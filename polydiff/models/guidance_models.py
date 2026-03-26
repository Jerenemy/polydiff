"""Time-conditioned guidance models for sampling guidance."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.polygon_dataset import PolygonGraphBatch, build_polygon_graph_batch
from .diffusion import SinusoidalPosEmb, _cycle_adj, _cycle_edge_index, _cycle_positional_features
from .gat import GAT
from .gcn import GraphConvolution


def _canonical_pooling(pooling: str) -> str:
    pooling = str(pooling).strip().lower()
    if pooling == "avg":
        pooling = "mean"
    if pooling not in {"mean", "max", "sum"}:
        raise ValueError(f"Unsupported pooling {pooling!r}; expected avg, mean, max, or sum")
    return pooling


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, values.shape[-1]))
    out.index_add_(0, index, values)
    return out


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    sums = _scatter_sum(values, index, dim_size=dim_size)
    counts = values.new_zeros((dim_size,))
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return sums / counts.clamp(min=1.0).unsqueeze(-1)


def _scatter_max(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.new_zeros((dim_size, values.shape[-1]))
    out = values.new_full((dim_size, values.shape[-1]), float("-inf"))
    for graph_index in range(dim_size):
        mask = index == graph_index
        if bool(mask.any()):
            out[graph_index] = values[mask].max(dim=0).values
    out[torch.isinf(out)] = 0.0
    return out


def _pool_graph_features(
    values: torch.Tensor,
    batch: PolygonGraphBatch,
    *,
    pooling: str,
) -> torch.Tensor:
    pooling = _canonical_pooling(pooling)
    if pooling == "sum":
        return _scatter_sum(values, batch.graph_index, dim_size=batch.batch_size)
    if pooling == "max":
        return _scatter_max(values, batch.graph_index, dim_size=batch.batch_size)
    return _scatter_mean(values, batch.graph_index, dim_size=batch.batch_size)


def _as_graph_batch(
    x: torch.Tensor,
    *,
    batch: PolygonGraphBatch | None,
    expected_data_dim: int | None,
) -> tuple[PolygonGraphBatch, torch.Tensor]:
    if batch is not None:
        if x.ndim != 2 or x.shape != (batch.total_vertices, 2):
            raise ValueError(f"x must have shape ({batch.total_vertices}, 2), got {tuple(x.shape)}")
        return batch, x

    if x.ndim != 2 or x.shape[1] % 2 != 0:
        if expected_data_dim is None:
            raise ValueError("dense graph guidance input must have shape (batch, n_vertices * 2)")
        raise ValueError(f"x must have shape (batch, {expected_data_dim}), got {tuple(x.shape)}")
    if expected_data_dim is not None and x.shape[1] != expected_data_dim:
        raise ValueError(f"x must have shape (batch, {expected_data_dim}), got {tuple(x.shape)}")
    num_vertices = x.shape[1] // 2
    batch_size = x.shape[0]
    coords = x.reshape(batch_size * num_vertices, 2)
    graph_batch = build_polygon_graph_batch(
        torch.full((batch_size,), num_vertices, dtype=torch.long, device=x.device),
        coords=coords,
    )
    return graph_batch, coords


class _NoisyPolygonTimeMLPTrunk(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        timestep_conditioning: bool,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.timestep_conditioning = bool(timestep_conditioning)
        if self.timestep_conditioning:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
            )
            in_dim = data_dim + time_emb_dim
        else:
            self.time_mlp = None
            in_dim = data_dim

        layers: list[nn.Module] = []
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        if batch is not None:
            if x.ndim != 2 or x.shape != (batch.total_vertices, 2):
                raise ValueError(f"x must have shape ({batch.total_vertices}, 2), got {tuple(x.shape)}")
            n_vertices = batch.uniform_num_vertices()
            if self.data_dim != n_vertices * 2:
                raise ValueError(
                    f"MLP guidance expects uniform polygons with data_dim={self.data_dim}, got n_vertices={n_vertices}"
                )
            x = x.reshape(batch.batch_size, self.data_dim)
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape (batch, {self.data_dim}), got {tuple(x.shape)}")
        if self.time_mlp is None:
            return self.net(x)
        if t is None or t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape ({x.shape[0]},), got {None if t is None else tuple(t.shape)}")
        time_emb = self.time_mlp(t)
        return self.net(torch.cat([x, time_emb], dim=1))


class NoisyPolygonClassifierMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        num_classes: int,
        timestep_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.trunk = _NoisyPolygonTimeMLPTrunk(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            timestep_conditioning=timestep_conditioning,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t, batch=batch))


class NoisyPolygonRegressorMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        timestep_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.trunk = _NoisyPolygonTimeMLPTrunk(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            timestep_conditioning=timestep_conditioning,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t, batch=batch)).squeeze(-1)


class _NoisyPolygonGraphTrunk(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        pooling: str,
        timestep_conditioning: bool,
    ) -> None:
        super().__init__()
        self.model_type = str(model_type).lower()
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.pooling = _canonical_pooling(pooling)
        self.timestep_conditioning = bool(timestep_conditioning)
        self.coord_dim = 2
        self.pos_enc_dim = 4
        node_in_dim = self.coord_dim + self.pos_enc_dim + (int(time_emb_dim) if self.timestep_conditioning else 0)

        if self.timestep_conditioning:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
            )
        else:
            self.time_mlp = None

        if self.model_type == "gat":
            hidden_heads = 4 if hidden_dim >= 4 else 1
            hidden_per_head = max(1, hidden_dim // hidden_heads)
            num_heads_per_layer = [hidden_heads] * max(0, num_layers - 1) + [1]
            num_features_per_layer = [node_in_dim]
            if num_layers > 1:
                num_features_per_layer.extend([hidden_per_head] * (num_layers - 1))
            num_features_per_layer.append(hidden_dim)
            self.gat = GAT(
                num_of_layers=max(1, int(num_layers)),
                num_heads_per_layer=num_heads_per_layer,
                num_features_per_layer=num_features_per_layer,
                dropout=0.0,
            )
            self.gcn_layers = None
        elif self.model_type == "gcn":
            gcn_layers: list[nn.Module] = []
            in_dim = node_in_dim
            for _ in range(max(1, int(num_layers))):
                gcn_layers.append(GraphConvolution(in_dim, hidden_dim))
                in_dim = hidden_dim
            self.gcn_layers = nn.ModuleList(gcn_layers)
            self.gat = None
        else:
            raise ValueError(f"Unsupported graph guidance model type {model_type!r}; expected gat or gcn")

    def forward_features(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        graph_batch, coords = _as_graph_batch(x, batch=batch, expected_data_dim=self.data_dim if batch is None else None)
        batch_size = graph_batch.batch_size
        if self.time_mlp is not None:
            if t is None or t.ndim != 1 or t.shape[0] != batch_size:
                raise ValueError(f"t must have shape ({batch_size},), got {None if t is None else tuple(t.shape)}")
            time_emb = self.time_mlp(t).index_select(0, graph_batch.graph_index)
        else:
            time_emb = None

        pos_enc = _cycle_positional_features(graph_batch, dtype=coords.dtype)
        node_features = [coords, pos_enc]
        if time_emb is not None:
            node_features.append(time_emb)
        node_features_cat = torch.cat(node_features, dim=1)

        if self.gat is not None:
            edge_index = _cycle_edge_index(graph_batch)
            node_hidden, _ = self.gat((node_features_cat, edge_index))
        else:
            assert self.gcn_layers is not None
            adj = _cycle_adj(graph_batch, device=coords.device)
            node_hidden = node_features_cat
            for layer in self.gcn_layers:
                node_hidden = F.silu(layer(node_hidden, adj))

        return _pool_graph_features(node_hidden, graph_batch, pooling=self.pooling)


class NoisyPolygonClassifierGraph(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        pooling: str,
        num_classes: int,
        timestep_conditioning: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.trunk = _NoisyPolygonGraphTrunk(
            model_type=model_type,
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            pooling=pooling,
            timestep_conditioning=timestep_conditioning,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t, batch=batch))


class NoisyPolygonRegressorGraph(nn.Module):
    def __init__(
        self,
        *,
        model_type: str,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        pooling: str,
        timestep_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.trunk = _NoisyPolygonGraphTrunk(
            model_type=model_type,
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            pooling=pooling,
            timestep_conditioning=timestep_conditioning,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t, batch=batch)).squeeze(-1)


def build_guidance_model(
    *,
    task: str,
    data_dim: int,
    model_cfg: Mapping[str, Any] | None = None,
    num_classes: int = 2,
) -> nn.Module:
    cfg = {} if model_cfg is None else dict(model_cfg)
    model_type = str(cfg.get("type", "mlp")).lower()
    hidden_dim = int(cfg.get("hidden_dim", 256))
    time_emb_dim = int(cfg.get("time_emb_dim", 64))
    num_layers = int(cfg.get("num_layers", 3))
    pooling = str(cfg.get("pooling", "mean"))
    timestep_conditioning = bool(cfg.get("timestep_conditioning", True))
    task = str(task).lower()

    if model_type == "mlp":
        if task == "classifier":
            return NoisyPolygonClassifierMLP(
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                time_emb_dim=time_emb_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                timestep_conditioning=timestep_conditioning,
            )
        if task == "regressor":
            return NoisyPolygonRegressorMLP(
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                time_emb_dim=time_emb_dim,
                num_layers=num_layers,
                timestep_conditioning=timestep_conditioning,
            )
    elif model_type in {"gat", "gcn"}:
        if task == "classifier":
            return NoisyPolygonClassifierGraph(
                model_type=model_type,
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                time_emb_dim=time_emb_dim,
                num_layers=num_layers,
                pooling=pooling,
                num_classes=num_classes,
                timestep_conditioning=timestep_conditioning,
            )
        if task == "regressor":
            return NoisyPolygonRegressorGraph(
                model_type=model_type,
                data_dim=data_dim,
                hidden_dim=hidden_dim,
                time_emb_dim=time_emb_dim,
                num_layers=num_layers,
                pooling=pooling,
                timestep_conditioning=timestep_conditioning,
            )
    else:
        raise ValueError(
            f"Unsupported guidance model type {model_type!r}; expected one of: mlp, gat, gcn"
        )

    raise ValueError(f"Unsupported guidance task {task!r}; expected one of: classifier, regressor")


def build_guidance_classifier(
    *,
    data_dim: int,
    model_cfg: Mapping[str, Any] | None = None,
    num_classes: int = 2,
) -> nn.Module:
    return build_guidance_model(
        task="classifier",
        data_dim=data_dim,
        model_cfg=model_cfg,
        num_classes=num_classes,
    )


def build_guidance_regressor(
    *,
    data_dim: int,
    model_cfg: Mapping[str, Any] | None = None,
) -> nn.Module:
    return build_guidance_model(
        task="regressor",
        data_dim=data_dim,
        model_cfg=model_cfg,
    )
