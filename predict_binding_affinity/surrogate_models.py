"""Permutation-invariant ligand-context surrogate models."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from polydiff.models.diffusion import SinusoidalPosEmb

from .surrogate_data import LIGAND_NODE, SurrogateBatch


PoolingMode = Literal["mean", "avg", "max", "sum"]


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
    counts = counts.clamp(min=1.0)
    return sums / counts.unsqueeze(-1)


def _scatter_max(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    out = values.new_full((dim_size, values.shape[-1]), float("-inf"))
    expanded_index = index.unsqueeze(-1).expand(-1, values.shape[-1])
    out.scatter_reduce_(0, expanded_index, values, reduce="amax", include_self=True)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def _pool_graph_features(
    values: torch.Tensor,
    batch: torch.Tensor,
    *,
    mask: torch.Tensor,
    pooling: str,
    num_graphs: int,
) -> torch.Tensor:
    masked_values = values[mask]
    masked_batch = batch[mask]
    if masked_values.numel() == 0:
        return values.new_zeros((num_graphs, values.shape[-1]))
    if pooling == "sum":
        return _scatter_sum(masked_values, masked_batch, dim_size=num_graphs)
    if pooling == "max":
        return _scatter_max(masked_values, masked_batch, dim_size=num_graphs)
    return _scatter_mean(masked_values, masked_batch, dim_size=num_graphs)


class GaussianRBF(nn.Module):
    def __init__(self, *, num_rbf: int = 32, cutoff: float = 6.0) -> None:
        super().__init__()
        if num_rbf < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        centers = torch.linspace(0.0, float(cutoff), int(num_rbf), dtype=torch.float32)
        self.register_buffer("centers", centers)
        step = 1.0 if centers.numel() == 1 else float((centers[1] - centers[0]).item())
        self.gamma = 1.0 / max(step * step, 1e-6)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * diff.square())


class ContextMessageBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_edge_types: int = 2,
        num_rbf: int = 32,
        cutoff: float = 6.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear((3 * hidden_dim) + num_rbf + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, batch: SurrogateBatch) -> torch.Tensor:
        if batch.edge_index.numel() == 0:
            return h

        src, dst = batch.edge_index
        rel = batch.pos.index_select(0, src) - batch.pos.index_select(0, dst)
        dist = rel.norm(dim=-1)
        radial = self.rbf(dist)
        edge_kind = self.edge_type_emb(batch.edge_type.long())
        m_in = torch.cat(
            [
                h.index_select(0, src),
                h.index_select(0, dst),
                edge_kind,
                radial,
                dist.unsqueeze(-1),
            ],
            dim=-1,
        )
        messages = self.msg_mlp(m_in)
        aggregated = _scatter_mean(messages, dst, dim_size=h.shape[0])
        update = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        ligand_mask = batch.node_type == LIGAND_NODE
        out = h.clone()
        out[ligand_mask] = self.norm(h[ligand_mask] + update[ligand_mask])
        return out


class LigandContextSurrogateModel(nn.Module):
    """Ligand-context GNN with configurable ligand pooling and optional `t` conditioning."""

    def __init__(
        self,
        *,
        max_atomic_number: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 32,
        cutoff: float = 6.0,
        dropout: float = 0.1,
        pooling: PoolingMode = "mean",
        timestep_conditioning: bool = True,
        time_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        if max_atomic_number < 2:
            raise ValueError(f"max_atomic_number must be >= 2, got {max_atomic_number}")
        self.max_atomic_number = int(max_atomic_number)
        self.pooling = _canonical_pooling(pooling)
        self.timestep_conditioning = bool(timestep_conditioning)

        self.atomic_number_emb = nn.Embedding(self.max_atomic_number, hidden_dim)
        self.node_type_emb = nn.Embedding(2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        if self.timestep_conditioning:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.time_mlp = None

        self.layers = nn.ModuleList(
            [
                ContextMessageBlock(
                    hidden_dim=hidden_dim,
                    num_edge_types=2,
                    num_rbf=num_rbf,
                    cutoff=cutoff,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: SurrogateBatch) -> torch.Tensor:
        clamped_z = batch.z.long().clamp(min=0, max=self.max_atomic_number - 1)
        h = self.atomic_number_emb(clamped_z) + self.node_type_emb(batch.node_type.long())
        if self.time_mlp is not None:
            if batch.t is None:
                raise ValueError("Batch is missing timestep tensor `t` for a timestep-conditioned surrogate model")
            time_features = self.time_mlp(batch.t.float()).index_select(0, batch.batch)
            h = h + time_features
        h = self.input_norm(h)

        for layer in self.layers:
            h = layer(h, batch)

        graph_features = _pool_graph_features(
            h,
            batch.batch,
            mask=batch.node_type == LIGAND_NODE,
            pooling=self.pooling,
            num_graphs=batch.batch_size,
        )
        return self.head(graph_features).squeeze(-1)
