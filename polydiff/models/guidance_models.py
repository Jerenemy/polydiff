"""Time-conditioned guidance models for sampling guidance."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .diffusion import SinusoidalPosEmb


class _NoisyPolygonTimeMLPTrunk(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        layers: list[nn.Module] = []
        in_dim = data_dim + time_emb_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape (batch, {self.data_dim}), got {tuple(x.shape)}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape ({x.shape[0]},), got {tuple(t.shape)}")
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
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t))


class NoisyPolygonRegressorMLP(nn.Module):
    def __init__(
        self,
        *,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.trunk = _NoisyPolygonTimeMLPTrunk(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk.forward_features(x, t)).squeeze(-1)


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
    task = str(task).lower()

    if model_type != "mlp":
        raise ValueError(
            f"Unsupported guidance model type {model_type!r}; "
            "expected one of: mlp. Future graph-based guidance models can plug in here."
        )
    if task == "classifier":
        return NoisyPolygonClassifierMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    if task == "regressor":
        return NoisyPolygonRegressorMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
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
