"""Sampling-time guidance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..models.guidance_models import build_guidance_model
from ..models.regularity_torch import regularity_score_torch, smooth_polygon_area_torch


@dataclass(frozen=True, slots=True)
class ClassifierGuidance:
    classifier: torch.nn.Module
    scale: float
    target_class: int

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.classifier.eval()
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            if logits.ndim != 2:
                raise ValueError(f"classifier logits must have shape (batch, num_classes), got {tuple(logits.shape)}")
            if not 0 <= self.target_class < logits.shape[1]:
                raise ValueError(
                    f"target_class must be in [0, {logits.shape[1] - 1}], got {self.target_class}"
                )
            log_probs = F.log_softmax(logits, dim=1)
            selected = log_probs[:, self.target_class].sum()
            grad = torch.autograd.grad(selected, x_in)[0]
        return self.scale * grad.detach()


@dataclass(frozen=True, slots=True)
class RegressorGuidance:
    regressor: torch.nn.Module
    scale: float
    target_value: float | None = None

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.regressor.eval()
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            pred = self.regressor(x_in, t)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred[:, 0]
            elif pred.ndim != 1:
                raise ValueError(
                    f"regressor output must have shape (batch,) or (batch, 1), got {tuple(pred.shape)}"
                )

            if self.target_value is None:
                objective = pred.sum()
            else:
                objective = -((pred - float(self.target_value)) ** 2).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


@dataclass(frozen=True, slots=True)
class RegularityScoreGuidance:
    n_vertices: int
    scale: float
    target_value: float | None = None
    alpha: float = 8.0
    beta: float = 5.0
    gamma: float = 4.0

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t  # analytic regularity guidance does not depend on timestep explicitly
        if x_t.ndim != 2 or x_t.shape[1] != self.n_vertices * 2:
            raise ValueError(
                f"x_t must have shape (batch, {self.n_vertices * 2}) for n_vertices={self.n_vertices}, "
                f"got {tuple(x_t.shape)}"
            )

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
            regularity = regularity_score_torch(
                coords,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            if self.target_value is None:
                objective = regularity.score.sum()
            else:
                objective = -((regularity.score - float(self.target_value)) ** 2).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


@dataclass(frozen=True, slots=True)
class AreaGuidance:
    n_vertices: int
    scale: float
    target_value: float | None = None
    absolute: bool = True

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t  # analytic area guidance does not depend on timestep explicitly
        if x_t.ndim != 2 or x_t.shape[1] != self.n_vertices * 2:
            raise ValueError(
                f"x_t must have shape (batch, {self.n_vertices * 2}) for n_vertices={self.n_vertices}, "
                f"got {tuple(x_t.shape)}"
            )

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
            area = smooth_polygon_area_torch(coords, absolute=self.absolute)

            if self.target_value is None:
                objective = area.sum()
            else:
                objective = -((area - float(self.target_value)) ** 2).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


SamplingGuidance = ClassifierGuidance | RegressorGuidance | RegularityScoreGuidance | AreaGuidance


def load_sampling_guidance(
    checkpoint_path: Path | None,
    *,
    device: torch.device,
    kind: str,
    scale: float,
    n_vertices: int | None = None,
    target_class: int | None = None,
    target_value: float | None = None,
    alpha: float = 8.0,
    beta: float = 5.0,
    gamma: float = 4.0,
) -> tuple[dict[str, Any], SamplingGuidance, int]:
    task = str(kind).lower()

    if task == "regularity":
        if n_vertices is None:
            raise ValueError("n_vertices is required for regularity guidance")
        metadata = {
            "guidance_task": "regularity",
            "n_vertices": int(n_vertices),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        }
        guidance = RegularityScoreGuidance(
            n_vertices=int(n_vertices),
            scale=float(scale),
            target_value=None if target_value is None else float(target_value),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        return metadata, guidance, int(n_vertices)

    if task == "area":
        if n_vertices is None:
            raise ValueError("n_vertices is required for area guidance")
        metadata = {
            "guidance_task": "area",
            "n_vertices": int(n_vertices),
            "absolute": True,
        }
        guidance = AreaGuidance(
            n_vertices=int(n_vertices),
            scale=float(scale),
            target_value=None if target_value is None else float(target_value),
            absolute=True,
        )
        return metadata, guidance, int(n_vertices)

    if checkpoint_path is None:
        raise ValueError(f"checkpoint_path is required for {task!r} guidance")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model_cfg = dict(checkpoint.get("model_cfg", {}))
    checkpoint_model_cfg.setdefault("type", "mlp")
    n_vertices = int(checkpoint.get("n_vertices", 6))

    checkpoint_task = checkpoint.get("guidance_task")
    if checkpoint_task is not None and str(checkpoint_task).lower() != task:
        raise ValueError(
            f"guidance checkpoint task={checkpoint_task!r} does not match requested kind={kind!r}"
        )

    if task == "classifier":
        num_classes = int(checkpoint.get("num_classes", 2))
        classifier = build_guidance_model(
            task="classifier",
            data_dim=n_vertices * 2,
            model_cfg=checkpoint_model_cfg,
            num_classes=num_classes,
        )
        classifier.load_state_dict(checkpoint["model_state"])
        classifier.to(device)
        classifier.eval()
        guidance = ClassifierGuidance(
            classifier=classifier,
            scale=float(scale),
            target_class=1 if target_class is None else int(target_class),
        )
        return checkpoint, guidance, n_vertices

    if task == "regressor":
        regressor = build_guidance_model(
            task="regressor",
            data_dim=n_vertices * 2,
            model_cfg=checkpoint_model_cfg,
        )
        regressor.load_state_dict(checkpoint["model_state"])
        regressor.to(device)
        regressor.eval()
        guidance = RegressorGuidance(
            regressor=regressor,
            scale=float(scale),
            target_value=None if target_value is None else float(target_value),
        )
        return checkpoint, guidance, n_vertices

    raise ValueError(f"Unsupported guidance kind {kind!r}; expected one of: classifier, regressor, regularity, area")


def load_guidance_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    kind: str,
    scale: float,
    target_class: int | None = None,
    target_value: float | None = None,
) -> tuple[dict[str, Any], ClassifierGuidance | RegressorGuidance, int]:
    checkpoint, guidance, n_vertices = load_sampling_guidance(
        checkpoint_path,
        device=device,
        kind=kind,
        scale=scale,
        target_class=target_class,
        target_value=target_value,
    )
    if not isinstance(guidance, (ClassifierGuidance, RegressorGuidance)):
        raise ValueError("load_guidance_from_checkpoint only supports checkpoint-backed classifier/regressor guidance")
    return checkpoint, guidance, n_vertices
