"""Sampling-time guidance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..data.polygon_dataset import PolygonGraphBatch
from ..models.guidance_models import build_guidance_model
from ..models.regularity_torch import regularity_score_torch, smooth_polygon_area_torch
from ..restoration import (
    RestorationProxyConfig,
    restoration_state_torch_dense,
    restoration_state_torch_graph,
)


@dataclass(frozen=True, slots=True)
class ClassifierGuidance:
    classifier: torch.nn.Module
    scale: float
    target_class: int

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        self.classifier.eval()
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            if graph_batch is None:
                classifier_input = x_in
            else:
                expected_data_dim = getattr(getattr(self.classifier, "trunk", None), "data_dim", None)
                classifier_input = _graph_batch_dense_input(x_in, graph_batch, expected_data_dim=expected_data_dim)
            logits = self.classifier(classifier_input, t)
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

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        self.regressor.eval()
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            if graph_batch is None:
                regressor_input = x_in
            else:
                expected_data_dim = getattr(getattr(self.regressor, "trunk", None), "data_dim", None)
                regressor_input = _graph_batch_dense_input(x_in, graph_batch, expected_data_dim=expected_data_dim)
            pred = self.regressor(regressor_input, t)
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
    n_vertices: int | None
    scale: float
    target_value: float | None = None
    alpha: float = 8.0
    beta: float = 5.0
    gamma: float = 4.0

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        del t  # analytic regularity guidance does not depend on timestep explicitly
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            if graph_batch is None:
                if self.n_vertices is None:
                    raise ValueError("n_vertices is required for dense analytic regularity guidance")
                if x_in.ndim != 2 or x_in.shape[1] != self.n_vertices * 2:
                    raise ValueError(
                        f"x_t must have shape (batch, {self.n_vertices * 2}) for n_vertices={self.n_vertices}, "
                        f"got {tuple(x_in.shape)}"
                    )
                coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
                scores = regularity_score_torch(
                    coords,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                ).score
            else:
                scores = torch.stack(
                    [
                        regularity_score_torch(
                            x_in[graph_batch.graph_slice(i)],
                            alpha=self.alpha,
                            beta=self.beta,
                            gamma=self.gamma,
                        ).score
                        for i in range(graph_batch.batch_size)
                    ],
                    dim=0,
                )

            if self.target_value is None:
                objective = scores.sum()
            else:
                objective = -((scores - float(self.target_value)) ** 2).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


@dataclass(frozen=True, slots=True)
class AreaGuidance:
    n_vertices: int | None
    scale: float
    target_value: float | None = None
    absolute: bool = True

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        del t  # analytic area guidance does not depend on timestep explicitly
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            if graph_batch is None:
                if self.n_vertices is None:
                    raise ValueError("n_vertices is required for dense analytic area guidance")
                if x_in.ndim != 2 or x_in.shape[1] != self.n_vertices * 2:
                    raise ValueError(
                        f"x_t must have shape (batch, {self.n_vertices * 2}) for n_vertices={self.n_vertices}, "
                        f"got {tuple(x_in.shape)}"
                    )
                coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
                area = smooth_polygon_area_torch(coords, absolute=self.absolute)
            else:
                area = torch.stack(
                    [
                        smooth_polygon_area_torch(
                            x_in[graph_batch.graph_slice(i)],
                            absolute=self.absolute,
                        )
                        for i in range(graph_batch.batch_size)
                    ],
                    dim=0,
                )

            if self.target_value is None:
                objective = area.sum()
            else:
                objective = -((area - float(self.target_value)) ** 2).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


@dataclass(frozen=True, slots=True)
class RestorationGuidance:
    n_vertices: int | None
    scale: float
    restoration: RestorationProxyConfig
    target_value: float | None = None
    num_steps: int = 1
    min_timestep_weight: float = 0.05
    timestep_power: float = 2.0

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            if graph_batch is None:
                if self.n_vertices is None:
                    raise ValueError("n_vertices is required for dense analytic restoration guidance")
                if x_in.ndim != 2 or x_in.shape[1] != self.n_vertices * 2:
                    raise ValueError(
                        f"x_t must have shape (batch, {self.n_vertices * 2}) for n_vertices={self.n_vertices}, "
                        f"got {tuple(x_in.shape)}"
                    )
                state = restoration_state_torch_dense(
                    x_in.reshape(x_in.shape[0], self.n_vertices, 2),
                    self.restoration,
                )
            else:
                state = restoration_state_torch_graph(x_in, graph_batch, self.restoration)

            timestep_weight = _restoration_timestep_weight(
                t,
                num_steps=self.num_steps,
                min_timestep_weight=self.min_timestep_weight,
                timestep_power=self.timestep_power,
                dtype=state.dna_distance.dtype,
            )
            dna_distance = state.dna_distance
            if self.target_value is None:
                objective = -(timestep_weight * dna_distance.square()).sum()
            else:
                objective = -(timestep_weight * ((dna_distance - float(self.target_value)) ** 2)).sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


AtomicSamplingGuidance = (
    ClassifierGuidance
    | RegressorGuidance
    | RegularityScoreGuidance
    | AreaGuidance
    | RestorationGuidance
)


@dataclass(frozen=True, slots=True)
class GuidanceSchedule:
    kind: str
    num_steps: int
    start_frac: float = 0.0
    end_frac: float = 1.0
    min_scale: float = 0.0

    def weights(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"t must have shape (batch,), got {tuple(t.shape)}")
        if self.num_steps <= 1:
            return torch.ones_like(t, dtype=dtype)
        progress = 1.0 - t.to(dtype) / float(self.num_steps - 1)
        progress = progress.clamp(0.0, 1.0)
        kind = self.kind.lower()
        if kind == "all":
            return torch.ones_like(progress, dtype=dtype)
        if kind == "early":
            return (progress <= 0.30).to(dtype)
        if kind == "mid":
            return ((progress >= 0.30) & (progress < 0.70)).to(dtype)
        if kind == "late":
            return (progress >= 0.70).to(dtype)
        if kind == "window":
            return ((progress >= float(self.start_frac)) & (progress <= float(self.end_frac))).to(dtype)
        if kind == "linear_ramp":
            return float(self.min_scale) + (1.0 - float(self.min_scale)) * progress
        if kind == "quadratic_ramp":
            return float(self.min_scale) + (1.0 - float(self.min_scale)) * progress.square()
        decay_progress = 1.0 - progress
        if kind == "linear_decay":
            return float(self.min_scale) + (1.0 - float(self.min_scale)) * decay_progress
        if kind == "quadratic_decay":
            return float(self.min_scale) + (1.0 - float(self.min_scale)) * decay_progress.square()
        raise ValueError(f"Unsupported guidance schedule {self.kind!r}")


@dataclass(frozen=True, slots=True)
class ScheduledGuidance:
    guidance: AtomicSamplingGuidance | CompositeGuidance
    schedule: GuidanceSchedule

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        grad = self.guidance(x_t, t, graph_batch=graph_batch)
        weights = self.schedule.weights(t, dtype=grad.dtype)
        if graph_batch is None:
            return grad * weights.unsqueeze(1)
        node_weights = weights.index_select(0, graph_batch.graph_index).unsqueeze(1)
        return grad * node_weights


@dataclass(frozen=True, slots=True)
class CompositeGuidance:
    components: tuple[AtomicSamplingGuidance, ...]

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        grad_total = torch.zeros_like(x_t)
        for component in self.components:
            grad_total = grad_total + component(x_t, t, graph_batch=graph_batch)
        return grad_total


SamplingGuidance = AtomicSamplingGuidance | CompositeGuidance | ScheduledGuidance


def _restoration_timestep_weight(
    t: torch.Tensor,
    *,
    num_steps: int,
    min_timestep_weight: float,
    timestep_power: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError(f"t must have shape (batch,), got {tuple(t.shape)}")
    if num_steps <= 1:
        return torch.ones_like(t, dtype=dtype)
    progress = 1.0 - t.to(dtype) / float(num_steps - 1)
    progress = progress.clamp(0.0, 1.0)
    return float(min_timestep_weight) + (1.0 - float(min_timestep_weight)) * progress.pow(float(timestep_power))


def _graph_batch_dense_input(
    x_t: torch.Tensor,
    graph_batch: PolygonGraphBatch,
    *,
    expected_data_dim: int | None,
) -> torch.Tensor:
    if x_t.ndim != 2 or x_t.shape != (graph_batch.total_vertices, 2):
        raise ValueError(
            f"x_t must have shape ({graph_batch.total_vertices}, 2) for graph guidance, got {tuple(x_t.shape)}"
        )
    n_vertices = graph_batch.uniform_num_vertices()
    if expected_data_dim is not None and expected_data_dim != n_vertices * 2:
        raise ValueError(
            f"checkpoint-backed guidance expects data_dim={expected_data_dim}, but graph batch has n_vertices={n_vertices}"
        )
    return x_t.reshape(graph_batch.batch_size, n_vertices * 2)


def load_sampling_guidance(
    checkpoint_path: Path | None,
    *,
    device: torch.device,
    kind: str,
    scale: float,
    num_steps: int | None = None,
    n_vertices: int | None = None,
    target_class: int | None = None,
    target_value: float | None = None,
    alpha: float = 8.0,
    beta: float = 5.0,
    gamma: float = 4.0,
    schedule: str = "all",
    schedule_start: float = 0.0,
    schedule_end: float = 1.0,
    schedule_min_scale: float = 0.0,
    min_timestep_weight: float = 0.05,
    timestep_power: float = 2.0,
    restoration: RestorationProxyConfig | None = None,
) -> tuple[dict[str, Any], SamplingGuidance, int | None]:
    task = str(kind).lower()
    schedule_name = str(schedule).lower()

    def _wrap_schedule(metadata: dict[str, Any], guidance: SamplingGuidance) -> tuple[dict[str, Any], SamplingGuidance]:
        if schedule_name == "all":
            return metadata, guidance
        if num_steps is None:
            raise ValueError("num_steps is required when sampling.guidance.schedule is not 'all'")
        metadata = {
            **metadata,
            "guidance_schedule": schedule_name,
            "guidance_schedule_start": float(schedule_start),
            "guidance_schedule_end": float(schedule_end),
            "guidance_schedule_min_scale": float(schedule_min_scale),
        }
        return metadata, ScheduledGuidance(
            guidance=guidance,
            schedule=GuidanceSchedule(
                kind=schedule_name,
                num_steps=int(num_steps),
                start_frac=float(schedule_start),
                end_frac=float(schedule_end),
                min_scale=float(schedule_min_scale),
            ),
        )

    if task == "regularity":
        metadata = {
            "guidance_task": "regularity",
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        }
        if n_vertices is not None:
            metadata["n_vertices"] = int(n_vertices)
        guidance = RegularityScoreGuidance(
            n_vertices=None if n_vertices is None else int(n_vertices),
            scale=float(scale),
            target_value=None if target_value is None else float(target_value),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )
        metadata, guidance = _wrap_schedule(metadata, guidance)
        return metadata, guidance, None if n_vertices is None else int(n_vertices)

    if task == "area":
        metadata = {
            "guidance_task": "area",
            "absolute": True,
        }
        if n_vertices is not None:
            metadata["n_vertices"] = int(n_vertices)
        guidance = AreaGuidance(
            n_vertices=None if n_vertices is None else int(n_vertices),
            scale=float(scale),
            target_value=None if target_value is None else float(target_value),
            absolute=True,
        )
        metadata, guidance = _wrap_schedule(metadata, guidance)
        return metadata, guidance, None if n_vertices is None else int(n_vertices)

    if task == "restoration":
        if restoration is None:
            raise ValueError("restoration config is required for restoration guidance")
        if num_steps is None:
            raise ValueError("num_steps is required for restoration guidance")
        metadata = {
            "guidance_task": "restoration",
            "restoration": restoration.to_dict(),
            "guidance_num_steps": int(num_steps),
            "min_timestep_weight": float(min_timestep_weight),
            "timestep_power": float(timestep_power),
        }
        if n_vertices is not None:
            metadata["n_vertices"] = int(n_vertices)
        guidance = RestorationGuidance(
            n_vertices=None if n_vertices is None else int(n_vertices),
            scale=float(scale),
            restoration=restoration,
            target_value=None if target_value is None else float(target_value),
            num_steps=int(num_steps),
            min_timestep_weight=float(min_timestep_weight),
            timestep_power=float(timestep_power),
        )
        metadata, guidance = _wrap_schedule(metadata, guidance)
        return metadata, guidance, None if n_vertices is None else int(n_vertices)

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
        checkpoint, guidance = _wrap_schedule(dict(checkpoint), guidance)
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
        checkpoint, guidance = _wrap_schedule(dict(checkpoint), guidance)
        return checkpoint, guidance, n_vertices

    raise ValueError(
        f"Unsupported guidance kind {kind!r}; expected one of: classifier, regressor, regularity, area, restoration"
    )


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
