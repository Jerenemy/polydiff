"""Sampling-side animation and diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.diagnostics import (
    DEFAULT_SCORE_THRESHOLDS,
    compare_polygon_summaries,
    compare_polygon_metric_tables,
    format_polygon_delta_summary,
    format_polygon_summary,
    json_ready,
    metric_threshold_rates,
    outlier_polygon_indices,
    polygon_metric_table,
    representative_polygon_indices,
    summarize_polygon_dataset,
)
from ..data.gen_polygons import normalize_size_distribution
from ..data.polygon_dataset import load_polygon_dataset
from ..models.diffusion import Diffusion, DiffusionConfig, build_denoiser
from ..restoration import (
    RestorationProxyConfig,
    build_restoration_animation_overlay,
    format_restoration_summary,
    restoration_scene_coords_numpy,
    summarize_restoration_dataset,
)
from ..utils.runtime import resolve_project_path

DEFAULT_SAMPLES_OUT_NAME = "samples.npz"
DEFAULT_ANIMATION_DIR_NAME = "animations"


@dataclass(frozen=True, slots=True)
class AnimationOptions:
    out_path: Path
    sample_index: int
    count: int
    max_frames: int
    fps: int


@dataclass(frozen=True, slots=True)
class DiagnosticsOptions:
    enabled: bool
    out_path: Path | None
    reference_data_path: Path | None


@dataclass(frozen=True, slots=True)
class GuidanceComponentOptions:
    kind: str
    checkpoint_path: Path | None
    scale: float
    target_class: int
    target_value: float | None
    alpha: float
    beta: float
    gamma: float
    schedule: str
    schedule_start: float
    schedule_end: float
    schedule_min_scale: float
    min_timestep_weight: float
    timestep_power: float

    def to_config_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "kind": self.kind,
            "scale": float(self.scale),
        }
        if self.checkpoint_path is not None:
            out["checkpoint"] = str(self.checkpoint_path)
        if self.kind == "classifier":
            out["target_class"] = int(self.target_class)
        elif self.target_value is not None:
            out["target_value"] = float(self.target_value)
        if self.kind == "regularity":
            out["alpha"] = float(self.alpha)
            out["beta"] = float(self.beta)
            out["gamma"] = float(self.gamma)
        if self.schedule != "all":
            out["schedule"] = self.schedule
            out["schedule_start"] = float(self.schedule_start)
            out["schedule_end"] = float(self.schedule_end)
            out["schedule_min_scale"] = float(self.schedule_min_scale)
        if self.kind == "restoration":
            out["min_timestep_weight"] = float(self.min_timestep_weight)
            out["timestep_power"] = float(self.timestep_power)
        return out


@dataclass(frozen=True, slots=True)
class GuidanceOptions:
    enabled: bool
    components: tuple[GuidanceComponentOptions, ...]


@dataclass(frozen=True, slots=True)
class SizeDistributionOptions:
    values: tuple[int, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SamplingRequest:
    num_samples: int
    n_steps: int
    out_path: Path
    canonicalize_output: bool
    animation: AnimationOptions | None
    diagnostics: DiagnosticsOptions
    guidance: GuidanceOptions
    restoration: RestorationProxyConfig | None
    size_distribution: SizeDistributionOptions | None


def load_diffusion_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Diffusion, DiffusionConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_model_cfg = dict(checkpoint.get("model_cfg", {}))
    checkpoint_model_cfg.setdefault("type", "gat")
    max_vertices = int(checkpoint.get("max_vertices", checkpoint.get("n_vertices", 6)))
    model = build_denoiser(data_dim=max_vertices * 2, model_cfg=checkpoint_model_cfg)
    model.load_state_dict(checkpoint["model_state"])

    checkpoint_diffusion_cfg = checkpoint.get("diffusion", {})
    diffusion_config = DiffusionConfig(
        n_steps=int(checkpoint_diffusion_cfg.get("n_steps", 1000)),
        beta_start=float(checkpoint_diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(checkpoint_diffusion_cfg.get("beta_end", 2e-2)),
    )
    diffusion = Diffusion(model=model, config=diffusion_config, device=device)
    return checkpoint, diffusion, diffusion_config, max_vertices


def normalize_animation_out_path(path_like: str | Path) -> Path:
    out_path = resolve_project_path(path_like)
    if out_path.suffix and out_path.suffix.lower() != ".gif":
        raise ValueError(f"animation output must be a .gif file or directory, got {out_path}")
    return out_path


def animation_sample_indices(options: AnimationOptions) -> list[int]:
    return list(range(options.sample_index, options.sample_index + options.count))


def animation_output_paths(options: AnimationOptions) -> list[Path]:
    sample_indices = animation_sample_indices(options)
    if options.count == 1 and options.out_path.suffix.lower() == ".gif":
        return [options.out_path]

    out_dir = options.out_path if options.out_path.suffix == "" else options.out_path.parent / options.out_path.stem
    return [out_dir / f"sample_{sample_index:04d}.gif" for sample_index in sample_indices]


def resolve_animation_options(
    sampling_cfg: dict[str, Any],
    *,
    enable_animation: bool,
    default_out_path: Path | None,
    animation_out_path: str | None,
    animation_count: int | None,
    animation_sample_index: int | None,
    animation_max_frames: int | None,
    animation_fps: int | None,
) -> AnimationOptions | None:
    animation_cfg = sampling_cfg.get("animation", {})
    if animation_cfg is None:
        animation_cfg = {}
    if not isinstance(animation_cfg, dict):
        raise ValueError("sampling.animation must be a mapping if provided")

    out_path = animation_cfg.get("out_path")
    count = animation_cfg.get("count", 1)
    if enable_animation:
        out_path = animation_out_path or out_path or default_out_path or DEFAULT_ANIMATION_DIR_NAME

    if out_path is None:
        return None

    sample_index = animation_cfg.get("sample_index", 0)
    max_frames = animation_cfg.get("max_frames", 120)
    fps = animation_cfg.get("fps", 12)

    if animation_count is not None:
        count = animation_count
    if animation_sample_index is not None:
        sample_index = animation_sample_index
    if animation_max_frames is not None:
        max_frames = animation_max_frames
    if animation_fps is not None:
        fps = animation_fps

    sample_index = int(sample_index)
    count = int(count)
    max_frames = int(max_frames)
    fps = int(fps)

    if sample_index < 0:
        raise ValueError(f"animation sample_index must be >= 0, got {sample_index}")
    if count < 1:
        raise ValueError(f"animation count must be at least 1, got {count}")
    if max_frames < 2:
        raise ValueError(f"animation max_frames must be at least 2, got {max_frames}")
    if fps < 1:
        raise ValueError(f"animation fps must be at least 1, got {fps}")

    return AnimationOptions(
        out_path=normalize_animation_out_path(out_path),
        sample_index=sample_index,
        count=count,
        max_frames=max_frames,
        fps=fps,
    )


def resolve_diagnostics_options(sampling_cfg: dict[str, Any]) -> DiagnosticsOptions:
    diagnostics_cfg = sampling_cfg.get("diagnostics", {})
    if diagnostics_cfg is None:
        diagnostics_cfg = {}
    if not isinstance(diagnostics_cfg, dict):
        raise ValueError("sampling.diagnostics must be a mapping if provided")

    reference_data_path = diagnostics_cfg.get("reference_data_path")
    out_path = diagnostics_cfg.get("out_path")
    return DiagnosticsOptions(
        enabled=bool(diagnostics_cfg.get("enabled", True)),
        out_path=None if out_path is None else resolve_project_path(out_path),
        reference_data_path=None if reference_data_path is None else resolve_project_path(reference_data_path),
    )


def _resolve_xy_point(value: object, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a length-2 sequence, got {value!r}")
    return float(value[0]), float(value[1])


def _resolve_xy_points(value: object, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of [x, y] points, got {value!r}")
    return tuple(_resolve_xy_point(point, field_name=field_name) for point in value)


def resolve_restoration_options(sampling_cfg: dict[str, Any]) -> RestorationProxyConfig | None:
    restoration_cfg = sampling_cfg.get("restoration")
    if restoration_cfg is None:
        return None
    if not isinstance(restoration_cfg, dict):
        raise ValueError("sampling.restoration must be a mapping if provided")
    if not bool(restoration_cfg.get("enabled", False)):
        return None

    ligand_binding_site = _resolve_xy_point(
        restoration_cfg.get("ligand_binding_site", restoration_cfg.get("binding_site")),
        field_name="sampling.restoration.ligand_binding_site",
    )
    dna_unbound_position = _resolve_xy_point(
        restoration_cfg.get("dna_unbound_position", restoration_cfg.get("mutant_position")),
        field_name="sampling.restoration.dna_unbound_position",
    )
    dna_bound_position = _resolve_xy_point(
        restoration_cfg.get("dna_bound_position", restoration_cfg.get("wild_type_position")),
        field_name="sampling.restoration.dna_bound_position",
    )
    mutant_target_points = _resolve_xy_points(
        restoration_cfg.get("mutant_target_points", restoration_cfg.get("target_points", ())),
        field_name="sampling.restoration.mutant_target_points",
    )
    if not mutant_target_points:
        raise ValueError("sampling.restoration.mutant_target_points must contain at least one [x, y] point")
    wild_type_target_points = restoration_cfg.get("wild_type_target_points")
    if wild_type_target_points is not None:
        wild_type_target_points = _resolve_xy_points(
            wild_type_target_points,
            field_name="sampling.restoration.wild_type_target_points",
        )
        if len(wild_type_target_points) != len(mutant_target_points):
            raise ValueError(
                "sampling.restoration.wild_type_target_points must match mutant_target_points in length"
            )
    activation_sigma = float(restoration_cfg.get("activation_sigma", 0.35))
    contact_beta = float(restoration_cfg.get("contact_beta", 12.0))
    dna_binding_threshold = float(restoration_cfg.get("dna_binding_threshold", 0.65))
    dna_binding_steepness = float(restoration_cfg.get("dna_binding_steepness", 14.0))
    if activation_sigma <= 0.0:
        raise ValueError(f"sampling.restoration.activation_sigma must be > 0, got {activation_sigma}")
    if contact_beta <= 0.0:
        raise ValueError(f"sampling.restoration.contact_beta must be > 0, got {contact_beta}")
    if not 0.0 <= dna_binding_threshold <= 1.0:
        raise ValueError(
            f"sampling.restoration.dna_binding_threshold must be in [0, 1], got {dna_binding_threshold}"
        )
    if dna_binding_steepness <= 0.0:
        raise ValueError(
            f"sampling.restoration.dna_binding_steepness must be > 0, got {dna_binding_steepness}"
        )

    success_distance_raw = restoration_cfg.get("success_distance")
    if success_distance_raw is None:
        dna_unbound_np = np.asarray(dna_unbound_position, dtype=np.float32)
        dna_bound_np = np.asarray(dna_bound_position, dtype=np.float32)
        success_distance = max(float(np.linalg.norm(dna_unbound_np - dna_bound_np)) * 0.10, 1e-6)
    else:
        success_distance = float(success_distance_raw)
    if success_distance <= 0.0:
        raise ValueError(f"sampling.restoration.success_distance must be > 0, got {success_distance}")

    return RestorationProxyConfig(
        mutant_target_points=mutant_target_points,
        wild_type_target_points=wild_type_target_points,
        ligand_binding_site=ligand_binding_site,
        dna_unbound_position=dna_unbound_position,
        dna_bound_position=dna_bound_position,
        activation_sigma=activation_sigma,
        contact_beta=contact_beta,
        dna_binding_threshold=dna_binding_threshold,
        dna_binding_steepness=dna_binding_steepness,
        success_distance=success_distance,
    )


def _resolve_guidance_component(
    component_cfg: dict[str, Any],
    *,
    default_kind: str | None,
) -> GuidanceComponentOptions:
    kind = component_cfg.get("kind")
    kind_str = default_kind if kind is None else str(kind).lower()
    if kind_str is None:
        raise ValueError("guidance components must define a kind")
    if kind_str not in {"classifier", "regressor", "regularity", "area", "restoration"}:
        raise ValueError(
            f"Unsupported guidance kind {kind!r}; expected 'classifier', 'regressor', 'regularity', 'area', or 'restoration'"
        )

    checkpoint_path = component_cfg.get("checkpoint")
    scale = float(component_cfg.get("scale", 1.0))
    target_class = int(component_cfg.get("target_class", 1))
    target_value_raw = component_cfg.get("target_value")
    target_value = None if target_value_raw is None else float(target_value_raw)
    alpha = float(component_cfg.get("alpha", 8.0))
    beta = float(component_cfg.get("beta", 5.0))
    gamma = float(component_cfg.get("gamma", 4.0))
    schedule = str(component_cfg.get("schedule", "all")).lower()
    schedule_start = float(component_cfg.get("schedule_start", 0.0))
    schedule_end = float(component_cfg.get("schedule_end", 1.0))
    schedule_min_scale = float(component_cfg.get("schedule_min_scale", 0.0))
    min_timestep_weight = float(component_cfg.get("min_timestep_weight", 0.05))
    timestep_power = float(component_cfg.get("timestep_power", 2.0))
    valid_schedules = {
        "all",
        "early",
        "mid",
        "late",
        "window",
        "linear_ramp",
        "quadratic_ramp",
        "linear_decay",
        "quadratic_decay",
    }

    if scale < 0.0:
        raise ValueError(f"sampling.guidance.scale must be >= 0, got {scale}")
    if checkpoint_path is None and kind_str not in {"regularity", "area", "restoration"}:
        raise ValueError(f"sampling.guidance.checkpoint is required for {kind_str} guidance")
    if schedule not in valid_schedules:
        raise ValueError(
            f"sampling.guidance.schedule must be one of {sorted(valid_schedules)}, got {schedule!r}"
        )
    if not 0.0 <= schedule_start <= 1.0:
        raise ValueError(f"sampling.guidance.schedule_start must be in [0, 1], got {schedule_start}")
    if not 0.0 <= schedule_end <= 1.0:
        raise ValueError(f"sampling.guidance.schedule_end must be in [0, 1], got {schedule_end}")
    if schedule_start > schedule_end:
        raise ValueError(
            f"sampling.guidance.schedule_start must be <= schedule_end, got {schedule_start} > {schedule_end}"
        )
    if not 0.0 <= schedule_min_scale <= 1.0:
        raise ValueError(
            f"sampling.guidance.schedule_min_scale must be in [0, 1], got {schedule_min_scale}"
        )
    if not 0.0 <= min_timestep_weight <= 1.0:
        raise ValueError(
            f"sampling.guidance.min_timestep_weight must be in [0, 1], got {min_timestep_weight}"
        )
    if timestep_power <= 0.0:
        raise ValueError(f"sampling.guidance.timestep_power must be > 0, got {timestep_power}")

    return GuidanceComponentOptions(
        kind=kind_str,
        checkpoint_path=None if checkpoint_path is None else resolve_project_path(checkpoint_path),
        scale=scale,
        target_class=target_class,
        target_value=target_value,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        schedule=schedule,
        schedule_start=schedule_start,
        schedule_end=schedule_end,
        schedule_min_scale=schedule_min_scale,
        min_timestep_weight=min_timestep_weight,
        timestep_power=timestep_power,
    )


def resolve_guidance_options(
    sampling_cfg: dict[str, Any],
    *,
    restoration: RestorationProxyConfig | None,
) -> GuidanceOptions:
    guidance_cfg = sampling_cfg.get("guidance", {})
    if guidance_cfg is None:
        guidance_cfg = {}
    if not isinstance(guidance_cfg, dict):
        raise ValueError("sampling.guidance must be a mapping if provided")

    components_cfg = guidance_cfg.get("components")
    if components_cfg is None:
        if not bool(guidance_cfg.get("enabled", False)):
            return GuidanceOptions(enabled=False, components=())
        components = (_resolve_guidance_component(guidance_cfg, default_kind="classifier"),)
    else:
        if not isinstance(components_cfg, list):
            raise ValueError("sampling.guidance.components must be a list if provided")
        if not bool(guidance_cfg.get("enabled", True)):
            return GuidanceOptions(enabled=False, components=())
        components_list: list[GuidanceComponentOptions] = []
        for component_cfg in components_cfg:
            if not isinstance(component_cfg, dict):
                raise ValueError("each sampling.guidance.components entry must be a mapping")
            if not bool(component_cfg.get("enabled", True)):
                continue
            components_list.append(_resolve_guidance_component(component_cfg, default_kind=None))
        components = tuple(components_list)
        if not components:
            return GuidanceOptions(enabled=False, components=())

    if restoration is None and any(component.kind == "restoration" for component in components):
        raise ValueError("restoration guidance requires sampling.restoration.enabled: true")

    return GuidanceOptions(enabled=True, components=components)


def resolve_size_distribution_options(sampling_cfg: dict[str, Any]) -> SizeDistributionOptions | None:
    size_cfg = sampling_cfg.get("size_distribution")
    if size_cfg is None:
        return None
    if not isinstance(size_cfg, dict):
        raise ValueError("sampling.size_distribution must be a mapping if provided")
    values = size_cfg.get("values")
    if values is None:
        return None
    probabilities = size_cfg.get("probabilities")
    sizes, probs = normalize_size_distribution(values, probabilities)
    return SizeDistributionOptions(
        values=tuple(int(value) for value in sizes.tolist()),
        probabilities=tuple(float(prob) for prob in probs.tolist()),
    )


def resolve_sampling_request(
    sampling_cfg: dict[str, Any],
    *,
    checkpoint_n_steps: int,
    default_out_path: Path | None,
    default_animation_out_path: Path | None,
    enable_animation: bool,
    animation_out_path: str | None,
    animation_count: int | None,
    animation_sample_index: int | None,
    animation_max_frames: int | None,
    animation_fps: int | None,
) -> SamplingRequest:
    num_samples = int(sampling_cfg.get("num_samples", 64))
    n_steps = sampling_cfg.get("n_steps")
    n_steps = checkpoint_n_steps if n_steps is None else int(n_steps)
    if n_steps < 1 or n_steps > checkpoint_n_steps:
        raise ValueError(
            f"sampling.n_steps must be in [1, {checkpoint_n_steps}] "
            f"(checkpoint diffusion.n_steps), got {n_steps}"
        )
    if n_steps != checkpoint_n_steps:
        print(
            "[sample] warning: reduced-step sampling currently uses a naive schedule truncation, "
            "not a respaced sampler. Use full checkpoint_n_steps when comparing distribution quality."
        )

    animation = resolve_animation_options(
        sampling_cfg,
        enable_animation=enable_animation,
        default_out_path=default_animation_out_path,
        animation_out_path=animation_out_path,
        animation_count=animation_count,
        animation_sample_index=animation_sample_index,
        animation_max_frames=animation_max_frames,
        animation_fps=animation_fps,
    )
    if animation is not None and animation.sample_index >= num_samples:
        raise ValueError(
            f"animation sample_index must be in [0, {num_samples - 1}] for num_samples={num_samples}, "
            f"got {animation.sample_index}"
        )
    if animation is not None and animation.sample_index + animation.count > num_samples:
        raise ValueError(
            f"animation sample_index + count must be <= num_samples ({num_samples}), "
            f"got start={animation.sample_index}, count={animation.count}"
        )

    out_path_value = sampling_cfg.get("out_path")
    if out_path_value is None:
        out_path = default_out_path or resolve_project_path(f"data/processed/{DEFAULT_SAMPLES_OUT_NAME}")
    else:
        out_path = resolve_project_path(out_path_value)
    restoration = resolve_restoration_options(sampling_cfg)
    canonicalize_output_cfg = sampling_cfg.get("canonicalize_output")
    canonicalize_output = (restoration is None) if canonicalize_output_cfg is None else bool(canonicalize_output_cfg)
    if restoration is not None and canonicalize_output:
        raise ValueError("sampling.canonicalize_output must be false when restoration is enabled")
    return SamplingRequest(
        num_samples=num_samples,
        n_steps=n_steps,
        out_path=out_path,
        canonicalize_output=canonicalize_output,
        animation=animation,
        diagnostics=resolve_diagnostics_options(sampling_cfg),
        guidance=resolve_guidance_options(
            sampling_cfg,
            restoration=restoration,
        ),
        restoration=restoration,
        size_distribution=resolve_size_distribution_options(sampling_cfg),
    )


def save_animation(trajectory: torch.Tensor, n_vertices: int, options: AnimationOptions) -> Path:
    from ..data.plot_polygons import save_polygon_animation

    coords = trajectory.numpy().reshape(trajectory.shape[0], n_vertices, 2).astype(np.float32)
    return save_polygon_animation(
        coords,
        options.out_path,
        fps=options.fps,
        max_frames=options.max_frames,
    )


def save_animations(
    trajectories: torch.Tensor | list[torch.Tensor],
    n_vertices: int | None,
    options: AnimationOptions,
    *,
    restoration: RestorationProxyConfig | None = None,
) -> list[Path]:
    from ..data.plot_polygons import save_polygon_animation

    output_paths = animation_output_paths(options)
    if isinstance(trajectories, list):
        if len(trajectories) != len(output_paths):
            raise ValueError(f"expected {len(output_paths)} trajectories, got {len(trajectories)}")
        saved_paths: list[Path] = []
        for trajectory, out_path in zip(trajectories, output_paths):
            trajectory_cpu = trajectory.detach().cpu()
            if trajectory_cpu.ndim != 3 or trajectory_cpu.shape[-1] != 2:
                raise ValueError(
                    f"graph trajectory must have shape (frames, n_vertices, 2), got {tuple(trajectory_cpu.shape)}"
                )
            display_coords = trajectory_cpu.numpy().astype(np.float32)
            overlay = None
            if restoration is not None:
                display_coords = restoration_scene_coords_numpy(display_coords, restoration)
                overlay = build_restoration_animation_overlay(
                    display_coords,
                    restoration,
                )
            saved_paths.append(
                save_polygon_animation(
                    display_coords,
                    out_path,
                    fps=options.fps,
                    max_frames=options.max_frames,
                    restoration_overlay=overlay,
                )
            )
        return saved_paths

    if n_vertices is None:
        raise ValueError("n_vertices is required when saving dense trajectories")

    trajectories_cpu = trajectories.detach().cpu()
    if trajectories_cpu.ndim == 2:
        trajectories_cpu = trajectories_cpu.unsqueeze(1)
    if trajectories_cpu.ndim != 3:
        raise ValueError(f"trajectories must have shape (frames, count, data_dim), got {tuple(trajectories_cpu.shape)}")
    if trajectories_cpu.shape[1] != len(output_paths):
        raise ValueError(
            f"expected {len(output_paths)} trajectories for animation save, got {trajectories_cpu.shape[1]}"
        )

    saved_paths: list[Path] = []
    for i, out_path in enumerate(output_paths):
        coords = trajectories_cpu[:, i, :].numpy().reshape(trajectories_cpu.shape[0], n_vertices, 2).astype(np.float32)
        display_coords = coords
        overlay = None
        if restoration is not None:
            display_coords = restoration_scene_coords_numpy(coords, restoration)
            overlay = build_restoration_animation_overlay(display_coords, restoration)
        saved_paths.append(
            save_polygon_animation(
                display_coords,
                out_path,
                fps=options.fps,
                max_frames=options.max_frames,
                restoration_overlay=overlay,
            )
        )
    return saved_paths


def default_diagnostics_out_path(samples_out_path: Path) -> Path:
    return samples_out_path.with_name(f"{samples_out_path.stem}.diagnostics.json")


def default_metrics_out_path(samples_out_path: Path) -> Path:
    return samples_out_path.with_name(f"{samples_out_path.stem}.metrics.csv")


def resolve_reference_summary(
    checkpoint: dict[str, Any],
    options: DiagnosticsOptions,
) -> tuple[dict[str, float | int] | None, Path | None, str | None]:
    if options.reference_data_path is not None:
        dataset = load_polygon_dataset(options.reference_data_path)
        summary = summarize_polygon_dataset(dataset)
        return summary, options.reference_data_path, "reference_file"

    summary = checkpoint.get("training_data_summary")
    training_data_path = checkpoint.get("training_data_path")
    if summary is not None:
        resolved_path = None if training_data_path is None else resolve_project_path(training_data_path)
        return summary, resolved_path, "checkpoint"

    if training_data_path is not None:
        resolved_path = resolve_project_path(training_data_path)
        if resolved_path.exists():
            dataset = load_polygon_dataset(resolved_path)
            summary = summarize_polygon_dataset(dataset)
            return summary, resolved_path, "training_data_file"

    return None, None, None


def write_sampling_diagnostics(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    samples_out_path: Path,
    coords: np.ndarray,
    num_vertices: np.ndarray | None,
    options: DiagnosticsOptions,
    sampling_n_steps: int,
    sample_run_name: str | None = None,
    output_pose_normalized: bool = False,
    restoration: RestorationProxyConfig | None = None,
    restoration_trajectory: dict[str, object] | None = None,
) -> Path | None:
    if not options.enabled:
        return None

    generated_table = polygon_metric_table(coords, num_vertices=num_vertices)
    generated_summary = summarize_polygon_dataset(coords, num_vertices=num_vertices)
    reference_summary, reference_path, reference_source = resolve_reference_summary(checkpoint, options)
    delta_vs_reference = (
        None if reference_summary is None else compare_polygon_summaries(reference_summary, generated_summary)
    )
    reference_dataset = None
    reference_table = None
    if reference_path is not None and reference_path.exists():
        reference_dataset = load_polygon_dataset(reference_path)
        reference_table = polygon_metric_table(reference_dataset)
    distribution_distances = (
        None if reference_table is None else compare_polygon_metric_tables(reference_table, generated_table)
    )
    score_thresholds = metric_threshold_rates(
        generated_table["score"].to_numpy(dtype=np.float64, copy=False),
        metric_name="score",
        thresholds=DEFAULT_SCORE_THRESHOLDS,
    )
    representative_indices = None
    outlier_indices = None
    if reference_dataset is not None:
        representative_indices = representative_polygon_indices(
            reference_dataset,
            coords,
            count=min(16, int(generated_table.shape[0])),
            observed_num_vertices=num_vertices,
        ).tolist()
        outlier_indices = outlier_polygon_indices(
            reference_dataset,
            coords,
            count=min(16, int(generated_table.shape[0])),
            observed_num_vertices=num_vertices,
        ).tolist()

    diagnostics_out_path = options.out_path or default_diagnostics_out_path(samples_out_path)
    metrics_out_path = default_metrics_out_path(samples_out_path)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_table.to_csv(metrics_out_path, index=False)
    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("global_step"),
        "run_name": checkpoint.get("run_name"),
        "sample_run_name": sample_run_name,
        "samples_path": str(samples_out_path),
        "metrics_path": str(metrics_out_path),
        "sampling_n_steps": int(sampling_n_steps),
        "output_pose_normalized": bool(output_pose_normalized),
        "generated_summary": generated_summary,
        "score_threshold_rates": score_thresholds,
        "reference_summary": reference_summary,
        "reference_data_path": None if reference_path is None else str(reference_path),
        "reference_summary_source": reference_source,
        "delta_vs_reference": delta_vs_reference,
        "distribution_distances": distribution_distances,
        "representative_polygon_indices": representative_indices,
        "outlier_polygon_indices": outlier_indices,
    }
    if restoration is not None:
        payload["restoration_config"] = restoration.to_dict()
        payload["restoration_summary"] = summarize_restoration_dataset(
            coords,
            restoration,
            num_vertices=num_vertices,
        )
        payload["restoration_trajectory"] = restoration_trajectory

    diagnostics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[sample] generated summary: {format_polygon_summary(generated_summary)}")
    if delta_vs_reference is not None:
        print(f"[sample] generated vs reference: {format_polygon_delta_summary(delta_vs_reference)}")
    else:
        print("[sample] no reference summary available for direct comparison")
    if distribution_distances is not None:
        shape_shift = distribution_distances.get("shape_distribution_shift_mean_normalized_w1")
        pose_shift = distribution_distances.get("pose_distribution_shift_mean_normalized_w1")
        overall_shift = distribution_distances.get("distribution_shift_mean_normalized_w1")
        parts: list[str] = []
        if shape_shift is not None:
            parts.append(f"shape_mean_normalized_w1={float(shape_shift):.4f}")
        if pose_shift is not None:
            parts.append(f"pose_mean_normalized_w1={float(pose_shift):.4f}")
        if overall_shift is not None:
            parts.append(f"overall_mean_normalized_w1={float(overall_shift):.4f}")
        if parts:
            print(f"[sample] distribution shift: {', '.join(parts)}")
    restoration_summary = payload.get("restoration_summary")
    if isinstance(restoration_summary, dict):
        print(f"[sample] restoration summary: {format_restoration_summary(restoration_summary)}")
    print(f"[sample] saved metrics {metrics_out_path}")
    print(f"[sample] saved diagnostics {diagnostics_out_path}")
    return diagnostics_out_path
