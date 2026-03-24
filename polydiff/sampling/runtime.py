"""Sampling-side animation and diagnostics helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.diagnostics import (
    compare_polygon_summaries,
    format_polygon_delta_summary,
    format_polygon_summary,
    json_ready,
    summarize_polygon_dataset,
)
from ..data.gen_polygons import normalize_size_distribution
from ..data.polygon_dataset import load_polygon_dataset
from ..models.diffusion import Diffusion, DiffusionConfig, build_denoiser
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
class GuidanceOptions:
    enabled: bool
    kind: str | None
    checkpoint_path: Path | None
    scale: float
    target_class: int
    target_value: float | None
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True, slots=True)
class SizeDistributionOptions:
    values: tuple[int, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SamplingRequest:
    num_samples: int
    n_steps: int
    out_path: Path
    animation: AnimationOptions | None
    diagnostics: DiagnosticsOptions
    guidance: GuidanceOptions
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


def resolve_guidance_options(sampling_cfg: dict[str, Any]) -> GuidanceOptions:
    guidance_cfg = sampling_cfg.get("guidance", {})
    if guidance_cfg is None:
        guidance_cfg = {}
    if not isinstance(guidance_cfg, dict):
        raise ValueError("sampling.guidance must be a mapping if provided")

    enabled = bool(guidance_cfg.get("enabled", False))
    kind = guidance_cfg.get("kind")
    checkpoint_path = guidance_cfg.get("checkpoint")
    scale = float(guidance_cfg.get("scale", 1.0))
    target_class = int(guidance_cfg.get("target_class", 1))
    target_value_raw = guidance_cfg.get("target_value")
    target_value = None if target_value_raw is None else float(target_value_raw)
    alpha = float(guidance_cfg.get("alpha", 8.0))
    beta = float(guidance_cfg.get("beta", 5.0))
    gamma = float(guidance_cfg.get("gamma", 4.0))

    if not enabled:
        return GuidanceOptions(
            enabled=False,
            kind=None if kind is None else str(kind),
            checkpoint_path=None if checkpoint_path is None else resolve_project_path(checkpoint_path),
            scale=scale,
            target_class=target_class,
            target_value=target_value,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    kind_str = "classifier" if kind is None else str(kind).lower()
    if kind_str not in {"classifier", "regressor", "regularity", "area"}:
        raise ValueError(
            f"Unsupported sampling.guidance.kind {kind!r}; expected 'classifier', 'regressor', 'regularity', or 'area'. "
            "Future graph-based guidance models can reuse this interface."
        )
    if checkpoint_path is None and kind_str not in {"regularity", "area"}:
        raise ValueError("sampling.guidance.checkpoint is required when guidance is enabled")
    if scale < 0.0:
        raise ValueError(f"sampling.guidance.scale must be >= 0, got {scale}")

    return GuidanceOptions(
        enabled=True,
        kind=kind_str,
        checkpoint_path=None if checkpoint_path is None else resolve_project_path(checkpoint_path),
        scale=scale,
        target_class=target_class,
        target_value=target_value,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


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
    return SamplingRequest(
        num_samples=num_samples,
        n_steps=n_steps,
        out_path=out_path,
        animation=animation,
        diagnostics=resolve_diagnostics_options(sampling_cfg),
        guidance=resolve_guidance_options(sampling_cfg),
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
            saved_paths.append(
                save_polygon_animation(
                    trajectory_cpu.numpy().astype(np.float32),
                    out_path,
                    fps=options.fps,
                    max_frames=options.max_frames,
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
        saved_paths.append(
            save_polygon_animation(
                coords,
                out_path,
                fps=options.fps,
                max_frames=options.max_frames,
            )
        )
    return saved_paths


def default_diagnostics_out_path(samples_out_path: Path) -> Path:
    return samples_out_path.with_name(f"{samples_out_path.stem}.diagnostics.json")


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
) -> Path | None:
    if not options.enabled:
        return None

    generated_summary = summarize_polygon_dataset(coords, num_vertices=num_vertices)
    reference_summary, reference_path, reference_source = resolve_reference_summary(checkpoint, options)
    delta_vs_reference = (
        None if reference_summary is None else compare_polygon_summaries(reference_summary, generated_summary)
    )

    diagnostics_out_path = options.out_path or default_diagnostics_out_path(samples_out_path)
    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("global_step"),
        "run_name": checkpoint.get("run_name"),
        "sample_run_name": sample_run_name,
        "samples_path": str(samples_out_path),
        "sampling_n_steps": int(sampling_n_steps),
        "generated_summary": generated_summary,
        "reference_summary": reference_summary,
        "reference_data_path": None if reference_path is None else str(reference_path),
        "reference_summary_source": reference_source,
        "delta_vs_reference": delta_vs_reference,
    }

    diagnostics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[sample] generated summary: {format_polygon_summary(generated_summary)}")
    if delta_vs_reference is not None:
        print(f"[sample] generated vs reference: {format_polygon_delta_summary(delta_vs_reference)}")
    else:
        print("[sample] no reference summary available for direct comparison")
    print(f"[sample] saved diagnostics {diagnostics_out_path}")
    return diagnostics_out_path
