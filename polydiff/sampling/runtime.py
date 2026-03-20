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
from ..models.diffusion import DenoiseMLP, Diffusion, DiffusionConfig
from ..utils.runtime import resolve_project_path

DEFAULT_ANIMATION_OUT_PATH = "data/processed/sample_trajectory.gif"


@dataclass(frozen=True, slots=True)
class AnimationOptions:
    out_path: Path
    sample_index: int
    max_frames: int
    fps: int


@dataclass(frozen=True, slots=True)
class DiagnosticsOptions:
    enabled: bool
    out_path: Path | None
    reference_data_path: Path | None


@dataclass(frozen=True, slots=True)
class SamplingRequest:
    num_samples: int
    n_steps: int
    out_path: Path
    animation: AnimationOptions | None
    diagnostics: DiagnosticsOptions


def load_diffusion_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Diffusion, DiffusionConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_model_cfg = checkpoint.get("model_cfg", {})
    n_vertices = int(checkpoint.get("n_vertices", 6))
    model = DenoiseMLP(
        data_dim=n_vertices * 2,
        hidden_dim=int(checkpoint_model_cfg.get("hidden_dim", 256)),
        time_emb_dim=int(checkpoint_model_cfg.get("time_emb_dim", 64)),
        num_layers=int(checkpoint_model_cfg.get("num_layers", 3)),
    )
    model.load_state_dict(checkpoint["model_state"])

    checkpoint_diffusion_cfg = checkpoint.get("diffusion", {})
    diffusion_config = DiffusionConfig(
        n_steps=int(checkpoint_diffusion_cfg.get("n_steps", 1000)),
        beta_start=float(checkpoint_diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(checkpoint_diffusion_cfg.get("beta_end", 2e-2)),
    )
    diffusion = Diffusion(model=model, config=diffusion_config, device=device)
    return checkpoint, diffusion, diffusion_config, n_vertices


def normalize_animation_out_path(path_like: str | Path) -> Path:
    out_path = resolve_project_path(path_like)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".gif")
    if out_path.suffix.lower() != ".gif":
        raise ValueError(f"animation output must be a .gif file, got {out_path}")
    return out_path


def resolve_animation_options(
    sampling_cfg: dict[str, Any],
    *,
    enable_animation: bool,
    animation_out_path: str | None,
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
    if enable_animation:
        out_path = animation_out_path or out_path or DEFAULT_ANIMATION_OUT_PATH

    if out_path is None:
        return None

    sample_index = animation_cfg.get("sample_index", 0)
    max_frames = animation_cfg.get("max_frames", 120)
    fps = animation_cfg.get("fps", 12)

    if animation_sample_index is not None:
        sample_index = animation_sample_index
    if animation_max_frames is not None:
        max_frames = animation_max_frames
    if animation_fps is not None:
        fps = animation_fps

    sample_index = int(sample_index)
    max_frames = int(max_frames)
    fps = int(fps)

    if sample_index < 0:
        raise ValueError(f"animation sample_index must be >= 0, got {sample_index}")
    if max_frames < 2:
        raise ValueError(f"animation max_frames must be at least 2, got {max_frames}")
    if fps < 1:
        raise ValueError(f"animation fps must be at least 1, got {fps}")

    return AnimationOptions(
        out_path=normalize_animation_out_path(out_path),
        sample_index=sample_index,
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


def resolve_sampling_request(
    sampling_cfg: dict[str, Any],
    *,
    checkpoint_n_steps: int,
    enable_animation: bool,
    animation_out_path: str | None,
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
        animation_out_path=animation_out_path,
        animation_sample_index=animation_sample_index,
        animation_max_frames=animation_max_frames,
        animation_fps=animation_fps,
    )
    if animation is not None and animation.sample_index >= num_samples:
        raise ValueError(
            f"animation sample_index must be in [0, {num_samples - 1}] for num_samples={num_samples}, "
            f"got {animation.sample_index}"
        )

    out_path = resolve_project_path(sampling_cfg.get("out_path", "data/processed/samples.npz"))
    return SamplingRequest(
        num_samples=num_samples,
        n_steps=n_steps,
        out_path=out_path,
        animation=animation,
        diagnostics=resolve_diagnostics_options(sampling_cfg),
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


def default_diagnostics_out_path(samples_out_path: Path) -> Path:
    return samples_out_path.with_name(f"{samples_out_path.stem}.diagnostics.json")


def resolve_reference_summary(
    checkpoint: dict[str, Any],
    options: DiagnosticsOptions,
) -> tuple[dict[str, float | int] | None, Path | None, str | None]:
    if options.reference_data_path is not None:
        data = np.load(options.reference_data_path, allow_pickle=True)
        summary = summarize_polygon_dataset(np.asarray(data["coords"], dtype=np.float32))
        return summary, options.reference_data_path, "reference_file"

    summary = checkpoint.get("training_data_summary")
    training_data_path = checkpoint.get("training_data_path")
    if summary is not None:
        resolved_path = None if training_data_path is None else resolve_project_path(training_data_path)
        return summary, resolved_path, "checkpoint"

    if training_data_path is not None:
        resolved_path = resolve_project_path(training_data_path)
        if resolved_path.exists():
            data = np.load(resolved_path, allow_pickle=True)
            summary = summarize_polygon_dataset(np.asarray(data["coords"], dtype=np.float32))
            return summary, resolved_path, "training_data_file"

    return None, None, None


def write_sampling_diagnostics(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    config_path: Path,
    samples_out_path: Path,
    coords: np.ndarray,
    options: DiagnosticsOptions,
    sampling_n_steps: int,
) -> Path | None:
    if not options.enabled:
        return None

    generated_summary = summarize_polygon_dataset(coords)
    reference_summary, reference_path, reference_source = resolve_reference_summary(checkpoint, options)
    delta_vs_reference = (
        None if reference_summary is None else compare_polygon_summaries(reference_summary, generated_summary)
    )

    diagnostics_out_path = options.out_path or default_diagnostics_out_path(samples_out_path)
    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("global_step"),
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
