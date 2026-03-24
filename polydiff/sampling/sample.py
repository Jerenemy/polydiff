"""Sampling entrypoint for polygon diffusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..runs import (
    create_sampling_run_paths,
    infer_run_name_from_checkpoint_path,
    latest_model_run_dir,
    resolve_model_run_dir,
    slugify,
    write_sampling_run_files,
)
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed
from .guidance import load_sampling_guidance
from .runtime import (
    DEFAULT_ANIMATION_DIR_NAME,
    DEFAULT_SAMPLES_OUT_NAME,
    animation_sample_indices,
    load_diffusion_from_checkpoint,
    resolve_sampling_request,
    save_animations,
    write_sampling_diagnostics,
)


@dataclass(frozen=True, slots=True)
class SampleCliOverrides:
    run: str | None = None
    enable_animation: bool = False
    animation_out_path: str | None = None
    animation_count: int | None = None
    animation_sample_index: int | None = None
    animation_max_frames: int | None = None
    animation_fps: int | None = None


def build_sampling_run_label(cfg: dict[str, object], sampling_cfg: dict[str, object]) -> str:
    parts = [str(cfg.get("experiment_name", "polydiff-sample"))]
    guidance_cfg = sampling_cfg.get("guidance")
    if isinstance(guidance_cfg, dict) and bool(guidance_cfg.get("enabled", False)):
        parts.append(str(guidance_cfg.get("kind", "classifier")).lower())
        parts.append("guided")
    else:
        parts.append("unguided")

    n_steps = sampling_cfg.get("n_steps")
    if n_steps is not None:
        parts.append(f"{int(n_steps)}steps")

    return "-".join(slugify(part) for part in parts if str(part).strip())


def resolve_checkpoint_from_run_selection(
    model_cfg: dict[str, object] | None,
    *,
    run_override: str | None,
) -> tuple[Path, str | None]:
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping if provided")

    run_value = run_override if run_override is not None else model_cfg.get("run")
    checkpoint_value = model_cfg.get("checkpoint")

    if run_value is not None:
        run_dir = resolve_model_run_dir(str(run_value))
        checkpoint_path = run_dir / "diffusion_final.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No diffusion_final.pt found in {run_dir}")
        return checkpoint_path, run_dir.name

    if checkpoint_value is None or str(checkpoint_value) == "models/diffusion_final.pt":
        try:
            run_dir = latest_model_run_dir()
            checkpoint_path = run_dir / "diffusion_final.pt"
            if checkpoint_path.exists():
                return checkpoint_path, run_dir.name
        except FileNotFoundError:
            pass

    if checkpoint_value is None:
        checkpoint_path = resolve_project_path("models/diffusion_final.pt")
    else:
        checkpoint_path = resolve_project_path(str(checkpoint_value))
    return checkpoint_path, infer_run_name_from_checkpoint_path(checkpoint_path)


def sample_from_config(config_path: Path, *, cli_overrides: SampleCliOverrides | None = None) -> None:
    cli_overrides = cli_overrides or SampleCliOverrides()
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = device_from_config(cfg)
    model_cfg = cfg.get("model") or {}
    sampling_cfg = cfg.get("sampling") or {}
    checkpoint_path, requested_run_name = resolve_checkpoint_from_run_selection(
        model_cfg,
        run_override=cli_overrides.run,
    )
    checkpoint, diffusion, diffusion_config, n_vertices = load_diffusion_from_checkpoint(
        checkpoint_path,
        device=device,
    )
    run_name = str(checkpoint.get("run_name") or requested_run_name or infer_run_name_from_checkpoint_path(checkpoint_path) or "")
    sample_run_paths = None
    if run_name:
        sample_run_paths = create_sampling_run_paths(
            run_name=run_name,
            label=build_sampling_run_label(cfg, sampling_cfg),
        )
        default_out_path = sample_run_paths.processed_dir / DEFAULT_SAMPLES_OUT_NAME
        default_animation_out_path = sample_run_paths.media_dir / DEFAULT_ANIMATION_DIR_NAME
        print(
            f"[sample] using run {run_name} from checkpoint {checkpoint_path} "
            f"-> sampling subdir {sample_run_paths.sample_run_name}"
        )
    else:
        default_out_path = paths.PROCESSED_DATA_DIR / DEFAULT_SAMPLES_OUT_NAME
        default_animation_out_path = paths.PROCESSED_DATA_DIR / "media" / DEFAULT_ANIMATION_DIR_NAME
        print(f"[sample] using checkpoint {checkpoint_path}")

    request = resolve_sampling_request(
        sampling_cfg,
        checkpoint_n_steps=diffusion_config.n_steps,
        default_out_path=default_out_path,
        default_animation_out_path=default_animation_out_path,
        enable_animation=cli_overrides.enable_animation,
        animation_out_path=cli_overrides.animation_out_path,
        animation_count=cli_overrides.animation_count,
        animation_sample_index=cli_overrides.animation_sample_index,
        animation_max_frames=cli_overrides.animation_max_frames,
        animation_fps=cli_overrides.animation_fps,
    )
    request.out_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_run_paths is not None:
        resolved_cfg = {
            **cfg,
            "seed": seed,
            "device": str(device),
            "model": {
                "checkpoint": str(checkpoint_path),
                "run": run_name,
            },
            "sampling": {
                **sampling_cfg,
                "num_samples": request.num_samples,
                "n_steps": request.n_steps,
                "out_path": str(request.out_path),
                "diagnostics": {
                    **(sampling_cfg.get("diagnostics") or {}),
                    "enabled": request.diagnostics.enabled,
                    "out_path": None if request.diagnostics.out_path is None else str(request.diagnostics.out_path),
                    "reference_data_path": (
                        None
                        if request.diagnostics.reference_data_path is None
                        else str(request.diagnostics.reference_data_path)
                    ),
                },
            },
            "model_run_name": run_name,
            "sample_run_name": sample_run_paths.sample_run_name,
        }
        if request.animation is not None:
            resolved_cfg["sampling"]["animation"] = {
                **(sampling_cfg.get("animation") or {}),
                "out_path": str(request.animation.out_path),
                "sample_index": request.animation.sample_index,
                "count": request.animation.count,
                "max_frames": request.animation.max_frames,
                "fps": request.animation.fps,
            }
        write_sampling_run_files(
            sample_run_paths,
            config=resolved_cfg,
            config_path=config_path,
            extra_metadata={
                "checkpoint_path": str(checkpoint_path),
                "model_run_name": run_name,
                "sample_out_path": str(request.out_path),
            },
        )

    guidance_grad = None
    if request.guidance.enabled:
        _, guidance, guidance_n_vertices = load_sampling_guidance(
            request.guidance.checkpoint_path,
            device=device,
            kind=request.guidance.kind or "classifier",
            scale=request.guidance.scale,
            n_vertices=n_vertices,
            target_class=request.guidance.target_class,
            target_value=request.guidance.target_value,
            alpha=request.guidance.alpha,
            beta=request.guidance.beta,
            gamma=request.guidance.gamma,
        )
        if guidance_n_vertices != n_vertices:
            raise ValueError(
                f"guidance model n_vertices={guidance_n_vertices} does not match "
                f"diffusion checkpoint n_vertices={n_vertices}"
            )
        guidance_grad = guidance
        guidance_desc = f"scale={request.guidance.scale}"
        if request.guidance.kind == "regularity":
            guidance_desc += (
                f", analytic_score(alpha={request.guidance.alpha}, "
                f"beta={request.guidance.beta}, gamma={request.guidance.gamma})"
            )
        elif request.guidance.kind == "area":
            guidance_desc += ", analytic_area(smooth_absolute_shoelace)"
        else:
            guidance_desc += f", checkpoint={request.guidance.checkpoint_path}"
        if request.guidance.kind == "classifier":
            guidance_desc += f", target_class={request.guidance.target_class}"
        elif request.guidance.target_value is not None:
            guidance_desc += f", target_value={request.guidance.target_value}"
        elif request.guidance.kind == "regularity":
            guidance_desc += ", objective=maximize_regularity_score"
        elif request.guidance.kind == "area":
            guidance_desc += ", objective=maximize_area"
        else:
            guidance_desc += ", objective=maximize_predicted_score"
        print(f"[sample] enabled {request.guidance.kind} guidance ({guidance_desc})")

    if request.animation is None:
        samples = diffusion.p_sample_loop(
            (request.num_samples, n_vertices * 2),
            n_steps=request.n_steps,
            guidance_grad=guidance_grad,
        )
        trajectories = None
    else:
        samples, trajectories = diffusion.p_sample_loop_trajectories(
            (request.num_samples, n_vertices * 2),
            n_steps=request.n_steps,
            trajectory_indices=animation_sample_indices(request.animation),
            guidance_grad=guidance_grad,
        )

    samples = samples.detach().cpu().numpy().reshape(request.num_samples, n_vertices, 2).astype(np.float32)
    np.savez_compressed(
        request.out_path,
        coords=samples,
        n=np.int32(n_vertices),
        meta=dict(
            checkpoint=str(checkpoint_path),
            run_name=run_name or None,
            sample_run_name=None if sample_run_paths is None else sample_run_paths.sample_run_name,
            num_samples=request.num_samples,
            n_steps=request.n_steps,
            checkpoint_n_steps=diffusion_config.n_steps,
        ),
    )

    print(
        f"[sample] saved {request.out_path} with {request.num_samples} samples "
        f"(sampling n_steps={request.n_steps}, checkpoint n_steps={diffusion_config.n_steps})"
    )
    if trajectories is not None and request.animation is not None:
        animation_out_paths = save_animations(trajectories, n_vertices, request.animation)
        print(
            f"[sample] saved {len(animation_out_paths)} animation(s) under {animation_out_paths[0].parent} "
            f"(start_index={request.animation.sample_index}, count={request.animation.count}, "
            f"max_frames={request.animation.max_frames}, fps={request.animation.fps})"
        )

    write_sampling_diagnostics(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        samples_out_path=request.out_path,
        coords=samples,
        options=request.diagnostics,
        sampling_n_steps=request.n_steps,
        sample_run_name=None if sample_run_paths is None else sample_run_paths.sample_run_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "sample_diffusion.yaml"),
        help="path to sampling config",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="run directory name or run number to sample from; defaults to the most recent models/run_*",
    )
    parser.add_argument(
        "--save_animation",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="save reverse diffusion GIFs; optionally provide an output path or directory",
    )
    parser.add_argument(
        "--animation_count",
        type=int,
        default=None,
        help="number of sample trajectories to save as GIFs starting from --animation_index",
    )
    parser.add_argument(
        "--animation_index",
        type=int,
        default=None,
        help="starting sample index within the batch for animation export",
    )
    parser.add_argument(
        "--animation_max_frames",
        type=int,
        default=None,
        help="maximum number of frames to encode in the GIF",
    )
    parser.add_argument(
        "--animation_fps",
        type=int,
        default=None,
        help="GIF playback rate",
    )
    args = parser.parse_args()
    sample_from_config(
        resolve_project_path(args.config),
        cli_overrides=SampleCliOverrides(
            run=args.run,
            enable_animation=args.save_animation is not None,
            animation_out_path=args.save_animation or None,
            animation_count=args.animation_count,
            animation_sample_index=args.animation_index,
            animation_max_frames=args.animation_max_frames,
            animation_fps=args.animation_fps,
        ),
    )


if __name__ == "__main__":
    main()
