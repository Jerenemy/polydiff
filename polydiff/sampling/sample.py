"""Sampling entrypoint for polygon diffusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed
from .runtime import (
    load_diffusion_from_checkpoint,
    resolve_sampling_request,
    save_animation,
    write_sampling_diagnostics,
)


@dataclass(frozen=True, slots=True)
class SampleCliOverrides:
    enable_animation: bool = False
    animation_out_path: str | None = None
    animation_sample_index: int | None = None
    animation_max_frames: int | None = None
    animation_fps: int | None = None


def sample_from_config(config_path: Path, *, cli_overrides: SampleCliOverrides | None = None) -> None:
    cli_overrides = cli_overrides or SampleCliOverrides()
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    checkpoint_path = resolve_project_path(cfg.get("model", {}).get("checkpoint", "models/diffusion_final.pt"))
    checkpoint, diffusion, diffusion_config, n_vertices = load_diffusion_from_checkpoint(
        checkpoint_path,
        device=device_from_config(cfg),
    )

    sampling_cfg = cfg.get("sampling", {})
    request = resolve_sampling_request(
        sampling_cfg,
        checkpoint_n_steps=diffusion_config.n_steps,
        enable_animation=cli_overrides.enable_animation,
        animation_out_path=cli_overrides.animation_out_path,
        animation_sample_index=cli_overrides.animation_sample_index,
        animation_max_frames=cli_overrides.animation_max_frames,
        animation_fps=cli_overrides.animation_fps,
    )
    request.out_path.parent.mkdir(parents=True, exist_ok=True)

    if request.animation is None:
        samples = diffusion.p_sample_loop((request.num_samples, n_vertices * 2), n_steps=request.n_steps)
        trajectory = None
    else:
        samples, trajectory = diffusion.p_sample_loop_trajectory(
            (request.num_samples, n_vertices * 2),
            n_steps=request.n_steps,
            trajectory_index=request.animation.sample_index,
        )

    samples = samples.detach().cpu().numpy().reshape(request.num_samples, n_vertices, 2).astype(np.float32)
    np.savez_compressed(
        request.out_path,
        coords=samples,
        n=np.int32(n_vertices),
        meta=dict(
            checkpoint=str(checkpoint_path),
            num_samples=request.num_samples,
            n_steps=request.n_steps,
            checkpoint_n_steps=diffusion_config.n_steps,
        ),
    )

    print(
        f"[sample] saved {request.out_path} with {request.num_samples} samples "
        f"(sampling n_steps={request.n_steps}, checkpoint n_steps={diffusion_config.n_steps})"
    )
    if trajectory is not None and request.animation is not None:
        animation_out = save_animation(trajectory, n_vertices, request.animation)
        print(
            f"[sample] saved animation {animation_out} "
            f"(sample_index={request.animation.sample_index}, "
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
        "--save_animation",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="save a GIF of one sample's reverse diffusion trajectory; optionally provide an output path",
    )
    parser.add_argument(
        "--animation_index",
        type=int,
        default=None,
        help="sample index within the batch to animate",
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
            enable_animation=args.save_animation is not None,
            animation_out_path=args.save_animation or None,
            animation_sample_index=args.animation_index,
            animation_max_frames=args.animation_max_frames,
            animation_fps=args.animation_fps,
        ),
    )


if __name__ == "__main__":
    main()
