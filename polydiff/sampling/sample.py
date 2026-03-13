"""Sampling entrypoint for polygon diffusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .. import paths
from ..models.diffusion import DenoiseMLP, Diffusion, DiffusionConfig

DEFAULT_ANIMATION_OUT_PATH = "data/processed/sample_trajectory.gif"


@dataclass(frozen=True, slots=True)
class SampleCliOverrides:
    enable_animation: bool = False
    animation_out_path: str | None = None
    animation_sample_index: int | None = None
    animation_max_frames: int | None = None
    animation_fps: int | None = None


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(cfg: Dict[str, Any]) -> torch.device:
    device = cfg.get("device", "auto")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        return paths.PROJECT_ROOT / p
    return p


def _normalize_animation_out_path(path_like: str | Path) -> Path:
    out_path = _resolve_project_path(path_like)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".gif")
    if out_path.suffix.lower() != ".gif":
        raise ValueError(f"animation output must be a .gif file, got {out_path}")
    return out_path


def _resolve_animation_options(
    sampling_cfg: Dict[str, Any],
    cli_overrides: SampleCliOverrides,
) -> Dict[str, Any] | None:
    animation_cfg = sampling_cfg.get("animation", {})
    if animation_cfg is None:
        animation_cfg = {}
    if not isinstance(animation_cfg, dict):
        raise ValueError("sampling.animation must be a mapping if provided")

    out_path = animation_cfg.get("out_path")
    if cli_overrides.enable_animation:
        out_path = cli_overrides.animation_out_path or out_path or DEFAULT_ANIMATION_OUT_PATH

    if out_path is None:
        return None

    sample_index = animation_cfg.get("sample_index", 0)
    max_frames = animation_cfg.get("max_frames", 120)
    fps = animation_cfg.get("fps", 12)

    if cli_overrides.animation_sample_index is not None:
        sample_index = cli_overrides.animation_sample_index
    if cli_overrides.animation_max_frames is not None:
        max_frames = cli_overrides.animation_max_frames
    if cli_overrides.animation_fps is not None:
        fps = cli_overrides.animation_fps

    sample_index = int(sample_index)
    max_frames = int(max_frames)
    fps = int(fps)

    if sample_index < 0:
        raise ValueError(f"animation sample_index must be >= 0, got {sample_index}")
    if max_frames < 2:
        raise ValueError(f"animation max_frames must be at least 2, got {max_frames}")
    if fps < 1:
        raise ValueError(f"animation fps must be at least 1, got {fps}")

    return {
        "out_path": _normalize_animation_out_path(out_path),
        "sample_index": sample_index,
        "max_frames": max_frames,
        "fps": fps,
    }


def _save_animation(trajectory: torch.Tensor, n_vertices: int, animation_cfg: Dict[str, Any]) -> Path:
    from ..data.plot_polygons import save_polygon_animation

    coords = trajectory.numpy().reshape(trajectory.shape[0], n_vertices, 2).astype(np.float32)
    return save_polygon_animation(
        coords,
        animation_cfg["out_path"],
        fps=int(animation_cfg["fps"]),
        max_frames=int(animation_cfg["max_frames"]),
    )


def sample_from_config(config_path: Path, *, cli_overrides: SampleCliOverrides | None = None) -> None:
    cli_overrides = cli_overrides or SampleCliOverrides()
    cfg = _load_config(config_path)

    seed = int(cfg.get("seed", 0))
    _set_seed(seed)

    model_cfg = cfg.get("model", {})
    checkpoint_path = _resolve_project_path(model_cfg.get("checkpoint", "models/diffusion_final.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    diffusion_cfg = checkpoint.get("diffusion", {})
    model_cfg_ckpt = checkpoint.get("model_cfg", {})
    n_vertices = int(checkpoint.get("n_vertices", 6))

    hidden_dim = int(model_cfg_ckpt.get("hidden_dim", 256))
    time_emb_dim = int(model_cfg_ckpt.get("time_emb_dim", 64))
    num_layers = int(model_cfg_ckpt.get("num_layers", 3))

    data_dim = n_vertices * 2
    model = DenoiseMLP(data_dim=data_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim, num_layers=num_layers)
    model.load_state_dict(checkpoint["model_state"])

    diffusion_config = DiffusionConfig(
        n_steps=int(diffusion_cfg.get("n_steps", 1000)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 2e-2)),
    )

    device = _device_from_config(cfg)
    diffusion = Diffusion(model=model, config=diffusion_config, device=device)

    sampling_cfg = cfg.get("sampling", {})
    num_samples = int(sampling_cfg.get("num_samples", 64))
    sampling_n_steps = sampling_cfg.get("n_steps")
    if sampling_n_steps is None:
        sampling_n_steps = diffusion_config.n_steps
    else:
        sampling_n_steps = int(sampling_n_steps)
    if sampling_n_steps < 1 or sampling_n_steps > diffusion_config.n_steps:
        raise ValueError(
            f"sampling.n_steps must be in [1, {diffusion_config.n_steps}] "
            f"(checkpoint diffusion.n_steps), got {sampling_n_steps}"
        )
    animation_cfg = _resolve_animation_options(sampling_cfg, cli_overrides)
    if animation_cfg is not None and animation_cfg["sample_index"] >= num_samples:
        raise ValueError(
            f"animation sample_index must be in [0, {num_samples - 1}] for num_samples={num_samples}, "
            f"got {animation_cfg['sample_index']}"
        )

    out_path = _resolve_project_path(sampling_cfg.get("out_path", "data/processed/samples.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if animation_cfg is None:
            samples = diffusion.p_sample_loop((num_samples, data_dim), n_steps=sampling_n_steps)
            trajectory = None
        else:
            samples, trajectory = diffusion.p_sample_loop_trajectory(
                (num_samples, data_dim),
                n_steps=sampling_n_steps,
                trajectory_index=int(animation_cfg["sample_index"]),
            )

    samples = samples.detach().cpu().numpy().reshape(num_samples, n_vertices, 2).astype(np.float32)

    np.savez_compressed(
        out_path,
        coords=samples,
        n=np.int32(n_vertices),
        meta=dict(
            checkpoint=str(checkpoint_path),
            num_samples=num_samples,
            n_steps=sampling_n_steps,
            checkpoint_n_steps=diffusion_config.n_steps,
        ),
    )

    print(
        f"[sample] saved {out_path} with {num_samples} samples "
        f"(sampling n_steps={sampling_n_steps}, checkpoint n_steps={diffusion_config.n_steps})"
    )
    if trajectory is not None:
        animation_out = _save_animation(trajectory, n_vertices, animation_cfg)
        print(
            f"[sample] saved animation {animation_out} "
            f"(sample_index={animation_cfg['sample_index']}, max_frames={animation_cfg['max_frames']}, fps={animation_cfg['fps']})"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "sample_diffusion.yaml"),
        help="path to sampling config",
    )
    p.add_argument(
        "--save_animation",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="save a GIF of one sample's reverse diffusion trajectory; optionally provide an output path",
    )
    p.add_argument(
        "--animation_index",
        type=int,
        default=None,
        help="sample index within the batch to animate",
    )
    p.add_argument(
        "--animation_max_frames",
        type=int,
        default=None,
        help="maximum number of frames to encode in the GIF",
    )
    p.add_argument(
        "--animation_fps",
        type=int,
        default=None,
        help="GIF playback rate",
    )
    args = p.parse_args()

    config_path = _resolve_project_path(args.config)
    sample_from_config(
        config_path,
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
