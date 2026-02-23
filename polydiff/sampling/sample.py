"""Sampling entrypoint for polygon diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .. import paths
from ..models.diffusion import DenoiseMLP, Diffusion, DiffusionConfig


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def sample_from_config(config_path: Path) -> None:
    cfg = _load_config(config_path)

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

    out_path = _resolve_project_path(sampling_cfg.get("out_path", "data/processed/samples.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        samples = diffusion.p_sample_loop((num_samples, data_dim), n_steps=sampling_n_steps)

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "sample_diffusion.yaml"),
        help="path to sampling config",
    )
    args = p.parse_args()

    config_path = _resolve_project_path(args.config)
    sample_from_config(config_path)


if __name__ == "__main__":
    main()
