"""Training loop for polygon diffusion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .. import paths
from ..models.diffusion import DenoiseMLP, Diffusion, DiffusionConfig


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


def train_from_config(config_path: Path) -> None:
    cfg = _load_config(config_path)

    seed = int(cfg.get("seed", 0))
    _set_seed(seed)

    data_cfg = cfg.get("data", {})
    data_path = _resolve_project_path(data_cfg.get("path", "data/raw/hexagons.npz"))
    batch_size = int(data_cfg.get("batch_size", 64))
    shuffle = bool(data_cfg.get("shuffle", True))
    num_workers = int(data_cfg.get("num_workers", 0))

    data = np.load(data_path)
    coords = data["coords"].astype(np.float32)
    n = coords.shape[1]
    x = coords.reshape(coords.shape[0], -1)

    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model_cfg = cfg.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    time_emb_dim = int(model_cfg.get("time_emb_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 3))

    diff_cfg = cfg.get("diffusion", {})
    diffusion_config = DiffusionConfig(
        n_steps=int(diff_cfg.get("n_steps", 1000)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 2e-2)),
    )

    device = _device_from_config(cfg)

    model = DenoiseMLP(data_dim=x.shape[1], hidden_dim=hidden_dim, time_emb_dim=time_emb_dim, num_layers=num_layers)
    diffusion = Diffusion(model=model, config=diffusion_config, device=device)

    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-4))
    log_every = int(train_cfg.get("log_every", 100))
    save_every = int(train_cfg.get("save_every", 1000))
    save_dir = paths.ensure_dir(_resolve_project_path(train_cfg.get("save_dir", "models")))

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            loss = diffusion.loss(batch_x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % log_every == 0:
                print(f"[train] epoch {epoch} step {global_step} loss {loss.item():.6f}")

            if global_step % save_every == 0 and global_step > 0:
                ckpt_path = save_dir / f"diffusion_step_{global_step}.pt"
                _save_checkpoint(
                    ckpt_path,
                    model,
                    diffusion_config,
                    model_cfg,
                    n,
                )
                print(f"[train] saved {ckpt_path}")

            global_step += 1

    final_path = save_dir / "diffusion_final.pt"
    _save_checkpoint(final_path, model, diffusion_config, model_cfg, n)
    print(f"[train] saved {final_path}")


def _save_checkpoint(
    path: Path,
    model: DenoiseMLP,
    diffusion_config: DiffusionConfig,
    model_cfg: Dict[str, Any],
    n_vertices: int,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "diffusion": asdict(diffusion_config),
        "model_cfg": model_cfg,
        "n_vertices": n_vertices,
    }
    torch.save(ckpt, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "train_diffusion.yaml"),
        help="path to training config",
    )
    args = p.parse_args()

    config_path = _resolve_project_path(args.config)
    train_from_config(config_path)


if __name__ == "__main__":
    main()
