"""Training loop for noisy-polygon guidance models."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import paths
from ..data.gen_polygons import regularity_score
from ..models.guidance_models import build_guidance_model
from ..models.diffusion import Diffusion, DiffusionConfig
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed


class _UnusedDenoiser(nn.Module):
    """Only used to reuse Diffusion.q_sample for guidance-model training."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover - never called
        raise RuntimeError("_UnusedDenoiser.forward should never be called")


def _load_scores(coords: np.ndarray, npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "score" in npz_data:
        return np.asarray(npz_data["score"], dtype=np.float32)
    return np.asarray([regularity_score(xy).score for xy in coords], dtype=np.float32)


def train_guidance_model_from_config(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    data_path = resolve_project_path(data_cfg.get("path", "data/raw/hexagons.npz"))
    batch_size = int(data_cfg.get("batch_size", 128))
    shuffle = bool(data_cfg.get("shuffle", True))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 2e-4))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 500))
    save_dir = paths.ensure_dir(resolve_project_path(train_cfg.get("save_dir", "models")))

    label_cfg = cfg.get("labels", {})
    label_type = str(label_cfg.get("type", "score_threshold")).lower()

    if label_type == "score_threshold":
        guidance_task = "classifier"
        checkpoint_name = str(train_cfg.get("checkpoint_name", "classifier_final.pt"))
    elif label_type in {"score_regression", "score_regressor", "score"}:
        guidance_task = "regressor"
        checkpoint_name = str(train_cfg.get("checkpoint_name", "regressor_final.pt"))
    else:
        raise ValueError(
            f"Unsupported labels.type {label_type!r}; expected one of: "
            "score_threshold, score_regression"
        )

    score_threshold = float(label_cfg.get("threshold", 0.8))

    model_cfg = dict(cfg.get("classifier", {}))
    model_cfg.setdefault("type", "mlp")

    diffusion_cfg = cfg.get("diffusion", {})
    diffusion_config = DiffusionConfig(
        n_steps=int(diffusion_cfg.get("n_steps", 1000)),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 2e-2)),
    )

    npz_data = np.load(data_path, allow_pickle=True)
    coords = np.asarray(npz_data["coords"], dtype=np.float32)
    n_vertices = int(npz_data["n"]) if "n" in npz_data else int(coords.shape[1])
    x = coords.reshape(coords.shape[0], -1)
    scores = _load_scores(coords, npz_data)

    if guidance_task == "classifier":
        targets_np = (scores >= score_threshold).astype(np.int64)
        target_tensor = torch.from_numpy(targets_np)
        num_classes = 2
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        targets_np = scores.astype(np.float32)
        target_tensor = torch.from_numpy(targets_np)
        num_classes = 1
        criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(x), target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    device = device_from_config(cfg)
    model = build_guidance_model(
        task=guidance_task,
        data_dim=x.shape[1],
        model_cfg=model_cfg,
        num_classes=max(2, num_classes),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    forward_diffusion = Diffusion(model=_UnusedDenoiser(), config=diffusion_config, device=device)

    print(
        f"[guidance-train] data={data_path} coords_shape={tuple(coords.shape)} "
        f"task={guidance_task} save_dir={save_dir}"
    )
    if guidance_task == "classifier":
        positive_fraction = float(targets_np.mean()) if len(targets_np) > 0 else 0.0
        print(
            f"[guidance-train] score_threshold={score_threshold:.4f} "
            f"positive_fraction={positive_fraction:.4f}"
        )
    else:
        print(
            f"[guidance-train] score_mean={float(scores.mean()):.4f} "
            f"score_std={float(scores.std()):.4f}"
        )
    print(
        f"[guidance-train] model_cfg={model_cfg} diffusion={diffusion_config}"
    )

    global_step = 0
    ema_loss: float | None = None
    start_time = time.perf_counter()

    for epoch in range(epochs):
        for batch_x, batch_target in loader:
            model.train()
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)

            t = torch.randint(0, diffusion_config.n_steps, (batch_x.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(batch_x)
            x_t = forward_diffusion.q_sample(batch_x, t, noise)
            pred = model(x_t, t)

            if guidance_task == "classifier":
                loss = criterion(pred, batch_target)
            else:
                batch_target = batch_target.float()
                loss = criterion(pred, batch_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().item())
            ema_loss = loss_value if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_value)

            if global_step % log_every == 0:
                elapsed = time.perf_counter() - start_time
                log_msg = (
                    f"[guidance-train] epoch={epoch} step={global_step} "
                    f"loss={loss_value:.6f} ema_loss={float(ema_loss):.6f} "
                    f"t_mean={float(t.float().mean().item()):.2f} "
                    f"steps_per_sec={global_step / max(elapsed, 1e-12):.2f}"
                )
                with torch.no_grad():
                    if guidance_task == "classifier":
                        acc = float((pred.argmax(dim=1) == batch_target).float().mean().item())
                        log_msg += f" acc={acc:.4f}"
                    else:
                        mae = float((pred - batch_target).abs().mean().item())
                        pred_mean = float(pred.mean().item())
                        log_msg += f" mae={mae:.4f} pred_mean={pred_mean:.4f}"
                print(log_msg)

            if global_step > 0 and save_every > 0 and global_step % save_every == 0:
                checkpoint_path = save_dir / (
                    f"{'classifier' if guidance_task == 'classifier' else 'regressor'}_step_{global_step}.pt"
                )
                payload: dict[str, object] = {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "guidance_task": guidance_task,
                    "n_vertices": n_vertices,
                    "diffusion": {
                        "n_steps": diffusion_config.n_steps,
                        "beta_start": diffusion_config.beta_start,
                        "beta_end": diffusion_config.beta_end,
                    },
                    "training_data_path": str(data_path),
                    "config_path": str(config_path),
                    "global_step": global_step,
                }
                if guidance_task == "classifier":
                    payload["num_classes"] = 2
                    payload["label_info"] = {
                        "type": "score_threshold",
                        "threshold": score_threshold,
                        "positive_class": 1,
                    }
                else:
                    payload["target_info"] = {
                        "type": "score_regression",
                        "metric": "regularity_score",
                    }
                torch.save(payload, checkpoint_path)
                print(f"[guidance-train] saved checkpoint {checkpoint_path}")

            global_step += 1

    final_path = save_dir / checkpoint_name
    payload = {
        "model_state": model.state_dict(),
        "model_cfg": model_cfg,
        "guidance_task": guidance_task,
        "n_vertices": n_vertices,
        "diffusion": {
            "n_steps": diffusion_config.n_steps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
        },
        "training_data_path": str(data_path),
        "config_path": str(config_path),
        "global_step": global_step,
    }
    if guidance_task == "classifier":
        payload["num_classes"] = 2
        payload["label_info"] = {
            "type": "score_threshold",
            "threshold": score_threshold,
            "positive_class": 1,
        }
    else:
        payload["target_info"] = {
            "type": "score_regression",
            "metric": "regularity_score",
        }
    torch.save(payload, final_path)
    print(f"[guidance-train] saved final checkpoint {final_path} at step={global_step}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "train_guidance_model.yaml"),
        help="path to guidance-model training config",
    )
    args = parser.parse_args()
    train_guidance_model_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
