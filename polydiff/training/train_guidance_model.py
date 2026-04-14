"""Training loop for noisy-polygon guidance models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .. import paths
from ..data.diagnostics import summarize_polygon_dataset
from ..data.gen_polygons import regularity_score
from ..data.polygon_dataset import PolygonGraphBatch, collate_polygon_graph_batch, load_polygon_dataset
from ..models.guidance_models import build_guidance_model
from ..models.diffusion import Diffusion, DiffusionConfig
from ..runs import create_run_paths, write_run_files
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed
from .runtime import append_jsonl, log_both_info, setup_train_logger


@dataclass(frozen=True, slots=True)
class GuidanceTrainResult:
    run_name: str
    model_dir: Path
    processed_dir: Path
    checkpoint_path: Path
    log_path: Path
    metrics_path: Path
    guidance_task: str
    training_data_path: Path
    config_path: Path


class _UnusedDenoiser(nn.Module):
    """Only used to reuse Diffusion.q_sample for guidance-model training."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover - never called
        raise RuntimeError("_UnusedDenoiser.forward should never be called")


class _PolygonTargetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, polygon_data, targets: np.ndarray) -> None:
        self.polygon_data = polygon_data
        self.targets = np.asarray(targets)

    def __len__(self) -> int:
        return int(self.polygon_data.num_polygons)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        polygon = torch.from_numpy(self.polygon_data.polygon(index)).to(torch.float32)
        target = torch.as_tensor(self.targets[index])
        return polygon, target


def _collate_polygon_targets(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[PolygonGraphBatch, torch.Tensor]:
    polygons, targets = zip(*items, strict=True)
    graph_batch = collate_polygon_graph_batch(list(polygons))
    target_tensor = torch.stack([target.reshape(()) for target in targets], dim=0)
    return graph_batch, target_tensor


def _load_scores(polygon_data, npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "score" in npz_data:
        return np.asarray(npz_data["score"], dtype=np.float32)
    return np.asarray([regularity_score(polygon_data.polygon(i)).score for i in range(polygon_data.num_polygons)], dtype=np.float32)


def _save_guidance_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    model_cfg: dict[str, object],
    guidance_task: str,
    diffusion_config: DiffusionConfig,
    n_vertices: int | None,
    max_vertices: int,
    training_data_path: Path,
    training_data_summary: dict[str, float | int | dict[int, int]],
    config_path: Path,
    global_step: int,
    run_name: str,
    score_threshold: float,
) -> None:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "model_cfg": model_cfg,
        "guidance_task": guidance_task,
        "n_vertices": n_vertices,
        "max_vertices": max_vertices,
        "diffusion": {
            "n_steps": diffusion_config.n_steps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
        },
        "training_data_path": str(training_data_path),
        "training_data_summary": training_data_summary,
        "config_path": str(config_path),
        "global_step": global_step,
        "run_name": run_name,
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
    torch.save(payload, path)


def train_guidance_model_from_loaded_config(cfg: dict[str, object], *, config_path: Path) -> GuidanceTrainResult:
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
    save_dir_root = resolve_project_path(train_cfg.get("save_dir", "models"))

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

    polygon_data = load_polygon_dataset(data_path)
    with np.load(data_path, allow_pickle=True) as npz_data:
        scores = _load_scores(polygon_data, npz_data)
    training_data_summary = summarize_polygon_dataset(polygon_data)
    coords_shape = tuple(polygon_data.coords.shape)
    n_vertices = int(polygon_data.num_vertices[0]) if polygon_data.is_uniform else None
    max_vertices = int(polygon_data.max_vertices)
    model_type = str(model_cfg.get("type", "mlp")).lower()

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

    if model_type == "mlp":
        if not polygon_data.is_uniform:
            raise ValueError("guidance-model training with model.type='mlp' supports fixed-size polygon datasets only")
        coords = polygon_data.to_dense()
        x = coords.reshape(coords.shape[0], -1)
        assert x is not None
        dataset = TensorDataset(torch.from_numpy(x), target_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        model_data_dim = x.shape[1]
    elif model_type in {"gat", "gcn"}:
        coords = None
        x = None
        dataset = _PolygonTargetDataset(polygon_data, targets_np)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_polygon_targets,
        )
        model_data_dim = max_vertices * 2
    else:
        raise ValueError(f"Unsupported classifier.type {model_type!r}; expected one of: mlp, gat, gcn")

    device = device_from_config(cfg)
    model = build_guidance_model(
        task=guidance_task,
        data_dim=model_data_dim,
        model_cfg=model_cfg,
        num_classes=max(2, num_classes),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    forward_diffusion = Diffusion(model=_UnusedDenoiser(), config=diffusion_config, device=device)
    run_paths = create_run_paths(
        experiment_name=str(cfg.get("experiment_name", "polydiff-guidance-train")),
        model_type=f"{model_type}-{guidance_task}",
        data_path=data_path,
        model_root=save_dir_root,
    )
    save_dir = run_paths.model_dir
    logger, log_path, metrics_path = setup_train_logger(save_dir)
    resolved_cfg = {
        **cfg,
        "experiment_name": str(cfg.get("experiment_name", "polydiff-guidance-train")),
        "seed": seed,
        "device": str(device),
        "data": {
            "path": str(data_path),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        },
        "classifier": model_cfg,
        "diffusion": {
            "n_steps": diffusion_config.n_steps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
        },
        "labels": {
            **label_cfg,
            "type": label_type,
            "threshold": score_threshold,
        },
        "training": {
            **train_cfg,
            "epochs": epochs,
            "lr": lr,
            "log_every": log_every,
            "save_every": save_every,
            "save_dir": str(save_dir_root),
            "checkpoint_name": checkpoint_name,
        },
        "run_name": run_paths.run_name,
    }
    write_run_files(
        run_paths,
        config=resolved_cfg,
        config_path=config_path,
        extra_metadata={
            "seed": seed,
            "guidance_task": guidance_task,
            "data_path": str(data_path),
            "model_cfg": model_cfg,
            "diffusion_cfg": {
                "n_steps": diffusion_config.n_steps,
                "beta_start": diffusion_config.beta_start,
                "beta_end": diffusion_config.beta_end,
            },
            "labels_cfg": label_cfg,
            "training_cfg": train_cfg,
        },
    )

    log_both_info(
        logger,
        "[guidance-train] run=%s model_dir=%s processed_dir=%s media_dir=%s",
        run_paths.run_name,
        run_paths.model_dir,
        run_paths.processed_dir,
        run_paths.media_dir,
    )
    log_both_info(
        logger,
        "[guidance-train] data=%s coords_shape=%s task=%s save_dir=%s",
        data_path,
        coords_shape,
        guidance_task,
        save_dir,
    )
    append_jsonl(
        metrics_path,
        {
            "event": "train_start",
            "run_name": run_paths.run_name,
            "config_path": str(config_path),
            "guidance_task": guidance_task,
            "seed": seed,
            "device": str(device),
            "data_path": str(data_path),
            "coords_shape": list(coords_shape),
            "model_cfg": model_cfg,
            "diffusion_cfg": {
                "n_steps": diffusion_config.n_steps,
                "beta_start": diffusion_config.beta_start,
                "beta_end": diffusion_config.beta_end,
            },
            "training_cfg": {
                **train_cfg,
                "epochs": epochs,
                "lr": lr,
                "log_every": log_every,
                "save_every": save_every,
                "save_dir": str(save_dir_root),
                "checkpoint_name": checkpoint_name,
            },
            "training_data_summary": training_data_summary,
        },
    )
    if guidance_task == "classifier":
        positive_fraction = float(targets_np.mean()) if len(targets_np) > 0 else 0.0
        log_both_info(
            logger,
            "[guidance-train] score_threshold=%.4f positive_fraction=%.4f",
            score_threshold,
            positive_fraction,
        )
    else:
        log_both_info(
            logger,
            "[guidance-train] score_mean=%.4f score_std=%.4f",
            float(scores.mean()),
            float(scores.std()),
        )
    log_both_info(logger, "[guidance-train] model_cfg=%s diffusion=%s", model_cfg, diffusion_config)

    global_step = 0
    ema_loss: float | None = None
    start_time = time.perf_counter()

    for epoch in range(epochs):
        for batch in loader:
            model.train()
            if model_type == "mlp":
                batch_x, batch_target = batch
                batch_x = batch_x.to(device)
                batch_target = batch_target.to(device)
                t = torch.randint(0, diffusion_config.n_steps, (batch_x.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(batch_x)
                x_t = forward_diffusion.q_sample(batch_x, t, noise)
                pred = model(x_t, t)
            else:
                batch_graph, batch_target = batch
                batch_graph = batch_graph.to(device)
                batch_target = batch_target.to(device)
                t = torch.randint(0, diffusion_config.n_steps, (batch_graph.batch_size,), device=device, dtype=torch.long)
                noise = torch.randn_like(batch_graph.coords)
                x_t = forward_diffusion.q_sample(batch_graph.coords, t, noise, graph_batch=batch_graph)
                pred = model(x_t, t, batch=batch_graph)

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
                        metrics_payload = {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": loss_value,
                            "ema_loss": float(ema_loss),
                            "t_mean": float(t.float().mean().item()),
                            "steps_per_sec": float(global_step / max(elapsed, 1e-12)),
                            "acc": acc,
                        }
                    else:
                        mae = float((pred - batch_target).abs().mean().item())
                        pred_mean = float(pred.mean().item())
                        log_msg += f" mae={mae:.4f} pred_mean={pred_mean:.4f}"
                        metrics_payload = {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": loss_value,
                            "ema_loss": float(ema_loss),
                            "t_mean": float(t.float().mean().item()),
                            "steps_per_sec": float(global_step / max(elapsed, 1e-12)),
                            "mae": mae,
                            "pred_mean": pred_mean,
                        }
                log_both_info(logger, "%s", log_msg)
                append_jsonl(metrics_path, metrics_payload)

            if global_step > 0 and save_every > 0 and global_step % save_every == 0:
                checkpoint_path = save_dir / (
                    f"{'classifier' if guidance_task == 'classifier' else 'regressor'}_step_{global_step}.pt"
                )
                _save_guidance_checkpoint(
                    checkpoint_path,
                    model=model,
                    model_cfg=model_cfg,
                    guidance_task=guidance_task,
                    diffusion_config=diffusion_config,
                    n_vertices=n_vertices,
                    max_vertices=max_vertices,
                    training_data_path=data_path,
                    training_data_summary=training_data_summary,
                    config_path=config_path,
                    global_step=global_step,
                    run_name=run_paths.run_name,
                    score_threshold=score_threshold,
                )
                log_both_info(logger, "[guidance-train] saved checkpoint %s", checkpoint_path)
                append_jsonl(
                    metrics_path,
                    {
                        "event": "checkpoint",
                        "step": global_step,
                        "checkpoint_path": str(checkpoint_path),
                    },
                )

            global_step += 1

    final_path = save_dir / checkpoint_name
    _save_guidance_checkpoint(
        final_path,
        model=model,
        model_cfg=model_cfg,
        guidance_task=guidance_task,
        diffusion_config=diffusion_config,
        n_vertices=n_vertices,
        max_vertices=max_vertices,
        training_data_path=data_path,
        training_data_summary=training_data_summary,
        config_path=config_path,
        global_step=global_step,
        run_name=run_paths.run_name,
        score_threshold=score_threshold,
    )
    log_both_info(logger, "[guidance-train] saved final checkpoint %s at step=%d", final_path, global_step)
    append_jsonl(
        metrics_path,
        {
            "event": "train_end",
            "final_checkpoint_path": str(final_path),
            "final_step": global_step,
        },
    )
    return GuidanceTrainResult(
        run_name=run_paths.run_name,
        model_dir=run_paths.model_dir,
        processed_dir=run_paths.processed_dir,
        checkpoint_path=final_path,
        log_path=log_path,
        metrics_path=metrics_path,
        guidance_task=guidance_task,
        training_data_path=data_path,
        config_path=config_path,
    )


def train_guidance_model_from_config(config_path: Path) -> GuidanceTrainResult:
    cfg = load_yaml_config(config_path)
    return train_guidance_model_from_loaded_config(cfg, config_path=config_path)


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
