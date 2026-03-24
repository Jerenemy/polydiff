"""Training loop for polygon diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .. import paths
from ..data.diagnostics import summarize_polygon_dataset
from ..data.polygon_dataset import PolygonDatasetArrays, collate_polygon_graph_batch, load_polygon_dataset
from ..models.diffusion import Diffusion, DiffusionConfig, build_denoiser
from ..runs import create_run_paths, write_run_files
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed
from .runtime import (
    grad_norm,
    log_both_info,
    log_sample_diagnostic,
    log_training_end,
    log_training_start,
    log_training_step,
    param_norm,
    resolve_training_options,
    run_sample_diagnostics,
    save_checkpoint,
    setup_train_logger,
)


class _PolygonArrayDataset(Dataset[torch.Tensor]):
    def __init__(self, data: PolygonDatasetArrays) -> None:
        self.data = data

    def __len__(self) -> int:
        return self.data.num_polygons

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data.polygon(index))


def train_from_config(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    data_path = resolve_project_path(data_cfg.get("path", "data/raw/hexagons.npz"))
    batch_size = int(data_cfg.get("batch_size", 64))
    shuffle = bool(data_cfg.get("shuffle", True))
    num_workers = int(data_cfg.get("num_workers", 0))

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("type", "gat")
    model_type = str(model_cfg.get("type", "gat")).lower()

    polygon_data = load_polygon_dataset(data_path)
    coords_shape = tuple(polygon_data.coords.shape)
    n_vertices = int(polygon_data.num_vertices[0]) if polygon_data.is_uniform and polygon_data.num_polygons > 0 else None
    max_vertices = int(polygon_data.max_vertices)

    if model_type == "mlp":
        if not polygon_data.is_uniform:
            raise ValueError("model.type='mlp' only supports fixed-size polygon datasets")
        coords = polygon_data.to_dense()
        x = coords.reshape(coords.shape[0], -1)
        dataset = TensorDataset(torch.from_numpy(x))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        model_data_dim = x.shape[1]
    elif model_type in {"gat", "gcn"}:
        coords = None
        x = None
        dataset = _PolygonArrayDataset(polygon_data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_polygon_graph_batch,
        )
        model_data_dim = max_vertices * 2
    else:
        raise ValueError(f"Unsupported model.type {model_type!r}; expected one of: mlp, gat, gcn")

    model = build_denoiser(data_dim=model_data_dim, model_cfg=model_cfg)

    diff_cfg = cfg.get("diffusion", {})
    diffusion_config = DiffusionConfig(
        n_steps=int(diff_cfg.get("n_steps", 1000)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 2e-2)),
    )

    device = device_from_config(cfg)
    diffusion = Diffusion(model=model, config=diffusion_config, device=device)

    train_cfg = cfg.get("training", {})
    train_options = resolve_training_options(train_cfg, batch_size=batch_size)
    run_paths = create_run_paths(
        experiment_name=str(cfg.get("experiment_name", "polydiff-train")),
        model_type=str(model_cfg.get("type", "gat")),
        data_path=data_path,
        model_root=resolve_project_path(train_options.save_dir),
    )
    save_dir = run_paths.model_dir

    logger, log_path, metrics_path = setup_train_logger(save_dir)
    resolved_cfg = {
        **cfg,
        "experiment_name": str(cfg.get("experiment_name", "polydiff-train")),
        "seed": seed,
        "device": str(device),
        "data": {
            "path": str(data_path),
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        },
        "model": model_cfg,
        "diffusion": {
            "n_steps": diffusion_config.n_steps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
        },
        "training": {
            **train_cfg,
            "epochs": train_options.epochs,
            "lr": train_options.lr,
            "log_every": train_options.log_every,
            "save_every": train_options.save_every,
            "save_dir": str(resolve_project_path(train_options.save_dir)),
            "sample_diagnostics_every": train_options.sample_diagnostics_every,
            "sample_diagnostics_num_samples": train_options.sample_diagnostics_num_samples,
            "sample_diagnostics_n_steps": train_options.sample_diagnostics_n_steps,
            "sample_diagnostics_seed": train_options.sample_diagnostics_seed,
        },
        "run_name": run_paths.run_name,
    }
    write_run_files(
        run_paths,
        config=resolved_cfg,
        config_path=config_path,
        extra_metadata={
            "seed": seed,
            "data_path": str(data_path),
            "model_cfg": model_cfg,
            "diffusion_cfg": {
                "n_steps": diffusion_config.n_steps,
                "beta_start": diffusion_config.beta_start,
                "beta_end": diffusion_config.beta_end,
            },
            "training_cfg": train_cfg,
        },
    )
    reference_summary = summarize_polygon_dataset(polygon_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_options.lr)

    log_both_info(
        logger,
        "[train] run=%s model_dir=%s processed_dir=%s media_dir=%s",
        run_paths.run_name,
        run_paths.model_dir,
        run_paths.processed_dir,
        run_paths.media_dir,
    )

    log_training_start(
        logger=logger,
        metrics_path=metrics_path,
        config_path=config_path,
        seed=seed,
        device=device,
        data_path=data_path,
        coords_shape=coords_shape,
        model_cfg=model_cfg,
        diffusion_config=diffusion_config,
        training_cfg=train_cfg,
        training_data_summary=reference_summary,
        log_path=log_path,
    )

    global_step = 0
    ema_loss: float | None = None
    last_log_time = time.perf_counter()
    last_log_step = -1

    for epoch in range(train_options.epochs):
        for batch in loader:
            model.train()

            if model_type == "mlp":
                (batch_x,) = batch
                batch_x = batch_x.to(device)
                loss, loss_stats = diffusion.loss(batch_x, return_stats=True)
            else:
                batch_graph = batch.to(device)
                loss, loss_stats = diffusion.loss(batch_graph.coords, graph_batch=batch_graph, return_stats=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            current_grad_norm = grad_norm(model.parameters())
            optimizer.step()

            loss_value = float(loss_stats["loss"])
            ema_loss = loss_value if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_value)

            if global_step % train_options.log_every == 0:
                now = time.perf_counter()
                steps_since_last = global_step - last_log_step
                steps_per_sec = steps_since_last / max(now - last_log_time, 1e-12)
                current_param_norm = param_norm(model.parameters())
                log_training_step(
                    logger=logger,
                    metrics_path=metrics_path,
                    epoch=epoch,
                    step=global_step,
                    ema_loss=float(ema_loss),
                    grad_norm_value=float(current_grad_norm),
                    param_norm_value=float(current_param_norm),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    steps_per_sec=float(steps_per_sec),
                    loss_stats=loss_stats,
                )
                last_log_time = now
                last_log_step = global_step

            if (
                train_options.sample_diagnostics_every > 0
                and global_step > 0
                and global_step % train_options.sample_diagnostics_every == 0
            ):
                sample_summary, sample_deltas = run_sample_diagnostics(
                    diffusion,
                    num_samples=train_options.sample_diagnostics_num_samples,
                    data_dim=None if x is None else x.shape[1],
                    n_vertices=n_vertices,
                    reference_num_vertices=None if model_type == "mlp" else polygon_data.num_vertices,
                    n_steps=train_options.sample_diagnostics_n_steps,
                    seed=train_options.sample_diagnostics_seed,
                    reference_summary=reference_summary,
                )
                log_sample_diagnostic(
                    logger=logger,
                    metrics_path=metrics_path,
                    epoch=epoch,
                    step=global_step,
                    num_samples=train_options.sample_diagnostics_num_samples,
                    n_steps=(
                        diffusion_config.n_steps
                        if train_options.sample_diagnostics_n_steps is None
                        else train_options.sample_diagnostics_n_steps
                    ),
                    sample_summary=sample_summary,
                    sample_deltas=sample_deltas,
                )

            if global_step % train_options.save_every == 0 and global_step > 0:
                checkpoint_path = save_dir / f"diffusion_step_{global_step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    diffusion_config=diffusion_config,
                    model_cfg=model_cfg,
                    n_vertices=n_vertices,
                    max_vertices=max_vertices,
                    global_step=global_step,
                    run_name=run_paths.run_name,
                    training_data_path=data_path,
                    training_data_summary=reference_summary,
                    config_path=config_path,
                )
                log_both_info(logger, "[train] saved checkpoint %s", checkpoint_path)

            global_step += 1

    final_path = save_dir / "diffusion_final.pt"
    save_checkpoint(
        final_path,
        model=model,
        diffusion_config=diffusion_config,
        model_cfg=model_cfg,
        n_vertices=n_vertices,
        max_vertices=max_vertices,
        global_step=global_step,
        run_name=run_paths.run_name,
        training_data_path=data_path,
        training_data_summary=reference_summary,
        config_path=config_path,
    )
    log_training_end(logger=logger, metrics_path=metrics_path, final_checkpoint=final_path, final_step=global_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "train_diffusion.yaml"),
        help="path to training config",
    )
    args = parser.parse_args()
    train_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
