"""Training loop for polygon diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import paths
from ..data.diagnostics import summarize_polygon_dataset
from ..models.diffusion import Diffusion, DiffusionConfig, build_denoiser
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


def train_from_config(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    data_path = resolve_project_path(data_cfg.get("path", "data/raw/hexagons.npz"))
    batch_size = int(data_cfg.get("batch_size", 64))
    shuffle = bool(data_cfg.get("shuffle", True))
    num_workers = int(data_cfg.get("num_workers", 0))

    data = np.load(data_path)
    coords = data["coords"].astype(np.float32)
    n_vertices = coords.shape[1]
    x = coords.reshape(coords.shape[0], -1)

    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model_cfg = dict(cfg.get("model", {}))
    model_cfg.setdefault("type", "gat")
    model = build_denoiser(data_dim=x.shape[1], model_cfg=model_cfg)

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
    save_dir = paths.ensure_dir(resolve_project_path(train_options.save_dir))

    logger, log_path, metrics_path = setup_train_logger(save_dir)
    reference_summary = summarize_polygon_dataset(coords)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_options.lr)

    log_training_start(
        logger=logger,
        metrics_path=metrics_path,
        config_path=config_path,
        seed=seed,
        device=device,
        data_path=data_path,
        coords_shape=tuple(coords.shape),
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
        for (batch_x,) in loader:
            model.train()

            batch_x = batch_x.to(device)
            loss, loss_stats = diffusion.loss(batch_x, return_stats=True)

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
                    data_dim=x.shape[1],
                    n_vertices=n_vertices,
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
                    global_step=global_step,
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
        global_step=global_step,
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
