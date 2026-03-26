"""Train pooled ligand-context surrogate models for guidance experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from polydiff import paths
from polydiff.runs import create_run_paths, slugify
from polydiff.utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed

from .surrogate_data import (
    LigandContextSurrogateDataset,
    SurrogateBatch,
    SurrogateNoiseSchedule,
    collate_surrogate_graphs,
)
from .surrogate_models import LigandContextSurrogateModel


@dataclass(frozen=True, slots=True)
class SurrogateTrainResult:
    checkpoint_path: Path
    final_checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    config_path: Path
    run_name: str
    model_dir: Path
    data_source_path: Path
    best_epoch: int
    best_val_mae: float
    test_mae: float
    test_rmse: float
    test_pearson: float
    pooling: str
    timestep_conditioning: bool
    noisy_training: bool


def train_surrogate_model_from_loaded_config(
    cfg: dict[str, Any],
    *,
    config_path: Path,
) -> SurrogateTrainResult:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping")
    pt_path = data_cfg.get("pt_path")
    pt_dir = data_cfg.get("pt_dir")
    pdb_path = resolve_project_path(data_cfg.get("pdb_path", ""))
    if not str(pdb_path):
        raise ValueError("data.pdb_path is required for surrogate training")
    vina_mode = str(data_cfg.get("vina_mode", "score_only"))
    max_affinity = data_cfg.get("max_affinity")
    results_key = data_cfg.get("results_key")
    r_ligand = float(data_cfg.get("r_ligand", 5.0))
    r_cross = float(data_cfg.get("r_cross", 6.0))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    test_fraction = float(data_cfg.get("test_fraction", 0.1))
    split_seed = int(data_cfg.get("split_seed", seed))
    if val_fraction < 0.0 or test_fraction < 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError("data.val_fraction and data.test_fraction must be >= 0 and sum to < 1")

    dataset = LigandContextSurrogateDataset(
        pt_path=None if pt_path is None else resolve_project_path(pt_path),
        pt_dir=None if pt_dir is None else resolve_project_path(pt_dir),
        pdb_path=pdb_path,
        vina_mode=vina_mode,
        max_affinity=None if max_affinity is None else float(max_affinity),
        results_key=None if results_key is None else str(results_key),
    )

    training_cfg = cfg.get("training") or {}
    if not isinstance(training_cfg, dict):
        raise ValueError("training config must be a mapping")
    batch_size = int(training_cfg.get("batch_size", 16))
    epochs = int(training_cfg.get("epochs", 30))
    lr = float(training_cfg.get("lr", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    log_every = int(training_cfg.get("log_every", 1))
    eval_repeats = int(training_cfg.get("eval_repeats", 4))
    checkpoint_name = str(training_cfg.get("checkpoint_name", "surrogate_best.pt"))
    final_checkpoint_name = str(training_cfg.get("final_checkpoint_name", "surrogate_final.pt"))
    experiment_name = str(cfg.get("experiment_name", "surrogate-guidance"))
    model_root = paths.ensure_dir(resolve_project_path(training_cfg.get("save_dir", "models/surrogate_guidance")))
    processed_root = paths.ensure_dir(
        resolve_project_path(training_cfg.get("processed_dir", "data/surrogate_guidance"))
    )

    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping")
    pooling = str(model_cfg.get("pooling", "mean"))
    timestep_conditioning = bool(model_cfg.get("timestep_conditioning", True))

    noise_cfg = cfg.get("noise") or {}
    if not isinstance(noise_cfg, dict):
        raise ValueError("noise config must be a mapping")
    noisy_training = bool(noise_cfg.get("enabled", True))
    noise_schedule = SurrogateNoiseSchedule(
        enabled=noisy_training,
        n_steps=int(noise_cfg.get("n_steps", 1000)),
        beta_start=float(noise_cfg.get("beta_start", 1e-4)),
        beta_end=float(noise_cfg.get("beta_end", 2e-2)),
    )

    source_path = dataset.source_path
    run_paths = create_run_paths(
        experiment_name=experiment_name,
        model_type=f"surrogate-{slugify(pooling)}-{'t' if timestep_conditioning else 'no-t'}",
        data_path=source_path if source_path is not None else pdb_path,
        model_root=model_root,
        processed_root=processed_root,
    )

    resolved_config_path = run_paths.model_dir / "config.resolved.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(cfg, f, sort_keys=False)
    shutil.copyfile(config_path, run_paths.model_dir / "config.source.yaml")

    train_dataset, val_dataset, test_dataset = _split_dataset(
        dataset,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    collate_fn = lambda graphs: collate_surrogate_graphs(graphs, r_ligand=r_ligand, r_cross=r_cross)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = device_from_config(cfg)
    model = LigandContextSurrogateModel(
        max_atomic_number=int(model_cfg.get("max_atomic_number", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_rbf=int(model_cfg.get("num_rbf", 32)),
        cutoff=float(model_cfg.get("cutoff", max(r_ligand, r_cross))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=pooling,
        timestep_conditioning=timestep_conditioning,
        time_emb_dim=int(model_cfg.get("time_emb_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion: nn.Module = nn.SmoothL1Loss(beta=float(training_cfg.get("huber_beta", 1.0)))

    print(
        f"[surrogate-train] run={run_paths.run_name} data={source_path} "
        f"pooling={pooling} timestep_conditioning={timestep_conditioning} noisy_training={noisy_training}"
    )
    print(
        f"[surrogate-train] splits train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"r_ligand={r_ligand} r_cross={r_cross}"
    )

    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_val_mae = float("inf")
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_training_epoch(
            model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
            r_ligand=r_ligand,
            r_cross=r_cross,
        )
        val_metrics = _evaluate_model(
            model,
            loader=val_loader,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
            r_ligand=r_ligand,
            r_cross=r_cross,
            repeats=eval_repeats,
            seed=seed + (1000 * epoch),
        )
        history_row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(history_row)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            _save_checkpoint(
                run_paths.model_dir / checkpoint_name,
                model=model,
                cfg=cfg,
                config_path=config_path,
                run_name=run_paths.run_name,
                source_path=source_path,
                best_epoch=best_epoch,
                best_val_mae=best_val_mae,
                metrics=val_metrics,
            )

        if epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == epochs:
            elapsed = time.perf_counter() - start
            print(
                f"[surrogate-train] epoch={epoch}/{epochs} "
                f"train_loss={train_metrics['train_loss']:.5f} "
                f"train_mae={train_metrics['train_mae']:.5f} "
                f"val_mae={val_metrics['mae']:.5f} "
                f"val_rmse={val_metrics['rmse']:.5f} "
                f"val_pearson={val_metrics['pearson']:.5f} "
                f"elapsed_sec={elapsed:.1f}"
            )

    final_checkpoint_path = run_paths.model_dir / final_checkpoint_name
    _save_checkpoint(
        final_checkpoint_path,
        model=model,
        cfg=cfg,
        config_path=config_path,
        run_name=run_paths.run_name,
        source_path=source_path,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        metrics=history[-1] if history else {},
    )

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_metrics = _evaluate_model(
        model,
        loader=test_loader,
        device=device,
        noise_schedule=noise_schedule,
        timestep_conditioning=timestep_conditioning,
        r_ligand=r_ligand,
        r_cross=r_cross,
        repeats=max(1, eval_repeats),
        seed=seed + 99999,
    )

    history_path = run_paths.processed_dir / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)
        f.write("\n")

    metrics_payload = {
        "run_name": run_paths.run_name,
        "source_path": str(source_path),
        "pdb_path": str(pdb_path),
        "pooling": pooling,
        "timestep_conditioning": timestep_conditioning,
        "noisy_training": noisy_training,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_metrics": test_metrics,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    metrics_path = run_paths.processed_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return SurrogateTrainResult(
        checkpoint_path=run_paths.model_dir / checkpoint_name,
        final_checkpoint_path=final_checkpoint_path,
        history_path=history_path,
        metrics_path=metrics_path,
        config_path=config_path,
        run_name=run_paths.run_name,
        model_dir=run_paths.model_dir,
        data_source_path=source_path,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        test_mae=float(test_metrics["mae"]),
        test_rmse=float(test_metrics["rmse"]),
        test_pearson=float(test_metrics["pearson"]),
        pooling=pooling,
        timestep_conditioning=timestep_conditioning,
        noisy_training=noisy_training,
    )


def train_surrogate_model_from_config(config_path: Path) -> SurrogateTrainResult:
    cfg = load_yaml_config(config_path)
    return train_surrogate_model_from_loaded_config(cfg, config_path=config_path)


def _split_dataset(
    dataset: LigandContextSurrogateDataset,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[Subset[SurrogateGraph], Subset[SurrogateGraph], Subset[SurrogateGraph]]:
    num_items = len(dataset)
    indices = np.arange(num_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    num_test = int(round(num_items * test_fraction))
    num_val = int(round(num_items * val_fraction))
    num_test = min(num_test, max(num_items - 2, 0))
    num_val = min(num_val, max(num_items - num_test - 1, 0))
    num_train = num_items - num_val - num_test
    if num_train < 1:
        raise ValueError("Dataset split leaves no training examples")

    train_idx = indices[:num_train].tolist()
    val_idx = indices[num_train : num_train + num_val].tolist()
    test_idx = indices[num_train + num_val :].tolist()
    if not val_idx:
        val_idx = train_idx[-1:]
    if not test_idx:
        test_idx = val_idx[-1:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def _prepare_model_batch(
    batch: SurrogateBatch,
    *,
    device: torch.device,
    noise_schedule: SurrogateNoiseSchedule,
    timestep_conditioning: bool,
    r_ligand: float,
    r_cross: float,
    generator: torch.Generator | None = None,
) -> SurrogateBatch:
    return noise_schedule.sample_batch(
        batch.to(device),
        include_timestep=timestep_conditioning,
        r_ligand=r_ligand,
        r_cross=r_cross,
        generator=generator,
    )


def _run_training_epoch(
    model: LigandContextSurrogateModel,
    *,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    noise_schedule: SurrogateNoiseSchedule,
    timestep_conditioning: bool,
    r_ligand: float,
    r_cross: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    total_count = 0
    for batch in loader:
        prepared = _prepare_model_batch(
            batch,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
            r_ligand=r_ligand,
            r_cross=r_cross,
        )
        pred = model(prepared)
        target = prepared.y.float()
        loss = criterion(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item()) * int(target.shape[0])
        total_abs_error += float((pred.detach() - target).abs().sum().item())
        total_count += int(target.shape[0])
    return {
        "train_loss": total_loss / max(total_count, 1),
        "train_mae": total_abs_error / max(total_count, 1),
    }


@torch.inference_mode()
def _evaluate_model(
    model: LigandContextSurrogateModel,
    *,
    loader: DataLoader,
    device: torch.device,
    noise_schedule: SurrogateNoiseSchedule,
    timestep_conditioning: bool,
    r_ligand: float,
    r_cross: float,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    pred_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    for repeat in range(max(1, repeats)):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + repeat)
        for batch in loader:
            prepared = _prepare_model_batch(
                batch,
                device=device,
                noise_schedule=noise_schedule,
                timestep_conditioning=timestep_conditioning,
                r_ligand=r_ligand,
                r_cross=r_cross,
                generator=generator,
            )
            pred = model(prepared)
            pred_values.append(pred.detach().cpu().numpy())
            target_values.append(prepared.y.detach().cpu().numpy())

    if not pred_values:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    pred_arr = np.concatenate(pred_values, axis=0).astype(np.float64, copy=False)
    target_arr = np.concatenate(target_values, axis=0).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(pred_arr - target_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))
    if pred_arr.size < 2 or np.std(pred_arr) <= 1e-12 or np.std(target_arr) <= 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(pred_arr, target_arr)[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson": pearson}


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: LigandContextSurrogateModel,
    cfg: dict[str, Any],
    config_path: Path,
    run_name: str,
    source_path: Path,
    best_epoch: int,
    best_val_mae: float,
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "model_cfg": dict(cfg.get("model") or {}),
        "noise_cfg": dict(cfg.get("noise") or {}),
        "data_cfg": dict(cfg.get("data") or {}),
        "config_path": str(config_path),
        "run_name": run_name,
        "data_source_path": str(source_path),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "train_surrogate_guidance_model.yaml"),
        help="Path to surrogate-model training YAML config",
    )
    args = parser.parse_args()
    train_surrogate_model_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
