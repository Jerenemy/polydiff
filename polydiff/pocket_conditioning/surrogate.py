"""Pocket-context surrogate model and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import paths
from ..models.diffusion import SinusoidalPosEmb
from ..runs import create_run_paths, slugify
from ..utils.runtime import device_from_config, resolve_project_path, set_seed
from .data import PocketFitPairDataset, load_pocket_fit_pair_arrays


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, values.shape[-1]))
    out.index_add_(0, index, values)
    return out


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    sums = _scatter_sum(values, index, dim_size=dim_size)
    counts = values.new_zeros((dim_size,))
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    counts = counts.clamp(min=1.0)
    return sums / counts.unsqueeze(-1)


@dataclass(slots=True)
class PocketConditionBatch:
    ligand_coords: torch.Tensor
    pocket_coords: torch.Tensor
    fit_score: torch.Tensor
    inside_fraction: torch.Tensor
    outside_penalty: torch.Tensor
    area_ratio: torch.Tensor
    clearance_mean: torch.Tensor
    t: torch.Tensor | None = None
    clean_ligand_coords: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "PocketConditionBatch":
        self.ligand_coords = self.ligand_coords.to(device)
        self.pocket_coords = self.pocket_coords.to(device)
        self.fit_score = self.fit_score.to(device)
        self.inside_fraction = self.inside_fraction.to(device)
        self.outside_penalty = self.outside_penalty.to(device)
        self.area_ratio = self.area_ratio.to(device)
        self.clearance_mean = self.clearance_mean.to(device)
        if self.t is not None:
            self.t = self.t.to(device)
        if self.clean_ligand_coords is not None:
            self.clean_ligand_coords = self.clean_ligand_coords.to(device)
        return self

    def clone(self) -> "PocketConditionBatch":
        return PocketConditionBatch(
            ligand_coords=self.ligand_coords.clone(),
            pocket_coords=self.pocket_coords.clone(),
            fit_score=self.fit_score.clone(),
            inside_fraction=self.inside_fraction.clone(),
            outside_penalty=self.outside_penalty.clone(),
            area_ratio=self.area_ratio.clone(),
            clearance_mean=self.clearance_mean.clone(),
            t=None if self.t is None else self.t.clone(),
            clean_ligand_coords=None if self.clean_ligand_coords is None else self.clean_ligand_coords.clone(),
        )


@dataclass(frozen=True, slots=True)
class PocketSurrogateTrainResult:
    checkpoint_path: Path
    final_checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    run_name: str
    model_dir: Path
    timestep_conditioning: bool
    best_epoch: int
    best_val_mae: float
    test_mae: float
    test_pearson: float


class PocketSurrogateNoiseSchedule:
    def __init__(
        self,
        *,
        enabled: bool,
        n_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.n_steps = int(n_steps)
        betas = torch.linspace(float(beta_start), float(beta_end), self.n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def sample_batch(
        self,
        batch: PocketConditionBatch,
        *,
        include_timestep: bool,
        generator: torch.Generator | None = None,
    ) -> PocketConditionBatch:
        prepared = batch.clone()
        prepared.clean_ligand_coords = batch.ligand_coords.clone()
        if not self.enabled:
            prepared.t = (
                torch.zeros((prepared.ligand_coords.shape[0],), dtype=torch.long, device=prepared.ligand_coords.device)
                if include_timestep
                else None
            )
            return prepared

        timestep = torch.randint(
            0,
            self.n_steps,
            (prepared.ligand_coords.shape[0],),
            device=prepared.ligand_coords.device,
            generator=generator,
            dtype=torch.long,
        )
        alpha_bar = self.alpha_bars.to(prepared.ligand_coords.device).index_select(0, timestep).view(-1, 1, 1)
        noise = torch.randn(
            prepared.ligand_coords.shape,
            device=prepared.ligand_coords.device,
            dtype=prepared.ligand_coords.dtype,
            generator=generator,
        )
        prepared.ligand_coords = alpha_bar.sqrt() * prepared.ligand_coords + (1.0 - alpha_bar).sqrt() * noise
        prepared.t = timestep if include_timestep else None
        return prepared


class GaussianRBF(nn.Module):
    def __init__(self, *, num_rbf: int = 16, cutoff: float = 4.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff), int(num_rbf), dtype=torch.float32)
        self.register_buffer("centers", centers)
        step = 1.0 if centers.numel() == 1 else float((centers[1] - centers[0]).item())
        self.gamma = 1.0 / max(step * step, 1e-6)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers.view(1, -1)
        return torch.exp(-self.gamma * diff.square())


class PocketContextMessageBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, num_edge_types: int, num_rbf: int, cutoff: float, dropout: float) -> None:
        super().__init__()
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=cutoff)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear((3 * hidden_dim) + num_rbf + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        rel = coords.index_select(0, src) - coords.index_select(0, dst)
        dist = rel.norm(dim=-1)
        radial = self.rbf(dist)
        edge_features = self.edge_type_emb(edge_type)
        messages = self.msg_mlp(
            torch.cat(
                [
                    h.index_select(0, src),
                    h.index_select(0, dst),
                    edge_features,
                    radial,
                    dist.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        aggregated = _scatter_mean(messages, dst, dim_size=h.shape[0])
        return self.norm(h + self.update_mlp(torch.cat([h, aggregated], dim=-1)))


class PocketContextSurrogateModel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 16,
        cutoff: float = 4.0,
        dropout: float = 0.1,
        timestep_conditioning: bool = True,
        time_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        self.timestep_conditioning = bool(timestep_conditioning)
        self.input_proj = nn.Linear(4, hidden_dim)
        self.node_type_emb = nn.Embedding(2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        if self.timestep_conditioning:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(time_emb_dim),
                nn.Linear(time_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.time_mlp = None
        self.layers = nn.ModuleList(
            [
                PocketContextMessageBlock(
                    hidden_dim=hidden_dim,
                    num_edge_types=4,
                    num_rbf=num_rbf,
                    cutoff=cutoff,
                    dropout=dropout,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ligand_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ligand_coords.ndim != 3 or ligand_coords.shape[-1] != 2:
            raise ValueError(f"ligand_coords must have shape (batch, n_ligand, 2), got {tuple(ligand_coords.shape)}")
        if pocket_coords.ndim != 3 or pocket_coords.shape[-1] != 2:
            raise ValueError(f"pocket_coords must have shape (batch, n_pocket, 2), got {tuple(pocket_coords.shape)}")
        if pocket_coords.shape[0] != ligand_coords.shape[0]:
            raise ValueError("ligand_coords and pocket_coords must have matching batch size")

        batch_size, n_ligand, _ = ligand_coords.shape
        n_pocket = int(pocket_coords.shape[1])
        coords = torch.cat([ligand_coords, pocket_coords], dim=1)
        cycle_features = torch.cat(
            [
                _cycle_positional_encoding(batch_size, n_ligand, coords.device, coords.dtype),
                _cycle_positional_encoding(batch_size, n_pocket, coords.device, coords.dtype),
            ],
            dim=1,
        )
        node_features = torch.cat([coords, cycle_features], dim=-1)
        node_features = node_features.reshape(batch_size * (n_ligand + n_pocket), 4)
        coords_flat = coords.reshape(batch_size * (n_ligand + n_pocket), 2)
        node_type = torch.cat(
            [
                torch.ones((batch_size, n_ligand), dtype=torch.long, device=coords.device),
                torch.zeros((batch_size, n_pocket), dtype=torch.long, device=coords.device),
            ],
            dim=1,
        ).reshape(-1)
        batch_index = torch.arange(batch_size, device=coords.device, dtype=torch.long).repeat_interleave(n_ligand + n_pocket)
        edge_index, edge_type = _batched_context_edges(
            batch_size=batch_size,
            n_ligand=n_ligand,
            n_pocket=n_pocket,
            device=coords.device,
        )

        h = self.input_proj(node_features) + self.node_type_emb(node_type)
        if self.time_mlp is not None:
            if t is None:
                raise ValueError("timestep-conditioned surrogate requires `t`")
            time_features = self.time_mlp(t.float()).index_select(0, batch_index)
            h = h + time_features
        h = self.input_norm(h)

        for layer in self.layers:
            h = layer(h, coords_flat, edge_index, edge_type)

        ligand_mask = node_type == 1
        pooled = _scatter_mean(h[ligand_mask], batch_index[ligand_mask], dim_size=batch_size)
        return self.head(pooled).squeeze(-1)


def collate_pocket_condition_batch(items: Sequence[dict[str, torch.Tensor]]) -> PocketConditionBatch:
    return PocketConditionBatch(
        ligand_coords=torch.stack([item["ligand_coords"] for item in items], dim=0).to(torch.float32),
        pocket_coords=torch.stack([item["pocket_coords"] for item in items], dim=0).to(torch.float32),
        fit_score=torch.stack([item["fit_score"] for item in items], dim=0).to(torch.float32),
        inside_fraction=torch.stack([item["inside_fraction"] for item in items], dim=0).to(torch.float32),
        outside_penalty=torch.stack([item["outside_penalty"] for item in items], dim=0).to(torch.float32),
        area_ratio=torch.stack([item["area_ratio"] for item in items], dim=0).to(torch.float32),
        clearance_mean=torch.stack([item["clearance_mean"] for item in items], dim=0).to(torch.float32),
    )


def train_pocket_surrogate(
    *,
    train_pairs_path: Path,
    val_pairs_path: Path,
    test_pairs_path: Path,
    config: dict[str, Any],
    config_path: Path,
) -> PocketSurrogateTrainResult:
    seed = int(config.get("seed", 0))
    set_seed(seed)

    study_name = str(config.get("study_name", "pocket-fit-conditioning"))
    timestep_conditioning = bool(config.get("timestep_conditioning", True))
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    noise_cfg = dict(config.get("noise", {}))

    train_dataset = PocketFitPairDataset(load_pocket_fit_pair_arrays(train_pairs_path))
    val_dataset = PocketFitPairDataset(load_pocket_fit_pair_arrays(val_pairs_path))
    test_dataset = PocketFitPairDataset(load_pocket_fit_pair_arrays(test_pairs_path))

    batch_size = int(training_cfg.get("batch_size", 64))
    epochs = int(training_cfg.get("epochs", 20))
    lr = float(training_cfg.get("lr", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    log_every = int(training_cfg.get("log_every", 1))
    eval_repeats = int(training_cfg.get("eval_repeats", 4))
    checkpoint_name = str(training_cfg.get("checkpoint_name", "pocket_surrogate_best.pt"))
    final_checkpoint_name = str(training_cfg.get("final_checkpoint_name", "pocket_surrogate_final.pt"))

    save_dir = paths.ensure_dir(resolve_project_path(training_cfg.get("save_dir", "models/pocket_conditioning")))
    processed_dir = paths.ensure_dir(
        resolve_project_path(training_cfg.get("processed_dir", "data/pocket_conditioning"))
    )
    run_paths = create_run_paths(
        experiment_name=f"{study_name}-{'with-t' if timestep_conditioning else 'no-t'}",
        model_type=f"pocket-surrogate-{'with-t' if timestep_conditioning else 'no-t'}",
        data_path=train_pairs_path,
        model_root=save_dir,
        processed_root=processed_dir,
    )

    config_suffix = config_path.suffix or ".json"
    shutil.copyfile(config_path, run_paths.model_dir / f"config.source{config_suffix}")
    (run_paths.model_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_pocket_condition_batch,
        "num_workers": 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = device_from_config(config)
    model = PocketContextSurrogateModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_rbf=int(model_cfg.get("num_rbf", 16)),
        cutoff=float(model_cfg.get("cutoff", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        timestep_conditioning=timestep_conditioning,
        time_emb_dim=int(model_cfg.get("time_emb_dim", 64)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion: nn.Module = nn.SmoothL1Loss(beta=float(training_cfg.get("huber_beta", 1.0)))
    noise_schedule = PocketSurrogateNoiseSchedule(
        enabled=bool(noise_cfg.get("enabled", True)),
        n_steps=int(noise_cfg.get("n_steps", 1000)),
        beta_start=float(noise_cfg.get("beta_start", 1e-4)),
        beta_end=float(noise_cfg.get("beta_end", 2e-2)),
    )

    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
        )
        val_metrics = _evaluate_surrogate(
            model,
            loader=val_loader,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
            repeats=eval_repeats,
            seed=seed + (1000 * epoch),
        )
        timestep_metrics = _evaluate_by_timestep_bin(
            model,
            loader=val_loader,
            device=device,
            noise_schedule=noise_schedule,
            timestep_conditioning=timestep_conditioning,
            num_bins=5,
            seed=seed + (2000 * epoch),
        )
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_timestep_mae": timestep_metrics,
        }
        history.append(row)
        if float(val_metrics["mae"]) < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_state = {"model_state": model.state_dict()}
            _save_checkpoint(
                run_paths.model_dir / checkpoint_name,
                model=model,
                config=config,
                run_name=run_paths.run_name,
                best_epoch=best_epoch,
                best_val_mae=best_val_mae,
            )
        if epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == epochs:
            elapsed = time.perf_counter() - start
            print(
                f"[pocket-surrogate] run={run_paths.run_name} epoch={epoch}/{epochs} "
                f"train_loss={train_metrics['train_loss']:.5f} train_mae={train_metrics['train_mae']:.5f} "
                f"val_mae={val_metrics['mae']:.5f} val_pearson={val_metrics['pearson']:.5f} "
                f"elapsed_sec={elapsed:.1f}"
            )

    _save_checkpoint(
        run_paths.model_dir / final_checkpoint_name,
        model=model,
        config=config,
        run_name=run_paths.run_name,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
    )
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_metrics = _evaluate_surrogate(
        model,
        loader=test_loader,
        device=device,
        noise_schedule=noise_schedule,
        timestep_conditioning=timestep_conditioning,
        repeats=max(1, eval_repeats),
        seed=seed + 123456,
    )
    test_timestep_metrics = _evaluate_by_timestep_bin(
        model,
        loader=test_loader,
        device=device,
        noise_schedule=noise_schedule,
        timestep_conditioning=timestep_conditioning,
        num_bins=5,
        seed=seed + 234567,
    )

    history_path = run_paths.processed_dir / "surrogate_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_payload = {
        "run_name": run_paths.run_name,
        "timestep_conditioning": timestep_conditioning,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_metrics": test_metrics,
        "test_timestep_mae": test_timestep_metrics,
        "train_pairs_path": str(train_pairs_path),
        "val_pairs_path": str(val_pairs_path),
        "test_pairs_path": str(test_pairs_path),
    }
    metrics_path = run_paths.processed_dir / "surrogate_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return PocketSurrogateTrainResult(
        checkpoint_path=run_paths.model_dir / checkpoint_name,
        final_checkpoint_path=run_paths.model_dir / final_checkpoint_name,
        history_path=history_path,
        metrics_path=metrics_path,
        run_name=run_paths.run_name,
        model_dir=run_paths.model_dir,
        timestep_conditioning=timestep_conditioning,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        test_mae=float(test_metrics["mae"]),
        test_pearson=float(test_metrics["pearson"]),
    )


def load_pocket_surrogate_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], PocketContextSurrogateModel]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = dict(checkpoint.get("config", {}).get("model", {}))
    timestep_conditioning = bool(checkpoint.get("config", {}).get("timestep_conditioning", True))
    model = PocketContextSurrogateModel(
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_rbf=int(model_cfg.get("num_rbf", 16)),
        cutoff=float(model_cfg.get("cutoff", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        timestep_conditioning=timestep_conditioning,
        time_emb_dim=int(model_cfg.get("time_emb_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


def _run_epoch(
    model: PocketContextSurrogateModel,
    *,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    noise_schedule: PocketSurrogateNoiseSchedule,
    timestep_conditioning: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    for batch in loader:
        prepared = noise_schedule.sample_batch(batch.to(device), include_timestep=timestep_conditioning)
        pred = model(prepared.ligand_coords, prepared.pocket_coords, t=prepared.t)
        target = prepared.fit_score
        loss = criterion(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_mae += float((pred.detach() - target).abs().sum().item())
        total_count += batch_size
    return {
        "train_loss": total_loss / max(total_count, 1),
        "train_mae": total_mae / max(total_count, 1),
    }


@torch.inference_mode()
def _evaluate_surrogate(
    model: PocketContextSurrogateModel,
    *,
    loader: DataLoader,
    device: torch.device,
    noise_schedule: PocketSurrogateNoiseSchedule,
    timestep_conditioning: bool,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for repeat in range(max(1, repeats)):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + repeat)
        for batch in loader:
            prepared = noise_schedule.sample_batch(
                batch.to(device),
                include_timestep=timestep_conditioning,
                generator=generator,
            )
            pred = model(prepared.ligand_coords, prepared.pocket_coords, t=prepared.t)
            preds.append(pred.detach().cpu().numpy())
            targets.append(prepared.fit_score.detach().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0).astype(np.float64, copy=False)
    target_arr = np.concatenate(targets, axis=0).astype(np.float64, copy=False)
    mae = float(np.mean(np.abs(pred_arr - target_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))
    if pred_arr.size < 2 or np.std(pred_arr) <= 1e-12 or np.std(target_arr) <= 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(pred_arr, target_arr)[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson": pearson}


@torch.inference_mode()
def _evaluate_by_timestep_bin(
    model: PocketContextSurrogateModel,
    *,
    loader: DataLoader,
    device: torch.device,
    noise_schedule: PocketSurrogateNoiseSchedule,
    timestep_conditioning: bool,
    num_bins: int,
    seed: int,
) -> dict[str, float]:
    if noise_schedule.n_steps < 2:
        return {"bin_0": float("nan")}
    model.eval()
    bin_edges = np.linspace(0, noise_schedule.n_steps, int(num_bins) + 1, dtype=np.int64)
    metrics: dict[str, float] = {}
    for bin_index in range(len(bin_edges) - 1):
        low = int(bin_edges[bin_index])
        high = max(int(bin_edges[bin_index + 1]), low + 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + bin_index)
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for batch in loader:
            prepared = batch.to(device).clone()
            prepared.clean_ligand_coords = prepared.ligand_coords.clone()
            t = torch.randint(
                low,
                high,
                (prepared.ligand_coords.shape[0],),
                generator=generator,
                device=device,
                dtype=torch.long,
            )
            alpha_bar = noise_schedule.alpha_bars.to(device).index_select(0, t).view(-1, 1, 1)
            noise = torch.randn(
                prepared.ligand_coords.shape,
                generator=generator,
                device=device,
                dtype=prepared.ligand_coords.dtype,
            )
            prepared.ligand_coords = alpha_bar.sqrt() * prepared.ligand_coords + (1.0 - alpha_bar).sqrt() * noise
            prepared.t = t if timestep_conditioning else None
            pred = model(prepared.ligand_coords, prepared.pocket_coords, t=prepared.t)
            preds.append(pred.detach().cpu().numpy())
            targets.append(prepared.fit_score.detach().cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0).astype(np.float64, copy=False)
        target_arr = np.concatenate(targets, axis=0).astype(np.float64, copy=False)
        metrics[f"bin_{bin_index}"] = float(np.mean(np.abs(pred_arr - target_arr)))
    return metrics


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: PocketContextSurrogateModel,
    config: dict[str, Any],
    run_name: str,
    best_epoch: int,
    best_val_mae: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "run_name": run_name,
            "best_epoch": int(best_epoch),
            "best_val_mae": float(best_val_mae),
        },
        checkpoint_path,
    )


def _cycle_positional_encoding(
    batch_size: int,
    num_vertices: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(num_vertices, device=device, dtype=dtype)
    angles = 2.0 * torch.pi * positions / max(float(num_vertices), 1.0)
    encoding = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
    return encoding.unsqueeze(0).repeat(batch_size, 1, 1)


def _batched_context_edges(
    *,
    batch_size: int,
    n_ligand: int,
    n_pocket: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges_src: list[torch.Tensor] = []
    edges_dst: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    nodes_per_graph = n_ligand + n_pocket
    for batch_index in range(batch_size):
        base = batch_index * nodes_per_graph
        ligand_nodes = torch.arange(base, base + n_ligand, device=device, dtype=torch.long)
        pocket_nodes = torch.arange(base + n_ligand, base + nodes_per_graph, device=device, dtype=torch.long)

        ligand_next = torch.roll(ligand_nodes, shifts=-1)
        pocket_next = torch.roll(pocket_nodes, shifts=-1)

        edges_src.extend([ligand_nodes, ligand_next, pocket_nodes, pocket_next])
        edges_dst.extend([ligand_next, ligand_nodes, pocket_next, pocket_nodes])
        edge_types.extend(
            [
                torch.zeros((n_ligand,), dtype=torch.long, device=device),
                torch.zeros((n_ligand,), dtype=torch.long, device=device),
                torch.ones((n_pocket,), dtype=torch.long, device=device),
                torch.ones((n_pocket,), dtype=torch.long, device=device),
            ]
        )

        pocket_repeat = pocket_nodes.repeat_interleave(n_ligand)
        ligand_repeat = ligand_nodes.repeat(n_pocket)
        edges_src.extend([pocket_repeat, ligand_repeat])
        edges_dst.extend([ligand_repeat, pocket_repeat])
        edge_types.extend(
            [
                torch.full((n_ligand * n_pocket,), 2, dtype=torch.long, device=device),
                torch.full((n_ligand * n_pocket,), 3, dtype=torch.long, device=device),
            ]
        )

    edge_index = torch.stack([torch.cat(edges_src), torch.cat(edges_dst)], dim=0)
    edge_type = torch.cat(edge_types, dim=0)
    return edge_index, edge_type
