"""Run the pocket-fit conditioning study for polygon guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..data.diagnostics import compare_polygon_metric_tables, polygon_metric_table, summarize_polygon_dataset
from ..data.plot_polygons import select_animation_frames
from ..data.polygon_dataset import load_polygon_dataset
from ..runs import slugify
from ..sampling.guidance import GuidanceSchedule, ScheduledGuidance
from ..sampling.runtime import load_diffusion_from_checkpoint
from ..studies.runtime import apply_dotted_overrides, create_study_paths, write_yaml_config
from ..training.train import train_from_loaded_config
from ..utils.runtime import load_yaml_config, resolve_project_path
from .data import default_generated_data_dir, generate_pocket_fit_pair_splits, load_eval_pockets
from .geometry import PocketFitConfig, pocket_fit_state_numpy, pocket_fit_state_torch
from .surrogate import (
    PocketContextSurrogateModel,
    load_pocket_surrogate_checkpoint,
    train_pocket_surrogate,
)


class AnalyticPocketFitGuidance:
    def __init__(
        self,
        *,
        pocket_coords: torch.Tensor,
        n_vertices: int,
        reward: PocketFitConfig,
        scale: float,
    ) -> None:
        self.pocket_coords = pocket_coords
        self.n_vertices = int(n_vertices)
        self.reward = reward
        self.scale = float(scale)

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor, *, graph_batch=None) -> torch.Tensor:
        del t, graph_batch
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
            state = pocket_fit_state_torch(coords, self.pocket_coords, self.reward)
            objective = state.fit_score.sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


class SurrogatePocketFitGuidance:
    def __init__(
        self,
        *,
        model: PocketContextSurrogateModel,
        pocket_coords: torch.Tensor,
        n_vertices: int,
        scale: float,
        timestep_conditioning: bool,
    ) -> None:
        self.model = model
        self.pocket_coords = pocket_coords
        self.n_vertices = int(n_vertices)
        self.scale = float(scale)
        self.timestep_conditioning = bool(timestep_conditioning)

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor, *, graph_batch=None) -> torch.Tensor:
        del graph_batch
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            coords = x_in.reshape(x_in.shape[0], self.n_vertices, 2)
            pocket = self.pocket_coords.unsqueeze(0).expand(coords.shape[0], -1, -1)
            timestep = t if self.timestep_conditioning else None
            pred = self.model(coords, pocket, t=timestep)
            objective = pred.sum()
            grad = torch.autograd.grad(objective, x_in)[0]
        return self.scale * grad.detach()


def run_pocket_fit_study_from_config(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml_config(config_path)
    study_cfg = cfg.get("study") or {}
    if not isinstance(study_cfg, dict):
        raise ValueError("study config must be a mapping")
    study_name = str(study_cfg.get("name", "pocket-fit-conditioning"))
    study_root = resolve_project_path(study_cfg.get("root_dir", "data/studies"))
    paths_obj = create_study_paths(name=study_name, root_dir=study_root)
    write_yaml_config(paths_obj.configs_dir / "00__study_manifest.yaml", cfg)

    reward = _load_reward_config(cfg.get("reward") or {})
    data_paths = _prepare_pair_data(cfg, study_name=study_name)
    diffusion_result = _prepare_diffusion_prior(cfg, paths_obj=paths_obj)
    surrogate_results = _train_surrogates(
        cfg,
        data_paths=data_paths,
        paths_obj=paths_obj,
        study_name=study_name,
    )

    eval_pockets = load_eval_pockets(data_paths["eval_pockets"])
    ligand_reference_path = resolve_project_path((cfg.get("data") or {}).get("ligand_data_path", "data/raw/hexagons.npz"))
    reference_table = polygon_metric_table(load_polygon_dataset(ligand_reference_path))
    checkpoint, diffusion, _, max_vertices = load_diffusion_from_checkpoint(
        resolve_project_path(diffusion_result["checkpoint_path"]),
        device=torch.device(str(cfg.get("device", "cpu")) if cfg.get("device") not in {None, "auto"} else ("cuda" if torch.cuda.is_available() else "cpu")),
    )
    n_vertices = int(checkpoint.get("n_vertices", max_vertices))

    guidance_cfg = cfg.get("guidance") or {}
    if not isinstance(guidance_cfg, dict):
        raise ValueError("guidance config must be a mapping")
    num_samples_per_pocket = int(guidance_cfg.get("num_samples_per_pocket", 256))
    n_steps = int(guidance_cfg.get("n_steps", 250))
    schedule_name = str(guidance_cfg.get("schedule", "late"))
    common_scale = float(guidance_cfg.get("scale", 6.0))
    analytic_scale = float(guidance_cfg.get("analytic_scale", common_scale))
    surrogate_scale = float(guidance_cfg.get("surrogate_scale", common_scale))
    sampling_seed = int(guidance_cfg.get("seed", int(cfg.get("seed", 0))))

    with_t_checkpoint, with_t_model = load_pocket_surrogate_checkpoint(
        surrogate_results["with_t"]["checkpoint_path"],
        device=diffusion.device,
    )
    no_t_checkpoint, no_t_model = load_pocket_surrogate_checkpoint(
        surrogate_results["no_t"]["checkpoint_path"],
        device=diffusion.device,
    )
    del with_t_checkpoint, no_t_checkpoint

    case_specs = [
        {"case_name": "unguided", "guidance": None, "guidance_kind": "unguided"},
        {"case_name": "analytic-pocket-fit", "guidance_kind": "analytic"},
        {"case_name": "surrogate-no-t", "guidance_kind": "surrogate_no_t"},
        {"case_name": "surrogate-with-t", "guidance_kind": "surrogate_with_t"},
    ]

    case_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    case_payloads: dict[str, dict[str, Any]] = {}
    representative_samples: dict[str, dict[str, Any]] = {}
    guidance_builders_by_case: dict[str, Any] = {}
    for case_index, case_spec in enumerate(case_specs, start=1):
        guidance_kind = case_spec["guidance_kind"]
        guidance_builder = _build_case_guidance_builder(
            guidance_kind=guidance_kind,
            n_vertices=n_vertices,
            reward=reward,
            analytic_scale=analytic_scale,
            surrogate_scale=surrogate_scale,
            no_t_model=no_t_model,
            with_t_model=with_t_model,
            schedule_name=schedule_name,
            n_steps=n_steps,
        )
        guidance_builders_by_case[case_spec["case_name"]] = guidance_builder

        case_result = _run_guidance_case(
            diffusion=diffusion,
            case_name=case_spec["case_name"],
            guidance_kind=guidance_kind,
            guidance_builder=guidance_builder,
            eval_pockets=eval_pockets,
            reward=reward,
            reference_table=reference_table,
            num_samples_per_pocket=num_samples_per_pocket,
            n_vertices=n_vertices,
            n_steps=n_steps,
            seed=sampling_seed + (1000 * case_index),
            study_dir=paths_obj.study_dir,
            surrogate_eval_model=(
                None
                if guidance_kind == "unguided"
                else (with_t_model if guidance_kind == "surrogate_with_t" else no_t_model if guidance_kind == "surrogate_no_t" else None)
            ),
            surrogate_eval_uses_t=(guidance_kind == "surrogate_with_t"),
        )
        case_rows.append(case_result["summary_row"])
        sample_rows.extend(case_result["sample_rows"])
        case_payloads[case_spec["case_name"]] = case_result["case_payload"]
        representative_samples[case_spec["case_name"]] = case_result["representative"]

    reports_dir = paths_obj.study_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    case_results_path = reports_dir / "pocket_case_results.json"
    case_results_path.write_text(json.dumps(case_payloads, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_df = pd.DataFrame(case_rows)
    summary_csv_path = reports_dir / "pocket_case_summary.csv"
    summary_json_path = reports_dir / "pocket_case_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(case_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sample_df = pd.DataFrame(sample_rows)
    sample_csv_path = reports_dir / "pocket_sample_metrics.csv"
    sample_json_path = reports_dir / "pocket_sample_metrics.json"
    sample_df.to_csv(sample_csv_path, index=False)
    sample_json_path.write_text(json.dumps(sample_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figures_dir = paths_obj.study_dir / "figures"
    animation_paths, extra_figure_paths = _save_trajectory_visualizations(
        cfg=cfg,
        diffusion=diffusion,
        eval_pockets=eval_pockets,
        reward=reward,
        n_vertices=n_vertices,
        n_steps=n_steps,
        guidance_builders_by_case=guidance_builders_by_case,
        figures_dir=figures_dir,
        available_case_names=[str(spec["case_name"]) for spec in case_specs],
    )
    figure_paths = {
        "surrogate_timestep_mae": _stringify_path(
            _save_surrogate_timestep_figure(surrogate_results, figures_dir / "surrogate_timestep_mae.png")
        ),
        "guidance_metric_panel": _stringify_path(
            _save_guidance_metric_panel(summary_df, figures_dir / "pocket_guidance_metric_panel.png")
        ),
        "guidance_tradeoff": _stringify_path(
            _save_guidance_tradeoff(summary_df, figures_dir / "pocket_guidance_tradeoff.png")
        ),
        "best_sample_gallery": _stringify_path(
            _save_best_sample_gallery(representative_samples, figures_dir / "pocket_best_sample_gallery.png")
        ),
    }
    figure_paths.update(extra_figure_paths)
    figure_paths = {key: value for key, value in figure_paths.items() if value is not None}

    interpretation_path = _write_interpretation_guide(
        paths_obj.study_dir / "INTERPRET_RESULTS.md",
        reward=reward,
        summary_df=summary_df,
        figure_paths=figure_paths,
        animation_paths=animation_paths,
    )
    study_report = {
        "status": "completed",
        "study_name": paths_obj.study_name,
        "study_dir": str(paths_obj.study_dir),
        "config_path": str(resolve_project_path(config_path)),
        "data_paths": {key: str(value) for key, value in data_paths.items()},
        "diffusion_checkpoint_path": str(diffusion_result["checkpoint_path"]),
        "surrogate_results": surrogate_results,
        "case_results_path": str(case_results_path),
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "sample_metrics_csv_path": str(sample_csv_path),
        "sample_metrics_json_path": str(sample_json_path),
        "figure_paths": figure_paths,
        "animation_paths": animation_paths,
        "interpretation_guide_path": str(interpretation_path),
    }
    report_path = reports_dir / "study_report.json"
    report_path.write_text(json.dumps(study_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return study_report


def _prepare_pair_data(cfg: dict[str, Any], *, study_name: str) -> dict[str, Path]:
    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping")
    ligand_data_path = resolve_project_path(data_cfg.get("ligand_data_path", "data/raw/hexagons.npz"))
    generated_data_dir = resolve_project_path(
        data_cfg.get("generated_data_dir", default_generated_data_dir(study_name))
    )
    reward = _load_reward_config(cfg.get("reward") or {})
    required_paths = {
        "train": generated_data_dir / "pairs_train.npz",
        "val": generated_data_dir / "pairs_val.npz",
        "test": generated_data_dir / "pairs_test.npz",
        "eval_pockets": generated_data_dir / "eval_pockets.npz",
        "metadata": generated_data_dir / "metadata.json",
    }
    if not bool(data_cfg.get("force_regenerate", False)) and all(path.exists() for path in required_paths.values()):
        return required_paths
    return generate_pocket_fit_pair_splits(
        ligand_data_path=ligand_data_path,
        out_dir=generated_data_dir,
        reward=reward,
        seed=int(cfg.get("seed", 0)),
        ligand_size=int(data_cfg.get("ligand_size", 6)),
        pocket_size=int(data_cfg.get("pocket_size", 10)),
        num_train_pairs=int(data_cfg.get("num_train_pairs", 12000)),
        num_val_pairs=int(data_cfg.get("num_val_pairs", 2000)),
        num_test_pairs=int(data_cfg.get("num_test_pairs", 2000)),
        num_eval_pockets=int(data_cfg.get("num_eval_pockets", 6)),
        pocket_radial_sigma=float(data_cfg.get("pocket_radial_sigma", 0.08)),
        pocket_angle_sigma=float(data_cfg.get("pocket_angle_sigma", 0.03)),
        pocket_smooth_passes=int(data_cfg.get("pocket_smooth_passes", 6)),
        pocket_scale_min=float(data_cfg.get("pocket_scale_min", 1.9)),
        pocket_scale_max=float(data_cfg.get("pocket_scale_max", 2.4)),
        ligand_scale_min=float(data_cfg.get("ligand_scale_min", 0.65)),
        ligand_scale_max=float(data_cfg.get("ligand_scale_max", 1.45)),
    )


def _prepare_diffusion_prior(cfg: dict[str, Any], *, paths_obj) -> dict[str, Any]:
    diffusion_cfg = cfg.get("diffusion") or {}
    if not isinstance(diffusion_cfg, dict):
        raise ValueError("diffusion config must be a mapping")
    checkpoint_path = diffusion_cfg.get("checkpoint_path")
    if checkpoint_path is not None:
        resolved = resolve_project_path(checkpoint_path)
        if not resolved.exists():
            raise FileNotFoundError(f"diffusion.checkpoint_path does not exist: {resolved}")
        return {"checkpoint_path": str(resolved), "trained": False}

    train_config_path = resolve_project_path(diffusion_cfg.get("train_config", "configs/train_diffusion.yaml"))
    base_cfg = load_yaml_config(train_config_path)
    overrides = diffusion_cfg.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("diffusion.overrides must be a mapping")
    resolved_cfg = apply_dotted_overrides(base_cfg, overrides)
    resolved_cfg.setdefault("experiment_name", f"{paths_obj.study_name}-pocket-fit-prior")
    resolved_config_path = write_yaml_config(paths_obj.configs_dir / "00__pocket_fit_prior.yaml", resolved_cfg)
    result = train_from_loaded_config(resolved_cfg, config_path=resolved_config_path)
    return {
        "checkpoint_path": str(result.checkpoint_path),
        "run_name": result.run_name,
        "trained": True,
    }


def _train_surrogates(
    cfg: dict[str, Any],
    *,
    data_paths: dict[str, Path],
    paths_obj,
    study_name: str,
) -> dict[str, dict[str, Any]]:
    surrogate_cfg = cfg.get("surrogate") or {}
    if not isinstance(surrogate_cfg, dict):
        raise ValueError("surrogate config must be a mapping")
    shared_config = {
        "seed": int(cfg.get("seed", 0)),
        "device": cfg.get("device", "auto"),
        "study_name": study_name,
        "data": {},
        "model": dict(surrogate_cfg.get("model", {})),
        "noise": dict(surrogate_cfg.get("noise", {})),
        "training": dict(surrogate_cfg.get("training", {})),
    }

    results: dict[str, dict[str, Any]] = {}
    for label, with_t in (("with_t", True), ("no_t", False)):
        resolved_cfg = json.loads(json.dumps(shared_config))
        resolved_cfg["timestep_conditioning"] = with_t
        resolved_cfg["model"]["timestep_conditioning"] = with_t
        config_path = paths_obj.configs_dir / f"01__pocket_surrogate_{label}.json"
        config_path.write_text(json.dumps(resolved_cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = train_pocket_surrogate(
            train_pairs_path=data_paths["train"],
            val_pairs_path=data_paths["val"],
            test_pairs_path=data_paths["test"],
            config=resolved_cfg,
            config_path=config_path,
        )
        results[label] = {
            "checkpoint_path": str(result.checkpoint_path),
            "final_checkpoint_path": str(result.final_checkpoint_path),
            "history_path": str(result.history_path),
            "metrics_path": str(result.metrics_path),
            "run_name": result.run_name,
            "best_epoch": int(result.best_epoch),
            "best_val_mae": float(result.best_val_mae),
            "test_mae": float(result.test_mae),
            "test_pearson": float(result.test_pearson),
            "timestep_conditioning": bool(result.timestep_conditioning),
        }
    return results


def _build_case_guidance_builder(
    *,
    guidance_kind: str,
    n_vertices: int,
    reward: PocketFitConfig,
    analytic_scale: float,
    surrogate_scale: float,
    no_t_model: PocketContextSurrogateModel,
    with_t_model: PocketContextSurrogateModel,
    schedule_name: str,
    n_steps: int,
):
    if guidance_kind == "analytic":
        return lambda pocket: _wrap_guidance_schedule(
            AnalyticPocketFitGuidance(
                pocket_coords=pocket,
                n_vertices=n_vertices,
                reward=reward,
                scale=analytic_scale,
            ),
            schedule_name=schedule_name,
            num_steps=n_steps,
        )
    if guidance_kind == "surrogate_no_t":
        return lambda pocket: _wrap_guidance_schedule(
            SurrogatePocketFitGuidance(
                model=no_t_model,
                pocket_coords=pocket,
                n_vertices=n_vertices,
                scale=surrogate_scale,
                timestep_conditioning=False,
            ),
            schedule_name=schedule_name,
            num_steps=n_steps,
        )
    if guidance_kind == "surrogate_with_t":
        return lambda pocket: _wrap_guidance_schedule(
            SurrogatePocketFitGuidance(
                model=with_t_model,
                pocket_coords=pocket,
                n_vertices=n_vertices,
                scale=surrogate_scale,
                timestep_conditioning=True,
            ),
            schedule_name=schedule_name,
            num_steps=n_steps,
        )
    return lambda pocket: None


def _run_guidance_case(
    *,
    diffusion,
    case_name: str,
    guidance_kind: str,
    guidance_builder,
    eval_pockets: np.ndarray,
    reward: PocketFitConfig,
    reference_table: pd.DataFrame,
    num_samples_per_pocket: int,
    n_vertices: int,
    n_steps: int,
    seed: int,
    study_dir: Path,
    surrogate_eval_model: PocketContextSurrogateModel | None,
    surrogate_eval_uses_t: bool,
) -> dict[str, Any]:
    case_dir = study_dir / "cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    all_samples: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    representative_payload: dict[str, Any] | None = None

    for pocket_index, pocket_coords in enumerate(eval_pockets):
        pocket_tensor = torch.from_numpy(pocket_coords).to(diffusion.device, dtype=torch.float32)
        guidance = None if guidance_kind == "unguided" else guidance_builder(pocket_tensor)
        samples = _sample_with_seed(
            diffusion,
            shape=(num_samples_per_pocket, n_vertices * 2),
            n_steps=n_steps,
            guidance=guidance,
            seed=seed + pocket_index,
        ).reshape(num_samples_per_pocket, n_vertices, 2).astype(np.float32)
        fit_state = pocket_fit_state_numpy(samples, pocket_coords, reward)
        regularity_table = polygon_metric_table(samples)

        if surrogate_eval_model is not None:
            surrogate_pred = _predict_surrogate_scores(
                surrogate_eval_model,
                samples=samples,
                pocket_coords=pocket_coords,
                device=diffusion.device,
                uses_t=surrogate_eval_uses_t,
            )
        else:
            surrogate_pred = None

        for sample_index in range(num_samples_per_pocket):
            row = {
                "case_name": case_name,
                "guidance_kind": guidance_kind,
                "pocket_index": pocket_index,
                "sample_index": sample_index,
                "fit_score": float(fit_state.fit_score[sample_index]),
                "inside_fraction": float(fit_state.inside_fraction[sample_index]),
                "outside_penalty": float(fit_state.outside_penalty[sample_index]),
                "area_ratio": float(fit_state.area_ratio[sample_index]),
                "clearance_mean": float(fit_state.clearance_mean[sample_index]),
                "self_intersection": float(regularity_table.iloc[sample_index]["self_intersection"]),
                "regularity_score": float(regularity_table.iloc[sample_index]["score"]),
            }
            if surrogate_pred is not None:
                row["surrogate_pred"] = float(surrogate_pred[sample_index])
                row["surrogate_true_gap"] = float(surrogate_pred[sample_index] - fit_state.fit_score[sample_index])
            sample_rows.append(row)

        all_samples.append(samples)
        if pocket_index == 0:
            order = np.argsort(-np.asarray(fit_state.fit_score, dtype=np.float32))
            representative_payload = {
                "pocket_coords": pocket_coords.astype(np.float32),
                "samples": samples[order[: min(3, samples.shape[0])]].astype(np.float32),
                "fit_score": np.asarray(fit_state.fit_score, dtype=np.float32)[order[: min(3, samples.shape[0])]],
            }

    stacked_samples = np.concatenate(all_samples, axis=0)
    sample_archive_path = case_dir / "samples.npz"
    np.savez_compressed(sample_archive_path, coords=stacked_samples.astype(np.float32), n=np.int32(n_vertices))
    sample_df = pd.DataFrame(sample_rows)
    sample_metrics_path = case_dir / "sample_metrics.csv"
    sample_df.to_csv(sample_metrics_path, index=False)

    generated_table = polygon_metric_table(stacked_samples)
    generated_summary = summarize_polygon_dataset(stacked_samples)
    distribution_distances = compare_polygon_metric_tables(reference_table, generated_table)
    fit_score = sample_df["fit_score"].to_numpy(dtype=float, copy=False)
    summary_row = {
        "case_name": case_name,
        "guidance_kind": guidance_kind,
        "samples_path": str(sample_archive_path),
        "sample_metrics_path": str(sample_metrics_path),
        "fit_score_mean": float(fit_score.mean()),
        "fit_score_std": float(fit_score.std()),
        "fit_score_p95": float(np.quantile(fit_score, 0.95)),
        "fit_success_rate": float(np.mean(fit_score >= float(reward.success_threshold))),
        "inside_fraction_mean": float(sample_df["inside_fraction"].mean()),
        "outside_penalty_mean": float(sample_df["outside_penalty"].mean()),
        "area_ratio_mean": float(sample_df["area_ratio"].mean()),
        "clearance_mean": float(sample_df["clearance_mean"].mean()),
        "generated_summary.score_mean": float(generated_summary["score_mean"]),
        "generated_summary.self_intersection_rate": float(generated_summary["self_intersection_rate"]),
        "distribution_distances.shape_distribution_shift_mean_normalized_w1": float(
            distribution_distances["shape_distribution_shift_mean_normalized_w1"]
        ),
    }
    if "surrogate_true_gap" in sample_df.columns:
        summary_row["surrogate_true_gap_mean"] = float(sample_df["surrogate_true_gap"].mean())
        summary_row["surrogate_pred_mean"] = float(sample_df["surrogate_pred"].mean())
    case_payload = {
        "summary_row": summary_row,
        "generated_summary": generated_summary,
        "distribution_distances": distribution_distances,
        "reward": reward.to_dict(),
    }
    (case_dir / "case_report.json").write_text(json.dumps(case_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary_row": summary_row,
        "sample_rows": sample_rows,
        "case_payload": case_payload,
        "representative": representative_payload,
    }


def _sample_with_seed(diffusion, *, shape: tuple[int, int], n_steps: int, guidance, seed: int) -> np.ndarray:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        samples = diffusion.p_sample_loop(shape, n_steps=n_steps, guidance_grad=guidance)
        return samples.detach().cpu().numpy()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _sample_trajectory_with_seed(
    diffusion,
    *,
    shape: tuple[int, int],
    n_steps: int,
    guidance,
    seed: int,
) -> np.ndarray:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        _, trajectory = diffusion.p_sample_loop_trajectory(
            shape,
            n_steps=n_steps,
            trajectory_index=0,
            guidance_grad=guidance,
        )
        return trajectory.detach().cpu().numpy().reshape(-1, shape[1] // 2, 2).astype(np.float32)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _predict_surrogate_scores(
    model: PocketContextSurrogateModel,
    *,
    samples: np.ndarray,
    pocket_coords: np.ndarray,
    device: torch.device,
    uses_t: bool,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        ligand = torch.from_numpy(samples).to(device=device, dtype=torch.float32)
        pocket = torch.from_numpy(pocket_coords).to(device=device, dtype=torch.float32).unsqueeze(0).expand(ligand.shape[0], -1, -1)
        timestep = torch.zeros((ligand.shape[0],), dtype=torch.long, device=device) if uses_t else None
        pred = model(ligand, pocket, t=timestep)
    return pred.detach().cpu().numpy().astype(np.float32, copy=False)


def _wrap_guidance_schedule(guidance, *, schedule_name: str, num_steps: int):
    if str(schedule_name).lower() == "all":
        return guidance
    return ScheduledGuidance(
        guidance=guidance,
        schedule=GuidanceSchedule(kind=str(schedule_name).lower(), num_steps=int(num_steps)),
    )


def _save_surrogate_timestep_figure(
    surrogate_results: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path | None:
    rows = []
    for label, result in surrogate_results.items():
        metrics_path = Path(result["metrics_path"])
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key, value in payload.get("test_timestep_mae", {}).items():
            rows.append(
                {
                    "label": "with t" if label == "with_t" else "no t",
                    "bin": int(key.split("_")[-1]),
                    "mae": float(value),
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, group in frame.groupby("label", sort=False):
        group = group.sort_values("bin")
        ax.plot(group["bin"], group["mae"], marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("noise bin (higher = noisier x_t)")
    ax.set_ylabel("test MAE")
    ax.set_title("Pocket Surrogate Error By Noise Level")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _save_guidance_metric_panel(summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    metrics = [
        ("fit_score_mean", "fit score mean"),
        ("fit_success_rate", "fit success rate"),
        ("distribution_distances.shape_distribution_shift_mean_normalized_w1", "shape shift"),
        ("generated_summary.self_intersection_rate", "self-intersection rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    axes = axes.reshape(-1)
    x = np.arange(summary_df.shape[0], dtype=float)
    labels = [str(value) for value in summary_df["case_name"].tolist()]
    for axis, (key, title) in zip(axes, metrics, strict=True):
        axis.bar(x, summary_df[key].to_numpy(dtype=float), color="tab:blue")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Pocket Guidance Comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _save_guidance_tradeoff(summary_df: pd.DataFrame, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = summary_df["fit_score_mean"].to_numpy(dtype=float)
    y = summary_df["distribution_distances.shape_distribution_shift_mean_normalized_w1"].to_numpy(dtype=float)
    ax.scatter(x, y, s=56, color="tab:blue")
    for _, row in summary_df.iterrows():
        ax.annotate(str(row["case_name"]), (float(row["fit_score_mean"]), float(row["distribution_distances.shape_distribution_shift_mean_normalized_w1"])), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("true pocket-fit mean")
    ax.set_ylabel("shape distribution shift (mean normalized W1)")
    ax.set_title("Pocket Guidance Tradeoff")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _save_best_sample_gallery(representative_samples: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    if not representative_samples:
        return None
    case_names = list(representative_samples.keys())
    num_cols = 3
    fig, axes = plt.subplots(len(case_names), num_cols, figsize=(3.2 * num_cols, 2.8 * len(case_names)))
    if len(case_names) == 1:
        axes = np.asarray([axes])
    for row_index, case_name in enumerate(case_names):
        payload = representative_samples[case_name]
        pocket = np.asarray(payload["pocket_coords"], dtype=np.float32)
        samples = np.asarray(payload["samples"], dtype=np.float32)
        scores = np.asarray(payload["fit_score"], dtype=np.float32)
        for col_index in range(num_cols):
            axis = axes[row_index, col_index]
            axis.set_aspect("equal")
            axis.axis("off")
            _draw_polygon(axis, pocket, color="black", linewidth=1.2)
            if col_index < samples.shape[0]:
                _draw_polygon(axis, samples[col_index], color="tab:blue", linewidth=1.1)
                axis.set_title(f"{case_name}\nfit={scores[col_index]:.2f}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _save_trajectory_visualizations(
    *,
    cfg: dict[str, Any],
    diffusion,
    eval_pockets: np.ndarray,
    reward: PocketFitConfig,
    n_vertices: int,
    n_steps: int,
    guidance_builders_by_case: dict[str, Any],
    figures_dir: Path,
    available_case_names: list[str],
) -> tuple[dict[str, str], dict[str, str | None]]:
    visualization_cfg = cfg.get("visualization") or {}
    if not isinstance(visualization_cfg, dict):
        raise ValueError("visualization config must be a mapping if provided")
    if not bool(visualization_cfg.get("enabled", True)):
        return {}, {}
    if eval_pockets.shape[0] == 0:
        return {}, {}

    pocket_index = int(visualization_cfg.get("pocket_index", 0))
    if not 0 <= pocket_index < eval_pockets.shape[0]:
        raise ValueError(f"visualization.pocket_index must be in [0, {eval_pockets.shape[0] - 1}], got {pocket_index}")
    fps = int(visualization_cfg.get("fps", 12))
    max_frames = int(visualization_cfg.get("max_frames", 90))
    trajectory_seed = int(visualization_cfg.get("trajectory_seed", int(cfg.get("seed", 0)) + 4242))

    default_individual_cases = [name for name in ("unguided", "analytic-pocket-fit", "surrogate-with-t") if name in available_case_names]
    panel_cases = _resolve_case_sequence(
        visualization_cfg.get("panel_cases"),
        available=available_case_names,
        default=available_case_names,
        field_name="visualization.panel_cases",
    )
    individual_cases = _resolve_case_sequence(
        visualization_cfg.get("individual_cases"),
        available=available_case_names,
        default=default_individual_cases,
        field_name="visualization.individual_cases",
    )
    compare_cases = _resolve_case_sequence(
        visualization_cfg.get("compare_cases"),
        available=available_case_names,
        default=[name for name in ("unguided", "surrogate-with-t") if name in available_case_names],
        field_name="visualization.compare_cases",
    )

    pocket_coords = np.asarray(eval_pockets[pocket_index], dtype=np.float32)
    pocket_tensor = torch.from_numpy(pocket_coords).to(diffusion.device, dtype=torch.float32)
    trajectory_payloads: dict[str, dict[str, np.ndarray | float]] = {}
    trajectory_case_names = list(dict.fromkeys(panel_cases + individual_cases + compare_cases))
    for case_name in trajectory_case_names:
        guidance_builder = guidance_builders_by_case[case_name]
        guidance = guidance_builder(pocket_tensor)
        trajectory = _sample_trajectory_with_seed(
            diffusion,
            shape=(1, n_vertices * 2),
            n_steps=n_steps,
            guidance=guidance,
            seed=trajectory_seed,
        )
        state = pocket_fit_state_numpy(trajectory, pocket_coords, reward)
        trajectory_payloads[case_name] = {
            "trajectory": trajectory.astype(np.float32),
            "fit_score": np.asarray(state.fit_score, dtype=np.float32),
            "inside_fraction": np.asarray(state.inside_fraction, dtype=np.float32),
            "outside_penalty": np.asarray(state.outside_penalty, dtype=np.float32),
        }

    animation_dir = figures_dir / "animations"
    animation_paths: dict[str, str] = {}
    for case_name in individual_cases:
        out_path = animation_dir / f"trajectory__pocket_{pocket_index:02d}__{slugify(case_name)}.gif"
        saved = _save_pocket_trajectory_gif(
            np.asarray(trajectory_payloads[case_name]["trajectory"], dtype=np.float32),
            pocket_coords=pocket_coords,
            reward=reward,
            out_path=out_path,
            title=case_name,
            fps=fps,
            max_frames=max_frames,
        )
        if saved is not None:
            animation_paths[f"trajectory_gif__{slugify(case_name)}"] = str(saved)

    if len(compare_cases) >= 2:
        ordered_compare = {case_name: np.asarray(trajectory_payloads[case_name]["trajectory"], dtype=np.float32) for case_name in compare_cases}
        compare_slug = "__vs__".join(slugify(case_name) for case_name in compare_cases)
        compare_path = animation_dir / f"trajectory_compare__pocket_{pocket_index:02d}__{compare_slug}.gif"
        saved = _save_pocket_comparison_gif(
            ordered_compare,
            pocket_coords=pocket_coords,
            reward=reward,
            out_path=compare_path,
            fps=fps,
            max_frames=max_frames,
        )
        if saved is not None:
            animation_paths["trajectory_compare_gif"] = str(saved)

    panel_path = _save_matched_seed_final_panel(
        {case_name: np.asarray(trajectory_payloads[case_name]["trajectory"], dtype=np.float32) for case_name in panel_cases},
        pocket_coords=pocket_coords,
        reward=reward,
        out_path=figures_dir / f"pocket_matched_seed_final_panel__pocket_{pocket_index:02d}.png",
    )
    figure_paths = {"matched_seed_final_panel": _stringify_path(panel_path)}
    return animation_paths, figure_paths


def _resolve_case_sequence(
    value: object,
    *,
    available: list[str],
    default: list[str],
    field_name: str,
) -> list[str]:
    if value is None:
        resolved = list(default)
    else:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a sequence of case names")
        resolved = [str(item) for item in value]
    for case_name in resolved:
        if case_name not in available:
            raise ValueError(f"{field_name} contains unknown case {case_name!r}; available={available}")
    return resolved


def _save_pocket_trajectory_gif(
    trajectory: np.ndarray,
    *,
    pocket_coords: np.ndarray,
    reward: PocketFitConfig,
    out_path: Path,
    title: str,
    fps: int,
    max_frames: int,
) -> Path | None:
    try:
        import PIL  # noqa: F401
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return None

    frames, frame_indices = select_animation_frames(np.asarray(trajectory, dtype=np.float32), max_frames=max_frames)
    state = pocket_fit_state_numpy(frames, pocket_coords, reward)
    x_all = np.concatenate([frames[:, :, 0].reshape(-1), pocket_coords[:, 0]], axis=0)
    y_all = np.concatenate([frames[:, :, 1].reshape(-1), pocket_coords[:, 1]], axis=0)
    x_pad = max((float(x_all.max()) - float(x_all.min())) * 0.15, 0.25)
    y_pad = max((float(y_all.max()) - float(y_all.min())) * 0.15, 0.25)

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.82)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(float(x_all.min()) - x_pad, float(x_all.max()) + x_pad)
    ax.set_ylim(float(y_all.min()) - y_pad, float(y_all.max()) + y_pad)
    _draw_polygon(ax, pocket_coords, color="0.7", linewidth=1.6)
    line, = ax.plot([], [], color="tab:blue", linewidth=2.0)
    points = ax.scatter([], [], s=18, color="tab:blue")
    title_artist = fig.suptitle("", y=0.97, fontsize=12)

    def _update(frame_idx: int):
        xy = frames[frame_idx]
        closed = np.vstack([xy, xy[:1]])
        line.set_data(closed[:, 0], closed[:, 1])
        points.set_offsets(xy)
        title_artist.set_text(
            f"{title}\nframe {frame_indices[frame_idx] + 1}/{trajectory.shape[0]} | "
            f"fit={float(state.fit_score[frame_idx]):.2f} | inside={float(state.inside_fraction[frame_idx]):.2f}"
        )
        return line, points, title_artist

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=int(round(1000 / fps)),
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def _save_pocket_comparison_gif(
    trajectories: dict[str, np.ndarray],
    *,
    pocket_coords: np.ndarray,
    reward: PocketFitConfig,
    out_path: Path,
    fps: int,
    max_frames: int,
) -> Path | None:
    try:
        import PIL  # noqa: F401
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return None

    if not trajectories:
        return None
    case_names = list(trajectories.keys())
    first_trajectory = np.asarray(next(iter(trajectories.values())), dtype=np.float32)
    _, frame_indices = select_animation_frames(first_trajectory, max_frames=max_frames)
    frames_by_case = {case_name: np.asarray(trajectory, dtype=np.float32)[frame_indices] for case_name, trajectory in trajectories.items()}
    state_by_case = {
        case_name: pocket_fit_state_numpy(frames, pocket_coords, reward)
        for case_name, frames in frames_by_case.items()
    }

    x_arrays = [pocket_coords[:, 0]]
    y_arrays = [pocket_coords[:, 1]]
    for frames in frames_by_case.values():
        x_arrays.append(frames[:, :, 0].reshape(-1))
        y_arrays.append(frames[:, :, 1].reshape(-1))
    x_all = np.concatenate(x_arrays, axis=0)
    y_all = np.concatenate(y_arrays, axis=0)
    x_pad = max((float(x_all.max()) - float(x_all.min())) * 0.15, 0.25)
    y_pad = max((float(y_all.max()) - float(y_all.min())) * 0.15, 0.25)

    fig, axes = plt.subplots(1, len(case_names), figsize=(4.1 * len(case_names), 4.4))
    if len(case_names) == 1:
        axes = np.asarray([axes])
    lines = []
    points = []
    title_artists = []
    for axis, case_name in zip(axes, case_names, strict=True):
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_xlim(float(x_all.min()) - x_pad, float(x_all.max()) + x_pad)
        axis.set_ylim(float(y_all.min()) - y_pad, float(y_all.max()) + y_pad)
        _draw_polygon(axis, pocket_coords, color="0.7", linewidth=1.5)
        line, = axis.plot([], [], color="tab:blue", linewidth=1.9)
        point = axis.scatter([], [], s=18, color="tab:blue")
        title_artist = axis.set_title(case_name, fontsize=10)
        lines.append(line)
        points.append(point)
        title_artists.append(title_artist)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.82, wspace=0.10)
    figure_title = fig.suptitle("", y=0.96, fontsize=12)

    def _update(frame_idx: int):
        artists: list[object] = [figure_title]
        figure_title.set_text(f"Matched-seed pocket trajectory comparison | frame {frame_indices[frame_idx] + 1}/{first_trajectory.shape[0]}")
        for idx, case_name in enumerate(case_names):
            xy = frames_by_case[case_name][frame_idx]
            closed = np.vstack([xy, xy[:1]])
            lines[idx].set_data(closed[:, 0], closed[:, 1])
            points[idx].set_offsets(xy)
            title_artists[idx].set_text(
                f"{case_name}\nfit={float(state_by_case[case_name].fit_score[frame_idx]):.2f} | "
                f"inside={float(state_by_case[case_name].inside_fraction[frame_idx]):.2f}"
            )
            artists.extend([lines[idx], points[idx]])
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        interval=int(round(1000 / fps)),
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def _save_matched_seed_final_panel(
    trajectories: dict[str, np.ndarray],
    *,
    pocket_coords: np.ndarray,
    reward: PocketFitConfig,
    out_path: Path,
) -> Path | None:
    if not trajectories:
        return None
    case_names = list(trajectories.keys())
    fig, axes = plt.subplots(1, len(case_names), figsize=(3.5 * len(case_names), 3.7))
    if len(case_names) == 1:
        axes = np.asarray([axes])

    x_arrays = [pocket_coords[:, 0]]
    y_arrays = [pocket_coords[:, 1]]
    for trajectory in trajectories.values():
        final_xy = np.asarray(trajectory, dtype=np.float32)[-1]
        x_arrays.append(final_xy[:, 0])
        y_arrays.append(final_xy[:, 1])
    x_all = np.concatenate(x_arrays, axis=0)
    y_all = np.concatenate(y_arrays, axis=0)
    x_pad = max((float(x_all.max()) - float(x_all.min())) * 0.15, 0.25)
    y_pad = max((float(y_all.max()) - float(y_all.min())) * 0.15, 0.25)

    for axis, case_name in zip(axes, case_names, strict=True):
        final_xy = np.asarray(trajectories[case_name], dtype=np.float32)[-1]
        state = pocket_fit_state_numpy(final_xy[None, ...], pocket_coords, reward)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_xlim(float(x_all.min()) - x_pad, float(x_all.max()) + x_pad)
        axis.set_ylim(float(y_all.min()) - y_pad, float(y_all.max()) + y_pad)
        _draw_polygon(axis, pocket_coords, color="0.7", linewidth=1.5)
        _draw_polygon(axis, final_xy, color="tab:blue", linewidth=1.8)
        axis.scatter(final_xy[:, 0], final_xy[:, 1], s=16, color="tab:blue")
        axis.set_title(
            f"{case_name}\nfit={float(state.fit_score[0]):.2f} | inside={float(state.inside_fraction[0]):.2f}",
            fontsize=9,
        )
    fig.suptitle("Matched-Seed Final Samples On One Pocket", y=0.98, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _draw_polygon(axis, xy: np.ndarray, *, color: str, linewidth: float) -> None:
    coords = np.asarray(xy, dtype=np.float32)
    closed = np.vstack([coords, coords[:1]])
    axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth)


def _write_interpretation_guide(
    out_path: Path,
    *,
    reward: PocketFitConfig,
    summary_df: pd.DataFrame,
    figure_paths: dict[str, str],
    animation_paths: dict[str, str],
) -> Path:
    lines = [
        f"# How To Read {out_path.parent.name}",
        "",
        "This study uses an exact pocket-fit reward on centered ligand polygons and fixed pocket polygons.",
        "The pocket stays fixed in a canonical frame, so this is a conditional-shape analogue rather than a full pose-generation benchmark.",
        "",
        "## What The Study Is Testing",
        "",
        "- whether a learned pocket-conditioned surrogate should be trained on noisy ligand coordinates",
        "- whether conditioning that surrogate on timestep `t` improves guidance quality",
        "- whether a learned surrogate can approach analytic guidance without paying an avoidable manifold-fidelity cost",
        "- whether better reward optimization also preserves manifold fidelity",
        "",
        "## Reward Terms",
        "",
        f"- target area ratio: `{reward.target_area_ratio:.3f}`",
        f"- target clearance: `{reward.target_clearance:.3f}`",
        "- the true fit score rewards staying inside the pocket while matching a target fill level rather than collapsing to a tiny central shape",
        "",
        "## Metrics",
        "",
        "- `fit_score_mean`: true pocket-fit reward on final samples. Higher is better.",
        "- `fit_success_rate`: share of samples clearing the study success threshold.",
        "- `distribution_distances.shape_distribution_shift_mean_normalized_w1`: manifold drift relative to the ligand training distribution. Lower is better.",
        "- `generated_summary.self_intersection_rate`: explicit geometric invalidity. Lower is better.",
        "",
        "## Reading The Figures",
        "",
    ]
    if "surrogate_timestep_mae" in figure_paths:
        lines.append("- `surrogate_timestep_mae.png`: lower means better surrogate prediction; if the `with t` curve stays below `no t` at high-noise bins, timestep conditioning is helping in the regime that matters for diffusion guidance.")
    if "guidance_metric_panel" in figure_paths:
        lines.append("- `pocket_guidance_metric_panel.png`: the main case comparison across reward gain and safety metrics.")
    if "guidance_tradeoff" in figure_paths:
        lines.append("- `pocket_guidance_tradeoff.png`: right and down is preferred, because it means better true pocket fit with less manifold drift.")
    if "best_sample_gallery" in figure_paths:
        lines.append("- `pocket_best_sample_gallery.png`: best-scoring samples on one held-out pocket. Use this to verify that reward gains correspond to visually plausible fits rather than pathological shrinkage or protrusion.")
    if "matched_seed_final_panel" in figure_paths:
        lines.append("- `pocket_matched_seed_final_panel__*.png`: same initial noise, same pocket, different guidance. Use this to compare end states fairly.")
    if animation_paths:
        lines.append("- `figures/animations/*.gif`: matched-seed denoising movies. These are the clearest visual comparison for unguided versus guided generation on the same pocket.")
    lines.extend(
        [
            "",
        "## Thesis Framing",
        "",
        "- This study does not claim to solve molecule generation.",
        "- Analytic pocket-fit guidance is the upper-bound reference because the true reward is available exactly in polygons.",
        "- It isolates whether a pocket-conditioned guidance surrogate should see noisy states and timestep information before you deploy the same design in the binding-affinity pipeline.",
        "- The key decision rule is whether `with t` improves true reward optimization enough to justify the extra conditioning without creating a worse score-fidelity tradeoff.",
        ]
    )
    if not summary_df.empty:
        lines.extend(
            [
                "",
                "## Case Summary",
                "",
                "| Case | Fit Mean | Success Rate | Shape Shift | Self-Intersection |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in summary_df.iterrows():
            lines.append(
                f"| {row['case_name']} | {float(row['fit_score_mean']):.4f} | {float(row['fit_success_rate']):.4f} | "
                f"{float(row['distribution_distances.shape_distribution_shift_mean_normalized_w1']):.4f} | "
                f"{float(row['generated_summary.self_intersection_rate']):.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _load_reward_config(cfg: dict[str, Any]) -> PocketFitConfig:
    return PocketFitConfig(
        target_area_ratio=float(cfg.get("target_area_ratio", 0.22)),
        target_clearance=float(cfg.get("target_clearance", 0.10)),
        inside_temperature=float(cfg.get("inside_temperature", 0.06)),
        outside_weight=float(cfg.get("outside_weight", 3.5)),
        area_weight=float(cfg.get("area_weight", 8.0)),
        clearance_weight=float(cfg.get("clearance_weight", 4.0)),
        sample_fractions=tuple(float(value) for value in cfg.get("sample_fractions", (0.0, 0.25, 0.5, 0.75))),
        success_threshold=float(cfg.get("success_threshold", 0.55)),
    )


def _stringify_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_project_path("configs/study_pocket_fit_conditioning.yaml")),
        help="Path to pocket-fit study config",
    )
    args = parser.parse_args()
    run_pocket_fit_study_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
