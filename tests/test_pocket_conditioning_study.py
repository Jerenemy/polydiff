import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from polydiff.data.gen_polygons import batch
from polydiff.pocket_conditioning.data import generate_pocket_fit_pair_splits
from polydiff.pocket_conditioning.geometry import PocketFitConfig, pocket_fit_state_numpy
from polydiff.pocket_conditioning.study import run_pocket_fit_study_from_config


def test_pocket_fit_reward_prefers_matched_scale():
    reward = PocketFitConfig()
    pocket = _regular_polygon(10, radius=2.0)
    matched = _regular_polygon(6, radius=1.55)
    tiny = _regular_polygon(6, radius=0.35)
    oversized = _regular_polygon(6, radius=2.2)

    matched_score = float(pocket_fit_state_numpy(matched, pocket, reward).fit_score[0])
    tiny_score = float(pocket_fit_state_numpy(tiny, pocket, reward).fit_score[0])
    oversized_score = float(pocket_fit_state_numpy(oversized, pocket, reward).fit_score[0])

    assert matched_score > tiny_score
    assert matched_score > oversized_score


def test_generate_pocket_fit_pair_splits_writes_expected_files(tmp_path):
    ligand_path = tmp_path / "hexagons.npz"
    coords, _, _ = batch(n=6, num=24, seed=0, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=5)
    np.savez_compressed(ligand_path, coords=coords.astype(np.float32), n=np.int32(6))

    result = generate_pocket_fit_pair_splits(
        ligand_data_path=ligand_path,
        out_dir=tmp_path / "pairs",
        reward=PocketFitConfig(),
        seed=0,
        ligand_size=6,
        pocket_size=10,
        num_train_pairs=12,
        num_val_pairs=4,
        num_test_pairs=4,
        num_eval_pockets=2,
        pocket_radial_sigma=0.08,
        pocket_angle_sigma=0.03,
        pocket_smooth_passes=5,
        pocket_scale_min=1.9,
        pocket_scale_max=2.2,
        ligand_scale_min=0.6,
        ligand_scale_max=1.4,
    )

    assert result["train"].exists()
    assert result["val"].exists()
    assert result["test"].exists()
    assert result["eval_pockets"].exists()
    assert result["metadata"].exists()


def test_run_pocket_fit_study_writes_reports_and_figures(tmp_path):
    pytest.importorskip("PIL")
    ligand_path = tmp_path / "hexagons.npz"
    coords, _, _ = batch(n=6, num=48, seed=1, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=5)
    np.savez_compressed(ligand_path, coords=coords.astype(np.float32), n=np.int32(6))

    config_path = tmp_path / "study_pocket_fit.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "synthetic-pocket-fit",
                    "root_dir": str(tmp_path / "studies"),
                },
                "seed": 0,
                "device": "cpu",
                "data": {
                    "ligand_data_path": str(ligand_path),
                    "generated_data_dir": str(tmp_path / "generated_pairs"),
                    "force_regenerate": True,
                    "ligand_size": 6,
                    "pocket_size": 10,
                    "num_train_pairs": 24,
                    "num_val_pairs": 8,
                    "num_test_pairs": 8,
                    "num_eval_pockets": 2,
                    "pocket_radial_sigma": 0.08,
                    "pocket_angle_sigma": 0.03,
                    "pocket_smooth_passes": 5,
                    "pocket_scale_min": 1.9,
                    "pocket_scale_max": 2.2,
                    "ligand_scale_min": 0.6,
                    "ligand_scale_max": 1.4,
                },
                "reward": {
                    "target_area_ratio": 0.22,
                    "target_clearance": 0.10,
                    "inside_temperature": 0.06,
                    "outside_weight": 3.5,
                    "area_weight": 8.0,
                    "clearance_weight": 4.0,
                    "success_threshold": 0.55,
                },
                "diffusion": {
                    "train_config": str(Path("configs/train_diffusion.yaml")),
                    "overrides": {
                        "experiment_name": "synthetic-pocket-fit-prior",
                        "device": "cpu",
                        "data.path": str(ligand_path),
                        "data.batch_size": 8,
                        "model.type": "mlp",
                        "model.hidden_dim": 16,
                        "model.time_emb_dim": 8,
                        "model.num_layers": 1,
                        "diffusion.n_steps": 6,
                        "training.epochs": 1,
                        "training.lr": 1e-3,
                        "training.log_every": 1,
                        "training.save_dir": str(tmp_path / "models"),
                    },
                },
                "surrogate": {
                    "model": {
                        "hidden_dim": 24,
                        "num_layers": 2,
                        "num_rbf": 8,
                        "cutoff": 4.0,
                        "dropout": 0.0,
                        "time_emb_dim": 8,
                    },
                    "noise": {
                        "enabled": True,
                        "n_steps": 32,
                        "beta_start": 1e-4,
                        "beta_end": 1e-2,
                    },
                    "training": {
                        "batch_size": 8,
                        "epochs": 2,
                        "lr": 1e-3,
                        "weight_decay": 0.0,
                        "eval_repeats": 1,
                        "log_every": 1,
                        "save_dir": str(tmp_path / "surrogate_models"),
                        "processed_dir": str(tmp_path / "surrogate_processed"),
                    },
                },
                "guidance": {
                    "num_samples_per_pocket": 6,
                    "n_steps": 6,
                    "schedule": "late",
                    "scale": 4.0,
                    "seed": 0,
                },
                "visualization": {
                    "enabled": True,
                    "pocket_index": 0,
                    "trajectory_seed": 123,
                    "individual_cases": ["unguided", "surrogate-with-t"],
                    "compare_cases": ["unguided", "surrogate-with-t"],
                    "panel_cases": ["unguided", "analytic-pocket-fit", "surrogate-with-t"],
                    "max_frames": 7,
                    "fps": 6,
                },
            },
            f,
            sort_keys=False,
        )

    report = run_pocket_fit_study_from_config(config_path)

    assert Path(report["summary_csv_path"]).exists()
    assert Path(report["summary_json_path"]).exists()
    assert Path(report["sample_metrics_csv_path"]).exists()
    assert Path(report["interpretation_guide_path"]).exists()
    assert Path(report["figure_paths"]["surrogate_timestep_mae"]).exists()
    assert Path(report["figure_paths"]["guidance_metric_panel"]).exists()
    assert Path(report["figure_paths"]["guidance_tradeoff"]).exists()
    assert Path(report["figure_paths"]["best_sample_gallery"]).exists()
    assert Path(report["figure_paths"]["matched_seed_final_panel"]).exists()
    assert Path(report["animation_paths"]["trajectory_compare_gif"]).exists()
    assert Path(report["animation_paths"]["trajectory_gif__unguided"]).exists()


def _regular_polygon(num_vertices: int, *, radius: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, int(num_vertices), endpoint=False, dtype=np.float32)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)
