import json
from pathlib import Path

import numpy as np
import torch
import yaml

from polydiff.data.diagnostics import (
    compare_polygon_metric_tables,
    outlier_polygon_indices,
    polygon_metric_table,
    representative_polygon_indices,
    summarize_polygon_dataset,
)
from polydiff.data.gen_polygons import batch
from polydiff.models.diffusion import build_denoiser
from polydiff.studies.run import run_study_from_config
from polydiff.studies.runtime import apply_dotted_overrides, resolve_case_placeholders


def test_metric_tables_capture_distribution_shift():
    reference_coords, _, _ = batch(n=6, num=32, seed=0, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=5)
    observed_coords, _, _ = batch(n=6, num=32, seed=1, radial_sigma=0.30, angle_sigma=0.20, smooth_passes=1)

    reference_table = polygon_metric_table(reference_coords)
    observed_table = polygon_metric_table(observed_coords)
    distances = compare_polygon_metric_tables(reference_table, observed_table)

    assert reference_table.shape[0] == 32
    assert observed_table.shape[0] == 32
    assert distances["score_normalized_w1"] > 0.0
    assert distances["distribution_shift_mean_normalized_w1"] > 0.0


def test_representative_and_outlier_indices_use_reference_anomaly():
    reference_coords, _, _ = batch(n=6, num=12, seed=0, radial_sigma=0.12, angle_sigma=0.06, smooth_passes=4)
    observed_coords = np.asarray(reference_coords, dtype=np.float32, copy=True)
    observed_coords[0] = observed_coords[0] + np.asarray([4.0, -3.0], dtype=np.float32)

    representative = representative_polygon_indices(reference_coords, observed_coords, count=3)
    outliers = outlier_polygon_indices(reference_coords, observed_coords, count=3)

    assert 0 not in representative.tolist()
    assert outliers[0] == 0


def test_runtime_override_helpers_support_placeholders():
    cfg = {"model": {"type": "mlp"}, "sampling": {"num_samples": 8}}
    placeholders = resolve_case_placeholders(
        {"model.run": "{{baseline.run_name}}", "sampling.num_samples": 16},
        case_results={"baseline": {"run_name": "run_0001__baseline"}},
    )
    resolved = apply_dotted_overrides(cfg, placeholders)

    assert resolved["model"]["run"] == "run_0001__baseline"
    assert resolved["sampling"]["num_samples"] == 16


def test_run_study_from_config_executes_sample_case_and_writes_reports(tmp_path):
    reference_coords, _, _ = batch(n=4, num=24, seed=4, radial_sigma=0.10, angle_sigma=0.05, smooth_passes=4)
    reference_path = tmp_path / "reference.npz"
    np.savez_compressed(reference_path, coords=reference_coords.astype(np.float32), n=np.int32(4))

    model = build_denoiser(
        data_dim=8,
        model_cfg={
            "type": "mlp",
            "hidden_dim": 16,
            "time_emb_dim": 8,
            "num_layers": 1,
        },
    )
    checkpoint_path = tmp_path / "diffusion_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "diffusion": {"n_steps": 4, "beta_start": 1e-4, "beta_end": 2e-2},
            "model_cfg": {"type": "mlp", "hidden_dim": 16, "time_emb_dim": 8, "num_layers": 1},
            "n_vertices": 4,
            "max_vertices": 4,
            "global_step": 0,
            "run_name": None,
            "training_data_path": str(reference_path),
            "training_data_summary": summarize_polygon_dataset(reference_coords),
        },
        checkpoint_path,
    )

    sample_config_path = tmp_path / "sample_config.yaml"
    with open(sample_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment_name": "study-sample",
                "seed": 0,
                "device": "cpu",
                "model": {},
                "sampling": {
                    "num_samples": 8,
                    "out_path": str(tmp_path / "samples.npz"),
                    "guidance": {"enabled": False},
                    "restoration": {"enabled": False},
                    "diagnostics": {"enabled": True, "reference_data_path": str(reference_path)},
                },
            },
            f,
            sort_keys=False,
        )

    study_config_path = tmp_path / "study.yaml"
    with open(study_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "unit-study",
                    "root_dir": str(tmp_path / "studies"),
                    "summary": {
                        "reference_data_path": str(reference_path),
                        "representative_count": 4,
                        "outlier_count": 4,
                        "max_projection_points": 128,
                    },
                },
                "cases": [
                    {
                        "name": "sample-unguided",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.num_samples": 8,
                        },
                    }
                ],
            },
            f,
            sort_keys=False,
        )

    report_path = run_study_from_config(study_config_path)

    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        report_payload = json.load(f)
    summary_csv_path = Path(report_payload["summary_csv_path"])
    case_results_path = Path(report_payload["case_results_path"])
    assert summary_csv_path.exists()
    assert case_results_path.exists()
    figure_paths = report_payload["figure_paths_by_case"]["sample-unguided"]
    assert Path(figure_paths["score_distribution"]).exists()
    assert Path(figure_paths["representative_gallery"]).exists()
    assert Path(figure_paths["outlier_gallery"]).exists()
    assert Path(figure_paths["pca_projection"]).exists()
