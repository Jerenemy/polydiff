import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from polydiff.data.diagnostics import (
    compare_polygon_metric_tables,
    outlier_failure_mode_summary,
    outlier_polygon_indices,
    polygon_metric_table,
    representative_polygon_indices,
    summarize_polygon_dataset,
)
from polydiff.data.polygon_dataset import load_polygon_dataset
from polydiff.data.gen_polygons import batch
from polydiff.models.diffusion import build_denoiser
from polydiff.studies.plots import save_metric_sweep_figure
from polydiff.studies.run import _add_tradeoff_labels, refresh_study_outputs, run_study_from_config
from polydiff.studies.runtime import apply_dotted_overrides, load_study_spec, resolve_case_placeholders


def test_metric_tables_capture_distribution_shift():
    reference_coords, _, _ = batch(n=6, num=32, seed=0, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=5)
    observed_coords, _, _ = batch(n=6, num=32, seed=1, radial_sigma=0.30, angle_sigma=0.20, smooth_passes=1)

    reference_table = polygon_metric_table(reference_coords)
    observed_table = polygon_metric_table(observed_coords)
    distances = compare_polygon_metric_tables(reference_table, observed_table)

    assert reference_table.shape[0] == 32
    assert observed_table.shape[0] == 32
    assert distances["score_normalized_w1"] > 0.0
    assert distances["shape_distribution_shift_mean_normalized_w1"] > 0.0
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


def test_load_study_spec_parses_parallel_options(tmp_path):
    sample_config_path = tmp_path / "sample.yaml"
    sample_config_path.write_text("sampling:\n  num_samples: 4\n", encoding="utf-8")
    config_path = tmp_path / "study.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "parallel-study",
                    "parallel": {
                        "enabled": True,
                        "max_workers": 3,
                        "require_cuda": False,
                    },
                },
                "cases": [
                    {
                        "name": "sample-a",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                    }
                ],
            },
            f,
            sort_keys=False,
        )

    spec = load_study_spec(config_path)

    assert spec.parallel.enabled is True
    assert spec.parallel.max_workers == 3
    assert spec.parallel.require_cuda is False


def test_load_study_spec_parses_case_tags(tmp_path):
    sample_config_path = tmp_path / "sample.yaml"
    sample_config_path.write_text("sampling:\n  num_samples: 4\n", encoding="utf-8")
    config_path = tmp_path / "study_tags.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {"name": "tagged-study"},
                "cases": [
                    {
                        "name": "sample-a",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "architecture_baseline",
                            "architecture": "mlp",
                            "dataset_noise": "baseline",
                        },
                    }
                ],
            },
            f,
            sort_keys=False,
        )

    spec = load_study_spec(config_path)

    assert spec.cases[0].tags["analysis_group"] == "architecture_baseline"
    assert spec.cases[0].tags["architecture"] == "mlp"


def test_tradeoff_label_layout_avoids_overlap_for_close_points():
    summary_df = pd.DataFrame(
        [
            {
                "case_name": "sample-a",
                "generated_summary.score_mean": 0.4218,
                "distribution_distances.distribution_shift_mean_normalized_w1": 220000.0,
            },
            {
                "case_name": "sample-b",
                "generated_summary.score_mean": 0.4222,
                "distribution_distances.distribution_shift_mean_normalized_w1": 219900.0,
            },
            {
                "case_name": "sample-c",
                "generated_summary.score_mean": 0.4226,
                "distribution_distances.distribution_shift_mean_normalized_w1": 220100.0,
            },
            {
                "case_name": "sample-d",
                "generated_summary.score_mean": 0.4230,
                "distribution_distances.distribution_shift_mean_normalized_w1": 220050.0,
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x_key = "generated_summary.score_mean"
    y_key = "distribution_distances.distribution_shift_mean_normalized_w1"
    ax.scatter(summary_df[x_key], summary_df[y_key], s=46, color="tab:blue")
    ax.margins(x=0.14, y=0.10)
    annotations = _add_tradeoff_labels(ax, summary_df, x_key=x_key, y_key=y_key)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [annotation.get_window_extent(renderer).expanded(1.01, 1.08) for annotation in annotations]
    assert len(annotations) == len(summary_df)
    for index, bbox in enumerate(boxes):
        for other in boxes[index + 1 :]:
            assert not bbox.overlaps(other)
    plt.close(fig)


def test_metric_sweep_figure_supports_symlog_strength_axis(tmp_path):
    summary_df = pd.DataFrame(
        [
            {
                "case_tags.guidance_scale": 0.0,
                "generated_summary.score_mean": 0.41,
                "generated_summary.score_p99": 0.72,
                "score_threshold_rates.score_ge_0p7_rate": 0.02,
                "distribution_distances.shape_distribution_shift_mean_normalized_w1": 0.40,
            },
            {
                "case_tags.guidance_scale": 1.0,
                "generated_summary.score_mean": 0.43,
                "generated_summary.score_p99": 0.76,
                "score_threshold_rates.score_ge_0p7_rate": 0.04,
                "distribution_distances.shape_distribution_shift_mean_normalized_w1": 0.38,
            },
            {
                "case_tags.guidance_scale": 1000.0,
                "generated_summary.score_mean": 0.39,
                "generated_summary.score_p99": 0.63,
                "score_threshold_rates.score_ge_0p7_rate": 0.01,
                "distribution_distances.shape_distribution_shift_mean_normalized_w1": 1.50,
            },
        ]
    )

    out_path = tmp_path / "guidance_strength_sweep.png"
    result = save_metric_sweep_figure(
        summary_df,
        out_path,
        x_key="case_tags.guidance_scale",
        x_label="guidance scale",
        x_scale="symlog",
        title="Guidance Strength Sweep",
    )

    assert result == out_path
    assert out_path.exists()


def test_outlier_failure_mode_summary_counts_self_intersections():
    reference_coords, _, _ = batch(n=6, num=24, seed=0, radial_sigma=0.10, angle_sigma=0.05, smooth_passes=4)
    reference_table = polygon_metric_table(reference_coords)

    observed_table = reference_table.copy()
    observed_table.loc[0, "self_intersection"] = 1.0
    summary = outlier_failure_mode_summary(reference_table, observed_table, outlier_indices=[0, 1, 2])

    assert summary["outlier_count"] == 3
    assert summary["self_intersection_count"] == 1
    assert summary["self_intersection_rate"] == 1.0 / 3.0


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
    assert report_payload["status"] == "completed"
    summary_csv_path = Path(report_payload["summary_csv_path"])
    summary_json_path = Path(report_payload["summary_json_path"])
    case_results_path = Path(report_payload["case_results_path"])
    interpretation_guide_path = Path(report_payload["interpretation_guide_path"])
    assert summary_csv_path.exists()
    assert summary_json_path.exists()
    assert case_results_path.exists()
    assert interpretation_guide_path.exists()
    figure_paths = report_payload["figure_paths_by_case"]["sample-unguided"]
    case_report_path = Path(report_payload["case_report_paths_by_case"]["sample-unguided"])
    assert case_report_path.exists()
    assert Path(figure_paths["score_distribution"]).exists()
    assert Path(figure_paths["representative_gallery"]).exists()
    assert Path(figure_paths["outlier_gallery"]).exists()
    assert Path(figure_paths["pca_projection"]).exists()

    with open(case_results_path, "r", encoding="utf-8") as f:
        case_results_payload = json.load(f)
    sample_case_result = case_results_payload["sample-unguided"]
    samples_out_path = Path(sample_case_result["samples_out_path"])
    with np.load(samples_out_path, allow_pickle=True) as npz_data:
        coords = np.asarray(npz_data["coords"], dtype=np.float32)
        meta = dict(npz_data["meta"].item())
    np.savez_compressed(
        samples_out_path,
        coords=(coords * 1.25 + np.asarray([0.40, -0.30], dtype=np.float32)).astype(np.float32),
        n=np.int32(coords.shape[1]),
        meta={**meta, "canonicalize_output": False},
    )

    report_path.unlink()
    summary_csv_path.unlink()
    summary_json_path.unlink()

    refreshed_report_path = refresh_study_outputs(Path(report_payload["study_dir"]))
    assert refreshed_report_path.exists()
    with open(refreshed_report_path, "r", encoding="utf-8") as f:
        refreshed_report_payload = json.load(f)
    assert refreshed_report_payload["status"] == "completed"
    assert Path(refreshed_report_payload["summary_csv_path"]).exists()
    refreshed_dataset = load_polygon_dataset(samples_out_path)
    refreshed_summary = summarize_polygon_dataset(refreshed_dataset)
    assert refreshed_summary["raw_centroid_norm_mean"] < 1e-5
    assert abs(refreshed_summary["raw_rms_radius_mean"] - 1.0) < 1e-5

    refreshed_summary_df = pd.read_csv(Path(refreshed_report_payload["summary_csv_path"]))
    assert "distribution_distances.shape_distribution_shift_mean_normalized_w1" in refreshed_summary_df.columns
    assert "generated_summary.score_p99" in refreshed_summary_df.columns
    assert "score_threshold_rates.score_ge_0p7_rate" in refreshed_summary_df.columns


def test_run_study_from_config_supports_parallel_case_execution(tmp_path):
    reference_coords, _, _ = batch(n=4, num=16, seed=11, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=4)
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

    sample_config_path = tmp_path / "sample_parallel.yaml"
    with open(sample_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment_name": "parallel-study-sample",
                "seed": 0,
                "device": "cpu",
                "model": {},
                "sampling": {
                    "num_samples": 4,
                    "guidance": {"enabled": False},
                    "restoration": {"enabled": False},
                    "diagnostics": {"enabled": True, "reference_data_path": str(reference_path)},
                },
            },
            f,
            sort_keys=False,
        )

    study_config_path = tmp_path / "study_parallel.yaml"
    with open(study_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "parallel-study",
                    "root_dir": str(tmp_path / "studies"),
                    "parallel": {
                        "enabled": True,
                        "max_workers": 2,
                        "require_cuda": False,
                    },
                    "summary": {
                        "reference_data_path": str(reference_path),
                        "representative_count": 4,
                        "outlier_count": 4,
                        "max_projection_points": 128,
                    },
                },
                "cases": [
                    {
                        "name": "sample-a",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "architecture_baseline",
                            "architecture": "mlp",
                            "dataset_noise": "baseline",
                            "guidance_schedule": "unguided",
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_a.npz"),
                        },
                    },
                    {
                        "name": "sample-b",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "architecture_baseline",
                            "architecture": "gat",
                            "dataset_noise": "baseline",
                            "guidance_schedule": "unguided",
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_b.npz"),
                        },
                    },
                ],
            },
            f,
            sort_keys=False,
        )

    report_path = run_study_from_config(study_config_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_payload = json.load(f)

    assert report_payload["status"] == "completed"
    assert report_payload["parallel"]["enabled"] is True
    assert report_payload["current_case_names"] == []
    assert Path(report_payload["interpretation_guide_path"]).exists()
    assert Path(report_payload["study_figure_paths"]["architecture_score_overlay__baseline"]).exists()
    assert Path(report_payload["study_figure_paths"]["architecture_metric_panel__baseline"]).exists()

    case_results_path = Path(report_payload["case_results_path"])
    with open(case_results_path, "r", encoding="utf-8") as f:
        case_results_payload = json.load(f)

    for case_name in ("sample-a", "sample-b"):
        case_result = case_results_payload[case_name]
        assert Path(case_result["samples_out_path"]).exists()
        assert Path(case_result["diagnostics_path"]).exists()
        assert Path(case_result["study_case_log_path"]).exists()
