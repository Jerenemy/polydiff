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
from polydiff.data.gen_polygons import batch, batch_variable_sizes
from polydiff.models.diffusion import build_denoiser
from polydiff.studies.plots import _guidance_case_color_map, _plot_guidance_summary_bars, save_metric_sweep_figure
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


def test_guidance_summary_bars_keep_case_colors_consistent():
    summary_df = pd.DataFrame(
        [
            {"case_name": "polygon-gat-avg-with-t", "final_mae": 0.11, "final_ema_loss": 0.0201},
            {"case_name": "polygon-gat-max-with-t", "final_mae": 0.14, "final_ema_loss": 0.0212},
            {"case_name": "polygon-gat-sum-with-t", "final_mae": 0.12, "final_ema_loss": 0.0198},
        ]
    )

    case_colors = _guidance_case_color_map(summary_df, [])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    _plot_guidance_summary_bars(
        axes[0],
        summary_df,
        metric_key="final_mae",
        metric_label="final MAE",
        higher_is_better=False,
        case_colors=case_colors,
    )
    _plot_guidance_summary_bars(
        axes[1],
        summary_df,
        metric_key="final_ema_loss",
        metric_label="final EMA loss",
        higher_is_better=False,
        case_colors=case_colors,
    )

    def label_to_color(ax):
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        return {
            label: patch.get_facecolor()
            for label, patch in zip(labels, ax.patches, strict=True)
        }

    colors_a = label_to_color(axes[0])
    colors_b = label_to_color(axes[1])
    assert colors_a.keys() == colors_b.keys()
    for label in colors_a:
        assert colors_a[label] == colors_b[label]
    plt.close(fig)


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


def test_run_study_from_config_writes_guidance_training_summary_plot(tmp_path):
    data_path = tmp_path / "mixed_polygons.npz"
    coords, num_vertices, score, _ = batch_variable_sizes(
        size_values=[5, 6, 7],
        size_probabilities=[0.3, 0.4, 0.3],
        num=18,
        seed=3,
        radial_sigma=0.18,
        angle_sigma=0.12,
        smooth_passes=3,
    )
    np.savez_compressed(
        data_path,
        coords=coords.astype(np.float32),
        num_vertices=num_vertices.astype(np.int32),
        score=score.astype(np.float32),
    )

    train_config_path = tmp_path / "train_guidance.yaml"
    with open(train_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "experiment_name": "guidance-study-base",
                "data": {
                    "path": str(data_path),
                    "batch_size": 6,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "classifier": {
                    "type": "gat",
                    "hidden_dim": 16,
                    "time_emb_dim": 8,
                    "num_layers": 2,
                    "pooling": "avg",
                    "timestep_conditioning": True,
                },
                "diffusion": {
                    "n_steps": 16,
                    "beta_start": 1e-4,
                    "beta_end": 1e-2,
                },
                "labels": {
                    "type": "score_regression",
                },
                "training": {
                    "epochs": 1,
                    "lr": 1e-3,
                    "log_every": 1,
                    "save_every": 0,
                    "save_dir": str(tmp_path / "models"),
                    "checkpoint_name": "regressor_final.pt",
                },
            },
            f,
            sort_keys=False,
        )

    study_config_path = tmp_path / "guidance_study.yaml"
    with open(study_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "guidance-training-study",
                    "root_dir": str(tmp_path / "studies"),
                },
                "cases": [
                    {
                        "name": "avg-with-t",
                        "kind": "train_guidance_model",
                        "config": str(train_config_path),
                        "overrides": {
                            "experiment_name": "avg-with-t",
                            "classifier.pooling": "avg",
                            "classifier.timestep_conditioning": True,
                        },
                        "tags": {
                            "pooling": "avg",
                            "timestep_conditioning": True,
                        },
                    },
                    {
                        "name": "max-no-t",
                        "kind": "train_guidance_model",
                        "config": str(train_config_path),
                        "overrides": {
                            "experiment_name": "max-no-t",
                            "classifier.pooling": "max",
                            "classifier.timestep_conditioning": False,
                        },
                        "tags": {
                            "pooling": "max",
                            "timestep_conditioning": False,
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
    assert Path(report_payload["guidance_summary_csv_path"]).exists()
    assert Path(report_payload["guidance_summary_json_path"]).exists()
    assert Path(report_payload["guidance_comparison_plot_path"]).exists()
    assert "guidance_training_comparison" in report_payload["study_figure_paths"]


def test_run_study_writes_distribution_guidance_sweep(tmp_path):
    reference_coords, _, _ = batch(n=4, num=24, seed=9, radial_sigma=0.10, angle_sigma=0.05, smooth_passes=4)
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

    sample_config_path = tmp_path / "sample_guidance_compare.yaml"
    with open(sample_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment_name": "distribution-guidance-study",
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

    study_config_path = tmp_path / "study_distribution_guidance.yaml"
    with open(study_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "distribution-guidance-study",
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
                        "name": "sample-mlp-unguided",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "architecture_baseline",
                            "architecture": "mlp",
                            "guidance_schedule": "unguided",
                            "guidance_scale": 0.0,
                            "guidance_baseline": True,
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_mlp_unguided.npz"),
                        },
                    },
                    {
                        "name": "sample-gat-unguided",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "architecture_baseline",
                            "architecture": "gat",
                            "guidance_schedule": "unguided",
                            "guidance_scale": 0.0,
                            "guidance_baseline": True,
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_gat_unguided.npz"),
                        },
                    },
                    {
                        "name": "sample-mlp-guided",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "distribution_guidance",
                            "architecture": "mlp",
                            "guidance_schedule": "late",
                            "guidance_scale": 2.0,
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_mlp_guided.npz"),
                            "sampling.guidance": {
                                "enabled": True,
                                "components": [
                                    {
                                        "kind": "regularity",
                                        "scale": 2.0,
                                        "schedule": "late",
                                    }
                                ],
                            },
                        },
                    },
                    {
                        "name": "sample-gat-guided",
                        "kind": "sample_diffusion",
                        "config": str(sample_config_path),
                        "tags": {
                            "analysis_group": "distribution_guidance",
                            "architecture": "gat",
                            "guidance_schedule": "late",
                            "guidance_scale": 2.0,
                        },
                        "overrides": {
                            "model.checkpoint": str(checkpoint_path),
                            "sampling.out_path": str(tmp_path / "sample_gat_guided.npz"),
                            "sampling.guidance": {
                                "enabled": True,
                                "components": [
                                    {
                                        "kind": "regularity",
                                        "scale": 2.0,
                                        "schedule": "late",
                                    }
                                ],
                            },
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
    assert Path(report_payload["study_figure_paths"]["distribution_guidance_sweep"]).exists()


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
