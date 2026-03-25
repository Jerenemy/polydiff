"""Run config-driven Polydiff studies and collect summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd
import torch

from ..data.diagnostics import (
    DEFAULT_SCORE_THRESHOLDS,
    compare_polygon_metric_tables,
    compare_polygon_summaries,
    json_ready,
    metric_threshold_rates,
    outlier_failure_mode_summary,
    outlier_polygon_indices,
    polygon_metric_table,
    representative_polygon_indices,
    summarize_polygon_dataset,
)
from ..data.polygon_dataset import load_polygon_dataset
from ..runs import slugify
from ..sampling.sample import SampleCliOverrides, rewrite_sample_archive, sample_from_loaded_config
from ..sampling.runtime import (
    default_diagnostics_out_path,
    default_metrics_out_path,
    resolve_diagnostics_options,
    resolve_reference_summary,
    resolve_restoration_options,
)
from ..training.train import train_from_loaded_config
from ..training.train_guidance_model import train_guidance_model_from_loaded_config
from ..utils.runtime import load_yaml_config, resolve_project_path
from .plots import (
    save_failure_mode_rate_figure,
    save_metric_sweep_figure,
    save_multi_case_score_distribution_figure,
    save_pca_projection_figure,
    save_polygon_gallery,
    save_score_distribution_figure,
)
from .runtime import (
    StudyCase,
    StudyParallelOptions,
    StudyPaths,
    StudySpec,
    apply_dotted_overrides,
    case_result_to_dict,
    create_study_paths,
    flatten_mapping,
    load_study_spec,
    resolve_case_placeholders,
    write_study_metadata,
    write_yaml_config,
)

_PLACEHOLDER_RE = re.compile(r"^\{\{([^{}.]+)\.([^{}.]+)\}\}$")


def _bbox_overlap_area(a: Bbox, b: Bbox) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return float(width * height)


def _axes_overflow_penalty(bbox: Bbox, axes_bbox: Bbox) -> float:
    overflow_x = max(axes_bbox.x0 - bbox.x0, 0.0) + max(bbox.x1 - axes_bbox.x1, 0.0)
    overflow_y = max(axes_bbox.y0 - bbox.y0, 0.0) + max(bbox.y1 - axes_bbox.y1, 0.0)
    return float(overflow_x + overflow_y)


def _tradeoff_label_candidates() -> list[tuple[int, int, str, str]]:
    candidates: list[tuple[int, int, str, str]] = []
    for dx, ha in ((8, "left"), (-8, "right")):
        for dy, va in ((8, "bottom"), (18, "bottom"), (28, "bottom"), (-8, "top"), (-18, "top"), (-28, "top")):
            candidates.append((dx, dy, ha, va))
    candidates.extend(
        [
            (0, 14, "center", "bottom"),
            (0, -14, "center", "top"),
            (18, 0, "left", "center"),
            (-18, 0, "right", "center"),
        ]
    )
    return candidates


def _add_tradeoff_labels(
    ax,
    frame: pd.DataFrame,
    *,
    x_key: str = "generated_summary.score_mean",
    y_key: str = "distribution_distances.distribution_shift_mean_normalized_w1",
) -> list[Any]:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer).expanded(0.98, 0.96)
    placed_boxes: list[Bbox] = []
    annotations: list[Any] = []

    ordered = frame.sort_values([x_key, y_key, "case_name"], ascending=[True, False, True])
    candidates = _tradeoff_label_candidates()
    for _, row in ordered.iterrows():
        x = float(row[x_key])
        y = float(row[y_key])
        label = str(row["case_name"])
        point_x, point_y = ax.transData.transform((x, y))

        best_annotation = None
        best_bbox = None
        best_score = None
        for dx, dy, ha, va in candidates:
            annotation = ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.55",
                    "lw": 0.7,
                    "alpha": 0.55,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                annotation_clip=False,
            )
            fig.canvas.draw()
            bbox = annotation.get_window_extent(renderer).expanded(1.02, 1.10)
            overlap_penalty = sum(_bbox_overlap_area(bbox, other) for other in placed_boxes)
            overflow_penalty = _axes_overflow_penalty(bbox, axes_bbox)
            point_penalty = 1_000_000.0 if bbox.contains(point_x, point_y) else 0.0
            distance_penalty = 0.25 * (abs(dx) + abs(dy))
            score = (10_000.0 * overlap_penalty) + (1_000.0 * overflow_penalty) + point_penalty + distance_penalty

            if best_score is None or score < best_score:
                if best_annotation is not None:
                    best_annotation.remove()
                best_annotation = annotation
                best_bbox = bbox
                best_score = score
                if overlap_penalty == 0.0 and overflow_penalty == 0.0 and point_penalty == 0.0:
                    break
            else:
                annotation.remove()

        if best_annotation is not None and best_bbox is not None:
            placed_boxes.append(best_bbox)
            annotations.append(best_annotation)
    return annotations


def _case_placeholder_dependencies(value: Any) -> set[str]:
    dependencies: set[str] = set()
    if isinstance(value, dict):
        for inner in value.values():
            dependencies.update(_case_placeholder_dependencies(inner))
        return dependencies
    if isinstance(value, list):
        for inner in value:
            dependencies.update(_case_placeholder_dependencies(inner))
        return dependencies
    if not isinstance(value, str):
        return dependencies

    match = _PLACEHOLDER_RE.match(value.strip())
    if match is not None:
        dependencies.add(match.group(1))
    return dependencies


def _study_case_dependencies(spec: StudySpec) -> dict[str, set[str]]:
    known_case_names = {case.name for case in spec.cases}
    dependencies: dict[str, set[str]] = {}
    for case in spec.cases:
        case_dependencies = _case_placeholder_dependencies(case.overrides)
        unknown = case_dependencies - known_case_names
        if unknown:
            raise KeyError(
                f"Study case {case.name!r} references unknown placeholder case(s): {sorted(unknown)}"
            )
        if case.name in case_dependencies:
            raise ValueError(f"Study case {case.name!r} cannot depend on itself")
        dependencies[case.name] = case_dependencies
    return dependencies


def _parallel_execution_enabled(parallel: StudyParallelOptions) -> tuple[bool, str | None]:
    if not parallel.enabled:
        return False, None
    if parallel.max_workers < 2:
        return False, "study.parallel.max_workers < 2"
    if parallel.require_cuda and not torch.cuda.is_available():
        return False, "CUDA is not available"
    return True, None


def _resolve_case_execution(
    *,
    case_index: int,
    case: StudyCase,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    base_cfg = load_yaml_config(case.config_path)
    resolved_overrides = resolve_case_placeholders(case.overrides, case_results=case_results)
    resolved_cfg = apply_dotted_overrides(base_cfg, resolved_overrides)
    if "experiment_name" not in case.overrides:
        resolved_cfg["experiment_name"] = f"{spec.name}-{case.name}"
    resolved_config_path = write_yaml_config(
        paths_obj.configs_dir / f"{case_index:02d}__{slugify(case.name)}.yaml",
        resolved_cfg,
    )
    return resolved_cfg, resolved_config_path


def _case_result_temp_path(*, paths_obj: StudyPaths, case_index: int, case_name: str) -> Path:
    return paths_obj.reports_dir / "tmp" / f"{case_index:02d}__{slugify(case_name)}.result.json"


def _case_log_path(*, paths_obj: StudyPaths, case_index: int, case_name: str) -> Path:
    return paths_obj.reports_dir / "logs" / f"{case_index:02d}__{slugify(case_name)}.log"


def _run_case_entrypoint(*, kind: str, name: str, config_path: Path, result_path: Path) -> Path:
    resolved_config = load_yaml_config(config_path)
    case = StudyCase(name=name, kind=kind, config_path=config_path, overrides={}, tags={})
    result = _run_case(case, resolved_config=resolved_config, resolved_config_path=config_path)
    return _write_json(result_path, result)


def _launch_case_process(
    *,
    case: StudyCase,
    case_index: int,
    resolved_config_path: Path,
    paths_obj: StudyPaths,
) -> dict[str, Any]:
    result_path = _case_result_temp_path(paths_obj=paths_obj, case_index=case_index, case_name=case.name)
    log_path = _case_log_path(paths_obj=paths_obj, case_index=case_index, case_name=case.name)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "polydiff.studies.run",
            "--run-case-kind",
            case.kind,
            "--run-case-name",
            case.name,
            "--run-case-config",
            str(resolved_config_path),
            "--run-case-result",
            str(result_path),
        ],
        cwd=str(resolve_project_path(".")),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "case_index": case_index,
        "case": case,
        "result_path": result_path,
        "log_path": log_path,
        "log_file": log_file,
        "process": process,
    }


def _terminate_running_processes(running_processes: dict[str, dict[str, Any]]) -> None:
    for entry in running_processes.values():
        process = entry["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5.0
    for entry in running_processes.values():
        process = entry["process"]
        if process.poll() is not None:
            continue
        timeout = max(deadline - time.time(), 0.0)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
    for entry in running_processes.values():
        log_file = entry.get("log_file")
        if log_file is not None and not log_file.closed:
            log_file.close()


def _run_case(case: StudyCase, *, resolved_config: dict[str, Any], resolved_config_path: Path) -> dict[str, Any]:
    if case.kind == "train_diffusion":
        result = train_from_loaded_config(resolved_config, config_path=resolved_config_path)
    elif case.kind == "train_guidance_model":
        result = train_guidance_model_from_loaded_config(resolved_config, config_path=resolved_config_path)
    elif case.kind == "sample_diffusion":
        result = sample_from_loaded_config(
            resolved_config,
            config_path=resolved_config_path,
            cli_overrides=SampleCliOverrides(),
        )
    else:
        raise ValueError(f"Unsupported study case kind {case.kind!r}")
    return case_result_to_dict(result, kind=case.kind, name=case.name, config_path=resolved_config_path)


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def _sampling_output_pose_normalized(sampling_cfg: dict[str, Any]) -> bool:
    canonicalize_output = sampling_cfg.get("canonicalize_output")
    restoration_cfg = sampling_cfg.get("restoration")
    restoration_enabled = isinstance(restoration_cfg, dict) and bool(restoration_cfg.get("enabled", False))
    return (not restoration_enabled) if canonicalize_output is None else bool(canonicalize_output)


def _refresh_sample_case_artifacts(case_result: dict[str, Any]) -> dict[str, Any]:
    samples_out_path = resolve_project_path(case_result["samples_out_path"])
    if not samples_out_path.exists():
        return case_result

    resolved_config_path_value = case_result.get("resolved_config_path") or case_result.get("config_path")
    if resolved_config_path_value is None:
        raise KeyError(f"sample case {case_result.get('case_name', '<unknown>')} is missing resolved_config_path")
    resolved_config_path = resolve_project_path(resolved_config_path_value)
    cfg = load_yaml_config(resolved_config_path)
    sampling_cfg = cfg.get("sampling") or {}
    if not isinstance(sampling_cfg, dict):
        raise ValueError(f"sampling config at {resolved_config_path} must be a mapping")

    canonicalize_output = _sampling_output_pose_normalized(sampling_cfg)
    coords, num_vertices = rewrite_sample_archive(
        samples_out_path,
        canonicalize_output=canonicalize_output,
    )

    metrics_path = Path(case_result.get("metrics_path") or default_metrics_out_path(samples_out_path))
    diagnostics_path = Path(case_result.get("diagnostics_path") or default_diagnostics_out_path(samples_out_path))
    existing_diagnostics = _load_json(diagnostics_path) or {}

    generated_table = polygon_metric_table(coords, num_vertices=num_vertices)
    generated_summary = summarize_polygon_dataset(coords, num_vertices=num_vertices)
    generated_table.to_csv(metrics_path, index=False)

    reference_summary = existing_diagnostics.get("reference_summary")
    reference_path_value = existing_diagnostics.get("reference_data_path")
    reference_source = existing_diagnostics.get("reference_summary_source")
    if reference_summary is None and reference_path_value is None:
        checkpoint_path = resolve_project_path(case_result["checkpoint_path"])
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        diagnostics_options = resolve_diagnostics_options(sampling_cfg)
        reference_summary, reference_path, reference_source = resolve_reference_summary(checkpoint, diagnostics_options)
        reference_path_value = None if reference_path is None else str(reference_path)

    reference_path = None if reference_path_value is None else resolve_project_path(reference_path_value)
    distribution_distances = existing_diagnostics.get("distribution_distances")
    if reference_path is not None and reference_path.exists():
        reference_table = polygon_metric_table(load_polygon_dataset(reference_path))
        distribution_distances = compare_polygon_metric_tables(reference_table, generated_table)
    delta_vs_reference = (
        None if reference_summary is None else compare_polygon_summaries(reference_summary, generated_summary)
    )
    score_thresholds = metric_threshold_rates(
        generated_table["score"].to_numpy(dtype=float, copy=False),
        metric_name="score",
        thresholds=DEFAULT_SCORE_THRESHOLDS,
    )

    payload = dict(existing_diagnostics)
    payload.update(
        {
            "config_path": str(resolved_config_path),
            "checkpoint_path": case_result["checkpoint_path"],
            "run_name": case_result.get("run_name"),
            "sample_run_name": case_result.get("sample_run_name"),
            "samples_path": str(samples_out_path),
            "metrics_path": str(metrics_path),
            "sampling_n_steps": int(sampling_cfg.get("n_steps", existing_diagnostics.get("sampling_n_steps", 0))),
            "output_pose_normalized": bool(canonicalize_output),
            "generated_summary": generated_summary,
            "score_threshold_rates": score_thresholds,
            "reference_summary": reference_summary,
            "reference_data_path": reference_path_value,
            "reference_summary_source": reference_source,
            "delta_vs_reference": delta_vs_reference,
            "distribution_distances": distribution_distances,
        }
    )
    restoration = resolve_restoration_options(sampling_cfg)
    if restoration is not None:
        payload["restoration_config"] = restoration.to_dict()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")

    refreshed = dict(case_result)
    refreshed["samples_out_path"] = str(samples_out_path)
    refreshed["metrics_path"] = str(metrics_path)
    refreshed["diagnostics_path"] = str(diagnostics_path)
    return refreshed


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _write_case_results(path: Path, case_results: dict[str, dict[str, Any]]) -> Path:
    return _write_json(path, case_results)


def _write_case_report(
    *,
    paths_obj: StudyPaths,
    case: StudyCase,
    case_name: str,
    case_result: dict[str, Any],
    figure_paths: dict[str, str] | None = None,
    summary_row: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "case_name": case_name,
        "case_kind": case_result.get("case_kind"),
        "case_tags": case.tags,
        "case_result": case_result,
        "figure_paths": figure_paths or {},
        "summary_row": summary_row,
    }
    return _write_json(paths_obj.reports_dir / "cases" / f"{slugify(case_name)}.json", payload)


def _select_reference_path(spec: StudySpec, diagnostics_payload: dict[str, Any] | None) -> Path | None:
    if spec.summary.reference_data_path is not None:
        return spec.summary.reference_data_path
    if diagnostics_payload is None:
        return None
    reference_path = diagnostics_payload.get("reference_data_path")
    return None if reference_path is None else resolve_project_path(reference_path)


def _load_case_comparison_artifacts(
    *,
    spec: StudySpec,
    case_result: dict[str, Any],
    diagnostics_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    sample_path = Path(case_result["samples_out_path"])
    if not sample_path.exists():
        return None

    generated_dataset = load_polygon_dataset(sample_path)
    generated_metrics_path = case_result.get("metrics_path")
    if generated_metrics_path is None or not Path(generated_metrics_path).exists():
        generated_table = polygon_metric_table(generated_dataset)
    else:
        generated_table = pd.read_csv(generated_metrics_path)

    reference_path = _select_reference_path(spec, diagnostics_payload)
    if reference_path is None or not reference_path.exists():
        return {
            "sample_path": sample_path,
            "generated_dataset": generated_dataset,
            "generated_table": generated_table,
            "reference_path": None,
            "reference_dataset": None,
            "reference_table": None,
        }

    reference_dataset = load_polygon_dataset(reference_path)
    reference_table = polygon_metric_table(reference_dataset)
    return {
        "sample_path": sample_path,
        "generated_dataset": generated_dataset,
        "generated_table": generated_table,
        "reference_path": reference_path,
        "reference_dataset": reference_dataset,
        "reference_table": reference_table,
    }


def _generate_case_figures(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case_name: str,
    case_result: dict[str, Any],
    diagnostics_payload: dict[str, Any] | None,
    comparison_artifacts: dict[str, Any] | None,
) -> dict[str, str]:
    if comparison_artifacts is None:
        return {}

    generated_dataset = comparison_artifacts["generated_dataset"]
    generated_table = comparison_artifacts["generated_table"]
    reference_dataset = comparison_artifacts["reference_dataset"]
    reference_table = comparison_artifacts["reference_table"]
    if reference_dataset is None or reference_table is None:
        return {}

    case_figure_dir = paths_obj.figures_dir / slugify(case_name)
    case_figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, str] = {}
    score_dist_path = save_score_distribution_figure(
        reference_table,
        generated_table,
        case_figure_dir / "score_distribution.png",
    )
    figure_paths["score_distribution"] = str(score_dist_path)

    pca_path = save_pca_projection_figure(
        reference_dataset,
        generated_dataset,
        case_figure_dir / "pca_projection.png",
        max_points=spec.summary.max_projection_points,
    )
    if pca_path is not None:
        figure_paths["pca_projection"] = str(pca_path)

    representative_indices = None if diagnostics_payload is None else diagnostics_payload.get("representative_polygon_indices")
    outlier_indices = None if diagnostics_payload is None else diagnostics_payload.get("outlier_polygon_indices")
    if representative_indices is None:
        representative_indices = representative_polygon_indices(
            reference_dataset,
            generated_dataset,
            count=spec.summary.representative_count,
        ).tolist()
    if outlier_indices is None:
        outlier_indices = outlier_polygon_indices(
            reference_dataset,
            generated_dataset,
            count=spec.summary.outlier_count,
        ).tolist()

    representative_gallery = save_polygon_gallery(
        generated_dataset,
        representative_indices[: spec.summary.representative_count],
        case_figure_dir / "representative_gallery.png",
        title=f"{case_name}: representative samples",
    )
    if representative_gallery is not None:
        figure_paths["representative_gallery"] = str(representative_gallery)

    outlier_gallery = save_polygon_gallery(
        generated_dataset,
        outlier_indices[: spec.summary.outlier_count],
        case_figure_dir / "outlier_gallery.png",
        title=f"{case_name}: outliers",
    )
    if outlier_gallery is not None:
        figure_paths["outlier_gallery"] = str(outlier_gallery)
    return figure_paths


def _build_sample_case_summary_entry(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case: StudyCase,
    case_name: str,
    case_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    diagnostics_payload = _load_json(case_result.get("diagnostics_path"))
    row = {
        "case_name": case_name,
        "case_kind": case_result.get("case_kind"),
        "resolved_config_path": case_result.get("resolved_config_path"),
        "samples_out_path": case_result.get("samples_out_path"),
        "metrics_path": case_result.get("metrics_path"),
        "diagnostics_path": case_result.get("diagnostics_path"),
    }
    row.update(flatten_mapping(case.tags, prefix="case_tags"))
    if diagnostics_payload is not None:
        row.update(flatten_mapping(diagnostics_payload))
    comparison_artifacts = _load_case_comparison_artifacts(
        spec=spec,
        case_result=case_result,
        diagnostics_payload=diagnostics_payload,
    )
    if comparison_artifacts is not None:
        generated_table = comparison_artifacts["generated_table"]
        row.setdefault("generated_summary.score_p99", float(generated_table["score"].quantile(0.99)))
        for key, value in metric_threshold_rates(
            generated_table["score"].to_numpy(dtype=float, copy=False),
            metric_name="score",
            thresholds=DEFAULT_SCORE_THRESHOLDS,
        ).items():
            row.setdefault(f"score_threshold_rates.{key}", value)
        reference_table = comparison_artifacts["reference_table"]
        if reference_table is not None:
            outlier_indices = None if diagnostics_payload is None else diagnostics_payload.get("outlier_polygon_indices")
            if outlier_indices is None:
                outlier_indices = outlier_polygon_indices(
                    comparison_artifacts["reference_dataset"],
                    comparison_artifacts["generated_dataset"],
                    count=spec.summary.outlier_count,
                ).tolist()
            failure_summary = outlier_failure_mode_summary(
                reference_table,
                generated_table,
                outlier_indices=outlier_indices[: spec.summary.outlier_count],
            )
            row.update(flatten_mapping({"outlier_failure_modes": failure_summary}))
    figure_paths = _generate_case_figures(
        spec=spec,
        paths_obj=paths_obj,
        case_name=case_name,
        case_result=case_result,
        diagnostics_payload=diagnostics_payload,
        comparison_artifacts=comparison_artifacts,
    )
    return row, figure_paths


def _write_tradeoff_plot(summary_df: pd.DataFrame, *, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None

    x_key = "generated_summary.score_mean"
    if "distribution_distances.shape_distribution_shift_mean_normalized_w1" in summary_df.columns:
        y_key = "distribution_distances.shape_distribution_shift_mean_normalized_w1"
    elif "distribution_distances.distribution_shift_mean_normalized_w1" in summary_df.columns:
        y_key = "distribution_distances.distribution_shift_mean_normalized_w1"
    else:
        y_key = "generated_summary.self_intersection_rate"
    if x_key not in summary_df.columns or y_key not in summary_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(summary_df[x_key], summary_df[y_key], s=46, color="tab:blue")
    ax.margins(x=0.14, y=0.10)
    _add_tradeoff_labels(ax, summary_df, x_key=x_key, y_key=y_key)
    ax.set_xlabel("generated score mean")
    y_label = {
        "distribution_distances.shape_distribution_shift_mean_normalized_w1": "shape distribution shift (mean normalized W1)",
        "distribution_distances.distribution_shift_mean_normalized_w1": "distribution shift (mean normalized W1)",
        "generated_summary.self_intersection_rate": "self-intersection rate",
    }.get(y_key, y_key.split(".")[-1].replace("_", " "))
    ax.set_ylabel(y_label)
    ax.set_title("Study Tradeoff Summary")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _frame_for_analysis_group(
    summary_df: pd.DataFrame,
    *,
    analysis_group: str,
    include_guidance_baseline: bool = False,
) -> pd.DataFrame:
    if "case_tags.analysis_group" not in summary_df.columns:
        return summary_df.iloc[0:0].copy()
    mask = summary_df["case_tags.analysis_group"].astype(str) == analysis_group
    if include_guidance_baseline and "case_tags.guidance_baseline" in summary_df.columns:
        baseline_mask = summary_df["case_tags.guidance_baseline"].fillna(False).astype(bool)
        mask = mask | baseline_mask
    return summary_df.loc[mask].copy()


def _load_case_score_tables(frame: pd.DataFrame, *, label_key: str) -> list[tuple[str, pd.DataFrame]]:
    case_tables: list[tuple[str, pd.DataFrame]] = []
    if "metrics_path" not in frame.columns or label_key not in frame.columns:
        return case_tables
    for _, row in frame.iterrows():
        metrics_path = row.get("metrics_path")
        if metrics_path is None:
            continue
        resolved = Path(metrics_path)
        if not resolved.exists():
            continue
        case_tables.append((str(row[label_key]), pd.read_csv(resolved)))
    return case_tables


def _ordered_noise_labels(frame: pd.DataFrame) -> list[str]:
    if "case_tags.dataset_noise" not in frame.columns:
        return []
    if "case_tags.dataset_noise_rank" not in frame.columns:
        return [str(value) for value in frame["case_tags.dataset_noise"].dropna().drop_duplicates().tolist()]
    order_frame = (
        frame.loc[:, ["case_tags.dataset_noise", "case_tags.dataset_noise_rank"]]
        .drop_duplicates()
        .sort_values(["case_tags.dataset_noise_rank", "case_tags.dataset_noise"], kind="stable")
    )
    return [str(value) for value in order_frame["case_tags.dataset_noise"].tolist()]


def _group_key_for_frame(frame: pd.DataFrame) -> tuple[str | None, list[str] | None]:
    if "case_tags.dataset_noise" in frame.columns and frame["case_tags.dataset_noise"].nunique(dropna=True) > 1:
        return "case_tags.dataset_noise", _ordered_noise_labels(frame)
    if "case_tags.architecture" in frame.columns and frame["case_tags.architecture"].nunique(dropna=True) > 1:
        order = [arch for arch in ("mlp", "gat", "gcn") if arch in frame["case_tags.architecture"].astype(str).unique()]
        return "case_tags.architecture", order or None
    return None, None


def _write_analysis_figures(summary_df: pd.DataFrame, *, paths_obj: StudyPaths) -> dict[str, str]:
    if summary_df.empty:
        return {}

    figure_paths: dict[str, str] = {}
    architecture_order = [arch for arch in ("mlp", "gat", "gcn") if arch in summary_df.get("case_tags.architecture", pd.Series(dtype=str)).astype(str).unique()]

    architecture_frame = summary_df.copy()
    if "case_tags.analysis_group" in architecture_frame.columns:
        architecture_frame = architecture_frame[
            architecture_frame["case_tags.analysis_group"].astype(str).isin({"architecture_baseline", "architecture_noise"})
        ]
    if "case_tags.guidance_schedule" in architecture_frame.columns:
        architecture_frame = architecture_frame[architecture_frame["case_tags.guidance_schedule"].astype(str) == "unguided"]
    elif "case_tags.guidance_enabled" in architecture_frame.columns:
        architecture_frame = architecture_frame[~architecture_frame["case_tags.guidance_enabled"].fillna(False).astype(bool)]
    if not architecture_frame.empty and "case_tags.architecture" in architecture_frame.columns:
        if "case_tags.dataset_noise" in architecture_frame.columns:
            group_values = [
                (noise_label, architecture_frame[architecture_frame["case_tags.dataset_noise"].astype(str) == noise_label].copy())
                for noise_label in _ordered_noise_labels(architecture_frame)
            ]
        else:
            group_values = [("architecture", architecture_frame)]
        for noise_label, group_frame in group_values:
            if group_frame.empty or group_frame["case_tags.architecture"].nunique(dropna=True) < 2:
                continue
            slug = slugify(str(noise_label))
            title_suffix = "" if str(noise_label) == "architecture" else f" ({noise_label})"
            overlay_path = save_multi_case_score_distribution_figure(
                _load_case_score_tables(group_frame, label_key="case_tags.architecture"),
                paths_obj.figures_dir / f"architecture_score_overlay__{slug}.png",
                title=f"Architecture Score Distribution{title_suffix}",
            )
            if overlay_path is not None:
                figure_paths[f"architecture_score_overlay__{slug}"] = str(overlay_path)
            metric_panel_path = save_metric_sweep_figure(
                group_frame,
                paths_obj.figures_dir / f"architecture_metric_panel__{slug}.png",
                x_key="case_tags.architecture",
                x_label="architecture",
                x_order=architecture_order or None,
                title=f"Architecture Comparison{title_suffix}",
            )
            if metric_panel_path is not None:
                figure_paths[f"architecture_metric_panel__{slug}"] = str(metric_panel_path)

    schedule_frame = _frame_for_analysis_group(
        summary_df,
        analysis_group="guidance_schedule",
        include_guidance_baseline=True,
    )
    if not schedule_frame.empty and "case_tags.guidance_schedule" in schedule_frame.columns:
        group_key, group_order = _group_key_for_frame(schedule_frame)
        schedule_path = save_metric_sweep_figure(
            schedule_frame,
            paths_obj.figures_dir / "guidance_schedule_sweep.png",
            x_key="case_tags.guidance_schedule",
            x_label="guidance schedule",
            x_order=["unguided", "all", "early", "mid", "late", "linear_ramp", "quadratic_ramp"],
            group_key=group_key,
            group_order=group_order,
            title="Guidance Timing Sweep",
        )
        if schedule_path is not None:
            figure_paths["guidance_schedule_sweep"] = str(schedule_path)

    strength_frame = _frame_for_analysis_group(
        summary_df,
        analysis_group="guidance_strength",
        include_guidance_baseline=True,
    )
    if not strength_frame.empty and "case_tags.guidance_scale" in strength_frame.columns:
        group_key, group_order = _group_key_for_frame(strength_frame)
        strength_path = save_metric_sweep_figure(
            strength_frame,
            paths_obj.figures_dir / "guidance_strength_sweep.png",
            x_key="case_tags.guidance_scale",
            x_label="guidance scale",
            x_scale="symlog",
            group_key=group_key,
            group_order=group_order,
            title="Guidance Strength Sweep",
        )
        if strength_path is not None:
            figure_paths["guidance_strength_sweep"] = str(strength_path)

    noise_frame = _frame_for_analysis_group(summary_df, analysis_group="architecture_noise")
    if (
        not noise_frame.empty
        and "case_tags.architecture" in noise_frame.columns
        and "case_tags.dataset_noise" in noise_frame.columns
        and noise_frame["case_tags.architecture"].nunique(dropna=True) > 1
        and noise_frame["case_tags.dataset_noise"].nunique(dropna=True) > 1
    ):
        noise_path = save_metric_sweep_figure(
            noise_frame,
            paths_obj.figures_dir / "architecture_noise_sweep.png",
            x_key="case_tags.dataset_noise",
            x_label="training dataset noise",
            x_order=_ordered_noise_labels(noise_frame),
            group_key="case_tags.architecture",
            group_order=architecture_order or None,
            title="Architecture vs Training Dataset Noise",
        )
        if noise_path is not None:
            figure_paths["architecture_noise_sweep"] = str(noise_path)

    failure_modes_path = save_failure_mode_rate_figure(
        summary_df,
        paths_obj.figures_dir / "outlier_failure_modes.png",
    )
    if failure_modes_path is not None:
        figure_paths["outlier_failure_modes"] = str(failure_modes_path)
    return figure_paths


def _write_interpretation_guide(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    summary_df: pd.DataFrame,
    study_figure_paths: dict[str, str],
) -> Path:
    lines = [
        f"# How To Read {paths_obj.study_name}",
        "",
        "This note is written automatically by the study runner. Use it as a quick reading guide for the figures, then write the thesis interpretation separately.",
        "",
        "## Core Metrics",
        "",
        "- `generated_summary.score_mean`: typical sample regularity. Higher is better.",
        "- `generated_summary.score_p99`: best-tail quality. Higher means the method produces stronger rare samples.",
        "- `score_threshold_rates.score_ge_0p7_rate`: share of samples clearing a high-quality regularity threshold.",
        "- `distribution_distances.shape_distribution_shift_mean_normalized_w1`: shape-only manifold drift relative to the training/reference dataset. Lower is better.",
        "- `generated_summary.self_intersection_rate`: explicit geometric invalidity rate. Lower is better.",
        "",
        "## Reading The Figures",
        "",
        "- `study_tradeoff.png`: lower-right is the preferred region because it combines higher score with lower shape drift.",
    ]
    if any(key.startswith("architecture_score_overlay__") for key in study_figure_paths):
        lines.append("- `architecture_score_overlay__*.png`: compare the full score distributions, not just the mean. This is where tail advantages or mode shifts become visible.")
    if any(key.startswith("architecture_metric_panel__") for key in study_figure_paths):
        lines.append("- `architecture_metric_panel__*.png`: compare typical quality, tail quality, high-threshold success, and fidelity side by side for the architecture cases.")
    if "guidance_schedule_sweep" in study_figure_paths:
        lines.append("- `guidance_schedule_sweep.png`: isolates timing effects at fixed guidance strength. Look for schedules that improve score without sharply increasing shape drift.")
    if "guidance_strength_sweep" in study_figure_paths:
        lines.append("- `guidance_strength_sweep.png`: uses a wide dynamic range to show whether regularity guidance is monotonic, saturating, or destabilizing as scale increases.")
    if "architecture_noise_sweep" in study_figure_paths:
        lines.append("- `architecture_noise_sweep.png`: tests whether the architecture ranking changes as the training polygons become rougher.")
    if "outlier_failure_modes" in study_figure_paths:
        lines.append("- `outlier_failure_modes.png`: counts heuristic failure labels among the stored outlier set. Use this to distinguish high-score methods from low-degeneracy methods.")
    lines.extend(
        [
            "",
            "## Thesis Guardrail",
            "",
            "- Do not assume `mlp` wins by default.",
            "- Treat `mlp > gnn` as a hypothesis to prove or reject using mean score, tail score, high-threshold rate, shape drift, and failure behavior together.",
            "- A tradeoff result is acceptable: for example, `gat` may win on the score tail while `mlp` remains closer to the training manifold.",
            "",
            "## Guidance Guardrail",
            "",
            "- Regularity guidance is an aligned, relatively low-risk objective in this project.",
            "- It is useful for studying controllability under favorable conditions, but it should not be presented as proof that arbitrary sampling-time guidance will behave equally well.",
            "- The strength sweep is meant to locate the plateau and failure regime, not just the first improvement regime.",
        ]
    )
    if not summary_df.empty and "case_name" in summary_df.columns:
        lines.extend(
            [
                "",
                "## Case Tags",
                "",
                "- The study runner uses `cases[*].tags` in the manifest to decide which grouped figures to build.",
                "- For architecture studies, tag `architecture`, `dataset_noise`, and `analysis_group`.",
                "- For guidance studies, tag `guidance_schedule`, `guidance_scale`, `analysis_group`, and set `guidance_baseline: true` on the unguided reference case.",
            ]
        )
    lines.extend(
        [
            "",
            f"Source manifest: `{spec.config_path}`",
            "",
        ]
    )
    guide_path = paths_obj.study_dir / "INTERPRET_RESULTS.md"
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text("\n".join(lines), encoding="utf-8")
    return guide_path


def _write_sample_summary_outputs(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    rows: list[dict[str, Any]],
    figure_paths_by_case: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not rows:
        return {
            "summary_csv_path": None,
            "summary_json_path": None,
            "tradeoff_plot_path": None,
            "study_figure_paths": {},
            "interpretation_guide_path": None,
            "figure_paths_by_case": figure_paths_by_case,
        }

    summary_df = pd.DataFrame(rows)
    summary_csv_path = paths_obj.reports_dir / "sample_case_summary.csv"
    summary_json_path = paths_obj.reports_dir / "sample_case_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    _write_json(summary_json_path, rows)

    tradeoff_path = _write_tradeoff_plot(summary_df, out_path=paths_obj.figures_dir / "study_tradeoff.png")
    study_figure_paths = _write_analysis_figures(summary_df, paths_obj=paths_obj)
    if tradeoff_path is not None:
        study_figure_paths = {"study_tradeoff": str(tradeoff_path), **study_figure_paths}
    interpretation_guide_path = _write_interpretation_guide(
        spec=spec,
        paths_obj=paths_obj,
        summary_df=summary_df,
        study_figure_paths=study_figure_paths,
    )
    return {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "tradeoff_plot_path": None if tradeoff_path is None else str(tradeoff_path),
        "study_figure_paths": study_figure_paths,
        "interpretation_guide_path": str(interpretation_guide_path),
        "figure_paths_by_case": figure_paths_by_case,
    }


def _summarize_sample_cases(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    figure_paths_by_case: dict[str, dict[str, str]] = {}
    for case in spec.cases:
        case_result = case_results.get(case.name)
        if case_result is None or case_result.get("case_kind") != "sample_diffusion":
            continue
        row, figure_paths = _build_sample_case_summary_entry(
            spec=spec,
            paths_obj=paths_obj,
            case=case,
            case_name=case.name,
            case_result=case_result,
        )
        rows.append(row)
        figure_paths_by_case[case.name] = figure_paths
        _write_case_report(
            paths_obj=paths_obj,
            case=case,
            case_name=case.name,
            case_result=case_result,
            figure_paths=figure_paths,
            summary_row=row,
        )

    summary_payload = _write_sample_summary_outputs(
        spec=spec,
        paths_obj=paths_obj,
        rows=rows,
        figure_paths_by_case=figure_paths_by_case,
    )
    return summary_payload, rows, figure_paths_by_case


def _ingest_case_result(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case: StudyCase,
    case_result: dict[str, Any],
    case_results: dict[str, dict[str, Any]],
    case_results_path: Path,
    case_report_paths_by_case: dict[str, str],
    sample_rows: list[dict[str, Any]],
    figure_paths_by_case: dict[str, dict[str, str]],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    case_results[case.name] = case_result
    _write_case_results(case_results_path, case_results)

    case_report_path = _write_case_report(paths_obj=paths_obj, case=case, case_name=case.name, case_result=case_result)
    case_report_paths_by_case[case.name] = str(case_report_path)

    if case_result.get("case_kind") == "sample_diffusion":
        row, figure_paths = _build_sample_case_summary_entry(
            spec=spec,
            paths_obj=paths_obj,
            case=case,
            case_name=case.name,
            case_result=case_result,
        )
        sample_rows.append(row)
        figure_paths_by_case[case.name] = figure_paths
        case_report_path = _write_case_report(
            paths_obj=paths_obj,
            case=case,
            case_name=case.name,
            case_result=case_result,
            figure_paths=figure_paths,
            summary_row=row,
        )
        case_report_paths_by_case[case.name] = str(case_report_path)
        return _write_sample_summary_outputs(
            spec=spec,
            paths_obj=paths_obj,
            rows=sample_rows,
            figure_paths_by_case=figure_paths_by_case,
        )
    return summary_payload


def _write_study_report(
    *,
    paths_obj: StudyPaths,
    spec: StudySpec,
    case_results: dict[str, dict[str, Any]],
    case_results_path: Path | None,
    summary_payload: dict[str, Any],
    case_report_paths_by_case: dict[str, str],
    status: str,
    current_case_name: str | None = None,
    current_case_names: list[str] | None = None,
    error: dict[str, str] | None = None,
) -> Path:
    completed_case_names = [case.name for case in spec.cases if case.name in case_results]
    missing_case_names = [case.name for case in spec.cases if case.name not in case_results]
    running_case_names = [] if current_case_names is None else [str(name) for name in current_case_names]
    if current_case_name is None and running_case_names:
        current_case_name = running_case_names[0]
    report_payload = {
        "study_name": paths_obj.study_name,
        "study_dir": str(paths_obj.study_dir),
        "status": status,
        "current_case_name": current_case_name,
        "current_case_names": running_case_names,
        "completed_cases": len(completed_case_names),
        "total_cases": len(spec.cases),
        "completed_case_names": completed_case_names,
        "missing_case_names": missing_case_names,
        "case_results_path": None if case_results_path is None else str(case_results_path),
        "case_report_paths_by_case": case_report_paths_by_case,
        "parallel": {
            "enabled": spec.parallel.enabled,
            "max_workers": spec.parallel.max_workers,
            "require_cuda": spec.parallel.require_cuda,
        },
        **summary_payload,
    }
    if error is not None:
        report_payload["error"] = error
    return _write_json(paths_obj.reports_dir / "study_report.json", report_payload)


def run_study_from_config(config_path: Path) -> Path:
    spec = load_study_spec(config_path)
    paths_obj = create_study_paths(name=spec.name, root_dir=spec.root_dir)
    write_study_metadata(paths_obj.study_dir / "study_metadata.json", spec=spec, paths_obj=paths_obj)
    dependency_map = _study_case_dependencies(spec)
    parallel_enabled, parallel_disabled_reason = _parallel_execution_enabled(spec.parallel)
    if spec.parallel.enabled and not parallel_enabled and parallel_disabled_reason is not None:
        print(f"[study] parallel execution disabled: {parallel_disabled_reason}; falling back to serial execution")

    case_results: dict[str, dict[str, Any]] = {}
    case_results_path = paths_obj.reports_dir / "case_results.json"
    case_report_paths_by_case: dict[str, str] = {}
    sample_rows: list[dict[str, Any]] = []
    figure_paths_by_case: dict[str, dict[str, str]] = {}
    summary_payload = _write_sample_summary_outputs(
        spec=spec,
        paths_obj=paths_obj,
        rows=sample_rows,
        figure_paths_by_case=figure_paths_by_case,
    )
    final_report_path = _write_study_report(
        paths_obj=paths_obj,
        spec=spec,
        case_results=case_results,
        case_results_path=None,
        summary_payload=summary_payload,
        case_report_paths_by_case=case_report_paths_by_case,
        status="running",
    )

    pending_cases: dict[str, tuple[int, StudyCase]] = {
        case.name: (case_index, case)
        for case_index, case in enumerate(spec.cases, start=1)
    }
    running_processes: dict[str, dict[str, Any]] = {}
    max_parallel_workers = spec.parallel.max_workers if parallel_enabled else 1
    try:
        while pending_cases or running_processes:
            completed_case_names = set(case_results.keys())
            ready_cases = [
                (case_index, case)
                for case_index, case in enumerate(spec.cases, start=1)
                if case.name in pending_cases and dependency_map[case.name].issubset(completed_case_names)
            ]

            if not parallel_enabled:
                if not ready_cases:
                    unresolved = {
                        case.name: sorted(dependency_map[case.name] - completed_case_names)
                        for _, case in pending_cases.values()
                    }
                    raise RuntimeError(f"No runnable study cases remain; unresolved dependencies: {unresolved}")

                case_index, case = ready_cases[0]
                del pending_cases[case.name]
                _write_study_report(
                    paths_obj=paths_obj,
                    spec=spec,
                    case_results=case_results,
                    case_results_path=case_results_path if case_results else None,
                    summary_payload=summary_payload,
                    case_report_paths_by_case=case_report_paths_by_case,
                    status="running",
                    current_case_name=case.name,
                    current_case_names=[case.name],
                )
                resolved_cfg, resolved_config_path = _resolve_case_execution(
                    case_index=case_index,
                    case=case,
                    spec=spec,
                    paths_obj=paths_obj,
                    case_results=case_results,
                )
                print(f"[study] running case {case_index}/{len(spec.cases)} name={case.name} kind={case.kind}")
                case_result = _run_case(
                    case,
                    resolved_config=resolved_cfg,
                    resolved_config_path=resolved_config_path,
                )
                summary_payload = _ingest_case_result(
                    spec=spec,
                    paths_obj=paths_obj,
                    case=case,
                    case_result=case_result,
                    case_results=case_results,
                    case_results_path=case_results_path,
                    case_report_paths_by_case=case_report_paths_by_case,
                    sample_rows=sample_rows,
                    figure_paths_by_case=figure_paths_by_case,
                    summary_payload=summary_payload,
                )
                final_report_path = _write_study_report(
                    paths_obj=paths_obj,
                    spec=spec,
                    case_results=case_results,
                    case_results_path=case_results_path,
                    summary_payload=summary_payload,
                    case_report_paths_by_case=case_report_paths_by_case,
                    status="running",
                )
                continue

            available_slots = max(0, max_parallel_workers - len(running_processes))
            for case_index, case in ready_cases[:available_slots]:
                del pending_cases[case.name]
                _, resolved_config_path = _resolve_case_execution(
                    case_index=case_index,
                    case=case,
                    spec=spec,
                    paths_obj=paths_obj,
                    case_results=case_results,
                )
                print(f"[study] launching case {case_index}/{len(spec.cases)} name={case.name} kind={case.kind}")
                running_processes[case.name] = _launch_case_process(
                    case=case,
                    case_index=case_index,
                    resolved_config_path=resolved_config_path,
                    paths_obj=paths_obj,
                )

            running_case_names = list(running_processes.keys())
            _write_study_report(
                paths_obj=paths_obj,
                spec=spec,
                case_results=case_results,
                case_results_path=case_results_path if case_results else None,
                summary_payload=summary_payload,
                case_report_paths_by_case=case_report_paths_by_case,
                status="running",
                current_case_names=running_case_names,
            )

            if not running_processes:
                unresolved = {
                    case.name: sorted(dependency_map[case.name] - completed_case_names)
                    for _, case in pending_cases.values()
                }
                raise RuntimeError(f"No runnable study cases remain; unresolved dependencies: {unresolved}")

            completed_entries: list[dict[str, Any]] = []
            while not completed_entries:
                for case_name, entry in list(running_processes.items()):
                    return_code = entry["process"].poll()
                    if return_code is None:
                        continue
                    entry["log_file"].close()
                    del running_processes[case_name]
                    entry["return_code"] = int(return_code)
                    completed_entries.append(entry)
                if not completed_entries:
                    time.sleep(0.25)

            for entry in completed_entries:
                case = entry["case"]
                log_path = entry["log_path"]
                if entry["return_code"] != 0:
                    _terminate_running_processes(running_processes)
                    raise RuntimeError(
                        f"Study case {case.name!r} failed with exit code {entry['return_code']}; see {log_path}"
                    )
                case_result = _load_json(entry["result_path"])
                if case_result is None:
                    _terminate_running_processes(running_processes)
                    raise FileNotFoundError(
                        f"Study case {case.name!r} did not write a result file at {entry['result_path']}"
                    )
                case_result["study_case_log_path"] = str(log_path)
                summary_payload = _ingest_case_result(
                    spec=spec,
                    paths_obj=paths_obj,
                    case=case,
                    case_result=case_result,
                    case_results=case_results,
                    case_results_path=case_results_path,
                    case_report_paths_by_case=case_report_paths_by_case,
                    sample_rows=sample_rows,
                    figure_paths_by_case=figure_paths_by_case,
                    summary_payload=summary_payload,
                )
                final_report_path = _write_study_report(
                    paths_obj=paths_obj,
                    spec=spec,
                    case_results=case_results,
                    case_results_path=case_results_path,
                    summary_payload=summary_payload,
                    case_report_paths_by_case=case_report_paths_by_case,
                    status="running",
                    current_case_names=list(running_processes.keys()),
                )
    except Exception as exc:
        _terminate_running_processes(running_processes)
        final_report_path = _write_study_report(
            paths_obj=paths_obj,
            spec=spec,
            case_results=case_results,
            case_results_path=case_results_path if case_results else None,
            summary_payload=summary_payload,
            case_report_paths_by_case=case_report_paths_by_case,
            status="failed",
            current_case_names=list(running_processes.keys()),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise

    final_report_path = _write_study_report(
        paths_obj=paths_obj,
        spec=spec,
        case_results=case_results,
        case_results_path=case_results_path,
        summary_payload=summary_payload,
        case_report_paths_by_case=case_report_paths_by_case,
        status="completed",
    )
    print(f"[study] wrote report {final_report_path}")
    return final_report_path


def refresh_study_outputs(study_dir: Path) -> Path:
    resolved_study_dir = resolve_project_path(study_dir)
    metadata_path = resolved_study_dir / "study_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Study metadata not found at {metadata_path}")

    metadata_payload = _load_json(metadata_path)
    if metadata_payload is None:
        raise ValueError(f"Failed to load study metadata from {metadata_path}")

    source_config_path = metadata_payload.get("source_config_path")
    if source_config_path is None:
        raise KeyError(f"study metadata at {metadata_path} is missing source_config_path")

    spec = load_study_spec(resolve_project_path(source_config_path))
    study_number = metadata_payload.get("study_number")
    if study_number is None:
        study_number = 0
    paths_obj = StudyPaths(
        study_name=str(metadata_payload.get("study_name", resolved_study_dir.name)),
        study_number=int(study_number),
        study_dir=resolved_study_dir,
        configs_dir=resolved_study_dir / "configs",
        reports_dir=resolved_study_dir / "reports",
        figures_dir=resolved_study_dir / "figures",
    )
    for directory in (paths_obj.study_dir, paths_obj.configs_dir, paths_obj.reports_dir, paths_obj.figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_study_metadata(paths_obj.study_dir / "study_metadata.json", spec=spec, paths_obj=paths_obj)

    case_results_path = paths_obj.reports_dir / "case_results.json"
    case_results_payload = _load_json(case_results_path)
    if case_results_payload is None:
        raise FileNotFoundError(f"Case results not found at {case_results_path}")

    case_results = {str(key): dict(value) for key, value in case_results_payload.items()}
    for case in spec.cases:
        case_result = case_results.get(case.name)
        if case_result is None or case_result.get("case_kind") != "sample_diffusion":
            continue
        case_results[case.name] = _refresh_sample_case_artifacts(case_result)
    _write_case_results(case_results_path, case_results)
    summary_payload, rows, figure_paths_by_case = _summarize_sample_cases(
        spec=spec,
        paths_obj=paths_obj,
        case_results=case_results,
    )

    case_report_paths_by_case: dict[str, str] = {}
    for case in spec.cases:
        case_result = case_results.get(case.name)
        if case_result is None:
            continue
        summary_row = next((row for row in rows if row["case_name"] == case.name), None)
        case_report_path = _write_case_report(
            paths_obj=paths_obj,
            case=case,
            case_name=case.name,
            case_result=case_result,
            figure_paths=figure_paths_by_case.get(case.name),
            summary_row=summary_row,
        )
        case_report_paths_by_case[case.name] = str(case_report_path)

    status = "completed" if len(case_results) == len(spec.cases) else "partial"
    report_path = _write_study_report(
        paths_obj=paths_obj,
        spec=spec,
        case_results=case_results,
        case_results_path=case_results_path,
        summary_payload=summary_payload,
        case_report_paths_by_case=case_report_paths_by_case,
        status=status,
    )
    print(f"[study] refreshed report {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="path to study manifest yaml",
    )
    group.add_argument(
        "--refresh-study-dir",
        type=str,
        help="rebuild study reports and figures from an existing study directory",
    )
    group.add_argument(
        "--run-case-config",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-case-kind",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-case-name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-case-result",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.run_case_config is not None:
        if args.run_case_kind is None or args.run_case_name is None or args.run_case_result is None:
            raise ValueError(
                "--run-case-config requires --run-case-kind, --run-case-name, and --run-case-result"
            )
        _run_case_entrypoint(
            kind=str(args.run_case_kind),
            name=str(args.run_case_name),
            config_path=resolve_project_path(args.run_case_config),
            result_path=resolve_project_path(args.run_case_result),
        )
        return
    if args.refresh_study_dir is not None:
        refresh_study_outputs(resolve_project_path(args.refresh_study_dir))
        return
    run_study_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
