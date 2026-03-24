"""Run config-driven Polydiff studies and collect summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import pandas as pd

from ..data.diagnostics import (
    DEFAULT_SCORE_THRESHOLDS,
    compare_polygon_metric_tables,
    compare_polygon_summaries,
    json_ready,
    metric_threshold_rates,
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
from .plots import save_pca_projection_figure, save_polygon_gallery, save_score_distribution_figure
from .runtime import (
    StudyCase,
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
    case_name: str,
    case_result: dict[str, Any],
    figure_paths: dict[str, str] | None = None,
    summary_row: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "case_name": case_name,
        "case_kind": case_result.get("case_kind"),
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


def _generate_case_figures(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case_name: str,
    case_result: dict[str, Any],
    diagnostics_payload: dict[str, Any] | None,
) -> dict[str, str]:
    sample_path = Path(case_result["samples_out_path"])
    if not sample_path.exists():
        return {}

    generated_dataset = load_polygon_dataset(sample_path)
    reference_path = _select_reference_path(spec, diagnostics_payload)
    if reference_path is None or not reference_path.exists():
        return {}
    reference_dataset = load_polygon_dataset(reference_path)

    case_figure_dir = paths_obj.figures_dir / slugify(case_name)
    case_figure_dir.mkdir(parents=True, exist_ok=True)

    generated_metrics_path = case_result.get("metrics_path")
    if generated_metrics_path is None or not Path(generated_metrics_path).exists():
        generated_table = None
    else:
        generated_table = pd.read_csv(generated_metrics_path)

    from ..data.diagnostics import polygon_metric_table

    reference_table = polygon_metric_table(reference_dataset)
    if generated_table is None:
        generated_table = polygon_metric_table(generated_dataset)

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
    if diagnostics_payload is not None:
        row.update(flatten_mapping(diagnostics_payload))
    figure_paths = _generate_case_figures(
        spec=spec,
        paths_obj=paths_obj,
        case_name=case_name,
        case_result=case_result,
        diagnostics_payload=diagnostics_payload,
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


def _write_sample_summary_outputs(
    *,
    paths_obj: StudyPaths,
    rows: list[dict[str, Any]],
    figure_paths_by_case: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if not rows:
        return {
            "summary_csv_path": None,
            "summary_json_path": None,
            "tradeoff_plot_path": None,
            "figure_paths_by_case": figure_paths_by_case,
        }

    summary_df = pd.DataFrame(rows)
    summary_csv_path = paths_obj.reports_dir / "sample_case_summary.csv"
    summary_json_path = paths_obj.reports_dir / "sample_case_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    _write_json(summary_json_path, rows)

    tradeoff_path = _write_tradeoff_plot(summary_df, out_path=paths_obj.figures_dir / "study_tradeoff.png")
    return {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "tradeoff_plot_path": None if tradeoff_path is None else str(tradeoff_path),
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
            case_name=case.name,
            case_result=case_result,
        )
        rows.append(row)
        figure_paths_by_case[case.name] = figure_paths
        _write_case_report(
            paths_obj=paths_obj,
            case_name=case.name,
            case_result=case_result,
            figure_paths=figure_paths,
            summary_row=row,
        )

    summary_payload = _write_sample_summary_outputs(
        paths_obj=paths_obj,
        rows=rows,
        figure_paths_by_case=figure_paths_by_case,
    )
    return summary_payload, rows, figure_paths_by_case


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
    error: dict[str, str] | None = None,
) -> Path:
    completed_case_names = [case.name for case in spec.cases if case.name in case_results]
    missing_case_names = [case.name for case in spec.cases if case.name not in case_results]
    report_payload = {
        "study_name": paths_obj.study_name,
        "study_dir": str(paths_obj.study_dir),
        "status": status,
        "current_case_name": current_case_name,
        "completed_cases": len(completed_case_names),
        "total_cases": len(spec.cases),
        "completed_case_names": completed_case_names,
        "missing_case_names": missing_case_names,
        "case_results_path": None if case_results_path is None else str(case_results_path),
        "case_report_paths_by_case": case_report_paths_by_case,
        **summary_payload,
    }
    if error is not None:
        report_payload["error"] = error
    return _write_json(paths_obj.reports_dir / "study_report.json", report_payload)


def run_study_from_config(config_path: Path) -> Path:
    spec = load_study_spec(config_path)
    paths_obj = create_study_paths(name=spec.name, root_dir=spec.root_dir)
    write_study_metadata(paths_obj.study_dir / "study_metadata.json", spec=spec, paths_obj=paths_obj)

    case_results: dict[str, dict[str, Any]] = {}
    case_results_path = paths_obj.reports_dir / "case_results.json"
    case_report_paths_by_case: dict[str, str] = {}
    sample_rows: list[dict[str, Any]] = []
    figure_paths_by_case: dict[str, dict[str, str]] = {}
    summary_payload = _write_sample_summary_outputs(
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

    current_case_name: str | None = None
    try:
        for case_index, case in enumerate(spec.cases, start=1):
            current_case_name = case.name
            _write_study_report(
                paths_obj=paths_obj,
                spec=spec,
                case_results=case_results,
                case_results_path=case_results_path if case_results else None,
                summary_payload=summary_payload,
                case_report_paths_by_case=case_report_paths_by_case,
                status="running",
                current_case_name=current_case_name,
            )

            base_cfg = load_yaml_config(case.config_path)
            resolved_overrides = resolve_case_placeholders(case.overrides, case_results=case_results)
            resolved_cfg = apply_dotted_overrides(base_cfg, resolved_overrides)
            if "experiment_name" not in case.overrides:
                resolved_cfg["experiment_name"] = f"{spec.name}-{case.name}"
            resolved_config_path = write_yaml_config(
                paths_obj.configs_dir / f"{case_index:02d}__{slugify(case.name)}.yaml",
                resolved_cfg,
            )
            print(f"[study] running case {case_index}/{len(spec.cases)} name={case.name} kind={case.kind}")

            case_result = _run_case(
                case,
                resolved_config=resolved_cfg,
                resolved_config_path=resolved_config_path,
            )
            case_results[case.name] = case_result
            _write_case_results(case_results_path, case_results)

            case_report_path = _write_case_report(paths_obj=paths_obj, case_name=case.name, case_result=case_result)
            case_report_paths_by_case[case.name] = str(case_report_path)

            if case_result.get("case_kind") == "sample_diffusion":
                row, figure_paths = _build_sample_case_summary_entry(
                    spec=spec,
                    paths_obj=paths_obj,
                    case_name=case.name,
                    case_result=case_result,
                )
                sample_rows.append(row)
                figure_paths_by_case[case.name] = figure_paths
                case_report_path = _write_case_report(
                    paths_obj=paths_obj,
                    case_name=case.name,
                    case_result=case_result,
                    figure_paths=figure_paths,
                    summary_row=row,
                )
                case_report_paths_by_case[case.name] = str(case_report_path)
                summary_payload = _write_sample_summary_outputs(
                    paths_obj=paths_obj,
                    rows=sample_rows,
                    figure_paths_by_case=figure_paths_by_case,
                )

            final_report_path = _write_study_report(
                paths_obj=paths_obj,
                spec=spec,
                case_results=case_results,
                case_results_path=case_results_path,
                summary_payload=summary_payload,
                case_report_paths_by_case=case_report_paths_by_case,
                status="running",
                current_case_name=current_case_name,
            )
            current_case_name = None
    except Exception as exc:
        final_report_path = _write_study_report(
            paths_obj=paths_obj,
            spec=spec,
            case_results=case_results,
            case_results_path=case_results_path if case_results else None,
            summary_payload=summary_payload,
            case_report_paths_by_case=case_report_paths_by_case,
            status="failed",
            current_case_name=current_case_name,
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
    args = parser.parse_args()

    if args.refresh_study_dir is not None:
        refresh_study_outputs(resolve_project_path(args.refresh_study_dir))
        return
    run_study_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
