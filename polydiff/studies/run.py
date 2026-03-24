"""Run config-driven Polydiff studies and collect summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..data.diagnostics import outlier_polygon_indices, representative_polygon_indices
from ..data.polygon_dataset import load_polygon_dataset
from ..runs import slugify
from ..training.train import train_from_loaded_config
from ..training.train_guidance_model import train_guidance_model_from_loaded_config
from ..utils.runtime import load_yaml_config, resolve_project_path
from ..sampling.sample import SampleCliOverrides, sample_from_loaded_config
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


def _write_case_results(path: Path, case_results: dict[str, dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case_results, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


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


def _write_tradeoff_plot(summary_df: pd.DataFrame, *, out_path: Path) -> Path | None:
    if summary_df.empty:
        return None

    x_key = "generated_summary.score_mean"
    y_key = (
        "distribution_distances.distribution_shift_mean_normalized_w1"
        if "distribution_distances.distribution_shift_mean_normalized_w1" in summary_df.columns
        else "generated_summary.self_intersection_rate"
    )
    if x_key not in summary_df.columns or y_key not in summary_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(summary_df[x_key], summary_df[y_key], s=46, color="tab:blue")
    for _, row in summary_df.iterrows():
        ax.annotate(str(row["case_name"]), (row[x_key], row[y_key]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("generated score mean")
    ax.set_ylabel(y_key.split(".")[-1].replace("_", " "))
    ax.set_title("Study Tradeoff Summary")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _summarize_sample_cases(
    *,
    spec: StudySpec,
    paths_obj: StudyPaths,
    case_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    figure_paths_by_case: dict[str, dict[str, str]] = {}
    for case_name, case_result in case_results.items():
        if case_result.get("case_kind") != "sample_diffusion":
            continue
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
        rows.append(row)
        figure_paths_by_case[case_name] = _generate_case_figures(
            spec=spec,
            paths_obj=paths_obj,
            case_name=case_name,
            case_result=case_result,
            diagnostics_payload=diagnostics_payload,
        )

    summary_df = pd.DataFrame(rows)
    summary_csv_path = paths_obj.reports_dir / "sample_case_summary.csv"
    summary_json_path = paths_obj.reports_dir / "sample_case_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")

    tradeoff_path = _write_tradeoff_plot(summary_df, out_path=paths_obj.figures_dir / "study_tradeoff.png")
    return {
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "tradeoff_plot_path": None if tradeoff_path is None else str(tradeoff_path),
        "figure_paths_by_case": figure_paths_by_case,
    }


def run_study_from_config(config_path: Path) -> Path:
    spec = load_study_spec(config_path)
    paths_obj = create_study_paths(name=spec.name, root_dir=spec.root_dir)
    write_study_metadata(paths_obj.study_dir / "study_metadata.json", spec=spec, paths_obj=paths_obj)

    case_results: dict[str, dict[str, Any]] = {}
    for case_index, case in enumerate(spec.cases, start=1):
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
        case_results[case.name] = _run_case(
            case,
            resolved_config=resolved_cfg,
            resolved_config_path=resolved_config_path,
        )

    case_results_path = _write_case_results(paths_obj.reports_dir / "case_results.json", case_results)
    summary_payload = _summarize_sample_cases(spec=spec, paths_obj=paths_obj, case_results=case_results)
    report_payload = {
        "study_name": paths_obj.study_name,
        "study_dir": str(paths_obj.study_dir),
        "case_results_path": str(case_results_path),
        **summary_payload,
    }
    final_report_path = paths_obj.reports_dir / "study_report.json"
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[study] wrote report {final_report_path}")
    return final_report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to study manifest yaml",
    )
    args = parser.parse_args()
    run_study_from_config(resolve_project_path(args.config))


if __name__ == "__main__":
    main()
