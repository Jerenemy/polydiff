"""Study-level plots built from saved sample outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.diagnostics import (
    DEFAULT_OUTLIER_FAILURE_MODE_ORDER,
    canonical_polygon_feature_matrix,
    polygon_metric_table,
)
from ..data.plot_polygons import plot_polygon
from ..data.polygon_dataset import PolygonDatasetArrays


def _as_dataset(
    coords: np.ndarray | PolygonDatasetArrays,
    num_vertices: np.ndarray | None = None,
) -> PolygonDatasetArrays:
    if isinstance(coords, PolygonDatasetArrays):
        if num_vertices is not None:
            raise ValueError("num_vertices must not be provided when coords is already a PolygonDatasetArrays")
        return coords
    coords_array = np.asarray(coords, dtype=np.float32)
    if num_vertices is None:
        if coords_array.ndim != 3:
            raise ValueError(
                "coords must have shape (num_polygons, n_vertices, 2) when num_vertices is omitted, "
                f"got {coords_array.shape}"
            )
        num_vertices = np.full((coords_array.shape[0],), coords_array.shape[1], dtype=np.int32)
    return PolygonDatasetArrays(coords=coords_array, num_vertices=num_vertices)


def _sample_rows(array: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    if array.shape[0] <= max_rows:
        return array
    rng = np.random.default_rng(seed)
    indices = rng.choice(array.shape[0], size=max_rows, replace=False)
    return array[np.sort(indices)]


def save_score_distribution_figure(
    reference_table,
    observed_table,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0.0, 1.0, num=31, dtype=np.float64)
    ax.hist(reference_table["score"], bins=bins, density=True, alpha=0.45, label="reference", color="tab:blue")
    ax.hist(observed_table["score"], bins=bins, density=True, alpha=0.45, label="generated", color="tab:orange")
    ax.set_xlabel("regularity score")
    ax.set_ylabel("density")
    ax.set_title("Score Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_pca_projection_figure(
    reference_coords: np.ndarray | PolygonDatasetArrays,
    observed_coords: np.ndarray | PolygonDatasetArrays,
    out_path: str | Path,
    *,
    reference_num_vertices: np.ndarray | None = None,
    observed_num_vertices: np.ndarray | None = None,
    max_points: int = 2000,
) -> Path | None:
    reference_dataset = _as_dataset(reference_coords, num_vertices=reference_num_vertices)
    observed_dataset = _as_dataset(observed_coords, num_vertices=observed_num_vertices)
    if (
        not reference_dataset.is_uniform
        or not observed_dataset.is_uniform
        or int(reference_dataset.num_vertices[0]) != int(observed_dataset.num_vertices[0])
    ):
        return None

    reference_features = _sample_rows(
        canonical_polygon_feature_matrix(reference_dataset),
        max_rows=max_points,
        seed=0,
    )
    observed_features = _sample_rows(
        canonical_polygon_feature_matrix(observed_dataset),
        max_rows=max_points,
        seed=1,
    )
    combined = np.concatenate([reference_features, observed_features], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    reference_projection = projection[: reference_features.shape[0]]
    observed_projection = projection[reference_features.shape[0] :]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(reference_projection[:, 0], reference_projection[:, 1], s=10, alpha=0.25, label="reference")
    ax.scatter(observed_projection[:, 0], observed_projection[:, 1], s=10, alpha=0.25, label="generated")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Canonical Polygon PCA")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_polygon_gallery(
    coords: np.ndarray | PolygonDatasetArrays,
    indices: np.ndarray | list[int],
    out_path: str | Path,
    *,
    num_vertices: np.ndarray | None = None,
    title: str | None = None,
) -> Path | None:
    dataset = _as_dataset(coords, num_vertices=num_vertices)
    index_array = np.asarray(indices, dtype=np.int32).reshape(-1)
    if index_array.size == 0:
        return None

    metric_table = polygon_metric_table(dataset)
    count = int(index_array.size)
    n_cols = min(4, count)
    n_rows = int(math.ceil(count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for ax, polygon_index in zip(axes_array, index_array.tolist()):
        xy = dataset.polygon(int(polygon_index))
        score = float(metric_table.loc[int(polygon_index), "score"])
        plot_polygon(ax, xy, score=score, color_by_score=True)
        ax.set_title(f"#{int(polygon_index)}  score={score:.3f}", fontsize=9)
    for ax in axes_array[count:]:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_multi_case_score_distribution_figure(
    case_tables: list[tuple[str, pd.DataFrame]],
    out_path: str | Path,
    *,
    title: str = "Score Distribution Comparison",
) -> Path | None:
    if len(case_tables) < 2:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bins = np.linspace(0.0, 1.0, num=41, dtype=np.float64)
    colors = plt.get_cmap("tab10")
    for index, (label, table) in enumerate(case_tables):
        if "score" not in table.columns:
            continue
        ax.hist(
            table["score"].to_numpy(dtype=np.float64, copy=False),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=label,
            color=colors(index % 10),
        )
    ax.set_xlabel("regularity score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _guidance_metric_spec(summary_df: pd.DataFrame) -> tuple[str, str, bool] | None:
    candidates = (
        ("final_mae", "final MAE", False),
        ("best_mae", "best MAE", False),
        ("final_acc", "final accuracy", True),
        ("best_acc", "best accuracy", True),
        ("final_ema_loss", "final EMA loss", False),
        ("final_loss", "final loss", False),
    )
    for key, label, higher_is_better in candidates:
        if key in summary_df.columns and summary_df[key].notna().any():
            return key, label, higher_is_better
    return None


def _guidance_secondary_metric_spec(summary_df: pd.DataFrame, *, primary_key: str) -> tuple[str, str, bool] | None:
    candidates = (
        ("final_ema_loss", "final EMA loss", False),
        ("final_loss", "final loss", False),
        ("best_mae", "best MAE", False),
        ("best_acc", "best accuracy", True),
    )
    for key, label, higher_is_better in candidates:
        if key == primary_key:
            continue
        if key in summary_df.columns and summary_df[key].notna().any():
            return key, label, higher_is_better
    return None


def _guidance_case_color_map(
    summary_df: pd.DataFrame,
    case_histories: list[tuple[str, pd.DataFrame]],
) -> dict[str, tuple[float, float, float, float]]:
    ordered_case_names: list[str] = []
    if "case_name" in summary_df.columns:
        for case_name in summary_df["case_name"].astype(str).tolist():
            if case_name not in ordered_case_names:
                ordered_case_names.append(case_name)
    for case_name, _ in case_histories:
        label = str(case_name)
        if label not in ordered_case_names:
            ordered_case_names.append(label)

    cmap = plt.get_cmap("tab10")
    return {
        case_name: cmap(index % 10)
        for index, case_name in enumerate(ordered_case_names)
    }


def _plot_guidance_summary_bars(
    ax,
    summary_df: pd.DataFrame,
    *,
    metric_key: str,
    metric_label: str,
    higher_is_better: bool,
    case_colors: dict[str, tuple[float, float, float, float]],
) -> None:
    plot_df = summary_df.loc[:, ["case_name", metric_key]].dropna().copy()
    if plot_df.empty:
        ax.axis("off")
        return
    plot_df = plot_df.sort_values(metric_key, ascending=not higher_is_better, kind="stable")
    values = plot_df[metric_key].to_numpy(dtype=np.float64, copy=False)
    labels = plot_df["case_name"].astype(str).tolist()
    colors = [case_colors[label] for label in labels]
    y_positions = np.arange(len(plot_df), dtype=np.float64)
    ax.barh(y_positions, values, color=colors, alpha=0.85)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    ax.set_title(f"{metric_label} ({'higher' if higher_is_better else 'lower'} is better)")
    ax.grid(axis="x", alpha=0.2)


def save_guidance_training_comparison_figure(
    summary_df: pd.DataFrame,
    case_histories: list[tuple[str, pd.DataFrame]],
    out_path: str | Path,
    *,
    title: str = "Guidance Training Comparison",
) -> Path | None:
    primary_spec = _guidance_metric_spec(summary_df)
    has_histories = any(not history.empty for _, history in case_histories)
    if primary_spec is None and not has_histories:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes_array = np.atleast_1d(axes).reshape(-1)
    case_colors = _guidance_case_color_map(summary_df, case_histories)

    if has_histories:
        for case_name, history in case_histories:
            if history.empty or "step" not in history.columns:
                continue
            color = case_colors[str(case_name)]
            if "loss" in history.columns and history["loss"].notna().any():
                axes_array[0].plot(
                    history["step"].to_numpy(dtype=np.float64, copy=False),
                    history["loss"].to_numpy(dtype=np.float64, copy=False),
                    label=case_name,
                    color=color,
                    linewidth=1.5,
                )
            if "ema_loss" in history.columns and history["ema_loss"].notna().any():
                axes_array[1].plot(
                    history["step"].to_numpy(dtype=np.float64, copy=False),
                    history["ema_loss"].to_numpy(dtype=np.float64, copy=False),
                    label=case_name,
                    color=color,
                    linewidth=1.5,
                )
        axes_array[0].set_xlabel("step")
        axes_array[0].set_ylabel("loss")
        axes_array[0].set_title("Training Loss")
        axes_array[0].grid(alpha=0.2)
        axes_array[1].set_xlabel("step")
        axes_array[1].set_ylabel("EMA loss")
        axes_array[1].set_title("Training EMA Loss")
        axes_array[1].grid(alpha=0.2)
        axes_array[1].legend(frameon=False, fontsize=8)
    else:
        axes_array[0].axis("off")
        axes_array[1].axis("off")

    if primary_spec is None:
        axes_array[2].axis("off")
        axes_array[3].axis("off")
    else:
        primary_key, primary_label, primary_higher_is_better = primary_spec
        _plot_guidance_summary_bars(
            axes_array[2],
            summary_df,
            metric_key=primary_key,
            metric_label=primary_label,
            higher_is_better=primary_higher_is_better,
            case_colors=case_colors,
        )
        secondary_spec = _guidance_secondary_metric_spec(summary_df, primary_key=primary_key)
        if secondary_spec is None:
            axes_array[3].axis("off")
        else:
            secondary_key, secondary_label, secondary_higher_is_better = secondary_spec
            _plot_guidance_summary_bars(
                axes_array[3],
                summary_df,
                metric_key=secondary_key,
                metric_label=secondary_label,
                higher_is_better=secondary_higher_is_better,
                case_colors=case_colors,
            )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _metric_label(metric_key: str) -> str:
    labels = {
        "generated_summary.score_mean": "score mean",
        "generated_summary.score_p95": "score p95",
        "generated_summary.score_p99": "score p99",
        "score_threshold_rates.score_ge_0p6_rate": "rate score >= 0.6",
        "score_threshold_rates.score_ge_0p7_rate": "rate score >= 0.7",
        "score_threshold_rates.score_ge_0p8_rate": "rate score >= 0.8",
        "distribution_distances.shape_distribution_shift_mean_normalized_w1": "shape shift",
        "distribution_distances.pose_distribution_shift_mean_normalized_w1": "pose drift",
        "generated_summary.self_intersection_rate": "self-intersection rate",
    }
    return labels.get(metric_key, metric_key.split(".")[-1].replace("_", " "))


def save_metric_sweep_figure(
    summary_df: pd.DataFrame,
    out_path: str | Path,
    *,
    x_key: str,
    x_label: str,
    title: str,
    x_scale: str = "linear",
    metric_keys: tuple[str, ...] = (
        "generated_summary.score_mean",
        "generated_summary.score_p99",
        "score_threshold_rates.score_ge_0p7_rate",
        "distribution_distances.shape_distribution_shift_mean_normalized_w1",
    ),
    group_key: str | None = None,
    group_order: list[str] | None = None,
    x_order: list[str] | None = None,
) -> Path | None:
    available_metrics = [metric for metric in metric_keys if metric in summary_df.columns]
    if summary_df.empty or x_key not in summary_df.columns or not available_metrics:
        return None

    plot_df = summary_df.copy()
    if x_order is not None:
        order_lookup = {value: index for index, value in enumerate(x_order)}
        plot_df["_x_sort_key"] = plot_df[x_key].map(lambda value: order_lookup.get(value, len(order_lookup)))
    else:
        plot_df["_x_sort_key"] = plot_df[x_key]

    if group_key is None or group_key not in plot_df.columns:
        group_values = [("__single__", plot_df)]
    else:
        if group_order is None:
            ordered_values = sorted(str(value) for value in plot_df[group_key].dropna().unique().tolist())
        else:
            ordered_values = list(group_order)
        group_values = [
            (group_value, plot_df[plot_df[group_key].astype(str) == str(group_value)].copy())
            for group_value in ordered_values
        ]
        group_values = [(name, frame) for name, frame in group_values if not frame.empty]
        if not group_values:
            return None

    n_metrics = min(4, len(available_metrics))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    axes_array = np.atleast_1d(axes).reshape(-1)
    colors = plt.get_cmap("tab10")
    categorical_x = x_order is not None or plot_df[x_key].dtype == object
    x_tick_positions = None
    x_tick_labels = None
    if categorical_x:
        categories = x_order or [str(value) for value in plot_df.sort_values("_x_sort_key")[x_key].drop_duplicates().tolist()]
        x_tick_positions = np.arange(len(categories), dtype=np.float64)
        x_tick_labels = categories

    for axis_index, metric_key in enumerate(available_metrics[:n_metrics]):
        ax = axes_array[axis_index]
        for group_index, (group_name, group_frame) in enumerate(group_values):
            ordered = group_frame.sort_values(["_x_sort_key", x_key], kind="stable")
            if categorical_x:
                position_lookup = {label: position for position, label in enumerate(x_tick_labels or [])}
                x_values = np.asarray(
                    [position_lookup[str(value)] for value in ordered[x_key].tolist()],
                    dtype=np.float64,
                )
            else:
                x_values = ordered[x_key].to_numpy(dtype=np.float64, copy=False)
            y_values = ordered[metric_key].to_numpy(dtype=np.float64, copy=False)
            label = None if group_name == "__single__" else str(group_name)
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.6,
                markersize=4.5,
                color=colors(group_index % 10),
                label=label,
            )
        ax.set_ylabel(_metric_label(metric_key))
        ax.grid(alpha=0.2)
        if categorical_x and x_tick_positions is not None and x_tick_labels is not None:
            ax.set_xticks(x_tick_positions, x_tick_labels, rotation=20, ha="right")
        elif x_scale == "symlog":
            ax.set_xscale("symlog", linthresh=1.0)
        elif x_scale == "log":
            ax.set_xscale("log")
        ax.set_xlabel(x_label)

    for ax in axes_array[n_metrics:]:
        ax.axis("off")

    if len(group_values) > 1:
        axes_array[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_failure_mode_rate_figure(
    summary_df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str = "Outlier Failure Modes",
    prefix: str = "outlier_failure_modes",
) -> Path | None:
    if summary_df.empty or "case_name" not in summary_df.columns:
        return None
    count_columns = [f"{prefix}.{mode}_count" for mode in DEFAULT_OUTLIER_FAILURE_MODE_ORDER if f"{prefix}.{mode}_count" in summary_df.columns]
    if not count_columns:
        return None

    plot_df = summary_df.loc[:, ["case_name", *count_columns]].copy()
    if plot_df[count_columns].to_numpy(dtype=np.float64).sum() <= 0.0:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 0.65 * len(plot_df))))
    colors = plt.get_cmap("tab20")
    left = np.zeros((len(plot_df),), dtype=np.float64)
    y_positions = np.arange(len(plot_df), dtype=np.float64)
    for index, column in enumerate(count_columns):
        label = column.split(".")[-1].replace("_count", "").replace("_", " ")
        values = plot_df[column].to_numpy(dtype=np.float64, copy=False)
        ax.barh(
            y_positions,
            values,
            left=left,
            color=colors(index % 20),
            label=label,
            alpha=0.9,
        )
        left = left + values
    ax.set_yticks(y_positions, plot_df["case_name"].tolist())
    ax.set_xlabel("count among labeled outliers")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
