"""Study-level plots built from saved sample outputs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..data.diagnostics import canonical_polygon_feature_matrix, polygon_metric_table
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
