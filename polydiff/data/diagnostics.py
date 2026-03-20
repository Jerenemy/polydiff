"""Dataset-level diagnostics for polygon training and sampling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .gen_polygons import (
    centroid_xy,
    edge_lengths,
    enforce_ccw,
    is_self_intersecting,
    normalize_scale_rms,
    polygon_signed_area_xy,
    regularity_score,
)


def anchor_index(xy: np.ndarray) -> int:
    """Return a stable anchor vertex index for cyclic alignment."""
    return int(np.lexsort((xy[:, 1], -xy[:, 0]))[0])


def canonicalize_polygon(xy: np.ndarray) -> np.ndarray:
    """Center, normalize, orient, and cyclically align a polygon."""
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - centroid_xy(xy)
    xy = normalize_scale_rms(xy)
    xy = enforce_ccw(xy)

    start = anchor_index(xy)
    xy = np.roll(xy, -start, axis=0)

    theta = math.atan2(float(xy[0, 1]), float(xy[0, 0]))
    c = math.cos(-theta)
    s = math.sin(-theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy = xy @ rotation.T
    return xy.astype(np.float32)


def polygon_metric_row(xy: np.ndarray) -> dict[str, float]:
    """Compute raw and canonicalized metrics for a single polygon."""
    xy = np.asarray(xy, dtype=np.float64)
    centroid = centroid_xy(xy)
    centered = xy - centroid
    radii = np.linalg.norm(centered, axis=1)
    rms_radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    signed_area = float(polygon_signed_area_xy(xy))

    xy_canon = canonicalize_polygon(xy)
    reg = regularity_score(xy_canon.astype(np.float64))
    lengths_canon = edge_lengths(xy_canon).astype(np.float64)
    canon_area = abs(float(polygon_signed_area_xy(xy_canon)))
    canon_perimeter = float(lengths_canon.sum())
    compactness = float(4.0 * math.pi * canon_area / (canon_perimeter ** 2 + 1e-12))

    return {
        "raw_centroid_norm": float(np.linalg.norm(centroid)),
        "raw_rms_radius": rms_radius,
        "raw_ccw": float(signed_area > 0.0),
        "self_intersection": float(is_self_intersecting(xy)),
        "score": float(reg.score),
        "edge_cv": float(reg.edge_cv),
        "angle_cv": float(reg.angle_cv),
        "radius_cv": float(reg.radius_cv),
        "area": canon_area,
        "compactness": compactness,
        "min_radius": float(radii.min()),
        "max_radius": float(radii.max()),
    }


def summarize_polygon_dataset(coords: np.ndarray) -> dict[str, float | int]:
    """Aggregate polygon metrics for an entire dataset."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (num_polygons, n_vertices, 2), got {coords.shape}")

    rows = [polygon_metric_row(xy) for xy in coords]
    metrics = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }

    score = metrics["score"]
    out: dict[str, float | int] = {
        "num_polygons": int(coords.shape[0]),
        "n_vertices": int(coords.shape[1]),
        "raw_centroid_norm_mean": float(metrics["raw_centroid_norm"].mean()),
        "raw_centroid_norm_std": float(metrics["raw_centroid_norm"].std()),
        "raw_rms_radius_mean": float(metrics["raw_rms_radius"].mean()),
        "raw_rms_radius_std": float(metrics["raw_rms_radius"].std()),
        "raw_ccw_rate": float(metrics["raw_ccw"].mean()),
        "self_intersection_rate": float(metrics["self_intersection"].mean()),
        "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_p05": float(np.quantile(score, 0.05)),
        "score_p50": float(np.quantile(score, 0.50)),
        "score_p95": float(np.quantile(score, 0.95)),
        "edge_cv_mean": float(metrics["edge_cv"].mean()),
        "angle_cv_mean": float(metrics["angle_cv"].mean()),
        "radius_cv_mean": float(metrics["radius_cv"].mean()),
        "area_mean": float(metrics["area"].mean()),
        "area_std": float(metrics["area"].std()),
        "compactness_mean": float(metrics["compactness"].mean()),
        "compactness_std": float(metrics["compactness"].std()),
        "min_radius_mean": float(metrics["min_radius"].mean()),
        "max_radius_mean": float(metrics["max_radius"].mean()),
    }
    return out


def compare_polygon_summaries(
    reference: dict[str, float | int],
    observed: dict[str, float | int],
) -> dict[str, float]:
    """Return observed-minus-reference deltas for key diagnostics."""
    keys = [
        "raw_centroid_norm_mean",
        "raw_rms_radius_mean",
        "raw_ccw_rate",
        "self_intersection_rate",
        "score_mean",
        "score_std",
        "edge_cv_mean",
        "angle_cv_mean",
        "radius_cv_mean",
        "area_mean",
        "compactness_mean",
    ]
    deltas: dict[str, float] = {}
    for key in keys:
        if key in reference and key in observed:
            deltas[f"{key}_delta"] = float(observed[key]) - float(reference[key])
    return deltas


def format_polygon_summary(summary: dict[str, float | int]) -> str:
    """Compact human-readable summary for logs."""
    return (
        f"n={int(summary['num_polygons'])} "
        f"score={float(summary['score_mean']):.4f}±{float(summary['score_std']):.4f} "
        f"edge_cv={float(summary['edge_cv_mean']):.4f} "
        f"angle_cv={float(summary['angle_cv_mean']):.4f} "
        f"radius_cv={float(summary['radius_cv_mean']):.4f} "
        f"area={float(summary['area_mean']):.4f} "
        f"compactness={float(summary['compactness_mean']):.4f} "
        f"centroid_norm={float(summary['raw_centroid_norm_mean']):.4f} "
        f"rms_radius={float(summary['raw_rms_radius_mean']):.4f} "
        f"ccw_rate={float(summary['raw_ccw_rate']):.4f} "
        f"self_intersection={float(summary['self_intersection_rate']):.4f}"
    )


def format_polygon_delta_summary(deltas: dict[str, float]) -> str:
    """Compact human-readable delta summary for logs."""
    ordered = [
        ("score_mean_delta", "score"),
        ("score_std_delta", "score_std"),
        ("edge_cv_mean_delta", "edge_cv"),
        ("angle_cv_mean_delta", "angle_cv"),
        ("radius_cv_mean_delta", "radius_cv"),
        ("area_mean_delta", "area"),
        ("compactness_mean_delta", "compactness"),
        ("raw_centroid_norm_mean_delta", "centroid_norm"),
        ("raw_rms_radius_mean_delta", "rms_radius"),
        ("raw_ccw_rate_delta", "ccw_rate"),
        ("self_intersection_rate_delta", "self_intersection"),
    ]
    pieces = [
        f"{label}Δ={float(deltas[key]):+.4f}"
        for key, label in ordered
        if key in deltas
    ]
    return " ".join(pieces)


def json_ready(value: Any) -> Any:
    """Convert numpy scalars/arrays into plain Python objects for JSON output."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
