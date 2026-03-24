import numpy as np

from polydiff.data.diagnostics import (
    compare_polygon_summaries,
    format_polygon_delta_summary,
    format_polygon_summary,
    summarize_polygon_dataset,
)
from polydiff.data.gen_polygons import batch, make_polygon
from polydiff.data.polygon_dataset import concatenate_polygons


def test_summarize_polygon_dataset_returns_expected_keys():
    coords, _, _ = batch(n=6, num=16, seed=0)

    summary = summarize_polygon_dataset(coords)

    assert summary["num_polygons"] == 16
    assert summary["n_vertices"] == 6
    assert 0.0 < summary["score_mean"] <= 1.0
    assert summary["raw_ccw_rate"] == 1.0
    assert np.isfinite(summary["area_mean"])
    assert np.isfinite(summary["compactness_mean"])


def test_compare_polygon_summaries_detects_distribution_shift():
    coords_a, _, _ = batch(n=6, num=32, seed=0, radial_sigma=0.08, angle_sigma=0.04, smooth_passes=5)
    coords_b, _, _ = batch(n=6, num=32, seed=1, radial_sigma=0.30, angle_sigma=0.20, smooth_passes=1)

    summary_a = summarize_polygon_dataset(coords_a)
    summary_b = summarize_polygon_dataset(coords_b)
    deltas = compare_polygon_summaries(summary_a, summary_b)

    assert deltas["score_mean_delta"] < 0.0
    assert deltas["edge_cv_mean_delta"] > 0.0
    assert deltas["angle_cv_mean_delta"] > 0.0
    assert "score" in format_polygon_delta_summary(deltas)
    assert "score=" in format_polygon_summary(summary_a)


def test_summarize_polygon_dataset_handles_ragged_storage():
    rng = np.random.default_rng(7)
    polygons = [
        make_polygon(n=5, deform=0.2, rng=rng),
        make_polygon(n=7, deform=0.4, rng=rng),
        make_polygon(n=5, deform=0.6, rng=rng),
    ]
    coords, num_vertices = concatenate_polygons(polygons)

    summary = summarize_polygon_dataset(coords, num_vertices=num_vertices)

    assert summary["num_polygons"] == 3
    assert summary["min_vertices"] == 5
    assert summary["max_vertices"] == 7
    assert summary["vertex_count_histogram"] == {5: 2, 7: 1}
    assert "vertices=5-7" in format_polygon_summary(summary)
