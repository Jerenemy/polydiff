import numpy as np

from polydiff.data.gen_polygons import (
    batch,
    make_polygon,
    polygon_signed_area_xy,
    regularity_score,
)


def test_make_polygon_properties():
    rng = np.random.default_rng(0)
    xy = make_polygon(n=6, deform=0.3, rng=rng)
    assert xy.shape == (6, 2)

    rms = np.sqrt(np.mean(np.sum(xy * xy, axis=1)))
    assert np.isclose(rms, 1.0, atol=1e-3)
    assert polygon_signed_area_xy(xy) > 0

    reg = regularity_score(xy)
    assert np.isfinite(reg.score)
    assert 0.0 < reg.score <= 1.0


def test_batch_shapes():
    X, score, deform = batch(n=5, num=8, seed=123)
    assert X.shape == (8, 5, 2)
    assert score.shape == (8,)
    assert deform.shape == (8,)
    assert np.all(score > 0.0)
    assert np.all(score <= 1.0)
    assert np.all(deform >= 0.0)
    assert np.all(deform <= 1.0)
