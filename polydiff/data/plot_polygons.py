#!/usr/bin/env python3
"""
Visualize polygons saved by gen_polygons.py

Usage:
  python -m polydiff.data.plot_polygons hexagons.npz --num 16
  python -m polydiff.data.plot_polygons triangles.npz --num 25 --color_by_score
  python -m polydiff.data.plot_polygons data/processed/samples.npz --num 16 --compute_score
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import paths
from .gen_polygons import centroid_xy, enforce_ccw, normalize_scale_rms, regularity_score


def _compute_scores(coords: np.ndarray) -> np.ndarray:
    """Compute regularity scores from polygon coordinates."""
    score = np.empty((coords.shape[0],), dtype=np.float32)
    for i in range(coords.shape[0]):
        xy = coords[i].astype(np.float64, copy=True)
        # Match generator scoring assumptions.
        xy = xy - centroid_xy(xy)
        xy = normalize_scale_rms(xy)
        xy = enforce_ccw(xy)
        score[i] = np.float32(regularity_score(xy).score)
    return score


def plot_polygon(ax, xy, score=None, color_by_score=False):
    # Close the polygon
    xy_closed = np.vstack([xy, xy[0]])

    if color_by_score and score is not None:
        # green = good, red = bad
        c = plt.cm.viridis(score)
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color=c, linewidth=2)
    else:
        ax.plot(xy_closed[:, 0], xy_closed[:, 1], color="black", linewidth=2)

    ax.scatter(xy[:, 0], xy[:, 1], s=10, color="black")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_file(
    file: str | Path,
    *,
    num: int = 16,
    compute_score: bool = False,
    show_scores: bool = False,
    color_by_score: bool = False,
    no_color_by_score: bool = False,
):
    """Plot polygons from a saved .npz file and return (fig, axes)."""
    data_dir = paths.RAW_DATA_DIR
    file_path = paths.resolve_path(file, data_dir)

    data = np.load(file_path)
    coords = data["coords"]
    score = data["score"] if "score" in data else None
    if score is None and compute_score:
        score = _compute_scores(coords)

    use_score_color = color_by_score or (score is not None and not no_color_by_score)
    # Preserve original behavior for training/generated data:
    # if score exists in the file, show numeric labels by default.
    use_show_scores = (score is not None) or show_scores or compute_score

    num = min(num, coords.shape[0])
    cols = int(math.ceil(math.sqrt(num)))
    rows = int(math.ceil(num / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num):
        plot_polygon(
            axes[i],
            coords[i],
            score=None if score is None else score[i],
            color_by_score=use_score_color,
        )
        if use_show_scores and score is not None:
            axes[i].set_title(f"{score[i]:.3f}", fontsize=9)

    for j in range(num, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "file",
        type=str,
        help=".npz file from generator (if no path, load from data/raw/)",
    )
    p.add_argument("--num", type=int, default=16, help="number of polygons to show")
    p.add_argument(
        "--compute_score",
        action="store_true",
        help="compute regularity scores from coords (for files without stored score)",
    )
    p.add_argument("--show_scores", action="store_true", help="show numeric score in subplot titles")
    p.add_argument("--color_by_score", action="store_true", help="force color polygons by regularity score")
    p.add_argument("--no_color_by_score", action="store_true", help="disable score-based coloring")
    args = p.parse_args()

    plot_file(
        args.file,
        num=args.num,
        compute_score=args.compute_score,
        show_scores=args.show_scores,
        color_by_score=args.color_by_score,
        no_color_by_score=args.no_color_by_score,
    )
    plt.show()


if __name__ == "__main__":
    main()
