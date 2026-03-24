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
from ..restoration import RestorationAnimationOverlay
from .gen_polygons import centroid_xy, enforce_ccw, normalize_scale_rms, regularity_score
from .polygon_dataset import PolygonDatasetArrays, load_polygon_dataset


def _compute_scores(dataset: PolygonDatasetArrays) -> np.ndarray:
    """Compute regularity scores from polygon coordinates."""
    score = np.empty((dataset.num_polygons,), dtype=np.float32)
    for i, xy_source in enumerate(dataset.iter_polygons()):
        xy = xy_source.astype(np.float64, copy=True)
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


def select_animation_frames(coords: np.ndarray, *, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Select evenly spaced animation frames while always keeping endpoints."""
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (frames, vertices, 2), got {coords.shape}")
    if max_frames < 2:
        raise ValueError(f"max_frames must be at least 2, got {max_frames}")

    total_frames = coords.shape[0]
    if total_frames <= max_frames:
        indices = np.arange(total_frames, dtype=np.int32)
        return coords, indices

    raw = np.linspace(0, total_frames - 1, num=max_frames, dtype=np.float64)
    indices = np.unique(np.round(raw).astype(np.int32))
    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)
    if indices[-1] != total_frames - 1:
        indices = np.append(indices, total_frames - 1)
    return coords[indices], indices


def save_polygon_animation(
    coords: np.ndarray,
    out_path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 120,
    restoration_overlay: RestorationAnimationOverlay | None = None,
) -> Path:
    """Save a GIF showing one polygon's denoising trajectory."""
    if fps < 1:
        raise ValueError(f"fps must be at least 1, got {fps}")

    out_path = Path(out_path)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".gif")
    if out_path.suffix.lower() != ".gif":
        raise ValueError(f"animation output must be a .gif file, got {out_path}")

    try:
        import PIL  # noqa: F401
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError("saving GIF animations requires Pillow; install it with `poetry add pillow`") from exc

    frames, frame_indices = select_animation_frames(coords, max_frames=max_frames)
    overlay = None if restoration_overlay is None else restoration_overlay.select_frames(frame_indices)

    x_arrays = [frames[:, :, 0].reshape(-1)]
    y_arrays = [frames[:, :, 1].reshape(-1)]
    if overlay is not None:
        if overlay.mutant_target_points.size > 0:
            x_arrays.append(overlay.mutant_target_points[:, 0])
            y_arrays.append(overlay.mutant_target_points[:, 1])
        if overlay.wild_type_target_points.size > 0:
            x_arrays.append(overlay.wild_type_target_points[:, 0])
            y_arrays.append(overlay.wild_type_target_points[:, 1])
        if overlay.protein_points.size > 0:
            x_arrays.append(overlay.protein_points[:, :, 0].reshape(-1))
            y_arrays.append(overlay.protein_points[:, :, 1].reshape(-1))
        x_arrays.extend(
            [
                overlay.ligand_binding_site[0:1],
                overlay.dna_unbound_position[0:1],
                overlay.dna_bound_position[0:1],
                overlay.contact_points[:, 0],
                overlay.dna_positions[:, 0],
            ]
        )
        y_arrays.extend(
            [
                overlay.ligand_binding_site[1:2],
                overlay.dna_unbound_position[1:2],
                overlay.dna_bound_position[1:2],
                overlay.contact_points[:, 1],
                overlay.dna_positions[:, 1],
            ]
        )

    x_all = np.concatenate(x_arrays, axis=0)
    y_all = np.concatenate(y_arrays, axis=0)
    x_min = float(x_all.min())
    x_max = float(x_all.max())
    y_min = float(y_all.min())
    y_max = float(y_all.max())

    x_pad = max((x_max - x_min) * 0.15, 0.25)
    y_pad = max((y_max - y_min) * 0.15, 0.25)

    if overlay is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
        title_x = 0.5
        title_size = 11
    else:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        fig.subplots_adjust(left=0.06, right=0.68, bottom=0.08, top=0.82)
        title_x = 0.36
        title_size = 14
    line, = ax.plot([], [], color="black", linewidth=2)
    points = ax.scatter([], [], s=18, color="black")
    title = fig.suptitle("", x=title_x, y=0.96, ha="center", va="top", fontsize=title_size)
    protein_line = None
    contact_point_artist = None
    contact_line_artist = None
    dna_point_artist = None

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if overlay is not None:
        if overlay.mutant_target_points.shape[0] >= 2:
            mutant_closed = np.vstack([overlay.mutant_target_points, overlay.mutant_target_points[0]])
            ax.plot(
                mutant_closed[:, 0],
                mutant_closed[:, 1],
                color="0.75",
                linewidth=1.5,
                linestyle="-",
                label="mutant protein",
            )
        elif overlay.mutant_target_points.shape[0] == 1:
            ax.scatter(
                overlay.mutant_target_points[:, 0],
                overlay.mutant_target_points[:, 1],
                s=24,
                color="0.75",
                label="mutant protein",
            )
        if overlay.wild_type_target_points.shape[0] >= 2:
            wt_closed = np.vstack([overlay.wild_type_target_points, overlay.wild_type_target_points[0]])
            ax.plot(
                wt_closed[:, 0],
                wt_closed[:, 1],
                color="tab:green",
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                label="WT protein",
            )
        elif overlay.wild_type_target_points.shape[0] == 1:
            ax.scatter(
                overlay.wild_type_target_points[:, 0],
                overlay.wild_type_target_points[:, 1],
                s=24,
                color="tab:green",
                alpha=0.7,
                label="WT protein",
            )
        protein_line, = ax.plot([], [], color="tab:blue", linewidth=1.8, label="restored protein")
        ax.plot(
            [overlay.dna_unbound_position[0], overlay.dna_bound_position[0]],
            [overlay.dna_unbound_position[1], overlay.dna_bound_position[1]],
            color="0.7",
            linewidth=1.0,
            linestyle="--",
        )
        ax.scatter(
            [overlay.ligand_binding_site[0]],
            [overlay.ligand_binding_site[1]],
            marker="x",
            s=48,
            color="tab:purple",
            label="ligand site",
        )
        ax.scatter(
            [overlay.dna_unbound_position[0]],
            [overlay.dna_unbound_position[1]],
            s=36,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.5,
            label="DNA unbound",
        )
        ax.scatter(
            [overlay.dna_bound_position[0]],
            [overlay.dna_bound_position[1]],
            s=36,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.5,
            label="DNA bound",
        )
        contact_line_artist, = ax.plot([], [], color="tab:blue", linewidth=1.0, linestyle="--")
        contact_point_artist = ax.scatter([], [], marker="x", s=36, color="tab:blue", label="ligand contact")
        dna_point_artist = ax.scatter([], [], s=32, color="tab:orange", label="DNA")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            fontsize=8,
        )

    def _update(frame_idx: int):
        xy = frames[frame_idx]
        xy_closed = np.vstack([xy, xy[0]])
        line.set_data(xy_closed[:, 0], xy_closed[:, 1])
        points.set_offsets(xy)
        if overlay is None:
            title.set_text(f"frame {frame_indices[frame_idx] + 1}/{coords.shape[0]}")
            return line, points, title

        contact_xy = overlay.contact_points[frame_idx]
        dna_xy = overlay.dna_positions[frame_idx]
        protein_xy = overlay.protein_points[frame_idx]
        protein_closed = np.vstack([protein_xy, protein_xy[0]])
        protein_line.set_data(protein_closed[:, 0], protein_closed[:, 1])
        contact_point_artist.set_offsets(contact_xy.reshape(1, 2))
        dna_point_artist.set_offsets(dna_xy.reshape(1, 2))
        contact_line_artist.set_data(
            [contact_xy[0], overlay.ligand_binding_site[0]],
            [contact_xy[1], overlay.ligand_binding_site[1]],
        )
        title.set_text(
            f"frame {frame_indices[frame_idx] + 1}/{coords.shape[0]} | "
            f"restore={overlay.protein_restoration[frame_idx]:.2f} | "
            f"dna_bind={overlay.dna_binding_activation[frame_idx]:.2f}\n"
            f"contact_d={overlay.contact_drift[frame_idx]:.2f} | "
            f"dna_d={overlay.dna_distance[frame_idx]:.2f}"
        )
        return line, points, protein_line, title, contact_point_artist, dna_point_artist, contact_line_artist

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=int(round(1000 / fps)),
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


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

    dataset = load_polygon_dataset(file_path)
    with np.load(file_path, allow_pickle=True) as data:
        score = np.asarray(data["score"], dtype=np.float32) if "score" in data else None
    if score is None and compute_score:
        score = _compute_scores(dataset)

    use_score_color = color_by_score or (score is not None and not no_color_by_score)
    # Preserve original behavior for training/generated data:
    # if score exists in the file, show numeric labels by default.
    use_show_scores = (score is not None) or show_scores or compute_score

    num = min(num, dataset.num_polygons)
    cols = int(math.ceil(math.sqrt(num)))
    rows = int(math.ceil(num / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(num):
        plot_polygon(
            axes[i],
            dataset.polygon(i),
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
