#!/usr/bin/env python3
"""
Synthetic polygon generator for near-regular simple polygons.

Supports either:
- fixed-size datasets stored densely as `(num_polygons, n_vertices, 2)`, or
- variable-size datasets stored raggedly as concatenated `(total_vertices, 2)`
  coordinates plus a `num_vertices` array.

Usage:
  python -m polydiff.data.gen_polygons --n 6 --num 50000 --out hexagons.npz
  python -m polydiff.data.gen_polygons --n 3 --num 20000 --out triangles.npz --no_reject
  python -m polydiff.data.gen_polygons --n 5 --num 10000 --out pentagons.npz --radial_sigma 0.25 --angle_sigma 0.08
  python -m polydiff.data.gen_polygons --size_values 5,6,7,8 --size_probabilities 0.1,0.2,0.4,0.3 --num 50000 --out mixed.npz

Or import and use sample_polygon(), batch(), or batch_variable_sizes().
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .. import paths
from .polygon_dataset import concatenate_polygons

# ----------------------------
# Geometry helpers
# ----------------------------

def polygon_signed_area_xy(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def enforce_ccw(xy: np.ndarray) -> np.ndarray:
    if polygon_signed_area_xy(xy) < 0:
        return xy[::-1].copy()
    return xy


def centroid_xy(xy: np.ndarray) -> np.ndarray:
    # For our purposes, vertex-mean centroid is fine and stable.
    return np.mean(xy, axis=0)


def normalize_scale_rms(xy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Scale so RMS radius = 1
    r2 = np.mean(np.sum(xy * xy, axis=1))
    s = math.sqrt(r2 + eps)
    return xy / s


def normalize_pose_xy(xy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Center a polygon, normalize RMS scale, and enforce CCW orientation."""
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - centroid_xy(xy)
    xy = normalize_scale_rms(xy, eps=eps)
    xy = enforce_ccw(xy)
    return xy.astype(np.float32)


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    # Proper segment intersection test (excluding shared endpoints handled outside).
    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r) -> bool:
        # q lies on pr
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # General case
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    # Colinear cases
    if abs(o1) < 1e-12 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-12 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-12 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-12 and on_segment(c, b, d):
        return True

    return False


def is_self_intersecting(xy: np.ndarray) -> bool:
    n = xy.shape[0]
    # edges: (i -> i+1)
    for i in range(n):
        a = xy[i]
        b = xy[(i + 1) % n]
        for j in range(i + 1, n):
            # Skip adjacent edges and the same edge
            if j == i:
                continue
            if (j + 1) % n == i:
                continue
            if (i + 1) % n == j:
                continue

            c = xy[j]
            d = xy[(j + 1) % n]
            # Also skip if they share endpoints
            if np.allclose(a, c) or np.allclose(a, d) or np.allclose(b, c) or np.allclose(b, d):
                continue
            if segments_intersect(a, b, c, d):
                return True
    return False


# ----------------------------
# Regularity metrics
# ----------------------------

def edge_lengths(xy: np.ndarray) -> np.ndarray:
    diffs = np.roll(xy, -1, axis=0) - xy
    return np.sqrt(np.sum(diffs * diffs, axis=1))


def interior_angles(xy: np.ndarray) -> np.ndarray:
    # Interior angle at vertex i using vectors (i-1 -> i) and (i+1 -> i)
    n = xy.shape[0]
    ang = np.empty(n, dtype=np.float64)
    for i in range(n):
        v1 = xy[(i - 1) % n] - xy[i]
        v2 = xy[(i + 1) % n] - xy[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            ang[i] = np.nan
            continue
        cosv = float(np.dot(v1, v2) / (n1 * n2))
        cosv = max(-1.0, min(1.0, cosv))
        ang[i] = math.acos(cosv)
    return ang


@dataclass
class Regularity:
    edge_cv: float
    angle_cv: float
    radius_cv: float
    score: float


def regularity_score(xy: np.ndarray, alpha: float = 8.0, beta: float = 5.0, gamma: float = 4.0) -> Regularity:
    # Expect xy is centered + normalized
    el = edge_lengths(xy).astype(np.float64)
    ang = interior_angles(xy).astype(np.float64)
    rad = np.sqrt(np.sum(xy.astype(np.float64) ** 2, axis=1))

    def cv(v: np.ndarray) -> float:
        m = float(np.nanmean(v))
        s = float(np.nanstd(v))
        return float(s / (m + 1e-12))

    ecv = cv(el)
    acv = cv(ang)
    rcv = cv(rad)

    # Smooth score: 1.0 for perfect regular, decays with irregularity
    score = float(math.exp(-alpha * ecv - beta * acv - gamma * rcv))
    return Regularity(ecv, acv, rcv, score)


# ----------------------------
# Noise helpers
# ----------------------------

def smooth_circular_noise(n: int, sigma: float, smooth_passes: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(0.0, sigma, size=n).astype(np.float64)
    if smooth_passes <= 0:
        return x
    for _ in range(smooth_passes):
        x = 0.25 * np.roll(x, -1) + 0.5 * x + 0.25 * np.roll(x, 1)
    return x


# ----------------------------
# Polygon generator
# ----------------------------

def make_polygon(
    n: int,
    deform: float,
    rng: np.random.Generator,
    base_radius: float = 1.0,
    radial_sigma: float = 0.18,
    angle_sigma: float = 0.12,
    smooth_passes: int = 3,
) -> np.ndarray:
    """
    Create a near-regular n-gon with smooth radial + angle jitter.
    deform in [0,1] controls how ugly it gets.
    """
    deform = float(np.clip(deform, 0.0, 1.0))

    theta0 = rng.uniform(-math.pi, math.pi)
    k = np.arange(n, dtype=np.float64)
    theta = theta0 + 2.0 * math.pi * k / n

    # Smooth perturbations
    d_r = smooth_circular_noise(n, sigma=radial_sigma * deform, smooth_passes=smooth_passes, rng=rng)
    d_t = smooth_circular_noise(n, sigma=angle_sigma * deform, smooth_passes=smooth_passes, rng=rng)

    r = base_radius * (1.0 + d_r)
    r = np.maximum(r, 0.05 * base_radius)  # keep radii positive

    theta = theta + d_t

    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float64)

    # Canonicalize pose without removing the random global rotation.
    return normalize_pose_xy(xy)


def sample_polygon(
    n: int,
    rng: np.random.Generator,
    reject_self_intersections: bool = True,
    max_tries: int = 200,
    deform_dist: str = "beta",
    radial_sigma: float = 0.18,
    angle_sigma: float = 0.12,
    smooth_passes: int = 3,
) -> Tuple[np.ndarray, Regularity, float]:
    """
    Returns (xy, regularity, deform).
    deform is the sampled deformation level in [0,1].
    """
    for _ in range(max_tries):
        if deform_dist == "beta":
            # Concentrate around medium deformations but still produce very regular and very ugly samples sometimes
            deform = float(rng.beta(2.0, 2.0))
        elif deform_dist == "uniform":
            deform = float(rng.uniform(0.0, 1.0))
        else:
            raise ValueError(f"Unknown deform_dist: {deform_dist}")

        xy = make_polygon(
            n=n,
            deform=deform,
            rng=rng,
            radial_sigma=radial_sigma,
            angle_sigma=angle_sigma,
            smooth_passes=smooth_passes,
        )
        if reject_self_intersections and is_self_intersecting(xy):
            continue

        reg = regularity_score(xy)
        if not np.isfinite(reg.score):
            continue
        return xy, reg, deform

    raise RuntimeError("Failed to sample a valid polygon within max_tries. Reduce deform or disable rejection.")


def batch(
    n: int,
    num: int,
    seed: int,
    reject_self_intersections: bool = True,
    deform_dist: str = "beta",
    radial_sigma: float = 0.18,
    angle_sigma: float = 0.12,
    smooth_passes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (num, n, 2) float32
      score: (num,) float32
      deform: (num,) float32
    """
    rng = np.random.default_rng(seed)
    X = np.empty((num, n, 2), dtype=np.float32)
    score = np.empty((num,), dtype=np.float32)
    deform = np.empty((num,), dtype=np.float32)

    for i in range(num):
        xy, reg, d = sample_polygon(
            n=n,
            rng=rng,
            reject_self_intersections=reject_self_intersections,
            deform_dist=deform_dist,
            radial_sigma=radial_sigma,
            angle_sigma=angle_sigma,
            smooth_passes=smooth_passes,
        )
        X[i] = xy
        score[i] = np.float32(reg.score)
        deform[i] = np.float32(d)

    return X, score, deform


def normalize_size_distribution(
    size_values: Sequence[int],
    size_probabilities: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray(size_values, dtype=np.int32).reshape(-1)
    if sizes.size == 0:
        raise ValueError("size_values must contain at least one polygon size")
    if np.any(sizes < 3):
        raise ValueError(f"all polygon sizes must be >= 3, got {sizes.tolist()}")

    if size_probabilities is None:
        probs = np.full((sizes.size,), 1.0 / sizes.size, dtype=np.float64)
    else:
        probs = np.asarray(size_probabilities, dtype=np.float64).reshape(-1)
        if probs.shape[0] != sizes.shape[0]:
            raise ValueError(
                f"size_probabilities must have length {sizes.shape[0]}, got {probs.shape[0]}"
            )
        if np.any(probs < 0.0):
            raise ValueError("size_probabilities must be non-negative")
        total = float(probs.sum())
        if total <= 0.0:
            raise ValueError("size_probabilities must sum to a positive value")
        probs = probs / total

    return sizes, probs


def sample_polygon_sizes(
    *,
    num: int,
    rng: np.random.Generator,
    size_values: Sequence[int],
    size_probabilities: Sequence[float] | None = None,
) -> np.ndarray:
    sizes, probs = normalize_size_distribution(size_values, size_probabilities)
    return rng.choice(sizes, size=num, replace=True, p=probs).astype(np.int32)


def batch_variable_sizes(
    *,
    size_values: Sequence[int],
    num: int,
    seed: int,
    size_probabilities: Sequence[float] | None = None,
    reject_self_intersections: bool = True,
    deform_dist: str = "beta",
    radial_sigma: float = 0.18,
    angle_sigma: float = 0.12,
    smooth_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      coords: (total_vertices, 2) float32
      num_vertices: (num,) int32
      score: (num,) float32
      deform: (num,) float32
    """
    rng = np.random.default_rng(seed)
    sampled_sizes = sample_polygon_sizes(
        num=num,
        rng=rng,
        size_values=size_values,
        size_probabilities=size_probabilities,
    )
    polygons: list[np.ndarray] = []
    score = np.empty((num,), dtype=np.float32)
    deform = np.empty((num,), dtype=np.float32)

    for i, n in enumerate(sampled_sizes.tolist()):
        xy, reg, d = sample_polygon(
            n=int(n),
            rng=rng,
            reject_self_intersections=reject_self_intersections,
            deform_dist=deform_dist,
            radial_sigma=radial_sigma,
            angle_sigma=angle_sigma,
            smooth_passes=smooth_passes,
        )
        polygons.append(xy)
        score[i] = np.float32(reg.score)
        deform[i] = np.float32(d)

    coords, num_vertices = concatenate_polygons(polygons)
    return coords, num_vertices, score, deform


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("expected a comma-separated list of integers")
    return [int(part) for part in parts]


def _parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("expected a comma-separated list of floats")
    return [float(part) for part in parts]


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="number of vertices for a fixed-size dataset")
    p.add_argument(
        "--size_values",
        type=str,
        default=None,
        help="comma-separated polygon sizes for a variable-size dataset, e.g. 5,6,7,8",
    )
    p.add_argument(
        "--size_probabilities",
        type=str,
        default=None,
        help="comma-separated probabilities aligned with --size_values; defaults to uniform",
    )
    p.add_argument("--num", type=int, default=10000, help="number of samples to generate")
    p.add_argument("--seed", type=int, default=0, help="rng seed")
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="output .npz file (if no path, saves to data/raw/)",
    )
    p.add_argument("--no_reject", action="store_true", help="do not reject self-intersecting polygons")
    p.add_argument("--deform_dist", type=str, default="beta", choices=["beta", "uniform"])
    p.add_argument("--radial_sigma", type=float, default=0.18, help="base radial jitter std (scaled by deform)")
    p.add_argument("--angle_sigma", type=float, default=0.12, help="base angle jitter std (scaled by deform)")
    p.add_argument("--smooth_passes", type=int, default=3, help="smoothing passes for jitter noise")
    args = p.parse_args()

    data_dir = paths.ensure_dir(paths.RAW_DATA_DIR)
    out_path = paths.resolve_path(args.out, data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_values = _parse_csv_ints(args.size_values)
    size_probabilities = _parse_csv_floats(args.size_probabilities)
    if args.n is None and size_values is None:
        raise ValueError("either --n or --size_values is required")
    if args.n is not None and size_values is not None:
        raise ValueError("use either --n or --size_values, not both")

    if args.n is not None:
        X, score, deform = batch(
            n=args.n,
            num=args.num,
            seed=args.seed,
            reject_self_intersections=not args.no_reject,
            deform_dist=args.deform_dist,
            radial_sigma=args.radial_sigma,
            angle_sigma=args.angle_sigma,
            smooth_passes=args.smooth_passes,
        )
        np.savez_compressed(
            out_path,
            coords=X,
            score=score,
            deform=deform,
            n=np.int32(args.n),
        )
        print(f"Saved {args.num} fixed-size samples to {out_path}")
        print(
            f"coords shape: {X.shape}  score: [{score.min():.4f}, {score.max():.4f}]  "
            f"deform: [{deform.min():.4f}, {deform.max():.4f}]"
        )
        return

    coords, num_vertices, score, deform = batch_variable_sizes(
        size_values=size_values or [],
        size_probabilities=size_probabilities,
        num=args.num,
        seed=args.seed,
        reject_self_intersections=not args.no_reject,
        deform_dist=args.deform_dist,
        radial_sigma=args.radial_sigma,
        angle_sigma=args.angle_sigma,
        smooth_passes=args.smooth_passes,
    )
    sizes, probs = normalize_size_distribution(size_values or [], size_probabilities)
    np.savez_compressed(
        out_path,
        coords=coords,
        num_vertices=num_vertices,
        score=score,
        deform=deform,
        size_values=sizes.astype(np.int32),
        size_probabilities=probs.astype(np.float32),
    )

    print(f"Saved {args.num} variable-size samples to {out_path}")
    print(
        f"coords shape: {coords.shape}  polygons: {num_vertices.shape[0]}  "
        f"min_vertices={int(num_vertices.min())} max_vertices={int(num_vertices.max())}  "
        f"score: [{score.min():.4f}, {score.max():.4f}]  deform: [{deform.min():.4f}, {deform.max():.4f}]"
    )


if __name__ == "__main__":
    main()
