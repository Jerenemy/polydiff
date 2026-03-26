"""Analytic pocket-fit reward for polygon-conditioning studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch

from ..data.gen_polygons import is_self_intersecting
from ..models.regularity_torch import smooth_polygon_area_torch


@dataclass(frozen=True, slots=True)
class PocketFitConfig:
    target_area_ratio: float = 0.22
    target_clearance: float = 0.10
    inside_temperature: float = 0.06
    outside_weight: float = 3.5
    area_weight: float = 8.0
    clearance_weight: float = 4.0
    sample_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    success_threshold: float = 0.55

    def to_dict(self) -> dict[str, object]:
        return {
            "target_area_ratio": float(self.target_area_ratio),
            "target_clearance": float(self.target_clearance),
            "inside_temperature": float(self.inside_temperature),
            "outside_weight": float(self.outside_weight),
            "area_weight": float(self.area_weight),
            "clearance_weight": float(self.clearance_weight),
            "sample_fractions": [float(value) for value in self.sample_fractions],
            "success_threshold": float(self.success_threshold),
        }


@dataclass(frozen=True, slots=True)
class PocketFitStateTorch:
    fit_score: torch.Tensor
    inside_fraction: torch.Tensor
    outside_penalty: torch.Tensor
    area_ratio: torch.Tensor
    area_ratio_penalty: torch.Tensor
    clearance_mean: torch.Tensor
    clearance_penalty: torch.Tensor


@dataclass(frozen=True, slots=True)
class PocketFitStateNumpy:
    fit_score: np.ndarray
    inside_fraction: np.ndarray
    outside_penalty: np.ndarray
    area_ratio: np.ndarray
    area_ratio_penalty: np.ndarray
    clearance_mean: np.ndarray
    clearance_penalty: np.ndarray
    self_intersection: np.ndarray


def rotate_polygon_numpy(xy: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.asarray(
        [
            [math.cos(float(angle)), -math.sin(float(angle))],
            [math.sin(float(angle)), math.cos(float(angle))],
        ],
        dtype=np.float32,
    )
    return np.asarray(xy, dtype=np.float32) @ rotation.T


def rotate_polygon_torch(xy: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    angle_tensor = torch.as_tensor(angle, dtype=xy.dtype, device=xy.device)
    cos_angle = torch.cos(angle_tensor)
    sin_angle = torch.sin(angle_tensor)
    if angle_tensor.ndim == 0:
        rotation = torch.stack(
            [
                torch.stack([cos_angle, -sin_angle]),
                torch.stack([sin_angle, cos_angle]),
            ],
            dim=0,
        )
        return xy @ rotation.transpose(0, 1)
    rotation = torch.stack(
        [
            torch.stack([cos_angle, -sin_angle], dim=-1),
            torch.stack([sin_angle, cos_angle], dim=-1),
        ],
        dim=-2,
    )
    return torch.einsum("...nd,...dc->...nc", xy, rotation.transpose(-1, -2))


def is_strictly_convex_polygon(xy: np.ndarray, *, tol: float = 1e-6) -> bool:
    coords = np.asarray(xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 3:
        return False
    cross_values: list[float] = []
    for index in range(coords.shape[0]):
        a = coords[index]
        b = coords[(index + 1) % coords.shape[0]]
        c = coords[(index + 2) % coords.shape[0]]
        ab = b - a
        bc = c - b
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        cross_values.append(cross)
    cross_arr = np.asarray(cross_values, dtype=np.float64)
    return bool(np.all(cross_arr > tol) or np.all(cross_arr < -tol))


def pocket_fit_state_torch(
    ligand_xy: torch.Tensor,
    pocket_xy: torch.Tensor,
    config: PocketFitConfig,
) -> PocketFitStateTorch:
    ligand_batched, squeeze = _as_batched_polygons_torch(ligand_xy)
    pocket = _as_polygon_torch(pocket_xy)
    samples = _edge_samples_torch(ligand_batched, config.sample_fractions)
    margins = _convex_margins_torch(samples, pocket)
    inside_fraction = torch.sigmoid(margins / float(config.inside_temperature)).mean(dim=1)
    outside_penalty = torch.relu(-margins).mean(dim=1)
    clearance_mean = margins.clamp_min(0.0).mean(dim=1)

    ligand_area = smooth_polygon_area_torch(ligand_batched, absolute=True)
    pocket_area = smooth_polygon_area_torch(pocket, absolute=True).reshape(1).expand_as(ligand_area)
    area_ratio = ligand_area / pocket_area.clamp_min(1e-8)
    area_ratio_penalty = (area_ratio - float(config.target_area_ratio)).square()
    clearance_penalty = (clearance_mean - float(config.target_clearance)).square()

    fit_score = (
        inside_fraction
        - float(config.outside_weight) * outside_penalty
        - float(config.area_weight) * area_ratio_penalty
        - float(config.clearance_weight) * clearance_penalty
    )
    if squeeze:
        return PocketFitStateTorch(
            fit_score=fit_score[0],
            inside_fraction=inside_fraction[0],
            outside_penalty=outside_penalty[0],
            area_ratio=area_ratio[0],
            area_ratio_penalty=area_ratio_penalty[0],
            clearance_mean=clearance_mean[0],
            clearance_penalty=clearance_penalty[0],
        )
    return PocketFitStateTorch(
        fit_score=fit_score,
        inside_fraction=inside_fraction,
        outside_penalty=outside_penalty,
        area_ratio=area_ratio,
        area_ratio_penalty=area_ratio_penalty,
        clearance_mean=clearance_mean,
        clearance_penalty=clearance_penalty,
    )


def pocket_fit_state_numpy(
    ligand_xy: np.ndarray,
    pocket_xy: np.ndarray,
    config: PocketFitConfig,
) -> PocketFitStateNumpy:
    ligand_batched, squeeze = _as_batched_polygons_numpy(ligand_xy)
    pocket = _as_polygon_numpy(pocket_xy)
    samples = _edge_samples_numpy(ligand_batched, config.sample_fractions)
    margins = _convex_margins_numpy(samples, pocket)
    inside_fraction = 1.0 / (1.0 + np.exp(-margins / float(config.inside_temperature)))
    inside_fraction = inside_fraction.mean(axis=1)
    outside_penalty = np.maximum(-margins, 0.0).mean(axis=1)
    clearance_mean = np.maximum(margins, 0.0).mean(axis=1)
    ligand_area = np.asarray([_polygon_area_numpy(polygon) for polygon in ligand_batched], dtype=np.float32)
    pocket_area = np.float32(_polygon_area_numpy(pocket))
    area_ratio = ligand_area / max(float(pocket_area), 1e-8)
    area_ratio_penalty = (area_ratio - float(config.target_area_ratio)) ** 2
    clearance_penalty = (clearance_mean - float(config.target_clearance)) ** 2
    fit_score = (
        inside_fraction
        - float(config.outside_weight) * outside_penalty
        - float(config.area_weight) * area_ratio_penalty
        - float(config.clearance_weight) * clearance_penalty
    ).astype(np.float32)
    self_intersection = np.asarray(
        [float(is_self_intersecting(polygon)) for polygon in ligand_batched],
        dtype=np.float32,
    )
    if squeeze:
        return PocketFitStateNumpy(
            fit_score=fit_score[:1],
            inside_fraction=inside_fraction[:1].astype(np.float32),
            outside_penalty=outside_penalty[:1].astype(np.float32),
            area_ratio=area_ratio[:1].astype(np.float32),
            area_ratio_penalty=area_ratio_penalty[:1].astype(np.float32),
            clearance_mean=clearance_mean[:1].astype(np.float32),
            clearance_penalty=clearance_penalty[:1].astype(np.float32),
            self_intersection=self_intersection[:1],
        )
    return PocketFitStateNumpy(
        fit_score=fit_score,
        inside_fraction=inside_fraction.astype(np.float32),
        outside_penalty=outside_penalty.astype(np.float32),
        area_ratio=area_ratio.astype(np.float32),
        area_ratio_penalty=area_ratio_penalty.astype(np.float32),
        clearance_mean=clearance_mean.astype(np.float32),
        clearance_penalty=clearance_penalty.astype(np.float32),
        self_intersection=self_intersection,
    )


def summarize_pocket_fit(
    ligand_xy: np.ndarray,
    pocket_xy: np.ndarray,
    config: PocketFitConfig,
) -> dict[str, float]:
    state = pocket_fit_state_numpy(ligand_xy, pocket_xy, config)
    fit_score = np.asarray(state.fit_score, dtype=np.float64).reshape(-1)
    inside_fraction = np.asarray(state.inside_fraction, dtype=np.float64).reshape(-1)
    outside_penalty = np.asarray(state.outside_penalty, dtype=np.float64).reshape(-1)
    area_ratio = np.asarray(state.area_ratio, dtype=np.float64).reshape(-1)
    clearance_mean = np.asarray(state.clearance_mean, dtype=np.float64).reshape(-1)
    self_intersection = np.asarray(state.self_intersection, dtype=np.float64).reshape(-1)
    return {
        "fit_score_mean": float(fit_score.mean()),
        "fit_score_std": float(fit_score.std()),
        "fit_score_p95": float(np.quantile(fit_score, 0.95)),
        "fit_success_rate": float(np.mean(fit_score >= float(config.success_threshold))),
        "inside_fraction_mean": float(inside_fraction.mean()),
        "outside_penalty_mean": float(outside_penalty.mean()),
        "area_ratio_mean": float(area_ratio.mean()),
        "clearance_mean": float(clearance_mean.mean()),
        "self_intersection_rate": float(self_intersection.mean()),
    }


def _as_batched_polygons_torch(xy: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if xy.ndim == 2 and xy.shape[-1] == 2:
        return xy.unsqueeze(0), True
    if xy.ndim == 3 and xy.shape[-1] == 2:
        return xy, False
    raise ValueError(f"xy must have shape (n, 2) or (batch, n, 2), got {tuple(xy.shape)}")


def _as_polygon_torch(xy: torch.Tensor) -> torch.Tensor:
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"pocket polygon must have shape (n, 2), got {tuple(xy.shape)}")
    return xy


def _as_batched_polygons_numpy(xy: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return arr[None, ...], True
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr, False
    raise ValueError(f"xy must have shape (n, 2) or (batch, n, 2), got {arr.shape}")


def _as_polygon_numpy(xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"pocket polygon must have shape (n, 2), got {arr.shape}")
    return arr


def _edge_samples_torch(polygons: torch.Tensor, fractions: Sequence[float]) -> torch.Tensor:
    fraction_tensor = torch.as_tensor(tuple(float(value) for value in fractions), dtype=polygons.dtype, device=polygons.device)
    next_points = torch.roll(polygons, shifts=-1, dims=1)
    samples = polygons.unsqueeze(2) * (1.0 - fraction_tensor.view(1, 1, -1, 1))
    samples = samples + next_points.unsqueeze(2) * fraction_tensor.view(1, 1, -1, 1)
    return samples.reshape(polygons.shape[0], polygons.shape[1] * fraction_tensor.shape[0], 2)


def _edge_samples_numpy(polygons: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    fraction_array = np.asarray(tuple(float(value) for value in fractions), dtype=np.float32)
    next_points = np.roll(polygons, shift=-1, axis=1)
    samples = polygons[:, :, None, :] * (1.0 - fraction_array.reshape(1, 1, -1, 1))
    samples = samples + next_points[:, :, None, :] * fraction_array.reshape(1, 1, -1, 1)
    return samples.reshape(polygons.shape[0], polygons.shape[1] * fraction_array.shape[0], 2)


def _convex_margins_torch(points: torch.Tensor, pocket: torch.Tensor) -> torch.Tensor:
    edges = torch.roll(pocket, shifts=-1, dims=0) - pocket
    rel = points.unsqueeze(2) - pocket.view(1, 1, pocket.shape[0], 2)
    cross = edges.view(1, 1, pocket.shape[0], 2)[..., 0] * rel[..., 1] - edges.view(1, 1, pocket.shape[0], 2)[..., 1] * rel[..., 0]
    edge_norm = torch.linalg.norm(edges, dim=1).clamp_min(1e-8).view(1, 1, pocket.shape[0])
    signed_distance = cross / edge_norm
    return signed_distance.min(dim=2).values


def _convex_margins_numpy(points: np.ndarray, pocket: np.ndarray) -> np.ndarray:
    edges = np.roll(pocket, shift=-1, axis=0) - pocket
    rel = points[:, :, None, :] - pocket.reshape(1, 1, pocket.shape[0], 2)
    cross = edges.reshape(1, 1, pocket.shape[0], 2)[..., 0] * rel[..., 1] - edges.reshape(1, 1, pocket.shape[0], 2)[..., 1] * rel[..., 0]
    edge_norm = np.linalg.norm(edges, axis=1).reshape(1, 1, pocket.shape[0]).clip(1e-8, None)
    return (cross / edge_norm).min(axis=2).astype(np.float32)


def _polygon_area_numpy(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))
