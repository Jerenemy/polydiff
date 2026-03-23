"""Differentiable regularity score for polygon guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RegularityTensors:
    edge_cv: torch.Tensor
    angle_cv: torch.Tensor
    radius_cv: torch.Tensor
    score: torch.Tensor


def _as_batched_polygons(xy: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if xy.ndim == 2 and xy.shape[-1] == 2:
        return xy.unsqueeze(0), True
    if xy.ndim == 3 and xy.shape[-1] == 2:
        return xy, False
    raise ValueError(f"xy must have shape (n, 2) or (batch, n, 2), got {tuple(xy.shape)}")


def center_and_normalize_scale_rms_torch(xy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xy_batched, squeeze = _as_batched_polygons(xy)
    centered = xy_batched - xy_batched.mean(dim=1, keepdim=True)
    mean_radius_sq = centered.square().sum(dim=-1).mean(dim=1, keepdim=True)
    rms = torch.sqrt(mean_radius_sq + eps).unsqueeze(-1)
    normalized = centered / rms
    return normalized[0] if squeeze else normalized


def edge_lengths_torch(xy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xy_batched, squeeze = _as_batched_polygons(xy)
    diffs = torch.roll(xy_batched, shifts=-1, dims=1) - xy_batched
    lengths = torch.sqrt(diffs.square().sum(dim=-1) + eps)
    return lengths[0] if squeeze else lengths


def signed_polygon_area_torch(xy: torch.Tensor) -> torch.Tensor:
    xy_batched, squeeze = _as_batched_polygons(xy)
    x = xy_batched[..., 0]
    y = xy_batched[..., 1]
    area = 0.5 * (x * torch.roll(y, shifts=-1, dims=1) - y * torch.roll(x, shifts=-1, dims=1)).sum(dim=1)
    return area[0] if squeeze else area


def smooth_polygon_area_torch(xy: torch.Tensor, eps: float = 1e-8, absolute: bool = True) -> torch.Tensor:
    area = signed_polygon_area_torch(xy)
    if absolute:
        return torch.sqrt(area.square() + eps)
    return area


def interior_angles_torch(xy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xy_batched, squeeze = _as_batched_polygons(xy)

    prev_vec = torch.roll(xy_batched, shifts=1, dims=1) - xy_batched
    next_vec = torch.roll(xy_batched, shifts=-1, dims=1) - xy_batched

    prev_unit = prev_vec / prev_vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    next_unit = next_vec / next_vec.norm(dim=-1, keepdim=True).clamp_min(eps)

    dot = (prev_unit * next_unit).sum(dim=-1)
    cross_z = prev_unit[..., 0] * next_unit[..., 1] - prev_unit[..., 1] * next_unit[..., 0]

    # Smooth absolute value keeps the angle objective differentiable around 0.
    cross_mag = torch.sqrt(cross_z.square() + eps)
    angles = torch.atan2(cross_mag, dot)
    return angles[0] if squeeze else angles


def _coefficient_of_variation(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = values.mean(dim=-1)
    std = torch.sqrt((values - mean.unsqueeze(-1)).square().mean(dim=-1) + eps)
    return std / (mean + eps)


def regularity_score_torch(
    xy: torch.Tensor,
    *,
    alpha: float = 8.0,
    beta: float = 5.0,
    gamma: float = 4.0,
    eps: float = 1e-8,
    center_and_normalize: bool = True,
) -> RegularityTensors:
    xy_batched, squeeze = _as_batched_polygons(xy)
    if center_and_normalize:
        xy_batched = center_and_normalize_scale_rms_torch(xy_batched, eps=eps)

    edge_lengths = edge_lengths_torch(xy_batched, eps=eps)
    angles = interior_angles_torch(xy_batched, eps=eps)
    radii = torch.sqrt(xy_batched.square().sum(dim=-1) + eps)

    edge_cv = _coefficient_of_variation(edge_lengths, eps=eps)
    angle_cv = _coefficient_of_variation(angles, eps=eps)
    radius_cv = _coefficient_of_variation(radii, eps=eps)
    score = torch.exp(
        -float(alpha) * edge_cv
        - float(beta) * angle_cv
        - float(gamma) * radius_cv
    )

    if squeeze:
        return RegularityTensors(
            edge_cv=edge_cv[0],
            angle_cv=angle_cv[0],
            radius_cv=radius_cv[0],
            score=score[0],
        )
    return RegularityTensors(
        edge_cv=edge_cv,
        angle_cv=angle_cv,
        radius_cv=radius_cv,
        score=score,
    )
