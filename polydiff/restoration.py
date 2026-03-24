"""Three-body restoration analogue helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data.polygon_dataset import PolygonDatasetArrays, PolygonGraphBatch


@dataclass(frozen=True, slots=True)
class RestorationProxyConfig:
    mutant_target_points: tuple[tuple[float, float], ...]
    wild_type_target_points: tuple[tuple[float, float], ...] | None
    ligand_binding_site: tuple[float, float]
    dna_unbound_position: tuple[float, float]
    dna_bound_position: tuple[float, float]
    activation_sigma: float
    contact_beta: float
    dna_binding_threshold: float
    dna_binding_steepness: float
    success_distance: float

    @property
    def target_points(self) -> tuple[tuple[float, float], ...]:
        return self.mutant_target_points

    @property
    def binding_site(self) -> tuple[float, float]:
        return self.ligand_binding_site

    @property
    def mutant_position(self) -> tuple[float, float]:
        return self.dna_unbound_position

    @property
    def wild_type_position(self) -> tuple[float, float]:
        return self.dna_bound_position

    def to_dict(self) -> dict[str, object]:
        return {
            "mutant_target_points": [list(point) for point in self.mutant_target_points],
            "wild_type_target_points": [list(point) for point in self.resolved_wild_type_target_points()],
            "ligand_binding_site": list(self.ligand_binding_site),
            "dna_unbound_position": list(self.dna_unbound_position),
            "dna_bound_position": list(self.dna_bound_position),
            "activation_sigma": float(self.activation_sigma),
            "contact_beta": float(self.contact_beta),
            "dna_binding_threshold": float(self.dna_binding_threshold),
            "dna_binding_steepness": float(self.dna_binding_steepness),
            "success_distance": float(self.success_distance),
        }

    def resolved_wild_type_target_points(self) -> tuple[tuple[float, float], ...]:
        if self.wild_type_target_points is not None:
            return self.wild_type_target_points
        mutant = np.asarray(self.mutant_target_points, dtype=np.float32)
        derived = _derive_wild_type_target_points(
            mutant,
            dna_bound_position=np.asarray(self.dna_bound_position, dtype=np.float32),
            ligand_binding_site=np.asarray(self.ligand_binding_site, dtype=np.float32),
        )
        return tuple((float(point[0]), float(point[1])) for point in derived.tolist())


@dataclass(frozen=True, slots=True)
class RestorationStateTorch:
    contact_point: torch.Tensor
    protein_restoration: torch.Tensor
    protein_points: torch.Tensor
    dna_binding_activation: torch.Tensor
    dna_position: torch.Tensor
    contact_drift: torch.Tensor
    dna_distance: torch.Tensor

    @property
    def activation(self) -> torch.Tensor:
        return self.protein_restoration

    @property
    def centering_drift(self) -> torch.Tensor:
        return self.contact_drift

    @property
    def secondary_position(self) -> torch.Tensor:
        return self.dna_position

    @property
    def secondary_body_distance(self) -> torch.Tensor:
        return self.dna_distance


@dataclass(frozen=True, slots=True)
class RestorationStateNumpy:
    contact_point: np.ndarray
    protein_restoration: np.ndarray
    protein_points: np.ndarray
    dna_binding_activation: np.ndarray
    dna_position: np.ndarray
    contact_drift: np.ndarray
    dna_distance: np.ndarray

    @property
    def activation(self) -> np.ndarray:
        return self.protein_restoration

    @property
    def centering_drift(self) -> np.ndarray:
        return self.contact_drift

    @property
    def secondary_position(self) -> np.ndarray:
        return self.dna_position

    @property
    def secondary_body_distance(self) -> np.ndarray:
        return self.dna_distance


@dataclass(frozen=True, slots=True)
class RestorationAnimationOverlay:
    mutant_target_points: np.ndarray
    wild_type_target_points: np.ndarray
    protein_points: np.ndarray
    ligand_binding_site: np.ndarray
    dna_unbound_position: np.ndarray
    dna_bound_position: np.ndarray
    contact_points: np.ndarray
    dna_positions: np.ndarray
    protein_restoration: np.ndarray
    dna_binding_activation: np.ndarray
    contact_drift: np.ndarray
    dna_distance: np.ndarray

    @property
    def target_points(self) -> np.ndarray:
        return self.protein_points

    @property
    def binding_site(self) -> np.ndarray:
        return self.ligand_binding_site

    @property
    def mutant_position(self) -> np.ndarray:
        return self.dna_unbound_position

    @property
    def wild_type_position(self) -> np.ndarray:
        return self.dna_bound_position

    @property
    def secondary_positions(self) -> np.ndarray:
        return self.dna_positions

    @property
    def activation(self) -> np.ndarray:
        return self.protein_restoration

    @property
    def centering_drift(self) -> np.ndarray:
        return self.contact_drift

    @property
    def secondary_body_distance(self) -> np.ndarray:
        return self.dna_distance

    def select_frames(self, frame_indices: np.ndarray) -> RestorationAnimationOverlay:
        indices = np.asarray(frame_indices, dtype=np.int64)
        return RestorationAnimationOverlay(
            mutant_target_points=self.mutant_target_points,
            wild_type_target_points=self.wild_type_target_points,
            protein_points=self.protein_points[indices],
            ligand_binding_site=self.ligand_binding_site,
            dna_unbound_position=self.dna_unbound_position,
            dna_bound_position=self.dna_bound_position,
            contact_points=self.contact_points[indices],
            dna_positions=self.dna_positions[indices],
            protein_restoration=self.protein_restoration[indices],
            dna_binding_activation=self.dna_binding_activation[indices],
            contact_drift=self.contact_drift[indices],
            dna_distance=self.dna_distance[indices],
        )


def _numpy_point(point: tuple[float, float]) -> np.ndarray:
    return np.asarray(point, dtype=np.float32)


def _torch_point(point: tuple[float, float], *, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(point, dtype=like.dtype, device=like.device)


def _normalize_numpy(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def _normalize_torch(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)


def _derive_wild_type_target_points(
    mutant_target_points: np.ndarray,
    *,
    dna_bound_position: np.ndarray,
    ligand_binding_site: np.ndarray,
) -> np.ndarray:
    centroid = mutant_target_points.mean(axis=0, keepdims=True)
    offsets = mutant_target_points - centroid
    radial_dir = _normalize_numpy(offsets)

    dna_axis = _normalize_numpy((dna_bound_position.reshape(1, 2) - centroid).astype(np.float32))[0]
    ligand_axis = _normalize_numpy((ligand_binding_site.reshape(1, 2) - centroid).astype(np.float32))[0]
    dna_face = np.clip(radial_dir @ dna_axis, 0.0, 1.0)
    ligand_face = np.clip(radial_dir @ ligand_axis, 0.0, 1.0)
    tangent = np.asarray([-dna_axis[1], dna_axis[0]], dtype=np.float32)

    radial_scale = (0.18 * dna_face - 0.06 * ligand_face).reshape(-1, 1)
    tangential_shift = (0.08 * dna_face - 0.03 * ligand_face).reshape(-1, 1) * tangent.reshape(1, 2)
    return mutant_target_points + radial_scale * offsets + tangential_shift


def _resolved_wild_type_target_points_numpy(config: RestorationProxyConfig) -> np.ndarray:
    return np.asarray(config.resolved_wild_type_target_points(), dtype=np.float32).reshape(-1, 2)


def _resolved_wild_type_target_points_torch(config: RestorationProxyConfig, *, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(config.resolved_wild_type_target_points(), dtype=like.dtype, device=like.device)


def _scene_anchor_numpy(config: RestorationProxyConfig) -> np.ndarray:
    return np.asarray(config.mutant_target_points, dtype=np.float32).mean(axis=0)


def _scene_anchor_torch(config: RestorationProxyConfig, *, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(config.mutant_target_points, dtype=like.dtype, device=like.device).mean(dim=0)


def restoration_scene_coords_torch_dense(
    coords: torch.Tensor,
    config: RestorationProxyConfig,
) -> torch.Tensor:
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (batch, n_vertices, 2), got {tuple(coords.shape)}")
    scene_anchor = _scene_anchor_torch(config, like=coords).view(1, 1, 2)
    return coords - coords.mean(dim=1, keepdim=True) + scene_anchor


def restoration_scene_coords_torch_graph(
    coords: torch.Tensor,
    graph_batch: PolygonGraphBatch,
    config: RestorationProxyConfig,
) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape != (graph_batch.total_vertices, 2):
        raise ValueError(
            f"coords must have shape ({graph_batch.total_vertices}, 2), got {tuple(coords.shape)}"
        )
    graph_means = torch.zeros((graph_batch.batch_size, 2), dtype=coords.dtype, device=coords.device)
    graph_means.index_add_(0, graph_batch.graph_index, coords)
    graph_means = graph_means / graph_batch.num_vertices.to(coords.dtype).unsqueeze(1).clamp_min(1.0)
    scene_anchor = _scene_anchor_torch(config, like=coords).view(1, 2)
    return coords - graph_means.index_select(0, graph_batch.graph_index) + scene_anchor


def restoration_scene_coords_numpy(
    coords: np.ndarray,
    config: RestorationProxyConfig,
) -> np.ndarray:
    coords_array = np.asarray(coords, dtype=np.float32)
    squeeze = False
    if coords_array.ndim == 2:
        coords_array = coords_array[None, :, :]
        squeeze = True
    if coords_array.ndim != 3 or coords_array.shape[-1] != 2:
        raise ValueError(f"coords must have shape (batch, n_vertices, 2), got {coords_array.shape}")

    scene_anchor = _scene_anchor_numpy(config).reshape(1, 1, 2)
    scene_coords = coords_array - coords_array.mean(axis=1, keepdims=True) + scene_anchor
    return scene_coords[0] if squeeze else scene_coords


def _soft_contact_weights_dense(
    coords: torch.Tensor,
    ligand_binding_site: torch.Tensor,
    *,
    contact_beta: float,
) -> torch.Tensor:
    distance_sq = (coords - ligand_binding_site.view(1, 1, 2)).square().sum(dim=-1)
    return torch.softmax(-float(contact_beta) * distance_sq, dim=1)


def _soft_contact_weights_graph(
    coords: torch.Tensor,
    graph_batch: PolygonGraphBatch,
    ligand_binding_site: torch.Tensor,
    *,
    contact_beta: float,
) -> torch.Tensor:
    distance_sq = (coords - ligand_binding_site.view(1, 2)).square().sum(dim=-1)
    logits = -float(contact_beta) * distance_sq
    max_per_graph = torch.segment_reduce(logits, reduce="max", lengths=graph_batch.num_vertices)
    stabilized = logits - max_per_graph.repeat_interleave(graph_batch.num_vertices)
    numerators = stabilized.exp()
    denominators = torch.segment_reduce(numerators, reduce="sum", lengths=graph_batch.num_vertices).clamp_min(1e-12)
    return numerators / denominators.repeat_interleave(graph_batch.num_vertices)


def _protein_restoration_from_contact_torch(
    contact_point: torch.Tensor,
    ligand_binding_site: torch.Tensor,
    *,
    activation_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    contact_drift = torch.linalg.norm(contact_point - ligand_binding_site.view(1, 2), dim=1)
    protein_restoration = torch.exp(-0.5 * contact_drift.square() / (float(activation_sigma) ** 2))
    return protein_restoration, contact_drift


def _protein_restoration_from_contact_numpy(
    contact_point: np.ndarray,
    ligand_binding_site: np.ndarray,
    *,
    activation_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    contact_drift = np.linalg.norm(contact_point - ligand_binding_site.reshape(1, 2), axis=1)
    protein_restoration = np.exp(-0.5 * contact_drift**2 / (float(activation_sigma) ** 2))
    return protein_restoration.astype(np.float32), contact_drift.astype(np.float32)


def _dna_binding_from_restoration_torch(
    protein_restoration: torch.Tensor,
    *,
    threshold: float,
    steepness: float,
) -> torch.Tensor:
    return torch.sigmoid((protein_restoration - float(threshold)) * float(steepness))


def _dna_binding_from_restoration_numpy(
    protein_restoration: np.ndarray,
    *,
    threshold: float,
    steepness: float,
) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-(protein_restoration - float(threshold)) * float(steepness)))).astype(np.float32)


def restoration_state_torch_dense(
    coords: torch.Tensor,
    config: RestorationProxyConfig,
) -> RestorationStateTorch:
    scene_coords = restoration_scene_coords_torch_dense(coords, config)
    ligand_binding_site = _torch_point(config.ligand_binding_site, like=scene_coords)
    dna_unbound_position = _torch_point(config.dna_unbound_position, like=scene_coords)
    dna_bound_position = _torch_point(config.dna_bound_position, like=scene_coords)
    mutant_target_points = torch.tensor(config.mutant_target_points, dtype=scene_coords.dtype, device=scene_coords.device)
    wild_type_target_points = _resolved_wild_type_target_points_torch(config, like=scene_coords)

    weights = _soft_contact_weights_dense(scene_coords, ligand_binding_site, contact_beta=config.contact_beta)
    contact_point = (weights.unsqueeze(-1) * scene_coords).sum(dim=1)
    protein_restoration, contact_drift = _protein_restoration_from_contact_torch(
        contact_point,
        ligand_binding_site,
        activation_sigma=config.activation_sigma,
    )
    dna_binding_activation = _dna_binding_from_restoration_torch(
        protein_restoration,
        threshold=config.dna_binding_threshold,
        steepness=config.dna_binding_steepness,
    )
    protein_points = mutant_target_points.unsqueeze(0) + protein_restoration.view(-1, 1, 1) * (
        wild_type_target_points - mutant_target_points
    ).unsqueeze(0)
    dna_position = dna_unbound_position.view(1, 2) + dna_binding_activation.unsqueeze(1) * (
        dna_bound_position - dna_unbound_position
    ).view(1, 2)
    dna_distance = torch.linalg.norm(dna_position - dna_bound_position.view(1, 2), dim=1)
    return RestorationStateTorch(
        contact_point=contact_point,
        protein_restoration=protein_restoration,
        protein_points=protein_points,
        dna_binding_activation=dna_binding_activation,
        dna_position=dna_position,
        contact_drift=contact_drift,
        dna_distance=dna_distance,
    )


def restoration_state_torch_graph(
    coords: torch.Tensor,
    graph_batch: PolygonGraphBatch,
    config: RestorationProxyConfig,
) -> RestorationStateTorch:
    scene_coords = restoration_scene_coords_torch_graph(coords, graph_batch, config)
    ligand_binding_site = _torch_point(config.ligand_binding_site, like=scene_coords)
    dna_unbound_position = _torch_point(config.dna_unbound_position, like=scene_coords)
    dna_bound_position = _torch_point(config.dna_bound_position, like=scene_coords)
    mutant_target_points = torch.tensor(config.mutant_target_points, dtype=scene_coords.dtype, device=scene_coords.device)
    wild_type_target_points = _resolved_wild_type_target_points_torch(config, like=scene_coords)

    weights = _soft_contact_weights_graph(
        scene_coords,
        graph_batch,
        ligand_binding_site,
        contact_beta=config.contact_beta,
    )
    weighted_coords = weights.unsqueeze(1) * scene_coords
    contact_point = torch.zeros((graph_batch.batch_size, 2), dtype=scene_coords.dtype, device=scene_coords.device)
    contact_point.index_add_(0, graph_batch.graph_index, weighted_coords)

    protein_restoration, contact_drift = _protein_restoration_from_contact_torch(
        contact_point,
        ligand_binding_site,
        activation_sigma=config.activation_sigma,
    )
    dna_binding_activation = _dna_binding_from_restoration_torch(
        protein_restoration,
        threshold=config.dna_binding_threshold,
        steepness=config.dna_binding_steepness,
    )
    protein_points = mutant_target_points.unsqueeze(0) + protein_restoration.view(-1, 1, 1) * (
        wild_type_target_points - mutant_target_points
    ).unsqueeze(0)
    dna_position = dna_unbound_position.view(1, 2) + dna_binding_activation.unsqueeze(1) * (
        dna_bound_position - dna_unbound_position
    ).view(1, 2)
    dna_distance = torch.linalg.norm(dna_position - dna_bound_position.view(1, 2), dim=1)
    return RestorationStateTorch(
        contact_point=contact_point,
        protein_restoration=protein_restoration,
        protein_points=protein_points,
        dna_binding_activation=dna_binding_activation,
        dna_position=dna_position,
        contact_drift=contact_drift,
        dna_distance=dna_distance,
    )


def restoration_state_numpy(
    coords: np.ndarray,
    config: RestorationProxyConfig,
) -> RestorationStateNumpy:
    scene_coords = restoration_scene_coords_numpy(coords, config)
    squeeze = False
    if scene_coords.ndim == 2:
        scene_coords = scene_coords[None, :, :]
        squeeze = True
    if scene_coords.ndim != 3 or scene_coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (batch, n_vertices, 2), got {scene_coords.shape}")

    ligand_binding_site = _numpy_point(config.ligand_binding_site)
    dna_unbound_position = _numpy_point(config.dna_unbound_position)
    dna_bound_position = _numpy_point(config.dna_bound_position)
    mutant_target_points = np.asarray(config.mutant_target_points, dtype=np.float32).reshape(1, -1, 2)
    wild_type_target_points = _resolved_wild_type_target_points_numpy(config).reshape(1, -1, 2)

    distance_sq = np.sum((scene_coords - ligand_binding_site.reshape(1, 1, 2)) ** 2, axis=-1)
    logits = -float(config.contact_beta) * distance_sq
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    contact_point = np.sum(weights[:, :, None] * scene_coords, axis=1).astype(np.float32)

    protein_restoration, contact_drift = _protein_restoration_from_contact_numpy(
        contact_point,
        ligand_binding_site,
        activation_sigma=config.activation_sigma,
    )
    dna_binding_activation = _dna_binding_from_restoration_numpy(
        protein_restoration,
        threshold=config.dna_binding_threshold,
        steepness=config.dna_binding_steepness,
    )
    protein_points = mutant_target_points + protein_restoration.reshape(-1, 1, 1) * (
        wild_type_target_points - mutant_target_points
    )
    dna_position = dna_unbound_position.reshape(1, 2) + dna_binding_activation.reshape(-1, 1) * (
        dna_bound_position - dna_unbound_position
    ).reshape(1, 2)
    dna_distance = np.linalg.norm(dna_position - dna_bound_position.reshape(1, 2), axis=1).astype(np.float32)

    if squeeze:
        return RestorationStateNumpy(
            contact_point=contact_point[0],
            protein_restoration=protein_restoration[0],
            protein_points=protein_points[0],
            dna_binding_activation=dna_binding_activation[0],
            dna_position=dna_position[0],
            contact_drift=contact_drift[0],
            dna_distance=dna_distance[0],
        )
    return RestorationStateNumpy(
        contact_point=contact_point,
        protein_restoration=protein_restoration,
        protein_points=protein_points.astype(np.float32),
        dna_binding_activation=dna_binding_activation,
        dna_position=dna_position.astype(np.float32),
        contact_drift=contact_drift,
        dna_distance=dna_distance,
    )


def _as_polygon_dataset(
    coords: np.ndarray | PolygonDatasetArrays,
    *,
    num_vertices: np.ndarray | None = None,
) -> PolygonDatasetArrays:
    if isinstance(coords, PolygonDatasetArrays):
        if num_vertices is not None:
            raise ValueError("num_vertices must not be provided when coords is already a PolygonDatasetArrays object")
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


def summarize_restoration_dataset(
    coords: np.ndarray | PolygonDatasetArrays,
    config: RestorationProxyConfig,
    *,
    num_vertices: np.ndarray | None = None,
) -> dict[str, object]:
    dataset = _as_polygon_dataset(coords, num_vertices=num_vertices)
    if dataset.num_polygons == 0:
        raise ValueError("cannot summarize restoration on an empty polygon dataset")

    states = [restoration_state_numpy(polygon, config) for polygon in dataset.iter_polygons()]
    protein_restoration = np.asarray([float(state.protein_restoration) for state in states], dtype=np.float64)
    dna_binding_activation = np.asarray([float(state.dna_binding_activation) for state in states], dtype=np.float64)
    contact_drift = np.asarray([float(state.contact_drift) for state in states], dtype=np.float64)
    dna_distance = np.asarray([float(state.dna_distance) for state in states], dtype=np.float64)
    dna_positions = np.asarray([np.asarray(state.dna_position, dtype=np.float64) for state in states], dtype=np.float64)

    dna_unbound_position = _numpy_point(config.dna_unbound_position).astype(np.float64)
    dna_bound_position = _numpy_point(config.dna_bound_position).astype(np.float64)
    dna_unbound_distance = float(np.linalg.norm(dna_unbound_position - dna_bound_position))

    summary = {
        "protein_restoration_mean": float(protein_restoration.mean()),
        "protein_restoration_std": float(protein_restoration.std()),
        "protein_restoration_p05": float(np.quantile(protein_restoration, 0.05)),
        "protein_restoration_p50": float(np.quantile(protein_restoration, 0.50)),
        "protein_restoration_p95": float(np.quantile(protein_restoration, 0.95)),
        "dna_binding_activation_mean": float(dna_binding_activation.mean()),
        "dna_binding_activation_std": float(dna_binding_activation.std()),
        "contact_drift_mean": float(contact_drift.mean()),
        "contact_drift_std": float(contact_drift.std()),
        "dna_distance_mean": float(dna_distance.mean()),
        "dna_distance_std": float(dna_distance.std()),
        "dna_distance_p05": float(np.quantile(dna_distance, 0.05)),
        "dna_distance_p50": float(np.quantile(dna_distance, 0.50)),
        "dna_distance_p95": float(np.quantile(dna_distance, 0.95)),
        "dna_position_mean": dna_positions.mean(axis=0).astype(np.float32),
        "dna_position_std": dna_positions.std(axis=0).astype(np.float32),
        "restoration_success_rate": float(np.mean(dna_distance <= float(config.success_distance))),
        "dna_unbound_distance": dna_unbound_distance,
        "success_distance": float(config.success_distance),
        # Backward-compatible aliases.
        "activation_mean": float(protein_restoration.mean()),
        "activation_std": float(protein_restoration.std()),
        "activation_p05": float(np.quantile(protein_restoration, 0.05)),
        "activation_p50": float(np.quantile(protein_restoration, 0.50)),
        "activation_p95": float(np.quantile(protein_restoration, 0.95)),
        "centering_drift_mean": float(contact_drift.mean()),
        "centering_drift_std": float(contact_drift.std()),
        "secondary_body_distance_mean": float(dna_distance.mean()),
        "secondary_body_distance_std": float(dna_distance.std()),
        "secondary_body_distance_p05": float(np.quantile(dna_distance, 0.05)),
        "secondary_body_distance_p50": float(np.quantile(dna_distance, 0.50)),
        "secondary_body_distance_p95": float(np.quantile(dna_distance, 0.95)),
        "secondary_body_position_mean": dna_positions.mean(axis=0).astype(np.float32),
        "secondary_body_position_std": dna_positions.std(axis=0).astype(np.float32),
        "mutant_secondary_body_distance": dna_unbound_distance,
    }
    return summary


def format_restoration_summary(summary: dict[str, object]) -> str:
    dna_mean = np.asarray(summary["dna_position_mean"], dtype=np.float64)
    return (
        f"protein_restore={float(summary['protein_restoration_mean']):.4f}±{float(summary['protein_restoration_std']):.4f} "
        f"dna_bind={float(summary['dna_binding_activation_mean']):.4f}±{float(summary['dna_binding_activation_std']):.4f} "
        f"contact_drift={float(summary['contact_drift_mean']):.4f} "
        f"dna_distance={float(summary['dna_distance_mean']):.4f} "
        f"success_rate={float(summary['restoration_success_rate']):.4f} "
        f"dna_mean=({dna_mean[0]:+.3f}, {dna_mean[1]:+.3f})"
    )


def build_restoration_animation_overlay(
    coords: np.ndarray,
    config: RestorationProxyConfig,
) -> RestorationAnimationOverlay:
    state = restoration_state_numpy(coords, config)
    if not isinstance(state.protein_restoration, np.ndarray):
        raise ValueError("animation overlay requires a full trajectory with shape (frames, n_vertices, 2)")

    return RestorationAnimationOverlay(
        mutant_target_points=np.asarray(config.mutant_target_points, dtype=np.float32).reshape(-1, 2),
        wild_type_target_points=_resolved_wild_type_target_points_numpy(config),
        protein_points=np.asarray(state.protein_points, dtype=np.float32),
        ligand_binding_site=_numpy_point(config.ligand_binding_site),
        dna_unbound_position=_numpy_point(config.dna_unbound_position),
        dna_bound_position=_numpy_point(config.dna_bound_position),
        contact_points=np.asarray(state.contact_point, dtype=np.float32),
        dna_positions=np.asarray(state.dna_position, dtype=np.float32),
        protein_restoration=np.asarray(state.protein_restoration, dtype=np.float32),
        dna_binding_activation=np.asarray(state.dna_binding_activation, dtype=np.float32),
        contact_drift=np.asarray(state.contact_drift, dtype=np.float32),
        dna_distance=np.asarray(state.dna_distance, dtype=np.float32),
    )


class RestorationTrajectoryRecorder:
    """Track restoration metrics across a reverse diffusion trajectory."""

    def __init__(self, config: RestorationProxyConfig) -> None:
        self.config = config
        self._diffusion_steps: list[int] = []
        self._protein_restoration_mean: list[float] = []
        self._protein_restoration_std: list[float] = []
        self._dna_binding_activation_mean: list[float] = []
        self._dna_binding_activation_std: list[float] = []
        self._contact_drift_mean: list[float] = []
        self._contact_drift_std: list[float] = []
        self._dna_distance_mean: list[float] = []
        self._dna_distance_std: list[float] = []
        self._dna_position_mean: list[list[float]] = []

    def observe(
        self,
        coords: torch.Tensor,
        step: int,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> None:
        if graph_batch is None:
            if coords.ndim != 2 or coords.shape[1] % 2 != 0:
                raise ValueError(f"dense coords must have shape (batch, data_dim), got {tuple(coords.shape)}")
            state = restoration_state_torch_dense(
                coords.reshape(coords.shape[0], coords.shape[1] // 2, 2),
                self.config,
            )
        else:
            state = restoration_state_torch_graph(coords, graph_batch, self.config)

        dna_mean = state.dna_position.mean(dim=0)
        self._diffusion_steps.append(int(step))
        self._protein_restoration_mean.append(float(state.protein_restoration.mean().item()))
        self._protein_restoration_std.append(float(state.protein_restoration.std(unbiased=False).item()))
        self._dna_binding_activation_mean.append(float(state.dna_binding_activation.mean().item()))
        self._dna_binding_activation_std.append(float(state.dna_binding_activation.std(unbiased=False).item()))
        self._contact_drift_mean.append(float(state.contact_drift.mean().item()))
        self._contact_drift_std.append(float(state.contact_drift.std(unbiased=False).item()))
        self._dna_distance_mean.append(float(state.dna_distance.mean().item()))
        self._dna_distance_std.append(float(state.dna_distance.std(unbiased=False).item()))
        self._dna_position_mean.append([float(dna_mean[0].item()), float(dna_mean[1].item())])

    def to_dict(self) -> dict[str, object]:
        payload = {
            "diffusion_step": list(self._diffusion_steps),
            "protein_restoration_mean": list(self._protein_restoration_mean),
            "protein_restoration_std": list(self._protein_restoration_std),
            "dna_binding_activation_mean": list(self._dna_binding_activation_mean),
            "dna_binding_activation_std": list(self._dna_binding_activation_std),
            "contact_drift_mean": list(self._contact_drift_mean),
            "contact_drift_std": list(self._contact_drift_std),
            "dna_distance_mean": list(self._dna_distance_mean),
            "dna_distance_std": list(self._dna_distance_std),
            "dna_position_mean": list(self._dna_position_mean),
        }
        payload["activation_mean"] = list(self._protein_restoration_mean)
        payload["activation_std"] = list(self._protein_restoration_std)
        payload["centering_drift_mean"] = list(self._contact_drift_mean)
        payload["centering_drift_std"] = list(self._contact_drift_std)
        payload["secondary_body_distance_mean"] = list(self._dna_distance_mean)
        payload["secondary_body_distance_std"] = list(self._dna_distance_std)
        payload["secondary_body_position_mean"] = list(self._dna_position_mean)
        return payload
