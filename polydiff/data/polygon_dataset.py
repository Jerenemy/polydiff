"""Helpers for fixed-size and ragged polygon datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PolygonDatasetArrays:
    coords: np.ndarray
    num_vertices: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=np.float32)
        num_vertices = np.asarray(self.num_vertices, dtype=np.int32).reshape(-1)
        if coords.ndim not in {2, 3} or coords.shape[-1] != 2:
            raise ValueError(
                "coords must have shape (num_polygons, n_vertices, 2) or (total_vertices, 2), "
                f"got {coords.shape}"
            )
        if num_vertices.ndim != 1:
            raise ValueError(f"num_vertices must be 1D, got shape {num_vertices.shape}")
        if np.any(num_vertices < 3):
            raise ValueError("all polygons must have at least 3 vertices")
        if coords.ndim == 3 and coords.shape[0] != num_vertices.shape[0]:
            raise ValueError(
                f"dense coords first dimension {coords.shape[0]} does not match num_vertices length {num_vertices.shape[0]}"
            )
        if coords.ndim == 2 and int(num_vertices.sum()) != int(coords.shape[0]):
            raise ValueError(
                f"ragged coords row count {coords.shape[0]} does not match sum(num_vertices)={int(num_vertices.sum())}"
            )
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "num_vertices", num_vertices)

    @property
    def num_polygons(self) -> int:
        return int(self.num_vertices.shape[0])

    @property
    def max_vertices(self) -> int:
        return int(self.num_vertices.max()) if self.num_polygons > 0 else 0

    @property
    def is_uniform(self) -> bool:
        return bool(self.num_polygons == 0 or np.all(self.num_vertices == self.num_vertices[0]))

    @property
    def is_ragged_storage(self) -> bool:
        return self.coords.ndim == 2

    @property
    def ptr(self) -> np.ndarray:
        ptr = np.zeros(self.num_polygons + 1, dtype=np.int64)
        if self.num_polygons > 0:
            ptr[1:] = np.cumsum(self.num_vertices, dtype=np.int64)
        return ptr

    def polygon(self, index: int) -> np.ndarray:
        index = int(index)
        n = int(self.num_vertices[index])
        if self.coords.ndim == 3:
            return np.asarray(self.coords[index, :n], dtype=np.float32)
        start = int(self.ptr[index])
        end = start + n
        return np.asarray(self.coords[start:end], dtype=np.float32)

    def iter_polygons(self) -> Iterator[np.ndarray]:
        for i in range(self.num_polygons):
            yield self.polygon(i)

    def to_dense(self) -> np.ndarray:
        if self.coords.ndim == 3 and self.is_uniform:
            n = int(self.num_vertices[0]) if self.num_polygons > 0 else 0
            return np.asarray(self.coords[:, :n], dtype=np.float32)
        if not self.is_uniform:
            raise ValueError("cannot convert variable-size polygons to a dense array")
        n = int(self.num_vertices[0]) if self.num_polygons > 0 else 0
        return np.asarray(self.coords, dtype=np.float32).reshape(self.num_polygons, n, 2)


def polygon_dataset_from_npz(npz_data: np.lib.npyio.NpzFile) -> PolygonDatasetArrays:
    coords = np.asarray(npz_data["coords"], dtype=np.float32)
    if "num_vertices" in npz_data:
        num_vertices = np.asarray(npz_data["num_vertices"], dtype=np.int32).reshape(-1)
        return PolygonDatasetArrays(coords=coords, num_vertices=num_vertices)

    if coords.ndim == 3:
        num_vertices = np.full((coords.shape[0],), coords.shape[1], dtype=np.int32)
        return PolygonDatasetArrays(coords=coords, num_vertices=num_vertices)

    if "n" in npz_data:
        n = int(npz_data["n"])
        if coords.ndim != 2:
            raise ValueError(f"ragged coords with scalar n expect shape (total_vertices, 2), got {coords.shape}")
        if n < 3:
            raise ValueError(f"n must be >= 3, got {n}")
        if coords.shape[0] % n != 0:
            raise ValueError(f"coords row count {coords.shape[0]} is not divisible by n={n}")
        num_polygons = coords.shape[0] // n
        num_vertices = np.full((num_polygons,), n, dtype=np.int32)
        return PolygonDatasetArrays(coords=coords, num_vertices=num_vertices)

    raise ValueError("dataset must store num_vertices for ragged polygon coordinates")


def load_polygon_dataset(path: str | Path) -> PolygonDatasetArrays:
    with np.load(Path(path), allow_pickle=True) as npz_data:
        return polygon_dataset_from_npz(npz_data)


def concatenate_polygons(polygons: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    polygon_list = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
    if not polygon_list:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    num_vertices = np.asarray([polygon.shape[0] for polygon in polygon_list], dtype=np.int32)
    coords = np.concatenate(polygon_list, axis=0).astype(np.float32, copy=False)
    return coords, num_vertices


def vertex_count_histogram(num_vertices: np.ndarray) -> dict[int, int]:
    counts = np.asarray(num_vertices, dtype=np.int32).reshape(-1)
    unique, freq = np.unique(counts, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique.tolist(), freq.tolist())}


@dataclass(frozen=True, slots=True)
class PolygonGraphBatch:
    coords: torch.Tensor
    num_vertices: torch.Tensor
    ptr: torch.Tensor
    graph_index: torch.Tensor
    node_index: torch.Tensor

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (total_vertices, 2), got {tuple(self.coords.shape)}")
        if self.num_vertices.ndim != 1:
            raise ValueError(f"num_vertices must have shape (batch,), got {tuple(self.num_vertices.shape)}")
        if self.ptr.ndim != 1 or self.ptr.shape[0] != self.num_vertices.shape[0] + 1:
            raise ValueError(
                f"ptr must have shape ({self.num_vertices.shape[0] + 1},), got {tuple(self.ptr.shape)}"
            )
        if self.graph_index.ndim != 1 or self.graph_index.shape[0] != self.coords.shape[0]:
            raise ValueError(
                f"graph_index must have shape ({self.coords.shape[0]},), got {tuple(self.graph_index.shape)}"
            )
        if self.node_index.ndim != 1 or self.node_index.shape[0] != self.coords.shape[0]:
            raise ValueError(
                f"node_index must have shape ({self.coords.shape[0]},), got {tuple(self.node_index.shape)}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.num_vertices.shape[0])

    @property
    def total_vertices(self) -> int:
        return int(self.coords.shape[0])

    @property
    def max_vertices(self) -> int:
        return int(self.num_vertices.max().item()) if self.batch_size > 0 else 0

    @property
    def is_uniform(self) -> bool:
        return bool(self.batch_size == 0 or torch.all(self.num_vertices == self.num_vertices[0]).item())

    def uniform_num_vertices(self) -> int:
        if not self.is_uniform:
            raise ValueError("batch does not have a uniform number of vertices")
        return int(self.num_vertices[0].item()) if self.batch_size > 0 else 0

    def graph_slice(self, index: int) -> slice:
        start = int(self.ptr[index].item())
        end = int(self.ptr[index + 1].item())
        return slice(start, end)

    def graph_coords(self, index: int) -> torch.Tensor:
        return self.coords[self.graph_slice(index)]

    def to(self, device: torch.device | str) -> PolygonGraphBatch:
        return PolygonGraphBatch(
            coords=self.coords.to(device),
            num_vertices=self.num_vertices.to(device),
            ptr=self.ptr.to(device),
            graph_index=self.graph_index.to(device),
            node_index=self.node_index.to(device),
        )

    def to_dense(self) -> torch.Tensor:
        n = self.uniform_num_vertices()
        return self.coords.reshape(self.batch_size, n, 2)


def build_polygon_graph_batch(
    num_vertices: torch.Tensor | np.ndarray | Iterable[int],
    *,
    coords: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> PolygonGraphBatch:
    num_vertices_tensor = torch.as_tensor(num_vertices, dtype=torch.long, device=device)
    if num_vertices_tensor.ndim != 1:
        raise ValueError(f"num_vertices must be 1D, got shape {tuple(num_vertices_tensor.shape)}")
    if num_vertices_tensor.numel() > 0 and torch.any(num_vertices_tensor < 3):
        raise ValueError("all polygons must have at least 3 vertices")

    ptr = torch.zeros(num_vertices_tensor.shape[0] + 1, dtype=torch.long, device=num_vertices_tensor.device)
    if num_vertices_tensor.numel() > 0:
        ptr[1:] = torch.cumsum(num_vertices_tensor, dim=0)
    total_vertices = int(ptr[-1].item()) if ptr.numel() > 0 else 0

    if coords is None:
        coords_tensor = torch.empty((total_vertices, 2), dtype=torch.float32, device=num_vertices_tensor.device)
    else:
        coords_tensor = coords.to(num_vertices_tensor.device)
        if coords_tensor.ndim != 2 or coords_tensor.shape != (total_vertices, 2):
            raise ValueError(
                f"coords must have shape ({total_vertices}, 2), got {tuple(coords_tensor.shape)}"
            )

    graph_index = torch.repeat_interleave(
        torch.arange(num_vertices_tensor.shape[0], device=num_vertices_tensor.device, dtype=torch.long),
        num_vertices_tensor,
    )
    start_per_node = ptr[:-1].repeat_interleave(num_vertices_tensor) if total_vertices > 0 else ptr[:-1]
    node_index = (
        torch.arange(total_vertices, device=num_vertices_tensor.device, dtype=torch.long) - start_per_node
        if total_vertices > 0
        else torch.empty((0,), dtype=torch.long, device=num_vertices_tensor.device)
    )
    return PolygonGraphBatch(
        coords=coords_tensor,
        num_vertices=num_vertices_tensor,
        ptr=ptr,
        graph_index=graph_index,
        node_index=node_index,
    )


def collate_polygon_graph_batch(polygons: list[torch.Tensor]) -> PolygonGraphBatch:
    if not polygons:
        raise ValueError("cannot collate an empty polygon batch")
    num_vertices = torch.tensor([int(polygon.shape[0]) for polygon in polygons], dtype=torch.long)
    coords = torch.cat([polygon.to(torch.float32) for polygon in polygons], dim=0)
    return build_polygon_graph_batch(num_vertices, coords=coords)
