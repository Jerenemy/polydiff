"""Synthetic paired pocket-fit data for polygon-conditioning studies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import paths
from ..data.gen_polygons import sample_polygon
from ..data.polygon_dataset import load_polygon_dataset
from .geometry import (
    PocketFitConfig,
    is_strictly_convex_polygon,
    pocket_fit_state_numpy,
    rotate_polygon_numpy,
)


@dataclass(frozen=True, slots=True)
class PocketFitPairArrays:
    ligand_coords: np.ndarray
    pocket_coords: np.ndarray
    fit_score: np.ndarray
    inside_fraction: np.ndarray
    outside_penalty: np.ndarray
    area_ratio: np.ndarray
    clearance_mean: np.ndarray

    def __post_init__(self) -> None:
        ligand_coords = np.asarray(self.ligand_coords, dtype=np.float32)
        pocket_coords = np.asarray(self.pocket_coords, dtype=np.float32)
        fit_score = np.asarray(self.fit_score, dtype=np.float32).reshape(-1)
        inside_fraction = np.asarray(self.inside_fraction, dtype=np.float32).reshape(-1)
        outside_penalty = np.asarray(self.outside_penalty, dtype=np.float32).reshape(-1)
        area_ratio = np.asarray(self.area_ratio, dtype=np.float32).reshape(-1)
        clearance_mean = np.asarray(self.clearance_mean, dtype=np.float32).reshape(-1)
        if ligand_coords.ndim != 3 or ligand_coords.shape[-1] != 2:
            raise ValueError(f"ligand_coords must have shape (num_pairs, n_ligand, 2), got {ligand_coords.shape}")
        if pocket_coords.ndim != 3 or pocket_coords.shape[-1] != 2:
            raise ValueError(f"pocket_coords must have shape (num_pairs, n_pocket, 2), got {pocket_coords.shape}")
        num_pairs = ligand_coords.shape[0]
        if pocket_coords.shape[0] != num_pairs:
            raise ValueError("ligand_coords and pocket_coords must have the same first dimension")
        for value in (fit_score, inside_fraction, outside_penalty, area_ratio, clearance_mean):
            if value.shape[0] != num_pairs:
                raise ValueError("all label arrays must have length num_pairs")
        object.__setattr__(self, "ligand_coords", ligand_coords)
        object.__setattr__(self, "pocket_coords", pocket_coords)
        object.__setattr__(self, "fit_score", fit_score)
        object.__setattr__(self, "inside_fraction", inside_fraction)
        object.__setattr__(self, "outside_penalty", outside_penalty)
        object.__setattr__(self, "area_ratio", area_ratio)
        object.__setattr__(self, "clearance_mean", clearance_mean)

    @property
    def num_pairs(self) -> int:
        return int(self.ligand_coords.shape[0])


class PocketFitPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: PocketFitPairArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays.num_pairs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "ligand_coords": torch.from_numpy(self.arrays.ligand_coords[index]),
            "pocket_coords": torch.from_numpy(self.arrays.pocket_coords[index]),
            "fit_score": torch.tensor(float(self.arrays.fit_score[index]), dtype=torch.float32),
            "inside_fraction": torch.tensor(float(self.arrays.inside_fraction[index]), dtype=torch.float32),
            "outside_penalty": torch.tensor(float(self.arrays.outside_penalty[index]), dtype=torch.float32),
            "area_ratio": torch.tensor(float(self.arrays.area_ratio[index]), dtype=torch.float32),
            "clearance_mean": torch.tensor(float(self.arrays.clearance_mean[index]), dtype=torch.float32),
        }


def load_pocket_fit_pair_arrays(path: str | Path) -> PocketFitPairArrays:
    with np.load(Path(path), allow_pickle=False) as npz_data:
        return PocketFitPairArrays(
            ligand_coords=npz_data["ligand_coords"],
            pocket_coords=npz_data["pocket_coords"],
            fit_score=npz_data["fit_score"],
            inside_fraction=npz_data["inside_fraction"],
            outside_penalty=npz_data["outside_penalty"],
            area_ratio=npz_data["area_ratio"],
            clearance_mean=npz_data["clearance_mean"],
        )


def generate_pocket_fit_pair_splits(
    *,
    ligand_data_path: Path,
    out_dir: Path,
    reward: PocketFitConfig,
    seed: int,
    ligand_size: int,
    pocket_size: int,
    num_train_pairs: int,
    num_val_pairs: int,
    num_test_pairs: int,
    num_eval_pockets: int,
    pocket_radial_sigma: float,
    pocket_angle_sigma: float,
    pocket_smooth_passes: int,
    pocket_scale_min: float,
    pocket_scale_max: float,
    ligand_scale_min: float,
    ligand_scale_max: float,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ligand_data = load_polygon_dataset(ligand_data_path)
    if not ligand_data.is_uniform:
        raise ValueError("pocket-fit study currently expects a fixed-size ligand dataset")
    if int(ligand_data.num_vertices[0]) != int(ligand_size):
        raise ValueError(
            f"ligand_data_path stores polygons with n={int(ligand_data.num_vertices[0])}, expected ligand_size={ligand_size}"
        )
    ligand_bank = ligand_data.to_dense()

    split_sizes = {
        "train": int(num_train_pairs),
        "val": int(num_val_pairs),
        "test": int(num_test_pairs),
    }
    split_paths: dict[str, Path] = {}
    for split_name, split_num_pairs in split_sizes.items():
        arrays = _generate_pair_arrays(
            ligand_bank=ligand_bank,
            pocket_size=pocket_size,
            reward=reward,
            rng=rng,
            num_pairs=split_num_pairs,
            pocket_radial_sigma=pocket_radial_sigma,
            pocket_angle_sigma=pocket_angle_sigma,
            pocket_smooth_passes=pocket_smooth_passes,
            pocket_scale_min=pocket_scale_min,
            pocket_scale_max=pocket_scale_max,
            ligand_scale_min=ligand_scale_min,
            ligand_scale_max=ligand_scale_max,
        )
        split_path = out_dir / f"pairs_{split_name}.npz"
        _save_pair_arrays(split_path, arrays)
        split_paths[split_name] = split_path

    eval_pockets = np.stack(
        [
            _sample_convex_pocket(
                pocket_size=pocket_size,
                rng=rng,
                radial_sigma=pocket_radial_sigma,
                angle_sigma=pocket_angle_sigma,
                smooth_passes=pocket_smooth_passes,
                scale_min=pocket_scale_min,
                scale_max=pocket_scale_max,
            )
            for _ in range(int(num_eval_pockets))
        ],
        axis=0,
    ).astype(np.float32)
    eval_pockets_path = out_dir / "eval_pockets.npz"
    np.savez_compressed(eval_pockets_path, pocket_coords=eval_pockets)

    metadata = {
        "ligand_data_path": str(ligand_data_path),
        "reward": reward.to_dict(),
        "ligand_size": int(ligand_size),
        "pocket_size": int(pocket_size),
        "num_train_pairs": int(num_train_pairs),
        "num_val_pairs": int(num_val_pairs),
        "num_test_pairs": int(num_test_pairs),
        "num_eval_pockets": int(num_eval_pockets),
        "seed": int(seed),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_paths["eval_pockets"] = eval_pockets_path
    split_paths["metadata"] = metadata_path
    return split_paths


def load_eval_pockets(path: str | Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as npz_data:
        return np.asarray(npz_data["pocket_coords"], dtype=np.float32)


def default_generated_data_dir(study_name: str) -> Path:
    return paths.DATA_DIR / "pocket_conditioning" / study_name


def _generate_pair_arrays(
    *,
    ligand_bank: np.ndarray,
    pocket_size: int,
    reward: PocketFitConfig,
    rng: np.random.Generator,
    num_pairs: int,
    pocket_radial_sigma: float,
    pocket_angle_sigma: float,
    pocket_smooth_passes: int,
    pocket_scale_min: float,
    pocket_scale_max: float,
    ligand_scale_min: float,
    ligand_scale_max: float,
) -> PocketFitPairArrays:
    ligands = np.empty((num_pairs, ligand_bank.shape[1], 2), dtype=np.float32)
    pockets = np.empty((num_pairs, pocket_size, 2), dtype=np.float32)
    fit_score = np.empty((num_pairs,), dtype=np.float32)
    inside_fraction = np.empty((num_pairs,), dtype=np.float32)
    outside_penalty = np.empty((num_pairs,), dtype=np.float32)
    area_ratio = np.empty((num_pairs,), dtype=np.float32)
    clearance_mean = np.empty((num_pairs,), dtype=np.float32)

    ligand_indices = rng.integers(0, ligand_bank.shape[0], size=num_pairs)
    for index in range(num_pairs):
        pocket = _sample_convex_pocket(
            pocket_size=pocket_size,
            rng=rng,
            radial_sigma=pocket_radial_sigma,
            angle_sigma=pocket_angle_sigma,
            smooth_passes=pocket_smooth_passes,
            scale_min=pocket_scale_min,
            scale_max=pocket_scale_max,
        )
        ligand = np.asarray(ligand_bank[int(ligand_indices[index])], dtype=np.float32)
        angle = float(rng.uniform(-np.pi, np.pi))
        scale = float(rng.uniform(ligand_scale_min, ligand_scale_max))
        ligand = rotate_polygon_numpy(ligand * scale, angle)
        state = pocket_fit_state_numpy(ligand, pocket, reward)

        ligands[index] = ligand
        pockets[index] = pocket
        fit_score[index] = float(state.fit_score[0])
        inside_fraction[index] = float(state.inside_fraction[0])
        outside_penalty[index] = float(state.outside_penalty[0])
        area_ratio[index] = float(state.area_ratio[0])
        clearance_mean[index] = float(state.clearance_mean[0])

    return PocketFitPairArrays(
        ligand_coords=ligands,
        pocket_coords=pockets,
        fit_score=fit_score,
        inside_fraction=inside_fraction,
        outside_penalty=outside_penalty,
        area_ratio=area_ratio,
        clearance_mean=clearance_mean,
    )


def _save_pair_arrays(path: Path, arrays: PocketFitPairArrays) -> None:
    np.savez_compressed(
        path,
        ligand_coords=arrays.ligand_coords.astype(np.float32),
        pocket_coords=arrays.pocket_coords.astype(np.float32),
        fit_score=arrays.fit_score.astype(np.float32),
        inside_fraction=arrays.inside_fraction.astype(np.float32),
        outside_penalty=arrays.outside_penalty.astype(np.float32),
        area_ratio=arrays.area_ratio.astype(np.float32),
        clearance_mean=arrays.clearance_mean.astype(np.float32),
    )


def _sample_convex_pocket(
    *,
    pocket_size: int,
    rng: np.random.Generator,
    radial_sigma: float,
    angle_sigma: float,
    smooth_passes: int,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    for _ in range(512):
        pocket, _, _ = sample_polygon(
            n=int(pocket_size),
            rng=rng,
            reject_self_intersections=True,
            deform_dist="uniform",
            radial_sigma=float(radial_sigma),
            angle_sigma=float(angle_sigma),
            smooth_passes=int(smooth_passes),
        )
        if not is_strictly_convex_polygon(pocket):
            continue
        pocket = pocket * float(rng.uniform(scale_min, scale_max))
        return pocket.astype(np.float32)
    raise RuntimeError("Failed to sample a convex pocket polygon")
