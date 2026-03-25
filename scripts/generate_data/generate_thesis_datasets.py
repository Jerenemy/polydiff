#!/usr/bin/env python3
"""Generate fixed-size hexagon datasets for thesis comparison studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from polydiff import paths
from polydiff.data.diagnostics import format_polygon_summary, summarize_polygon_dataset
from polydiff.data.gen_polygons import batch

PRESETS = {
    "baseline": {
        "filename": "hexagons.npz",
        "radial_sigma": 0.18,
        "angle_sigma": 0.12,
        "smooth_passes": 3,
        "deform_dist": "beta",
        "description": "default near-regular training set",
    },
    "noisy": {
        "filename": "hexagons_noisy.npz",
        "radial_sigma": 0.30,
        "angle_sigma": 0.20,
        "smooth_passes": 1,
        "deform_dist": "uniform",
        "description": "rougher shapes with broader deformation coverage",
    },
    "very_noisy": {
        "filename": "hexagons_very_noisy.npz",
        "radial_sigma": 0.38,
        "angle_sigma": 0.26,
        "smooth_passes": 0,
        "deform_dist": "uniform",
        "description": "stress-test distribution with sharp local irregularity",
    },
}


def _write_preset_dataset(
    *,
    out_dir: Path,
    preset_name: str,
    num: int,
    seed: int,
    force: bool,
) -> Path:
    preset = PRESETS[preset_name]
    out_path = out_dir / preset["filename"]
    if out_path.exists() and not force:
        print(f"[skip] {preset_name}: {out_path} already exists")
        return out_path

    coords, score, deform = batch(
        n=6,
        num=num,
        seed=seed,
        deform_dist=str(preset["deform_dist"]),
        radial_sigma=float(preset["radial_sigma"]),
        angle_sigma=float(preset["angle_sigma"]),
        smooth_passes=int(preset["smooth_passes"]),
    )
    np.savez_compressed(
        out_path,
        coords=coords.astype(np.float32),
        score=score.astype(np.float32),
        deform=deform.astype(np.float32),
        n=np.int32(6),
        preset=preset_name,
        radial_sigma=np.float32(preset["radial_sigma"]),
        angle_sigma=np.float32(preset["angle_sigma"]),
        smooth_passes=np.int32(preset["smooth_passes"]),
        deform_dist=str(preset["deform_dist"]),
    )
    summary = summarize_polygon_dataset(coords)
    print(f"[write] {preset_name}: {out_path}")
    print(f"        {preset['description']}")
    print(f"        {format_polygon_summary(summary)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--presets",
        type=str,
        default="baseline,noisy,very_noisy",
        help="comma-separated preset names to generate",
    )
    parser.add_argument("--num", type=int, default=10000, help="number of polygons per dataset")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(paths.RAW_DATA_DIR),
        help="output directory for generated datasets",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    out_dir = paths.ensure_dir(Path(args.out_dir))
    requested_presets = [part.strip() for part in str(args.presets).split(",") if part.strip()]
    unknown = [name for name in requested_presets if name not in PRESETS]
    if unknown:
        raise ValueError(f"unknown preset(s): {unknown}; expected one of {sorted(PRESETS)}")

    for offset, preset_name in enumerate(requested_presets):
        _write_preset_dataset(
            out_dir=out_dir,
            preset_name=preset_name,
            num=int(args.num),
            seed=int(args.seed) + offset,
            force=bool(args.force),
        )


if __name__ == "__main__":
    main()
