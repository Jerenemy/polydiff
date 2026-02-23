"""Project-relative paths and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Repo root: .../polydiff/polydiff/paths.py -> parents[1] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"


def resolve_path(path: PathLike, default_dir: Path) -> Path:
    """Resolve a path, defaulting to default_dir when only a filename is provided."""
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        return default_dir / p
    return p


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
