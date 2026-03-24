"""Helpers for organizing diffusion outputs by run directory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from . import paths

_RUN_NAME_RE = re.compile(r"^run_(\d{4})(?:__(.+))?$")
_SAMPLE_RUN_NAME_RE = re.compile(r"^sample_(\d{4})(?:__(.+))?$")


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_name: str
    run_number: int
    model_dir: Path
    processed_dir: Path
    media_dir: Path


@dataclass(frozen=True, slots=True)
class SampleRunPaths:
    model_run_name: str
    sample_run_name: str
    sample_run_number: int
    processed_dir: Path
    media_dir: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")
    return slug or "run"


def parse_run_number(name: str) -> int | None:
    match = _RUN_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def is_run_dir_name(name: str) -> bool:
    return parse_run_number(name) is not None


def parse_sample_run_number(name: str) -> int | None:
    match = _SAMPLE_RUN_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def is_sample_run_dir_name(name: str) -> bool:
    return parse_sample_run_number(name) is not None


def list_model_run_dirs(model_root: Path | None = None) -> list[Path]:
    root = paths.ensure_dir(model_root or paths.MODELS_DIR)
    run_dirs = [p for p in root.iterdir() if p.is_dir() and is_run_dir_name(p.name)]
    return sorted(run_dirs, key=lambda p: (parse_run_number(p.name) or -1, p.name))


def latest_model_run_dir(model_root: Path | None = None) -> Path:
    run_dirs = list_model_run_dirs(model_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {model_root or paths.MODELS_DIR}")
    return run_dirs[-1]


def resolve_model_run_dir(run: str | int | None, model_root: Path | None = None) -> Path:
    run_dirs = list_model_run_dirs(model_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {model_root or paths.MODELS_DIR}")

    if run is None or str(run).lower() in {"latest", "most_recent", "recent"}:
        return run_dirs[-1]

    run_str = str(run)
    run_number = int(run_str) if run_str.isdigit() else None

    for run_dir in run_dirs:
        if run_dir.name == run_str:
            return run_dir
        if run_number is not None and parse_run_number(run_dir.name) == run_number:
            return run_dir

    raise FileNotFoundError(
        f"Could not resolve run={run!r}. Available runs: {[p.name for p in run_dirs]}"
    )


def infer_run_name_from_checkpoint_path(checkpoint_path: Path) -> str | None:
    parent_name = checkpoint_path.parent.name
    return parent_name if is_run_dir_name(parent_name) else None


def list_sampling_run_dirs(
    run_name: str,
    *,
    processed_root: Path | None = None,
) -> list[Path]:
    parent_dir = run_paths_for_name(run_name, processed_root=processed_root, create_missing=False).processed_dir
    if not parent_dir.exists():
        return []
    sample_dirs = [p for p in parent_dir.iterdir() if p.is_dir() and is_sample_run_dir_name(p.name)]
    return sorted(sample_dirs, key=lambda p: (parse_sample_run_number(p.name) or -1, p.name))


def latest_sampling_run_dir(
    run_name: str,
    *,
    processed_root: Path | None = None,
) -> Path:
    sample_dirs = list_sampling_run_dirs(run_name, processed_root=processed_root)
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_* directories found under {run_paths_for_name(run_name, processed_root=processed_root).processed_dir}")
    return sample_dirs[-1]


def run_paths_for_name(
    run_name: str,
    *,
    model_root: Path | None = None,
    processed_root: Path | None = None,
    create_missing: bool = False,
) -> RunPaths:
    model_root = model_root or paths.MODELS_DIR
    processed_root = processed_root or paths.PROCESSED_DATA_DIR
    run_number = parse_run_number(run_name)
    if run_number is None:
        raise ValueError(f"{run_name!r} is not a valid run directory name")

    model_dir = model_root / run_name
    processed_dir = processed_root / run_name
    media_dir = processed_dir / "media"

    if create_missing:
        paths.ensure_dir(model_dir)
        paths.ensure_dir(processed_dir)
        paths.ensure_dir(media_dir)

    return RunPaths(
        run_name=run_name,
        run_number=run_number,
        model_dir=model_dir,
        processed_dir=processed_dir,
        media_dir=media_dir,
    )


def sample_run_paths_for_name(
    run_name: str,
    sample_run_name: str,
    *,
    processed_root: Path | None = None,
    create_missing: bool = False,
) -> SampleRunPaths:
    run_paths = run_paths_for_name(run_name, processed_root=processed_root, create_missing=False)
    sample_run_number = parse_sample_run_number(sample_run_name)
    if sample_run_number is None:
        raise ValueError(f"{sample_run_name!r} is not a valid sample run directory name")

    processed_dir = run_paths.processed_dir / sample_run_name
    media_dir = processed_dir / "media"
    if create_missing:
        paths.ensure_dir(run_paths.processed_dir)
        paths.ensure_dir(processed_dir)
        paths.ensure_dir(media_dir)

    return SampleRunPaths(
        model_run_name=run_name,
        sample_run_name=sample_run_name,
        sample_run_number=sample_run_number,
        processed_dir=processed_dir,
        media_dir=media_dir,
    )


def create_run_paths(
    *,
    experiment_name: str,
    model_type: str,
    data_path: Path,
    model_root: Path | None = None,
    processed_root: Path | None = None,
) -> RunPaths:
    model_root = paths.ensure_dir(model_root or paths.MODELS_DIR)
    processed_root = paths.ensure_dir(processed_root or paths.PROCESSED_DATA_DIR)

    existing_numbers = [parse_run_number(p.name) or 0 for p in list_model_run_dirs(model_root)]
    next_number = (max(existing_numbers) + 1) if existing_numbers else 1

    slug_parts = [slugify(experiment_name), slugify(model_type), slugify(data_path.stem)]
    run_name = f"run_{next_number:04d}__{'-'.join(part for part in slug_parts if part)}"
    return run_paths_for_name(
        run_name,
        model_root=model_root,
        processed_root=processed_root,
        create_missing=True,
    )


def create_sampling_run_paths(
    *,
    run_name: str,
    label: str,
    processed_root: Path | None = None,
) -> SampleRunPaths:
    existing_numbers = [
        parse_sample_run_number(p.name) or 0 for p in list_sampling_run_dirs(run_name, processed_root=processed_root)
    ]
    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    slug = slugify(label)
    sample_run_name = f"sample_{next_number:04d}__{slug}" if slug else f"sample_{next_number:04d}"
    return sample_run_paths_for_name(
        run_name,
        sample_run_name,
        processed_root=processed_root,
        create_missing=True,
    )


def write_run_files(
    run_paths: RunPaths,
    *,
    config: dict[str, Any],
    config_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    source_config_path = run_paths.model_dir / "config.source.yaml"
    resolved_config_path = run_paths.model_dir / "config.resolved.yaml"
    metadata_path = run_paths.model_dir / "run_metadata.json"

    if config_path.exists():
        shutil.copy2(config_path, source_config_path)
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = {
        "run_name": run_paths.run_name,
        "run_number": run_paths.run_number,
        "created_at": datetime.now().astimezone().isoformat(),
        "model_dir": str(run_paths.model_dir),
        "processed_dir": str(run_paths.processed_dir),
        "media_dir": str(run_paths.media_dir),
        "config_path": str(config_path),
        "source_config_path": str(source_config_path),
        "resolved_config_path": str(resolved_config_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(metadata), f, indent=2, sort_keys=True)
        f.write("\n")


def write_sampling_run_files(
    sample_run_paths: SampleRunPaths,
    *,
    config: dict[str, Any],
    config_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    source_config_path = sample_run_paths.processed_dir / "config.source.yaml"
    resolved_config_path = sample_run_paths.processed_dir / "config.resolved.yaml"
    metadata_path = sample_run_paths.processed_dir / "sampling_metadata.json"

    if config_path.exists():
        shutil.copy2(config_path, source_config_path)
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = {
        "model_run_name": sample_run_paths.model_run_name,
        "sample_run_name": sample_run_paths.sample_run_name,
        "sample_run_number": sample_run_paths.sample_run_number,
        "created_at": datetime.now().astimezone().isoformat(),
        "processed_dir": str(sample_run_paths.processed_dir),
        "media_dir": str(sample_run_paths.media_dir),
        "config_path": str(config_path),
        "source_config_path": str(source_config_path),
        "resolved_config_path": str(resolved_config_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(metadata), f, indent=2, sort_keys=True)
        f.write("\n")
