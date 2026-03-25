"""Helpers for config-driven study execution and aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .. import paths
from ..runs import slugify
from ..utils.runtime import load_yaml_config, resolve_project_path

_STUDY_NAME_RE = re.compile(r"^study_(\d{4})(?:__(.+))?$")
_PLACEHOLDER_RE = re.compile(r"^\{\{([^{}.]+)\.([^{}.]+)\}\}$")


@dataclass(frozen=True, slots=True)
class StudySummaryOptions:
    reference_data_path: Path | None
    representative_count: int
    outlier_count: int
    max_projection_points: int


@dataclass(frozen=True, slots=True)
class StudyParallelOptions:
    enabled: bool
    max_workers: int
    require_cuda: bool


@dataclass(frozen=True, slots=True)
class StudyCase:
    name: str
    kind: str
    config_path: Path
    overrides: dict[str, Any]
    tags: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StudySpec:
    name: str
    config_path: Path
    root_dir: Path
    summary: StudySummaryOptions
    parallel: StudyParallelOptions
    cases: tuple[StudyCase, ...]


@dataclass(frozen=True, slots=True)
class StudyPaths:
    study_name: str
    study_number: int
    study_dir: Path
    configs_dir: Path
    reports_dir: Path
    figures_dir: Path


def _parse_study_number(name: str) -> int | None:
    match = _STUDY_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def create_study_paths(*, name: str, root_dir: Path | None = None) -> StudyPaths:
    study_root = paths.ensure_dir(root_dir or paths.STUDIES_DIR)
    existing_numbers = [
        _parse_study_number(path.name) or 0
        for path in study_root.iterdir()
        if path.is_dir() and _parse_study_number(path.name) is not None
    ]
    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    study_name = f"study_{next_number:04d}__{slugify(name)}"
    study_dir = study_root / study_name
    configs_dir = study_dir / "configs"
    reports_dir = study_dir / "reports"
    figures_dir = study_dir / "figures"
    for directory in (study_dir, configs_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return StudyPaths(
        study_name=study_name,
        study_number=next_number,
        study_dir=study_dir,
        configs_dir=configs_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )


def load_study_spec(config_path: Path) -> StudySpec:
    cfg = load_yaml_config(config_path)
    study_cfg = cfg.get("study", {})
    if study_cfg is None:
        study_cfg = {}
    if not isinstance(study_cfg, dict):
        raise ValueError("study config must be a mapping")

    cases_cfg = cfg.get("cases")
    if not isinstance(cases_cfg, list) or not cases_cfg:
        raise ValueError("study config must define a non-empty cases list")

    summary_cfg = study_cfg.get("summary", {})
    if summary_cfg is None:
        summary_cfg = {}
    if not isinstance(summary_cfg, dict):
        raise ValueError("study.summary must be a mapping if provided")
    parallel_cfg = study_cfg.get("parallel", {})
    if parallel_cfg is None:
        parallel_cfg = {}
    if not isinstance(parallel_cfg, dict):
        raise ValueError("study.parallel must be a mapping if provided")
    max_workers = int(parallel_cfg.get("max_workers", 2))
    if max_workers < 1:
        raise ValueError(f"study.parallel.max_workers must be >= 1, got {max_workers}")

    cases: list[StudyCase] = []
    for case_cfg in cases_cfg:
        if not isinstance(case_cfg, dict):
            raise ValueError("each study case must be a mapping")
        name = str(case_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("study case is missing a name")
        kind = str(case_cfg.get("kind", "")).strip().lower()
        if kind not in {"train_diffusion", "train_guidance_model", "sample_diffusion"}:
            raise ValueError(
                f"Unsupported study case kind {kind!r}; expected train_diffusion, train_guidance_model, or sample_diffusion"
            )
        case_config = case_cfg.get("config")
        if case_config is None:
            raise ValueError(f"study case {name!r} is missing config")
        overrides = case_cfg.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise ValueError(f"study case {name!r} overrides must be a mapping")
        tags = case_cfg.get("tags", {})
        if tags is None:
            tags = {}
        if not isinstance(tags, dict):
            raise ValueError(f"study case {name!r} tags must be a mapping")
        cases.append(
            StudyCase(
                name=name,
                kind=kind,
                config_path=resolve_project_path(case_config),
                overrides=dict(overrides),
                tags=dict(tags),
            )
        )

    return StudySpec(
        name=str(study_cfg.get("name", config_path.stem)),
        config_path=config_path,
        root_dir=resolve_project_path(study_cfg.get("root_dir", paths.STUDIES_DIR)),
        summary=StudySummaryOptions(
            reference_data_path=(
                None
                if summary_cfg.get("reference_data_path") is None
                else resolve_project_path(summary_cfg["reference_data_path"])
            ),
            representative_count=int(summary_cfg.get("representative_count", 16)),
            outlier_count=int(summary_cfg.get("outlier_count", 16)),
            max_projection_points=int(summary_cfg.get("max_projection_points", 2000)),
        ),
        parallel=StudyParallelOptions(
            enabled=bool(parallel_cfg.get("enabled", False)),
            max_workers=max_workers,
            require_cuda=bool(parallel_cfg.get("require_cuda", True)),
        ),
        cases=tuple(cases),
    )


def resolve_case_placeholders(value: Any, *, case_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        return {key: resolve_case_placeholders(val, case_results=case_results) for key, val in value.items()}
    if isinstance(value, list):
        return [resolve_case_placeholders(item, case_results=case_results) for item in value]
    if not isinstance(value, str):
        return value

    match = _PLACEHOLDER_RE.match(value.strip())
    if match is None:
        return value
    case_name, field_name = match.groups()
    if case_name not in case_results:
        raise KeyError(f"Unknown study case placeholder {case_name!r}")
    if field_name not in case_results[case_name]:
        raise KeyError(f"Study case {case_name!r} has no result field {field_name!r}")
    return case_results[case_name][field_name]


def apply_dotted_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(cfg))
    for dotted_key, value in overrides.items():
        if not isinstance(dotted_key, str) or not dotted_key:
            raise ValueError(f"Override keys must be non-empty strings, got {dotted_key!r}")
        parts = dotted_key.split(".")
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            next_value = cursor.get(part)
            if next_value is None:
                next_value = {}
                cursor[part] = next_value
            if not isinstance(next_value, dict):
                raise ValueError(f"Cannot descend into non-mapping key {part!r} while applying {dotted_key!r}")
            cursor = next_value
        leaf_key = parts[-1]
        if value is None:
            cursor.pop(leaf_key, None)
        else:
            cursor[leaf_key] = value
    return result


def write_yaml_config(path: Path, cfg: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def flatten_mapping(value: Any, *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_mapping(inner, prefix=next_prefix))
        return flat
    flat[prefix] = value
    return flat


def case_result_to_dict(result: Any, *, kind: str, name: str, config_path: Path) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        raise TypeError(f"Unsupported study case result type {type(result)!r}")
    normalized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    normalized["case_name"] = name
    normalized["case_kind"] = kind
    normalized["resolved_config_path"] = str(config_path)
    return normalized


def write_study_metadata(path: Path, *, spec: StudySpec, paths_obj: StudyPaths) -> Path:
    payload = {
        "study_name": paths_obj.study_name,
        "study_number": paths_obj.study_number,
        "study_dir": str(paths_obj.study_dir),
        "configs_dir": str(paths_obj.configs_dir),
        "reports_dir": str(paths_obj.reports_dir),
        "figures_dir": str(paths_obj.figures_dir),
        "source_config_path": str(spec.config_path),
        "summary": {
            "reference_data_path": None if spec.summary.reference_data_path is None else str(spec.summary.reference_data_path),
            "representative_count": spec.summary.representative_count,
            "outlier_count": spec.summary.outlier_count,
            "max_projection_points": spec.summary.max_projection_points,
        },
        "parallel": {
            "enabled": spec.parallel.enabled,
            "max_workers": spec.parallel.max_workers,
            "require_cuda": spec.parallel.require_cuda,
        },
        "cases": [
            {
                "name": case.name,
                "kind": case.kind,
                "config_path": str(case.config_path),
                "overrides": case.overrides,
                "tags": case.tags,
            }
            for case in spec.cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path
