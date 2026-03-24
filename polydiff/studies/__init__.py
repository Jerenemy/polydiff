"""Study orchestration and reporting for Polydiff."""

from pathlib import Path


def run_study_from_config(config_path: Path):
    from .run import run_study_from_config as _run_study_from_config

    return _run_study_from_config(config_path)


def refresh_study_outputs(study_dir: Path):
    from .run import refresh_study_outputs as _refresh_study_outputs

    return _refresh_study_outputs(study_dir)


__all__ = ["refresh_study_outputs", "run_study_from_config"]
