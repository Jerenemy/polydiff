#!/usr/bin/env python3
"""CLI wrapper for polydiff.sampling.sample."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _ensure_repo_on_path()
    from polydiff.sampling.sample import main as _main

    _main()


if __name__ == "__main__":
    main()
