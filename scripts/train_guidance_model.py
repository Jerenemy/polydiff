#!/usr/bin/env python3
"""CLI wrapper for guidance-model training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polydiff.training.train_guidance_model import main


if __name__ == "__main__":
    main()
