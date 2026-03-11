#!/usr/bin/env python3
"""CLI wrapper for polydiff.training.train."""

from __future__ import annotations



def main() -> None:
    from polydiff.training.train import main as _main

    _main()


if __name__ == "__main__":
    main()
