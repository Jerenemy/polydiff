#!/usr/bin/env python3
"""CLI wrapper for polydiff.sampling.sample."""

from __future__ import annotations



def main() -> None:
    from polydiff.sampling.sample import main as _main

    _main()


if __name__ == "__main__":
    main()
