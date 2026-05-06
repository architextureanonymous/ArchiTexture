"""Thin launcher for the repository CLI.

This script makes ``python main.py ...`` behave like the installed
``rwtd-sam3`` console entrypoint during local development. It adds ``src/`` to
``sys.path`` and delegates all argument parsing and run orchestration to
``rwtd_sam3.cli.main``.

Primary entrypoint:
- ``main`` from ``rwtd_sam3.cli``: parse CLI arguments and dispatch to the
  requested dataset inspection or evaluation routine.

Inputs are shell argv strings. Outputs are console logs plus the run artifacts
written by the delegated evaluation modules under ``outputs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rwtd_sam3.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
