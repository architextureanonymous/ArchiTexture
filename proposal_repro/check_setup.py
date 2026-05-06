#!/usr/bin/env python3
"""
Preflight check for proposal_repro.

Run before running proposal_repro/run_proposal_repro.py to validate that
required datasets and dependencies are in place.

Usage:
    python proposal_repro/check_setup.py \
        --rwtd-root datasets/RWTD \
        --stld-root datasets/STLD
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_dir(label: str, path: Path, min_files: int = 1) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"MISSING {label}: {path} does not exist")
        return errors
    if not path.is_dir():
        errors.append(f"NOT A DIRECTORY {label}: {path}")
        return errors
    files = list(path.rglob("*"))
    if len(files) < min_files:
        errors.append(
            f"EMPTY {label}: {path} has {len(files)} file(s), expected at least {min_files}"
        )
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for mod in ["numpy", "cv2", "sklearn", "pandas"]:
        try:
            __import__(mod)
        except ImportError:
            errors.append(f"MISSING IMPORT: {mod} — run: pip install -r requirements.txt")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for proposal_repro.")
    parser.add_argument("--rwtd-root", type=Path, required=True, help="RWTD dataset root")
    parser.add_argument("--stld-root", type=Path, required=True, help="STLD dataset root")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    # Check datasets
    errors += check_dir("RWTD dataset", args.rwtd_root, min_files=10)
    errors += check_dir("STLD dataset", args.stld_root, min_files=10)

    # Check that run_proposal_repro.py exists
    script = Path(__file__).parent / "run_proposal_repro.py"
    if not script.exists():
        errors.append(
            f"MISSING SCRIPT: {script}\n"
            "  run_proposal_repro.py must be present in proposal_repro/ for the live smoke path to work.\n"
            "  See AUDIT_status.md for details."
        )

    # Check imports
    errors += check_imports()

    # Warn about GPU
    try:
        import torch  # noqa: F401
        warnings.append(
            "NOTE: smoke path uses handcrafted descriptors by default (CPU). "
            "GPU is not required for the default run."
        )
    except ImportError:
        errors.append("MISSING IMPORT: torch — run: pip install torch")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()

    if errors:
        print("Errors (must fix before running proposal_repro):")
        for e in errors:
            print(f"  {e}")
        return 1

    print("All checks passed. You can now run:")
    print(
        "  python proposal_repro/run_proposal_repro.py"
        f" --output-root outputs/proposal_repro"
        f" --rwtd-root {args.rwtd_root}"
        f" --stld-root {args.stld_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
