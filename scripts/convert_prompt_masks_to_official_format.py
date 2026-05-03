#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert prompt-mask filenames rwtd_<imageId>_pXX_mYY.png into official "
            "TextureSAM eval format mask_<proposalIdx>_<imageId>.png via symlinks."
        )
    )
    p.add_argument("--src", type=Path, required=True, help="Source prompt mask directory.")
    p.add_argument("--dst", type=Path, required=True, help="Output directory for mask_* files.")
    p.add_argument(
        "--clear-dst",
        action="store_true",
        help="Delete existing mask_* files in destination before writing new links.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    if args.clear_dst:
        for p in args.dst.glob("mask_*_*.png"):
            p.unlink(missing_ok=True)

    pat = re.compile(r"^rwtd_(\d+)_p(\d+)_m(\d+)\.png$")
    converted = 0
    skipped = 0
    collisions = 0

    for src in sorted(args.src.glob("*.png")):
        m = pat.match(src.name)
        if m is None:
            skipped += 1
            continue

        image_id = int(m.group(1))
        prompt_id = int(m.group(2))
        mask_id = int(m.group(3))
        proposal_idx = prompt_id * 100 + mask_id

        dst = args.dst / f"mask_{proposal_idx}_{image_id}.png"
        if dst.exists() or dst.is_symlink():
            collisions += 1
            extra = 10000
            while True:
                alt = args.dst / f"mask_{proposal_idx + extra}_{image_id}.png"
                if not alt.exists() and not alt.is_symlink():
                    dst = alt
                    break
                extra += 1

        rel = os.path.relpath(src, start=args.dst)
        dst.symlink_to(rel)
        converted += 1

    print(
        {
            "src": str(args.src),
            "dst": str(args.dst),
            "converted": converted,
            "skipped": skipped,
            "collisions": collisions,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

