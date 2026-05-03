#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export single-mask predictions (<imageId>.png) into official TextureSAM "
            "filename format mask_0_<imageId>.png."
        )
    )
    p.add_argument("--src-masks-dir", type=Path, required=True)
    p.add_argument("--dst-dir", type=Path, required=True)
    p.add_argument("--clear-dst", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_dst:
        for p in args.dst_dir.glob("mask_*_*.png"):
            p.unlink(missing_ok=True)

    written = 0
    skipped = 0
    for src in sorted(args.src_masks_dir.glob("*.png")):
        try:
            image_id = int(src.stem)
        except ValueError:
            skipped += 1
            continue
        arr = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            skipped += 1
            continue
        out = (arr > 0).astype(np.uint8) * 255
        dst = args.dst_dir / f"mask_0_{image_id}.png"
        ok = cv2.imwrite(str(dst), out)
        if not ok:
            raise RuntimeError(f"failed writing {dst}")
        written += 1

    print(
        {
            "src_masks_dir": str(args.src_masks_dir),
            "dst_dir": str(args.dst_dir),
            "written": written,
            "skipped": skipped,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
