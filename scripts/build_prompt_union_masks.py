#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROMPT_RE = re.compile(r"^rwtd_(?P<image_id>\d+)_p(?P<prompt>\d+)_m(?P<mask>\d+)\.png$")
OFFICIAL_RE = re.compile(r"^mask_(?P<proposal>\d+)_(?P<image_id>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Union all proposal masks per image into a single binary baseline mask."
    )
    p.add_argument("--prompt-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--clear-out", action="store_true")
    return p.parse_args()


def _image_id(name: str) -> str | None:
    match = PROMPT_RE.match(name)
    if match is not None:
        return match.group("image_id")
    match = OFFICIAL_RE.match(name)
    if match is not None:
        return match.group("image_id")
    return None


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.clear_out:
        for path in args.out_root.glob("*.png"):
            path.unlink(missing_ok=True)

    files_by_id: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.prompt_root.glob("*.png")):
        image_id = _image_id(path.name)
        if image_id is None:
            continue
        files_by_id[image_id].append(path)

    rows: list[dict[str, int]] = []
    for image_id, paths in sorted(files_by_id.items(), key=lambda kv: int(kv[0])):
        union: np.ndarray | None = None
        for path in paths:
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                continue
            mask = arr > 0
            union = mask if union is None else np.logical_or(union, mask)
        if union is None:
            continue
        out = (union.astype(np.uint8) * 255)
        dst = args.out_root / f"{int(image_id)}.png"
        ok = cv2.imwrite(str(dst), out)
        if not ok:
            raise RuntimeError(f"Failed to write union mask: {dst}")
        rows.append(
            {
                "image_id": int(image_id),
                "proposal_count": int(len(paths)),
                "union_area": int(union.sum()),
            }
        )

    summary = {
        "prompt_root": str(args.prompt_root),
        "out_root": str(args.out_root),
        "num_images": int(len(rows)),
        "mean_proposal_count": float(np.mean([r["proposal_count"] for r in rows])) if rows else 0.0,
        "mean_union_area": float(np.mean([r["union_area"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    (args.out_root / "union_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
