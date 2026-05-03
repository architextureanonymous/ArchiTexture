#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
import sys
import heapq

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary


PROMPT_RE = re.compile(r"^rwtd_(?P<image_id>\d+)_p(?P<prompt>\d+)_m(?P<mask>\d+)\.png$")
OFFICIAL_RE = re.compile(r"^mask_(?P<proposal>\d+)_(?P<image_id>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a single prompt-style proposal bank by merging multiple roots. "
            "Accepted source filename patterns: rwtd_<id>_pXX_mYY.png or mask_<idx>_<id>.png"
        )
    )
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--source-root", action="append", dest="source_roots", required=True)
    p.add_argument("--clear-out", action="store_true")
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--disable-dedupe", action="store_true")
    p.add_argument("--copy-files", action="store_true", help="Copy instead of symlink.")
    p.add_argument(
        "--max-per-image",
        type=int,
        default=0,
        help="If >0, keep only the largest deduped proposals per image after merging all sources.",
    )
    return p.parse_args()


def _iter_source_files(src: Path) -> list[Path]:
    if not src.exists():
        return []
    return sorted([p for p in src.glob("*.png") if p.is_file()])


def _parse_image_id(name: str) -> str | None:
    m = PROMPT_RE.match(name)
    if m is not None:
        return m.group("image_id")
    m = OFFICIAL_RE.match(name)
    if m is not None:
        return m.group("image_id")
    return None


def _link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to read source mask for copy: {src}")
        ok = cv2.imwrite(str(dst), arr)
        if not ok:
            raise RuntimeError(f"Failed to write copied mask: {dst}")
        return
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.clear_out:
        for p in args.out_root.glob("rwtd_*_p*_m*.png"):
            p.unlink(missing_ok=True)

    src_roots = [Path(s) for s in args.source_roots]
    counters: dict[tuple[int, str], int] = defaultdict(int)  # (src_idx, image_id) -> next mask idx
    seen: dict[str, set[bytes]] = defaultdict(set)  # image_id -> fingerprints
    selected: dict[str, list[tuple[int, int, str, Path]]] = defaultdict(list)  # image_id -> min-heap(area, src_idx, name, path)

    total_seen = 0
    total_written = 0
    total_skipped_area = 0
    total_skipped_name = 0
    total_skipped_dup = 0
    total_pruned_cap = 0

    for src_idx, src_root in enumerate(src_roots):
        files = _iter_source_files(src_root)
        for src in files:
            total_seen += 1
            image_id = _parse_image_id(src.name)
            if image_id is None:
                total_skipped_name += 1
                continue

            arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if arr is None:
                total_skipped_name += 1
                continue
            m = ensure_binary(arr)
            if int(m.sum()) < int(args.min_area):
                total_skipped_area += 1
                continue

            if not args.disable_dedupe:
                key = np.packbits(m.astype(np.uint8), axis=None).tobytes()
                if key in seen[image_id]:
                    total_skipped_dup += 1
                    continue
                seen[image_id].add(key)

            if int(args.max_per_image) > 0:
                heap = selected[image_id]
                item = (int(m.sum()), int(src_idx), src.name, src)
                if len(heap) < int(args.max_per_image):
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
                    total_pruned_cap += 1
                else:
                    total_pruned_cap += 1
                continue

            local = counters[(src_idx, image_id)]
            counters[(src_idx, image_id)] += 1
            dst = args.out_root / f"rwtd_{image_id}_p{src_idx:02d}_m{local:03d}.png"
            _link_or_copy(src, dst, copy_files=bool(args.copy_files))
            total_written += 1

    if int(args.max_per_image) > 0:
        for image_id, heap in selected.items():
            items = sorted(heap, key=lambda x: (-x[0], x[1], x[2]))
            for local, (_area, src_idx, _name, src) in enumerate(items):
                dst = args.out_root / f"rwtd_{image_id}_p{src_idx:02d}_m{local:03d}.png"
                _link_or_copy(src, dst, copy_files=bool(args.copy_files))
                total_written += 1

    by_image: dict[str, int] = defaultdict(int)
    for p in args.out_root.glob("rwtd_*_p*_m*.png"):
        m = PROMPT_RE.match(p.name)
        if m is not None:
            by_image[m.group("image_id")] += 1

    if by_image:
        vals = np.asarray(list(by_image.values()), dtype=np.int32)
        mean = float(vals.mean())
        median = float(np.median(vals))
        minv = int(vals.min())
        maxv = int(vals.max())
    else:
        mean = median = 0.0
        minv = maxv = 0

    print(
        {
            "out_root": str(args.out_root),
            "sources": [str(x) for x in src_roots],
            "total_seen": int(total_seen),
            "total_written": int(total_written),
            "skipped_name": int(total_skipped_name),
            "skipped_area": int(total_skipped_area),
            "skipped_dup": int(total_skipped_dup),
            "pruned_by_cap": int(total_pruned_cap),
            "max_per_image": int(args.max_per_image),
            "num_images": int(len(by_image)),
            "masks_per_image_mean": float(mean),
            "masks_per_image_median": float(median),
            "masks_per_image_min": int(minv),
            "masks_per_image_max": int(maxv),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
