#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary_gt, read_mask_raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert the CAID/VOC-style shoreline dataset into the local binary benchmark "
            "layout used by the ArchiTexture binary evaluation pipeline."
        )
    )
    p.add_argument("--src-root", type=Path, required=True, help="Extracted CAID root or a parent folder containing it.")
    p.add_argument("--out-root", type=Path, required=True, help="Output benchmark root with images/ and labels/.")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Requested split name. Falls back to val/train/all if the exact split file is absent.",
    )
    p.add_argument("--clear-out", action="store_true", help="Delete any existing output benchmark root first.")
    return p.parse_args()


def _resolve_root(src_root: Path) -> Path:
    candidates = [
        src_root,
        src_root / "CAID",
        src_root / "voc2012",
        src_root / "VOC2012",
    ]
    for root in candidates:
        if (root / "JPEGImages").exists() and (root / "SegmentationClass").exists():
            return root
    raise FileNotFoundError(
        f"Could not locate a CAID/VOC-style root under {src_root}. Expected JPEGImages/ and SegmentationClass/."
    )


def _read_split_ids(root: Path, split: str) -> tuple[list[str], str]:
    seg_root = root / "ImageSets" / "Segmentation"
    preferred = [split]
    if split != "all":
        preferred.extend(["test", "val", "train"])
    preferred.append("all")

    for name in preferred:
        if name == "all":
            jpg_dir = root / "JPEGImages"
            ids = sorted(p.stem for p in jpg_dir.glob("*") if p.is_file())
            if ids:
                return ids, "all_files"
            continue
        path = seg_root / f"{name}.txt"
        if path.exists():
            ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if ids:
                return ids, name
    raise FileNotFoundError(f"Could not find any usable split file under {seg_root}.")


def _resolve_image(root: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = root / "JPEGImages" / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing CAID image for id {stem}")


def _resolve_label(root: Path, stem: str) -> Path:
    for subdir in ("SegmentationClass", "SegmentationClassAug", "GT_label", "labels"):
        for suffix in (".png", ".bmp", ".jpg"):
            path = root / subdir / f"{stem}{suffix}"
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing CAID label for id {stem}")


def _copy_binary_label(src: Path, dst: Path) -> None:
    raw = read_mask_raw(src)
    mask = ensure_binary_gt(raw, strict=False, source_name=str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), mask.astype("uint8") * 255)
    if not ok:
        raise RuntimeError(f"Failed to write binary label to {dst}")


def main() -> int:
    args = parse_args()
    root = _resolve_root(args.src_root)
    split_ids, split_used = _read_split_ids(root, args.split)

    if args.clear_out and args.out_root.exists():
        shutil.rmtree(args.out_root)

    images_dir = args.out_root / "images"
    labels_dir = args.out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for new_id, stem in enumerate(split_ids):
        src_img = _resolve_image(root, stem)
        src_lbl = _resolve_label(root, stem)
        dst_img = images_dir / f"{new_id}{src_img.suffix.lower()}"
        dst_lbl = labels_dir / f"{new_id}.png"
        shutil.copy2(src_img, dst_img)
        _copy_binary_label(src_lbl, dst_lbl)
        rows.append(
            {
                "new_id": int(new_id),
                "original_id": stem,
                "image_name": src_img.name,
                "label_name": src_lbl.name,
            }
        )

    with (args.out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["new_id", "original_id", "image_name", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "src_root": str(root),
        "requested_split": args.split,
        "resolved_split": split_used,
        "num_images": int(len(rows)),
        "out_root": str(args.out_root),
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
