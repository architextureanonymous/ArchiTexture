#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run upstream TextureSAM/SAM2 automatic mask inference and save masks "
            "in official eval filename format: mask_<proposalIdx>_<imageId>.png"
        )
    )
    p.add_argument(
        "--sam2-root",
        type=Path,
        default=Path("external/TextureSAM_upstream_20260303/sam2"),
        help="Path to upstream sam2 repo root (contains sam2/ package).",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("external/TextureSAM_upstream_20260303/Kaust256/images"),
        help="Directory with input images (e.g., Kaust256/images).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/TextureSAM_upstream_assets/checkpoints/checkpoints/sam2.1_hiera_small_0.3.pt"),
        help="Path to model checkpoint.",
    )
    p.add_argument(
        "--model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Hydra config path relative to sam2 package.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/repro_upstream_eval/official_0p3_masks"),
        help="Directory where mask_*.png files will be written.",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--points-per-side", type=int, default=64)
    p.add_argument("--pred-iou-thresh", type=float, default=0.8)
    p.add_argument("--stability-score-thresh", type=float, default=0.2)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    p.add_argument("--min-mask-region-area", type=int, default=0)
    p.add_argument("--multimask-output", action="store_true")
    p.add_argument("--max-images", type=int, default=0, help="0 means all.")
    p.add_argument("--start-index", type=int, default=0, help="Start index in sorted image list.")
    p.add_argument("--resume", action="store_true", help="Skip image IDs that already have masks in out-dir.")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def natural_sort_key(s: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def existing_image_ids(out_dir: Path) -> set[int]:
    done: set[int] = set()
    pat = re.compile(r"^mask_\d+_(\d+)\.png$")
    for p in out_dir.glob("mask_*_*.png"):
        m = pat.match(p.name)
        if m is not None:
            done.add(int(m.group(1)))
    return done


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    sys.path.insert(0, str(args.sam2_root))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    print(f"[load] building model on {args.device}...")
    sam2 = build_sam2(args.model_cfg, str(args.checkpoint), device=args.device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        mask_threshold=args.mask_threshold,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
        multimask_output=args.multimask_output,
    )

    image_files = sorted(
        [p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: natural_sort_key(p.name),
    )
    if args.start_index > 0:
        image_files = image_files[args.start_index :]
    if args.max_images > 0:
        image_files = image_files[: args.max_images]

    done_ids = existing_image_ids(args.out_dir) if args.resume else set()
    manifest_path = args.out_dir / "manifest.csv"
    write_header = not manifest_path.exists()
    manifest_fp = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(manifest_fp)
    if write_header:
        writer.writerow(["image_id", "image_name", "num_masks", "elapsed_sec"])

    total = len(image_files)
    processed = 0
    started = time.time()
    try:
        for i, img_path in enumerate(image_files, start=1):
            image_id = int(img_path.stem)
            if image_id in done_ids:
                continue

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[warn] failed to read {img_path}")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            masks = mask_generator.generate(img)
            dt = time.time() - t0

            for midx, m in enumerate(masks):
                out_path = args.out_dir / f"mask_{midx}_{image_id}.png"
                arr = (m["segmentation"].astype(np.uint8)) * 255
                ok = cv2.imwrite(str(out_path), arr)
                if not ok:
                    raise RuntimeError(f"failed to write mask: {out_path}")

            writer.writerow([image_id, img_path.name, len(masks), f"{dt:.4f}"])
            manifest_fp.flush()
            processed += 1

            if processed % max(1, args.log_every) == 0:
                elapsed = time.time() - started
                print(
                    f"[progress] {processed} new images, index={i}/{total}, "
                    f"last_image_id={image_id}, last_num_masks={len(masks)}, "
                    f"last_time={dt:.2f}s, elapsed={elapsed/60.0:.1f}m"
                )
    finally:
        manifest_fp.close()

    print(f"[done] processed {processed} new images into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
