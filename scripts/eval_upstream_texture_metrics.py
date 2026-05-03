#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate masks with upstream TextureSAM-style metrics. "
            "Expected mask filenames: mask_<proposalIdx>_<imageId>.png"
        )
    )
    p.add_argument("--pred-folder", type=Path, required=True)
    p.add_argument(
        "--gt-folder",
        type=Path,
        default=Path("external/TextureSAM_upstream_20260303/Kaust256/labeles"),
    )
    p.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("external/TextureSAM_upstream_20260303"),
        help="Directory containing eval_no_agg_masks.py",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Where to write combined metric summary JSON.",
    )
    return p.parse_args()


def _match_instance_labels(pred_masks: list[np.ndarray], label: np.ndarray) -> list[np.ndarray]:
    label_unique = np.unique(label)
    matched: list[np.ndarray] = []
    for pred_mask in pred_masks:
        best_label = int(label_unique[0])
        max_overlap = -1
        for lbl in label_unique:
            overlap = int(np.count_nonzero((pred_mask > 0) & (label == lbl)))
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = int(lbl)
        mm = np.zeros_like(label, dtype=np.int64)
        mm[pred_mask > 0] = best_label
        matched.append(mm)
    matched.sort(key=lambda x: int(np.count_nonzero(x)), reverse=True)
    return matched


def _mean_iou_over_gt_classes(pred: np.ndarray, gt: np.ndarray) -> float:
    classes = np.unique(gt)
    ious: list[float] = []
    for c in classes:
        pred_c = pred == c
        gt_c = gt == c
        union = int(np.logical_or(pred_c, gt_c).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(pred_c, gt_c).sum())
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def eval_agg_like(pred_folder: Path, gt_folder: Path) -> dict[str, float | int]:
    import re

    pat = re.compile(r"^mask_\d+_(\d+)\.png$")
    pred_by_image: dict[str, list[np.ndarray]] = {}
    for p in pred_folder.glob("mask_*_*.png"):
        m = pat.match(p.name)
        if m is None:
            continue
        image_id = m.group(1)
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        pred_by_image.setdefault(image_id, []).append((arr == 255).astype(np.uint8))

    rows: list[tuple[float, float, float]] = []
    for gt_path in sorted(gt_folder.glob("*.png"), key=lambda p: int(p.stem)):
        image_id = gt_path.stem
        if image_id not in pred_by_image:
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        pred_masks = pred_by_image[image_id]

        matched = _match_instance_labels(pred_masks, gt)
        pred_label = np.zeros_like(gt, dtype=np.int64)
        for mm in matched:
            nz = mm > 0
            pred_label[nz] = mm[nz]
        iou = _mean_iou_over_gt_classes(pred_label, gt.astype(np.int64))

        gt_inv = 255 - gt
        matched_inv = _match_instance_labels(pred_masks, gt_inv)
        pred_label_inv = np.zeros_like(gt_inv, dtype=np.int64)
        for mm in matched_inv:
            nz = mm > 0
            pred_label_inv[nz] = mm[nz]
        iou_inv = _mean_iou_over_gt_classes(pred_label_inv, gt_inv.astype(np.int64))

        rows.append((iou, iou_inv, (iou + iou_inv) / 2.0))

    if not rows:
        return {
            "num_images": 0,
            "mean_iou": 0.0,
            "mean_iou_inverse": 0.0,
            "mean_iou_rwtd_average": 0.0,
        }

    arr = np.asarray(rows, dtype=np.float64)
    return {
        "num_images": int(arr.shape[0]),
        "mean_iou": float(arr[:, 0].mean()),
        "mean_iou_inverse": float(arr[:, 1].mean()),
        "mean_iou_rwtd_average": float(arr[:, 2].mean()),
    }


def main() -> int:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    if not args.pred_folder.exists():
        raise FileNotFoundError(f"pred folder not found: {args.pred_folder}")
    if not args.gt_folder.exists():
        raise FileNotFoundError(f"gt folder not found: {args.gt_folder}")

    sys.path.insert(0, str(args.upstream_root))
    import eval_no_agg_masks as noagg

    gt_masks_by_image = noagg.load_gt_masks_as_instances(str(args.gt_folder))
    pred_masks_by_image = noagg.load_masks_from_folder(str(args.pred_folder))
    metrics_results, overall_iou, overall_ari = noagg.calculate_metrics_per_gt(pred_masks_by_image, gt_masks_by_image)

    noagg_summary = {
        "script": "eval_no_agg_masks.py (official import)",
        "num_gt_images": int(len(gt_masks_by_image)),
        "num_pred_image_ids": int(len(pred_masks_by_image)),
        "num_gt_instances_evaluated": int(len(metrics_results)),
        "overall_average_iou": float(overall_iou),
        "overall_average_rand_index": float(overall_ari),
    }

    agg_like_summary = eval_agg_like(args.pred_folder, args.gt_folder)
    combined = {
        "pred_folder": str(args.pred_folder),
        "gt_folder": str(args.gt_folder),
        "noagg_official": noagg_summary,
        "agg_like_custom": agg_like_summary,
    }
    args.out_json.write_text(json.dumps(combined, indent=2))
    print(json.dumps(combined, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
