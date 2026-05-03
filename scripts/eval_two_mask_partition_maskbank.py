#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, read_mask_raw
from texturesam_v2.metrics import ari_score_binary, iou_score


MASKBANK_RE = re.compile(r"^mask_(\d+)_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate binary predictions against two GT masks per image. Pixels where both GT "
            "masks are active or inactive are ignored."
        )
    )
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--method", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def load_binary(path: Path, *, gt: bool) -> np.ndarray:
    raw = read_mask_raw(path)
    return ensure_binary_gt(raw, strict=False, source_name=str(path)) if gt else ensure_binary(raw, source_name=str(path))


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _image_ids(root: Path) -> list[int]:
    ids = {
        int(path.stem)
        for path in (root / "images").glob("*")
        if path.is_file() and path.stem.isdigit() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    }
    return sorted(ids)


def _load_predictions(root: Path) -> tuple[dict[int, list[np.ndarray]], str]:
    single_files = sorted(
        p
        for p in root.glob("*")
        if p.is_file() and p.suffix.lower() == ".png" and p.stem.isdigit()
    )
    if single_files and not any(MASKBANK_RE.match(p.name) for p in single_files):
        out: dict[int, list[np.ndarray]] = {}
        for p in single_files:
            out[int(p.stem)] = [load_binary(p, gt=False)]
        return out, "single_mask"

    out: dict[int, list[np.ndarray]] = {}
    for p in sorted(root.glob("mask_*_*.png")):
        m = MASKBANK_RE.match(p.name)
        if m is None:
            continue
        image_id = int(m.group(2))
        out.setdefault(image_id, []).append(load_binary(p, gt=False))
    return out, "mask_bank"


def _masked_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() <= 0:
        return 0.0
    return float(iou_score(pred[valid], gt[valid]))


def _masked_ari(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() <= 0:
        return 0.0
    return float(ari_score_binary(pred[valid], gt[valid]))


def _score_image_direct(gt_a: np.ndarray, valid: np.ndarray, preds: list[np.ndarray]) -> tuple[float, float, bool]:
    if not preds:
        return 0.0, 0.0, False
    overlapping = [p for p in preds if np.logical_and(p > 0, gt_a > 0).sum() > 0]
    chosen = overlapping if overlapping else preds
    ious = [_masked_iou(p, gt_a, valid) for p in chosen]
    aris = [_masked_ari(p, gt_a, valid) for p in chosen]
    return _mean(ious), _mean(aris), True


def _score_image_invariant(
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    valid: np.ndarray,
    preds: list[np.ndarray],
) -> tuple[float, float, bool]:
    if not preds:
        return 0.0, 0.0, False
    ious: list[float] = []
    aris: list[float] = []
    for pred in preds:
        iou_a = _masked_iou(pred, gt_a, valid)
        iou_b = _masked_iou(pred, gt_b, valid)
        if iou_b > iou_a:
            ious.append(iou_b)
            aris.append(_masked_ari(pred, gt_b, valid))
        else:
            ious.append(iou_a)
            aris.append(_masked_ari(pred, gt_a, valid))
    return _mean(ious), _mean(aris), True


def main() -> int:
    args = parse_args()
    image_ids = _image_ids(args.benchmark_root)
    methods = [(name, Path(path)) for name, path in args.method]
    loaded = {name: _load_predictions(root) for name, root in methods}

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "benchmark_root": str(args.benchmark_root),
        "gt_mode": "two_mask_exclusive_valid_pixels",
        "num_images": int(len(image_ids)),
        "methods": {},
    }

    per_method_scores: dict[str, list[tuple[float, float, bool, float, float, bool]]] = {
        name: [] for name, _ in methods
    }
    valid_fracs: list[float] = []
    overlap_fracs: list[float] = []
    void_fracs: list[float] = []

    for image_id in image_ids:
        gt_a = load_binary(args.benchmark_root / "labels_a" / f"{image_id}.png", gt=True)
        gt_b = load_binary(args.benchmark_root / "labels_b" / f"{image_id}.png", gt=True)
        if gt_a.shape != gt_b.shape:
            raise ValueError(f"GT shape mismatch for image_id={image_id}: {gt_a.shape} vs {gt_b.shape}")
        valid = np.logical_xor(gt_a > 0, gt_b > 0)
        overlap = np.logical_and(gt_a > 0, gt_b > 0)
        void = np.logical_not(np.logical_or(gt_a > 0, gt_b > 0))
        valid_fracs.append(float(valid.mean()))
        overlap_fracs.append(float(overlap.mean()))
        void_fracs.append(float(void.mean()))

        row: dict[str, object] = {
            "image_id": int(image_id),
            "valid_fraction": float(valid.mean()),
            "overlap_fraction": float(overlap.mean()),
            "void_fraction": float(void.mean()),
        }
        for name, _ in methods:
            pred_map, mode = loaded[name]
            preds = pred_map.get(image_id, [])
            d_miou, d_ari, d_present = _score_image_direct(gt_a, valid, preds)
            i_miou, i_ari, i_present = _score_image_invariant(gt_a, gt_b, valid, preds)
            row[f"{name}_mode"] = mode
            row[f"{name}_present"] = bool(d_present or i_present)
            row[f"{name}_mask_count"] = int(len(preds))
            row[f"{name}_direct_miou"] = float(d_miou)
            row[f"{name}_direct_ari"] = float(d_ari)
            row[f"{name}_invariant_miou"] = float(i_miou)
            row[f"{name}_invariant_ari"] = float(i_ari)
            per_method_scores[name].append((d_miou, d_ari, d_present, i_miou, i_ari, i_present))
        rows.append(row)

    summary["valid_fraction_mean"] = _mean(valid_fracs)
    summary["overlap_fraction_mean"] = _mean(overlap_fracs)
    summary["void_fraction_mean"] = _mean(void_fracs)

    for name, _ in methods:
        scores = per_method_scores[name]
        direct_cov = [(miou, ari) for miou, ari, present, _, _, _ in scores if present]
        inv_cov = [(miou, ari) for _, _, _, miou, ari, present in scores if present]
        direct_all = [(miou, ari) for miou, ari, _, _, _, _ in scores]
        inv_all = [(miou, ari) for _, _, _, miou, ari, _ in scores]
        pred_map, mode = loaded[name]
        summary["methods"][name] = {
            "mode": mode,
            "coverage": int(sum(1 for _, _, present, _, _, _ in scores if present)),
            "direct": {
                "all": {
                    "miou": _mean([miou for miou, _ in direct_all]),
                    "ari": _mean([ari for _, ari in direct_all]),
                },
                "covered": {
                    "miou": _mean([miou for miou, _ in direct_cov]),
                    "ari": _mean([ari for _, ari in direct_cov]),
                },
            },
            "invariant": {
                "all": {
                    "miou": _mean([miou for miou, _ in inv_all]),
                    "ari": _mean([ari for _, ari in inv_all]),
                },
                "covered": {
                    "miou": _mean([miou for miou, _ in inv_cov]),
                    "ari": _mean([ari for _, ari in inv_cov]),
                },
            },
            "mean_mask_count_on_present": _mean(
                [len(pred_map.get(image_id, [])) for image_id in image_ids if image_id in pred_map]
            ),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    fieldnames = ["image_id", "valid_fraction", "overlap_fraction", "void_fraction"]
    for name, _ in methods:
        fieldnames.extend(
            [
                f"{name}_mode",
                f"{name}_present",
                f"{name}_mask_count",
                f"{name}_direct_miou",
                f"{name}_direct_ari",
                f"{name}_invariant_miou",
                f"{name}_invariant_ari",
            ]
        )
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
