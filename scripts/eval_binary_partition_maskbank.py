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
from texturesam_v2.metrics import ari_score_binary, iou_score, rwtd_invariant_metrics


MASKBANK_RE = re.compile(r"^mask_(\d+)_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate binary texture-partition predictions for both mask-bank methods "
            "(mask_<proposal>_<image>.png) and single-mask outputs. Reports both "
            "direct foreground metrics and partition-invariant metrics."
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


def _load_predictions(root: Path) -> tuple[dict[int, list[np.ndarray]], str]:
    single_files = sorted(p for p in root.glob("*.png") if p.is_file() and p.stem.isdigit())
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


def _score_image_direct(gt: np.ndarray, preds: list[np.ndarray]) -> tuple[float, float, bool]:
    if not preds:
        return 0.0, 0.0, False
    overlapping = [p for p in preds if np.logical_and(p > 0, gt > 0).sum() > 0]
    chosen = overlapping if overlapping else preds
    ious = [float(iou_score(p, gt)) for p in chosen]
    aris = [float(ari_score_binary(p, gt)) for p in chosen]
    return _mean(ious), _mean(aris), True


def _score_image_invariant(gt: np.ndarray, preds: list[np.ndarray]) -> tuple[float, float, bool]:
    if not preds:
        return 0.0, 0.0, False
    mets = [rwtd_invariant_metrics(p, gt) for p in preds]
    return _mean([float(m.iou) for m in mets]), _mean([float(m.ari) for m in mets]), True


def main() -> int:
    args = parse_args()
    image_ids = sorted(int(p.stem) for p in (args.benchmark_root / "images").glob("*.png"))
    methods = [(name, Path(path)) for name, path in args.method]
    loaded = {name: _load_predictions(root) for name, root in methods}

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "benchmark_root": str(args.benchmark_root),
        "num_images": int(len(image_ids)),
        "methods": {},
    }

    per_method_scores: dict[str, list[tuple[float, float, bool, float, float, bool]]] = {
        name: [] for name, _ in methods
    }
    for image_id in image_ids:
        gt = load_binary(args.benchmark_root / "labels" / f"{image_id}.png", gt=True)
        row: dict[str, object] = {"image_id": int(image_id)}
        for name, _ in methods:
            pred_map, mode = loaded[name]
            preds = pred_map.get(image_id, [])
            d_miou, d_ari, d_present = _score_image_direct(gt, preds)
            i_miou, i_ari, i_present = _score_image_invariant(gt, preds)
            row[f"{name}_mode"] = mode
            row[f"{name}_present"] = bool(d_present or i_present)
            row[f"{name}_mask_count"] = int(len(preds))
            row[f"{name}_direct_miou"] = float(d_miou)
            row[f"{name}_direct_ari"] = float(d_ari)
            row[f"{name}_invariant_miou"] = float(i_miou)
            row[f"{name}_invariant_ari"] = float(i_ari)
            per_method_scores[name].append((d_miou, d_ari, d_present, i_miou, i_ari, i_present))
        rows.append(row)

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

    fieldnames = ["image_id"]
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
