#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, read_mask_raw
from texturesam_v2.metrics import ari_score_binary, iou_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a held-out STLD-style benchmark with direct foreground metrics. "
            "This is separate from the RWTD partition-invariant evaluator."
        )
    )
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--proposal-union-root", type=Path, required=True)
    p.add_argument("--handcrafted-root", type=Path, required=True)
    p.add_argument("--heuristic-root", type=Path, required=True)
    p.add_argument("--learned-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    return p.parse_args()


def load_binary(path: Path, *, gt: bool) -> np.ndarray:
    raw = read_mask_raw(path)
    return ensure_binary_gt(raw, strict=False, source_name=str(path)) if gt else ensure_binary(raw, source_name=str(path))


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _summary_for(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    ious = [float(r[f"{key}_iou"]) for r in rows if r[f"{key}_iou"] is not None]
    aris = [float(r[f"{key}_ari"]) for r in rows if r[f"{key}_ari"] is not None]
    return {
        "count": int(len(ious)),
        "miou": _mean(ious),
        "ari": _mean(aris),
    }


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    image_ids = sorted(int(p.stem) for p in (args.benchmark_root / "images").glob("*.png"))
    rows: list[dict[str, object]] = []
    proposal_miss_ids: list[int] = []

    method_roots = {
        "proposal_union": args.proposal_union_root,
        "handcrafted": args.handcrafted_root,
        "ptd_heuristic": args.heuristic_root,
        "architexture": args.learned_root,
    }

    for image_id in image_ids:
        gt = load_binary(args.benchmark_root / "labels" / f"{image_id}.png", gt=True)
        row: dict[str, object] = {"image_id": int(image_id)}
        for name, root in method_roots.items():
            mask_path = root / f"{image_id}.png"
            present = mask_path.exists()
            row[f"{name}_present"] = bool(present)
            if not present:
                row[f"{name}_iou"] = None
                row[f"{name}_ari"] = None
                continue
            pred = load_binary(mask_path, gt=False)
            row[f"{name}_iou"] = float(iou_score(pred, gt))
            row[f"{name}_ari"] = float(ari_score_binary(pred, gt))
        if not bool(row["proposal_union_present"]):
            proposal_miss_ids.append(int(image_id))
        rows.append(row)

    covered_rows = [r for r in rows if bool(r["proposal_union_present"])]
    summary = {
        "benchmark_root": str(args.benchmark_root),
        "num_images": int(len(rows)),
        "proposal_union_covered": int(len(covered_rows)),
        "proposal_union_missing": int(len(proposal_miss_ids)),
        "proposal_union_missing_ids": proposal_miss_ids,
        "methods": {
            "proposal_union": {
                "all": _summary_for(rows, "proposal_union"),
                "covered": _summary_for(covered_rows, "proposal_union"),
            },
            "handcrafted": {
                "all": _summary_for(rows, "handcrafted"),
                "covered": _summary_for(covered_rows, "handcrafted"),
            },
            "ptd_heuristic": {
                "all": _summary_for(rows, "ptd_heuristic"),
                "covered": _summary_for(covered_rows, "ptd_heuristic"),
            },
            "architexture": {
                "all": _summary_for(rows, "architexture"),
                "covered": _summary_for(covered_rows, "architexture"),
            },
        },
    }
    summary["delta_vs_union"] = {
        "covered_miou": (
            float(summary["methods"]["architexture"]["covered"]["miou"])
            - float(summary["methods"]["proposal_union"]["covered"]["miou"])
        ),
        "covered_ari": (
            float(summary["methods"]["architexture"]["covered"]["ari"])
            - float(summary["methods"]["proposal_union"]["covered"]["ari"])
        ),
    }

    with (out_root / "direct_foreground_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    keys = [
        "image_id",
        "proposal_union_present",
        "proposal_union_iou",
        "proposal_union_ari",
        "handcrafted_present",
        "handcrafted_iou",
        "handcrafted_ari",
        "ptd_heuristic_present",
        "ptd_heuristic_iou",
        "ptd_heuristic_ari",
        "architexture_present",
        "architexture_iou",
        "architexture_ari",
    ]
    with (out_root / "direct_foreground_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
