#!/usr/bin/env python3
"""
Feature-clustering eval on the bundle's 56-image DeTexture ADE20K validation set.

Reads images/labels_a/labels_b directly from the bundle benchmark root
(detexture_validation_refined) — no separate dataset download required.

Usage:
    python scripts/eval_bundle_detexture_fc.py \
        --benchmark-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/experiments/detexture_ade20k_eval_20260317/benchmarks/detexture_validation_refined \
        --output-dir outputs/bundle_detexture_fc_eval \
        [--variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2] \
        [--device cuda] \
        [--failure-policy skip]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from tqdm import tqdm

FC_SRC = Path(__file__).resolve().parents[1] / "scripts/feature_clustering/src"
if str(FC_SRC) not in sys.path:
    sys.path.insert(0, str(FC_SRC))

from rwtd_sam3.eval.experiment_registry import (
    DEFAULT_CROSS_DATASET_EXPERIMENT,
    build_cross_dataset_experiment_runner,
)
from rwtd_sam3.eval.sam3_auto import select_best_binary_assignment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument(
        "--variant",
        default=DEFAULT_CROSS_DATASET_EXPERIMENT,
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--failure-policy", choices=["abort", "skip"], default="abort")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _load_binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def _image_ids(bench: Path) -> list[int]:
    return sorted(int(p.stem) for p in (bench / "images").glob("*.png") if p.stem.isdigit())


def main() -> int:
    args = parse_args()
    bench = args.benchmark_root.resolve()

    images_dir  = bench / "images"
    labels_a_dir = bench / "labels_a"
    labels_b_dir = bench / "labels_b"

    for d in (images_dir, labels_a_dir, labels_b_dir):
        if not d.exists():
            print(f"MISSING: {d}", file=sys.stderr)
            return 1

    all_ids = _image_ids(bench)
    if args.limit:
        all_ids = all_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = build_cross_dataset_experiment_runner(
        args.variant,
        model_id="facebook/sam2.1-hiera-small",
        device=args.device,
        hf_token=None,
        official_checkpoint_path=None,
    )

    rows: list[dict] = []
    failures: list[dict] = []

    for img_id in tqdm(all_ids, desc="eval:bundle_detexture", unit="img"):
        img_path = images_dir / f"{img_id}.png"
        la_path  = labels_a_dir / f"{img_id}.png"
        lb_path  = labels_b_dir / f"{img_id}.png"
        if not img_path.exists() or not la_path.exists() or not lb_path.exists():
            failures.append({"image_id": img_id, "error": "missing file"})
            continue

        image    = Image.open(img_path).convert("RGB")
        mask_a   = _load_binary(la_path)
        mask_b   = _load_binary(lb_path)

        try:
            refinement = runner.generate_feature_clusters(image)
        except Exception as exc:
            if args.failure_policy == "skip":
                failures.append({"image_id": img_id, "error": str(exc)})
                continue
            raise

        pred_a = np.asarray(refinement.refined_mask_a, dtype=bool)
        pred_b = np.asarray(refinement.refined_mask_b, dtype=bool)

        assignment = select_best_binary_assignment(
            prediction_a=pred_a,
            prediction_b=pred_b,
            target_a=mask_a,
            target_b=mask_b,
            source_a="cluster_a",
            source_b="cluster_b",
        )

        rows.append({
            "image_id":   img_id,
            "variant":    args.variant,
            "eval_miou":  assignment.chosen_miou,
            "eval_ari":   assignment.chosen_ari,
            "miou":       assignment.chosen_miou,
            "ari":        assignment.chosen_ari,
            "assignment": assignment.assignment_used,
        })

    if not rows:
        print("No successful evaluations.", file=sys.stderr)
        return 1

    csv_path = args.output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "detexture_ade20k_bundle",
        "benchmark_root": str(bench),
        "variant": args.variant,
        "num_total": len(all_ids),
        "num_valid": len(rows),
        "num_failures": len(failures),
        "mean_miou": mean(r["miou"] for r in rows),
        "mean_ari":  mean(r["ari"]  for r in rows),
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDeTexture bundle FC eval — {len(rows)}/{len(all_ids)} images")
    print(f"  mIoU = {summary['mean_miou']:.4f}")
    print(f"  ARI  = {summary['mean_ari']:.4f}")
    print(f"Saved → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
