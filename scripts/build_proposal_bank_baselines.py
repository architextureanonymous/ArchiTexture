#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary, infer_rwtd_dirs, list_rwtd_images, read_mask_raw, write_binary_mask
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig
from texturesam_v2.ptd_v8_partition import _proposal_union


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build simple proposal-bank baselines for reviewer isolation studies "
            "(compatibility-only, repair-only, ranking-only, union/intersection heuristics, lightweight reranker)."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--dense-prompt-masks-root", type=Path, required=True)
    p.add_argument("--v9-masks-root", type=Path, required=True)
    p.add_argument("--diagnostics-per-image-root", type=Path, required=True)
    p.add_argument("--audit-cases-root", type=Path, required=True)
    p.add_argument("--baseline-official-root", type=Path, required=True, help="Path to reproduced TextureSAM official masks.")
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--consensus-thresh", type=float, default=0.56, help="Higher threshold approximates intersection-like heuristic.")
    return p.parse_args()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    return ensure_binary(read_mask_raw(path))


def _candidate_index_map(cand_dir: Path, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not cand_dir.exists():
        return out
    pat = re.compile(r"rank\d+_idx(\d+)\.png$")
    for p in sorted(cand_dir.glob("rank*_idx*.png")):
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        out[idx] = _read_mask(p, shape)
    return out


def _parse_baseline_ids(mask_root: Path) -> set[int]:
    out: set[int] = set()
    pat = re.compile(r"mask_\d+_(\d+)\.png$")
    for p in mask_root.glob("mask_*_*.png"):
        m = pat.match(p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def _run_eval(root: Path, pred: Path, gt: Path, upstream: Path, out_json: Path) -> dict:
    cmd = [
        "python",
        str(root / "scripts" / "eval_upstream_texture_metrics.py"),
        "--pred-folder",
        str(pred),
        "--gt-folder",
        str(gt),
        "--upstream-root",
        str(upstream),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    image_dir, _ = infer_rwtd_dirs(args.rwtd_root)
    images = list_rwtd_images(image_dir)
    store = PromptMaskProposalStore(args.dense_prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))
    baseline_ids = _parse_baseline_ids(args.baseline_official_root)
    gt_dir = args.upstream_root / "Kaust256" / "labeles"

    methods = [
        "compatibility_only_core",
        "repair_only_prob",
        "ranking_only_gain",
        "union_heuristic",
        "intersection_heuristic",
        "lightweight_reranker",
    ]
    method_dirs: dict[str, Path] = {}
    for m in methods:
        d = args.out_root / m / "official_export"
        dc = args.out_root / m / "official_export_common253"
        if d.exists():
            shutil.rmtree(d)
        if dc.exists():
            shutil.rmtree(dc)
        d.mkdir(parents=True, exist_ok=True)
        dc.mkdir(parents=True, exist_ok=True)
        method_dirs[m] = d

    for i, image_path in enumerate(images, start=1):
        image_id = int(image_path.stem)
        shape = read_mask_raw(image_path).shape[:2]
        core = _read_mask(args.v9_masks_root / f"{image_id}.png", shape)
        proposals = store.load(image_id, expected_shape=shape)

        union = np.zeros(shape, dtype=np.uint8)
        inter_h = np.zeros(shape, dtype=np.uint8)
        if proposals:
            union = _proposal_union(proposals).astype(np.uint8)
            stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
            freq = stack.mean(axis=0)
            inter_h = (freq >= float(args.consensus_thresh)).astype(np.uint8)

        diag_path = args.diagnostics_per_image_root / f"{image_id}.json"
        cand_map = _candidate_index_map(args.audit_cases_root / str(image_id) / "candidates", shape)
        candidate_records: list[dict] = []
        if diag_path.exists():
            try:
                d = json.loads(diag_path.read_text(encoding="utf-8"))
                candidate_records = list(d.get("candidates", []))
            except Exception:
                candidate_records = []

        best_by_prob = None
        best_by_gain = None
        best_by_rerank = None
        if candidate_records:
            best_by_prob = max(candidate_records, key=lambda r: float(r.get("pred_prob", 0.0)))
            best_by_gain = max(candidate_records, key=lambda r: float(r.get("pred_gain", 0.0)))
            best_by_rerank = max(
                candidate_records,
                key=lambda r: (
                    float(r.get("pred_gain", 0.0))
                    + 0.25 * float(r.get("support_norm", 0.0))
                    - 0.20 * float(r.get("risk_prob", 0.0))
                ),
            )

        def _pick(rec: dict | None, fallback: np.ndarray) -> np.ndarray:
            if rec is None:
                return fallback
            idx = int(rec.get("idx_dense", -1))
            m = cand_map.get(idx)
            if m is None:
                return fallback
            return m

        preds = {
            "compatibility_only_core": core,
            "repair_only_prob": _pick(best_by_prob, core),
            "ranking_only_gain": _pick(best_by_gain, core),
            "union_heuristic": union if int(union.sum()) > 0 else core,
            "intersection_heuristic": inter_h if int(inter_h.sum()) > 0 else core,
            "lightweight_reranker": _pick(best_by_rerank, core),
        }

        for method, mask in preds.items():
            out = method_dirs[method] / f"mask_0_{image_id}.png"
            write_binary_mask(out, mask)
            if image_id in baseline_ids:
                outc = args.out_root / method / "official_export_common253" / f"mask_0_{image_id}.png"
                write_binary_mask(outc, mask)

        if i % 32 == 0:
            print(f"[baselines] processed {i}/{len(images)} images")

    rows: list[dict[str, str | float | int]] = []
    for method in methods:
        full_pred = args.out_root / method / "official_export"
        common_pred = args.out_root / method / "official_export_common253"
        full_json = args.out_root / method / "official_eval_full256.json"
        common_json = args.out_root / method / "official_eval_common253.json"
        full = _run_eval(root, full_pred, gt_dir, args.upstream_root, full_json)
        common = _run_eval(root, common_pred, gt_dir, args.upstream_root, common_json)
        rows.append(
            {
                "method": method,
                "full256_miou": float(full["noagg_official"]["overall_average_iou"]),
                "full256_ari": float(full["noagg_official"]["overall_average_rand_index"]),
                "full256_num_pred_image_ids": int(full["noagg_official"]["num_pred_image_ids"]),
                "common253_miou": float(common["noagg_official"]["overall_average_iou"]),
                "common253_ari": float(common["noagg_official"]["overall_average_rand_index"]),
                "common253_num_pred_image_ids": int(common["noagg_official"]["num_pred_image_ids"]),
            }
        )

    out_csv = args.out_root / "proposal_bank_baselines_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "full256_miou",
                "full256_ari",
                "full256_num_pred_image_ids",
                "common253_miou",
                "common253_ari",
                "common253_num_pred_image_ids",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    summary = {
        "num_images": len(images),
        "methods": rows,
        "summary_csv": str(out_csv.resolve()),
    }
    (args.out_root / "proposal_bank_baselines_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
