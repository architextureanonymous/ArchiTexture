#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig, _postprocess
from texturesam_v2.features import compute_texture_feature_map, region_descriptor
from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, infer_rwtd_dirs, list_rwtd_images, read_image_rgb, read_mask_raw, write_binary_mask
from texturesam_v2.merge import MergeConfig, merge_masks
from texturesam_v2.metrics import PairMetrics, rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig


FIXED_MERGE = MergeConfig(
    adjacency_dilation=3,
    merge_threshold=0.50,
    w_texture=0.65,
    w_boundary=0.30,
    w_hetero=0.20,
)


MASK_RE = re.compile(r"mask_\d+_(\d+)\.png$")
RANK_RE = re.compile(r"rank\d+_idx(\d+)\.png$")


@dataclass(frozen=True)
class OraclePick:
    mask: np.ndarray
    metrics: PairMetrics
    candidate_index: int
    candidate_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build reviewer-facing RWTD proposal-bank oracle analyses from the frozen prompt bank, "
            "dense rescue audit candidates, and the official upstream evaluator."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--v9-masks-root", type=Path, required=True)
    p.add_argument("--final-masks-root", type=Path, required=True)
    p.add_argument("--audit-cases-root", type=Path, required=True)
    p.add_argument("--baseline-official-root", type=Path, required=True, help="Public TextureSAM official mask export for common-253 IDs.")
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=24)
    return p.parse_args()


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    return ensure_binary(read_mask_raw(path))


def _dedupe_masks(masks: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for m in masks:
        b = (m > 0).astype(np.uint8)
        if int(b.sum()) < min_area:
            continue
        key = np.packbits(b, axis=None).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def _proposal_centrality_order(proposals: list[np.ndarray]) -> list[int]:
    if not proposals:
        return []
    n = len(proposals)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a = proposals[i] > 0
        for j in range(i, n):
            b = proposals[j] > 0
            inter = float(np.logical_and(a, b).sum())
            uni = float(np.logical_or(a, b).sum())
            sim = 1.0 if uni <= 0 else inter / uni
            mat[i, j] = sim
            mat[j, i] = sim
    return np.argsort(mat.mean(axis=1))[::-1].tolist()


def _topk_unions(proposals: list[np.ndarray], min_area: int, max_k: int = 6) -> list[np.ndarray]:
    if not proposals:
        return []
    order = _proposal_centrality_order(proposals)
    out: list[np.ndarray] = []
    for k in range(2, min(max_k, len(order)) + 1):
        mask = np.zeros_like(proposals[0], dtype=np.uint8)
        for idx in order[:k]:
            mask = np.logical_or(mask > 0, proposals[idx] > 0)
        out.append(mask.astype(np.uint8))
    return _dedupe_masks(out, min_area=min_area)


def _candidate_index_map(cand_dir: Path, shape: tuple[int, int], min_area: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not cand_dir.exists():
        return out
    for p in sorted(cand_dir.glob("rank*_idx*.png")):
        m = RANK_RE.match(p.name)
        if m is None:
            continue
        idx = int(m.group(1))
        arr = _read_mask(p, shape)
        if int(arr.sum()) >= min_area:
            out[idx] = arr
    return out


def _parse_baseline_ids(mask_root: Path) -> set[int]:
    ids: set[int] = set()
    for p in mask_root.glob("mask_*_*.png"):
        m = MASK_RE.match(p.name)
        if m is not None:
            ids.add(int(m.group(1)))
    return ids


def _select_best(candidates: list[np.ndarray], gt: np.ndarray) -> OraclePick:
    if not candidates:
        zero = np.zeros_like(gt, dtype=np.uint8)
        return OraclePick(mask=zero, metrics=rwtd_invariant_metrics(zero, gt), candidate_index=-1, candidate_count=0)

    best_idx = -1
    best_metrics: PairMetrics | None = None
    best_mask: np.ndarray | None = None
    for idx, cand in enumerate(candidates):
        met = rwtd_invariant_metrics(cand, gt)
        if best_metrics is None or (met.iou, met.ari) > (best_metrics.iou, best_metrics.ari):
            best_metrics = met
            best_idx = idx
            best_mask = cand

    assert best_metrics is not None and best_mask is not None
    return OraclePick(mask=best_mask, metrics=best_metrics, candidate_index=best_idx, candidate_count=len(candidates))


def _run_eval(pred: Path, gt_dir: Path, upstream_root: Path, out_json: Path) -> dict:
    cmd = [
        "python",
        str(ROOT / "scripts" / "eval_upstream_texture_metrics.py"),
        "--pred-folder",
        str(pred),
        "--gt-folder",
        str(gt_dir),
        "--upstream-root",
        str(upstream_root),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _compatible_components(
    image_rgb: np.ndarray,
    proposals: list[np.ndarray],
    min_area: int,
) -> list[np.ndarray]:
    if not proposals:
        return []
    feat_map = compute_texture_feature_map(image_rgb)
    descriptors = [region_descriptor(feat_map, m) for m in proposals]
    components, _ = merge_masks(
        image_rgb=image_rgb,
        proposals=proposals,
        descriptors=descriptors,
        feature_map=feat_map,
        cfg=FIXED_MERGE,
    )
    if not components:
        components = proposals

    post_cfg = ConsolidationConfig(min_area=min_area, merge=FIXED_MERGE)
    processed = [_postprocess(m, post_cfg) for m in components]
    return _dedupe_masks(processed, min_area=min_area)


def main() -> None:
    args = parse_args()
    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    images = list_rwtd_images(image_dir)
    gt_dir = args.upstream_root / "Kaust256" / "labeles"
    store = PromptMaskProposalStore(args.prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))
    baseline_ids = _parse_baseline_ids(args.baseline_official_root)

    methods = [
        "final_release",
        "core_v9",
        "single_proposal_oracle",
        "compatible_component_oracle",
        "topk_union_oracle",
        "rescue_candidate_oracle",
        "bank_upper_bound_oracle",
    ]

    method_dirs: dict[str, Path] = {}
    for method in methods:
        full_dir = args.out_root / method / "official_export"
        common_dir = args.out_root / method / "official_export_common253"
        if full_dir.exists():
            shutil.rmtree(full_dir)
        if common_dir.exists():
            shutil.rmtree(common_dir)
        full_dir.mkdir(parents=True, exist_ok=True)
        common_dir.mkdir(parents=True, exist_ok=True)
        method_dirs[method] = full_dir

    rows: list[dict[str, object]] = []
    for i, image_path in enumerate(images, start=1):
        image_id = int(image_path.stem)
        image_rgb = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_id}.png"), strict=False)
        shape = gt.shape

        proposals = store.load(image_id, expected_shape=shape)
        single_candidates = _dedupe_masks(proposals, min_area=int(args.min_area))
        component_candidates = _compatible_components(image_rgb, proposals, int(args.min_area))
        topk_candidates = _topk_unions(proposals, min_area=int(args.min_area), max_k=6)
        single_pick = _select_best(single_candidates, gt)
        component_pick = _select_best(component_candidates, gt)
        topk_pick = _select_best(topk_candidates, gt)

        rescue_map = _candidate_index_map(args.audit_cases_root / str(image_id) / "candidates", shape, int(args.min_area))
        rescue_pairs = sorted(rescue_map.items(), key=lambda kv: kv[0])
        rescue_candidates = [mask for _, mask in rescue_pairs]
        rescue_pick = _select_best(rescue_candidates, gt)
        rescue_idx_dense = rescue_pairs[rescue_pick.candidate_index][0] if rescue_pick.candidate_index >= 0 else -1

        core = _read_mask(args.v9_masks_root / f"{image_id}.png", shape)
        core_pick = OraclePick(mask=core, metrics=rwtd_invariant_metrics(core, gt), candidate_index=0, candidate_count=1)
        final_mask = _read_mask(args.final_masks_root / f"{image_id}.png", shape)
        final_pick = OraclePick(mask=final_mask, metrics=rwtd_invariant_metrics(final_mask, gt), candidate_index=0, candidate_count=1)

        upper_candidates = [final_pick.mask, core_pick.mask]
        upper_candidates.extend(single_candidates)
        upper_candidates.extend(component_candidates)
        upper_candidates.extend(topk_candidates)
        upper_candidates.extend(rescue_candidates)
        upper_pick = _select_best(_dedupe_masks(upper_candidates, min_area=int(args.min_area)), gt)

        picks = {
            "final_release": final_pick,
            "core_v9": core_pick,
            "single_proposal_oracle": single_pick,
            "compatible_component_oracle": component_pick,
            "topk_union_oracle": topk_pick,
            "rescue_candidate_oracle": rescue_pick,
            "bank_upper_bound_oracle": upper_pick,
        }

        for method, pick in picks.items():
            full_out = method_dirs[method] / f"mask_0_{image_id}.png"
            write_binary_mask(full_out, pick.mask)
            if image_id in baseline_ids:
                common_out = args.out_root / method / "official_export_common253" / f"mask_0_{image_id}.png"
                write_binary_mask(common_out, pick.mask)

        rows.append(
            {
                "image_id": image_id,
                "proposal_count": len(proposals),
                "single_candidate_count": int(single_pick.candidate_count),
                "single_iou": float(single_pick.metrics.iou),
                "single_ari": float(single_pick.metrics.ari),
                "single_index": int(single_pick.candidate_index),
                "component_candidate_count": int(component_pick.candidate_count),
                "component_iou": float(component_pick.metrics.iou),
                "component_ari": float(component_pick.metrics.ari),
                "component_index": int(component_pick.candidate_index),
                "topk_candidate_count": int(topk_pick.candidate_count),
                "topk_iou": float(topk_pick.metrics.iou),
                "topk_ari": float(topk_pick.metrics.ari),
                "topk_index": int(topk_pick.candidate_index),
                "rescue_candidate_count": int(rescue_pick.candidate_count),
                "rescue_iou": float(rescue_pick.metrics.iou),
                "rescue_ari": float(rescue_pick.metrics.ari),
                "rescue_index": int(rescue_pick.candidate_index),
                "rescue_idx_dense": int(rescue_idx_dense),
                "core_iou": float(core_pick.metrics.iou),
                "core_ari": float(core_pick.metrics.ari),
                "final_iou": float(final_pick.metrics.iou),
                "final_ari": float(final_pick.metrics.ari),
                "upper_candidate_count": int(upper_pick.candidate_count),
                "upper_iou": float(upper_pick.metrics.iou),
                "upper_ari": float(upper_pick.metrics.ari),
                "upper_index": int(upper_pick.candidate_index),
                "gain_single_vs_core_iou": float(single_pick.metrics.iou - core_pick.metrics.iou),
                "gain_component_vs_core_iou": float(component_pick.metrics.iou - core_pick.metrics.iou),
                "gain_topk_vs_core_iou": float(topk_pick.metrics.iou - core_pick.metrics.iou),
                "gain_rescue_vs_core_iou": float(rescue_pick.metrics.iou - core_pick.metrics.iou),
                "gain_upper_vs_final_iou": float(upper_pick.metrics.iou - final_pick.metrics.iou),
                "gain_upper_vs_final_ari": float(upper_pick.metrics.ari - final_pick.metrics.ari),
            }
        )

        if i % 32 == 0:
            print(f"[rwtd-proposal-oracles] processed {i}/{len(images)} images", flush=True)

    eval_rows: list[dict[str, object]] = []
    for method in methods:
        full_json = args.out_root / method / "official_eval_full256.json"
        common_json = args.out_root / method / "official_eval_common253.json"
        full = _run_eval(args.out_root / method / "official_export", gt_dir, args.upstream_root, full_json)
        common = _run_eval(args.out_root / method / "official_export_common253", gt_dir, args.upstream_root, common_json)
        eval_rows.append(
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

    per_image_csv = args.out_root / "rwtd_proposal_oracles_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = args.out_root / "rwtd_proposal_oracles_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
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
        writer.writeheader()
        writer.writerows(eval_rows)

    payload = {
        "num_images": len(images),
        "rwtd_root": str(args.rwtd_root),
        "prompt_masks_root": str(args.prompt_masks_root),
        "v9_masks_root": str(args.v9_masks_root),
        "final_masks_root": str(args.final_masks_root),
        "audit_cases_root": str(args.audit_cases_root),
        "baseline_official_root": str(args.baseline_official_root),
        "summary_csv": str(summary_csv.resolve()),
        "per_image_csv": str(per_image_csv.resolve()),
        "methods": eval_rows,
    }
    (args.out_root / "rwtd_proposal_oracles_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
