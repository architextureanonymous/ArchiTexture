#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig, _postprocess
from texturesam_v2.features import compute_texture_feature_map, region_descriptor
from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, read_image_rgb, read_mask_raw
from texturesam_v2.merge import MergeConfig, merge_masks
from texturesam_v2.metrics import ari_score_binary, iou_score
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig


FIXED_MERGE = MergeConfig(
    adjacency_dilation=3,
    merge_threshold=0.50,
    w_texture=0.65,
    w_boundary=0.30,
    w_hetero=0.20,
)


@dataclass(frozen=True)
class DirectMetrics:
    iou: float
    ari: float


@dataclass(frozen=True)
class OraclePick:
    metrics: DirectMetrics
    candidate_index: int
    candidate_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build reviewer-facing STLD proposal-bank oracle analyses under the same "
            "direct-foreground evaluator used by the main STLD comparison table."
        )
    )
    p.add_argument("--benchmark-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--final-masks-root", type=Path, required=True)
    p.add_argument("--texturesam-maskbank-csv", type=Path, required=True)
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
        b = ensure_binary(m)
        if int(b.sum()) < min_area:
            continue
        key = np.packbits(b, axis=None).tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(b.astype(np.uint8))
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


def _direct_metrics(pred: np.ndarray, gt: np.ndarray) -> DirectMetrics:
    p = ensure_binary(pred)
    g = ensure_binary_gt(gt, strict=False)
    return DirectMetrics(
        iou=float(iou_score(p, g)),
        ari=float(ari_score_binary(p, g)),
    )


def _select_best(candidates: list[np.ndarray], gt: np.ndarray) -> OraclePick:
    if not candidates:
        zero = np.zeros_like(gt, dtype=np.uint8)
        return OraclePick(metrics=_direct_metrics(zero, gt), candidate_index=-1, candidate_count=0)

    best_idx = -1
    best_metrics: DirectMetrics | None = None
    for idx, cand in enumerate(candidates):
        met = _direct_metrics(cand, gt)
        if best_metrics is None or (met.iou, met.ari) > (best_metrics.iou, best_metrics.ari):
            best_metrics = met
            best_idx = idx
    assert best_metrics is not None
    return OraclePick(metrics=best_metrics, candidate_index=best_idx, candidate_count=len(candidates))


def _load_common182_ids(csv_path: Path) -> set[int]:
    common: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("TextureSAM0p3_present", "").strip().lower() == "true":
                common.add(int(row["image_id"]))
    return common


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def _summarize(rows: list[dict[str, object]], key_prefix: str) -> dict[str, float | int]:
    return {
        "miou": _mean(rows, f"{key_prefix}_iou"),
        "ari": _mean(rows, f"{key_prefix}_ari"),
        "count": int(len(rows)),
    }


def main() -> None:
    args = parse_args()
    images_dir = args.benchmark_root / "images"
    labels_dir = args.benchmark_root / "labels"
    image_ids = sorted(int(p.stem) for p in images_dir.glob("*.png"))
    common_ids = _load_common182_ids(args.texturesam_maskbank_csv)
    store = PromptMaskProposalStore(args.prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))

    rows: list[dict[str, object]] = []
    for i, image_id in enumerate(image_ids, start=1):
        image_rgb = read_image_rgb(images_dir / f"{image_id}.png")
        gt = ensure_binary_gt(read_mask_raw(labels_dir / f"{image_id}.png"), strict=False)
        shape = gt.shape

        proposals = store.load(image_id, expected_shape=shape)
        single_candidates = _dedupe_masks(proposals, min_area=int(args.min_area))
        component_candidates = _compatible_components(image_rgb, proposals, int(args.min_area))
        topk_candidates = _topk_unions(proposals, min_area=int(args.min_area), max_k=6)
        final_mask = _read_mask(args.final_masks_root / f"{image_id}.png", shape)
        final_pick = OraclePick(metrics=_direct_metrics(final_mask, gt), candidate_index=0, candidate_count=1)
        single_pick = _select_best(single_candidates, gt)
        component_pick = _select_best(component_candidates, gt)
        topk_pick = _select_best(topk_candidates, gt)

        upper_candidates = [final_mask]
        upper_candidates.extend(single_candidates)
        upper_candidates.extend(component_candidates)
        upper_candidates.extend(topk_candidates)
        upper_candidates = _dedupe_masks(upper_candidates, min_area=int(args.min_area))
        upper_pick = _select_best(upper_candidates, gt)

        rows.append(
            {
                "image_id": int(image_id),
                "in_common182": bool(image_id in common_ids),
                "proposal_count": int(len(proposals)),
                "single_candidate_count": int(len(single_candidates)),
                "component_candidate_count": int(len(component_candidates)),
                "topk_candidate_count": int(len(topk_candidates)),
                "upper_candidate_count": int(len(upper_candidates)),
                "final_iou": float(final_pick.metrics.iou),
                "final_ari": float(final_pick.metrics.ari),
                "single_iou": float(single_pick.metrics.iou),
                "single_ari": float(single_pick.metrics.ari),
                "component_iou": float(component_pick.metrics.iou),
                "component_ari": float(component_pick.metrics.ari),
                "topk_iou": float(topk_pick.metrics.iou),
                "topk_ari": float(topk_pick.metrics.ari),
                "upper_iou": float(upper_pick.metrics.iou),
                "upper_ari": float(upper_pick.metrics.ari),
            }
        )
        if i % 32 == 0:
            print(f"[stld-proposal-oracles] processed {i}/{len(image_ids)} images", flush=True)

    all_rows = rows
    common_rows = [row for row in rows if bool(row["in_common182"])]

    summary = {
        "num_images": int(len(all_rows)),
        "common182_count": int(len(common_rows)),
        "benchmark_root": str(args.benchmark_root.resolve()),
        "prompt_masks_root": str(args.prompt_masks_root.resolve()),
        "final_masks_root": str(args.final_masks_root.resolve()),
        "texturesam_maskbank_csv": str(args.texturesam_maskbank_csv.resolve()),
        "methods": [
            {
                "method": "final_release",
                "all200_miou": _summarize(all_rows, "final")["miou"],
                "all200_ari": _summarize(all_rows, "final")["ari"],
                "common182_miou": _summarize(common_rows, "final")["miou"],
                "common182_ari": _summarize(common_rows, "final")["ari"],
            },
            {
                "method": "single_proposal_oracle",
                "all200_miou": _summarize(all_rows, "single")["miou"],
                "all200_ari": _summarize(all_rows, "single")["ari"],
                "common182_miou": _summarize(common_rows, "single")["miou"],
                "common182_ari": _summarize(common_rows, "single")["ari"],
            },
            {
                "method": "compatible_component_oracle",
                "all200_miou": _summarize(all_rows, "component")["miou"],
                "all200_ari": _summarize(all_rows, "component")["ari"],
                "common182_miou": _summarize(common_rows, "component")["miou"],
                "common182_ari": _summarize(common_rows, "component")["ari"],
            },
            {
                "method": "topk_union_oracle",
                "all200_miou": _summarize(all_rows, "topk")["miou"],
                "all200_ari": _summarize(all_rows, "topk")["ari"],
                "common182_miou": _summarize(common_rows, "topk")["miou"],
                "common182_ari": _summarize(common_rows, "topk")["ari"],
            },
            {
                "method": "bank_upper_bound_oracle",
                "all200_miou": _summarize(all_rows, "upper")["miou"],
                "all200_ari": _summarize(all_rows, "upper")["ari"],
                "common182_miou": _summarize(common_rows, "upper")["miou"],
                "common182_ari": _summarize(common_rows, "upper")["ari"],
            },
        ],
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    per_image_csv = args.out_root / "stld_proposal_oracles_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [
                "image_id",
                "in_common182",
                "proposal_count",
                "single_candidate_count",
                "component_candidate_count",
                "topk_candidate_count",
                "upper_candidate_count",
                "final_iou",
                "final_ari",
                "single_iou",
                "single_ari",
                "component_iou",
                "component_ari",
                "topk_iou",
                "topk_ari",
                "upper_iou",
                "upper_ari",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary["per_image_csv"] = str(per_image_csv.resolve())
    summary_json = args.out_root / "stld_proposal_oracles_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
