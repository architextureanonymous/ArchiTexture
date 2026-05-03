#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.features import compute_texture_feature_map, mean_feature, region_variance
from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, infer_rwtd_dirs, list_rwtd_images, read_image_rgb, read_mask_raw, write_binary_mask
from texturesam_v2.metrics import rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig
from texturesam_v2.ptd_v8_partition import _coverage_vs_union


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TextureSAM-v11 dense rescue: v9 baseline with conservative switch to dense-proposal "
            "consensus/union candidates only when under-coverage evidence is strong."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--dense-prompt-masks-root", type=Path, required=True)
    p.add_argument("--v9-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--image-ids-file", type=Path, default=None)

    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--cov9-low", type=float, default=0.20)
    p.add_argument("--cov-gain-min", type=float, default=0.22)
    p.add_argument("--prec-best-min", type=float, default=0.35)
    p.add_argument("--score-margin", type=float, default=0.22)
    p.add_argument("--area-ratio-min", type=float, default=0.05)
    p.add_argument("--area-ratio-max", type=float, default=2.2)
    p.add_argument("--w-cov", type=float, default=0.60)
    p.add_argument("--w-prec", type=float, default=0.18)
    p.add_argument("--w-delta", type=float, default=0.10)
    p.add_argument("--w-var", type=float, default=0.08)
    p.add_argument("--w-frag", type=float, default=0.10)
    return p.parse_args()


def _load_image_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keep: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        keep.add(str(int(s)))
    return keep or None


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    return ensure_binary(read_mask_raw(path))


def _dedupe(masks: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for m in masks:
        b = (m > 0).astype(np.uint8)
        if int(b.sum()) < min_area:
            continue
        k = np.packbits(b, axis=None).tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out


def _freq_candidates(proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    if not proposals:
        return []
    stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
    freq = stack.mean(axis=0)
    out: list[np.ndarray] = []
    for t in (0.10, 0.16, 0.24, 0.32, 0.42, 0.56, 0.70):
        m = (freq >= t).astype(np.uint8)
        if int(m.sum()) >= min_area:
            out.append(m)
    return _dedupe(out, min_area)


def _topk_unions(proposals: list[np.ndarray], min_area: int, max_k: int = 6) -> list[np.ndarray]:
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
            v = 1.0 if uni <= 0 else inter / uni
            mat[i, j] = v
            mat[j, i] = v
    central = mat.mean(axis=1)
    order = np.argsort(central)[::-1].tolist()

    out: list[np.ndarray] = []
    for k in range(2, min(max_k, n) + 1):
        u = np.zeros_like(proposals[0], dtype=np.uint8)
        for idx in order[:k]:
            u = np.logical_or(u > 0, proposals[idx] > 0)
        if int(u.sum()) >= min_area:
            out.append(u.astype(np.uint8))
    return _dedupe(out, min_area)


def _frag(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    n_cc, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cc_pen = max(0, n_cc - 2)
    inv = 1 - m
    n_h, lbl_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    holes = 0
    if n_h > 1:
        h, w = m.shape
        border = set(np.unique(lbl_h[0, :]).tolist())
        border.update(np.unique(lbl_h[h - 1, :]).tolist())
        border.update(np.unique(lbl_h[:, 0]).tolist())
        border.update(np.unique(lbl_h[:, w - 1]).tolist())
        for c in range(1, n_h):
            if c in border:
                continue
            if int(stats_h[c, cv2.CC_STAT_AREA]) > 0:
                holes += 1
    return float(cc_pen + 0.5 * holes)


def _score_candidate(
    mask: np.ndarray,
    union: np.ndarray,
    feat_map: np.ndarray,
    *,
    w_cov: float,
    w_prec: float,
    w_delta: float,
    w_var: float,
    w_frag: float,
) -> tuple[float, float, float, float]:
    m = (mask > 0).astype(np.uint8)
    cov, prec, area_vs_u = _coverage_vs_union(m, union)
    inside = mean_feature(feat_map, m)
    outside = mean_feature(feat_map, 1 - m)
    delta = float(np.linalg.norm(inside - outside))
    var = float(region_variance(feat_map, m))
    frag = _frag(m)
    score = w_cov * cov + w_prec * prec + w_delta * delta - w_var * var - w_frag * frag
    return float(score), float(cov), float(prec), float(area_vs_u)


def main() -> None:
    args = parse_args()
    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    images = list_rwtd_images(image_dir)
    keep_ids = _load_image_id_filter(args.image_ids_file)
    if keep_ids is not None:
        images = [p for p in images if p.stem in keep_ids]
    if args.max_images is not None:
        images = images[: int(args.max_images)]

    store = PromptMaskProposalStore(args.dense_prompt_masks_root, ProposalLoadConfig(min_area=args.min_area))

    out_dir = args.out_root / "strict_ptd_v11_dense_rescue"
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    switched = 0
    for i, image_path in enumerate(images, start=1):
        image_id = image_path.stem
        id_num = int(image_id)
        image = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_id}.png"), source_name=str(label_dir / f"{image_id}.png"))

        v9 = _read_mask(args.v9_masks_root / f"{image_id}.png", image.shape[:2])
        proposals = store.load(id_num, expected_shape=image.shape[:2])
        if not proposals:
            pred = v9
            source = "v9_no_proposals"
            sw = 0
            v9_cov = 0.0
            best_cov = 0.0
            best_prec = 0.0
            best_area_u = 0.0
        else:
            union = np.zeros_like(proposals[0], dtype=np.uint8)
            for p in proposals:
                union = np.logical_or(union > 0, p > 0).astype(np.uint8)

            feat_map = compute_texture_feature_map(image)
            s9, v9_cov, v9_prec, v9_area_u = _score_candidate(
                v9,
                union,
                feat_map,
                w_cov=args.w_cov,
                w_prec=args.w_prec,
                w_delta=args.w_delta,
                w_var=args.w_var,
                w_frag=args.w_frag,
            )

            cands = [v9]
            cands.extend(_freq_candidates(proposals, min_area=args.min_area))
            cands.extend(_topk_unions(proposals, min_area=args.min_area, max_k=6))
            cands = _dedupe(cands, min_area=args.min_area)
            if not cands:
                cands = [v9]

            best = v9
            best_score = s9
            best_cov = v9_cov
            best_prec = v9_prec
            best_area_u = v9_area_u
            for cm in cands:
                sc, cov, prec, area_u = _score_candidate(
                    cm,
                    union,
                    feat_map,
                    w_cov=args.w_cov,
                    w_prec=args.w_prec,
                    w_delta=args.w_delta,
                    w_var=args.w_var,
                    w_frag=args.w_frag,
                )
                if sc > best_score:
                    best = cm
                    best_score = sc
                    best_cov = cov
                    best_prec = prec
                    best_area_u = area_u

            sw = 0
            if not np.array_equal(best, v9):
                strong = (
                    v9_cov < args.cov9_low
                    and (best_cov - v9_cov) > args.cov_gain_min
                    and best_prec > args.prec_best_min
                    and (best_score - s9) > args.score_margin
                    and best_area_u > args.area_ratio_min
                    and best_area_u < args.area_ratio_max
                )
                if strong:
                    sw = 1
            pred = best if sw == 1 else v9
            source = "dense_rescue" if sw == 1 else "v9_keep"

        write_binary_mask(out_masks / f"{image_id}.png", pred)
        met = rwtd_invariant_metrics(pred, gt)
        met9 = rwtd_invariant_metrics(v9, gt)
        rows.append(
            {
                "image_id": image_id,
                "v11_iou": float(met.iou),
                "v11_ari": float(met.ari),
                "v9_iou": float(met9.iou),
                "v9_ari": float(met9.ari),
                "delta_iou_vs_v9": float(met.iou - met9.iou),
                "delta_ari_vs_v9": float(met.ari - met9.ari),
                "proposal_count_dense": int(len(proposals)),
                "source": source,
                "switched": int(sw),
                "v9_cov_union": float(v9_cov),
                "best_cov_union": float(best_cov),
                "best_prec_union": float(best_prec),
                "best_area_u": float(best_area_u),
            }
        )
        switched += int(sw)
        if i % 32 == 0:
            print(f"[v11_dense_rescue] processed {i}/{len(images)} images")

    def _mean(k: str) -> float:
        return float(np.mean([float(r[k]) for r in rows])) if rows else 0.0

    summary = {
        "dataset": "rwtd_kaust256",
        "num_images": int(len(rows)),
        "protocol": {
            "name": "strict_ptd_v11_dense_rescue",
            "uses_rwtd_labels_for_training": False,
            "uses_rwtd_labels_for_hyperparameter_search": False,
            "uses_rwtd_labels_for_final_metric_reporting": True,
            "external_training_data": "None (inference-only conservative dense rescue)",
            "frozen_after_ptd_validation_only": True,
        },
        "paths": {
            "dense_prompt_masks_root": str(args.dense_prompt_masks_root),
            "v9_masks_root": str(args.v9_masks_root),
            "rwtd_root": str(args.rwtd_root),
        },
        "metrics": {
            "v11_miou": _mean("v11_iou"),
            "v11_ari": _mean("v11_ari"),
            "v9_miou": _mean("v9_iou"),
            "v9_ari": _mean("v9_ari"),
            "delta_miou_vs_v9": _mean("delta_iou_vs_v9"),
            "delta_ari_vs_v9": _mean("delta_ari_vs_v9"),
        },
        "switching": {
            "switched_count": int(switched),
            "switch_rate": float(switched / max(len(rows), 1)),
        },
        "thresholds": {
            "cov9_low": float(args.cov9_low),
            "cov_gain_min": float(args.cov_gain_min),
            "prec_best_min": float(args.prec_best_min),
            "score_margin": float(args.score_margin),
            "area_ratio_min": float(args.area_ratio_min),
            "area_ratio_max": float(args.area_ratio_max),
        },
        "weights": {
            "w_cov": float(args.w_cov),
            "w_prec": float(args.w_prec),
            "w_delta": float(args.w_delta),
            "w_var": float(args.w_var),
            "w_frag": float(args.w_frag),
        },
        "artifacts": {
            "summary_json": str((out_dir / "summary.json").resolve()),
            "per_image_csv": str((out_dir / "per_image.csv").resolve()),
            "masks_dir": str(out_masks.resolve()),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
