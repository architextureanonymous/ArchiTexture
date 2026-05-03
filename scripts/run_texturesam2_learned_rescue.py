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

from texturesam_v2.features import compute_texture_feature_map
from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, infer_rwtd_dirs, list_rwtd_images, read_image_rgb, read_mask_raw, write_binary_mask
from texturesam_v2.metrics import rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig
from texturesam_v2.ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from texturesam_v2.ptd_v8_partition import PTDV8PartitionScorer

DEFAULT_V8_BUNDLE = ROOT / "artifacts/ptd_v8_partition_bundle_sanitized_cuda.pkl"
if not DEFAULT_V8_BUNDLE.exists():
    DEFAULT_V8_BUNDLE = ROOT / "artifacts/ptd_v8_partition_bundle.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "TextureSAM-2 learned rescue: keep the v9 PTD-trained core mask unless a denser "
            "candidate set is clearly better under the PTD-trained v8 partition scorer."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--dense-prompt-masks-root", type=Path, required=True)
    p.add_argument("--v9-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--ptd-checkpoint", type=Path, default=ROOT / "artifacts/ptd_convnext_tiny.pt")
    p.add_argument("--ptd-v8-bundle", type=Path, default=DEFAULT_V8_BUNDLE)
    p.add_argument("--ptd-device", type=str, default="cuda")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--image-ids-file", type=Path, default=None)

    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--cov9-low", type=float, default=0.20)
    p.add_argument("--cov-gain-min", type=float, default=0.22)
    p.add_argument("--prec-best-min", type=float, default=0.35)
    p.add_argument("--area-ratio-min", type=float, default=0.05)
    p.add_argument("--area-ratio-max", type=float, default=2.2)
    p.add_argument("--learned-margin", type=float, default=0.0)
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


def _coverage_vs_union(mask: np.ndarray, union: np.ndarray) -> tuple[float, float, float]:
    m = mask > 0
    u = union > 0
    ma = int(m.sum())
    ua = int(u.sum())
    if ma <= 0 or ua <= 0:
        return 0.0, 0.0, 0.0
    inter = float(np.logical_and(m, u).sum())
    cov = inter / max(ua, 1)
    prec = inter / max(ma, 1)
    area_u = ma / max(ua, 1)
    return float(cov), float(prec), float(area_u)


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
    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=args.ptd_checkpoint, device=args.ptd_device))
    scorer = PTDV8PartitionScorer(args.ptd_v8_bundle)

    out_dir = args.out_root / "texturesam2_learned_rescue"
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
            switched_flag = 0
            source = "v9_no_proposals"
            v9_cov = 0.0
            best_cov = 0.0
            best_prec = 0.0
            best_area_u = 0.0
            v9_score = 0.0
            best_score = 0.0
            candidate_count = 1
        else:
            proposal_union = np.zeros_like(proposals[0], dtype=np.uint8)
            for p in proposals:
                proposal_union = np.logical_or(proposal_union > 0, p > 0).astype(np.uint8)

            feat_map = compute_texture_feature_map(image)
            descriptors = encoder.encode_regions(image, proposals)

            learned_components, _, _ = scorer.merge_components(
                image_rgb=image,
                proposals=proposals,
                descriptors=descriptors,
                feature_map=feat_map,
            )

            cands = [v9]
            cands.extend(learned_components)
            cands.extend(_freq_candidates(proposals, min_area=args.min_area))
            cands.extend(_topk_unions(proposals, min_area=args.min_area, max_k=6))
            cands = _dedupe(cands, min_area=args.min_area)
            if not cands:
                cands = [v9]

            scores = scorer.score_components(
                image_rgb=image,
                components=cands,
                proposals=proposals,
                descriptors=descriptors,
                feature_map=feat_map,
            )
            if len(scores) != len(cands):
                raise RuntimeError(f"score count mismatch on image {image_id}: {len(scores)} vs {len(cands)}")

            v9_index = 0
            best_index = int(np.argmax(scores))
            best = cands[best_index]
            v9_score = float(scores[v9_index])
            best_score = float(scores[best_index])

            v9_cov, _, _ = _coverage_vs_union(v9, proposal_union)
            best_cov, best_prec, best_area_u = _coverage_vs_union(best, proposal_union)

            switched_flag = 0
            source = "v9_keep"
            if not np.array_equal(best, v9):
                strong = (
                    v9_cov < args.cov9_low
                    and (best_cov - v9_cov) > args.cov_gain_min
                    and best_prec > args.prec_best_min
                    and best_area_u > args.area_ratio_min
                    and best_area_u < args.area_ratio_max
                    and (best_score - v9_score) > args.learned_margin
                )
                if strong:
                    switched_flag = 1
                    source = "learned_rescue"

            pred = best if switched_flag == 1 else v9
            switched += int(switched_flag)
            candidate_count = len(cands)

        write_binary_mask(out_masks / f"{image_id}.png", pred)
        met = rwtd_invariant_metrics(pred, gt)
        met9 = rwtd_invariant_metrics(v9, gt)
        rows.append(
            {
                "image_id": image_id,
                "v12_iou": float(met.iou),
                "v12_ari": float(met.ari),
                "v9_iou": float(met9.iou),
                "v9_ari": float(met9.ari),
                "delta_iou_vs_v9": float(met.iou - met9.iou),
                "delta_ari_vs_v9": float(met.ari - met9.ari),
                "proposal_count_dense": int(len(proposals)),
                "candidate_count": int(candidate_count),
                "source": source,
                "switched": int(switched_flag),
                "v9_cov_union": float(v9_cov),
                "best_cov_union": float(best_cov),
                "best_prec_union": float(best_prec),
                "best_area_u": float(best_area_u),
                "v9_learned_score": float(v9_score),
                "best_learned_score": float(best_score),
            }
        )
        if i % 32 == 0:
            print(f"[texturesam2_learned_rescue] processed {i}/{len(images)} images")

    def _mean(k: str) -> float:
        return float(np.mean([float(r[k]) for r in rows])) if rows else 0.0

    summary = {
        "dataset": "rwtd_kaust256",
        "num_images": int(len(rows)),
        "protocol": {
            "name": "texturesam2_learned_rescue",
            "uses_rwtd_labels_for_training": False,
            "uses_rwtd_labels_for_hyperparameter_search": False,
            "uses_rwtd_labels_for_final_metric_reporting": True,
            "external_training_data": "PTD only",
            "frozen_after_ptd_validation_only": True,
        },
        "paths": {
            "dense_prompt_masks_root": str(args.dense_prompt_masks_root),
            "v9_masks_root": str(args.v9_masks_root),
            "rwtd_root": str(args.rwtd_root),
            "ptd_checkpoint": str(args.ptd_checkpoint),
            "ptd_v8_bundle": str(args.ptd_v8_bundle),
        },
        "metrics": {
            "v12_miou": _mean("v12_iou"),
            "v12_ari": _mean("v12_ari"),
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
            "area_ratio_min": float(args.area_ratio_min),
            "area_ratio_max": float(args.area_ratio_max),
            "learned_margin": float(args.learned_margin),
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
