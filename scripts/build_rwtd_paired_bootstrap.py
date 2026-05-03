#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute paired bootstrap support for RWTD using the same official no-aggregation "
            "per-instance evaluator that backs the main paper table."
        )
    )
    p.add_argument("--architexture-common-root", type=Path, required=True)
    p.add_argument("--texturesam-common-root", type=Path, required=True)
    p.add_argument("--architexture-full-root", type=Path, required=True)
    p.add_argument("--sam-full-root", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--num-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    samples = (a[idx] - b[idx]).mean(axis=1)
    delta = a - b
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "prob_delta_gt_zero": float(np.mean(samples > 0.0)),
    }


def _load_rwtd_instance_metrics(
    pred_folder: Path,
    gt_folder: Path,
    upstream_root: Path,
) -> tuple[dict[tuple[int, int], dict[str, float]], set[int], list[tuple[int, int]]]:
    sys.path.insert(0, str(upstream_root.resolve()))
    import eval_no_agg_masks as noagg  # type: ignore

    gt_masks_by_image = noagg.load_gt_masks_as_instances(str(gt_folder))
    pred_masks_by_image = noagg.load_masks_from_folder(str(pred_folder))
    pred_image_ids = {int(k) for k in pred_masks_by_image.keys()}
    metrics_results, _, _ = noagg.calculate_metrics_per_gt(pred_masks_by_image, gt_masks_by_image)

    all_keys: list[tuple[int, int]] = []
    for image_id, inst_list in gt_masks_by_image.items():
        iid = int(image_id)
        for gt_id, _ in inst_list:
            all_keys.append((iid, int(gt_id)))

    out: dict[tuple[int, int], dict[str, float]] = {}
    for key in all_keys:
        if key in metrics_results:
            item = metrics_results[key]
            out[key] = {
                "iou": float(item.get("average_iou", 0.0)),
                "ari": float(item.get("average_rand_index", 0.0)),
                "overlap": float(1 if len(item.get("all_iou", [])) > 0 else 0),
            }
        else:
            out[key] = {"iou": 0.0, "ari": 0.0, "overlap": 0.0}
    return out, pred_image_ids, all_keys


def _load_rwtd_image_metrics(
    pred_folder: Path,
    gt_folder: Path,
    upstream_root: Path,
) -> tuple[dict[int, dict[str, float]], set[int]]:
    sys.path.insert(0, str(upstream_root.resolve()))
    import eval_no_agg_masks as noagg  # type: ignore

    gt_masks_by_image = noagg.load_gt_masks_as_instances(str(gt_folder))
    pred_masks_by_image = noagg.load_masks_from_folder(str(pred_folder))
    pred_image_ids = {int(k) for k in pred_masks_by_image.keys()}
    metrics_results, _, _ = noagg.calculate_metrics_per_gt(pred_masks_by_image, gt_masks_by_image)

    all_keys: list[tuple[int, int]] = []
    for image_id, inst_list in gt_masks_by_image.items():
        iid = int(image_id)
        for gt_id, _ in inst_list:
            all_keys.append((iid, int(gt_id)))

    fixed_instance: dict[tuple[int, int], dict[str, float]] = {}
    for key in all_keys:
        if key in metrics_results:
            item = metrics_results[key]
            fixed_instance[key] = {
                "iou": float(item.get("average_iou", 0.0)),
                "ari": float(item.get("average_rand_index", 0.0)),
            }
        else:
            fixed_instance[key] = {"iou": 0.0, "ari": 0.0}

    per_image_keys: dict[int, list[tuple[int, int]]] = {}
    for key in all_keys:
        per_image_keys.setdefault(key[0], []).append(key)

    out: dict[int, dict[str, float]] = {}
    for iid, keys in per_image_keys.items():
        out[iid] = {
            "iou": float(np.mean([fixed_instance[key]["iou"] for key in keys])),
            "ari": float(np.mean([fixed_instance[key]["ari"] for key in keys])),
        }
    return out, pred_image_ids


def _summarize_overlap_instances(
    *,
    a_name: str,
    b_name: str,
    a_root: Path,
    b_root: Path,
    gt_root: Path,
    upstream_root: Path,
    allowed_image_ids: set[int] | None,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, object]:
    a, a_pred_ids, all_keys = _load_rwtd_instance_metrics(a_root, gt_root, upstream_root)
    b, b_pred_ids, _ = _load_rwtd_instance_metrics(b_root, gt_root, upstream_root)
    shared_pred_ids = a_pred_ids & b_pred_ids
    if allowed_image_ids is not None:
        shared_pred_ids &= set(int(i) for i in allowed_image_ids)
    keys = [
        key
        for key in all_keys
        if key[0] in shared_pred_ids and int(a[key]["overlap"]) == 1 and int(b[key]["overlap"]) == 1
    ]
    if not keys:
        raise RuntimeError(f"No overlap-instance keys between {a_root} and {b_root}.")

    a_iou = np.asarray([a[key]["iou"] for key in keys], dtype=np.float64)
    a_ari = np.asarray([a[key]["ari"] for key in keys], dtype=np.float64)
    b_iou = np.asarray([b[key]["iou"] for key in keys], dtype=np.float64)
    b_ari = np.asarray([b[key]["ari"] for key in keys], dtype=np.float64)

    return {
        "count": int(len(keys)),
        "a_name": a_name,
        "b_name": b_name,
        "a_root": str(a_root.resolve()),
        "b_root": str(b_root.resolve()),
        "note": (
            "Paired bootstrap over GT instances overlapped by both methods. "
            "This matches the official overlap-significance view used to support the "
            "common-253 comparison, rather than a per-image mean that changes the denominator."
        ),
        "delta_a_minus_b": {
            "miou": _bootstrap_delta(a_iou, b_iou, rng, n_boot),
            "ari": _bootstrap_delta(a_ari, b_ari, rng, n_boot),
        },
    }


def _summarize_pair(
    *,
    a_name: str,
    b_name: str,
    a_root: Path,
    b_root: Path,
    gt_root: Path,
    upstream_root: Path,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, object]:
    a, a_pred_ids = _load_rwtd_image_metrics(a_root, gt_root, upstream_root)
    b, b_pred_ids = _load_rwtd_image_metrics(b_root, gt_root, upstream_root)
    ids = sorted((set(a) & set(b)) & a_pred_ids & b_pred_ids)
    if not ids:
        raise RuntimeError(f"No overlapping RWTD image ids between {a_root} and {b_root}.")

    a_iou = np.asarray([a[i]["iou"] for i in ids], dtype=np.float64)
    a_ari = np.asarray([a[i]["ari"] for i in ids], dtype=np.float64)
    b_iou = np.asarray([b[i]["iou"] for i in ids], dtype=np.float64)
    b_ari = np.asarray([b[i]["ari"] for i in ids], dtype=np.float64)

    return {
        "count": int(len(ids)),
        "a_name": a_name,
        "b_name": b_name,
        "a_root": str(a_root.resolve()),
        "b_root": str(b_root.resolve()),
        "a": {
            "miou": _bootstrap_mean(a_iou, rng, n_boot),
            "ari": _bootstrap_mean(a_ari, rng, n_boot),
        },
        "b": {
            "miou": _bootstrap_mean(b_iou, rng, n_boot),
            "ari": _bootstrap_mean(b_ari, rng, n_boot),
        },
        "delta_a_minus_b": {
            "miou": _bootstrap_delta(a_iou, b_iou, rng, n_boot),
            "ari": _bootstrap_delta(a_ari, b_ari, rng, n_boot),
        },
    }


def main() -> None:
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    payload = {
        "config": {
            "num_bootstrap": int(args.num_bootstrap),
            "seed": int(args.seed),
            "gt_root": str(args.gt_root.resolve()),
            "upstream_root": str(args.upstream_root.resolve()),
        },
        "rwtd_common253_overlap_instances": _summarize_overlap_instances(
            a_name="ArchiTexture",
            b_name="TextureSAM-0.3",
            a_root=args.architexture_common_root,
            b_root=args.texturesam_common_root,
            gt_root=args.gt_root,
            upstream_root=args.upstream_root,
            allowed_image_ids=None,
            rng=rng,
            n_boot=int(args.num_bootstrap),
        ),
        "rwtd_common253_per_image_aux": _summarize_pair(
            a_name="ArchiTexture",
            b_name="TextureSAM-0.3",
            a_root=args.architexture_common_root,
            b_root=args.texturesam_common_root,
            gt_root=args.gt_root,
            upstream_root=args.upstream_root,
            rng=rng,
            n_boot=int(args.num_bootstrap),
        ),
        "rwtd_full256_per_image_aux": _summarize_pair(
            a_name="ArchiTexture",
            b_name="SAM2.1-small",
            a_root=args.architexture_full_root,
            b_root=args.sam_full_root,
            gt_root=args.gt_root,
            upstream_root=args.upstream_root,
            rng=rng,
            n_boot=int(args.num_bootstrap),
        ),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
