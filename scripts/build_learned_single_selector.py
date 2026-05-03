#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig, TextureSAMV2Consolidator, _postprocess
from texturesam_v2.features import compute_texture_feature_map
from texturesam_v2.io_utils import (
    ensure_binary_gt,
    infer_rwtd_dirs,
    list_rwtd_images,
    read_image_rgb,
    read_mask_raw,
    write_binary_mask,
)
from texturesam_v2.merge import MergeConfig
from texturesam_v2.metrics import ari_score_binary, iou_score, rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig


FIXED_MERGE = MergeConfig(
    adjacency_dilation=3,
    merge_threshold=0.50,
    w_texture=0.65,
    w_boundary=0.30,
    w_hetero=0.20,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a learned single-proposal selector that reuses the same Stage-A "
            "proposal descriptors and learned score model as ArchiTexture, but restricts "
            "inference to singleton proposals only."
        )
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def add_shared(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--prompt-masks-root", type=Path, required=True)
        sp.add_argument("--out-root", type=Path, required=True)
        sp.add_argument("--ptd-encoder-ckpt", type=Path, required=True)
        sp.add_argument("--ptd-learned-bundle", type=Path, required=True)
        sp.add_argument(
            "--descriptor-mode",
            type=str,
            default="hybrid_ptd",
            choices=["handcrafted", "ptd_convnext", "hybrid_ptd"],
        )
        sp.add_argument("--ptd-device", type=str, default="cuda")
        sp.add_argument("--min-area", type=int, default=32)
        sp.add_argument("--ptd-use-ring-context", dest="ptd_use_ring_context", action="store_true")
        sp.add_argument("--no-ptd-use-ring-context", dest="ptd_use_ring_context", action="store_false")
        sp.set_defaults(ptd_use_ring_context=True)
        sp.add_argument("--ptd-ring-dilation", type=int, default=9)
        sp.add_argument("--ptd-ring-min-pixels", type=int, default=24)

    rwtd = sub.add_parser("rwtd", help="Evaluate the learned single selector on RWTD.")
    add_shared(rwtd)
    rwtd.add_argument("--rwtd-root", type=Path, required=True)
    rwtd.add_argument("--baseline-official-root", type=Path, required=True)
    rwtd.add_argument("--upstream-root", type=Path, required=True)

    stld = sub.add_parser("stld", help="Evaluate the learned single selector on STLD.")
    add_shared(stld)
    stld.add_argument("--benchmark-root", type=Path, required=True)
    stld.add_argument("--texturesam-maskbank-csv", type=Path, required=True)

    return p.parse_args()


def _cfg(args: argparse.Namespace) -> ConsolidationConfig:
    return ConsolidationConfig(
        min_area=int(args.min_area),
        close_kernel=5,
        keep_largest_component=True,
        hole_area_threshold=64,
        descriptor_mode=str(args.descriptor_mode),
        ptd_checkpoint=args.ptd_encoder_ckpt,
        ptd_device=args.ptd_device,
        ptd_use_ring_context=bool(args.ptd_use_ring_context),
        ptd_ring_dilation=int(args.ptd_ring_dilation),
        ptd_ring_min_pixels=int(args.ptd_ring_min_pixels),
        learned_bundle=args.ptd_learned_bundle,
        merge=FIXED_MERGE,
        objective_lambda=0.45,
        objective_mu=0.30,
    )


def _clean_export_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _select_single(
    *,
    consolidator: TextureSAMV2Consolidator,
    image_rgb: np.ndarray,
    proposals: list[np.ndarray],
) -> tuple[np.ndarray, int, float, int]:
    shape = image_rgb.shape[:2]
    clean_props: list[np.ndarray] = []
    for p in proposals:
        m = (p > 0).astype(np.uint8)
        if int(m.sum()) >= int(consolidator.cfg.min_area):
            clean_props.append(m)
    if not clean_props:
        return np.zeros(shape, dtype=np.uint8), -1, float("-inf"), 0

    feature_map = compute_texture_feature_map(image_rgb)
    descriptors = consolidator._build_descriptors(image_rgb=image_rgb, feat_map=feature_map, proposals=clean_props)
    if consolidator.learned_models is None:
        raise RuntimeError("Expected a learned Stage-A bundle, but no learned model was loaded.")
    scores = consolidator.learned_models.score_components(
        image_rgb=image_rgb,
        components=clean_props,
        proposals=clean_props,
        descriptors=descriptors,
        feature_map=feature_map,
    )
    if len(scores) != len(clean_props):
        raise RuntimeError(f"Singleton score mismatch: {len(scores)} scores for {len(clean_props)} proposals.")
    idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    pred = _postprocess(clean_props[idx], consolidator.cfg).astype(np.uint8)
    return pred, idx, float(scores[idx]), len(clean_props)


def _load_common182_ids(csv_path: Path) -> set[int]:
    out: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("TextureSAM0p3_present", "").strip().lower() == "true":
                out.add(int(row["image_id"]))
    return out


def _parse_rwtd_common_ids(mask_root: Path) -> set[int]:
    out: set[int] = set()
    for p in mask_root.glob("mask_*_*.png"):
        try:
            out.add(int(p.stem.split("_")[-1]))
        except Exception:
            continue
    return out


def _run_rwtd_eval(pred: Path, gt_dir: Path, upstream_root: Path, out_json: Path) -> dict:
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


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def run_rwtd(args: argparse.Namespace) -> None:
    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    gt_dir = args.upstream_root / "Kaust256" / "labeles"
    images = list_rwtd_images(image_dir)
    common_ids = _parse_rwtd_common_ids(args.baseline_official_root)
    store = PromptMaskProposalStore(args.prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))
    consolidator = TextureSAMV2Consolidator(_cfg(args))

    full_dir = args.out_root / "official_export"
    common_dir = args.out_root / "official_export_common253"
    _clean_export_dir(full_dir)
    _clean_export_dir(common_dir)

    rows: list[dict[str, object]] = []
    for i, image_path in enumerate(images, start=1):
        image_id = image_path.stem
        image_num = int(image_id.split("_")[-1])
        image_rgb = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_id}.png"), strict=False)
        proposals = store.load(image_num, expected_shape=gt.shape)

        pred, selected_index, selected_score, proposal_count = _select_single(
            consolidator=consolidator,
            image_rgb=image_rgb,
            proposals=proposals,
        )
        met = rwtd_invariant_metrics(pred, gt)
        out_name = f"mask_0_{image_num}.png"
        write_binary_mask(full_dir / out_name, pred)
        if image_num in common_ids:
            write_binary_mask(common_dir / out_name, pred)

        rows.append(
            {
                "image_id": int(image_num),
                "in_common253": bool(image_num in common_ids),
                "proposal_count": int(proposal_count),
                "selected_index": int(selected_index),
                "selected_score": float(selected_score),
                "rwtd_iou": float(met.iou),
                "rwtd_ari": float(met.ari),
            }
        )
        if i % 32 == 0:
            print(f"[learned-single-selector:rwtd] processed {i}/{len(images)} images", flush=True)

    full_json = args.out_root / "official_eval_full256.json"
    common_json = args.out_root / "official_eval_common253.json"
    full_eval = _run_rwtd_eval(full_dir, gt_dir, args.upstream_root, full_json)
    common_eval = _run_rwtd_eval(common_dir, gt_dir, args.upstream_root, common_json)

    per_image_csv = args.out_root / "rwtd_learned_single_selector_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "rwtd",
        "selector": "learned_stageA_singleton",
        "num_images": int(len(rows)),
        "common253_count": int(sum(1 for row in rows if bool(row["in_common253"]))),
        "ptd_encoder_ckpt": str(args.ptd_encoder_ckpt.resolve()),
        "ptd_learned_bundle": str(args.ptd_learned_bundle.resolve()),
        "descriptor_mode": str(args.descriptor_mode),
        "methods": [
            {
                "method": "learned_single_selector",
                "full256_miou": float(full_eval["noagg_official"]["overall_average_iou"]),
                "full256_ari": float(full_eval["noagg_official"]["overall_average_rand_index"]),
                "full256_num_pred_image_ids": int(full_eval["noagg_official"]["num_pred_image_ids"]),
                "common253_miou": float(common_eval["noagg_official"]["overall_average_iou"]),
                "common253_ari": float(common_eval["noagg_official"]["overall_average_rand_index"]),
                "common253_num_pred_image_ids": int(common_eval["noagg_official"]["num_pred_image_ids"]),
                "internal_full256_miou": _mean(rows, "rwtd_iou"),
                "internal_full256_ari": _mean(rows, "rwtd_ari"),
                "internal_common253_miou": _mean([r for r in rows if bool(r["in_common253"])], "rwtd_iou"),
                "internal_common253_ari": _mean([r for r in rows if bool(r["in_common253"])], "rwtd_ari"),
            }
        ],
        "artifacts": {
            "official_eval_full256_json": str(full_json.resolve()),
            "official_eval_common253_json": str(common_json.resolve()),
            "full_export_dir": str(full_dir.resolve()),
            "common253_export_dir": str(common_dir.resolve()),
            "per_image_csv": str(per_image_csv.resolve()),
        },
    }
    out_json = args.out_root / "rwtd_learned_single_selector_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_stld(args: argparse.Namespace) -> None:
    images_dir = args.benchmark_root / "images"
    labels_dir = args.benchmark_root / "labels"
    image_ids = sorted(int(p.stem) for p in images_dir.glob("*.png"))
    common_ids = _load_common182_ids(args.texturesam_maskbank_csv)
    store = PromptMaskProposalStore(args.prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))
    consolidator = TextureSAMV2Consolidator(_cfg(args))

    masks_dir = args.out_root / "masks"
    _clean_export_dir(masks_dir)

    rows: list[dict[str, object]] = []
    for i, image_id in enumerate(image_ids, start=1):
        image_rgb = read_image_rgb(images_dir / f"{image_id}.png")
        gt = ensure_binary_gt(read_mask_raw(labels_dir / f"{image_id}.png"), strict=False)
        proposals = store.load(image_id, expected_shape=gt.shape)
        pred, selected_index, selected_score, proposal_count = _select_single(
            consolidator=consolidator,
            image_rgb=image_rgb,
            proposals=proposals,
        )
        write_binary_mask(masks_dir / f"{image_id}.png", pred)
        rows.append(
            {
                "image_id": int(image_id),
                "in_common182": bool(image_id in common_ids),
                "proposal_count": int(proposal_count),
                "selected_index": int(selected_index),
                "selected_score": float(selected_score),
                "direct_iou": float(iou_score(pred, gt)),
                "direct_ari": float(ari_score_binary(pred, gt)),
            }
        )
        if i % 32 == 0:
            print(f"[learned-single-selector:stld] processed {i}/{len(image_ids)} images", flush=True)

    per_image_csv = args.out_root / "stld_learned_single_selector_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_rows = rows
    common_rows = [row for row in rows if bool(row["in_common182"])]
    summary = {
        "dataset": "stld",
        "selector": "learned_stageA_singleton",
        "num_images": int(len(all_rows)),
        "common182_count": int(len(common_rows)),
        "ptd_encoder_ckpt": str(args.ptd_encoder_ckpt.resolve()),
        "ptd_learned_bundle": str(args.ptd_learned_bundle.resolve()),
        "descriptor_mode": str(args.descriptor_mode),
        "methods": [
            {
                "method": "learned_single_selector",
                "all200_miou": _mean(all_rows, "direct_iou"),
                "all200_ari": _mean(all_rows, "direct_ari"),
                "common182_miou": _mean(common_rows, "direct_iou"),
                "common182_ari": _mean(common_rows, "direct_ari"),
            }
        ],
        "artifacts": {
            "masks_dir": str(masks_dir.resolve()),
            "per_image_csv": str(per_image_csv.resolve()),
        },
    }
    out_json = args.out_root / "stld_learned_single_selector_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "rwtd":
        run_rwtd(args)
        return
    if args.mode == "stld":
        run_stld(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
