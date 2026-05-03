#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig, TextureSAMV2Consolidator
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
from texturesam_v2.ptd_learned import PTDLearnedTrainConfig, train_ptd_learned_models


FIXED_MERGE = MergeConfig(
    adjacency_dilation=3,
    merge_threshold=0.50,
    w_texture=0.65,
    w_boundary=0.30,
    w_hetero=0.20,
)


@dataclass(frozen=True)
class DatasetTrainSpec:
    name: str
    ptd_encoder_ckpt: Path
    hybrid_bundle: Path
    ptd_root: Path
    num_samples: int
    synthetic_layout: str
    shape_mask_root: Path | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the final-round descriptor ablation for ArchiTexture's proposal-space route. "
            "The current hybrid rows are reused from the retained bundle unless --retrain-hybrid is set."
        )
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "reports" / "final_submission_descriptor_ablation",
    )
    p.add_argument(
        "--descriptor-modes",
        nargs="+",
        default=["handcrafted", "ptd_convnext", "hybrid_ptd"],
        choices=["handcrafted", "ptd_convnext", "hybrid_ptd"],
    )
    p.add_argument("--retrain-hybrid", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--skip-rwtd-final", action="store_true")
    p.add_argument("--ptd-root", type=Path, default=Path("external/data/ptd"))

    p.add_argument("--rwtd-root", type=Path, default=ROOT / "data" / "rwtd_kaust256_clean")
    p.add_argument(
        "--rwtd-stagea-prompt-root",
        type=Path,
        default=ROOT / "reports" / "repro_upstream_eval" / "official_0p3_promptstyle",
    )
    p.add_argument(
        "--rwtd-dense-prompt-root",
        type=Path,
        default=ROOT / "reports" / "strict_ptd_v11_multibank" / "overnight_multibank_promptstyle_fixed",
    )
    p.add_argument(
        "--rwtd-common-baseline-root",
        type=Path,
        default=ROOT / "reports" / "repro_upstream_eval" / "official_0p3_masks",
    )
    p.add_argument(
        "--rwtd-ptd-encoder-ckpt",
        type=Path,
        default=ROOT / "artifacts" / "ptd_encoder_swinb_pre_ring.pt",
    )
    p.add_argument(
        "--rwtd-hybrid-bundle",
        type=Path,
        default=ROOT / "artifacts" / "ptd_learned_swinb_pre_ring.pkl",
    )
    p.add_argument(
        "--rwtd-acute-bundle",
        type=Path,
        default=ROOT / "artifacts" / "ptd_acute_rescue_repairmix_s256_safe_logreg_candidate.pkl",
    )
    p.add_argument(
        "--rwtd-acute-bundle-metrics",
        type=Path,
        default=ROOT / "artifacts" / "ptd_acute_rescue_repairmix_s256_safe_logreg_candidate_metrics.json",
    )
    p.add_argument(
        "--rwtd-v8-bundle",
        type=Path,
        default=ROOT / "artifacts" / "ptd_v8_partition_bundle_sanitized_cuda.pkl",
    )
    p.add_argument(
        "--rwtd-upstream-root",
        type=Path,
        default=Path("external/TextureSAM_upstream_20260303"),
    )

    p.add_argument(
        "--stld-benchmark-root",
        type=Path,
        default=ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "benchmark",
    )
    p.add_argument(
        "--stld-prompt-root",
        type=Path,
        default=ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "promptstyle",
    )
    p.add_argument(
        "--stld-common-csv",
        type=Path,
        default=ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "stld_texturesam0p3_maskbank_per_image.csv",
    )
    p.add_argument(
        "--stld-ptd-encoder-ckpt",
        type=Path,
        default=ROOT / "experiments" / "khan_stld_parallel_20260312" / "results_full" / "models" / "ptd_encoder.pt",
    )
    p.add_argument(
        "--stld-hybrid-bundle",
        type=Path,
        default=ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "models" / "ptd_learned_split.pkl",
    )
    p.add_argument(
        "--stld-shape-mask-root",
        type=Path,
        default=ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "assets" / "shape_split" / "train",
    )

    p.add_argument("--ptd-device", type=str, default="cuda")
    p.add_argument("--min-area", type=int, default=32)
    p.add_argument("--ptd-ring-dilation", type=int, default=9)
    p.add_argument("--ptd-ring-min-pixels", type=int, default=24)
    p.add_argument("--ptd-use-ring-context", dest="ptd_use_ring_context", action="store_true")
    p.add_argument("--no-ptd-use-ring-context", dest="ptd_use_ring_context", action="store_false")
    p.set_defaults(ptd_use_ring_context=True)
    return p.parse_args()


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _descriptor_cfg(
    *,
    descriptor_mode: str,
    learned_bundle: Path,
    ptd_encoder_ckpt: Path,
    ptd_device: str,
    min_area: int,
    ptd_use_ring_context: bool,
    ptd_ring_dilation: int,
    ptd_ring_min_pixels: int,
) -> ConsolidationConfig:
    return ConsolidationConfig(
        min_area=int(min_area),
        close_kernel=5,
        keep_largest_component=True,
        hole_area_threshold=64,
        descriptor_mode=str(descriptor_mode),
        ptd_checkpoint=ptd_encoder_ckpt,
        ptd_device=ptd_device,
        ptd_use_ring_context=bool(ptd_use_ring_context),
        ptd_ring_dilation=int(ptd_ring_dilation),
        ptd_ring_min_pixels=int(ptd_ring_min_pixels),
        learned_bundle=learned_bundle,
        merge=FIXED_MERGE,
        objective_lambda=0.45,
        objective_mu=0.30,
    )


def _run_json_eval(pred_dir: Path, gt_dir: Path, upstream_root: Path, out_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_upstream_texture_metrics.py"),
        "--pred-folder",
        str(pred_dir),
        "--gt-folder",
        str(gt_dir),
        "--upstream-root",
        str(upstream_root),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


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


def _mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def _export_rwtd_official(
    *,
    src_masks_dir: Path,
    full_dir: Path,
    common_dir: Path,
    common_ids: set[int],
) -> None:
    _clean_dir(full_dir)
    _clean_dir(common_dir)
    for src in sorted(src_masks_dir.glob("*.png")):
        image_id = int(src.stem)
        dst_name = f"mask_0_{image_id}.png"
        shutil.copy2(src, full_dir / dst_name)
        if image_id in common_ids:
            shutil.copy2(src, common_dir / dst_name)


def _ensure_bundle(
    *,
    args: argparse.Namespace,
    dataset_spec: DatasetTrainSpec,
    descriptor_mode: str,
    mode_root: Path,
) -> tuple[Path, Path, str]:
    if descriptor_mode == "hybrid_ptd" and not args.retrain_hybrid:
        return dataset_spec.hybrid_bundle, dataset_spec.hybrid_bundle.with_name(dataset_spec.hybrid_bundle.stem + "_metrics.json"), "reused"

    bundle_path = mode_root / "models" / f"{descriptor_mode}.pkl"
    metrics_path = mode_root / "models" / f"{descriptor_mode}_metrics.json"
    if bundle_path.exists() and metrics_path.exists() and not args.force_retrain:
        return bundle_path, metrics_path, "cached"

    mode_root.joinpath("models").mkdir(parents=True, exist_ok=True)
    metrics = train_ptd_learned_models(
        PTDLearnedTrainConfig(
            ptd_root=args.ptd_root,
            ptd_encoder_ckpt=dataset_spec.ptd_encoder_ckpt,
            out_bundle=bundle_path,
            out_metrics_json=metrics_path,
            descriptor_mode=descriptor_mode,
            num_samples=dataset_spec.num_samples,
            random_seed=1337,
            synthetic_layout=dataset_spec.synthetic_layout,
            shape_mask_root=dataset_spec.shape_mask_root,
        )
    )
    print(
        json.dumps(
            {
                "dataset": dataset_spec.name,
                "descriptor_mode": descriptor_mode,
                "train_status": "trained",
                "bundle_path": str(bundle_path),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return bundle_path, metrics_path, "trained"


def _evaluate_stld(
    *,
    args: argparse.Namespace,
    descriptor_mode: str,
    bundle_path: Path,
    out_root: Path,
) -> dict[str, object]:
    images_dir = args.stld_benchmark_root / "images"
    labels_dir = args.stld_benchmark_root / "labels"
    image_ids = sorted(int(p.stem) for p in images_dir.glob("*.png"))
    common_ids = _load_common182_ids(args.stld_common_csv)
    masks_dir = out_root / "masks"
    _clean_dir(masks_dir)

    store = PromptMaskProposalStore(args.stld_prompt_root, ProposalLoadConfig(min_area=int(args.min_area)))
    consolidator = TextureSAMV2Consolidator(
        _descriptor_cfg(
            descriptor_mode=descriptor_mode,
            learned_bundle=bundle_path,
            ptd_encoder_ckpt=args.stld_ptd_encoder_ckpt,
            ptd_device=args.ptd_device,
            min_area=args.min_area,
            ptd_use_ring_context=args.ptd_use_ring_context,
            ptd_ring_dilation=args.ptd_ring_dilation,
            ptd_ring_min_pixels=args.ptd_ring_min_pixels,
        )
    )

    rows: list[dict[str, object]] = []
    for i, image_id in enumerate(image_ids, start=1):
        image_rgb = read_image_rgb(images_dir / f"{image_id}.png")
        gt = ensure_binary_gt(read_mask_raw(labels_dir / f"{image_id}.png"), strict=False)
        proposals = store.load(image_id, expected_shape=gt.shape)
        pred, debug = consolidator(image_rgb, proposals)
        write_binary_mask(masks_dir / f"{image_id}.png", pred)
        rows.append(
            {
                "image_id": int(image_id),
                "in_common182": bool(image_id in common_ids),
                "proposal_count": int(len(proposals)),
                "merged_components": int(debug.num_merged_components),
                "direct_iou": float(iou_score(pred, gt)),
                "direct_ari": float(ari_score_binary(pred, gt)),
            }
        )
        if i % 32 == 0:
            print(f"[descriptor-ablation:stld:{descriptor_mode}] processed {i}/{len(image_ids)} images", flush=True)

    per_image_csv = out_root / "stld_stageA_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    common_rows = [row for row in rows if bool(row["in_common182"])]
    summary = {
        "dataset": "stld",
        "descriptor_mode": descriptor_mode,
        "bundle_path": str(bundle_path.resolve()),
        "num_images": int(len(rows)),
        "common182_count": int(len(common_rows)),
        "mean_proposal_count": _mean(rows, "proposal_count"),
        "mean_merged_components": _mean(rows, "merged_components"),
        "stageA": {
            "all200_miou": _mean(rows, "direct_iou"),
            "all200_ari": _mean(rows, "direct_ari"),
            "common182_miou": _mean(common_rows, "direct_iou"),
            "common182_ari": _mean(common_rows, "direct_ari"),
        },
        "artifacts": {
            "masks_dir": str(masks_dir.resolve()),
            "per_image_csv": str(per_image_csv.resolve()),
        },
    }
    out_json = out_root / "stld_descriptor_ablation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _evaluate_rwtd(
    *,
    args: argparse.Namespace,
    descriptor_mode: str,
    bundle_path: Path,
    out_root: Path,
) -> dict[str, object]:
    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    gt_dir = args.rwtd_upstream_root / "Kaust256" / "labeles"
    images = list_rwtd_images(image_dir)
    common_ids = _parse_rwtd_common_ids(args.rwtd_common_baseline_root)

    stagea_root = out_root / "stageA"
    stagea_masks_dir = stagea_root / "masks"
    stagea_export_full = stagea_root / "official_full256"
    stagea_export_common = stagea_root / "official_common253"
    _clean_dir(stagea_masks_dir)

    store = PromptMaskProposalStore(args.rwtd_stagea_prompt_root, ProposalLoadConfig(min_area=int(args.min_area)))
    consolidator = TextureSAMV2Consolidator(
        _descriptor_cfg(
            descriptor_mode=descriptor_mode,
            learned_bundle=bundle_path,
            ptd_encoder_ckpt=args.rwtd_ptd_encoder_ckpt,
            ptd_device=args.ptd_device,
            min_area=args.min_area,
            ptd_use_ring_context=args.ptd_use_ring_context,
            ptd_ring_dilation=args.ptd_ring_dilation,
            ptd_ring_min_pixels=args.ptd_ring_min_pixels,
        )
    )

    stagea_rows: list[dict[str, object]] = []
    for i, image_path in enumerate(images, start=1):
        image_num = int(image_path.stem)
        image_rgb = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_num}.png"), strict=False)
        proposals = store.load(image_num, expected_shape=gt.shape)
        pred, debug = consolidator(image_rgb, proposals)
        write_binary_mask(stagea_masks_dir / f"{image_num}.png", pred)
        met = rwtd_invariant_metrics(pred, gt)
        stagea_rows.append(
            {
                "image_id": int(image_num),
                "in_common253": bool(image_num in common_ids),
                "proposal_count": int(len(proposals)),
                "merged_components": int(debug.num_merged_components),
                "rwtd_iou": float(met.iou),
                "rwtd_ari": float(met.ari),
            }
        )
        if i % 32 == 0:
            print(f"[descriptor-ablation:rwtd:{descriptor_mode}] processed {i}/{len(images)} stage-A images", flush=True)

    stagea_per_image_csv = stagea_root / "stageA_per_image.csv"
    with stagea_per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stagea_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stagea_rows)

    _export_rwtd_official(
        src_masks_dir=stagea_masks_dir,
        full_dir=stagea_export_full,
        common_dir=stagea_export_common,
        common_ids=common_ids,
    )
    stagea_official_full = _run_json_eval(
        stagea_export_full,
        gt_dir,
        args.rwtd_upstream_root,
        stagea_root / "official_eval_full256.json",
    )
    stagea_official_common = _run_json_eval(
        stagea_export_common,
        gt_dir,
        args.rwtd_upstream_root,
        stagea_root / "official_eval_common253.json",
    )

    if args.skip_rwtd_final:
        summary = {
            "dataset": "rwtd",
            "descriptor_mode": descriptor_mode,
            "bundle_path": str(bundle_path.resolve()),
            "num_images": int(len(stagea_rows)),
            "common253_count": int(sum(1 for row in stagea_rows if bool(row["in_common253"]))),
            "stageA": {
                "mean_proposal_count": _mean(stagea_rows, "proposal_count"),
                "mean_merged_components": _mean(stagea_rows, "merged_components"),
                "internal_full256_miou": _mean(stagea_rows, "rwtd_iou"),
                "internal_full256_ari": _mean(stagea_rows, "rwtd_ari"),
                "internal_common253_miou": _mean([row for row in stagea_rows if bool(row["in_common253"])], "rwtd_iou"),
                "internal_common253_ari": _mean([row for row in stagea_rows if bool(row["in_common253"])], "rwtd_ari"),
                "official_full256_miou": float(stagea_official_full["noagg_official"]["overall_average_iou"]),
                "official_full256_ari": float(stagea_official_full["noagg_official"]["overall_average_rand_index"]),
                "official_common253_miou": float(stagea_official_common["noagg_official"]["overall_average_iou"]),
                "official_common253_ari": float(stagea_official_common["noagg_official"]["overall_average_rand_index"]),
            },
            "artifacts": {
                "stageA_per_image_csv": str(stagea_per_image_csv.resolve()),
                "stageA_masks_dir": str(stagea_masks_dir.resolve()),
                "stageA_official_full_json": str((stagea_root / "official_eval_full256.json").resolve()),
                "stageA_official_common_json": str((stagea_root / "official_eval_common253.json").resolve()),
            },
        }
        out_json = out_root / "rwtd_descriptor_ablation_summary.json"
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    final_root = out_root / "final_rwtd"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    rescue_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_texturesam2_acute_learned_rescue.py"),
        "--rwtd-root",
        str(args.rwtd_root),
        "--dense-prompt-masks-root",
        str(args.rwtd_dense_prompt_root),
        "--v9-masks-root",
        str(stagea_masks_dir),
        "--out-root",
        str(final_root),
        "--ptd-checkpoint",
        str(args.rwtd_ptd_encoder_ckpt),
        "--ptd-v8-bundle",
        str(args.rwtd_v8_bundle),
        "--bundle-path",
        str(args.rwtd_acute_bundle),
        "--bundle-metrics-json",
        str(args.rwtd_acute_bundle_metrics),
    ]
    subprocess.run(rescue_cmd, check=True)

    rescue_run_root = final_root / "texturesam2_acute_learned_rescue"
    final_masks_dir = rescue_run_root / "masks"
    final_export_full = final_root / "official_full256"
    final_export_common = final_root / "official_common253"
    _export_rwtd_official(
        src_masks_dir=final_masks_dir,
        full_dir=final_export_full,
        common_dir=final_export_common,
        common_ids=common_ids,
    )
    final_official_full = _run_json_eval(
        final_export_full,
        gt_dir,
        args.rwtd_upstream_root,
        final_root / "official_eval_full256.json",
    )
    final_official_common = _run_json_eval(
        final_export_common,
        gt_dir,
        args.rwtd_upstream_root,
        final_root / "official_eval_common253.json",
    )

    rescue_summary = json.loads((rescue_run_root / "summary.json").read_text(encoding="utf-8"))
    rescue_rows: list[dict[str, object]] = []
    with (rescue_run_root / "per_image.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rescue_rows.append(row)

    summary = {
        "dataset": "rwtd",
        "descriptor_mode": descriptor_mode,
        "bundle_path": str(bundle_path.resolve()),
        "num_images": int(len(stagea_rows)),
        "common253_count": int(sum(1 for row in stagea_rows if bool(row["in_common253"]))),
        "stageA": {
            "mean_proposal_count": _mean(stagea_rows, "proposal_count"),
            "mean_merged_components": _mean(stagea_rows, "merged_components"),
            "internal_full256_miou": _mean(stagea_rows, "rwtd_iou"),
            "internal_full256_ari": _mean(stagea_rows, "rwtd_ari"),
            "internal_common253_miou": _mean([row for row in stagea_rows if bool(row["in_common253"])], "rwtd_iou"),
            "internal_common253_ari": _mean([row for row in stagea_rows if bool(row["in_common253"])], "rwtd_ari"),
            "official_full256_miou": float(stagea_official_full["noagg_official"]["overall_average_iou"]),
            "official_full256_ari": float(stagea_official_full["noagg_official"]["overall_average_rand_index"]),
            "official_common253_miou": float(stagea_official_common["noagg_official"]["overall_average_iou"]),
            "official_common253_ari": float(stagea_official_common["noagg_official"]["overall_average_rand_index"]),
        },
        "final": {
            "mean_dense_proposal_count": float(
                np.mean([float(row["proposal_count_dense"]) for row in rescue_rows]) if rescue_rows else 0.0
            ),
            "mean_dense_candidate_count": float(
                np.mean([float(row["dense_candidate_count"]) for row in rescue_rows]) if rescue_rows else 0.0
            ),
            "switch_count": int(rescue_summary["switching"]["switched_count"]),
            "official_full256_miou": float(final_official_full["noagg_official"]["overall_average_iou"]),
            "official_full256_ari": float(final_official_full["noagg_official"]["overall_average_rand_index"]),
            "official_common253_miou": float(final_official_common["noagg_official"]["overall_average_iou"]),
            "official_common253_ari": float(final_official_common["noagg_official"]["overall_average_rand_index"]),
        },
        "artifacts": {
            "stageA_per_image_csv": str(stagea_per_image_csv.resolve()),
            "stageA_masks_dir": str(stagea_masks_dir.resolve()),
            "stageA_official_full_json": str((stagea_root / "official_eval_full256.json").resolve()),
            "stageA_official_common_json": str((stagea_root / "official_eval_common253.json").resolve()),
            "final_summary_json": str((rescue_run_root / "summary.json").resolve()),
            "final_per_image_csv": str((rescue_run_root / "per_image.csv").resolve()),
            "final_official_full_json": str((final_root / "official_eval_full256.json").resolve()),
            "final_official_common_json": str((final_root / "official_eval_common253.json").resolve()),
        },
    }
    out_json = out_root / "rwtd_descriptor_ablation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    rwtd_spec = DatasetTrainSpec(
        name="rwtd",
        ptd_encoder_ckpt=args.rwtd_ptd_encoder_ckpt,
        hybrid_bundle=args.rwtd_hybrid_bundle,
        ptd_root=args.ptd_root,
        num_samples=480,
        synthetic_layout="voronoi",
    )
    stld_spec = DatasetTrainSpec(
        name="stld",
        ptd_encoder_ckpt=args.stld_ptd_encoder_ckpt,
        hybrid_bundle=args.stld_hybrid_bundle,
        ptd_root=args.ptd_root,
        num_samples=1600,
        synthetic_layout="mpeg7_shapes",
        shape_mask_root=args.stld_shape_mask_root,
    )

    all_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {"descriptor_modes": list(args.descriptor_modes), "runs": {}}

    for descriptor_mode in args.descriptor_modes:
        mode_root = args.out_root / descriptor_mode
        bundle_path_rwtd, bundle_metrics_rwtd, bundle_status_rwtd = _ensure_bundle(
            args=args,
            dataset_spec=rwtd_spec,
            descriptor_mode=descriptor_mode,
            mode_root=mode_root / "rwtd",
        )
        rwtd_summary = _evaluate_rwtd(
            args=args,
            descriptor_mode=descriptor_mode,
            bundle_path=bundle_path_rwtd,
            out_root=mode_root / "rwtd",
        )
        rwtd_summary["bundle_metrics_path"] = str(bundle_metrics_rwtd.resolve()) if bundle_metrics_rwtd.exists() else ""
        rwtd_summary["bundle_status"] = bundle_status_rwtd
        payload["runs"][f"rwtd::{descriptor_mode}"] = rwtd_summary
        all_rows.extend(
            [
                {
                    "dataset": "RWTD",
                    "subset": "common-253",
                    "route": "stageA" if args.skip_rwtd_final else "full",
                    "descriptor_mode": descriptor_mode,
                    "miou": (
                        rwtd_summary["stageA"]["official_common253_miou"]
                        if args.skip_rwtd_final
                        else rwtd_summary["final"]["official_common253_miou"]
                    ),
                    "ari": (
                        rwtd_summary["stageA"]["official_common253_ari"]
                        if args.skip_rwtd_final
                        else rwtd_summary["final"]["official_common253_ari"]
                    ),
                    "bundle_status": bundle_status_rwtd,
                },
                {
                    "dataset": "RWTD",
                    "subset": "full-256",
                    "route": "stageA" if args.skip_rwtd_final else "full",
                    "descriptor_mode": descriptor_mode,
                    "miou": (
                        rwtd_summary["stageA"]["official_full256_miou"]
                        if args.skip_rwtd_final
                        else rwtd_summary["final"]["official_full256_miou"]
                    ),
                    "ari": (
                        rwtd_summary["stageA"]["official_full256_ari"]
                        if args.skip_rwtd_final
                        else rwtd_summary["final"]["official_full256_ari"]
                    ),
                    "bundle_status": bundle_status_rwtd,
                },
            ]
        )

        bundle_path_stld, bundle_metrics_stld, bundle_status_stld = _ensure_bundle(
            args=args,
            dataset_spec=stld_spec,
            descriptor_mode=descriptor_mode,
            mode_root=mode_root / "stld",
        )
        stld_summary = _evaluate_stld(
            args=args,
            descriptor_mode=descriptor_mode,
            bundle_path=bundle_path_stld,
            out_root=mode_root / "stld",
        )
        stld_summary["bundle_metrics_path"] = str(bundle_metrics_stld.resolve()) if bundle_metrics_stld.exists() else ""
        stld_summary["bundle_status"] = bundle_status_stld
        payload["runs"][f"stld::{descriptor_mode}"] = stld_summary
        all_rows.extend(
            [
                {
                    "dataset": "STLD",
                    "subset": "common-182",
                    "route": "stageA",
                    "descriptor_mode": descriptor_mode,
                    "miou": stld_summary["stageA"]["common182_miou"],
                    "ari": stld_summary["stageA"]["common182_ari"],
                    "bundle_status": bundle_status_stld,
                },
                {
                    "dataset": "STLD",
                    "subset": "all-200",
                    "route": "stageA",
                    "descriptor_mode": descriptor_mode,
                    "miou": stld_summary["stageA"]["all200_miou"],
                    "ari": stld_summary["stageA"]["all200_ari"],
                    "bundle_status": bundle_status_stld,
                },
            ]
        )

    summary_json = args.out_root / "descriptor_ablation_summary.json"
    summary_csv = args.out_root / "descriptor_ablation_summary.csv"
    payload["summary_csv"] = str(summary_csv.resolve())
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "subset", "route", "descriptor_mode", "miou", "ari", "bundle_status"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
