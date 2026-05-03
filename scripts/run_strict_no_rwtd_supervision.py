from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig
from texturesam_v2.dtd_encoder import DTDTrainConfig, train_dtd_encoder
from texturesam_v2.merge import MergeConfig
from texturesam_v2.pipeline import EvalConfig, TextureSAMV2Pipeline


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
            "Strict TextureSAM-v2 protocol: no RWTD-label training/tuning. "
            "Runs fixed handcrafted and fixed DTD-CNN consolidators, then reports metrics."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--baseline-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=None)

    p.add_argument(
        "--dtd-checkpoint",
        type=Path,
        default=Path("artifacts/dtd_small_cnn.pt"),
    )
    p.add_argument(
        "--dtd-data-root",
        type=Path,
        default=Path("external/data/dtd"),
    )
    p.add_argument("--train-dtd-if-missing", action="store_true")
    p.add_argument("--dtd-epochs", type=int, default=5)
    p.add_argument("--dtd-batch-size", type=int, default=64)
    return p.parse_args()


def fixed_cfg(descriptor_mode: str, dtd_checkpoint: Path | None = None) -> ConsolidationConfig:
    return ConsolidationConfig(
        min_area=32,
        close_kernel=5,
        keep_largest_component=True,
        hole_area_threshold=64,
        descriptor_mode=descriptor_mode,
        dtd_checkpoint=dtd_checkpoint,
        dtd_device="cuda",
        merge=FIXED_MERGE,
        objective_lambda=0.45,
        objective_mu=0.30,
    )


def maybe_train_dtd(args: argparse.Namespace) -> None:
    if args.dtd_checkpoint.exists():
        return
    if not args.train_dtd_if_missing:
        raise FileNotFoundError(
            f"Missing DTD checkpoint: {args.dtd_checkpoint}. "
            "Pass --train-dtd-if-missing or provide --dtd-checkpoint."
        )
    metrics = train_dtd_encoder(
        DTDTrainConfig(
            data_root=args.dtd_data_root,
            out_ckpt=args.dtd_checkpoint,
            epochs=args.dtd_epochs,
            batch_size=args.dtd_batch_size,
        )
    )
    print("Trained DTD encoder:", json.dumps(metrics, indent=2, sort_keys=True))


def run_one(
    *,
    descriptor_mode: str,
    out_dir: Path,
    rwtd_root: Path,
    prompt_masks_root: Path,
    baseline_masks_root: Path,
    max_images: int | None,
    dtd_checkpoint: Path | None = None,
) -> dict[str, float | int | str]:
    pipe = TextureSAMV2Pipeline(
        fixed_cfg(descriptor_mode=descriptor_mode, dtd_checkpoint=dtd_checkpoint)
    )
    return pipe.run(
        EvalConfig(
            rwtd_root=rwtd_root,
            prompt_masks_root=prompt_masks_root,
            baseline_masks_root=baseline_masks_root,
            out_dir=out_dir,
            max_images=max_images,
        )
    )


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["image_id"]] = row
    return out


def write_merged_per_image(
    *,
    handcrafted_csv: Path,
    dtd_csv: Path,
    out_csv: Path,
) -> None:
    h_rows = load_rows(handcrafted_csv)
    d_rows = load_rows(dtd_csv)
    image_ids = sorted(set(h_rows.keys()) & set(d_rows.keys()), key=lambda s: int(s.split("_")[-1]))

    fields = [
        "image_id",
        "baseline_iou",
        "baseline_ari",
        "handcrafted_iou",
        "handcrafted_ari",
        "dtd_iou",
        "dtd_ari",
        "delta_handcrafted_iou",
        "delta_handcrafted_ari",
        "delta_dtd_iou",
        "delta_dtd_ari",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for image_id in image_ids:
            hr = h_rows[image_id]
            dr = d_rows[image_id]
            b_iou = float(hr["baseline_iou"])
            b_ari = float(hr["baseline_ari"])
            h_iou = float(hr["v2_iou"])
            h_ari = float(hr["v2_ari"])
            d_iou = float(dr["v2_iou"])
            d_ari = float(dr["v2_ari"])
            w.writerow(
                {
                    "image_id": image_id,
                    "baseline_iou": b_iou,
                    "baseline_ari": b_ari,
                    "handcrafted_iou": h_iou,
                    "handcrafted_ari": h_ari,
                    "dtd_iou": d_iou,
                    "dtd_ari": d_ari,
                    "delta_handcrafted_iou": h_iou - b_iou,
                    "delta_handcrafted_ari": h_ari - b_ari,
                    "delta_dtd_iou": d_iou - b_iou,
                    "delta_dtd_ari": d_ari - b_ari,
                }
            )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    maybe_train_dtd(args)

    out_handcrafted = args.out_root / "handcrafted"
    out_dtd = args.out_root / "dtd_cnn"

    print("Running strict handcrafted...")
    h_summary = run_one(
        descriptor_mode="handcrafted",
        out_dir=out_handcrafted,
        rwtd_root=args.rwtd_root,
        prompt_masks_root=args.prompt_masks_root,
        baseline_masks_root=args.baseline_masks_root,
        max_images=args.max_images,
    )

    print("Running strict dtd_cnn...")
    d_summary = run_one(
        descriptor_mode="dtd_cnn",
        dtd_checkpoint=args.dtd_checkpoint,
        out_dir=out_dtd,
        rwtd_root=args.rwtd_root,
        prompt_masks_root=args.prompt_masks_root,
        baseline_masks_root=args.baseline_masks_root,
        max_images=args.max_images,
    )

    merged_csv = args.out_root / "per_image_strict.csv"
    write_merged_per_image(
        handcrafted_csv=out_handcrafted / "per_image.csv",
        dtd_csv=out_dtd / "per_image.csv",
        out_csv=merged_csv,
    )

    baseline_miou = float(h_summary["baseline_miou"])
    baseline_ari = float(h_summary["baseline_ari"])
    out = {
        "protocol": {
            "name": "strict_no_rwtd_supervision_v1",
            "uses_rwtd_labels_for_training": False,
            "uses_rwtd_labels_for_hyperparameter_search": False,
            "uses_rwtd_labels_for_final_metric_reporting": True,
            "selection_model": "none",
            "frozen_config": {
                "min_area": 32,
                "close_kernel": 5,
                "hole_area_threshold": 64,
                "merge_threshold": 0.50,
                "merge_dilation": 3,
                "w_texture": 0.65,
                "w_boundary": 0.30,
                "w_hetero": 0.20,
                "objective_lambda": 0.45,
                "objective_mu": 0.30,
            },
            "external_training_data": {
                "handcrafted": "none",
                "dtd_cnn": "DTD (train/val only)",
            },
        },
        "dataset": "rwtd_kaust256",
        "num_images": int(h_summary["num_images"]),
        "proposal_source": str(args.prompt_masks_root),
        "baseline": {
            "miou": baseline_miou,
            "ari": baseline_ari,
        },
        "strict_handcrafted": {
            "miou": float(h_summary["v2_miou"]),
            "ari": float(h_summary["v2_ari"]),
            "delta_miou_vs_baseline": float(h_summary["delta_miou_vs_baseline"]),
            "delta_ari_vs_baseline": float(h_summary["delta_ari_vs_baseline"]),
        },
        "strict_dtd_cnn": {
            "miou": float(d_summary["v2_miou"]),
            "ari": float(d_summary["v2_ari"]),
            "delta_miou_vs_baseline": float(d_summary["delta_miou_vs_baseline"]),
            "delta_ari_vs_baseline": float(d_summary["delta_ari_vs_baseline"]),
        },
        "artifacts": {
            "handcrafted_summary": str(out_handcrafted / "summary.json"),
            "dtd_summary": str(out_dtd / "summary.json"),
            "per_image_strict": str(merged_csv),
            "handcrafted_masks_dir": str(out_handcrafted / "masks"),
            "dtd_masks_dir": str(out_dtd / "masks"),
        },
    }

    out_json = args.out_root / "strict_summary.json"
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
