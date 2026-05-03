from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.consolidator import ConsolidationConfig
from texturesam_v2.merge import MergeConfig
from texturesam_v2.pipeline import EvalConfig, TextureSAMV2Pipeline
from texturesam_v2.ptd_encoder import PTDTrainConfig, train_ptd_encoder
from texturesam_v2.ptd_v3 import PTDV3TrainConfig, train_ptd_v3_models
from texturesam_v2.ptd_v8_partition import PTDV8PartitionTrainConfig, train_ptd_v8_partition_models


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
            "TextureSAM-v2 strict PTD-v8 partition protocol: PTD-only external training, "
            "no RWTD-label training/tuning, one-shot RWTD evaluation."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--baseline-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=None)

    p.add_argument("--ptd-root", type=Path, default=Path("external/data/ptd"))
    p.add_argument("--ptd-encoder-ckpt", type=Path, default=Path("artifacts/ptd_convnext_tiny.pt"))
    p.add_argument("--ptd-v3-bundle", type=Path, default=Path("artifacts/ptd_v3_graph_bundle.pkl"))
    p.add_argument(
        "--ptd-v3-metrics-json",
        type=Path,
        default=Path("artifacts/ptd_v3_graph_metrics.json"),
    )
    p.add_argument("--ptd-v8-bundle", type=Path, default=Path("artifacts/ptd_v8_partition_bundle.pkl"))
    p.add_argument(
        "--ptd-v8-metrics-json",
        type=Path,
        default=Path("artifacts/ptd_v8_partition_metrics.json"),
    )
    p.add_argument("--train-if-missing", action="store_true")

    p.add_argument("--ptd-epochs", type=int, default=2)
    p.add_argument("--ptd-batch-size", type=int, default=48)
    p.add_argument("--ptd-image-size", type=int, default=192)
    p.add_argument("--ptd-max-train-images", type=int, default=120000)
    p.add_argument("--synthetic-samples-v3", type=int, default=2200)
    p.add_argument("--synthetic-samples-v8", type=int, default=520)
    p.add_argument("--synthetic-image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("outputs/strict_ptd_v7_dualgate_on_official0p3/strict_ptd_v7_dualgate/summary.json"),
    )
    p.add_argument("--reference-name", type=str, default="strict_ptd_v7_dualgate")
    return p.parse_args()


def _cfg_ptd_v8(ptd_ckpt: Path, v8_bundle: Path) -> ConsolidationConfig:
    return ConsolidationConfig(
        min_area=32,
        close_kernel=5,
        keep_largest_component=False,
        hole_area_threshold=64,
        descriptor_mode="hybrid_ptd",
        ptd_checkpoint=ptd_ckpt,
        ptd_device="cuda",
        learned_bundle=v8_bundle,
        merge=FIXED_MERGE,
        objective_lambda=0.40,
        objective_mu=0.15,
    )


def maybe_train_ptd_encoder(args: argparse.Namespace) -> dict[str, float | int] | None:
    if args.ptd_encoder_ckpt.exists():
        return None
    if not args.train_if_missing:
        raise FileNotFoundError(
            f"Missing PTD encoder checkpoint: {args.ptd_encoder_ckpt}. "
            "Pass --train-if-missing to train it."
        )
    print("[strict_ptd_v8_partition] Training PTD encoder...")
    m = train_ptd_encoder(
        PTDTrainConfig(
            data_root=args.ptd_root,
            out_ckpt=args.ptd_encoder_ckpt,
            epochs=args.ptd_epochs,
            batch_size=args.ptd_batch_size,
            image_size=args.ptd_image_size,
            max_train_images=args.ptd_max_train_images,
            seed=args.seed,
        )
    )
    print("[strict_ptd_v8_partition] PTD encoder metrics:", json.dumps(m, indent=2, sort_keys=True))
    return m


def maybe_train_v3(args: argparse.Namespace) -> dict[str, float | int] | None:
    if args.ptd_v3_bundle.exists() and args.ptd_v3_metrics_json.exists():
        return None
    if not args.train_if_missing:
        raise FileNotFoundError(
            f"Missing PTD-v3 bundle/metrics: {args.ptd_v3_bundle}, {args.ptd_v3_metrics_json}. "
            "Pass --train-if-missing to train them."
        )
    print("[strict_ptd_v8_partition] Training PTD-v3 graph bundle...")
    m = train_ptd_v3_models(
        PTDV3TrainConfig(
            ptd_root=args.ptd_root,
            ptd_encoder_ckpt=args.ptd_encoder_ckpt,
            out_bundle=args.ptd_v3_bundle,
            out_metrics_json=args.ptd_v3_metrics_json,
            num_samples=args.synthetic_samples_v3,
            image_size=args.synthetic_image_size,
            random_seed=args.seed,
        )
    )
    print("[strict_ptd_v8_partition] PTD-v3 graph metrics:", json.dumps(m, indent=2, sort_keys=True))
    return m


def maybe_train_v8(args: argparse.Namespace) -> dict[str, float | int] | None:
    if args.ptd_v8_bundle.exists() and args.ptd_v8_metrics_json.exists():
        return None
    if not args.train_if_missing:
        raise FileNotFoundError(
            f"Missing PTD-v8 bundle/metrics: {args.ptd_v8_bundle}, {args.ptd_v8_metrics_json}. "
            "Pass --train-if-missing to train them."
        )
    print("[strict_ptd_v8_partition] Training PTD-v8 partition bundle...")
    m = train_ptd_v8_partition_models(
        PTDV8PartitionTrainConfig(
            ptd_root=args.ptd_root,
            ptd_encoder_ckpt=args.ptd_encoder_ckpt,
            ptd_v3_bundle=args.ptd_v3_bundle,
            out_bundle=args.ptd_v8_bundle,
            out_metrics_json=args.ptd_v8_metrics_json,
            num_samples=args.synthetic_samples_v8,
            image_size=args.synthetic_image_size,
            random_seed=args.seed,
        )
    )
    print("[strict_ptd_v8_partition] PTD-v8 partition metrics:", json.dumps(m, indent=2, sort_keys=True))
    return m


def run_v8(args: argparse.Namespace) -> dict[str, float | int | str]:
    out_v8 = args.out_root / "strict_ptd_v8_partition"
    pipe = TextureSAMV2Pipeline(_cfg_ptd_v8(args.ptd_encoder_ckpt, args.ptd_v8_bundle))
    return pipe.run(
        EvalConfig(
            rwtd_root=args.rwtd_root,
            prompt_masks_root=args.prompt_masks_root,
            baseline_masks_root=args.baseline_masks_root,
            out_dir=out_v8,
            max_images=args.max_images,
        )
    )


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    enc_metrics = maybe_train_ptd_encoder(args)
    v3_metrics = maybe_train_v3(args)
    v8_metrics = maybe_train_v8(args)

    print("[strict_ptd_v8_partition] Running strict PTD-v8 partition on RWTD...")
    v8_sum = run_v8(args)

    report: dict[str, object] = {
        "dataset": "rwtd_kaust256",
        "num_images": int(v8_sum["num_images"]),
        "proposal_source": str(args.prompt_masks_root),
        "protocol": {
            "name": "strict_ptd_v8_partition",
            "uses_rwtd_labels_for_training": False,
            "uses_rwtd_labels_for_hyperparameter_search": False,
            "uses_rwtd_labels_for_final_metric_reporting": True,
            "frozen_after_ptd_validation_only": True,
            "external_training_data": {
                "texture_encoder": "PTD only",
                "merge_model": "PTD-derived synthetic masks only (v3)",
                "partition_selector": "PTD-derived synthetic multi-region texture unions only (v8)",
            },
        },
        "strict_ptd_v8_partition": {
            "miou": float(v8_sum["v2_miou"]),
            "ari": float(v8_sum["v2_ari"]),
            "mean_merged_components": float(v8_sum["mean_merged_components"]),
        },
        "baseline_proxy_v5": {
            "miou": float(v8_sum.get("baseline_miou", 0.0)),
            "ari": float(v8_sum.get("baseline_ari", 0.0)),
        },
        "training": {
            "ptd_encoder_ckpt": str(args.ptd_encoder_ckpt),
            "ptd_encoder_metrics": enc_metrics,
            "ptd_v3_bundle": str(args.ptd_v3_bundle),
            "ptd_v3_metrics": v3_metrics,
            "ptd_v8_bundle": str(args.ptd_v8_bundle),
            "ptd_v8_metrics": v8_metrics,
            "ptd_v8_metrics_json": str(args.ptd_v8_metrics_json),
        },
        "artifacts": {
            "strict_ptd_v8_partition_summary": str(args.out_root / "strict_ptd_v8_partition" / "summary.json"),
            "strict_ptd_v8_partition_per_image": str(args.out_root / "strict_ptd_v8_partition" / "per_image.csv"),
            "strict_ptd_v8_partition_masks_dir": str(args.out_root / "strict_ptd_v8_partition" / "masks"),
        },
    }

    if args.reference_summary.exists():
        ref = json.loads(args.reference_summary.read_text(encoding="utf-8"))
        ref_miou = float(ref.get("v7_miou", ref.get("v2_miou", ref.get("miou", 0.0))))
        ref_ari = float(ref.get("v7_ari", ref.get("v2_ari", ref.get("ari", 0.0))))
        report["reference"] = {
            "name": args.reference_name,
            "path": str(args.reference_summary),
            "miou": ref_miou,
            "ari": ref_ari,
            "delta_miou_v8_vs_reference": float(v8_sum["v2_miou"]) - ref_miou,
            "delta_ari_v8_vs_reference": float(v8_sum["v2_ari"]) - ref_ari,
        }

    out_json = args.out_root / "strict_summary_ptd_v8_partition.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
