from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
SRC_DIR = SCRIPT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rwtd_sam3.data.architexture_binary import get_architexture_binary_sample
from rwtd_sam3.data.rwtd import get_rwtd_sample
from rwtd_sam3.eval.architexture_binary import evaluate_architexture_binary_sample
from rwtd_sam3.eval.experiment_registry import build_cross_dataset_experiment_runner
from rwtd_sam3.eval.sam3_auto import evaluate_sam3_auto_sample
from rwtd_sam3.utils.visualization import render_feature_cluster_figure2_panel, render_pooled_feature_pca_overlay


FEATURE_VARIANT = "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live Figure 2 feature-clustering reproduction.")
    parser.add_argument("--output-root", required=True, help="Fresh output root for this reproducibility run.")
    parser.add_argument("--rwtd-root", default=str(REPO_ROOT / "datasets" / "RWTD"))
    parser.add_argument("--stld-root", default=str(REPO_ROOT / "datasets" / "STLD"))
    parser.add_argument("--model-id", default="facebook/sam2-hiera-small")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--official-checkpoint-path", default=None)
    parser.add_argument("--examples-per-dataset", type=int, default=3)
    return parser.parse_args()


def _env_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _pick_indices(total: int, count: int) -> list[int]:
    if total < 1 or count < 1:
        return []
    if total == 1:
        return [0]
    candidates = [6, 11, 125]
    indices: list[int] = []
    for candidate in candidates:
        if candidate not in indices:
            indices.append(candidate)
        if len(indices) >= count:
            break
    return indices[:count]


def _save_panel(
    *,
    dataset: str,
    sample_index: int,
    crop_name: str,
    sample,
    output_dir: Path,
    runner,
    evaluation,
) -> dict[str, Any]:
    feature_map = runner.extract_pooled_feature_map_for_visualization(sample.image)
    pooled_overlay = render_pooled_feature_pca_overlay(sample.image, feature_map)
    panel = render_feature_cluster_figure2_panel(
        sample=sample,
        pooled_feature_pca_overlay=pooled_overlay,
        prediction_a=evaluation.prediction_masks[0],
        prediction_b=evaluation.prediction_masks[1],
    )
    dataset_dir = output_dir / dataset.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    panel_path = dataset_dir / f"{sample_index:03d}_{crop_name}_figure2.png"
    panel.save(panel_path)
    return {
        "dataset": dataset,
        "sample_index": sample_index,
        "crop_name": crop_name,
        "miou": float(evaluation.row["miou"]),
        "ari": float(evaluation.row["ari"]),
        "panel_path": str(panel_path),
        "metric_summary": evaluation.metric_summary,
    }


def _run_rwtd(output_root: Path, args: argparse.Namespace, runner) -> list[dict[str, Any]]:
    from rwtd_sam3.data.rwtd import load_split_overview

    overview = load_split_overview(split="all", dataset_id=str(Path(args.rwtd_root).expanduser()))
    indices = _pick_indices(overview.num_examples, args.examples_per_dataset)
    rows: list[dict[str, Any]] = []
    for index in indices:
        sample = get_rwtd_sample(split="all", index=index, dataset_id=str(Path(args.rwtd_root).expanduser()))
        evaluation = evaluate_sam3_auto_sample(sample=sample, variant=FEATURE_VARIANT, runner=runner)
        rows.append(
            _save_panel(
                dataset="RWTD",
                sample_index=index,
                crop_name=sample.crop_name,
                sample=sample,
                output_dir=output_root,
                runner=runner,
                evaluation=evaluation,
            )
        )
    return rows


def _run_stld(output_root: Path, args: argparse.Namespace, runner) -> list[dict[str, Any]]:
    from rwtd_sam3.data.architexture_binary import load_architexture_binary_overview

    overview = load_architexture_binary_overview(route="stld", benchmark_root=str(Path(args.stld_root).expanduser()))
    indices = _pick_indices(overview.num_examples, args.examples_per_dataset)
    rows: list[dict[str, Any]] = []
    for index in indices:
        sample = get_architexture_binary_sample(benchmark_root=str(Path(args.stld_root).expanduser()), route="stld", index=index)
        evaluation = evaluate_architexture_binary_sample(sample=sample, route="stld", variant=FEATURE_VARIANT, runner=runner)
        rows.append(
            _save_panel(
                dataset="STLD",
                sample_index=index,
                crop_name=sample.crop_name,
                sample=sample,
                output_dir=output_root,
                runner=runner,
                evaluation=evaluation,
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    rwtd_root = Path(args.rwtd_root).expanduser().resolve()
    stld_root = Path(args.stld_root).expanduser().resolve()
    if not rwtd_root.exists():
        raise FileNotFoundError(f"RWTD root does not exist: {rwtd_root}")
    if not stld_root.exists():
        raise FileNotFoundError(f"STLD root does not exist: {stld_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runner = build_cross_dataset_experiment_runner(
        FEATURE_VARIANT,
        model_id=args.model_id,
        device=args.device,
        hf_token=_env_token(),
        official_checkpoint_path=args.official_checkpoint_path,
    )

    rows = []
    rows.extend(_run_rwtd(output_root, args, runner))
    rows.extend(_run_stld(output_root, args, runner))

    csv_path = output_root / "figure_2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_index", "crop_name", "miou", "ari", "panel_path", "metric_summary"],
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "feature_variant": FEATURE_VARIANT,
        "output_root": str(output_root),
        "examples_per_dataset": args.examples_per_dataset,
        "rows": rows,
    }
    (output_root / "figure_2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote figure_2.csv to {csv_path}")
    for row in rows:
        print(f"{row['dataset']}[{row['sample_index']}]: mIoU={row['miou']:.6f} ARI={row['ari']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
