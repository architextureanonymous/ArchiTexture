from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
SRC_DIR = SCRIPT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rwtd_sam3.eval.architexture_binary import run_architexture_binary_evaluation
from rwtd_sam3.eval.cstd_binary import run_cstd_binary_evaluation
from rwtd_sam3.eval.sam3_auto import run_sam3_auto_evaluation


FEATURE_VARIANT = "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2"


@dataclass(frozen=True)
class DatasetRun:
    dataset: str
    output_dir: Path
    summary_path: Path
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live feature-clustering table-1 reproduction.")
    parser.add_argument("--output-root", required=True, help="Fresh output root for this reproducibility run.")
    parser.add_argument("--rwtd-root", default=str(REPO_ROOT / "datasets" / "RWTD"))
    parser.add_argument("--stld-root", default=str(REPO_ROOT / "datasets" / "STLD"))
    parser.add_argument("--cstd-root", default=str(REPO_ROOT / "datasets" / "CSTD"))
    parser.add_argument("--kaggle-dataset", default="architexanonymous/cstd-controlnet-synthetic-texture",
                        help="Kaggle dataset slug to download CSTD from if --cstd-root is absent.")
    parser.add_argument("--model-id", default="facebook/sam2-hiera-small")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--official-checkpoint-path", default=None)
    return parser.parse_args()


def _env_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _base_eval_args(**kwargs: Any) -> SimpleNamespace:
    payload = {
        "cache_dir": None,
        "hf_token": _env_token(),
        "wandb": False,
        "wandb_project": "repro-feature-clustering",
        "wandb_run_name": None,
        "log_every": 25,
        "failure_policy": "abort",
        "save_visuals": False,
        "dataset_partition": None,
    }
    payload.update(kwargs)
    return SimpleNamespace(**payload)


def _run_rwtd(output_dir: Path, args: argparse.Namespace) -> DatasetRun:
    run_args = _base_eval_args(
        command="eval-sam3-auto",
        dataset_id=str(Path(args.rwtd_root).expanduser()),
        split="all",
        variant=FEATURE_VARIANT,
        model_id=args.model_id,
        device=args.device,
        output_dir=str(output_dir),
        limit=None,
        score_threshold=0.0,
        mask_threshold=0.0,
        boundary_tolerance_px=2,
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        seed=args.seed,
        official_checkpoint_path=args.official_checkpoint_path,
    )
    run_sam3_auto_evaluation(run_args)
    summary_path = output_dir / "summary.json"
    command = (
        "python scripts/feature_clustering/main.py --seed "
        f"{args.seed} eval-sam3-auto --dataset-id {args.rwtd_root} --split all --variant {FEATURE_VARIANT} "
        f"--device {args.device} --output-dir {output_dir}"
    )
    return DatasetRun(dataset="RWTD", output_dir=output_dir, summary_path=summary_path, command=command)


def _ensure_cstd(cstd_root: Path, kaggle_dataset: str) -> Path:
    """Download CSTD from Kaggle if the local root is absent."""
    if (cstd_root / "images").is_dir() and (cstd_root / "regions").is_dir():
        return cstd_root
    print(f"CSTD not found at {cstd_root} — downloading from Kaggle ({kaggle_dataset}) …")
    try:
        import kaggle  # type: ignore
        # KGAT_ tokens require KAGGLE_API_TOKEN for Bearer auth; fall back to KAGGLE_KEY.
        if os.environ.get("KAGGLE_KEY") and not os.environ.get("KAGGLE_API_TOKEN"):
            os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]
        kaggle.api.authenticate()
        cstd_root.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(kaggle_dataset, path=str(cstd_root), unzip=True, quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download CSTD from Kaggle ({kaggle_dataset}): {exc}\n"
            "Set KAGGLE_API_TOKEN (or KAGGLE_KEY) env var, or place kaggle.json in ~/.kaggle/."
        ) from exc
    return cstd_root


def _run_stld(output_dir: Path, args: argparse.Namespace) -> DatasetRun:
    run_args = _base_eval_args(
        command="eval-architexture-binary",
        route="stld",
        benchmark_root=str(Path(args.stld_root).expanduser()),
        variant=FEATURE_VARIANT,
        model_id=args.model_id,
        device=args.device,
        output_dir=str(output_dir),
        limit=None,
        seed=args.seed,
        official_checkpoint_path=args.official_checkpoint_path,
    )
    run_architexture_binary_evaluation(run_args)
    summary_path = output_dir / "summary.json"
    command = (
        "python scripts/feature_clustering/main.py --seed "
        f"{args.seed} eval-architexture-binary --route stld --benchmark-root {args.stld_root} "
        f"--variant {FEATURE_VARIANT} --device {args.device} --output-dir {output_dir}"
    )
    return DatasetRun(dataset="STLD", output_dir=output_dir, summary_path=summary_path, command=command)


def _run_cstd(output_dir: Path, args: argparse.Namespace) -> DatasetRun:
    cstd_root = _ensure_cstd(
        Path(args.cstd_root).expanduser().resolve(),
        args.kaggle_dataset,
    )
    run_args = _base_eval_args(
        command="eval-cstd-binary",
        dataset_root=str(cstd_root),
        variant=FEATURE_VARIANT,
        model_id=args.model_id,
        device=args.device,
        output_dir=str(output_dir),
        limit=None,
        save_visuals=True,
        save_pooled_feature_pca_overlay=True,
        failure_policy="skip",
        wandb=False,
        wandb_project="repro-feature-clustering",
        wandb_run_name=None,
        log_every=25,
        dataset_partition=None,
        seed=args.seed,
        official_checkpoint_path=args.official_checkpoint_path,
    )
    run_cstd_binary_evaluation(run_args)
    summary_path = output_dir / "summary.json"
    command = (
        f"python scripts/feature_clustering/main.py --seed {args.seed} eval-cstd-binary "
        f"--dataset-root {cstd_root} --variant {FEATURE_VARIANT} "
        f"--device {args.device} --save-visuals --save-pooled-feature-pca-overlay "
        f"--output-dir {output_dir}"
    )
    return DatasetRun(dataset="CSTD", output_dir=output_dir, summary_path=summary_path, command=command)


def _read_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary file is not a JSON object: {summary_path}")
    return payload


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

    runs = [
        _run_rwtd(output_root / "rwtd_eval", args),
        _run_stld(output_root / "stld_eval", args),
        _run_cstd(output_root / "cstd_eval", args),
    ]

    rows = []
    for run in runs:
        summary = _read_summary(run.summary_path)
        mean_metrics = summary.get("mean_metrics", {})
        rows.append(
            {
                "dataset": run.dataset,
                "eval_miou": float(mean_metrics["eval_miou"]),
                "eval_ari": float(mean_metrics["eval_ari"]),
                "num_evaluated_samples": int(summary["num_evaluated_samples"]),
                "num_failed_samples": int(summary["num_failed_samples"]),
                "summary_path": str(run.summary_path),
                "output_dir": str(run.output_dir),
                "command": run.command,
            }
        )

    csv_path = output_root / "table_1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "eval_miou",
                "eval_ari",
                "num_evaluated_samples",
                "num_failed_samples",
                "summary_path",
                "output_dir",
                "command",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Table 1",
        "",
        "| Dataset | mIoU | ARI | Evaluated | Failed | Summary |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['dataset']}` | `{row['eval_miou']:.6f}` | `{row['eval_ari']:.6f}` | "
            f"`{row['num_evaluated_samples']}` | `{row['num_failed_samples']}` | `{row['summary_path']}` |"
        )
    (output_root / "table_1.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = {
        "feature_variant": FEATURE_VARIANT,
        "output_root": str(output_root),
        "runs": rows,
    }
    (output_root / "table_1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote table_1.csv to {csv_path}")
    for row in rows:
        print(f"{row['dataset']}: mIoU={row['eval_miou']:.6f} ARI={row['eval_ari']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
