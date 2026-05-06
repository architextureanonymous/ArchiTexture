"""RWTD crop-format SAM-2 baseline prediction and evaluation entrypoints.

This module implements the Hugging Face RWTD comparison path for the official
SAM-2 automatic mask generator. It orchestrates variant selection, metric
aggregation, visualization output, optional WandB logging, and run-artifact
generation for ``predict-sam2`` and ``eval-sam2``.

Primary entrypoints:
- ``run_sam2_predict_one()``: evaluate one RWTD sample with one or both SAM-2
  baseline variants.
- ``run_sam2_evaluation()``: evaluate a full RWTD split and write summary
  artifacts.

Inputs are decoded ``RwtdSample`` objects and raw mask predictions returned by
``Sam2BaselineRunner``. Outputs are run directories under ``outputs/rwtd_sam2/``
containing configs, summaries, per-sample CSV rows, and optional visual panels.
The default comparison view in this path is the repository-wide
ArchiTexture-style binary evaluator derived from the aggregated texture-A /
texture-B masks; the raw automatic-mask diagnostics are still exported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from tqdm import tqdm

from rwtd_sam3.data.rwtd import (
    DEFAULT_DATASET_ID,
    RwtdSample,
    get_rwtd_sample,
    iter_rwtd_samples,
    load_split_overview,
)
from rwtd_sam3.eval.metrics import (
    build_canonical_evaluation_fields,
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    compute_canonical_partition_metrics_for_mask_set,
    compute_mask_set_metrics,
)
from rwtd_sam3.eval.runner import (
    WandbSession,
    discovered_package_versions,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID, SAM2_BASELINE_PRESETS, Sam2BaselineRunner
from rwtd_sam3.utils.visualization import (
    build_visual_caption,
    build_visual_footer_lines,
    render_automatic_mask_panel,
    save_automatic_mask_panel,
)


LOGGER = logging.getLogger(__name__)

SAM2_SCALAR_FIELDS = (
    "eval_miou",
    "eval_ari",
    "miou",
    "ari",
    "miou_agg",
    "texture_a_best_iou",
    "texture_b_best_iou",
    "texture_a_agg_iou",
    "texture_b_agg_iou",
    "num_predicted_masks",
    "texture_a_overlap_mask_count",
    "texture_b_overlap_mask_count",
    "predicted_iou_mean",
    "predicted_iou_median",
    "stability_score_mean",
    "stability_score_median",
)


@dataclass(frozen=True)
class Sam2BaselineSampleResult:
    """Per-sample TextureSAM baseline evaluation payload."""

    row: dict[str, Any]
    metric_summary: str
    prediction_masks: tuple[np.ndarray, ...]
    aggregated_prediction_a: np.ndarray
    aggregated_prediction_b: np.ndarray


def run_sam2_predict_one(args) -> dict[str, Any]:
    """Run one RWTD sample through the official SAM-2 baseline variants."""

    sample = get_rwtd_sample(
        split=args.split,
        index=args.index,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
    )
    output_dir = resolve_sam2_output_dir(args.output_dir, variant=args.variant, split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam2_run_config(args, overview=None)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_sam2_experiment_terms_markdown(args))

    runner = Sam2BaselineRunner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        points_per_batch=args.points_per_batch,
        fast_cuda=args.fast_cuda,
        compile_image_encoder=args.compile_image_encoder,
    )

    results: dict[str, Any] = {}
    for variant in resolve_sam2_variants(args.variant):
        variant_dir = output_dir if args.variant != "both" else output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        evaluation = evaluate_sam2_baseline_sample(sample=sample, variant=variant, runner=runner)

        prediction_path = variant_dir / "prediction.json"
        write_json(prediction_path, evaluation.row)
        panel_path = None
        if args.save_visuals:
            panel_path = save_automatic_mask_panel(
                variant_dir / "prediction.png",
                sample=sample,
                prediction_masks=evaluation.prediction_masks,
                protocol=variant,
                aggregated_prediction_a=evaluation.aggregated_prediction_a,
                aggregated_prediction_b=evaluation.aggregated_prediction_b,
                metric_summary=evaluation.metric_summary,
            )
            write_jsonl(
                variant_dir / "visuals_manifest.jsonl",
                [
                    build_sam2_visual_record(
                        sample=sample,
                        evaluation=evaluation,
                        variant=variant,
                        visual_path=Path("prediction.png"),
                    )
                ],
            )

        results[variant] = {
            "prediction_json": str(prediction_path),
            "visualization": str(panel_path) if panel_path is not None else None,
            "metrics": evaluation.row,
        }
    return results


def run_sam2_evaluation(args) -> dict[str, Any]:
    """Run the SAM-2 / SAM-2* baseline evaluation on RWTD."""

    overview = load_split_overview(split=args.split, dataset_id=args.dataset_id, cache_dir=args.cache_dir)
    samples = list(
        iter_rwtd_samples(
            split=args.split,
            dataset_id=args.dataset_id,
            cache_dir=args.cache_dir,
            limit=args.limit,
        )
    )
    if not samples:
        raise RuntimeError(f"Split '{args.split}' did not yield any samples to evaluate.")

    output_dir = resolve_sam2_output_dir(args.output_dir, variant=args.variant, split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam2_run_config(args, overview=overview)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_sam2_experiment_terms_markdown(args))

    wandb_session = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or output_dir.name,
        config=config_payload,
    )
    runner = Sam2BaselineRunner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        points_per_batch=args.points_per_batch,
        fast_cuda=args.fast_cuda,
        compile_image_encoder=args.compile_image_encoder,
    )

    all_variant_summaries: dict[str, Any] = {}
    try:
        for variant in resolve_sam2_variants(args.variant):
            variant_dir = output_dir if args.variant != "both" else output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            if args.save_visuals:
                (variant_dir / "visuals").mkdir(parents=True, exist_ok=True)
            write_text(variant_dir / "experiment_terms.md", build_sam2_experiment_terms_markdown(args))

            rows: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            visual_records: list[dict[str, Any]] = []

            progress = tqdm(samples, desc=f"eval:{variant}", unit="sample")
            for step, sample in enumerate(progress):
                try:
                    evaluation = evaluate_sam2_baseline_sample(sample=sample, variant=variant, runner=runner)
                except Exception as exc:
                    failure = {"crop_name": sample.crop_name, "error": str(exc)}
                    if args.failure_policy == "skip":
                        failures.append(failure)
                        LOGGER.error("Skipping sample %s: %s", sample.crop_name, exc)
                        continue
                    raise

                rows.append(evaluation.row)
                progress.set_postfix({"crop": sample.crop_name, "eval_miou": f"{evaluation.row['eval_miou']:.3f}"})

                if args.save_visuals:
                    visual_path = Path("visuals") / f"{sample.crop_name}.png"
                    save_automatic_mask_panel(
                        variant_dir / visual_path,
                        sample=sample,
                        prediction_masks=evaluation.prediction_masks,
                        protocol=variant,
                        aggregated_prediction_a=evaluation.aggregated_prediction_a,
                        aggregated_prediction_b=evaluation.aggregated_prediction_b,
                        metric_summary=evaluation.metric_summary,
                    )
                    visual_records.append(
                        build_sam2_visual_record(
                            sample=sample,
                            evaluation=evaluation,
                            variant=variant,
                            visual_path=visual_path,
                        )
                    )

                if args.wandb:
                    scalar_log = {
                        "variant": variant,
                        "eval_miou": evaluation.row["eval_miou"],
                        "eval_ari": evaluation.row["eval_ari"],
                        "miou": evaluation.row["miou"],
                        "ari": evaluation.row["ari"],
                        "miou_agg": evaluation.row["miou_agg"],
                    }
                    wandb_session.log(scalar_log, step=step)
                    if step % args.log_every == 0:
                        preview_image = render_automatic_mask_panel(
                            sample=sample,
                            prediction_masks=evaluation.prediction_masks,
                            protocol=variant,
                            aggregated_prediction_a=evaluation.aggregated_prediction_a,
                            aggregated_prediction_b=evaluation.aggregated_prediction_b,
                            metric_summary=evaluation.metric_summary,
                        )
                        caption = build_visual_caption(
                            sample=sample,
                            protocol=variant,
                            metric_summary=evaluation.metric_summary,
                        )
                        wandb_session.log_preview(preview_image, caption=caption, step=step)

            if not rows:
                raise RuntimeError(
                    f"Variant '{variant}' produced no successful evaluations. "
                    "Check the recorded failures for details."
                )

            write_csv(variant_dir / "per_sample_metrics.csv", rows)
            if args.save_visuals:
                write_jsonl(variant_dir / "visuals_manifest.jsonl", visual_records)
            summary = build_sam2_variant_summary(
                variant=variant,
                split=args.split,
                dataset_id=args.dataset_id,
                model_id=args.model_id,
                rows=rows,
                failures=failures,
                num_total_samples=len(samples),
            )
            write_json(variant_dir / "summary.json", summary)
            write_text(variant_dir / "summary.md", build_sam2_markdown_summary(summary))
            all_variant_summaries[variant] = summary

        if args.variant == "both":
            write_json(output_dir / "summary.json", {"variants": all_variant_summaries})
            write_text(
                output_dir / "summary.md",
                "# RWTD SAM-2 Baseline Evaluation\n\nBoth baseline variants completed successfully.\n",
            )
    finally:
        wandb_session.finish()

    return all_variant_summaries


def evaluate_sam2_baseline_sample(
    sample: RwtdSample,
    variant: str,
    runner: Sam2BaselineRunner,
) -> Sam2BaselineSampleResult:
    """Evaluate one RWTD sample with one SAM-2 baseline variant."""

    predictions = runner.generate_masks(sample.image, variant=variant)
    prediction_masks = [prediction.segmentation for prediction in predictions]
    gt_regions = (sample.texture_a_mask, sample.texture_b_mask)
    metrics = compute_mask_set_metrics(prediction_masks, gt_regions)
    canonical_partition, aggregated_masks, overlap_counts = compute_canonical_partition_metrics_for_mask_set(
        prediction_masks,
        sample.texture_a_mask,
        sample.texture_b_mask,
    )

    predicted_ious = [prediction.predicted_iou for prediction in predictions]
    stability_scores = [prediction.stability_score for prediction in predictions]
    preset = SAM2_BASELINE_PRESETS[variant]
    row = {
        "variant": variant,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "points_per_side": int(preset["points_per_side"]),
        "stability_score_thresh": float(preset["stability_score_thresh"]),
        "num_predicted_masks": metrics.num_predicted_masks,
        "evaluation_view": "gt_overlap_aggregated_partition",
        "miou": metrics.miou,
        "ari": metrics.ari,
        "miou_agg": metrics.aggregated_miou,
        "texture_a_best_iou": metrics.region_best_ious[0],
        "texture_b_best_iou": metrics.region_best_ious[1],
        "texture_a_agg_iou": metrics.region_aggregated_ious[0],
        "texture_b_agg_iou": metrics.region_aggregated_ious[1],
        "texture_a_overlap_mask_count": overlap_counts[0],
        "texture_b_overlap_mask_count": overlap_counts[1],
        "predicted_iou_mean": float(np.mean(predicted_ious)) if predicted_ious else 0.0,
        "predicted_iou_median": float(np.median(predicted_ious)) if predicted_ious else 0.0,
        "stability_score_mean": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "stability_score_median": float(np.median(stability_scores)) if stability_scores else 0.0,
    }
    row.update(
        build_canonical_evaluation_fields(
            miou=canonical_partition.miou,
            ari=canonical_partition.ari,
            evaluation_view="gt_overlap_aggregated_partition",
        )
    )
    metric_summary = (
        f"Eval mIoU={canonical_partition.miou:.3f} "
        f"Eval ARI={canonical_partition.ari:.3f} "
        f"Raw mIoU={metrics.miou:.3f} "
        f"Raw ARI={metrics.ari:.3f} "
        f"Masks={metrics.num_predicted_masks}"
    )
    return Sam2BaselineSampleResult(
        row=row,
        metric_summary=metric_summary,
        prediction_masks=tuple(prediction_masks),
        aggregated_prediction_a=aggregated_masks[0],
        aggregated_prediction_b=aggregated_masks[1],
    )


def build_sam2_variant_summary(
    variant: str,
    split: str,
    dataset_id: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    num_total_samples: int,
) -> dict[str, Any]:
    """Aggregate one SAM-2 baseline variant over a split."""

    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in SAM2_SCALAR_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in SAM2_SCALAR_FIELDS}
    return {
        "dataset_id": dataset_id,
        "split": split,
        "variant": variant,
        "model_id": model_id,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": "gt_overlap_aggregated_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[CANONICAL_PRIMARY_METRIC],
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[CANONICAL_SECONDARY_METRIC],
        "primary_metric_reason": (
            "This repository now defaults to one ArchiTexture-style binary evaluator. "
            "For raw SAM-2 mask sets, the canonical view unions masks by GT-overlap into "
            "one texture-A mask and one texture-B mask before scoring mIoU/ARI."
        ),
        "num_total_samples": num_total_samples,
        "num_evaluated_samples": len(rows),
        "num_failed_samples": len(failures),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }


def build_sam2_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for one SAM-2 baseline variant."""

    lines = [
        "# RWTD SAM-2 Baseline Summary",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Split: `{summary['split']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Evaluated samples: `{summary['num_evaluated_samples']}` / `{summary['num_total_samples']}`",
        f"- Failed samples: `{summary['num_failed_samples']}`",
        f"- Default comparison view: `{summary['primary_metric_name']}` / "
        f"`{summary['secondary_metric_name']}` = "
        f"`{summary['primary_metric_value']:.6f}` / `{summary['secondary_metric_value']:.6f}`",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Mean | Median |",
        "| --- | ---: | ---: |",
    ]
    for metric_name in SAM2_SCALAR_FIELDS:
        lines.append(
            f"| `{metric_name}` | {summary['mean_metrics'][metric_name]:.6f} | "
            f"{summary['median_metrics'][metric_name]:.6f} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['crop_name']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def build_sam2_run_config(args, overview) -> dict[str, Any]:
    """Build a stable config payload for a SAM-2 baseline run."""

    config = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "variant": args.variant,
        "model_id": args.model_id,
        "device": args.device,
        "points_per_batch": args.points_per_batch,
        "fast_cuda": args.fast_cuda,
        "compile_image_encoder": args.compile_image_encoder,
        "limit": getattr(args, "limit", None),
        "failure_policy": getattr(args, "failure_policy", "abort"),
        "save_visuals": args.save_visuals,
        "wandb": getattr(args, "wandb", False),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant_presets": SAM2_BASELINE_PRESETS,
        "versions": discovered_package_versions(),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": "gt_overlap_aggregated_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
    }
    if overview is not None:
        config["discovered_split_size"] = overview.num_examples
        config["discovered_split_sizes"] = overview.split_sizes
        config["discovered_features"] = list(overview.features)
    return config


def build_sam2_experiment_terms_markdown(args) -> str:
    """Render a short glossary for the SAM-2 baseline run outputs."""

    variants = ", ".join(f"`{name}`" for name in resolve_sam2_variants(args.variant))
    return (
        "# TextureSAM Baseline Definitions\n\n"
        f"- Dataset: `{args.dataset_id}`\n"
        f"- Split: `{args.split}`\n"
        f"- Requested variant setting: `{args.variant}`\n"
        f"- Executed variant(s): {variants}\n"
        f"- Model: `{args.model_id}`\n\n"
        "## Runtime Settings\n\n"
        f"- `device`: `{args.device}`\n"
        f"- `points_per_batch`: `{args.points_per_batch}` "
        "(or the runtime default of `256` on CUDA / `64` on CPU when omitted)\n"
        f"- `fast_cuda`: `{args.fast_cuda}`\n"
        f"- `compile_image_encoder`: `{args.compile_image_encoder}`\n\n"
        "## Variant Definitions\n\n"
        "- `sam2`: official SAM-2 AMG defaults with `points_per_side=32` and `stability_score_thresh=0.95`.\n"
        "- `sam2_star`: same weights with `points_per_side=64` and `stability_score_thresh=0.2`.\n\n"
        "## Metric Definitions\n\n"
        "- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.\n"
        f"- `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}`: the repo-wide default ArchiTexture-style evaluator. In this path it first unions all raw masks with non-empty overlap against each GT region and then scores the resulting binary partition.\n"
        "- `miou`: non-aggregated mean IoU using one-to-one IoU matching between raw predicted masks and ground-truth regions.\n"
        "- `ari`: non-aggregated Adjusted Rand Index between ground-truth region labels and the pixelwise partition induced by raw predicted-mask memberships.\n"
        "- `miou_agg`: aggregated mean IoU after unioning all raw masks with non-empty overlap against each ground-truth region.\n"
        "- `texture_a_overlap_mask_count` / `texture_b_overlap_mask_count`: number of raw masks contributing to each aggregated region.\n\n"
        "## Evaluation Notes\n\n"
        f"- The default comparison view is `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}` under the shared `{CANONICAL_EVALUATION_CONTRACT}` contract.\n"
        "- Raw-mask `miou`, raw-mask `ari`, and `miou_agg` are retained as diagnostics.\n"
        "- Raw ARI uses the full predicted-mask membership partition, so overlapping and nested masks still penalize fragmentation.\n"
        "- Saved visuals include a raw categorical-mask panel with one color per visible proposal plus a separate aggregated A/B panel.\n"
    )


def build_sam2_visual_record(
    sample: RwtdSample,
    evaluation: Sam2BaselineSampleResult,
    variant: str,
    visual_path: Path,
) -> dict[str, Any]:
    """Build one machine-readable visual metadata record for the SAM-2 baseline."""

    record = dict(evaluation.row)
    record.update(
        {
            "visual_path": str(visual_path),
            "caption": build_visual_caption(
                sample=sample,
                protocol=variant,
                metric_summary=evaluation.metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=sample,
                protocol=variant,
                metric_summary=evaluation.metric_summary,
            ),
            "original_texture_a": sample.original_texture_a,
            "original_texture_b": sample.original_texture_b,
            "oracle_points_a_count": len(sample.oracle_points_a),
            "oracle_points_b_count": len(sample.oracle_points_b),
        }
    )
    return record


def resolve_sam2_variants(variant: str) -> tuple[str, ...]:
    if variant == "both":
        return ("sam2", "sam2_star")
    return (variant,)


def resolve_sam2_output_dir(output_dir: str | None, variant: str, split: str) -> Path:
    """Resolve the output directory for a SAM-2 baseline run."""

    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"sam2-baseline-{variant}-{split}-{timestamp}"
    return Path("outputs") / "rwtd_sam2" / run_name
