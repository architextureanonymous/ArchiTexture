"""Paper-faithful SAM-2 evaluation on the official TextureSAM RWTD release.

This module mirrors the scoring conventions of TextureSAM's released RWTD
scripts on the local ``Kaust256`` dataset layout. It exists separately from the
crop-format RWTD baseline so the repository can preserve the paper-facing
comparison view without conflating it with the Hugging Face RWTD adaptation.

Primary entrypoints:
- ``run_sam2_official_predict_one()``: evaluate one local ``Kaust256`` sample.
- ``run_sam2_official_evaluation()``: evaluate the full local dataset and write
  aggregate summaries.

Inputs are decoded ``Kaust256Sample`` objects and raw SAM-2 mask proposals.
Outputs are run directories under ``outputs/rwtd_sam2_official/`` containing
config metadata, per-sample rows, Markdown summaries, and optional visual
artifacts. The released TextureSAM metrics remain in the artifacts for audit
purposes, but the repo-default comparison view now follows the shared
ArchiTexture-style binary evaluator.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any

import numpy as np
from tqdm import tqdm

from rwtd_sam3.data.kaust256 import Kaust256Sample, get_kaust256_sample, iter_kaust256_samples, load_kaust256_overview
from rwtd_sam3.eval.metrics import (
    OfficialTextureSamMetrics,
    boundary_from_region_masks,
    build_canonical_evaluation_fields,
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    compute_canonical_partition_metrics_for_official_label_map,
    compute_official_texturesam_metrics,
    invert_binary_label_map,
    rasterize_matched_masks,
    reindex_label_map,
)
from rwtd_sam3.eval.runner import WandbSession, discovered_package_versions, write_csv, write_json, write_jsonl, write_text
from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID, SAM2_BASELINE_PRESETS, Sam2BaselineRunner
from rwtd_sam3.utils.visualization import (
    build_visual_caption,
    build_visual_footer_lines,
    render_automatic_mask_panel,
    save_automatic_mask_panel,
)


LOGGER = logging.getLogger(__name__)

OFFICIAL_SAM2_SCALAR_FIELDS = (
    "eval_miou",
    "eval_ari",
    "miou",
    "ari",
    "miou_agg",
    "miou_agg_original",
    "miou_agg_inverted",
    "label0_avg_iou",
    "label1_avg_iou",
    "label0_avg_ari",
    "label1_avg_ari",
    "label0_overlap_mask_count",
    "label1_overlap_mask_count",
    "num_predicted_masks",
    "predicted_iou_mean",
    "predicted_iou_median",
    "stability_score_mean",
    "stability_score_median",
)


@dataclass(frozen=True)
class Sam2OfficialSampleResult:
    """Per-sample official TextureSAM RWTD evaluation payload."""

    row: dict[str, Any]
    metric_summary: str
    prediction_masks: tuple[np.ndarray, ...]
    aggregated_prediction_a: np.ndarray
    aggregated_prediction_b: np.ndarray
    metrics: OfficialTextureSamMetrics


def run_sam2_official_predict_one(args) -> dict[str, Any]:
    """Run one official Kaust256 RWTD sample through the SAM-2 baseline variants."""

    sample = get_kaust256_sample(root_dir=args.rwtd_root, index=args.index)
    output_dir = resolve_sam2_official_output_dir(args.output_dir, variant=args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam2_official_run_config(args, overview=None)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_sam2_official_experiment_terms_markdown(args))

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
        evaluation = evaluate_sam2_official_sample(sample=sample, variant=variant, runner=runner)

        prediction_path = variant_dir / "prediction.json"
        write_json(prediction_path, evaluation.row)
        panel_path = None
        if args.save_visuals:
            panel_path = save_automatic_mask_panel(
                variant_dir / "prediction.png",
                sample=make_visual_sample(sample),
                prediction_masks=evaluation.prediction_masks,
                protocol=f"{variant}:official",
                aggregated_prediction_a=evaluation.aggregated_prediction_a,
                aggregated_prediction_b=evaluation.aggregated_prediction_b,
                metric_summary=evaluation.metric_summary,
            )
            write_jsonl(
                variant_dir / "visuals_manifest.jsonl",
                [
                    build_sam2_official_visual_record(
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


def run_sam2_official_evaluation(args) -> dict[str, Any]:
    """Run the official TextureSAM RWTD SAM-2 baseline evaluation."""

    overview = load_kaust256_overview(args.rwtd_root)
    samples = list(iter_kaust256_samples(root_dir=args.rwtd_root, limit=args.limit))
    if not samples:
        raise RuntimeError("Kaust256 did not yield any samples to evaluate.")

    output_dir = resolve_sam2_official_output_dir(args.output_dir, variant=args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam2_official_run_config(args, overview=overview)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_sam2_official_experiment_terms_markdown(args))

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
            write_text(variant_dir / "experiment_terms.md", build_sam2_official_experiment_terms_markdown(args))

            rows: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            visual_records: list[dict[str, Any]] = []

            progress = tqdm(samples, desc=f"eval:{variant}:official", unit="sample")
            for step, sample in enumerate(progress):
                try:
                    evaluation = evaluate_sam2_official_sample(sample=sample, variant=variant, runner=runner)
                except Exception as exc:
                    failure = {"image_id": sample.image_id, "error": str(exc)}
                    if args.failure_policy == "skip":
                        failures.append(failure)
                        LOGGER.error("Skipping sample %s: %s", sample.image_id, exc)
                        continue
                    raise

                rows.append(evaluation.row)
                progress.set_postfix({"image": sample.image_id, "eval_miou": f"{evaluation.row['eval_miou']:.3f}"})

                if args.save_visuals:
                    visual_path = Path("visuals") / f"{sample.image_id}.png"
                    save_automatic_mask_panel(
                        variant_dir / visual_path,
                        sample=make_visual_sample(sample),
                        prediction_masks=evaluation.prediction_masks,
                        protocol=f"{variant}:official",
                        aggregated_prediction_a=evaluation.aggregated_prediction_a,
                        aggregated_prediction_b=evaluation.aggregated_prediction_b,
                        metric_summary=evaluation.metric_summary,
                    )
                    visual_records.append(
                        build_sam2_official_visual_record(
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
                            sample=make_visual_sample(sample),
                            prediction_masks=evaluation.prediction_masks,
                            protocol=f"{variant}:official",
                            aggregated_prediction_a=evaluation.aggregated_prediction_a,
                            aggregated_prediction_b=evaluation.aggregated_prediction_b,
                            metric_summary=evaluation.metric_summary,
                        )
                        caption = build_visual_caption(
                            sample=make_visual_sample(sample),
                            protocol=f"{variant}:official",
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
            summary = build_sam2_official_variant_summary(
                variant=variant,
                model_id=args.model_id,
                rows=rows,
                failures=failures,
                num_total_samples=len(samples),
                kaust_root=overview.root_dir,
            )
            write_json(variant_dir / "summary.json", summary)
            write_text(variant_dir / "summary.md", build_sam2_official_markdown_summary(summary))
            all_variant_summaries[variant] = summary

        if args.variant == "both":
            write_json(output_dir / "summary.json", {"variants": all_variant_summaries})
            write_text(
                output_dir / "summary.md",
                "# TextureSAM Official RWTD SAM-2 Evaluation\n\nBoth baseline variants completed successfully.\n",
            )
    finally:
        wandb_session.finish()

    return all_variant_summaries


def evaluate_sam2_official_sample(
    sample: Kaust256Sample,
    variant: str,
    runner: Sam2BaselineRunner,
) -> Sam2OfficialSampleResult:
    """Evaluate one Kaust256 sample with TextureSAM's released RWTD scripts."""

    predictions = runner.generate_masks(sample.image, variant=variant)
    prediction_masks = [prediction.segmentation for prediction in predictions]
    metrics = compute_official_texturesam_metrics(prediction_masks, sample.label_mask)
    canonical_partition, canonical_predictions, _, _ = compute_canonical_partition_metrics_for_official_label_map(
        prediction_masks,
        sample.label_mask,
    )
    pred_label_map_original = canonical_predictions[0]
    pred_label_map_inverted = canonical_predictions[1]

    predicted_ious = [prediction.predicted_iou for prediction in predictions]
    stability_scores = [prediction.stability_score for prediction in predictions]
    preset = SAM2_BASELINE_PRESETS[variant]
    row = {
        "variant": variant,
        "dataset_format": "official_kaust256",
        "sample_index": sample.index,
        "image_id": sample.image_id,
        "label_values": list(metrics.label_values),
        "evaluation_view": "official_binary_partition",
        "points_per_side": int(preset["points_per_side"]),
        "stability_score_thresh": float(preset["stability_score_thresh"]),
        "num_predicted_masks": metrics.num_predicted_masks,
        "num_counted_regions": metrics.num_counted_regions,
        "official_raw_scored": metrics.num_counted_regions > 0,
        "official_agg_scored": metrics.num_predicted_masks > 0,
        "miou": metrics.miou,
        "ari": metrics.ari,
        "miou_agg": metrics.aggregated_miou,
        "miou_agg_original": metrics.aggregated_miou_original,
        "miou_agg_inverted": metrics.aggregated_miou_inverted,
        "label0_value": metrics.label_values[0],
        "label1_value": metrics.label_values[1],
        "label0_avg_iou": metrics.region_average_ious[0],
        "label1_avg_iou": metrics.region_average_ious[1],
        "label0_avg_ari": metrics.region_average_aris[0],
        "label1_avg_ari": metrics.region_average_aris[1],
        "label0_overlap_mask_count": metrics.region_overlap_counts[0],
        "label1_overlap_mask_count": metrics.region_overlap_counts[1],
        "predicted_iou_mean": float(np.mean(predicted_ious)) if predicted_ious else 0.0,
        "predicted_iou_median": float(np.median(predicted_ious)) if predicted_ious else 0.0,
        "stability_score_mean": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "stability_score_median": float(np.median(stability_scores)) if stability_scores else 0.0,
    }
    row.update(
        build_canonical_evaluation_fields(
            miou=canonical_partition.miou,
            ari=canonical_partition.ari,
            evaluation_view="official_binary_partition",
        )
    )
    metric_summary = (
        f"Eval mIoU={canonical_partition.miou:.3f} "
        f"Eval ARI={canonical_partition.ari:.3f} "
        f"Official mIoU={metrics.miou:.3f} "
        f"Official ARI={metrics.ari:.3f} "
        f"Masks={metrics.num_predicted_masks}"
    )
    return Sam2OfficialSampleResult(
        row=row,
        metric_summary=metric_summary,
        prediction_masks=tuple(prediction_masks),
        aggregated_prediction_a=pred_label_map_original == 1,
        aggregated_prediction_b=pred_label_map_inverted == 1,
        metrics=metrics,
    )


def build_sam2_official_variant_summary(
    variant: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    num_total_samples: int,
    kaust_root: Path,
) -> dict[str, Any]:
    """Aggregate one official TextureSAM SAM-2 baseline variant."""

    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in OFFICIAL_SAM2_SCALAR_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in OFFICIAL_SAM2_SCALAR_FIELDS}
    raw_weight = int(sum(int(row["num_counted_regions"]) for row in rows))
    if raw_weight > 0:
        mean_metrics["miou"] = float(
            sum(float(row["miou"]) * int(row["num_counted_regions"]) for row in rows) / raw_weight
        )
        mean_metrics["ari"] = float(
            sum(float(row["ari"]) * int(row["num_counted_regions"]) for row in rows) / raw_weight
        )
    else:
        mean_metrics["miou"] = 0.0
        mean_metrics["ari"] = 0.0

    agg_rows = [row for row in rows if int(row["num_predicted_masks"]) > 0]
    if agg_rows:
        mean_metrics["miou_agg"] = float(mean(float(row["miou_agg"]) for row in agg_rows))
        mean_metrics["miou_agg_original"] = float(mean(float(row["miou_agg_original"]) for row in agg_rows))
        mean_metrics["miou_agg_inverted"] = float(mean(float(row["miou_agg_inverted"]) for row in agg_rows))
    else:
        mean_metrics["miou_agg"] = 0.0
        mean_metrics["miou_agg_original"] = 0.0
        mean_metrics["miou_agg_inverted"] = 0.0

    return {
        "dataset_format": "official_kaust256",
        "dataset_root": str(kaust_root),
        "variant": variant,
        "model_id": model_id,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": "official_binary_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[CANONICAL_PRIMARY_METRIC],
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[CANONICAL_SECONDARY_METRIC],
        "primary_metric_reason": (
            "This repository now defaults to one ArchiTexture-style binary evaluator. "
            "The released TextureSAM raw/aggregated metrics are retained as official "
            "diagnostics, but the default comparison view is the canonical binary partition."
        ),
        "num_total_samples": num_total_samples,
        "num_evaluated_samples": len(rows),
        "num_failed_samples": len(failures),
        "num_raw_scored_regions": raw_weight,
        "num_samples_with_predictions": len(agg_rows),
        "num_samples_without_predictions": len(rows) - len(agg_rows),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }


def build_sam2_official_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for one official TextureSAM variant."""

    lines = [
        "# TextureSAM Official RWTD SAM-2 Summary",
        "",
        f"- Dataset format: `{summary['dataset_format']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Evaluated samples: `{summary['num_evaluated_samples']}` / `{summary['num_total_samples']}`",
        f"- Failed samples: `{summary['num_failed_samples']}`",
        f"- Default comparison view: `{summary['primary_metric_name']}` / "
        f"`{summary['secondary_metric_name']}` = "
        f"`{summary['primary_metric_value']:.6f}` / "
        f"`{summary['secondary_metric_value']:.6f}`",
        f"- Raw-script scored regions: `{summary['num_raw_scored_regions']}`",
        f"- Aggregated-script scored samples: `{summary['num_samples_with_predictions']}`",
        f"- Samples skipped by the aggregated script because no masks were emitted: "
        f"`{summary['num_samples_without_predictions']}`",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Mean | Median |",
        "| --- | ---: | ---: |",
    ]
    for metric_name in OFFICIAL_SAM2_SCALAR_FIELDS:
        lines.append(
            f"| `{metric_name}` | {summary['mean_metrics'][metric_name]:.6f} | "
            f"{summary['median_metrics'][metric_name]:.6f} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['image_id']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def build_sam2_official_run_config(args, overview) -> dict[str, Any]:
    """Build a stable config payload for the official TextureSAM RWTD run."""

    config = {
        "dataset_format": "official_kaust256",
        "rwtd_root": str(Path(args.rwtd_root).expanduser()),
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
        "evaluation_view": "official_binary_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
    }
    if overview is not None:
        config["discovered_split_size"] = overview.num_examples
        config["discovered_image_dir"] = str(overview.image_dir)
        config["discovered_label_dir"] = str(overview.label_dir)
    return config


def build_sam2_official_experiment_terms_markdown(args) -> str:
    """Render a short glossary for the official TextureSAM RWTD run outputs."""

    variants = ", ".join(f"`{name}`" for name in resolve_sam2_variants(args.variant))
    return (
        "# TextureSAM Official RWTD Definitions\n\n"
        f"- Dataset format: `official_kaust256`\n"
        f"- Dataset root: `{Path(args.rwtd_root).expanduser()}`\n"
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
        f"- `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}`: the repo-wide default ArchiTexture-style evaluator. In this path the matched-mask rasterization is converted into one canonical binary partition before scoring.\n"
        "- `miou`: mirrors `eval_no_agg_masks.py` from the released TextureSAM repo, averaging IoU over all raw predicted masks that overlap each GT label.\n"
        "- `ari`: mirrors `eval_no_agg_masks.py`, averaging binary-mask ARI over all raw predicted masks that overlap each GT label.\n"
        "- `miou_agg`: mirrors `eval_agg_masks.py`, including its pinned `torchmetrics==1.6.1` call pattern, assigning each raw mask to its best-overlap GT label, rasterizing only the positive class, and averaging the original plus inverted binary-label scores.\n"
        "- `miou_agg_original` / `miou_agg_inverted`: the two binary-label orientations averaged into `miou_agg`.\n"
        "- Dataset summaries follow the released script denominators exactly: raw `miou` / `ari` are weighted by the number of GT labels that actually received overlapping masks, while aggregated means skip samples where the predictor emitted zero masks.\n"
        "- The released TextureSAM raw `miou` / `ari` and `miou_agg` remain in the artifacts for auditability, but the repo-default comparison view is the shared canonical evaluator.\n"
        "- Saved visuals show raw SAM proposals plus the two positive-class rasterizations used by the original and inverted aggregated passes.\n"
    )


def build_sam2_official_visual_record(
    sample: Kaust256Sample,
    evaluation: Sam2OfficialSampleResult,
    variant: str,
    visual_path: Path,
) -> dict[str, Any]:
    """Build one machine-readable visual metadata record for the official RWTD run."""

    visual_sample = make_visual_sample(sample)
    record = dict(evaluation.row)
    record.update(
        {
            "visual_path": str(visual_path),
            "caption": build_visual_caption(
                sample=visual_sample,
                protocol=f"{variant}:official",
                metric_summary=evaluation.metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=visual_sample,
                protocol=f"{variant}:official",
                metric_summary=evaluation.metric_summary,
            ),
        }
    )
    return record


def resolve_sam2_variants(variant: str) -> tuple[str, ...]:
    if variant == "both":
        return ("sam2", "sam2_star")
    return (variant,)


def resolve_sam2_official_output_dir(output_dir: str | None, variant: str) -> Path:
    """Resolve the output directory for an official TextureSAM SAM-2 run."""

    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"sam2-official-{variant}-{timestamp}"
    return Path("outputs") / "rwtd_sam2_official" / run_name


def make_visual_sample(sample: Kaust256Sample):
    """Create a duck-typed visualization sample compatible with the existing panel renderer."""

    label0, label1 = sample.label_values
    mask_a = sample.label_mask == label0
    mask_b = sample.label_mask == label1
    boundary_mask = boundary_from_region_masks(mask_a, mask_b)
    return SimpleNamespace(
        crop_name=sample.image_id,
        split="official_kaust256",
        image=sample.image,
        boundary_mask=boundary_mask,
        texture_a_mask=mask_a,
        texture_b_mask=mask_b,
        texture_a=f"label={label0}",
        texture_b=f"label={label1}",
        original_texture_a=str(label0),
        original_texture_b=str(label1),
        oracle_points_a=(),
        oracle_points_b=(),
    )
