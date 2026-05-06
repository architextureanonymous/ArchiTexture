"""Prompt-conditioned SAM 3 inspection and evaluation entrypoints.

This module owns the RWTD dataset inspection path plus the prompt-conditioned
SAM 3 ``predict-one`` and ``eval`` commands. It wires together decoded RWTD
samples, the ``Sam3Runner`` backend wrapper, metric computation, visualization
generation, optional WandB logging, and the repository's run-artifact writers.

Primary entrypoints:
- ``inspect_dataset()``: inspect one RWTD split and preview a decoded sample.
- ``run_predict_one()``: evaluate one sample for ``text``, ``oracle_points``,
  or ``both`` and write single-sample artifacts.
- ``run_evaluation()``: evaluate a split and write CSV, JSON, Markdown, and
  visualization outputs.

Inputs are decoded ``RwtdSample`` objects, CLI settings, and optional model
credentials. Outputs are run directories under ``outputs/rwtd_sam3/`` containing
``config.json``, ``experiment_terms.md``, per-sample metrics, summaries, and
optional preview images. Missing optional dependencies, invalid prompt data, and
prediction/target shape mismatches are surfaced as explicit errors.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from tqdm import tqdm

from rwtd_sam3.data.rwtd import (
    DatasetSplitOverview,
    DEFAULT_DATASET_ID,
    DEFAULT_EVAL_SPLIT,
    RwtdDecodedDataset,
    RwtdSample,
    get_rwtd_sample,
    iter_rwtd_samples,
    load_split_overview,
)
from rwtd_sam3.eval.metrics import (
    BinaryMetrics,
    average_binary_metrics,
    boundary_from_region_masks,
    build_canonical_evaluation_fields,
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    compute_binary_metrics,
    compute_boundary_metrics,
    compute_partition_metrics,
)
from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID, PredictionMetadata, Sam3Runner
from rwtd_sam3.utils.visualization import (
    build_visual_caption,
    build_visual_footer_lines,
    render_prediction_panel,
    save_prediction_panel,
)


LOGGER = logging.getLogger(__name__)
HARDWARE_COMPATIBILITY_STANDARD = "eval_throughput_v1"
HARDWARE_COMPATIBILITY_DESCRIPTION = (
    "Evaluation entrypoints must default to a worker-backed dataloader with pinned host memory, "
    "must persist resolved throughput settings into run artifacts, and must avoid sample-by-sample "
    "GPU starvation when a batched model path exists."
)
PROMPT_EVAL_DEFAULT_BATCH_SIZE = 8
PROMPT_EVAL_DEFAULT_PREFETCH_FACTOR = 2
SCALAR_METRIC_FIELDS = (
    "eval_miou",
    "eval_ari",
    "texture_a_iou",
    "texture_a_dice",
    "texture_a_precision",
    "texture_a_recall",
    "texture_b_iou",
    "texture_b_dice",
    "texture_b_precision",
    "texture_b_recall",
    "sample_macro_iou",
    "sample_miou",
    "sample_macro_dice",
    "sample_macro_precision",
    "sample_macro_recall",
    "sample_ari",
    "boundary_iou",
    "boundary_dice",
    "boundary_precision",
    "boundary_recall",
)


@dataclass(frozen=True)
class SampleEvaluationResult:
    """Per-sample evaluation payload saved to CSV and JSON outputs."""

    row: dict[str, Any]
    metric_summary: str
    prediction_a: np.ndarray
    prediction_b: np.ndarray


@dataclass(frozen=True)
class DatasetPartitionSelection:
    """Resolved deterministic partition bounds for one dataset view."""

    raw_spec: str
    partition_index: int
    num_partitions: int
    start_index: int
    end_index: int
    partition_size: int
    total_size: int


class WandbSession:
    """Small wrapper around optional Weights & Biases logging."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        self.enabled = enabled
        self._run = None
        self._wandb = None
        if not enabled:
            return

        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - exercised at runtime
            raise RuntimeError(
                "WandB logging was requested, but the 'wandb' package is not installed."
            ) from exc

        try:
            self._run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        except Exception as exc:  # pragma: no cover - exercised at runtime
            raise RuntimeError(
                "WandB logging was requested, but initialization failed. "
                "Authenticate with Weights & Biases before enabling --wandb."
            ) from exc
        self._wandb = wandb

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._wandb.log(payload, step=step)

    def log_preview(self, image, caption: str, step: int | None = None) -> None:
        if not self.enabled:
            return
        self.log({"preview": self._wandb.Image(image, caption=caption)}, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def collate_rwtd_samples(batch: list[RwtdSample]) -> list[RwtdSample]:
    """Return decoded RWTD samples unchanged for eval-time batching."""

    return list(batch)


_DATASET_PARTITION_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_dataset_partition_spec(spec: str | None) -> tuple[int, int] | None:
    """Parse a deterministic partition spec of the form ``K/N``.

    ``K`` is 1-based and must satisfy ``1 <= K <= N``.
    """

    if spec is None:
        return None
    match = _DATASET_PARTITION_RE.fullmatch(str(spec))
    if match is None:
        raise ValueError("Dataset partition must use the form 'K/N', for example '1/10'.")
    partition_index = int(match.group(1))
    num_partitions = int(match.group(2))
    if num_partitions < 1:
        raise ValueError("Dataset partition denominator must be at least 1.")
    if partition_index < 1 or partition_index > num_partitions:
        raise ValueError("Dataset partition index must satisfy 1 <= K <= N.")
    return partition_index, num_partitions


def resolve_dataset_partition(
    total_size: int,
    dataset_partition: str | None,
) -> DatasetPartitionSelection | None:
    """Resolve deterministic slice bounds for ``dataset_partition`` over ``total_size`` items."""

    parsed = parse_dataset_partition_spec(dataset_partition)
    if parsed is None:
        return None
    partition_index, num_partitions = parsed
    start_index = (int(total_size) * (partition_index - 1)) // num_partitions
    end_index = (int(total_size) * partition_index) // num_partitions
    return DatasetPartitionSelection(
        raw_spec=str(dataset_partition),
        partition_index=partition_index,
        num_partitions=num_partitions,
        start_index=start_index,
        end_index=end_index,
        partition_size=end_index - start_index,
        total_size=int(total_size),
    )


def resolve_eval_sample_count(
    limit: int | None,
    discovered_split_size: int,
    dataset_partition: DatasetPartitionSelection | None = None,
) -> int:
    """Resolve the number of samples an eval run will process."""

    available_size = dataset_partition.partition_size if dataset_partition is not None else int(discovered_split_size)
    if limit is None:
        return int(available_size)
    return min(int(limit), int(available_size))


def append_dataset_partition_fields(
    payload: dict[str, Any],
    dataset_partition: DatasetPartitionSelection | None,
    *,
    selected_sample_count: int | None = None,
) -> None:
    """Persist dataset-partition metadata into a config or summary payload."""

    if dataset_partition is None:
        payload["dataset_partition"] = None
        return
    payload["dataset_partition"] = dataset_partition.raw_spec
    payload["dataset_partition_index"] = dataset_partition.partition_index
    payload["dataset_partition_count"] = dataset_partition.num_partitions
    payload["dataset_partition_start_index"] = dataset_partition.start_index
    payload["dataset_partition_end_index"] = dataset_partition.end_index
    payload["dataset_partition_size"] = dataset_partition.partition_size
    payload["dataset_partition_total_size"] = dataset_partition.total_size
    if selected_sample_count is not None:
        payload["dataset_partition_selected_sample_count"] = int(selected_sample_count)


def build_rwtd_eval_loader(args, num_total_samples: int, *, start_index: int = 0):
    """Create a worker-backed RWTD dataloader for the prompt-conditioned eval path."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(
            "RWTD evaluation requires PyTorch to build the evaluation dataloader."
        ) from exc

    dataset = RwtdDecodedDataset(
        split=args.split,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
        limit=args.limit,
        length=num_total_samples,
        start_index=start_index,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_rwtd_samples,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_kwargs)


def inspect_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_EVAL_SPLIT,
    cache_dir: str | None = None,
    limit: int | None = 1,
) -> dict[str, Any]:
    """Inspect RWTD dataset metadata and a small preview sample."""

    overview = load_split_overview(split=split, dataset_id=dataset_id, cache_dir=cache_dir)
    preview = next(iter_rwtd_samples(split=split, dataset_id=dataset_id, cache_dir=cache_dir, limit=1), None)

    result = {
        "dataset_id": overview.dataset_id,
        "split": overview.split,
        "num_examples": overview.num_examples,
        "features": list(overview.features),
        "split_sizes": overview.split_sizes,
    }
    if preview is not None:
        result["preview"] = {
            "crop_name": preview.crop_name,
            "source_split": preview.split,
            "image_size": [preview.width, preview.height],
            "texture_a": preview.texture_a,
            "texture_b": preview.texture_b,
            "oracle_points_a": list(preview.oracle_points_a[: min(limit or 1, 3)]),
            "oracle_points_b": list(preview.oracle_points_b[: min(limit or 1, 3)]),
        }
    return result


def run_predict_one(args) -> dict[str, Any]:
    """Evaluate one sample and save prediction artifacts for the selected protocol(s)."""

    sample = get_rwtd_sample(
        split=args.split,
        index=args.index,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
    )
    output_dir = resolve_output_dir(args.output_dir, protocol=args.protocol, split=args.split)
    runner = Sam3Runner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )

    config_payload = build_run_config(args=args, overview=None, protocol=args.protocol)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_experiment_terms_markdown(args))

    results: dict[str, Any] = {}
    for protocol in resolve_protocols(args.protocol):
        protocol_dir = output_dir if args.protocol != "both" else output_dir / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        evaluation = evaluate_sample(
            sample=sample,
            protocol=protocol,
            runner=runner,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            boundary_tolerance_px=args.boundary_tolerance_px,
        )
        prediction_path = protocol_dir / "prediction.json"
        write_json(prediction_path, evaluation.row)
        panel_path = None
        if args.save_visuals:
            panel_path = save_prediction_panel(
                protocol_dir / "prediction.png",
                sample=sample,
                prediction_a=evaluation.prediction_a,
                prediction_b=evaluation.prediction_b,
                protocol=protocol,
                metric_summary=evaluation.metric_summary,
            )
            write_jsonl(
                protocol_dir / "visuals_manifest.jsonl",
                [
                    build_visual_record(
                        sample=sample,
                        evaluation=evaluation,
                        protocol=protocol,
                        visual_path=Path("prediction.png"),
                    )
                ],
            )
        results[protocol] = {
            "prediction_json": str(prediction_path),
            "visualization": str(panel_path) if panel_path is not None else None,
            "metrics": evaluation.row,
        }
    return results


def run_evaluation(args) -> dict[str, Any]:
    """Run RWTD evaluation and write reproducible outputs to disk."""

    overview = load_split_overview(split=args.split, dataset_id=args.dataset_id, cache_dir=args.cache_dir)
    num_total_samples = resolve_eval_sample_count(args.limit, overview.num_examples)
    if num_total_samples < 1:
        raise RuntimeError(f"Split '{args.split}' did not yield any samples to evaluate.")

    output_dir = resolve_output_dir(args.output_dir, protocol=args.protocol, split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or output_dir.name
    config_payload = build_run_config(args=args, overview=overview, protocol=args.protocol)
    write_json(output_dir / "config.json", config_payload)
    write_text(output_dir / "experiment_terms.md", build_experiment_terms_markdown(args))

    wandb_session = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=run_name,
        config=config_payload,
    )

    runner = Sam3Runner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )
    LOGGER.info(
        "RWTD eval loader configured with batch_size=%d, num_workers=%d, pin_memory=true, prefetch_factor=%d.",
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
    )

    all_protocol_summaries: dict[str, Any] = {}
    try:
        for protocol in resolve_protocols(args.protocol):
            eval_loader = build_rwtd_eval_loader(args=args, num_total_samples=num_total_samples)
            protocol_dir = output_dir if args.protocol != "both" else output_dir / protocol
            protocol_dir.mkdir(parents=True, exist_ok=True)
            (protocol_dir / "visuals").mkdir(parents=True, exist_ok=True)
            write_text(protocol_dir / "experiment_terms.md", build_experiment_terms_markdown(args))

            rows: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            empty_texture_predictions = 0
            visual_records: list[dict[str, Any]] = []
            logged_samples = 0

            progress = tqdm(total=num_total_samples, desc=f"eval:{protocol}", unit="sample")
            for batch_samples in eval_loader:
                evaluations: list[SampleEvaluationResult] = []
                if protocol == "text":
                    try:
                        evaluations = evaluate_text_batch(
                            samples=batch_samples,
                            runner=runner,
                            protocol=protocol,
                            score_threshold=args.score_threshold,
                            mask_threshold=args.mask_threshold,
                            boundary_tolerance_px=args.boundary_tolerance_px,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "Text batch with %d samples failed; retrying sample-by-sample: %s",
                            len(batch_samples),
                            exc,
                        )
                for batch_index, sample in enumerate(batch_samples):
                    if protocol == "text" and evaluations:
                        evaluation = evaluations[batch_index]
                    else:
                        try:
                            evaluation = evaluate_sample(
                                sample=sample,
                                protocol=protocol,
                                runner=runner,
                                score_threshold=args.score_threshold,
                                mask_threshold=args.mask_threshold,
                                boundary_tolerance_px=args.boundary_tolerance_px,
                            )
                        except Exception as exc:
                            failure = {"crop_name": sample.crop_name, "error": str(exc)}
                            if args.failure_policy == "skip":
                                failures.append(failure)
                                LOGGER.error("Skipping sample %s: %s", sample.crop_name, exc)
                                progress.update(1)
                                continue
                            raise

                    rows.append(evaluation.row)
                    empty_texture_predictions += evaluation.row["empty_texture_predictions"]
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "crop": sample.crop_name,
                            "macro_iou": f"{evaluation.row['sample_macro_iou']:.3f}",
                        }
                    )

                    if args.save_visuals:
                        visual_path = Path("visuals") / f"{sample.crop_name}.png"
                        save_prediction_panel(
                            protocol_dir / visual_path,
                            sample=sample,
                            prediction_a=evaluation.prediction_a,
                            prediction_b=evaluation.prediction_b,
                            protocol=protocol,
                            metric_summary=evaluation.metric_summary,
                        )
                        visual_records.append(
                            build_visual_record(
                                sample=sample,
                                evaluation=evaluation,
                                protocol=protocol,
                                visual_path=visual_path,
                            )
                        )

                    if args.wandb:
                        scalar_log = {
                            "protocol": protocol,
                            "sample_macro_iou": evaluation.row["sample_macro_iou"],
                            "sample_macro_dice": evaluation.row["sample_macro_dice"],
                            "boundary_dice": evaluation.row["boundary_dice"],
                        }
                        wandb_session.log(scalar_log, step=logged_samples)
                        if logged_samples % args.log_every == 0:
                            preview_image = render_prediction_panel(
                                sample=sample,
                                prediction_a=evaluation.prediction_a,
                                prediction_b=evaluation.prediction_b,
                                protocol=protocol,
                                metric_summary=evaluation.metric_summary,
                            )
                            caption = build_visual_caption(
                                sample=sample,
                                protocol=protocol,
                                metric_summary=evaluation.metric_summary,
                            )
                            wandb_session.log_preview(preview_image, caption=caption, step=logged_samples)
                    logged_samples += 1
            progress.close()

            if not rows:
                raise RuntimeError(
                    f"Protocol '{protocol}' produced no successful evaluations. "
                    "Check the recorded failures for details."
                )

            write_csv(protocol_dir / "per_sample_metrics.csv", rows)
            if args.save_visuals:
                write_jsonl(protocol_dir / "visuals_manifest.jsonl", visual_records)
            summary = build_protocol_summary(
                protocol=protocol,
                split=args.split,
                dataset_id=args.dataset_id,
                model_id=args.model_id,
                rows=rows,
                failures=failures,
                empty_texture_predictions=empty_texture_predictions,
                num_total_samples=num_total_samples,
            )
            write_json(protocol_dir / "summary.json", summary)
            write_text(protocol_dir / "summary.md", build_markdown_summary(summary))
            all_protocol_summaries[protocol] = summary

        if args.protocol == "both":
            write_json(output_dir / "summary.json", {"protocols": all_protocol_summaries})
            write_text(
                output_dir / "summary.md",
                "# RWTD SAM 3 Evaluation\n\nBoth protocols completed successfully.\n",
            )
    finally:
        wandb_session.finish()

    return all_protocol_summaries


def evaluate_sample(
    sample: RwtdSample,
    protocol: str,
    runner: Sam3Runner,
    score_threshold: float,
    mask_threshold: float,
    boundary_tolerance_px: int,
    ) -> SampleEvaluationResult:
    """Run one protocol on one RWTD sample and return flattened metrics."""

    if protocol == "text":
        prediction_a, meta_a = runner.segment_text(
            image=sample.image,
            prompt=sample.texture_a,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )
        prediction_b, meta_b = runner.segment_text(
            image=sample.image,
            prompt=sample.texture_b,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )
    elif protocol == "oracle_points":
        if not sample.oracle_points_a:
            raise ValueError(f"Sample '{sample.crop_name}' has no oracle_points_a values.")
        if not sample.oracle_points_b:
            raise ValueError(f"Sample '{sample.crop_name}' has no oracle_points_b values.")
        prediction_a, meta_a = runner.segment_oracle_points(
            image=sample.image,
            points=sample.oracle_points_a,
            score_threshold=score_threshold,
        )
        prediction_b, meta_b = runner.segment_oracle_points(
            image=sample.image,
            points=sample.oracle_points_b,
            score_threshold=score_threshold,
        )
    else:
        raise ValueError(f"Unsupported protocol '{protocol}'.")

    return build_sample_evaluation_result(
        sample=sample,
        protocol=protocol,
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        meta_a=meta_a,
        meta_b=meta_b,
        boundary_tolerance_px=boundary_tolerance_px,
    )


def evaluate_text_batch(
    samples: list[RwtdSample],
    protocol: str,
    runner: Sam3Runner,
    score_threshold: float,
    mask_threshold: float,
    boundary_tolerance_px: int,
) -> list[SampleEvaluationResult]:
    """Run batched text prompting for one RWTD batch and compute per-sample metrics."""

    predictions_a = runner.segment_text_batch(
        images=[sample.image for sample in samples],
        prompts=[sample.texture_a for sample in samples],
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )
    predictions_b = runner.segment_text_batch(
        images=[sample.image for sample in samples],
        prompts=[sample.texture_b for sample in samples],
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )
    results: list[SampleEvaluationResult] = []
    for sample, prediction_a_result, prediction_b_result in zip(samples, predictions_a, predictions_b, strict=True):
        prediction_a, meta_a = prediction_a_result
        prediction_b, meta_b = prediction_b_result
        results.append(
            build_sample_evaluation_result(
                sample=sample,
                protocol=protocol,
                prediction_a=prediction_a,
                prediction_b=prediction_b,
                meta_a=meta_a,
                meta_b=meta_b,
                boundary_tolerance_px=boundary_tolerance_px,
            )
        )
    return results


def build_sample_evaluation_result(
    sample: RwtdSample,
    protocol: str,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    meta_a: PredictionMetadata,
    meta_b: PredictionMetadata,
    boundary_tolerance_px: int,
) -> SampleEvaluationResult:
    """Flatten one pair of predictions into the repository's per-sample metric row."""

    validate_prediction_shape(prediction_a, sample.texture_a_mask, sample.crop_name, "texture_a")
    validate_prediction_shape(prediction_b, sample.texture_b_mask, sample.crop_name, "texture_b")

    texture_a_metrics = compute_binary_metrics(prediction_a, sample.texture_a_mask)
    texture_b_metrics = compute_binary_metrics(prediction_b, sample.texture_b_mask)
    sample_macro = average_binary_metrics([texture_a_metrics, texture_b_metrics])
    sample_partition = compute_partition_metrics(
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        target_a=sample.texture_a_mask,
        target_b=sample.texture_b_mask,
    )
    boundary_prediction = boundary_from_region_masks(prediction_a, prediction_b)
    boundary_metrics = compute_boundary_metrics(
        prediction=boundary_prediction,
        target=sample.boundary_mask,
        tolerance_px=boundary_tolerance_px,
    )
    overlap_pixels = int(np.logical_and(prediction_a, prediction_b).sum())

    row = {
        "protocol": protocol,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "texture_a_prompt": sample.texture_a,
        "texture_b_prompt": sample.texture_b,
        "texture_a_backend": meta_a.backend,
        "texture_b_backend": meta_b.backend,
        "texture_a_num_instances": meta_a.num_instances,
        "texture_b_num_instances": meta_b.num_instances,
        "texture_a_scores": ",".join(f"{score:.6f}" for score in meta_a.kept_scores),
        "texture_b_scores": ",".join(f"{score:.6f}" for score in meta_b.kept_scores),
        "pred_overlap_pixels": overlap_pixels,
        "empty_texture_predictions": int(not prediction_a.any()) + int(not prediction_b.any()),
    }
    row.update(texture_a_metrics.to_prefixed_dict("texture_a"))
    row.update(texture_b_metrics.to_prefixed_dict("texture_b"))
    row.update(sample_macro.to_prefixed_dict("sample_macro"))
    row.update(sample_partition.to_prefixed_dict("sample"))
    row.update(
        build_canonical_evaluation_fields(
            miou=sample_partition.miou,
            ari=sample_partition.ari,
            evaluation_view="direct_binary_partition",
        )
    )
    row.update(boundary_metrics.to_prefixed_dict("boundary"))

    metric_summary = (
        f"A IoU={texture_a_metrics.iou:.3f} "
        f"B IoU={texture_b_metrics.iou:.3f} "
        f"mIoU={sample_partition.miou:.3f} "
        f"ARI={sample_partition.ari:.3f} "
        f"Boundary Dice={boundary_metrics.dice:.3f}"
    )
    return SampleEvaluationResult(
        row=row,
        metric_summary=metric_summary,
        prediction_a=prediction_a,
        prediction_b=prediction_b,
    )


def build_protocol_summary(
    protocol: str,
    split: str,
    dataset_id: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    empty_texture_predictions: int,
    num_total_samples: int,
) -> dict[str, Any]:
    """Aggregate one protocol's per-sample rows into JSON/Markdown summaries."""

    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in SCALAR_METRIC_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in SCALAR_METRIC_FIELDS}
    return {
        "dataset_id": dataset_id,
        "split": split,
        "protocol": protocol,
        "model_id": model_id,
        "hardware_compatibility_standard": HARDWARE_COMPATIBILITY_STANDARD,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": "direct_binary_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[CANONICAL_PRIMARY_METRIC],
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[CANONICAL_SECONDARY_METRIC],
        "num_total_samples": num_total_samples,
        "num_evaluated_samples": len(rows),
        "num_failed_samples": len(failures),
        "empty_texture_predictions": empty_texture_predictions,
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }


def build_run_config(
    args,
    overview: DatasetSplitOverview | None,
    protocol: str,
) -> dict[str, Any]:
    """Build a stable config payload saved alongside every run."""

    versions = discovered_package_versions()
    config = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "protocol": protocol,
        "model_id": args.model_id,
        "device": args.device,
        "hardware_compatibility_standard": HARDWARE_COMPATIBILITY_STANDARD,
        "hardware_compatibility_description": HARDWARE_COMPATIBILITY_DESCRIPTION,
        "batch_size": getattr(args, "batch_size", None),
        "num_workers": getattr(args, "num_workers", None),
        "prefetch_factor": getattr(args, "prefetch_factor", None),
        "pin_memory": True if hasattr(args, "batch_size") else None,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "boundary_tolerance_px": args.boundary_tolerance_px,
        "limit": getattr(args, "limit", None),
        "failure_policy": getattr(args, "failure_policy", "abort"),
        "save_visuals": args.save_visuals,
        "wandb": getattr(args, "wandb", False),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": versions,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": "direct_binary_partition",
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
    }
    if overview is not None:
        config["discovered_split_size"] = overview.num_examples
        config["discovered_split_sizes"] = overview.split_sizes
        config["discovered_features"] = list(overview.features)
    return config


def discovered_package_versions() -> dict[str, str | None]:
    """Collect version metadata for key runtime dependencies."""

    packages = ("datasets", "numpy", "Pillow", "torch", "transformers", "wandb")
    results: dict[str, str | None] = {}
    for package_name in packages:
        try:
            results[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            results[package_name] = None
    return results


def resolve_protocols(protocol: str) -> tuple[str, ...]:
    if protocol == "both":
        return ("text", "oracle_points")
    return (protocol,)


def resolve_output_dir(output_dir: str | None, protocol: str, split: str) -> Path:
    """Resolve and create the output directory for a run."""

    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"sam3-rwtd-{protocol}-{split}-{timestamp}"
    return Path("outputs") / "rwtd_sam3" / run_name


def validate_prediction_shape(
    prediction: np.ndarray,
    target: np.ndarray,
    crop_name: str,
    label: str,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Sample '{crop_name}' {label} prediction has shape {prediction.shape}, "
            f"expected {target.shape}."
        )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def build_visual_record(
    sample: RwtdSample,
    evaluation: SampleEvaluationResult,
    protocol: str,
    visual_path: Path,
) -> dict[str, Any]:
    """Build one per-visual metadata record for website scraping and auditing."""

    record = dict(evaluation.row)
    record.update(
        {
            "visual_path": str(visual_path),
            "caption": build_visual_caption(
                sample=sample,
                protocol=protocol,
                metric_summary=evaluation.metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=sample,
                protocol=protocol,
                metric_summary=evaluation.metric_summary,
            ),
            "original_texture_a": sample.original_texture_a,
            "original_texture_b": sample.original_texture_b,
            "oracle_points_a_count": len(sample.oracle_points_a),
            "oracle_points_b_count": len(sample.oracle_points_b),
        }
    )
    return record


def build_experiment_terms_markdown(args) -> str:
    """Render a short glossary saved with every prediction or evaluation run."""

    resolved_protocols = ", ".join(f"`{name}`" for name in resolve_protocols(args.protocol))
    lines = [
        "# Experiment Definitions and Terms",
        "",
        f"- Dataset: `{args.dataset_id}`",
        f"- Split: `{args.split}`",
        f"- Requested protocol setting: `{args.protocol}`",
        f"- Executed protocol(s): {resolved_protocols}",
        f"- Model: `{args.model_id}`",
        f"- Hardware compatibility standard: `{HARDWARE_COMPATIBILITY_STANDARD}`",
        f"- Eval dataloader: `batch_size={getattr(args, 'batch_size', 'n/a')}`, `num_workers={getattr(args, 'num_workers', 'n/a')}`, `pin_memory=true`, `prefetch_factor={getattr(args, 'prefetch_factor', 'n/a')}`",
        "",
        "## Core Terms",
        "",
        "- RWTD crop: one image crop with two annotated texture regions and one boundary mask.",
        "- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.",
        "- `texture_a` / `texture_b`: natural-language texture descriptions stored in RWTD.",
        "- `original_texture_a` / `original_texture_b`: shorter source labels stored in RWTD.",
        "- `oracle_points_a` / `oracle_points_b`: positive point prompts in pixel coordinates.",
        "- `text` protocol: predicts texture A from `texture_a` text and texture B from `texture_b` text.",
        "- `oracle_points` protocol: predicts texture A from `oracle_points_a` and texture B from `oracle_points_b`.",
        "- `both`: runs both protocols and writes separate protocol subdirectories.",
        f"- `{HARDWARE_COMPATIBILITY_STANDARD}`: prompt-conditioned eval throughput standard. The `eval` path defaults to `batch_size={PROMPT_EVAL_DEFAULT_BATCH_SIZE}`, `pin_memory=true`, `prefetch_factor={PROMPT_EVAL_DEFAULT_PREFETCH_FACTOR}`, and `num_workers=min(8, os.cpu_count())` so CUDA inference is not starved by host-side decoding.",
        "- `sample_macro_*`: arithmetic mean of the texture-A and texture-B metric for one sample.",
        "- `sample_miou`: explicit mean IoU over texture A and texture B for one sample.",
        "- `sample_ari`: Adjusted Rand Index over the per-pixel partition labels `{background, A-only, B-only, overlap}`.",
        f"- `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}`: the repo-wide default ArchiTexture-style binary evaluator. In this prompt-conditioned path they are identical to `sample_miou` / `sample_ari` because the model already emits one texture-A mask and one texture-B mask directly.",
        "- `boundary_*`: metrics computed on a derived boundary mask from the predicted A/B regions.",
        f"- `boundary_tolerance_px`: boundary matching radius in pixels, set to `{args.boundary_tolerance_px}` for this run.",
        "- `empty_texture_predictions`: number of empty texture masks for a sample or aggregated run.",
        "",
        "## Visual Artifacts",
        "",
        "- `visuals/*.png`: input, ground-truth, and prediction panels with wrapped footer captions.",
        "- `visuals_manifest.jsonl`: one JSON record per saved visual with the image path, full caption text, prompts, labels, oracle-point counts, and per-sample metrics.",
        "",
        "## Prompt Semantics",
        "",
        "- SAM 3 is always conditioned on a prompt in this repository.",
        "- `text` uses RWTD texture descriptions as prompts.",
        "- `oracle_points` uses RWTD positive oracle points as prompts.",
    ]
    return "\n".join(lines) + "\n"


def build_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for one protocol."""

    mean_metrics = summary["mean_metrics"]
    median_metrics = summary["median_metrics"]
    lines = [
        "# RWTD SAM 3 Evaluation Summary",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Split: `{summary['split']}`",
        f"- Protocol: `{summary['protocol']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Hardware compatibility standard: `{summary['hardware_compatibility_standard']}`",
        f"- Evaluated samples: `{summary['num_evaluated_samples']}` / `{summary['num_total_samples']}`",
        f"- Failed samples: `{summary['num_failed_samples']}`",
        f"- Empty texture predictions: `{summary['empty_texture_predictions']}`",
        f"- Default comparison view: `{summary['primary_metric_name']}` / `{summary['secondary_metric_name']}` = "
        f"`{summary['primary_metric_value']:.6f}` / `{summary['secondary_metric_value']:.6f}`",
        "",
        "## Mean Metrics",
        "",
        "| Metric | Mean | Median |",
        "| --- | ---: | ---: |",
    ]
    for metric_name in SCALAR_METRIC_FIELDS:
        lines.append(
            f"| `{metric_name}` | {mean_metrics[metric_name]:.6f} | {median_metrics[metric_name]:.6f} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['crop_name']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def set_global_seed(seed: int) -> None:
    """Set reproducible seeds for Python and NumPy."""

    random.seed(seed)
    np.random.seed(seed)
