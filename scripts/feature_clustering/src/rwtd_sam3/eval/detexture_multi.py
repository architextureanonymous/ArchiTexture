"""Multi-region DeTexture ADE20K evaluation for prompt-free pooled coarsest partitions.

This module adds a scientifically minimal multi-region extension of the current
proposal-free coarse-only SAM-feature method. It keeps the feature extraction,
pooling, normalization, and upsampling path fixed and changes only the partition
head from binary to deterministic K-way clustering.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from rwtd_sam3.data.detexture_multi import (
    DETEXTURE_MULTI_DATASET_ID,
    DETEXTURE_MULTI_GT_DECODE_SETTINGS,
    DETEXTURE_MULTI_REGION_FILTER_SETTINGS,
    DeTextureMultiOverview,
    DeTextureMultiSample,
    get_detexture_multi_sample,
    iter_detexture_multi_samples,
    load_detexture_multi_overview,
)
from rwtd_sam3.eval.metrics import (
    MultiPartitionMetrics,
    compute_multi_partition_metrics,
)
from rwtd_sam3.eval.runner import (
    append_dataset_partition_fields,
    DatasetPartitionSelection,
    resolve_dataset_partition,
    resolve_eval_sample_count,
    WandbSession,
    discovered_package_versions,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from rwtd_sam3.models.sam3_feature_cluster_multiregion_runner import (
    DETEXTURE_MULTI_DEFAULT_VARIANT,
    DETEXTURE_MULTI_SUPPORTED_VARIANTS,
    FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS,
    FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS,
    FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS,
    FeatureClusterCoarseGlobalMultiPartition,
    MULTI_DEEPDPM_FLIPAVG_VARIANT,
    MULTI_DEEPDPM_VARIANT,
    MULTI_HDBSCAN_FLIPAVG_VARIANT,
    MULTI_HDBSCAN_VARIANT,
    MULTI_ORACLE_K_FLIPAVG_VARIANT,
    MULTI_ORACLE_K_VARIANT,
    MULTI_PREDICTED_K_VARIANT,
    build_detexture_multi_runner,
)
from rwtd_sam3.utils.visualization import save_detexture_multi_partition_panel


LOGGER = logging.getLogger(__name__)

DETEXTURE_MULTI_OUTPUT_ROOT = Path("outputs") / "detexture_multi"
DETEXTURE_MULTI_EVALUATION_CONTRACT = "detexture_multi_partition_v1"
DETEXTURE_MULTI_PRIMARY_METRIC = "eval_miou"
DETEXTURE_MULTI_SECONDARY_METRIC = "eval_ari"
DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_STANDARD = "detexture_multi_streaming_eval_v1"
DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_DESCRIPTION = (
    "DeTexture multi evaluation must stream decoded image/labelmap samples instead of preloading the dataset, "
    "must resolve the detecture_data root once per run, and must persist the active sample-loading mode into run artifacts."
)
DETEXTURE_MULTI_VARIANT_SUMMARIES = {
    MULTI_ORACLE_K_VARIANT: (
        "Oracle-K prompt-free pooled coarsest clustering on frozen SAM features. GT supplies only the region count K; "
        "the clustering itself uses no prompts or GT geometry."
    ),
    MULTI_ORACLE_K_FLIPAVG_VARIANT: (
        "Oracle-K prompt-free pooled coarsest clustering on flip-averaged frozen SAM features. The clustering head "
        "is identical to oracle-K aside from the symmetric feature averaging."
    ),
    MULTI_PREDICTED_K_VARIANT: (
        "Predicted-K prompt-free pooled coarsest clustering on frozen SAM features using deterministic silhouette "
        "selection over a small K range."
    ),
    MULTI_DEEPDPM_VARIANT: (
        "A stronger nonparametric deep clustering ablation on frozen SAM coarse features. It keeps the frozen SAM "
        "coarsest pooled feature pipeline fixed and replaces the deterministic partition head with a trainable "
        "DeepDPM-style split/merge clustering module that infers K."
    ),
    MULTI_DEEPDPM_FLIPAVG_VARIANT: (
        "A stronger nonparametric deep clustering ablation on flip-averaged frozen SAM coarse features. It uses the "
        "same trainable DeepDPM-style split/merge partition head as the raw DeepDPM route, but averages identity, "
        "hflip, vflip, and hvflip coarsest features before pooling."
    ),
    MULTI_HDBSCAN_VARIANT: (
        "A simple HDBSCAN ablation on frozen SAM coarse features. It keeps the pooled coarsest frozen-feature pipeline "
        "fixed and replaces the deterministic partition head with sklearn HDBSCAN to infer K directly from the same pooled vectors."
    ),
    MULTI_HDBSCAN_FLIPAVG_VARIANT: (
        "A simple HDBSCAN ablation on flip-averaged frozen SAM coarse features. It uses the same sklearn HDBSCAN head as the raw "
        "HDBSCAN route, but averages identity, hflip, vflip, and hvflip coarsest features before pooling."
    ),
}
DETEXTURE_MULTI_VISUAL_CONTRACT = "input|gt_labelmap|pred_labelmap|pred_boundaries"
DETEXTURE_MULTI_SCALAR_FIELDS = (
    DETEXTURE_MULTI_PRIMARY_METRIC,
    DETEXTURE_MULTI_SECONDARY_METRIC,
    "eval_nmi",
    "count_accuracy",
    "coverage",
    "k_gt",
    "k_pred",
)


@dataclass(frozen=True)
class DeTextureMultiSampleResult:
    """Per-sample multi-region DeTexture payload."""

    row: dict[str, Any]
    metric_summary: str
    metrics: MultiPartitionMetrics
    partition: FeatureClusterCoarseGlobalMultiPartition


def _resolve_detexture_multi_runner_settings(args) -> dict[str, float | int | str] | None:
    """Resolve variant-specific runner overrides from the CLI surface."""

    overrides: dict[str, float | int | str] = {}
    if args.variant in {MULTI_DEEPDPM_VARIANT, MULTI_DEEPDPM_FLIPAVG_VARIANT}:
        overrides["deepdpm_seed"] = int(getattr(args, "seed", 0))
        for field_name in (
            "deepdpm_init_clusters",
            "deepdpm_min_clusters",
            "deepdpm_max_clusters",
            "deepdpm_hidden_dim",
            "deepdpm_embedding_dim",
            "deepdpm_outer_iterations",
            "deepdpm_inner_epochs",
            "deepdpm_learning_rate",
            "deepdpm_weight_decay",
            "deepdpm_split_dispersion_threshold",
            "deepdpm_merge_similarity_threshold",
        ):
            value = getattr(args, field_name, None)
            if value is not None:
                overrides[field_name] = value
    if args.variant in {MULTI_HDBSCAN_VARIANT, MULTI_HDBSCAN_FLIPAVG_VARIANT}:
        for field_name in (
            "hdbscan_min_cluster_size",
            "hdbscan_min_samples",
            "hdbscan_cluster_selection_epsilon",
        ):
            value = getattr(args, field_name, None)
            if value is not None:
                overrides[field_name] = value
    return overrides or None


def _detexture_multi_deepdpm_hyperparameter_overrides(args) -> dict[str, float | int | str]:
    """Return only the explicitly requested DeepDPM CLI overrides."""

    settings = _resolve_detexture_multi_runner_settings(args) or {}
    return {key: value for key, value in settings.items() if key.startswith("deepdpm_") and key != "deepdpm_seed"}


def _detexture_multi_hdbscan_hyperparameter_overrides(args) -> dict[str, float | int | str]:
    """Return only the explicitly requested HDBSCAN CLI overrides."""

    settings = _resolve_detexture_multi_runner_settings(args) or {}
    return {key: value for key, value in settings.items() if key.startswith("hdbscan_")}


def run_detexture_multi_predict_one(args) -> dict[str, Any]:
    """Run one multi-region DeTexture sample through the requested variant."""

    sample = get_detexture_multi_sample(dataset_root=args.dataset_root, index=args.index, split=args.split)
    output_dir = prepare_detexture_multi_output_dir(
        args.output_dir,
        variant=args.variant,
        run_kind="predict",
        sample_index=args.index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = load_detexture_multi_overview(args.dataset_root, split=args.split)
    config_payload = build_detexture_multi_run_config(
        args,
        overview=overview,
        dataset_partition=None,
        selected_sample_count=1,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_detexture_multi_experiment_terms_markdown(
            args,
            dataset_partition=None,
            selected_sample_count=1,
        ),
    )

    runner = build_detexture_multi_runner(
        args.variant,
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
        settings=_resolve_detexture_multi_runner_settings(args),
    )
    evaluation = evaluate_detexture_multi_sample(sample=sample, variant=args.variant, runner=runner)
    write_json(output_dir / "prediction.json", evaluation.row)
    np.save(output_dir / "predicted_label_map.npy", evaluation.partition.predicted_label_map.astype(np.int32))
    np.save(output_dir / "gt_label_map.npy", sample.gt_label_map.astype(np.int32))
    np.save(output_dir / "valid_pixel_mask.npy", sample.valid_pixel_mask.astype(bool))
    panel_path = None
    if args.save_visuals:
        panel_path = save_detexture_multi_panel(
            output_path=output_dir / "prediction.png",
            sample=sample,
            evaluation=evaluation,
            variant=args.variant,
        )
        write_jsonl(
            output_dir / "visuals_manifest.jsonl",
            [
                build_detexture_multi_visual_record(
                    sample=sample,
                    evaluation=evaluation,
                    variant=args.variant,
                    visual_path=Path("prediction.png"),
                    label_map_path=Path("predicted_label_map.npy"),
                )
            ],
        )
    return {
        args.variant: {
            "prediction_json": str(output_dir / "prediction.json"),
            "predicted_label_map": str(output_dir / "predicted_label_map.npy"),
            "visualization": str(panel_path) if panel_path is not None else None,
            "metrics": evaluation.row,
        }
    }


def run_detexture_multi_evaluation(args) -> dict[str, Any]:
    """Evaluate one multi-region DeTexture variant on the local dataset."""

    overview = load_detexture_multi_overview(args.dataset_root, split=args.split)
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_total_samples = resolve_eval_sample_count(args.limit, overview.num_examples, dataset_partition)
    if num_total_samples < 1:
        raise RuntimeError("The DeTexture multi root did not yield any samples to evaluate.")

    output_dir = prepare_detexture_multi_output_dir(
        args.output_dir,
        variant=args.variant,
        run_kind="eval",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_detexture_multi_run_config(
        args,
        overview=overview,
        dataset_partition=dataset_partition,
        selected_sample_count=num_total_samples,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_detexture_multi_experiment_terms_markdown(
            args,
            dataset_partition=dataset_partition,
            selected_sample_count=num_total_samples,
        ),
    )

    wandb_session = WandbSession(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or output_dir.name,
        config=config_payload,
    )
    runner = build_detexture_multi_runner(
        args.variant,
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
        settings=_resolve_detexture_multi_runner_settings(args),
    )
    LOGGER.info(
        "DeTexture multi eval uses streamed sample decoding under hardware standard %s; limit=%s split=%s.",
        DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_STANDARD,
        args.limit,
        args.split,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    visual_records: list[dict[str, Any]] = []
    label_map_dir = output_dir / "label_maps"
    label_map_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress = tqdm(total=num_total_samples, desc="eval:detexture-multi", unit="sample")
        for step, sample in enumerate(
            iter_detexture_multi_samples(
                dataset_root=overview.dataset_root,
                split=overview.split,
                limit=num_total_samples,
                start_index=dataset_partition.start_index if dataset_partition is not None else 0,
            )
        ):
            try:
                evaluation = evaluate_detexture_multi_sample(sample=sample, variant=args.variant, runner=runner)
            except Exception as exc:
                failure = {"crop_name": sample.crop_name, "error": str(exc)}
                if args.failure_policy == "skip":
                    failures.append(failure)
                    LOGGER.error("Skipping sample %s: %s", sample.crop_name, exc)
                    progress.update(1)
                    continue
                raise

            rows.append(evaluation.row)
            progress.update(1)
            progress.set_postfix({"crop": sample.crop_name, "eval_miou": f"{evaluation.row['eval_miou']:.3f}"})

            pred_path = Path("label_maps") / f"{sample.crop_name}.pred.npy"
            gt_path = Path("label_maps") / f"{sample.crop_name}.gt.npy"
            valid_path = Path("label_maps") / f"{sample.crop_name}.valid.npy"
            np.save(output_dir / pred_path, evaluation.partition.predicted_label_map.astype(np.int32))
            np.save(output_dir / gt_path, sample.gt_label_map.astype(np.int32))
            np.save(output_dir / valid_path, sample.valid_pixel_mask.astype(bool))

            if args.save_visuals:
                visual_path = Path("visuals") / f"{sample.crop_name}.png"
                save_detexture_multi_panel(
                    output_path=output_dir / visual_path,
                    sample=sample,
                    evaluation=evaluation,
                    variant=args.variant,
                )
                visual_records.append(
                    build_detexture_multi_visual_record(
                        sample=sample,
                        evaluation=evaluation,
                        variant=args.variant,
                        visual_path=visual_path,
                        label_map_path=pred_path,
                    )
                )

            if args.wandb:
                wandb_session.log(
                    {
                        "dataset_id": DETEXTURE_MULTI_DATASET_ID,
                        "variant": args.variant,
                        "eval_miou": evaluation.row[DETEXTURE_MULTI_PRIMARY_METRIC],
                        "eval_ari": evaluation.row[DETEXTURE_MULTI_SECONDARY_METRIC],
                        "eval_nmi": evaluation.row["eval_nmi"],
                        "k_gt": evaluation.row["k_gt"],
                        "k_pred": evaluation.row["k_pred"],
                        "count_accuracy": evaluation.row["count_accuracy"],
                    },
                    step=step,
                )
                if step % args.log_every == 0 and args.save_visuals:
                    wandb_session.log_preview(
                        Image.open(output_dir / visual_records[-1]["visual_path"]).convert("RGB"),
                        caption=visual_records[-1]["caption"],
                        step=step,
                    )
        progress.close()

        if not rows:
            raise RuntimeError(
                "The DeTexture multi benchmark produced no successful evaluations. Check the recorded failures for details."
            )

        write_csv(output_dir / "per_sample_metrics.csv", rows)
        if args.save_visuals:
            write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
        summary = build_detexture_multi_summary(
            overview=overview,
            variant=args.variant,
            model_id=args.model_id,
            rows=rows,
            failures=failures,
            num_total_samples=num_total_samples,
            dataset_partition=dataset_partition,
        )
        write_json(output_dir / "summary.json", summary)
        write_text(output_dir / "summary.md", build_detexture_multi_markdown_summary(summary))
    finally:
        wandb_session.finish()

    return {args.variant: summary}


def evaluate_detexture_multi_sample(
    sample: DeTextureMultiSample,
    variant: str,
    runner,
) -> DeTextureMultiSampleResult:
    """Evaluate one prompt-free multi-region DeTexture sample."""

    requested_k = sample.oracle_num_regions if variant in {MULTI_ORACLE_K_VARIANT, MULTI_ORACLE_K_FLIPAVG_VARIANT} else None
    partition = runner.generate_multi_partition(sample.image, num_clusters=requested_k)
    metrics = compute_multi_partition_metrics(
        partition.predicted_label_map,
        sample.gt_label_map,
        valid_mask=sample.valid_pixel_mask,
    )
    clustering_training_diagnostics = partition.clustering_training_diagnostics or {}

    row = {
        "dataset_id": DETEXTURE_MULTI_DATASET_ID,
        "variant": variant,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "evaluation_contract": DETEXTURE_MULTI_EVALUATION_CONTRACT,
        "evaluation_view": sample.evaluation_view,
        DETEXTURE_MULTI_PRIMARY_METRIC: metrics.miou,
        DETEXTURE_MULTI_SECONDARY_METRIC: metrics.ari,
        "eval_nmi": metrics.nmi,
        "k_gt": sample.oracle_num_regions,
        "k_gt_raw": sample.raw_oracle_num_regions,
        "k_pred": partition.num_clusters,
        "count_accuracy": metrics.count_accuracy,
        "coverage": metrics.coverage,
        "valid_pixel_count": metrics.valid_pixel_count,
        "total_pixel_count": metrics.total_pixel_count,
        "invalid_pixel_count": int(metrics.total_pixel_count - metrics.valid_pixel_count),
        "num_predicted_regions": metrics.num_predicted_regions,
        "num_target_regions": metrics.num_target_regions,
        "label_permutation_invariant": True,
        "evaluation_selector": "hungarian_iou_assignment",
        "matched_ious_json": json.dumps([float(value) for value in metrics.matched_ious]),
        "matched_pairs_json": json.dumps([
            {"pred_label": int(pred_label), "gt_label": int(gt_label), "iou": float(iou)}
            for pred_label, gt_label, iou in metrics.matched_pairs
        ]),
        "coarsest_level_name": partition.coarsest_level_name,
        "feature_level_names": partition.coarsest_level_name,
        "feature_level_resolutions": f"{partition.coarsest_level_resolution[0]}x{partition.coarsest_level_resolution[1]}",
        "num_feature_levels": 1,
        "coarsest_pool_kernel_size": partition.coarsest_pool_kernel_size,
        "coarsest_pool_stride": partition.coarsest_pool_stride,
        "pooled_grid_resolution": f"{partition.pooled_grid_resolution[0]}x{partition.pooled_grid_resolution[1]}",
        "coarsest_init_mode": partition.coarsest_init_mode,
        "flip_averaged_features_used": partition.flip_averaged_features_used,
        "partition_head": partition.partition_head,
        "sam_refinement_applied": False,
        "multiscale_refinement_applied": False,
        "predicted_k_selection_criterion": partition.predicted_k_selection_criterion,
        "predicted_k_criterion_value": partition.predicted_k_criterion_value,
        "predicted_k_scores_by_k_json": json.dumps(partition.predicted_k_scores_by_k, sort_keys=True)
        if partition.predicted_k_scores_by_k is not None
        else None,
        "cluster_pixel_counts_json": json.dumps([int(value) for value in partition.cluster_pixel_counts]),
        "clustering_diagnostics_json": json.dumps(clustering_training_diagnostics, sort_keys=True)
        if clustering_training_diagnostics
        else None,
        "hdbscan_noise_point_count": clustering_training_diagnostics.get("noise_point_count"),
        "hdbscan_noise_fraction": clustering_training_diagnostics.get("noise_fraction"),
        "hdbscan_noise_point_count_before_repair": clustering_training_diagnostics.get("noise_point_count_before_repair"),
        "hdbscan_noise_fraction_before_repair": clustering_training_diagnostics.get("noise_fraction_before_repair"),
        "hdbscan_all_noise_repair_used": clustering_training_diagnostics.get("all_noise_repair_used"),
        "hdbscan_all_noise_policy": clustering_training_diagnostics.get("all_noise_policy"),
        "deepdpm_epochs_completed": clustering_training_diagnostics.get("epochs_completed"),
        "deepdpm_final_loss": clustering_training_diagnostics.get("final_loss"),
        "deepdpm_inferred_k_trace_json": json.dumps(clustering_training_diagnostics.get("inferred_k_trace", [])),
        "deepdpm_split_merge_events_json": json.dumps(clustering_training_diagnostics.get("split_merge_events", [])),
        "deepdpm_training_diagnostics_json": json.dumps(clustering_training_diagnostics, sort_keys=True)
        if partition.partition_head.startswith("deepdpm") and clustering_training_diagnostics
        else None,
        "gt_decode_method": sample.gt_decode_method,
        "joker_label_value": sample.joker_label_value,
        "joker_region_count": sample.joker_region_count,
        "joker_pixel_count": sample.joker_pixel_count,
        "joker_pixel_fraction": sample.joker_pixel_fraction,
        "gt_decode_raw_oracle_num_regions": sample.gt_decode_diagnostics.get("raw_oracle_num_regions"),
        "gt_decode_effective_oracle_num_regions": sample.gt_decode_diagnostics.get("effective_oracle_num_regions"),
        "gt_decode_num_unique_rgb_colors": sample.gt_decode_diagnostics.get("num_unique_rgb_colors"),
        "gt_decode_scores_by_k_json": json.dumps(sample.gt_decode_diagnostics.get("scores_by_k", {}), sort_keys=True),
        "gt_decode_raw_region_sizes_json": json.dumps(sample.gt_decode_diagnostics.get("raw_region_sizes", [])),
        "gt_decode_effective_region_sizes_json": json.dumps(sample.gt_decode_diagnostics.get("effective_region_sizes", [])),
        "gt_decode_joker_region_labels_json": json.dumps(sample.gt_decode_diagnostics.get("joker_region_labels", [])),
        "gt_decode_joker_region_sizes_json": json.dumps(sample.gt_decode_diagnostics.get("joker_region_sizes", [])),
        "gt_decode_selection_criterion": sample.gt_decode_diagnostics.get("selection_criterion"),
        "gt_decode_selection_value": sample.gt_decode_diagnostics.get("selection_value"),
        f"{variant}_status": "ok",
    }
    metric_summary = (
        f"{variant} | mIoU={metrics.miou:.3f} ARI={metrics.ari:.3f} "
        f"NMI={metrics.nmi:.3f} K_gt={sample.oracle_num_regions} K_pred={partition.num_clusters} "
        f"coverage={metrics.coverage:.3f} joker_px={sample.joker_pixel_fraction:.3f}"
    )
    return DeTextureMultiSampleResult(
        row=row,
        metric_summary=metric_summary,
        metrics=metrics,
        partition=partition,
    )


def build_detexture_multi_run_config(
    args,
    overview: DeTextureMultiOverview | None,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> dict[str, Any]:
    """Build a stable run-config payload for DeTexture multi-region runs."""

    deepdpm_runner_settings = dict(FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS)
    hdbscan_runner_settings = dict(FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS)
    resolved_runner_settings = _resolve_detexture_multi_runner_settings(args) or {}
    deepdpm_runner_settings.update(resolved_runner_settings)
    hdbscan_runner_settings.update(resolved_runner_settings)
    config = {
        "command": getattr(args, "command", None),
        "dataset_id": DETEXTURE_MULTI_DATASET_ID,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "variant": args.variant,
        "variant_summary": DETEXTURE_MULTI_VARIANT_SUMMARIES[args.variant],
        "visual_contract": DETEXTURE_MULTI_VISUAL_CONTRACT,
        "model_id": args.model_id,
        "device": args.device,
        "hardware_compatibility_standard": DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_STANDARD,
        "hardware_compatibility_description": DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_DESCRIPTION,
        "sample_loading_mode": "streamed_iter",
        "official_checkpoint_path": getattr(args, "official_checkpoint_path", None),
        "limit": getattr(args, "limit", None),
        "failure_policy": getattr(args, "failure_policy", "abort"),
        "save_visuals": args.save_visuals,
        "wandb": getattr(args, "wandb", False),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_cluster_coarse_global_pooled_multi_settings": FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS,
        "feature_cluster_coarse_global_pooled_deepdpm_settings": deepdpm_runner_settings,
        "feature_cluster_coarse_global_pooled_hdbscan_settings": hdbscan_runner_settings,
        "deepdpm_cli_overrides": _detexture_multi_deepdpm_hyperparameter_overrides(args),
        "hdbscan_cli_overrides": _detexture_multi_hdbscan_hyperparameter_overrides(args),
        "deepdpm_implementation_note": (
            "The repository currently uses an in-repo compact DeepDPM-style split/merge head on frozen pooled SAM "
            "features because an official deepdpm package is not vendored in this environment."
        ),
        "hdbscan_implementation_note": (
            "The repository uses sklearn.cluster.HDBSCAN directly on the pooled frozen SAM feature vectors, with no prompts "
            "and no postprocessing beyond contiguous label reindexing. If sklearn HDBSCAN marks every pooled vector as noise, "
            "the route applies a deterministic single-cluster repair and records that repair explicitly in the artifacts."
        ),
        "gt_decode_settings": DETEXTURE_MULTI_GT_DECODE_SETTINGS,
        "gt_region_filter_settings": DETEXTURE_MULTI_REGION_FILTER_SETTINGS,
        "evaluation_contract": DETEXTURE_MULTI_EVALUATION_CONTRACT,
        "primary_metric_name": DETEXTURE_MULTI_PRIMARY_METRIC,
        "secondary_metric_name": DETEXTURE_MULTI_SECONDARY_METRIC,
        "primary_metric_reason": (
            "This multi-region route uses Hungarian-matched mIoU on valid pixels so cluster label identities remain "
            "permutation-invariant across variable-K partitions. Pixels belonging to GT regions smaller than the "
            "configured joker threshold are excluded through valid-pixel masking and do not affect the metrics."
        ),
        "versions": discovered_package_versions(),
    }
    if overview is not None:
        config["resolved_dataset_root"] = str(overview.dataset_root)
        config["resolved_detecture_data_root"] = str(overview.detecture_data_root)
        config["resolved_image_dir"] = str(overview.image_dir)
        config["resolved_mask_dir"] = str(overview.mask_dir)
        config["split"] = overview.split
        config["num_examples"] = overview.num_examples
        config["evaluation_view"] = overview.evaluation_view
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_detexture_multi_experiment_terms_markdown(
    args,
    *,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> str:
    """Render a route-specific protocol-spec document for DeTexture multi-region evaluation."""

    lines = [
        "# DeTexture Multi-Region Benchmark Terms",
        "",
        "## Run Scope",
        "",
        f"- Command family: `{getattr(args, 'command', 'detexture-multi')}`",
        f"- Dataset id: `{DETEXTURE_MULTI_DATASET_ID}`",
        f"- Requested dataset root: `{args.dataset_root}`",
        "- Expected local root form: either the directory containing `detecture_data/images` and `detecture_data/masks`, or `detecture_data/` itself.",
        f"- Executed variant: `{args.variant}`",
        f"- Variant summary: {DETEXTURE_MULTI_VARIANT_SUMMARIES[args.variant]}",
        f"- Model: `{args.model_id}`",
        f"- Device request: `{args.device}`",
        f"- Hardware compatibility standard: `{DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_STANDARD}`",
        "- Sample loading mode: `streamed_iter` (the adapter does not preload all crops into memory).",
        (
            f"- Dataset partition: `{dataset_partition.raw_spec}` -> indices `{dataset_partition.start_index}:{dataset_partition.end_index}` "
            f"({selected_sample_count} selected sample(s) from {dataset_partition.partition_size} partition items, full dataset size {dataset_partition.total_size})."
            if dataset_partition is not None
            else "- Dataset partition: full dataset order (no partitioning)."
        ),
        f"- Save visuals: `{getattr(args, 'save_visuals', True)}`",
        "",
        "## Dataset Semantics",
        "",
        "- Each sample is one RGB crop image from `detecture_data/images` and one JPEG-compressed multi-region color mask from `detecture_data/masks`.",
        "- GT is decoded into a single integer label map by deterministic weighted color clustering over the compressed RGB mask values.",
        f"- GT decode settings: criterion=`{DETEXTURE_MULTI_GT_DECODE_SETTINGS['selection_criterion']}`, metric=`{DETEXTURE_MULTI_GT_DECODE_SETTINGS['decode_metric']}`, max clusters=`{DETEXTURE_MULTI_GT_DECODE_SETTINGS['max_decode_clusters']}`.",
        "- Oracle K is the number of decoded GT regions in that sample.",
        "- Valid-pixel evaluation currently includes all pixels in the decoded mask. This route does not silently drop pixels; any future exclusions must be reported explicitly in row metadata.",
        "",
        "## Method Standard",
        "",
        "1. Extract frozen SAM dense features using the same official coarsest FPN level used by the current binary pooled coarse-only method.",
        "2. L2-normalize coarsest features per spatial location.",
        f"3. Average-pool the coarsest feature map with kernel={FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS['coarsest_init_pool_kernel_size']} and stride={FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS['coarsest_init_pool_stride']}.",
        "4. Flatten pooled features to `[N, C]`.",
        "5. Run the selected partition head on those pooled vectors: deterministic K-way clustering for the baseline multi routes, or a trainable DeepDPM-style split/merge clustering module for the DeepDPM ablations.",
        "6. Upsample the pooled label map back to native coarsest resolution, then to image resolution, with nearest-neighbor only.",
        "7. Do not add prompts, proposal banks, graph reasoning, component cleanup, CRFs, appended coordinates, or learned refinement.",
        "",
        "## Variant Definitions",
        "",
        f"- `{MULTI_ORACLE_K_VARIANT}`: use oracle K from the decoded GT region count; cluster raw coarsest pooled features directly with the deterministic K-way head.",
        f"- `{MULTI_ORACLE_K_FLIPAVG_VARIANT}`: same oracle-K route, but flip-average coarsest features across `identity`, `hflip`, `vflip`, and `hvflip` before pooling/clustering.",
        f"- `{MULTI_PREDICTED_K_VARIANT}`: choose K deterministically by silhouette score over a small candidate range on the pooled normalized feature vectors. This is a secondary ablation, not the main result route.",
        f"- `{MULTI_DEEPDPM_VARIANT}`: a stronger nonparametric deep clustering ablation on frozen SAM coarse features. It keeps the pooled coarsest feature pipeline fixed and replaces the deterministic partition head with a trainable DeepDPM-style split/merge module that infers K.",
        f"- `{MULTI_DEEPDPM_FLIPAVG_VARIANT}`: same DeepDPM partition head, but use flip-averaged coarsest SAM features before pooling.",
        f"- `{MULTI_HDBSCAN_VARIANT}`: replace the deterministic partition head with sklearn HDBSCAN on the same pooled normalized frozen SAM vectors; K is inferred automatically from density structure.",
        f"- `{MULTI_HDBSCAN_FLIPAVG_VARIANT}`: same HDBSCAN partition head, but use flip-averaged coarsest SAM features before pooling.",
        f"- DeepDPM settings exported in run config: init K=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_init_clusters']}`, max K=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_max_clusters']}`, hidden dim=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_hidden_dim']}`, embedding dim=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_embedding_dim']}`, outer iterations=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_outer_iterations']}`, inner epochs=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS['deepdpm_inner_epochs']}`.",
        f"- HDBSCAN settings exported in run config: min_cluster_size=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS['hdbscan_min_cluster_size']}`, min_samples=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS['hdbscan_min_samples']}`, cluster_selection_epsilon=`{FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS['hdbscan_cluster_selection_epsilon']}`.",
        "- Implementation note: this repository currently uses an in-repo compact DeepDPM-style adaptation because the official DeepDPM package is not vendored here; it remains a trainable clustering module rather than a fixed clustering rule.",
        "- HDBSCAN note: noise points are kept as a regular predicted region if they appear; there is no extra postprocessing or spatial cleanup.",
        "",
        "## Metric Standard",
        "",
        f"- `{DETEXTURE_MULTI_PRIMARY_METRIC}`: Hungarian-matched mean IoU across predicted and GT labels on valid pixels. When `K_pred != K_gt`, zero-IoU padded assignment slots penalize count mismatch.",
        f"- `{DETEXTURE_MULTI_SECONDARY_METRIC}`: pixel ARI on valid pixels.",
        "- `eval_nmi`: normalized mutual information on valid pixels.",
        "- `count_accuracy`: exact region-count match indicator, macro-averaged in summaries.",
        "- `coverage`: valid-pixel fraction used in evaluation.",
        "",
        "## Output Contract",
        "",
        "- `prediction.json`, `prediction.png`, `predicted_label_map.npy`, `gt_label_map.npy`, and `valid_pixel_mask.npy` for `predict-detexture-multi`.",
        "- `per_sample_metrics.csv`, `summary.json`, `summary.md`, `visuals_manifest.jsonl`, `visuals/*.png`, and `label_maps/*.npy` for `eval-detexture-multi`.",
        f"- Visual audit contract: `{DETEXTURE_MULTI_VISUAL_CONTRACT}`.",
        "",
        "## Failure Semantics",
        "",
        "- There is no silent fallback to the binary DeTexture route or any prompt-refined path.",
        "- Missing dataset roots, missing image/mask pairs, shape mismatches, GT decode failure, degenerate clustering, or missing official SAM features are explicit failures.",
        "- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.",
    ]
    return "\n".join(lines) + "\n"


def build_detexture_multi_summary(
    overview: DeTextureMultiOverview,
    variant: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    num_total_samples: int,
    dataset_partition: DatasetPartitionSelection | None,
) -> dict[str, Any]:
    """Aggregate one multi-region DeTexture evaluation run."""

    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in DETEXTURE_MULTI_SCALAR_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in DETEXTURE_MULTI_SCALAR_FIELDS}
    num_samples_with_joker_regions = sum(int(float(row.get("joker_region_count", 0)) > 0) for row in rows)
    mean_joker_pixel_fraction = mean(float(row.get("joker_pixel_fraction", 0.0)) for row in rows)
    max_joker_pixel_fraction = max(float(row.get("joker_pixel_fraction", 0.0)) for row in rows)
    summary = {
        "dataset_id": DETEXTURE_MULTI_DATASET_ID,
        "dataset_root": str(overview.dataset_root),
        "split": overview.split,
        "evaluation_view": overview.evaluation_view,
        "variant": variant,
        "variant_summary": DETEXTURE_MULTI_VARIANT_SUMMARIES[variant],
        "visual_contract": DETEXTURE_MULTI_VISUAL_CONTRACT,
        "model_id": model_id,
        "hardware_compatibility_standard": DETEXTURE_MULTI_HARDWARE_COMPATIBILITY_STANDARD,
        "evaluation_contract": DETEXTURE_MULTI_EVALUATION_CONTRACT,
        "primary_metric_name": DETEXTURE_MULTI_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[DETEXTURE_MULTI_PRIMARY_METRIC],
        "secondary_metric_name": DETEXTURE_MULTI_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[DETEXTURE_MULTI_SECONDARY_METRIC],
        "num_total_samples": num_total_samples,
        "full_dataset_num_examples": overview.num_examples,
        "num_successes": len(rows),
        "num_failures": len(failures),
        "num_samples_with_joker_regions": num_samples_with_joker_regions,
        "mean_joker_pixel_fraction": mean_joker_pixel_fraction,
        "max_joker_pixel_fraction": max_joker_pixel_fraction,
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }
    append_dataset_partition_fields(summary, dataset_partition, selected_sample_count=num_total_samples)
    return summary


def build_detexture_multi_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a compact Markdown summary for one multi-region DeTexture run."""

    lines = [
        "# DeTexture Multi Summary",
        "",
        f"- Dataset id: `{summary['dataset_id']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Split: `{summary['split']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Primary metric ({summary['primary_metric_name']}): {summary['primary_metric_value']:.4f}",
        f"- Secondary metric ({summary['secondary_metric_name']}): {summary['secondary_metric_value']:.4f}",
        f"- Mean NMI: {summary['mean_metrics']['eval_nmi']:.4f}",
        f"- Mean count accuracy: {summary['mean_metrics']['count_accuracy']:.4f}",
        f"- Mean coverage: {summary['mean_metrics']['coverage']:.4f}",
        f"- Mean joker pixel fraction: {summary['mean_joker_pixel_fraction']:.4f}",
        f"- Samples with joker regions: {summary['num_samples_with_joker_regions']}",
        f"- Successes / failures: {summary['num_successes']} / {summary['num_failures']}",
    ]
    return "\n".join(lines) + "\n"


def save_detexture_multi_panel(
    output_path: str | Path,
    sample: DeTextureMultiSample,
    evaluation: DeTextureMultiSampleResult,
    variant: str,
) -> Path:
    """Save the standard DeTexture multi-region audit panel."""

    return save_detexture_multi_partition_panel(
        output_path=output_path,
        sample=sample,
        gt_label_map=sample.gt_label_map,
        predicted_label_map=evaluation.partition.predicted_label_map,
        valid_pixel_mask=sample.valid_pixel_mask,
        protocol=_detexture_multi_protocol_label(variant),
        metric_summary=evaluation.metric_summary,
    )


def build_detexture_multi_visual_record(
    sample: DeTextureMultiSample,
    evaluation: DeTextureMultiSampleResult,
    variant: str,
    visual_path: Path,
    label_map_path: Path,
) -> dict[str, Any]:
    """Build the JSONL visual-manifest record for one multi-region sample."""

    return {
        "dataset_id": DETEXTURE_MULTI_DATASET_ID,
        "variant": variant,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "split": sample.split,
        "visual_path": str(visual_path),
        "label_map_path": str(label_map_path),
        "caption": _build_detexture_multi_caption(sample, evaluation, variant),
        "eval_miou": evaluation.row[DETEXTURE_MULTI_PRIMARY_METRIC],
        "eval_ari": evaluation.row[DETEXTURE_MULTI_SECONDARY_METRIC],
        "eval_nmi": evaluation.row["eval_nmi"],
        "k_gt": evaluation.row["k_gt"],
        "k_pred": evaluation.row["k_pred"],
    }


def _build_detexture_multi_caption(
    sample: DeTextureMultiSample,
    evaluation: DeTextureMultiSampleResult,
    variant: str,
) -> str:
    return (
        f"crop={sample.crop_name} | split={sample.split} | protocol={_detexture_multi_protocol_label(variant)} | "
        f"mIoU={evaluation.row[DETEXTURE_MULTI_PRIMARY_METRIC]:.3f} | "
        f"ARI={evaluation.row[DETEXTURE_MULTI_SECONDARY_METRIC]:.3f} | "
        f"NMI={evaluation.row['eval_nmi']:.3f} | K_gt={evaluation.row['k_gt']} | K_pred={evaluation.row['k_pred']} | "
        f"joker_px={evaluation.row['joker_pixel_fraction']:.3f}"
    )


def _detexture_multi_protocol_label(variant: str) -> str:
    return variant.replace("feature_cluster_", "").replace("_", "-")


def prepare_detexture_multi_output_dir(
    output_dir: str | None,
    *,
    variant: str,
    run_kind: str,
    sample_index: int | None = None,
) -> Path:
    """Resolve and validate the run output directory."""

    root = DETEXTURE_MULTI_OUTPUT_ROOT
    if output_dir is None:
        if run_kind == "predict":
            if sample_index is None:
                raise ValueError("predict output dir resolution requires sample_index.")
            resolved = root / f"{variant}_index{sample_index}"
        else:
            resolved = root / f"test_{variant}"
    else:
        requested = Path(output_dir).expanduser()
        resolved = requested if requested.is_absolute() else Path.cwd() / requested
        if resolved == (Path.cwd() / root):
            resolved = root / (f"{variant}_index{sample_index}" if run_kind == "predict" else f"test_{variant}")
    _validate_detexture_multi_output_dir(resolved, variant=variant, run_kind=run_kind)
    return resolved


def _validate_detexture_multi_output_dir(path: Path, *, variant: str, run_kind: str) -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        return
    existing = json.loads(config_path.read_text())
    existing_variant = existing.get("variant")
    existing_command = str(existing.get("command", ""))
    existing_is_predict = existing_command.startswith("predict-")
    requested_is_predict = run_kind == "predict"
    if existing_variant not in {None, variant}:
        raise RuntimeError(
            f"Refusing to reuse '{path}' because it already belongs to variant '{existing_variant}', not '{variant}'."
        )
    if existing_is_predict != requested_is_predict:
        raise RuntimeError(
            f"Refusing to mix predict/eval artifacts in '{path}'. Existing command='{existing_command}', requested run_kind='{run_kind}'."
        )
