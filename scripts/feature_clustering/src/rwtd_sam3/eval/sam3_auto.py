"""SAM-3 automatic-mask baselines and feature-based ablation entrypoints.

This module owns the ``predict-sam3-auto`` and ``eval-sam3-auto`` command
families. It runs the plain Transformers automatic-mask baselines, the
feature-based SAM refinement ablations, the prompt-invariance control, and the
shared output/summary logic for those experiment tracks.

Primary entrypoints:
- ``run_sam3_auto_predict_one()``: evaluate one RWTD sample for one automatic
  mask variant.
- ``run_sam3_auto_evaluation()``: evaluate a split and persist run artifacts.
- ``evaluate_sam3_auto_sample()``: score one decoded sample under the requested
  automatic-mask protocol.

Inputs are decoded ``RwtdSample`` objects plus runner instances that expose the
variant-specific generation methods. Outputs are run directories under
``outputs/rwtd_sam3_auto/`` with configs, summaries, CSV rows, JSONL visual
manifests, and variant-specific PNG panels. The module validates output
directories aggressively so predict/eval artifacts and conflicting variants are
not mixed silently.
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
from tqdm import tqdm

from rwtd_sam3.data.rwtd import DEFAULT_DATASET_ID, RwtdSample, get_rwtd_sample, iter_rwtd_samples, load_split_overview
from rwtd_sam3.eval.metrics import (
    aggregate_masks_by_regions,
    build_canonical_evaluation_fields,
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    compute_binary_metrics,
    compute_canonical_partition_metrics_for_mask_set,
    compute_mask_set_metrics,
    compute_partition_metrics,
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
from rwtd_sam3.eval.experiment_registry import (
    CROSS_DATASET_EXPERIMENT_VARIANTS,
    get_cross_dataset_experiment_spec,
    resolve_cross_dataset_experiment_model_id,
)
from rwtd_sam3.models.sam3_auto_runner import (
    DEFAULT_SAM3_AUTO_MODEL_ID,
    SAM3_AUTO_PRESETS,
    Sam3AutomaticMaskRunner,
)
from rwtd_sam3.models.sam3_feature_mask_runner import (
    FEATURE_MASK_SETTINGS,
    FeatureMaskRefinement,
    Sam3FeatureMaskRuntimeError,
    Sam3FeatureMaskRunner,
)
from rwtd_sam3.models.sam3_feature_cluster_global_runner import (
    FEATURE_CLUSTER_GLOBAL_SETTINGS,
    FeatureClusterGlobalRefinement,
    Sam3FeatureClusterGlobalRuntimeError,
    Sam3FeatureClusterGlobalRunner,
)
from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    FeatureClusterCoarseToFineGlobalRefinement,
    Sam3FeatureClusterCoarseToFineGlobalRuntimeError,
    Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner,
)
from rwtd_sam3.models.sam2_feature_cluster_runner import (
    SAM2_CFC_SETTINGS,
    Sam2FeatureClusterCoarseOnlyRunner,
    Sam2FeatureClusterFlipAvgCoarseOnlyRunner,
)
from rwtd_sam3.models.sam3_mask_prompt_invariance_runner import (
    MaskPromptInvarianceRefinement,
    Sam3MaskPromptInvarianceRunner,
)
from rwtd_sam3.models.sam3_boundary_refine_sweep_runner import (
    BOUNDARY_REFINE_SWEEP_SETTINGS,
    BOUNDARY_REFINE_SWEEP_VARIANT_IDS,
    BOUNDARY_REFINE_SWEEP_VARIANT_LABELS,
    BoundaryRefineSweepResult,
    Sam3BoundaryRefineSweepRunner,
    build_boundary_refine_sweep_summary,
)
from rwtd_sam3.utils.visualization import (
    build_visual_caption,
    build_visual_footer_lines,
    render_mask_prompt_invariance_panel,
    render_automatic_mask_panel,
    render_feature_cluster_edge_bias_panel,
    render_feature_cluster_coarse_to_fine_global_panel,
    render_feature_cluster_positionality_panel,
    render_feature_cluster_global_panel,
    render_feature_mask_panel,
    save_automatic_mask_panel,
    save_boundary_refine_sweep_panel,
    save_feature_cluster_edge_bias_panel,
    save_mask_prompt_invariance_panel,
    save_feature_cluster_coarse_to_fine_global_panel,
    save_feature_cluster_positionality_panel,
    save_feature_cluster_global_panel,
    save_feature_mask_panel,
)
from rwtd_sam3.eval.pooled_feature_pca_overlay import save_named_sample_pooled_feature_pca_overlay


LOGGER = logging.getLogger(__name__)
SAM3_AUTO_OUTPUT_ROOT = Path("outputs") / "rwtd_sam3_auto"

SAM3_AUTO_SCALAR_FIELDS = (
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
    "mask_score_mean",
    "mask_score_median",
)


@dataclass(frozen=True)
class Sam3AutoSampleResult:
    """Per-sample SAM-3 automatic-mask evaluation payload."""

    row: dict[str, Any]
    metric_summary: str
    prediction_masks: tuple[np.ndarray, ...]
    aggregated_prediction_a: np.ndarray
    aggregated_prediction_b: np.ndarray
    feature_mask_refinement: FeatureMaskRefinement | None = None
    feature_cluster_global_refinement: FeatureClusterGlobalRefinement | None = None
    feature_cluster_coarse_to_fine_global_refinement: FeatureClusterCoarseToFineGlobalRefinement | None = None
    mask_prompt_invariance_refinement: MaskPromptInvarianceRefinement | None = None
    boundary_refine_sweep_refinement: BoundaryRefineSweepResult | None = None


@dataclass(frozen=True)
class BinaryAssignmentSelection:
    """Best permutation-invariant assignment for two unlabeled binary predictions."""

    assignment_used: str
    chosen_prediction_a: np.ndarray
    chosen_prediction_b: np.ndarray
    chosen_source_a: str
    chosen_source_b: str
    direct_miou: float
    direct_ari: float
    swapped_miou: float
    swapped_ari: float
    chosen_miou: float
    chosen_ari: float


def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 1.0


def _compute_partition_alignment(
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    reference_a: np.ndarray,
    reference_b: np.ndarray,
) -> dict[str, Any]:
    direct_mean_iou = float(
        np.mean(
            [
                _compute_mask_iou(candidate_a, reference_a),
                _compute_mask_iou(candidate_b, reference_b),
            ]
        )
    )
    swapped_mean_iou = float(
        np.mean(
            [
                _compute_mask_iou(candidate_a, reference_b),
                _compute_mask_iou(candidate_b, reference_a),
            ]
        )
    )
    if swapped_mean_iou > direct_mean_iou:
        return {
            "mask_a": np.asarray(candidate_b, dtype=bool),
            "mask_b": np.asarray(candidate_a, dtype=bool),
            "alignment_used": "swapped",
            "chosen_mean_iou": swapped_mean_iou,
        }
    return {
        "mask_a": np.asarray(candidate_a, dtype=bool),
        "mask_b": np.asarray(candidate_b, dtype=bool),
        "alignment_used": "direct",
        "chosen_mean_iou": direct_mean_iou,
    }


def _flatten_component_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for cluster_key in ("cluster_0", "cluster_1"):
        cluster_stats = stats.get(cluster_key)
        if not isinstance(cluster_stats, dict):
            continue
        cluster_suffix = cluster_key.replace("cluster_", "cluster")
        row[f"{prefix}_{cluster_suffix}_num_components"] = cluster_stats.get("num_components")
        row[f"{prefix}_{cluster_suffix}_num_1_cell_components"] = cluster_stats.get("num_1_cell_components")
        row[f"{prefix}_{cluster_suffix}_num_2_cell_components"] = cluster_stats.get("num_2_cell_components")
        row[f"{prefix}_{cluster_suffix}_tiny_component_pixels"] = cluster_stats.get("tiny_component_pixels")
        row[f"{prefix}_{cluster_suffix}_tiny_component_pixel_fraction"] = cluster_stats.get(
            "tiny_component_pixel_fraction"
        )
        row[f"{prefix}_{cluster_suffix}_largest_component_size"] = cluster_stats.get("largest_component_size")
        row[f"{prefix}_{cluster_suffix}_smallest_component_size"] = cluster_stats.get("smallest_component_size")
    return row


def _sum_tiny_component_pixels(stats: dict[str, Any] | None) -> int:
    if not isinstance(stats, dict):
        return 0
    total = 0
    for cluster_key in ("cluster_0", "cluster_1"):
        cluster_stats = stats.get(cluster_key)
        if isinstance(cluster_stats, dict):
            total += int(cluster_stats.get("tiny_component_pixels", 0))
    return total


def _append_label_positionality_fields(row: dict[str, Any], prefix: str, diagnostics: dict[str, Any] | None) -> None:
    if not isinstance(diagnostics, dict):
        return
    row[f"{prefix}_label_positionality_index"] = diagnostics.get("label_positionality_index")
    row[f"{prefix}_label_positionality_best_model"] = diagnostics.get("best_model_name")
    row[f"{prefix}_label_positionality_best_iou"] = diagnostics.get("best_iou")
    row[f"{prefix}_label_positionality_best_balanced_accuracy"] = diagnostics.get("best_balanced_accuracy")
    row[f"{prefix}_label_positionality_axis_threshold_index"] = diagnostics.get("axis_threshold_index")
    if "models" in diagnostics:
        row[f"{prefix}_label_positionality_models_json"] = json.dumps(diagnostics["models"], sort_keys=True)


def _append_feature_positionality_fields(row: dict[str, Any], prefix: str, diagnostics: dict[str, Any] | None) -> None:
    if not isinstance(diagnostics, dict):
        return
    row[f"{prefix}_feature_positionality_index"] = diagnostics.get("feature_positionality_index")
    row[f"{prefix}_feature_positionality_mean_r2"] = diagnostics.get("mean_r2")
    row[f"{prefix}_feature_positionality_max_r2"] = diagnostics.get("max_r2")
    row[f"{prefix}_feature_positionality_coordinate_basis"] = diagnostics.get("coordinate_basis")
    row[f"{prefix}_feature_positionality_channel_count"] = diagnostics.get("channel_count")
    row[f"{prefix}_feature_positionality_spatial_resolution"] = diagnostics.get("spatial_resolution")
    row[f"{prefix}_spatial_coordinates_appended"] = diagnostics.get("spatial_coordinates_appended")
    if "top_channels" in diagnostics:
        row[f"{prefix}_feature_positionality_top_channels_json"] = json.dumps(
            diagnostics["top_channels"],
            sort_keys=True,
        )


def _append_edge_label_positionality_fields(row: dict[str, Any], prefix: str, diagnostics: dict[str, Any] | None) -> None:
    if not isinstance(diagnostics, dict):
        return
    row[f"{prefix}_edge_positionality_index"] = diagnostics.get("edge_positionality_index")
    row[f"{prefix}_edge_positionality_best_model"] = diagnostics.get("best_model_name")
    row[f"{prefix}_edge_positionality_best_iou"] = diagnostics.get("best_iou")
    row[f"{prefix}_edge_positionality_best_balanced_accuracy"] = diagnostics.get("best_balanced_accuracy")
    row[f"{prefix}_edge_positionality_coordinate_basis"] = diagnostics.get("coordinate_basis")
    if "models" in diagnostics:
        row[f"{prefix}_edge_positionality_models_json"] = json.dumps(diagnostics["models"], sort_keys=True)


def _append_edge_feature_positionality_fields(row: dict[str, Any], prefix: str, diagnostics: dict[str, Any] | None) -> None:
    if not isinstance(diagnostics, dict):
        return
    row[f"{prefix}_feature_edge_positionality_index"] = diagnostics.get("feature_positionality_index")
    row[f"{prefix}_feature_edge_positionality_mean_r2"] = diagnostics.get("mean_r2")
    row[f"{prefix}_feature_edge_positionality_max_r2"] = diagnostics.get("max_r2")
    row[f"{prefix}_feature_edge_positionality_coordinate_basis"] = diagnostics.get("coordinate_basis")
    row[f"{prefix}_feature_edge_positionality_channel_count"] = diagnostics.get("channel_count")
    row[f"{prefix}_feature_edge_positionality_spatial_resolution"] = diagnostics.get("spatial_resolution")
    row[f"{prefix}_feature_edge_spatial_coordinates_appended"] = diagnostics.get("spatial_coordinates_appended")
    if "top_channels" in diagnostics:
        row[f"{prefix}_feature_edge_positionality_top_channels_json"] = json.dumps(
            diagnostics["top_channels"],
            sort_keys=True,
        )


def _compute_unlabeled_partition_view(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    sample: RwtdSample,
    *,
    source_a: str,
    source_b: str,
) -> dict[str, Any]:
    assignment = select_best_binary_assignment(
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        target_a=sample.texture_a_mask,
        target_b=sample.texture_b_mask,
        source_a=source_a,
        source_b=source_b,
    )
    texture_a_iou = compute_binary_metrics(assignment.chosen_prediction_a, sample.texture_a_mask).iou
    texture_b_iou = compute_binary_metrics(assignment.chosen_prediction_b, sample.texture_b_mask).iou
    aggregated_masks, overlap_counts = aggregate_masks_by_regions(
        [assignment.chosen_prediction_a, assignment.chosen_prediction_b],
        (sample.texture_a_mask, sample.texture_b_mask),
    )
    texture_a_agg_iou = compute_binary_metrics(aggregated_masks[0], sample.texture_a_mask).iou
    texture_b_agg_iou = compute_binary_metrics(aggregated_masks[1], sample.texture_b_mask).iou
    return {
        "assignment": assignment,
        "texture_a_iou": texture_a_iou,
        "texture_b_iou": texture_b_iou,
        "aggregated_masks": aggregated_masks,
        "overlap_counts": overlap_counts,
        "texture_a_agg_iou": texture_a_agg_iou,
        "texture_b_agg_iou": texture_b_agg_iou,
    }


def run_sam3_auto_predict_one(args) -> dict[str, Any]:
    """Run one RWTD sample through the SAM-3 automatic-mask comparison variants."""

    sample = get_rwtd_sample(
        split=args.split,
        index=args.index,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
    )
    output_dir = prepare_sam3_auto_output_dir(
        args.output_dir,
        variant=args.variant,
        split=args.split,
        run_kind="predict",
        sample_index=args.index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam3_auto_run_config(
        args,
        overview=None,
        dataset_partition=None,
        selected_sample_count=1,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_sam3_auto_experiment_terms_markdown(
            args,
            dataset_partition=None,
            selected_sample_count=1,
        ),
    )

    auto_runner = Sam3AutomaticMaskRunner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    feature_mask_runner: Sam3FeatureMaskRunner | None = None
    feature_cluster_global_runner: Sam3FeatureClusterGlobalRunner | None = None
    feature_cluster_coarse_to_fine_global_runner: Sam3FeatureClusterCoarseToFineGlobalRunner | None = None
    feature_cluster_coarse_to_fine_global_pooled_init_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner: (
        Sam2FeatureClusterCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner: (
        Sam2FeatureClusterFlipAvgCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner: (
        Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_direct_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner | None
    ) = None
    boundary_refine_sweep_runner: Sam3BoundaryRefineSweepRunner | None = None
    mask_prompt_invariance_runner: Sam3MaskPromptInvarianceRunner | None = None

    results: dict[str, Any] = {}
    for variant in resolve_sam3_auto_variants(args.variant):
        variant_dir = output_dir if args.variant != "both" else output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        try:
            if variant == "feature_mask":
                if feature_mask_runner is None:
                    feature_mask_runner = Sam3FeatureMaskRunner(
                        model_id=args.model_id,
                        device=args.device,
                        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                    )
                evaluation = evaluate_sam3_auto_sample(sample=sample, variant=variant, runner=feature_mask_runner)
            elif variant == "feature_cluster_global":
                if feature_cluster_global_runner is None:
                    feature_cluster_global_runner = Sam3FeatureClusterGlobalRunner(
                        model_id=args.model_id,
                        device=args.device,
                        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_global_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global":
                if feature_cluster_coarse_to_fine_global_runner is None:
                    feature_cluster_coarse_to_fine_global_runner = Sam3FeatureClusterCoarseToFineGlobalRunner(
                        model_id=args.model_id,
                        device=args.device,
                        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init":
                if feature_cluster_coarse_to_fine_global_pooled_init_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only":
                if feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2":
                if feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner = (
                        Sam2FeatureClusterCoarseOnlyRunner(
                            model_id=resolve_cross_dataset_experiment_model_id(variant, args.model_id),
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2":
                if feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner = (
                        Sam2FeatureClusterFlipAvgCoarseOnlyRunner(
                            model_id=resolve_cross_dataset_experiment_model_id(variant, args.model_id),
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only":
                if feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner,
                )
            elif variant == "boundary_refine_sweep":
                if boundary_refine_sweep_runner is None:
                    boundary_refine_sweep_runner = Sam3BoundaryRefineSweepRunner(
                        model_id=args.model_id,
                        device=args.device,
                        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=boundary_refine_sweep_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
                if feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner,
                )
            elif variant == "flip_avg_plus_edge_debias":
                if feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner is None:
                    feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner,
                )
            elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_direct":
                if feature_cluster_coarse_to_fine_global_pooled_init_direct_runner is None:
                    feature_cluster_coarse_to_fine_global_pooled_init_direct_runner = (
                        Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner(
                            model_id=args.model_id,
                            device=args.device,
                            hf_token=args.hf_token
                            or os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                            official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                        )
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=feature_cluster_coarse_to_fine_global_pooled_init_direct_runner,
                )
            elif variant == "mask_prompt_invariance_control":
                if mask_prompt_invariance_runner is None:
                    mask_prompt_invariance_runner = Sam3MaskPromptInvarianceRunner(
                        model_id=args.model_id,
                        device=args.device,
                        hf_token=args.hf_token
                        or os.environ.get("HF_TOKEN")
                        or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                    )
                evaluation = evaluate_sam3_auto_sample(
                    sample=sample,
                    variant=variant,
                    runner=mask_prompt_invariance_runner,
                )
            else:
                evaluation = evaluate_sam3_auto_sample(sample=sample, variant=variant, runner=auto_runner)

            prediction_path = variant_dir / "prediction.json"
            write_json(prediction_path, evaluation.row)
            panel_path = None
            if args.save_visuals:
                panel_path = save_sam3_auto_panel(
                    output_path=variant_dir / "prediction.png",
                    sample=sample,
                    evaluation=evaluation,
                    variant=variant,
                )
                write_jsonl(
                    variant_dir / "visuals_manifest.jsonl",
                    [
                        build_sam3_auto_visual_record(
                            sample=sample,
                            evaluation=evaluation,
                            variant=variant,
                            visual_path=Path("prediction.png"),
                        )
                    ],
                )
                if args.save_pooled_feature_pca_overlay and hasattr(
                    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner
                    or feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner
                    or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner
                    or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner
                    or feature_cluster_coarse_to_fine_global_runner
                    or feature_cluster_global_runner
                    or feature_mask_runner
                    or auto_runner,
                    "extract_pooled_feature_map_for_visualization",
                ):
                    active_runner = (
                        feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner
                        or feature_cluster_coarse_to_fine_global_runner
                        or feature_cluster_global_runner
                        or feature_mask_runner
                        or auto_runner
                    )
                    save_named_sample_pooled_feature_pca_overlay(
                        variant_dir / "visuals",
                        sample=sample,
                        runner=active_runner,
                    )
        except (
            Sam3FeatureMaskRuntimeError,
            Sam3FeatureClusterGlobalRuntimeError,
            Sam3FeatureClusterCoarseToFineGlobalRuntimeError,
        ) as exc:
            if variant not in {
                "feature_mask",
                "feature_cluster_global",
                "feature_cluster_coarse_to_fine_global",
                "feature_cluster_coarse_to_fine_global_pooled_init",
                "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only",
                "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
                "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
                "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
                "boundary_refine_sweep",
                "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only",
                "flip_avg_plus_edge_debias",
                "feature_cluster_coarse_to_fine_global_pooled_init_direct",
                "mask_prompt_invariance_control",
            }:
                raise
            failure_row = build_sam3_auto_failure_row(sample=sample, variant=variant, error=exc)
            dense_diagnostic_masks: tuple[np.ndarray, np.ndarray] | None = None
            failure_metric_summary = f"FAILED: {exc}"
            if variant == "feature_mask":
                failure_row, dense_diagnostic_masks, failure_metric_summary = enrich_feature_mask_failure_row(
                    sample=sample,
                    failure_row=failure_row,
                    error=exc,
                )
            prediction_path = variant_dir / "prediction.json"
            write_json(prediction_path, failure_row)
            panel_path = None
            if variant == "feature_mask" and args.save_visuals and getattr(exc, "dense_prediction_masks", ()):
                panel_path = save_automatic_mask_panel(
                    variant_dir / "prediction.png",
                    sample=sample,
                    prediction_masks=exc.dense_prediction_masks,
                    protocol=f"sam3_auto:{variant}:failed",
                    aggregated_prediction_a=dense_diagnostic_masks[0],
                    aggregated_prediction_b=dense_diagnostic_masks[1],
                    metric_summary=failure_metric_summary,
                    raw_panel_title=(
                        "Dense proposals only | feature_mask failed before selecting/refining a pair "
                        f"| count={len(exc.dense_prediction_masks)}"
                    ),
                    aggregated_panel_title=(
                        "Dense baseline aggregation | diagnostic only; not a feature_mask output"
                    ),
                )
                write_jsonl(
                    variant_dir / "visuals_manifest.jsonl",
                    [
                        build_sam3_auto_failure_visual_record(
                            sample=sample,
                            variant=variant,
                            failure_row=failure_row,
                            visual_path=Path("prediction.png"),
                            failure_message=str(exc),
                            metric_summary=failure_metric_summary,
                        )
                    ],
                )
            results[variant] = {
                "prediction_json": str(prediction_path),
                "visualization": str(panel_path) if panel_path is not None else None,
                "metrics": failure_row,
                "error": str(exc),
            }
            continue

        results[variant] = {
            "prediction_json": str(prediction_path),
            "visualization": str(panel_path) if panel_path is not None else None,
            "metrics": evaluation.row,
        }
    return results


def run_sam3_auto_evaluation(args) -> dict[str, Any]:
    """Run the SAM-3 automatic-mask comparison on RWTD."""

    overview = load_split_overview(split=args.split, dataset_id=args.dataset_id, cache_dir=args.cache_dir)
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_total_samples = resolve_eval_sample_count(args.limit, overview.num_examples, dataset_partition)
    if num_total_samples < 1:
        raise RuntimeError(f"Split '{args.split}' did not yield any samples to evaluate.")
    samples = list(
        iter_rwtd_samples(
            split=args.split,
            dataset_id=args.dataset_id,
            cache_dir=args.cache_dir,
            limit=num_total_samples,
            start_index=dataset_partition.start_index if dataset_partition is not None else 0,
        )
    )
    if not samples:
        raise RuntimeError(f"Split '{args.split}' did not yield any samples to evaluate.")

    output_dir = prepare_sam3_auto_output_dir(
        args.output_dir,
        variant=args.variant,
        split=args.split,
        run_kind="eval",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_sam3_auto_run_config(
        args,
        overview=overview,
        dataset_partition=dataset_partition,
        selected_sample_count=num_total_samples,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_sam3_auto_experiment_terms_markdown(
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
    auto_runner = Sam3AutomaticMaskRunner(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    feature_mask_runner: Sam3FeatureMaskRunner | None = None
    feature_cluster_global_runner: Sam3FeatureClusterGlobalRunner | None = None
    feature_cluster_coarse_to_fine_global_runner: Sam3FeatureClusterCoarseToFineGlobalRunner | None = None
    feature_cluster_coarse_to_fine_global_pooled_init_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner: (
        Sam2FeatureClusterCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner: (
        Sam2FeatureClusterFlipAvgCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner: (
        Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner | None
    ) = None
    feature_cluster_coarse_to_fine_global_pooled_init_direct_runner: (
        Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner | None
    ) = None
    boundary_refine_sweep_runner: Sam3BoundaryRefineSweepRunner | None = None
    mask_prompt_invariance_runner: Sam3MaskPromptInvarianceRunner | None = None

    all_variant_summaries: dict[str, Any] = {}
    try:
        for variant in resolve_sam3_auto_variants(args.variant):
            variant_dir = output_dir if args.variant != "both" else output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            if args.save_visuals:
                (variant_dir / "visuals").mkdir(parents=True, exist_ok=True)
            write_text(
                variant_dir / "experiment_terms.md",
                build_sam3_auto_experiment_terms_markdown(
                    args,
                    dataset_partition=dataset_partition,
                    selected_sample_count=num_total_samples,
                ),
            )

            rows: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            visual_records: list[dict[str, Any]] = []

            progress = tqdm(samples, desc=f"eval:sam3-auto:{variant}", unit="sample")
            for step, sample in enumerate(progress):
                try:
                    if variant == "feature_mask":
                        if feature_mask_runner is None:
                            feature_mask_runner = Sam3FeatureMaskRunner(
                                model_id=args.model_id,
                                device=args.device,
                                hf_token=args.hf_token
                                or os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_mask_runner,
                        )
                    elif variant == "feature_cluster_global":
                        if feature_cluster_global_runner is None:
                            feature_cluster_global_runner = Sam3FeatureClusterGlobalRunner(
                                model_id=args.model_id,
                                device=args.device,
                                hf_token=args.hf_token
                                or os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_global_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global":
                        if feature_cluster_coarse_to_fine_global_runner is None:
                            feature_cluster_coarse_to_fine_global_runner = Sam3FeatureClusterCoarseToFineGlobalRunner(
                                model_id=args.model_id,
                                device=args.device,
                                hf_token=args.hf_token
                                or os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init":
                        if feature_cluster_coarse_to_fine_global_pooled_init_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only":
                        if feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2":
                        if feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner = (
                                Sam2FeatureClusterCoarseOnlyRunner(
                                    model_id=resolve_cross_dataset_experiment_model_id(variant, args.model_id),
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2":
                        if feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner = (
                                Sam2FeatureClusterFlipAvgCoarseOnlyRunner(
                                    model_id=resolve_cross_dataset_experiment_model_id(variant, args.model_id),
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only":
                        if feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner,
                        )
                    elif variant == "boundary_refine_sweep":
                        if boundary_refine_sweep_runner is None:
                            boundary_refine_sweep_runner = Sam3BoundaryRefineSweepRunner(
                                model_id=args.model_id,
                                device=args.device,
                                hf_token=args.hf_token
                                or os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=boundary_refine_sweep_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
                        if feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_runner,
                        )
                    elif variant == "flip_avg_plus_edge_debias":
                        if feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner is None:
                            feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_flip_avg_plus_edge_debias_runner,
                        )
                    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_direct":
                        if feature_cluster_coarse_to_fine_global_pooled_init_direct_runner is None:
                            feature_cluster_coarse_to_fine_global_pooled_init_direct_runner = (
                                Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner(
                                    model_id=args.model_id,
                                    device=args.device,
                                    hf_token=args.hf_token
                                    or os.environ.get("HF_TOKEN")
                                    or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                    official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                                )
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=feature_cluster_coarse_to_fine_global_pooled_init_direct_runner,
                        )
                    elif variant == "mask_prompt_invariance_control":
                        if mask_prompt_invariance_runner is None:
                            mask_prompt_invariance_runner = Sam3MaskPromptInvarianceRunner(
                                model_id=args.model_id,
                                device=args.device,
                                hf_token=args.hf_token
                                or os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                                official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
                            )
                        evaluation = evaluate_sam3_auto_sample(
                            sample=sample,
                            variant=variant,
                            runner=mask_prompt_invariance_runner,
                        )
                    else:
                        evaluation = evaluate_sam3_auto_sample(sample=sample, variant=variant, runner=auto_runner)
                except Exception as exc:
                    failure = build_sam3_auto_failure_row(sample=sample, variant=variant, error=exc)
                    if variant == "feature_mask":
                        failure, _, _ = enrich_feature_mask_failure_row(
                            sample=sample,
                            failure_row=failure,
                            error=exc,
                        )
                    if args.failure_policy == "skip":
                        failures.append(failure)
                        LOGGER.error("Skipping sample %s: %s", sample.crop_name, exc)
                        continue
                    raise

                rows.append(evaluation.row)
                progress.set_postfix({"crop": sample.crop_name, "eval_miou": f"{evaluation.row['eval_miou']:.3f}"})

                if args.save_visuals:
                    visual_path = Path("visuals") / f"{sample.crop_name}.png"
                    save_sam3_auto_panel(
                        output_path=variant_dir / visual_path,
                        sample=sample,
                        evaluation=evaluation,
                        variant=variant,
                    )
                    visual_records.append(
                        build_sam3_auto_visual_record(
                            sample=sample,
                            evaluation=evaluation,
                            variant=variant,
                            visual_path=visual_path,
                        )
                    )
                    if args.save_pooled_feature_pca_overlay and hasattr(
                        feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner
                        or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner
                        or feature_cluster_coarse_to_fine_global_runner
                        or feature_cluster_global_runner
                        or feature_mask_runner
                        or auto_runner,
                        "extract_pooled_feature_map_for_visualization",
                    ):
                        active_runner = (
                            feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_runner
                            or feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_runner
                            or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_runner
                            or feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_runner
                            or feature_cluster_coarse_to_fine_global_runner
                            or feature_cluster_global_runner
                            or feature_mask_runner
                            or auto_runner
                        )
                        save_named_sample_pooled_feature_pca_overlay(
                            variant_dir / "visuals",
                            sample=sample,
                            runner=active_runner,
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
                        preview_image = render_sam3_auto_panel(
                            sample=sample,
                            evaluation=evaluation,
                            variant=variant,
                        )
                        caption = build_visual_caption(
                            sample=sample,
                            protocol=f"sam3_auto:{variant}",
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
            summary = build_sam3_auto_variant_summary(
                variant=variant,
                split=args.split,
                dataset_id=args.dataset_id,
                model_id=args.model_id,
                rows=rows,
                failures=failures,
                num_total_samples=num_total_samples,
                full_dataset_num_examples=overview.num_examples,
                dataset_partition=dataset_partition,
            )
            write_json(variant_dir / "summary.json", summary)
            write_text(variant_dir / "summary.md", build_sam3_auto_markdown_summary(summary))
            all_variant_summaries[variant] = summary

        if args.variant == "both":
            write_json(output_dir / "summary.json", {"variants": all_variant_summaries})
            write_text(
                output_dir / "summary.md",
                "# RWTD SAM-3 Automatic Mask Evaluation\n\nBoth automatic-mask variants completed successfully.\n",
            )
    finally:
        wandb_session.finish()

    return all_variant_summaries


def evaluate_sam3_auto_sample(
    sample: RwtdSample,
    variant: str,
    runner: Sam3AutomaticMaskRunner,
) -> Sam3AutoSampleResult:
    """Evaluate one RWTD sample with one SAM-3 automatic-mask variant."""

    if variant == "feature_cluster_global":
        if not hasattr(runner, "generate_feature_clusters"):
            raise RuntimeError("feature_cluster_global evaluation requires a runner with generate_feature_clusters().")

        refinement = runner.generate_feature_clusters(sample.image)
        assignment = select_best_binary_assignment(
            prediction_a=refinement.refined_mask_a,
            prediction_b=refinement.refined_mask_b,
            target_a=sample.texture_a_mask,
            target_b=sample.texture_b_mask,
            source_a="cluster_a",
            source_b="cluster_b",
        )

        texture_a_iou = compute_binary_metrics(assignment.chosen_prediction_a, sample.texture_a_mask).iou
        texture_b_iou = compute_binary_metrics(assignment.chosen_prediction_b, sample.texture_b_mask).iou
        aggregated_masks, overlap_counts = aggregate_masks_by_regions(
            [assignment.chosen_prediction_a, assignment.chosen_prediction_b],
            (sample.texture_a_mask, sample.texture_b_mask),
        )
        texture_a_agg_iou = compute_binary_metrics(aggregated_masks[0], sample.texture_a_mask).iou
        texture_b_agg_iou = compute_binary_metrics(aggregated_masks[1], sample.texture_b_mask).iou
        refined_scores = []
        if refinement.refined_score_a is not None:
            refined_scores.append(refinement.refined_score_a)
        if refinement.refined_score_b is not None:
            refined_scores.append(refinement.refined_score_b)

        row = {
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "points_per_crop": 0,
            "stability_score_thresh": 0.0,
            "num_predicted_masks": 2,
            "evaluation_view": "partition_invariant",
            "miou": assignment.chosen_miou,
            "ari": assignment.chosen_ari,
            "miou_agg": float(np.mean([texture_a_agg_iou, texture_b_agg_iou])),
            "texture_a_best_iou": texture_a_iou,
            "texture_b_best_iou": texture_b_iou,
            "texture_a_agg_iou": texture_a_agg_iou,
            "texture_b_agg_iou": texture_b_agg_iou,
            "texture_a_overlap_mask_count": overlap_counts[0],
            "texture_b_overlap_mask_count": overlap_counts[1],
            "mask_score_mean": float(np.mean(refined_scores)) if refined_scores else 0.0,
            "mask_score_median": float(np.median(refined_scores)) if refined_scores else 0.0,
            "feature_cluster_global_status": "ok",
            "label_permutation_invariant": True,
            "refinement_applied": refinement.refinement_applied,
            "assignment_used": assignment.assignment_used,
            "assignment_source_for_texture_a": assignment.chosen_source_a,
            "assignment_source_for_texture_b": assignment.chosen_source_b,
            "assignment_direct_miou": assignment.direct_miou,
            "assignment_direct_ari": assignment.direct_ari,
            "assignment_swapped_miou": assignment.swapped_miou,
            "assignment_swapped_ari": assignment.swapped_ari,
            "cluster_pixel_count_a": refinement.cluster_pixel_count_a,
            "cluster_pixel_count_b": refinement.cluster_pixel_count_b,
            "rough_mask_positive_pixels_a": int(refinement.rough_mask_a.sum()),
            "rough_mask_positive_pixels_b": int(refinement.rough_mask_b.sum()),
            "refined_mask_positive_pixels_a": int(refinement.refined_mask_a.sum()),
            "refined_mask_positive_pixels_b": int(refinement.refined_mask_b.sum()),
            "refined_score_a": refinement.refined_score_a,
            "refined_score_b": refinement.refined_score_b,
            "refined_pair_selection_score": refinement.refined_pair_selection_score,
            "refined_pair_overlap_iou": refinement.refined_pair_overlap_iou,
            "output_stage": "refined" if refinement.refinement_applied else "rough_clusters",
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=assignment.chosen_miou,
                ari=assignment.chosen_ari,
                evaluation_view="partition_invariant",
            )
        )
        metric_summary = (
            f"Eval mIoU={assignment.chosen_miou:.3f} "
            f"Eval ARI={assignment.chosen_ari:.3f} "
            f"Aggr mIoU={row['miou_agg']:.3f} "
            f"Refine={'on' if refinement.refinement_applied else 'off'} "
            f"PairOv={float(refinement.refined_pair_overlap_iou or 0.0):.3f} "
            f"Assign={assignment.assignment_used}"
        )
        return Sam3AutoSampleResult(
            row=row,
            metric_summary=metric_summary,
            prediction_masks=(assignment.chosen_prediction_a, assignment.chosen_prediction_b),
            aggregated_prediction_a=aggregated_masks[0],
            aggregated_prediction_b=aggregated_masks[1],
            feature_cluster_global_refinement=refinement,
        )

    if variant == "mask_prompt_invariance_control":
        if not hasattr(runner, "generate_prompt_invariance"):
            raise RuntimeError(
                "mask_prompt_invariance_control evaluation requires a runner with generate_prompt_invariance()."
            )

        refinement = runner.generate_prompt_invariance(sample.image)

        def evaluate_branch(
            branch_name: str,
            prediction_a: np.ndarray,
            prediction_b: np.ndarray,
            score_a: float,
            score_b: float,
        ) -> dict[str, Any]:
            assignment = select_best_binary_assignment(
                prediction_a=prediction_a,
                prediction_b=prediction_b,
                target_a=sample.texture_a_mask,
                target_b=sample.texture_b_mask,
                source_a=f"{branch_name}_a",
                source_b=f"{branch_name}_b",
            )
            aggregated_masks, overlap_counts = aggregate_masks_by_regions(
                [assignment.chosen_prediction_a, assignment.chosen_prediction_b],
                (sample.texture_a_mask, sample.texture_b_mask),
            )
            return {
                "assignment": assignment,
                "aggregated_prediction_a": aggregated_masks[0],
                "aggregated_prediction_b": aggregated_masks[1],
                "overlap_counts": overlap_counts,
                "texture_a_agg_iou": compute_binary_metrics(aggregated_masks[0], sample.texture_a_mask).iou,
                "texture_b_agg_iou": compute_binary_metrics(aggregated_masks[1], sample.texture_b_mask).iou,
                "mask_score_mean": float(np.mean([score_a, score_b])),
                "mask_score_median": float(np.median([score_a, score_b])),
            }

        raw_metrics = evaluate_branch(
            "raw",
            refinement.raw_branch.final_mask_a,
            refinement.raw_branch.final_mask_b,
            refinement.raw_branch.refined_score_a,
            refinement.raw_branch.refined_score_b,
        )
        pooled_metrics = evaluate_branch(
            "pooled",
            refinement.pooled_branch.final_mask_a,
            refinement.pooled_branch.final_mask_b,
            refinement.pooled_branch.refined_score_a,
            refinement.pooled_branch.refined_score_b,
        )
        random_metrics = evaluate_branch(
            "random",
            refinement.random_branch.final_mask_a,
            refinement.random_branch.final_mask_b,
            refinement.random_branch.refined_score_a,
            refinement.random_branch.refined_score_b,
        )

        raw_to_pooled_prompt = _compute_partition_alignment(
            refinement.raw_branch.prompt_mask_a,
            refinement.raw_branch.prompt_mask_b,
            refinement.pooled_branch.prompt_mask_a,
            refinement.pooled_branch.prompt_mask_b,
        )
        random_to_pooled_prompt = _compute_partition_alignment(
            refinement.random_branch.prompt_mask_a,
            refinement.random_branch.prompt_mask_b,
            refinement.pooled_branch.prompt_mask_a,
            refinement.pooled_branch.prompt_mask_b,
        )
        raw_to_random_prompt = _compute_partition_alignment(
            refinement.raw_branch.prompt_mask_a,
            refinement.raw_branch.prompt_mask_b,
            refinement.random_branch.prompt_mask_a,
            refinement.random_branch.prompt_mask_b,
        )
        raw_to_random_final = _compute_partition_alignment(
            refinement.raw_branch.final_mask_a,
            refinement.raw_branch.final_mask_b,
            refinement.random_branch.final_mask_a,
            refinement.random_branch.final_mask_b,
        )

        pooled_assignment = pooled_metrics["assignment"]
        pooled_texture_a_iou = compute_binary_metrics(
            pooled_assignment.chosen_prediction_a,
            sample.texture_a_mask,
        ).iou
        pooled_texture_b_iou = compute_binary_metrics(
            pooled_assignment.chosen_prediction_b,
            sample.texture_b_mask,
        ).iou
        all_scores = [
            refinement.raw_branch.refined_score_a,
            refinement.raw_branch.refined_score_b,
            refinement.pooled_branch.refined_score_a,
            refinement.pooled_branch.refined_score_b,
            refinement.random_branch.refined_score_a,
            refinement.random_branch.refined_score_b,
        ]
        row = {
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "points_per_crop": 0,
            "stability_score_thresh": 0.0,
            "num_predicted_masks": 2,
            "evaluation_view": "partition_invariant",
            "miou": pooled_assignment.chosen_miou,
            "ari": pooled_assignment.chosen_ari,
            "miou_agg": float(
                np.mean([pooled_metrics["texture_a_agg_iou"], pooled_metrics["texture_b_agg_iou"]])
            ),
            "texture_a_best_iou": pooled_texture_a_iou,
            "texture_b_best_iou": pooled_texture_b_iou,
            "texture_a_agg_iou": pooled_metrics["texture_a_agg_iou"],
            "texture_b_agg_iou": pooled_metrics["texture_b_agg_iou"],
            "texture_a_overlap_mask_count": pooled_metrics["overlap_counts"][0],
            "texture_b_overlap_mask_count": pooled_metrics["overlap_counts"][1],
            "mask_score_mean": float(np.mean(all_scores)),
            "mask_score_median": float(np.median(all_scores)),
            "mask_prompt_invariance_control_status": "ok",
            "label_permutation_invariant": True,
            "assignment_used": pooled_assignment.assignment_used,
            "assignment_source_for_texture_a": pooled_assignment.chosen_source_a,
            "assignment_source_for_texture_b": pooled_assignment.chosen_source_b,
            "assignment_direct_miou": pooled_assignment.direct_miou,
            "assignment_direct_ari": pooled_assignment.direct_ari,
            "assignment_swapped_miou": pooled_assignment.swapped_miou,
            "assignment_swapped_ari": pooled_assignment.swapped_ari,
            "experiment_designation": "side_experiment",
            "canonical_branch": "pooled",
            "output_stage": "mask_prompt_invariance_control",
            "coarsest_level_name": refinement.coarsest_level_name,
            "feature_level_names": refinement.coarsest_level_name,
            "feature_level_resolutions": (
                f"{refinement.coarsest_level_resolution[0]}x{refinement.coarsest_level_resolution[1]}"
            ),
            "pooled_grid_resolution": (
                f"{refinement.pooled_grid_resolution[0]}x{refinement.pooled_grid_resolution[1]}"
            ),
            "raw_branch_miou": raw_metrics["assignment"].chosen_miou,
            "raw_branch_ari": raw_metrics["assignment"].chosen_ari,
            "pooled_branch_miou": pooled_metrics["assignment"].chosen_miou,
            "pooled_branch_ari": pooled_metrics["assignment"].chosen_ari,
            "random_branch_miou": random_metrics["assignment"].chosen_miou,
            "random_branch_ari": random_metrics["assignment"].chosen_ari,
            "raw_to_pooled_prompt_alignment_used": raw_to_pooled_prompt["alignment_used"],
            "raw_to_pooled_prompt_mean_iou": raw_to_pooled_prompt["chosen_mean_iou"],
            "raw_to_pooled_final_mean_iou": refinement.raw_to_pooled_final_mean_iou,
            "raw_to_pooled_final_alignment_used": refinement.raw_to_pooled_final_alignment_used,
            "random_to_pooled_prompt_alignment_used": random_to_pooled_prompt["alignment_used"],
            "random_to_pooled_prompt_mean_iou": random_to_pooled_prompt["chosen_mean_iou"],
            "random_to_pooled_final_mean_iou": refinement.random_to_pooled_final_mean_iou,
            "random_to_pooled_final_alignment_used": refinement.random_to_pooled_final_alignment_used,
            "raw_to_random_prompt_alignment_used": raw_to_random_prompt["alignment_used"],
            "raw_to_random_prompt_mean_iou": raw_to_random_prompt["chosen_mean_iou"],
            "raw_to_random_final_alignment_used": raw_to_random_final["alignment_used"],
            "raw_to_random_final_mean_iou": raw_to_random_final["chosen_mean_iou"],
            "raw_selected_candidate_rank_a": refinement.raw_branch.selected_candidate_rank_a,
            "raw_selected_candidate_rank_b": refinement.raw_branch.selected_candidate_rank_b,
            "pooled_selected_candidate_rank_a": refinement.pooled_branch.selected_candidate_rank_a,
            "pooled_selected_candidate_rank_b": refinement.pooled_branch.selected_candidate_rank_b,
            "random_selected_candidate_rank_a": refinement.random_branch.selected_candidate_rank_a,
            "random_selected_candidate_rank_b": refinement.random_branch.selected_candidate_rank_b,
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=pooled_assignment.chosen_miou,
                ari=pooled_assignment.chosen_ari,
                evaluation_view="partition_invariant",
            )
        )
        metric_summary = (
            f"Eval mIoU={pooled_assignment.chosen_miou:.3f} "
            f"Eval ARI={pooled_assignment.chosen_ari:.3f} "
            f"dirty->clean prompt/final={raw_to_pooled_prompt['chosen_mean_iou']:.3f}/{refinement.raw_to_pooled_final_mean_iou:.3f} "
            f"random->clean prompt/final={random_to_pooled_prompt['chosen_mean_iou']:.3f}/{refinement.random_to_pooled_final_mean_iou:.3f}"
        )
        return Sam3AutoSampleResult(
            row=row,
            metric_summary=metric_summary,
            prediction_masks=(
                pooled_assignment.chosen_prediction_a,
                pooled_assignment.chosen_prediction_b,
            ),
            aggregated_prediction_a=pooled_metrics["aggregated_prediction_a"],
            aggregated_prediction_b=pooled_metrics["aggregated_prediction_b"],
            mask_prompt_invariance_refinement=refinement,
        )

    if variant == "feature_mask":
        if not hasattr(runner, "generate_feature_masks"):
            raise RuntimeError("feature_mask evaluation requires a runner with generate_feature_masks().")

        refinement = runner.generate_feature_masks(sample.image)
        prediction_masks = [refinement.refined_mask_a, refinement.refined_mask_b]
        gt_regions = (sample.texture_a_mask, sample.texture_b_mask)
        metrics = compute_mask_set_metrics(prediction_masks, gt_regions)
        assignment = select_best_binary_assignment(
            prediction_a=refinement.refined_mask_a,
            prediction_b=refinement.refined_mask_b,
            target_a=sample.texture_a_mask,
            target_b=sample.texture_b_mask,
            source_a="feature_mask_a",
            source_b="feature_mask_b",
        )
        aggregated_masks, overlap_counts = aggregate_masks_by_regions(
            [assignment.chosen_prediction_a, assignment.chosen_prediction_b],
            gt_regions,
        )

        refined_scores = [refinement.refined_score_a]
        if refinement.refined_score_b is not None:
            refined_scores.append(refinement.refined_score_b)
        row = {
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "points_per_crop": int(SAM3_AUTO_PRESETS["dense"]["points_per_crop"]),
            "stability_score_thresh": float(SAM3_AUTO_PRESETS["dense"]["stability_score_thresh"]),
            "num_predicted_masks": metrics.num_predicted_masks,
            "evaluation_view": "partition_invariant",
            "miou": metrics.miou,
            "ari": metrics.ari,
            "miou_agg": metrics.aggregated_miou,
            "texture_a_best_iou": metrics.region_best_ious[0],
            "texture_b_best_iou": metrics.region_best_ious[1],
            "texture_a_agg_iou": metrics.region_aggregated_ious[0],
            "texture_b_agg_iou": metrics.region_aggregated_ious[1],
            "texture_a_overlap_mask_count": overlap_counts[0],
            "texture_b_overlap_mask_count": overlap_counts[1],
            "mask_score_mean": float(np.mean(refined_scores)) if refined_scores else 0.0,
            "mask_score_median": float(np.median(refined_scores)) if refined_scores else 0.0,
            "dense_num_predicted_masks": len(refinement.dense_prediction_masks),
            "dense_mask_score_mean": float(np.mean(refinement.dense_prediction_scores))
            if refinement.dense_prediction_scores
            else 0.0,
            "dense_mask_score_median": float(np.median(refinement.dense_prediction_scores))
            if refinement.dense_prediction_scores
            else 0.0,
            "feature_mask_mode": refinement.selection_mode,
            "selected_pair_id_a": refinement.selected_pair_ids[0],
            "selected_pair_id_b": refinement.selected_pair_ids[1],
            "pair_score": refinement.pair_score,
            "pair_feature_distance": refinement.pair_feature_distance,
            "pair_boundary_contact": refinement.pair_boundary_contact,
            "pair_union_pixels": refinement.pair_union_pixels,
            "local_region_pixels": refinement.local_region_pixels,
            "coarse_mask_positive_pixels_a": int(refinement.coarse_prompt_a.sum()),
            "coarse_mask_positive_pixels_b": int(refinement.coarse_prompt_b.sum()),
            "refined_score_a": refinement.refined_score_a,
            "refined_score_b": refinement.refined_score_b,
            "refined_mask_b_source": (
                "sam_refined" if refinement.selection_mode == "pair" else "complement_of_refined_a"
            ),
            "feature_mask_status": "ok",
            "label_permutation_invariant": True,
            "assignment_used": assignment.assignment_used,
            "assignment_source_for_texture_a": assignment.chosen_source_a,
            "assignment_source_for_texture_b": assignment.chosen_source_b,
            "assignment_direct_miou": assignment.direct_miou,
            "assignment_direct_ari": assignment.direct_ari,
            "assignment_swapped_miou": assignment.swapped_miou,
            "assignment_swapped_ari": assignment.swapped_ari,
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=assignment.chosen_miou,
                ari=assignment.chosen_ari,
                evaluation_view="partition_invariant",
            )
        )
        if refinement.selection_mode == "pair":
            selection_summary = (
                f"Mode=pair Pair={refinement.selected_pair_ids[0]}/{refinement.selected_pair_ids[1]}"
            )
        else:
            selection_summary = f"Mode=single Seed={refinement.selected_pair_ids[0]} Bg=local_complement"
        metric_summary = (
            f"Eval mIoU={assignment.chosen_miou:.3f} "
            f"Eval ARI={assignment.chosen_ari:.3f} "
            f"Aggr mIoU={metrics.aggregated_miou:.3f} "
            f"{selection_summary} "
            f"Score={refinement.pair_score:.3f}"
        )
        return Sam3AutoSampleResult(
            row=row,
            metric_summary=metric_summary,
            prediction_masks=(assignment.chosen_prediction_a, assignment.chosen_prediction_b),
            aggregated_prediction_a=aggregated_masks[0],
            aggregated_prediction_b=aggregated_masks[1],
            feature_mask_refinement=refinement,
        )

    if variant == "boundary_refine_sweep":
        if not hasattr(runner, "generate_feature_cluster_sweep"):
            raise RuntimeError("boundary_refine_sweep evaluation requires a runner with generate_feature_cluster_sweep().")

        sweep = runner.generate_feature_cluster_sweep(sample.image)
        baseline_output = sweep.variants["v0_no_refine"]
        baseline_view = _compute_unlabeled_partition_view(
            prediction_a=baseline_output.mask_a,
            prediction_b=baseline_output.mask_b,
            sample=sample,
            source_a="cluster_a",
            source_b="cluster_b",
        )
        baseline_assignment = baseline_view["assignment"]
        row = {
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "points_per_crop": 0,
            "stability_score_thresh": 0.0,
            "num_predicted_masks": 2,
            "evaluation_view": "partition_invariant",
            "miou": baseline_assignment.chosen_miou,
            "ari": baseline_assignment.chosen_ari,
            "miou_agg": float(
                np.mean([baseline_view["texture_a_agg_iou"], baseline_view["texture_b_agg_iou"]])
            ),
            "texture_a_best_iou": baseline_view["texture_a_iou"],
            "texture_b_best_iou": baseline_view["texture_b_iou"],
            "texture_a_agg_iou": baseline_view["texture_a_agg_iou"],
            "texture_b_agg_iou": baseline_view["texture_b_agg_iou"],
            "texture_a_overlap_mask_count": baseline_view["overlap_counts"][0],
            "texture_b_overlap_mask_count": baseline_view["overlap_counts"][1],
            "mask_score_mean": 0.0,
            "mask_score_median": 0.0,
            "boundary_refine_sweep_status": "ok",
            "label_permutation_invariant": True,
            "assignment_used": baseline_assignment.assignment_used,
            "assignment_source_for_texture_a": baseline_assignment.chosen_source_a,
            "assignment_source_for_texture_b": baseline_assignment.chosen_source_b,
            "assignment_direct_miou": baseline_assignment.direct_miou,
            "assignment_direct_ari": baseline_assignment.direct_ari,
            "assignment_swapped_miou": baseline_assignment.swapped_miou,
            "assignment_swapped_ari": baseline_assignment.swapped_ari,
            "num_feature_levels": len(sweep.level_names),
            "coarsest_level_name": sweep.level_names[0],
            "finest_level_name": sweep.level_names[-1],
            "feature_level_names": "|".join(sweep.level_names),
            "feature_level_resolutions": "|".join(
                f"{resolution[0]}x{resolution[1]}" for resolution in sweep.level_resolutions
            ),
            "boundary_touching_coarse_cell_pixels": int(sweep.boundary_cell_map.sum()),
            "boundary_touching_coarse_cell_fraction": float(sweep.boundary_cell_map.mean()),
            "coarsest_pool_kernel_size": sweep.coarsest_pool_kernel_size,
            "coarsest_pool_stride": sweep.coarsest_pool_stride,
            "pooled_grid_resolution": f"{sweep.pooled_grid_resolution[0]}x{sweep.pooled_grid_resolution[1]}",
            "output_stage": "boundary_refine_sweep",
            "coarsest_init_mode": "pooled_avg_pool_flip_avg_boundary_refine_sweep",
            "multiscale_refinement_applied": False,
            "sam_refinement_applied": False,
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=baseline_assignment.chosen_miou,
                ari=baseline_assignment.chosen_ari,
                evaluation_view="partition_invariant",
            )
        )
        best_variant_id = "v0_no_refine"
        best_tuple = (baseline_assignment.chosen_miou, -baseline_output.percent_pixels_changed)
        metric_pieces: list[str] = []
        for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
            variant_output = sweep.variants[variant_id]
            variant_view = _compute_unlabeled_partition_view(
                prediction_a=variant_output.mask_a,
                prediction_b=variant_output.mask_b,
                sample=sample,
                source_a="cluster_a",
                source_b="cluster_b",
            )
            variant_assignment = variant_view["assignment"]
            row[f"{variant_id}_assignment_used"] = variant_assignment.assignment_used
            row[f"{variant_id}_eval_miou"] = variant_assignment.chosen_miou
            row[f"{variant_id}_eval_ari"] = variant_assignment.chosen_ari
            row[f"{variant_id}_percent_pixels_changed"] = variant_output.percent_pixels_changed
            row[f"{variant_id}_num_changed_components"] = variant_output.num_changed_components
            row[f"{variant_id}_average_changed_margin"] = variant_output.average_changed_margin
            metric_pieces.append(f"{variant_id}={variant_assignment.chosen_miou:.3f}")
            rank_tuple = (variant_assignment.chosen_miou, -variant_output.percent_pixels_changed)
            if rank_tuple > best_tuple:
                best_variant_id = variant_id
                best_tuple = rank_tuple
        row["recommended_variant"] = best_variant_id
        row["recommended_variant_label"] = BOUNDARY_REFINE_SWEEP_VARIANT_LABELS[best_variant_id]
        return Sam3AutoSampleResult(
            row=row,
            metric_summary=f"boundary_refine_sweep | best={best_variant_id} | " + " ".join(metric_pieces),
            prediction_masks=(baseline_output.mask_a, baseline_output.mask_b),
            aggregated_prediction_a=baseline_view["aggregated_masks"][0],
            aggregated_prediction_b=baseline_view["aggregated_masks"][1],
            boundary_refine_sweep_refinement=sweep,
        )

    if variant in {
        "feature_cluster_coarse_to_fine_global",
        "feature_cluster_coarse_to_fine_global_pooled_init",
        "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only",
        "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
        "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only",
        "flip_avg_plus_edge_debias",
        "feature_cluster_coarse_to_fine_global_pooled_init_direct",
    }:
        if not hasattr(runner, "generate_feature_clusters"):
            raise RuntimeError(f"{variant} evaluation requires a runner with generate_feature_clusters().")

        refinement = runner.generate_feature_clusters(sample.image)
        active_view = _compute_unlabeled_partition_view(
            prediction_a=refinement.refined_mask_a,
            prediction_b=refinement.refined_mask_b,
            sample=sample,
            source_a="cluster_a",
            source_b="cluster_b",
        )
        assignment = active_view["assignment"]
        texture_a_iou = active_view["texture_a_iou"]
        texture_b_iou = active_view["texture_b_iou"]
        aggregated_masks = active_view["aggregated_masks"]
        overlap_counts = active_view["overlap_counts"]
        texture_a_agg_iou = active_view["texture_a_agg_iou"]
        texture_b_agg_iou = active_view["texture_b_agg_iou"]
        refined_scores = [
            score
            for score in (refinement.refined_score_a, refinement.refined_score_b)
            if refinement.sam_refinement_applied or score not in {0.0, None}
        ]
        status_field = f"{variant}_status"

        row = {
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "points_per_crop": 0,
            "stability_score_thresh": 0.0,
            "num_predicted_masks": 2,
            "evaluation_view": "partition_invariant",
            "miou": assignment.chosen_miou,
            "ari": assignment.chosen_ari,
            "miou_agg": float(np.mean([texture_a_agg_iou, texture_b_agg_iou])),
            "texture_a_best_iou": texture_a_iou,
            "texture_b_best_iou": texture_b_iou,
            "texture_a_agg_iou": texture_a_agg_iou,
            "texture_b_agg_iou": texture_b_agg_iou,
            "texture_a_overlap_mask_count": overlap_counts[0],
            "texture_b_overlap_mask_count": overlap_counts[1],
            "mask_score_mean": float(np.mean(refined_scores)) if refined_scores else 0.0,
            "mask_score_median": float(np.median(refined_scores)) if refined_scores else 0.0,
            status_field: "ok",
            "label_permutation_invariant": True,
            "assignment_used": assignment.assignment_used,
            "assignment_source_for_texture_a": assignment.chosen_source_a,
            "assignment_source_for_texture_b": assignment.chosen_source_b,
            "assignment_direct_miou": assignment.direct_miou,
            "assignment_direct_ari": assignment.direct_ari,
            "assignment_swapped_miou": assignment.swapped_miou,
            "assignment_swapped_ari": assignment.swapped_ari,
            "num_feature_levels": len(refinement.level_names),
            "coarsest_level_name": refinement.level_names[0],
            "finest_level_name": refinement.level_names[-1],
            "feature_level_names": "|".join(refinement.level_names),
            "feature_level_resolutions": "|".join(
                f"{resolution[0]}x{resolution[1]}" for resolution in refinement.level_resolutions
            ),
            "cluster_pixel_count_a": refinement.cluster_pixel_count_a,
            "cluster_pixel_count_b": refinement.cluster_pixel_count_b,
            "rough_mask_positive_pixels_a": int(refinement.rough_mask_a.sum()),
            "rough_mask_positive_pixels_b": int(refinement.rough_mask_b.sum()),
            "refined_mask_positive_pixels_a": int(refinement.refined_mask_a.sum()),
            "refined_mask_positive_pixels_b": int(refinement.refined_mask_b.sum()),
            "refined_score_a": refinement.refined_score_a,
            "refined_score_b": refinement.refined_score_b,
            "refined_pair_selection_score": refinement.refined_pair_selection_score,
            "refined_pair_overlap_iou": refinement.refined_pair_overlap_iou,
            "output_stage": (
                "coarsest_pooled_flip_avg_plus_edge_debias_direct"
                if variant == "flip_avg_plus_edge_debias"
                else "coarsest_pooled_projection_removed_direct"
                if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only"
                else "coarsest_pooled_direct"
                if not refinement.sam_refinement_applied and not refinement.multiscale_refinement_applied
                else "coarsest_prompt_refined"
                if refinement.sam_refinement_applied and not refinement.multiscale_refinement_applied
                else "refined"
            ),
            "coarsest_init_mode": refinement.coarsest_init_mode,
            "multiscale_refinement_applied": refinement.multiscale_refinement_applied,
            "sam_refinement_applied": refinement.sam_refinement_applied,
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=assignment.chosen_miou,
                ari=assignment.chosen_ari,
                evaluation_view="partition_invariant",
            )
        )
        if refinement.coarsest_pool_kernel_size is not None:
            row["coarsest_pool_kernel_size"] = refinement.coarsest_pool_kernel_size
        if refinement.coarsest_pool_stride is not None:
            row["coarsest_pool_stride"] = refinement.coarsest_pool_stride
        if refinement.pooled_grid_resolution is not None:
            row["pooled_grid_resolution"] = (
                f"{refinement.pooled_grid_resolution[0]}x{refinement.pooled_grid_resolution[1]}"
            )
        if refinement.coarsest_component_stats is not None:
            row.update(_flatten_component_stats("active_coarsest", refinement.coarsest_component_stats))
            row["active_coarsest_component_stats_json"] = json.dumps(
                refinement.coarsest_component_stats,
                sort_keys=True,
            )
        if refinement.prompt_candidates_a:
            row["prompt_top_candidate_a_prompt_iou"] = refinement.prompt_candidates_a[0]["prompt_iou"]
            row["prompt_top_candidate_a_score"] = refinement.prompt_candidates_a[0]["score"]
        if refinement.prompt_candidates_b:
            row["prompt_top_candidate_b_prompt_iou"] = refinement.prompt_candidates_b[0]["prompt_iou"]
            row["prompt_top_candidate_b_score"] = refinement.prompt_candidates_b[0]["score"]
        if refinement.selected_candidate_rank_a is not None:
            row["prompt_selected_candidate_rank_a"] = refinement.selected_candidate_rank_a
            row["prompt_selected_candidate_index_a"] = refinement.selected_candidate_index_a
        if refinement.selected_candidate_rank_b is not None:
            row["prompt_selected_candidate_rank_b"] = refinement.selected_candidate_rank_b
            row["prompt_selected_candidate_index_b"] = refinement.selected_candidate_index_b
        if refinement.comparison_baseline_alignment_used is not None:
            row["comparison_baseline_alignment_used"] = refinement.comparison_baseline_alignment_used
            row["comparison_baseline_direct_mean_iou"] = refinement.comparison_baseline_direct_mean_iou
            row["comparison_baseline_swapped_mean_iou"] = refinement.comparison_baseline_swapped_mean_iou
            row["comparison_baseline_chosen_mean_iou"] = refinement.comparison_baseline_chosen_mean_iou
        if refinement.comparison_baseline_prompt_candidates_a:
            row["comparison_baseline_prompt_top_candidate_a_prompt_iou"] = refinement.comparison_baseline_prompt_candidates_a[0]["prompt_iou"]
            row["comparison_baseline_prompt_top_candidate_a_score"] = refinement.comparison_baseline_prompt_candidates_a[0]["score"]
        if refinement.comparison_baseline_prompt_candidates_b:
            row["comparison_baseline_prompt_top_candidate_b_prompt_iou"] = refinement.comparison_baseline_prompt_candidates_b[0]["prompt_iou"]
            row["comparison_baseline_prompt_top_candidate_b_score"] = refinement.comparison_baseline_prompt_candidates_b[0]["score"]
        if refinement.comparison_baseline_selected_candidate_rank_a is not None:
            row["comparison_baseline_prompt_selected_candidate_rank_a"] = (
                refinement.comparison_baseline_selected_candidate_rank_a
            )
            row["comparison_baseline_prompt_selected_candidate_index_a"] = (
                refinement.comparison_baseline_selected_candidate_index_a
            )
        if refinement.comparison_baseline_selected_candidate_rank_b is not None:
            row["comparison_baseline_prompt_selected_candidate_rank_b"] = (
                refinement.comparison_baseline_selected_candidate_rank_b
            )
            row["comparison_baseline_prompt_selected_candidate_index_b"] = (
                refinement.comparison_baseline_selected_candidate_index_b
            )
        if refinement.comparison_baseline_component_stats is not None:
            row.update(
                _flatten_component_stats("baseline_coarsest", refinement.comparison_baseline_component_stats)
            )
            row["baseline_coarsest_component_stats_json"] = json.dumps(
                refinement.comparison_baseline_component_stats,
                sort_keys=True,
            )
            tiny_before = _sum_tiny_component_pixels(refinement.comparison_baseline_component_stats)
            tiny_after = _sum_tiny_component_pixels(refinement.coarsest_component_stats)
            row["coarsest_tiny_pixels_before"] = tiny_before
            row["coarsest_tiny_pixels_after"] = tiny_after
        if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
            row["label_positionality_index"] = (
                refinement.label_positionality_diagnostics or {}
            ).get("label_positionality_index")
            row["feature_positionality_index"] = (
                refinement.feature_positionality_diagnostics or {}
            ).get("feature_positionality_index")
            row["clustering_random_seed"] = None
            row["clustering_random_seed_used"] = "deterministic"
            row["clustering_reinitialized_after_coarsest"] = False
            row["coarsest_cluster_initialization"] = "deterministic_mean_farthest"
            _append_label_positionality_fields(row, "raw", refinement.label_positionality_diagnostics)
            _append_feature_positionality_fields(row, "raw", refinement.feature_positionality_diagnostics)
            _append_feature_positionality_fields(
                row,
                "projection_removed",
                refinement.projection_removed_feature_positionality_diagnostics,
            )
            _append_label_positionality_fields(row, "baseline_pooled", refinement.baseline_branch_diagnostics)
            _append_label_positionality_fields(row, "projection_removed", refinement.projection_removed_branch_diagnostics)
            _append_label_positionality_fields(row, "flip_avg", refinement.flip_avg_branch_diagnostics)
            if isinstance(refinement.flip_test_diagnostics, dict):
                row["flip_test_bias_interpretation"] = refinement.flip_test_diagnostics.get("bias_interpretation")
                row["flip_test_mean_unflipped_cluster_iou_to_original"] = refinement.flip_test_diagnostics.get(
                    "mean_unflipped_cluster_iou_to_original"
                )
                row["flip_test_max_unflipped_label_positionality_index"] = refinement.flip_test_diagnostics.get(
                    "max_unflipped_label_positionality_index"
                )
                for transform_name in ("hflip", "vflip"):
                    transform_diag = refinement.flip_test_diagnostics.get("transforms", {}).get(transform_name)
                    if isinstance(transform_diag, dict):
                        row[f"{transform_name}_unflipped_cluster_mean_iou_to_original"] = transform_diag.get(
                            "cluster_mean_iou_to_original"
                        )
                        row[f"{transform_name}_unflipped_label_positionality_index"] = transform_diag.get(
                            "label_positionality_index"
                        )
                        row[f"{transform_name}_unflipped_label_positionality_best_model"] = transform_diag.get(
                            "label_positionality_best_model"
                        )
            if isinstance(refinement.null_test_diagnostics, dict):
                for null_name in ("constant_gray", "weak_noise", "strong_blur"):
                    null_diag = refinement.null_test_diagnostics.get(null_name)
                    if isinstance(null_diag, dict):
                        row[f"{null_name}_label_positionality_index"] = null_diag.get("label_positionality_index")
                        row[f"{null_name}_label_positionality_best_model"] = null_diag.get("best_model_name")
                        row[f"{null_name}_axis_threshold_index"] = null_diag.get("axis_threshold_index")
                        row[f"{null_name}_axis_split_flag"] = null_diag.get("axis_split_flag")
            if refinement.baseline_branch_mask_a is not None and refinement.baseline_branch_mask_b is not None:
                baseline_view = _compute_unlabeled_partition_view(
                    prediction_a=refinement.baseline_branch_mask_a,
                    prediction_b=refinement.baseline_branch_mask_b,
                    sample=sample,
                    source_a="baseline_cluster_a",
                    source_b="baseline_cluster_b",
                )
                row["baseline_branch_eval_miou"] = baseline_view["assignment"].chosen_miou
                row["baseline_branch_eval_ari"] = baseline_view["assignment"].chosen_ari
            row["projection_removed_branch_eval_miou"] = assignment.chosen_miou
            row["projection_removed_branch_eval_ari"] = assignment.chosen_ari
            if refinement.flip_avg_mask_a is not None and refinement.flip_avg_mask_b is not None:
                flip_avg_view = _compute_unlabeled_partition_view(
                    prediction_a=refinement.flip_avg_mask_a,
                    prediction_b=refinement.flip_avg_mask_b,
                    sample=sample,
                    source_a="flip_avg_cluster_a",
                    source_b="flip_avg_cluster_b",
                )
                row["flip_avg_branch_eval_miou"] = flip_avg_view["assignment"].chosen_miou
                row["flip_avg_branch_eval_ari"] = flip_avg_view["assignment"].chosen_ari
            row["recommended_mitigation"] = refinement.recommended_mitigation
            row["recommendation_reason"] = refinement.recommendation_reason
        if variant == "flip_avg_plus_edge_debias":
            row["edge_positionality_index"] = (
                refinement.raw_edge_positionality_diagnostics or {}
            ).get("edge_positionality_index")
            row["feature_edge_positionality_index"] = (
                refinement.raw_feature_edge_positionality_diagnostics or {}
            ).get("feature_positionality_index")
            row["clustering_random_seed"] = None
            row["clustering_random_seed_used"] = "deterministic"
            row["clustering_reinitialized_after_coarsest"] = False
            row["coarsest_cluster_initialization"] = "deterministic_mean_farthest"
            _append_edge_label_positionality_fields(row, "raw", refinement.raw_edge_positionality_diagnostics)
            _append_edge_feature_positionality_fields(
                row,
                "raw",
                refinement.raw_feature_edge_positionality_diagnostics,
            )
            _append_edge_label_positionality_fields(
                row,
                "baseline_pooled",
                refinement.baseline_branch_edge_diagnostics,
            )
            _append_edge_label_positionality_fields(
                row,
                "flip_avg",
                refinement.flip_avg_branch_edge_diagnostics,
            )
            _append_edge_feature_positionality_fields(
                row,
                "flip_avg",
                refinement.flip_avg_feature_edge_positionality_diagnostics,
            )
            _append_edge_label_positionality_fields(
                row,
                "edge_debiased",
                refinement.edge_debiased_branch_edge_diagnostics,
            )
            _append_edge_feature_positionality_fields(
                row,
                "edge_debiased",
                refinement.edge_debiased_feature_edge_positionality_diagnostics,
            )
            _append_edge_label_positionality_fields(
                row,
                "flip_avg_plus_edge_debiased",
                refinement.flip_avg_plus_edge_debiased_branch_edge_diagnostics,
            )
            _append_edge_feature_positionality_fields(
                row,
                "flip_avg_plus_edge_debiased",
                refinement.flip_avg_plus_edge_debiased_feature_edge_positionality_diagnostics,
            )
            if refinement.baseline_branch_mask_a is not None and refinement.baseline_branch_mask_b is not None:
                baseline_view = _compute_unlabeled_partition_view(
                    prediction_a=refinement.baseline_branch_mask_a,
                    prediction_b=refinement.baseline_branch_mask_b,
                    sample=sample,
                    source_a="baseline_cluster_a",
                    source_b="baseline_cluster_b",
                )
                row["baseline_branch_eval_miou"] = baseline_view["assignment"].chosen_miou
                row["baseline_branch_eval_ari"] = baseline_view["assignment"].chosen_ari
            if refinement.flip_avg_mask_a is not None and refinement.flip_avg_mask_b is not None:
                flip_avg_view = _compute_unlabeled_partition_view(
                    prediction_a=refinement.flip_avg_mask_a,
                    prediction_b=refinement.flip_avg_mask_b,
                    sample=sample,
                    source_a="flip_avg_cluster_a",
                    source_b="flip_avg_cluster_b",
                )
                row["flip_avg_branch_eval_miou"] = flip_avg_view["assignment"].chosen_miou
                row["flip_avg_branch_eval_ari"] = flip_avg_view["assignment"].chosen_ari
            if refinement.edge_debiased_mask_a is not None and refinement.edge_debiased_mask_b is not None:
                edge_view = _compute_unlabeled_partition_view(
                    prediction_a=refinement.edge_debiased_mask_a,
                    prediction_b=refinement.edge_debiased_mask_b,
                    sample=sample,
                    source_a="edge_debiased_cluster_a",
                    source_b="edge_debiased_cluster_b",
                )
                row["edge_debiased_branch_eval_miou"] = edge_view["assignment"].chosen_miou
                row["edge_debiased_branch_eval_ari"] = edge_view["assignment"].chosen_ari
            row["flip_avg_plus_edge_debiased_branch_eval_miou"] = assignment.chosen_miou
            row["flip_avg_plus_edge_debiased_branch_eval_ari"] = assignment.chosen_ari
            row["recommended_mitigation"] = refinement.recommended_mitigation
            row["recommendation_reason"] = refinement.recommendation_reason
        metric_summary = (
            f"Eval mIoU={assignment.chosen_miou:.3f} "
            f"Eval ARI={assignment.chosen_ari:.3f} "
            f"Aggr mIoU={row['miou_agg']:.3f} "
            f"Stage={row['output_stage']} "
            f"Levels={len(refinement.level_names)} "
            f"PairOv={refinement.refined_pair_overlap_iou:.3f} "
            f"Assign={assignment.assignment_used}"
        )
        if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
            metric_summary = (
                f"{metric_summary} "
                f"RawPos={float(row.get('label_positionality_index', 0.0)):.3f} "
                f"BasePos={float(row.get('baseline_pooled_label_positionality_index', 0.0)):.3f} "
                f"ProjPos={float(row.get('projection_removed_label_positionality_index', 0.0)):.3f} "
                f"FlipPos={float(row.get('flip_avg_label_positionality_index', 0.0)):.3f}"
            )
        if variant == "flip_avg_plus_edge_debias":
            metric_summary = (
                f"{metric_summary} "
                f"RawEdge={float(row.get('raw_edge_positionality_index', 0.0)):.3f} "
                f"BaseEdge={float(row.get('baseline_pooled_edge_positionality_index', 0.0)):.3f} "
                f"FlipEdge={float(row.get('flip_avg_edge_positionality_index', 0.0)):.3f} "
                f"EdgeOnly={float(row.get('edge_debiased_edge_positionality_index', 0.0)):.3f} "
                f"Flip+Edge={float(row.get('flip_avg_plus_edge_debiased_edge_positionality_index', 0.0)):.3f}"
            )
        return Sam3AutoSampleResult(
            row=row,
            metric_summary=metric_summary,
            prediction_masks=(assignment.chosen_prediction_a, assignment.chosen_prediction_b),
            aggregated_prediction_a=aggregated_masks[0],
            aggregated_prediction_b=aggregated_masks[1],
            feature_cluster_coarse_to_fine_global_refinement=refinement,
        )

    predictions = runner.generate_masks(sample.image, variant=variant)
    prediction_masks = [prediction.segmentation for prediction in predictions]
    gt_regions = (sample.texture_a_mask, sample.texture_b_mask)
    metrics = compute_mask_set_metrics(prediction_masks, gt_regions)
    canonical_partition, aggregated_masks, overlap_counts = compute_canonical_partition_metrics_for_mask_set(
        prediction_masks,
        sample.texture_a_mask,
        sample.texture_b_mask,
    )

    scores = [prediction.score for prediction in predictions]
    preset = SAM3_AUTO_PRESETS[variant]
    row = {
        "variant": variant,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "points_per_crop": int(preset["points_per_crop"]),
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
        "mask_score_mean": float(np.mean(scores)) if scores else 0.0,
        "mask_score_median": float(np.median(scores)) if scores else 0.0,
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
    return Sam3AutoSampleResult(
        row=row,
        metric_summary=metric_summary,
        prediction_masks=tuple(prediction_masks),
        aggregated_prediction_a=aggregated_masks[0],
        aggregated_prediction_b=aggregated_masks[1],
    )


def select_best_binary_assignment(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    source_a: str,
    source_b: str,
) -> BinaryAssignmentSelection:
    """Pick the better GT alignment for two unlabeled binary predictions."""

    direct = compute_partition_metrics(prediction_a, prediction_b, target_a, target_b)
    swapped = compute_partition_metrics(prediction_b, prediction_a, target_a, target_b)
    direct_score = (direct.miou, direct.ari)
    swapped_score = (swapped.miou, swapped.ari)

    if swapped_score > direct_score:
        return BinaryAssignmentSelection(
            assignment_used=f"{source_b}->texture_a,{source_a}->texture_b",
            chosen_prediction_a=np.asarray(prediction_b, dtype=bool),
            chosen_prediction_b=np.asarray(prediction_a, dtype=bool),
            chosen_source_a=source_b,
            chosen_source_b=source_a,
            direct_miou=direct.miou,
            direct_ari=direct.ari,
            swapped_miou=swapped.miou,
            swapped_ari=swapped.ari,
            chosen_miou=swapped.miou,
            chosen_ari=swapped.ari,
        )

    return BinaryAssignmentSelection(
        assignment_used=f"{source_a}->texture_a,{source_b}->texture_b",
        chosen_prediction_a=np.asarray(prediction_a, dtype=bool),
        chosen_prediction_b=np.asarray(prediction_b, dtype=bool),
        chosen_source_a=source_a,
        chosen_source_b=source_b,
        direct_miou=direct.miou,
        direct_ari=direct.ari,
        swapped_miou=swapped.miou,
        swapped_ari=swapped.ari,
        chosen_miou=direct.miou,
        chosen_ari=direct.ari,
    )


def build_sam3_auto_variant_summary(
    variant: str,
    split: str,
    dataset_id: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    num_total_samples: int,
    full_dataset_num_examples: int,
    dataset_partition: DatasetPartitionSelection | None,
) -> dict[str, Any]:
    """Aggregate one SAM-3 automatic-mask variant over a dataset view."""

    cross_dataset_spec = (
        get_cross_dataset_experiment_spec(variant)
        if variant in CROSS_DATASET_EXPERIMENT_VARIANTS
        else None
    )
    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in SAM3_AUTO_SCALAR_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in SAM3_AUTO_SCALAR_FIELDS}
    summary = {
        "dataset_id": dataset_id,
        "split": split,
        "variant": variant,
        "model_id": model_id,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": str(rows[0].get("evaluation_view", "partition_invariant")),
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[CANONICAL_PRIMARY_METRIC],
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[CANONICAL_SECONDARY_METRIC],
        "primary_metric_reason": (
            "This repository now defaults to one ArchiTexture-style binary evaluator. "
            "Each variant maps its predictions into a canonical two-mask partition and "
            "reports that shared mIoU/ARI pair as the default comparison view."
        ),
        "num_total_samples": num_total_samples,
        "full_dataset_num_examples": int(full_dataset_num_examples),
        "num_evaluated_samples": len(rows),
        "num_failed_samples": len(failures),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }
    append_dataset_partition_fields(summary, dataset_partition, selected_sample_count=num_total_samples)
    if cross_dataset_spec is not None:
        summary["cross_dataset_registered"] = True
        summary["supported_dataset_ids"] = list(cross_dataset_spec.supported_datasets)
        summary["visual_contract"] = cross_dataset_spec.visual_contract
        summary["experiment_summary"] = cross_dataset_spec.summary
    if variant == "boundary_refine_sweep":
        sweep_summary = build_boundary_refine_sweep_summary(rows)
        summary["sweep_variant_metrics"] = sweep_summary["variant_metrics"]
        summary["recommended_variant"] = sweep_summary["recommended_variant"]
        summary["recommended_variant_reason"] = sweep_summary["recommended_variant_reason"]
        recommended_variant = sweep_summary["recommended_variant"]
        delta_field = f"{recommended_variant}_eval_miou"
        baseline_field = "v0_no_refine_eval_miou"
        ranked_rows = sorted(
            rows,
            key=lambda row: float(row[delta_field]) - float(row[baseline_field]),
            reverse=True,
        )
        summary["representative_examples"] = {
            "most_improved": [
                {
                    "crop_name": row["crop_name"],
                    "delta_eval_miou": float(row[delta_field]) - float(row[baseline_field]),
                }
                for row in ranked_rows[:3]
            ],
            "most_worsened": [
                {
                    "crop_name": row["crop_name"],
                    "delta_eval_miou": float(row[delta_field]) - float(row[baseline_field]),
                }
                for row in ranked_rows[-3:]
            ],
        }
    if variant == "mask_prompt_invariance_control":
        comparison_fields = (
            "raw_branch_miou",
            "pooled_branch_miou",
            "random_branch_miou",
            "raw_to_pooled_prompt_mean_iou",
            "raw_to_pooled_final_mean_iou",
            "random_to_pooled_prompt_mean_iou",
            "random_to_pooled_final_mean_iou",
            "raw_to_random_prompt_mean_iou",
            "raw_to_random_final_mean_iou",
        )
        summary["comparison_mean_metrics"] = {
            name: mean(float(row[name]) for row in rows)
            for name in comparison_fields
        }
        summary["comparison_median_metrics"] = {
            name: median(float(row[name]) for row in rows)
            for name in comparison_fields
        }
        summary["study_metric_name"] = "raw_to_pooled_final_mean_iou"
        summary["study_metric_value"] = summary["comparison_mean_metrics"]["raw_to_pooled_final_mean_iou"]
        summary["study_metric_reason"] = (
            "This side experiment is about SAM mask-prompt invariance: "
            "higher raw-to-pooled final IoU means dirty and cleaned prompts collapse to the same final partition."
        )
        summary["control_metric_name"] = "random_to_pooled_final_mean_iou"
        summary["control_metric_value"] = summary["comparison_mean_metrics"]["random_to_pooled_final_mean_iou"]
    if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
        comparison_fields = (
            "label_positionality_index",
            "feature_positionality_index",
            "raw_feature_positionality_index",
            "baseline_pooled_label_positionality_index",
            "projection_removed_label_positionality_index",
            "flip_avg_label_positionality_index",
            "projection_removed_feature_positionality_index",
            "baseline_branch_eval_miou",
            "baseline_branch_eval_ari",
            "projection_removed_branch_eval_miou",
            "projection_removed_branch_eval_ari",
            "flip_avg_branch_eval_miou",
            "flip_avg_branch_eval_ari",
            "hflip_unflipped_cluster_mean_iou_to_original",
            "vflip_unflipped_cluster_mean_iou_to_original",
            "constant_gray_label_positionality_index",
            "weak_noise_label_positionality_index",
            "strong_blur_label_positionality_index",
        )
        present_fields = tuple(
            field_name for field_name in comparison_fields if all(field_name in row and row[field_name] is not None for row in rows)
        )
        summary["comparison_mean_metrics"] = {
            name: mean(float(row[name]) for row in rows)
            for name in present_fields
        }
        summary["comparison_median_metrics"] = {
            name: median(float(row[name]) for row in rows)
            for name in present_fields
        }
        null_flag_fields = (
            "constant_gray_axis_split_flag",
            "weak_noise_axis_split_flag",
            "strong_blur_axis_split_flag",
        )
        summary["null_axis_split_flag_rates"] = {
            field_name: mean(1.0 if row.get(field_name) else 0.0 for row in rows)
            for field_name in null_flag_fields
        }
        projection_pos = summary["comparison_mean_metrics"].get("projection_removed_label_positionality_index")
        flip_pos = summary["comparison_mean_metrics"].get("flip_avg_label_positionality_index")
        baseline_pos = summary["comparison_mean_metrics"].get("baseline_pooled_label_positionality_index")
        projection_miou = summary["comparison_mean_metrics"].get("projection_removed_branch_eval_miou")
        flip_miou = summary["comparison_mean_metrics"].get("flip_avg_branch_eval_miou")
        baseline_miou = summary["comparison_mean_metrics"].get("baseline_branch_eval_miou")
        recommendation = "projection_removed"
        reason = "Projection removal is the direct de-biasing branch for this study."
        if None not in {projection_pos, flip_pos, baseline_pos, projection_miou, flip_miou, baseline_miou}:
            projection_tuple = (
                (baseline_pos - projection_pos),
                (projection_miou - baseline_miou),
            )
            flip_tuple = (
                (baseline_pos - flip_pos),
                (flip_miou - baseline_miou),
            )
            if flip_tuple[0] > projection_tuple[0] and flip_tuple[1] >= projection_tuple[1]:
                recommendation = "flip_averaged"
                reason = (
                    f"Flip averaging lowers mean pooled label positionality from {baseline_pos:.3f} to "
                    f"{flip_pos:.3f} while moving mean eval_miou from {baseline_miou:.3f} to {flip_miou:.3f}; "
                    f"projection removal reaches {projection_pos:.3f} / {projection_miou:.3f}."
                )
            else:
                recommendation = "projection_removed"
                reason = (
                    f"Projection removal lowers mean pooled label positionality from {baseline_pos:.3f} to "
                    f"{projection_pos:.3f} while moving mean eval_miou from {baseline_miou:.3f} to {projection_miou:.3f}; "
                    f"flip averaging reaches {flip_pos:.3f} / {flip_miou:.3f}."
                )
        summary["recommended_mitigation"] = recommendation
        summary["recommendation_reason"] = reason
        summary["study_metric_name"] = "projection_removed_label_positionality_index"
        summary["study_metric_value"] = projection_pos
    if variant == "flip_avg_plus_edge_debias":
        comparison_fields = (
            "raw_edge_positionality_index",
            "raw_feature_edge_positionality_index",
            "baseline_pooled_edge_positionality_index",
            "baseline_branch_eval_miou",
            "baseline_branch_eval_ari",
            "flip_avg_edge_positionality_index",
            "flip_avg_feature_edge_positionality_index",
            "flip_avg_branch_eval_miou",
            "flip_avg_branch_eval_ari",
            "edge_debiased_edge_positionality_index",
            "edge_debiased_feature_edge_positionality_index",
            "edge_debiased_branch_eval_miou",
            "edge_debiased_branch_eval_ari",
            "flip_avg_plus_edge_debiased_edge_positionality_index",
            "flip_avg_plus_edge_debiased_feature_edge_positionality_index",
            "flip_avg_plus_edge_debiased_branch_eval_miou",
            "flip_avg_plus_edge_debiased_branch_eval_ari",
        )
        present_fields = tuple(
            field_name
            for field_name in comparison_fields
            if all(field_name in row and row[field_name] is not None for row in rows)
        )
        summary["comparison_mean_metrics"] = {
            name: mean(float(row[name]) for row in rows)
            for name in present_fields
        }
        summary["comparison_median_metrics"] = {
            name: median(float(row[name]) for row in rows)
            for name in present_fields
        }
        active_edge_pos = summary["comparison_mean_metrics"].get("flip_avg_plus_edge_debiased_edge_positionality_index")
        baseline_edge_pos = summary["comparison_mean_metrics"].get("baseline_pooled_edge_positionality_index")
        flip_edge_pos = summary["comparison_mean_metrics"].get("flip_avg_edge_positionality_index")
        edge_only_pos = summary["comparison_mean_metrics"].get("edge_debiased_edge_positionality_index")
        active_miou = summary["comparison_mean_metrics"].get("flip_avg_plus_edge_debiased_branch_eval_miou")
        baseline_miou = summary["comparison_mean_metrics"].get("baseline_branch_eval_miou")
        flip_miou = summary["comparison_mean_metrics"].get("flip_avg_branch_eval_miou")
        edge_only_miou = summary["comparison_mean_metrics"].get("edge_debiased_branch_eval_miou")
        summary["recommended_mitigation"] = "flip_avg_plus_edge_debias"
        if None not in {
            baseline_edge_pos,
            flip_edge_pos,
            edge_only_pos,
            active_edge_pos,
            baseline_miou,
            flip_miou,
            edge_only_miou,
            active_miou,
        }:
            summary["recommendation_reason"] = (
                f"Mean edge positionality drops from {baseline_edge_pos:.3f} in the pooled baseline to "
                f"{flip_edge_pos:.3f} with flip averaging, {edge_only_pos:.3f} with edge-only de-biasing, "
                f"and {active_edge_pos:.3f} with the combined branch. The matching mean eval_mIoU values are "
                f"{baseline_miou:.3f}, {flip_miou:.3f}, {edge_only_miou:.3f}, and {active_miou:.3f}."
            )
        summary["study_metric_name"] = "flip_avg_plus_edge_debiased_edge_positionality_index"
        summary["study_metric_value"] = active_edge_pos
    return summary


def build_sam3_auto_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for one SAM-3 automatic-mask variant."""

    lines = [
        "# RWTD SAM-3 Automatic Mask Summary",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Split: `{summary['split']}`",
        f"- Variant: `{summary['variant']}`",
        (
            f"- Registered supported datasets: {', '.join(f'`{name}`' for name in summary.get('supported_dataset_ids', []))}"
            if summary.get("supported_dataset_ids")
            else "- Registered supported datasets: RWTD-only or not registered."
        ),
        f"- Model: `{summary['model_id']}`",
        f"- Evaluated samples: `{summary['num_evaluated_samples']}` / `{summary['num_total_samples']}`",
        (
            f"- Dataset partition: `{summary['dataset_partition']}` "
            f"(`{summary['dataset_partition_start_index']}:{summary['dataset_partition_end_index']}` of `{summary['full_dataset_num_examples']}` total items)"
            if summary.get("dataset_partition") is not None
            else f"- Dataset partition: full split order (`{summary['full_dataset_num_examples']}` total items)"
        ),
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
    for metric_name in SAM3_AUTO_SCALAR_FIELDS:
        lines.append(
            f"| `{metric_name}` | {summary['mean_metrics'][metric_name]:.6f} | "
            f"{summary['median_metrics'][metric_name]:.6f} |"
        )
    comparison_mean_metrics = summary.get("comparison_mean_metrics")
    comparison_median_metrics = summary.get("comparison_median_metrics")
    if isinstance(comparison_mean_metrics, dict) and isinstance(comparison_median_metrics, dict):
        lines.extend(["", "## Side Experiment Comparisons", "", "| Metric | Mean | Median |", "| --- | ---: | ---: |"])
        for metric_name, mean_value in comparison_mean_metrics.items():
            lines.append(
                f"| `{metric_name}` | {mean_value:.6f} | "
                f"{comparison_median_metrics[metric_name]:.6f} |"
            )
        study_metric_name = summary.get("study_metric_name")
        study_metric_value = summary.get("study_metric_value")
        if study_metric_name is not None and study_metric_value is not None:
            lines.extend(
                [
                    "",
                    f"- Study metric: `{study_metric_name}` = `{float(study_metric_value):.6f}`",
                ]
            )
        control_metric_name = summary.get("control_metric_name")
        control_metric_value = summary.get("control_metric_value")
        if control_metric_name is not None and control_metric_value is not None:
            lines.extend(
                [
                    "",
                    f"- Control metric: `{control_metric_name}` = `{float(control_metric_value):.6f}`",
                ]
            )
    sweep_variant_metrics = summary.get("sweep_variant_metrics")
    if isinstance(sweep_variant_metrics, dict):
        lines.extend(
            [
                "",
                "## Boundary Refine Sweep",
                "",
                "| Variant | Mean eval_mIoU | Mean eval_ARI | Mean % pixels changed | Improved vs V0 | Worsened vs V0 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
            metrics = sweep_variant_metrics[variant_id]
            lines.append(
                f"| `{variant_id}` | {metrics['eval_miou']:.6f} | {metrics['eval_ari']:.6f} | "
                f"{metrics['percent_pixels_changed']:.6f} | {metrics['improved_vs_v0_count']} | {metrics['worsened_vs_v0_count']} |"
            )
    recommended_mitigation = summary.get("recommended_mitigation")
    recommendation_reason = summary.get("recommendation_reason")
    if recommended_mitigation is not None:
        lines.extend(
            [
                "",
                f"- Recommended mitigation: `{recommended_mitigation}`",
            ]
        )
        if recommendation_reason is not None:
            lines.append(f"- Reason: {recommendation_reason}")
    if summary.get("recommended_variant") is not None:
        lines.extend(
            [
                "",
                f"- Recommended boundary-refine variant: `{summary['recommended_variant']}`",
            ]
        )
        if summary.get("recommended_variant_reason") is not None:
            lines.append(f"- Reason: {summary['recommended_variant_reason']}")
    null_axis_split_flag_rates = summary.get("null_axis_split_flag_rates")
    if isinstance(null_axis_split_flag_rates, dict):
        lines.extend(["", "## Null-Test Axis Split Rates", "", "| Flag | Mean |", "| --- | ---: |"])
        for field_name, mean_value in null_axis_split_flag_rates.items():
            lines.append(f"| `{field_name}` | {float(mean_value):.6f} |")
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['crop_name']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def build_sam3_auto_run_config(
    args,
    overview,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> dict[str, Any]:
    """Build a stable config payload for a SAM-3 automatic-mask run."""

    cross_dataset_spec = (
        get_cross_dataset_experiment_spec(args.variant)
        if args.variant in CROSS_DATASET_EXPERIMENT_VARIANTS
        else None
    )
    resolved_model_id = (
        resolve_cross_dataset_experiment_model_id(args.variant, args.model_id)
        if args.variant in CROSS_DATASET_EXPERIMENT_VARIANTS
        else args.model_id
    )
    config = {
        "command": getattr(args, "command", None),
        "dataset_id": args.dataset_id,
        "split": args.split,
        "variant": args.variant,
        "model_id": resolved_model_id,
        "device": args.device,
        "official_checkpoint_path": getattr(args, "official_checkpoint_path", None),
        "limit": getattr(args, "limit", None),
        "failure_policy": getattr(args, "failure_policy", "abort"),
        "save_visuals": args.save_visuals,
        "wandb": getattr(args, "wandb", False),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant_presets": SAM3_AUTO_PRESETS,
        "feature_mask_settings": FEATURE_MASK_SETTINGS,
        "feature_cluster_global_settings": FEATURE_CLUSTER_GLOBAL_SETTINGS,
        "feature_cluster_coarse_to_fine_global_settings": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS,
        "feature_cluster_coarse_to_fine_global_pooled_init_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_debiased_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_settings": SAM2_CFC_SETTINGS,
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_settings": SAM2_CFC_SETTINGS,
        "boundary_refine_sweep_settings": BOUNDARY_REFINE_SWEEP_SETTINGS,
        "flip_avg_plus_edge_debias_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS
        ),
        "mask_prompt_invariance_control_settings": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
        "versions": discovered_package_versions(),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
    }
    if overview is not None:
        config["discovered_split_size"] = overview.num_examples
        config["discovered_split_sizes"] = overview.split_sizes
        config["discovered_features"] = list(overview.features)
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    if cross_dataset_spec is not None:
        config["cross_dataset_registered"] = True
        config["supported_dataset_ids"] = list(cross_dataset_spec.supported_datasets)
        config["visual_contract"] = cross_dataset_spec.visual_contract
        config["experiment_summary"] = cross_dataset_spec.summary
        config["promotion_note"] = cross_dataset_spec.promotion_note
    return config


def build_sam3_auto_experiment_terms_markdown(
    args,
    *,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> str:
    """Render a standard protocol-spec document for SAM-3 automatic-mask runs."""

    executed_variants = resolve_sam3_auto_variants(args.variant)
    registered_variants = [name for name in executed_variants if name in CROSS_DATASET_EXPERIMENT_VARIANTS]
    lines = [
        "# SAM-3 Automatic Mask Comparison Terms",
        "",
        "## Run Scope",
        "",
        f"- Command family: `{getattr(args, 'command', 'sam3-auto')}`",
        f"- Dataset: `{args.dataset_id}`",
        f"- Split: `{args.split}`",
        f"- Requested variant setting: `{args.variant}`",
        f"- Executed variant(s): {', '.join(f'`{name}`' for name in executed_variants)}",
        f"- Model: `{args.model_id}`",
        f"- Device request: `{args.device}`",
        f"- Save visuals: `{getattr(args, 'save_visuals', True)}`",
    ]
    if hasattr(args, "limit"):
        lines.append(f"- Limit: `{getattr(args, 'limit', None)}`")
    lines.append(
        (
            f"- Dataset partition: `{dataset_partition.raw_spec}` -> indices `{dataset_partition.start_index}:{dataset_partition.end_index}` "
            f"({selected_sample_count} selected sample(s) from {dataset_partition.partition_size} partition items, full split size {dataset_partition.total_size})."
            if dataset_partition is not None
            else "- Dataset partition: full split order (no partitioning)."
        )
    )
    if hasattr(args, "failure_policy"):
        lines.append(f"- Failure policy: `{getattr(args, 'failure_policy', 'abort')}`")
    lines.extend(
        [
            f"- Official SAM checkpoint override: `{getattr(args, 'official_checkpoint_path', None)}`",
            "",
            "## Cross-Dataset Registration",
            "",
            "- Current-method SAM-3 feature experiments that are intended to run on every registered binary dataset must register once in `src/rwtd_sam3/eval/experiment_registry.py`.",
            "- Registered experiments must be documented in the root README before merge and become available through `eval-sam3-auto`, `eval-architexture-binary`, `eval-detexture-binary`, and `eval-cstd-binary` where those adapters apply.",
            (
                f"- Registered variant(s) in this run: {', '.join(f'`{name}`' for name in registered_variants)}."
                if registered_variants
                else "- Registered variant(s) in this run: none; this run is RWTD-only under the current contract."
            ),
            "",
            "## Core Terms",
            "",
            "- RWTD crop: one RGB image crop with two annotated texture regions (`texture_a_mask`, `texture_b_mask`) and one derived or stored boundary mask.",
            "- `all` dataset view: concatenates the public RWTD `train` and `test` splits in that order.",
            "- Raw predicted mask set: the unordered conceptual set of binary masks emitted by the active variant before region-to-GT assignment.",
            "- Rough mask: the pre-SAM-refinement binary mask produced directly by feature clustering or feature-margin thresholding.",
            "- Refined mask: the binary mask selected from the official Meta `sam3` mask-prompt path after feeding a rough mask as `input_masks` / mask prompt.",
            "- Permutation-invariant assignment: for unlabeled 2-cluster variants, both `(A->texture_a, B->texture_b)` and `(B->texture_a, A->texture_b)` are scored and the better one is reported.",
            "- Supplementary comparison track: these SAM-3 automatic-mask runs are not the TextureSAM paper baseline and should be read as a separate ablation track.",
            f"- Default evaluation contract: `{CANONICAL_EVALUATION_CONTRACT}`. Every variant now exports `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}` as the shared repo-wide comparison view.",
            "",
            "## Standard Per-Sample Row Fields",
            "",
            "- Common fields written for every successful SAM-3 auto sample: `variant`, `split`, `sample_index`, `crop_name`, `evaluation_view`, `eval_miou`, `eval_ari`, `num_predicted_masks`, `miou`, `ari`, `miou_agg`, `texture_a_best_iou`, `texture_b_best_iou`, `texture_a_agg_iou`, `texture_b_agg_iou`, `texture_a_overlap_mask_count`, `texture_b_overlap_mask_count`, `mask_score_mean`, and `mask_score_median`.",
            "- `mask_score_mean` / `mask_score_median`: arithmetic summaries of the retained SAM mask scores for the final raw mask set produced by the active variant.",
            "- Feature variants append additional protocol-specific fields such as selected support ids, assignment metadata, feature-level names, coarse-mask sizes, refined overlap diagnostics, and explicit status flags.",
            "",
            "## Variant Standard",
            "",
            "### `default`",
            "",
            "1. Use the Transformers `facebook/sam3` automatic mask-generation pipeline with `points_per_crop=32` and `stability_score_thresh=0.95`.",
            "2. Keep the returned raw mask set and score it directly with the repository automatic-mask metrics.",
            "3. Do not use text, points, boxes, proposals from other methods, or official Meta refinement.",
            "",
            "### `dense`",
            "",
            "1. Use the same Transformers automatic mask-generation path with `points_per_crop=64` and `stability_score_thresh=0.2`.",
            "2. Keep the denser raw mask set and score it directly with the repository automatic-mask metrics.",
            "3. Do not apply any additional feature-based refinement in this baseline.",
            "",
            "### `feature_mask`",
            "",
            "1. Run the existing `dense` automatic-mask generator once and cache its raw proposals.",
            "2. Extract official Meta `sam3` dense image features from the current image state.",
            "3. If at least two usable dense masks exist, score adjacent support pairs using feature distance plus boundary contact.",
            "4. If only one usable dense mask exists, synthesize a local complement region instead of silently falling back to `dense`.",
            "5. Mean-pool feature prototypes over eroded support regions, build a local feature-similarity margin, and threshold it into coarse mask prompts.",
            "6. Feed the coarse prompt(s) into the official Meta mask-prompt path and keep the final raw 2-mask partition for evaluation.",
            f"7. Current settings: base variant=`{FEATURE_MASK_SETTINGS['base_variant']}`, prototype erosion radius=`{FEATURE_MASK_SETTINGS['prototype_erosion_radius']}`, local dilation radius=`{FEATURE_MASK_SETTINGS['local_dilation_radius']}`, single-mask outer dilation radius=`{FEATURE_MASK_SETTINGS['single_mask_outer_dilation_radius']}`, min region pixels=`{FEATURE_MASK_SETTINGS['min_region_pixels']}`, preferred region pixels=`{FEATURE_MASK_SETTINGS['preferred_region_pixels']}`, min prototype pixels=`{FEATURE_MASK_SETTINGS['min_prototype_pixels']}`, min boundary contact=`{FEATURE_MASK_SETTINGS['min_boundary_contact']}`, margin threshold=`{FEATURE_MASK_SETTINGS['margin_threshold']}`, refine confidence threshold=`{FEATURE_MASK_SETTINGS['refine_confidence_threshold']}`.",
            "",
            "### `feature_cluster_global`",
            "",
            "1. Extract official Meta `sam3` dense image features from `backbone_fpn` and upsample the highest-resolution useful level to image resolution.",
            "2. L2-normalize the per-pixel feature vectors and run a global 2-way clustering over the full image with no appended spatial coordinates.",
            "3. Convert the binary label map into two disconnected-allowed rough masks over the whole image.",
            "4. Feed each rough mask independently into the official Meta mask-prompt path.",
            "5. Collect all candidate masks returned for prompt A and prompt B, then choose the final A/B pair jointly so prompt agreement is rewarded and A/B overlap is penalized.",
            "6. Evaluate the final refined pair permutation-invariantly against the two GT regions.",
            f"7. Current settings: feature source=`{FEATURE_CLUSTER_GLOBAL_SETTINGS['feature_source']}`, k-means max iterations=`{FEATURE_CLUSTER_GLOBAL_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_GLOBAL_SETTINGS['kmeans_convergence_tolerance']}`, apply refinement=`{FEATURE_CLUSTER_GLOBAL_SETTINGS['apply_refinement']}`, refine confidence threshold=`{FEATURE_CLUSTER_GLOBAL_SETTINGS['refine_confidence_threshold']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global`",
            "",
            "1. Extract the full official Meta `sam3` multiscale `backbone_fpn` feature pyramid and keep every level at its native spatial resolution.",
            "2. Order feature levels from coarsest to finest by spatial area.",
            "3. On the coarsest level only, flatten the feature grid, L2-normalize the vectors, and run a global 2-way clustering with no spatial coordinates.",
            "4. Upsample the current label map to the next finer level with nearest-neighbor interpolation.",
            "5. Build a local uncertainty band from 4-neighborhood label changes, dilate that band by a small radius, and freeze labels outside the band.",
            "6. Recompute cluster prototypes from the current labels and update only the uncertain pixels using finer-level cosine similarity to the two prototypes.",
            "7. Repeat the boundary-band refinement for every finer level, then convert the final label map into two rough masks.",
            "8. Feed both final rough masks into the official Meta mask-prompt path, collect prompt-aligned candidates, and choose the final refined A/B pair jointly with an overlap penalty.",
            "9. Evaluate the final refined pair permutation-invariantly against the two GT regions.",
            f"10. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['feature_source']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['kmeans_convergence_tolerance']}`, boundary band radius=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['boundary_band_radius']}`, refinement iterations per level=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['refinement_iterations_per_level']}`, refine confidence threshold=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS['refine_confidence_threshold']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global_pooled_init`",
            "",
            "1. Extract the same official Meta `sam3` multiscale `backbone_fpn` feature pyramid used by `feature_cluster_coarse_to_fine_global` and sort it from coarsest to finest by spatial area.",
            "2. Keep the original coarsest feature map only for diagnostics, then spatially average-pool that coarsest level before any clustering.",
            "3. L2-normalize the pooled coarsest features and run the same global 2-way clustering on that pooled grid with no spatial coordinates.",
            "4. Upsample the pooled label map back to the native coarsest feature resolution with nearest-neighbor interpolation.",
            "5. Reuse the unchanged boundary-band refinement logic on every finer feature level.",
            "6. Feed the final pooled-init rough masks into the official Meta mask-prompt path and choose the final refined A/B pair jointly with the same prompt-alignment and overlap-penalty logic.",
            "7. Evaluate the pooled-init refined pair permutation-invariantly against the two GT regions.",
            f"8. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_convergence_tolerance']}`, boundary band radius=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['boundary_band_radius']}`, refinement iterations per level=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['refinement_iterations_per_level']}`, refine confidence threshold=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['refine_confidence_threshold']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`",
            "",
            "1. Extract the same official Meta `sam3` `backbone_fpn` pyramid and sort feature levels from coarsest to finest by spatial area.",
            "2. Average-pool only the coarsest feature level before the initial global 2-way clustering.",
            "3. L2-normalize the pooled coarsest features, cluster the pooled grid globally into 2 unlabeled groups, and upsample that pooled label map back to the native coarsest resolution.",
            "4. Upsample the pooled coarsest partition directly to image resolution and treat those two coarse masks as the final raw prediction set.",
            "5. Do not run finer-level boundary-band refinement and do not send the coarse masks back through the official Meta SAM mask-prompt path in this ablation.",
            "6. Evaluate the unlabeled 2-mask partition permutation-invariantly against the two GT regions.",
            f"7. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_convergence_tolerance']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`",
            "",
            "1. Extract the raw coarsest official Meta `sam3` feature map and build four content-preserving views: `identity`, `hflip`, `vflip`, and `hvflip`.",
            "2. Undo each flipped coarsest feature map back into the original image coordinates and average the four restored coarsest feature tensors.",
            "3. L2-normalize that flip-averaged coarsest feature map, average-pool it on the coarsest level, and run the same global 2-way clustering on the pooled grid.",
            "4. Upsample the pooled label map back to the native coarsest resolution, then upsample that partition directly to image resolution.",
            "5. Treat the two image-space coarse masks as the final raw prediction set.",
            "6. Do not run finer-level boundary-band refinement and do not send the coarse masks through the official Meta SAM mask-prompt path in this ablation.",
            "7. Evaluate the unlabeled 2-mask partition permutation-invariantly against the two GT regions.",
            f"8. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['kmeans_convergence_tolerance']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only`",
            "",
            "1. Extract the raw coarsest official Meta `sam3` feature map before any spatial pooling and keep it as the diagnosis surface.",
            "2. Quantify raw label positionality by fitting coordinate-only separators to the raw 2-cluster map: vertical threshold, horizontal threshold, affine half-plane, and quadratic basis.",
            "3. Quantify raw feature positionality by regressing the flattened coarsest feature vectors on the normalized coordinate basis `[1, x, y, x^2, xy, y^2]` and reporting mean/max per-channel `R^2`.",
            "4. Run flip tests on the raw coarsest feature map using `identity`, `hflip`, `vflip`, and `hvflip`; unflip the resulting features back into the original image coordinates before comparing them.",
            "5. Run null-image controls on a constant gray image, a weak-noise image, and a strongly blurred image of the same size.",
            "6. Remove the coordinate subspace before clustering by projecting the raw coarsest feature vectors onto the coordinate basis and subtracting that projection, then L2-normalize the debiased features.",
            "7. Average-pool the debiased coarsest features, run the same global 2-way clustering, upsample back to the native coarsest resolution, and then upsample directly to image resolution.",
            "8. Do not run finer-level refinement and do not send the resulting masks into the official Meta SAM mask-prompt path in this debiasing ablation.",
            "9. Also run two comparison branches: the baseline pooled-init coarse-only branch and a flip-averaged feature branch built from `mean(untransform(E(T(I))))` over `{id, hflip, vflip, hvflip}`.",
            "10. Use the projection-removed branch as the active final prediction set, but export per-image and dataset-level comparisons for the baseline and flip-averaged branches as diagnostics.",
            f"11. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['kmeans_convergence_tolerance']}`, null noise seed=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['null_noise_seed']}`, null noise std=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['null_noise_std']}`, null blur radius=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS['null_blur_radius']}`.",
            "",
            "### `flip_avg_plus_edge_debias`",
            "",
            "1. This is a designated diagnosis-plus-mitigation side experiment focused on residual flip-symmetric border leakage.",
            "2. Extract the raw coarsest official Meta `sam3` feature map and use it only as the diagnosis surface, before any pooling or mitigation.",
            "3. Quantify raw label edge positionality by fitting only flip-symmetric coordinate models to the raw 2-cluster map: `1`, `x^2`, `y^2`, `x^2+y^2`, `x^2-y^2`, `d_edge`, `d_edge^2`, plus a joint linear separator over that same basis.",
            "4. Quantify raw feature-space edge bias by regressing the flattened coarsest feature vectors on the same flip-symmetric basis and reporting mean/max per-channel `R^2`.",
            "5. Build four pooled coarse-only branches for direct comparison: baseline pooled clustering, flip-only pooled clustering, edge-debias-only pooled clustering, and flip-averaged plus edge-debiased pooled clustering.",
            "6. The active branch first averages `identity`, `hflip`, `vflip`, and `hvflip` feature maps after unflipping them back into original coordinates, then projects out the flip-symmetric edge basis, L2-normalizes the residual features, average-pools them, and runs the same global 2-way clustering.",
            "7. Do not run finer-level refinement and do not send the resulting masks through the official Meta mask-prompt path in this side experiment; the goal is to isolate cluster-map positionality, not SAM prompt sensitivity.",
            "8. Export per-image comparisons for baseline, flip-only, edge-only, and flip-plus-edge branches, including both edge positionality metrics and downstream RWTD metrics.",
            f"9. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS['kmeans_convergence_tolerance']}`.",
            "",
            "### `feature_cluster_coarse_to_fine_global_pooled_init_direct`",
            "",
            "1. Extract the same official Meta `sam3` `backbone_fpn` pyramid and sort feature levels from coarsest to finest by spatial area.",
            "2. Average-pool only the coarsest feature level before the initial global 2-way clustering.",
            "3. L2-normalize the pooled features, cluster the pooled grid globally into 2 unlabeled groups, and upsample that pooled label map back to the native coarsest resolution.",
            "4. Upsample the coarsest label map directly to image resolution and use that rough 2-mask partition as the SAM mask prompt input.",
            "5. Do not run any finer-level boundary-band refinement in this ablation.",
            "6. Still run the official Meta mask-prompt refinement path once per coarsest cluster mask and choose the final refined A/B pair jointly.",
            "7. Evaluate the final unlabeled 2-mask partition permutation-invariantly against the two GT regions.",
            f"8. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_convergence_tolerance']}`.",
            "",
            "### `boundary_refine_sweep`",
            "",
            "1. Start from the existing strong coarse segmentation path: flip-average the coarsest SAM features across `identity`, `hflip`, `vflip`, and `hvflip`, average-pool the coarsest level, run the pooled global 2-way clustering, and upsample that pooled partition back to image resolution.",
            "2. Keep the coarse partition fixed as V0 and identify only the coarse cells that touch a 4-neighbor of the opposite label.",
            "3. Freeze the confident interior regions and use them as the anchored prototype supports for every refinement branch.",
            "4. V1 `refine_global_anchored`: open each boundary-touching coarse cell on the next finer feature level, classify its finer subcells by cosine similarity to the global anchored prototypes, and aggregate that local update back into the image-space partition.",
            "5. V2 `refine_global_anchored_margin`: same as V1, but only accept finer-level reassignments whose cosine-margin magnitude exceeds the configured threshold; ambiguous subcells keep the original coarse label.",
            "6. V3 `refine_global_local_blend_margin`: same as V2, but blend the global anchored prototypes with nearby local confident supports on each side of the current boundary.",
            "7. V4 `refine_two_level_global_margin`: apply the conservative anchored margin-gated update first on the next finer level and then once more on the finest level, still only inside the current boundary band.",
            "8. Do not re-cluster globally, do not refine the full image, do not update prototypes from uncertain boundary pixels, and do not run morphology, graph cuts, CRFs, or connected-component cleanup.",
            "9. Export the baseline coarse partition, the eligible boundary-cell map, every variant output, and per-variant change statistics, then score all five variants against the same GT regions.",
            f"10. Current settings: margin threshold=`{BOUNDARY_REFINE_SWEEP_SETTINGS['boundary_refine_margin_threshold']}`, local blend alpha=`{BOUNDARY_REFINE_SWEEP_SETTINGS['boundary_refine_local_blend_alpha']}`, interior exclusion radius=`{BOUNDARY_REFINE_SWEEP_SETTINGS['boundary_refine_interior_exclusion_radius']}`, local support radius cells=`{BOUNDARY_REFINE_SWEEP_SETTINGS['boundary_refine_local_support_radius_cells']}`, pooled init kernel=`{BOUNDARY_REFINE_SWEEP_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{BOUNDARY_REFINE_SWEEP_SETTINGS['coarsest_init_pool_stride']}`.",
            "",
            "### `mask_prompt_invariance_control`",
            "",
            "1. This is a designated side experiment, not a mainline benchmark baseline.",
            "2. Extract the same coarsest `backbone_fpn` feature level used by the pooled-init ablations.",
            "3. Build three prompt branches that all go through the same official Meta SAM mask-prompt refinement path: a dirty raw coarsest prompt, a cleaner pooled coarsest prompt, and a deterministic random prompt control with the same label balance as the pooled prompt.",
            "4. Do not run finer-level feature refinement in this side experiment; the point is to isolate prompt cleanliness, not multiscale updating.",
            "5. Refine all three prompt partitions through SAM, align raw/random final partitions to the pooled branch label order for comparison, and keep the pooled branch as the canonical branch for the standard automatic-mask fields.",
            "6. Report the side-experiment comparisons that matter: dirty-vs-clean prompt similarity, dirty-vs-clean final similarity, random-vs-clean prompt similarity, random-vs-clean final similarity, and per-branch GT scores.",
            "7. Save visuals that show the three prompt partitions directly beside the three final SAM outputs so prompt insensitivity is visible at a glance.",
            f"8. Current settings: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_convergence_tolerance']}`.",
            "",
            "### `both`",
            "",
            "1. `both` is intentionally narrow in this repository.",
            "2. It executes only `default` and `dense` and writes separate subdirectories for those two legacy automatic-mask baselines.",
            "3. It does not automatically include `feature_mask`, `feature_cluster_global`, `feature_cluster_coarse_to_fine_global`, `feature_cluster_coarse_to_fine_global_pooled_init`, `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only`, `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only`, `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only`, `flip_avg_plus_edge_debias`, `feature_cluster_coarse_to_fine_global_pooled_init_direct`, or `mask_prompt_invariance_control`.",
            "",
            "## Metric Standard",
            "",
            f"- `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}`: the repo-wide default ArchiTexture-style evaluator. Raw-mask variants first collapse into a GT-overlap aggregated binary partition; unlabeled 2-mask variants use permutation-invariant assignment and then score that binary partition.",
            "- `miou`: non-aggregated mean IoU from one-to-one IoU matching between the raw predicted mask set and the two GT texture regions. Higher is better. Range `[0, 1]`.",
            "- `ari`: Adjusted Rand Index between GT region labels and the pixelwise partition induced by raw predicted-mask memberships. Higher is better. Range `[-1, 1]`, with practical values usually in `[0, 1]` here.",
            "- `miou_agg`: aggregated mean IoU after unioning every raw predicted mask with non-empty overlap against each GT region. Higher is better. Range `[0, 1]`.",
            "- `texture_a_overlap_mask_count` / `texture_b_overlap_mask_count`: number of raw masks contributing to the aggregated texture-A / texture-B region.",
            "- `assignment_direct_*` and `assignment_swapped_*`: unlabeled 2-cluster diagnostics reported for the feature-cluster variants before the final permutation-invariant choice is made.",
            "- `refined_pair_overlap_iou`: IoU overlap between the final selected refined A and refined B masks for feature-cluster variants. Lower is better; values near `1` indicate collapse onto the same region.",
            "",
            "## Visual Artifact Standard",
            "",
            "- `prediction.json`: one per-sample JSON object for `predict-sam3-auto` containing the exact per-sample row written by the variant.",
            "- `prediction.png`: one per-sample visual panel for `predict-sam3-auto`.",
            "- `per_sample_metrics.csv`: one flattened row per evaluated sample for `eval-sam3-auto`.",
            "- `summary.json` / `summary.md`: aggregate metrics, failure counts, and primary metric selection for the current run.",
            "- `visuals_manifest.jsonl`: one machine-readable record per saved PNG with the full wrapped footer text and per-sample metric fields.",
            "- Default / dense visuals: input, GT, raw categorical proposals, and aggregated A/B view.",
            "- `feature_mask` visuals: input, GT, dense proposals, selected support pair, coarse feature prior, and final raw A/B partition.",
            "- `feature_cluster_global` visuals: input, GT, global 2-cluster map, rough masks, refined masks, and chosen permutation-aligned result.",
            "- `feature_cluster_coarse_to_fine_global` visuals: input, GT, coarsest cluster map, one panel per refined feature level, final rough partition before SAM, refined masks, and chosen permutation-aligned result.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init` visuals: input, GT, pooled coarsest cluster map, one panel per refined feature level, final rough partition before SAM, refined cluster A/B masks, and the chosen permutation-aligned result.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` visuals: input, GT, and the pooled coarsest partition only; the coarsest partition is already the final output in this ablation.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` visuals: input, GT, and the flip-averaged pooled coarsest partition only; that coarsest partition is already the final output in this ablation.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only` visuals: input, GT, raw coarsest cluster map, baseline pooled partition, projection-removed pooled partition, flip-averaged pooled partition, and the three null-image control maps.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_direct` visuals: input, GT, pooled coarsest cluster map, final rough partition before SAM, SAM-refined cluster A/B masks, and the chosen permutation-aligned result.",
            "- `boundary_refine_sweep` visuals: input, GT, the fixed coarsest partition, the eligible boundary-touching coarse cells, and one output panel per refinement variant V0-V4 with percent-changed annotations.",
            "- `flip_avg_plus_edge_debias` visuals: input, GT, raw coarsest cluster map, baseline pooled partition, flip-only pooled partition, edge-debias-only pooled partition, and the active flip-plus-edge-debiased pooled partition.",
            "- `mask_prompt_invariance_control` visuals: input, GT, dirty raw prompt partition, cleaner pooled prompt partition, deterministic random prompt control, and the three corresponding final SAM outputs with raw/random finals aligned to the pooled label order for comparison.",
            "",
            "## Failure Semantics",
            "",
            "- No SAM-feature variant silently falls back to another protocol when a required intermediate state is invalid.",
            "- `feature_mask` raises explicit failures when it cannot build a valid support pair or complement, coarse prompt, or refined mask; on `predict-sam3-auto` it may still save dense-only diagnostics when the dense proposals exist.",
            "- `feature_cluster_global` raises explicit failures when feature extraction, global clustering, prompt refinement, or joint A/B selection fails.",
            "- `feature_cluster_coarse_to_fine_global` raises explicit failures when multiscale features are missing, coarsest clustering collapses, finer-level prototype refinement cannot proceed, or final SAM refinement fails.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init` raises explicit failures when coarsest pooling is invalid, pooled coarsest clustering collapses, the reused finer-level refinement cannot proceed, or final SAM refinement fails.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only` raises explicit failures when coarsest pooling is invalid or the pooled coarsest partition collapses into an empty binary output.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only` raises explicit failures when flipped coarsest feature extraction returns incompatible shapes, pooled clustering collapses after flip averaging, or the final coarse binary partition becomes empty.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only` raises explicit failures when raw coarsest features are missing, coordinate projection removal becomes degenerate, any diagnosis branch collapses into an empty binary output, or the null/flip control extractions return incompatible coarsest shapes.",
            "- `feature_cluster_coarse_to_fine_global_pooled_init_direct` raises explicit failures when coarsest pooling is invalid or the pooled coarsest prompt cannot produce a valid prompt-refined binary output.",
            "- `boundary_refine_sweep` raises explicit failures when the fixed coarse partition is missing, the boundary-cell map cannot be constructed, an anchored prototype cannot be computed from confident interiors, or any requested finer feature level is missing or inconsistent.",
            "- `flip_avg_plus_edge_debias` raises explicit failures when the raw coarsest feature map is missing, flip-averaged features cannot be restored to a common shape, the symmetric edge-basis projection becomes degenerate, or any comparison branch collapses into an empty binary output.",
            "- `mask_prompt_invariance_control` raises explicit failures when any of the three prompt branches cannot be constructed or any of the shared SAM prompt-refinement passes fails; there is no fallback that drops the random control or replaces a failed branch.",
            "- `failure_policy=skip` applies only to evaluation commands and records the failed sample in the run summary instead of aborting immediately.",
            "",
            "## Backend And Dependency Assumptions",
            "",
            "- `default` and `dense` use the Transformers `facebook/sam3` automatic mask-generation path.",
            "- The SAM-feature variants rely on the official Meta `sam3` image backend because this repository needs access to the official multiscale image features and, for the prompt-based variants, the mask-prompt refinement hook exposed there.",
            "- Missing official `sam3` imports, missing gated model access, invalid checkpoints, or unexpected tensor shapes are treated as explicit runtime failures, not hidden fallbacks.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_sam3_auto_visual_record(
    sample: RwtdSample,
    evaluation: Sam3AutoSampleResult,
    variant: str,
    visual_path: Path,
) -> dict[str, Any]:
    """Build one machine-readable visual metadata record for the SAM-3 comparison."""

    cross_dataset_spec = (
        get_cross_dataset_experiment_spec(variant)
        if variant in CROSS_DATASET_EXPERIMENT_VARIANTS
        else None
    )
    record = dict(evaluation.row)
    record.update(
        {
            "visual_path": str(visual_path),
            "caption": build_visual_caption(
                sample=sample,
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=sample,
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            ),
            "original_texture_a": sample.original_texture_a,
            "original_texture_b": sample.original_texture_b,
            "oracle_points_a_count": len(sample.oracle_points_a),
            "oracle_points_b_count": len(sample.oracle_points_b),
        }
    )
    if cross_dataset_spec is not None:
        record["visual_contract"] = cross_dataset_spec.visual_contract
        record["supported_dataset_ids"] = list(cross_dataset_spec.supported_datasets)
    return record


def build_sam3_auto_failure_row(
    sample: RwtdSample,
    variant: str,
    error: Exception,
) -> dict[str, Any]:
    """Build a stable failure record for feature-mask predict/eval outputs."""

    row: dict[str, Any] = {
        "variant": variant,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "error": str(error),
    }
    if variant == "feature_mask":
        row["feature_mask_status"] = "failed"
    elif variant == "feature_cluster_global":
        row["feature_cluster_global_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global":
        row["feature_cluster_coarse_to_fine_global_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init":
        row["feature_cluster_coarse_to_fine_global_pooled_init_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only":
        row["feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2":
        row["feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2":
        row["feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only":
        row["feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_status"] = "failed"
    elif variant == "boundary_refine_sweep":
        row["boundary_refine_sweep_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
        row["feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only_status"] = "failed"
    elif variant == "flip_avg_plus_edge_debias":
        row["flip_avg_plus_edge_debias_status"] = "failed"
    elif variant == "feature_cluster_coarse_to_fine_global_pooled_init_direct":
        row["feature_cluster_coarse_to_fine_global_pooled_init_direct_status"] = "failed"
    elif variant == "mask_prompt_invariance_control":
        row["mask_prompt_invariance_control_status"] = "failed"
    else:
        row["status"] = "error"
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, dict):
        row.update(diagnostics)
    return row


def enrich_feature_mask_failure_row(
    sample: RwtdSample,
    failure_row: dict[str, Any],
    error: Exception,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray], str]:
    """Attach dense-baseline diagnostics to a feature-mask failure row when possible."""

    row = dict(failure_row)
    zero_mask = np.zeros_like(sample.texture_a_mask, dtype=bool)
    default_metric_summary = f"FAILED before feature_mask metric computation: {error}"
    if not isinstance(error, Sam3FeatureMaskRuntimeError) or not error.dense_prediction_masks:
        row.setdefault("refined_output_available", False)
        return row, (zero_mask, zero_mask), default_metric_summary

    gt_regions = (sample.texture_a_mask, sample.texture_b_mask)
    dense_metrics = compute_mask_set_metrics(error.dense_prediction_masks, gt_regions)
    dense_aggregated_masks, dense_overlap_counts = aggregate_masks_by_regions(
        error.dense_prediction_masks,
        gt_regions,
    )
    dense_scores = tuple(float(score) for score in error.dense_prediction_scores)
    row.update(
        {
            "refined_output_available": False,
            "dense_num_predicted_masks": len(error.dense_prediction_masks),
            "dense_mask_score_mean": float(np.mean(dense_scores)) if dense_scores else 0.0,
            "dense_mask_score_median": float(np.median(dense_scores)) if dense_scores else 0.0,
            "dense_miou": dense_metrics.miou,
            "dense_ari": dense_metrics.ari,
            "dense_miou_agg": dense_metrics.aggregated_miou,
            "dense_texture_a_best_iou": dense_metrics.region_best_ious[0],
            "dense_texture_b_best_iou": dense_metrics.region_best_ious[1],
            "dense_texture_a_agg_iou": dense_metrics.region_aggregated_ious[0],
            "dense_texture_b_agg_iou": dense_metrics.region_aggregated_ious[1],
            "dense_texture_a_overlap_mask_count": dense_overlap_counts[0],
            "dense_texture_b_overlap_mask_count": dense_overlap_counts[1],
        }
    )
    metric_summary = (
        f"FAILED before feature_mask metric computation: {error} | "
        f"Dense-only mIoU={dense_metrics.miou:.3f} "
        f"ARI={dense_metrics.ari:.3f} "
        f"Aggr mIoU={dense_metrics.aggregated_miou:.3f}"
    )
    return row, dense_aggregated_masks, metric_summary


def build_sam3_auto_failure_visual_record(
    sample: RwtdSample,
    variant: str,
    failure_row: dict[str, Any],
    visual_path: Path,
    failure_message: str,
    metric_summary: str | None = None,
) -> dict[str, Any]:
    """Build a manifest record for a saved feature-mask failure visual."""

    resolved_metric_summary = metric_summary or f"FAILED: {failure_message}"
    record = dict(failure_row)
    record.update(
        {
            "visual_path": str(visual_path),
            "caption": build_visual_caption(
                sample=sample,
                protocol=f"sam3_auto:{variant}:failed",
                metric_summary=resolved_metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=sample,
                protocol=f"sam3_auto:{variant}:failed",
                metric_summary=resolved_metric_summary,
            ),
            "original_texture_a": sample.original_texture_a,
            "original_texture_b": sample.original_texture_b,
            "oracle_points_a_count": len(sample.oracle_points_a),
            "oracle_points_b_count": len(sample.oracle_points_b),
        }
    )
    return record


def save_sam3_auto_panel(
    output_path: str | Path,
    sample: RwtdSample,
    evaluation: Sam3AutoSampleResult,
    variant: str,
) -> Path:
    """Save the repo-native panel for the selected SAM-3 automatic-mask variant."""

    if (
        evaluation.feature_mask_refinement is None
        and evaluation.feature_cluster_global_refinement is None
        and evaluation.feature_cluster_coarse_to_fine_global_refinement is None
        and evaluation.mask_prompt_invariance_refinement is None
        and evaluation.boundary_refine_sweep_refinement is None
    ):
        return save_automatic_mask_panel(
            output_path=output_path,
            sample=sample,
            prediction_masks=evaluation.prediction_masks,
            protocol=f"sam3_auto:{variant}",
            aggregated_prediction_a=evaluation.aggregated_prediction_a,
            aggregated_prediction_b=evaluation.aggregated_prediction_b,
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.feature_cluster_global_refinement is not None:
        return save_feature_cluster_global_panel(
            output_path=output_path,
            sample=sample,
            cluster_label_map=evaluation.feature_cluster_global_refinement.cluster_label_map,
            rough_mask_a=evaluation.feature_cluster_global_refinement.rough_mask_a,
            rough_mask_b=evaluation.feature_cluster_global_refinement.rough_mask_b,
            refinement_applied=evaluation.feature_cluster_global_refinement.refinement_applied,
            refined_mask_a=evaluation.feature_cluster_global_refinement.refined_mask_a,
            refined_mask_b=evaluation.feature_cluster_global_refinement.refined_mask_b,
            chosen_prediction_a=evaluation.prediction_masks[0],
            chosen_prediction_b=evaluation.prediction_masks[1],
            assignment_used=str(evaluation.row.get("assignment_used", "")),
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.mask_prompt_invariance_refinement is not None:
        return save_mask_prompt_invariance_panel(
            output_path=output_path,
            sample=sample,
            raw_prompt_a=evaluation.mask_prompt_invariance_refinement.raw_branch.prompt_mask_a,
            raw_prompt_b=evaluation.mask_prompt_invariance_refinement.raw_branch.prompt_mask_b,
            pooled_prompt_a=evaluation.mask_prompt_invariance_refinement.pooled_branch.prompt_mask_a,
            pooled_prompt_b=evaluation.mask_prompt_invariance_refinement.pooled_branch.prompt_mask_b,
            random_prompt_a=evaluation.mask_prompt_invariance_refinement.random_branch.prompt_mask_a,
            random_prompt_b=evaluation.mask_prompt_invariance_refinement.random_branch.prompt_mask_b,
            raw_final_a=evaluation.mask_prompt_invariance_refinement.raw_to_pooled_aligned_final_mask_a,
            raw_final_b=evaluation.mask_prompt_invariance_refinement.raw_to_pooled_aligned_final_mask_b,
            pooled_final_a=evaluation.mask_prompt_invariance_refinement.pooled_branch.final_mask_a,
            pooled_final_b=evaluation.mask_prompt_invariance_refinement.pooled_branch.final_mask_b,
            random_final_a=evaluation.mask_prompt_invariance_refinement.random_to_pooled_aligned_final_mask_a,
            random_final_b=evaluation.mask_prompt_invariance_refinement.random_to_pooled_aligned_final_mask_b,
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.boundary_refine_sweep_refinement is not None:
        return save_boundary_refine_sweep_panel(
            output_path=output_path,
            sample=sample,
            sweep=evaluation.boundary_refine_sweep_refinement,
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.feature_cluster_coarse_to_fine_global_refinement is not None:
        if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
            return save_feature_cluster_positionality_panel(
                output_path=output_path,
                sample=sample,
                raw_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.raw_diagnostic_label_map,
                baseline_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.baseline_branch_label_map,
                projection_removed_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.projection_removed_label_map
                ),
                flip_avg_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_label_map,
                null_constant_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.null_constant_label_map
                ),
                null_noise_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.null_noise_label_map,
                null_blur_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.null_blur_label_map,
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            )
        if variant == "flip_avg_plus_edge_debias":
            return save_feature_cluster_edge_bias_panel(
                output_path=output_path,
                sample=sample,
                raw_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.raw_diagnostic_label_map,
                baseline_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.baseline_branch_label_map,
                flip_avg_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_label_map,
                edge_debiased_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.edge_debiased_label_map,
                flip_avg_plus_edge_debiased_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_plus_edge_debiased_label_map
                ),
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            )
        return save_feature_cluster_coarse_to_fine_global_panel(
            output_path=output_path,
            sample=sample,
            level_label_maps=evaluation.feature_cluster_coarse_to_fine_global_refinement.level_label_maps,
            level_names=evaluation.feature_cluster_coarse_to_fine_global_refinement.level_names,
            rough_mask_a=evaluation.feature_cluster_coarse_to_fine_global_refinement.rough_mask_a,
            rough_mask_b=evaluation.feature_cluster_coarse_to_fine_global_refinement.rough_mask_b,
            refined_mask_a=evaluation.feature_cluster_coarse_to_fine_global_refinement.refined_mask_a,
            refined_mask_b=evaluation.feature_cluster_coarse_to_fine_global_refinement.refined_mask_b,
            chosen_prediction_a=evaluation.prediction_masks[0],
            chosen_prediction_b=evaluation.prediction_masks[1],
            assignment_used=str(evaluation.row.get("assignment_used", "")),
            multiscale_refinement_applied=(
                evaluation.feature_cluster_coarse_to_fine_global_refinement.multiscale_refinement_applied
            ),
            sam_refinement_applied=(
                evaluation.feature_cluster_coarse_to_fine_global_refinement.sam_refinement_applied
            ),
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    return save_feature_mask_panel(
        output_path=output_path,
        sample=sample,
        dense_prediction_masks=evaluation.feature_mask_refinement.dense_prediction_masks,
        selection_mode=evaluation.feature_mask_refinement.selection_mode,
        selected_pair_masks=evaluation.feature_mask_refinement.selected_pair_masks,
        selected_pair_ids=evaluation.feature_mask_refinement.selected_pair_ids,
        coarse_margin=evaluation.feature_mask_refinement.coarse_margin,
        final_prediction_a=evaluation.prediction_masks[0],
        final_prediction_b=evaluation.prediction_masks[1],
        protocol=f"sam3_auto:{variant}",
        metric_summary=evaluation.metric_summary,
    )


def render_sam3_auto_panel(
    sample: RwtdSample,
    evaluation: Sam3AutoSampleResult,
    variant: str,
):
    """Render the repo-native panel for previews and artifact export."""

    if (
        evaluation.feature_mask_refinement is None
        and evaluation.feature_cluster_global_refinement is None
        and evaluation.feature_cluster_coarse_to_fine_global_refinement is None
        and evaluation.boundary_refine_sweep_refinement is None
        and evaluation.mask_prompt_invariance_refinement is None
    ):
        return render_automatic_mask_panel(
            sample=sample,
            prediction_masks=evaluation.prediction_masks,
            protocol=f"sam3_auto:{variant}",
            aggregated_prediction_a=evaluation.aggregated_prediction_a,
            aggregated_prediction_b=evaluation.aggregated_prediction_b,
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.feature_cluster_global_refinement is not None:
        return render_feature_cluster_global_panel(
            sample=sample,
            cluster_label_map=evaluation.feature_cluster_global_refinement.cluster_label_map,
            rough_mask_a=evaluation.feature_cluster_global_refinement.rough_mask_a,
            rough_mask_b=evaluation.feature_cluster_global_refinement.rough_mask_b,
            refinement_applied=evaluation.feature_cluster_global_refinement.refinement_applied,
            refined_mask_a=evaluation.feature_cluster_global_refinement.refined_mask_a,
            refined_mask_b=evaluation.feature_cluster_global_refinement.refined_mask_b,
            chosen_prediction_a=evaluation.prediction_masks[0],
            chosen_prediction_b=evaluation.prediction_masks[1],
            assignment_used=str(evaluation.row.get("assignment_used", "")),
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.mask_prompt_invariance_refinement is not None:
        return render_mask_prompt_invariance_panel(
            sample=sample,
            raw_prompt_a=evaluation.mask_prompt_invariance_refinement.raw_branch.prompt_mask_a,
            raw_prompt_b=evaluation.mask_prompt_invariance_refinement.raw_branch.prompt_mask_b,
            pooled_prompt_a=evaluation.mask_prompt_invariance_refinement.pooled_branch.prompt_mask_a,
            pooled_prompt_b=evaluation.mask_prompt_invariance_refinement.pooled_branch.prompt_mask_b,
            random_prompt_a=evaluation.mask_prompt_invariance_refinement.random_branch.prompt_mask_a,
            random_prompt_b=evaluation.mask_prompt_invariance_refinement.random_branch.prompt_mask_b,
            raw_final_a=evaluation.mask_prompt_invariance_refinement.raw_to_pooled_aligned_final_mask_a,
            raw_final_b=evaluation.mask_prompt_invariance_refinement.raw_to_pooled_aligned_final_mask_b,
            pooled_final_a=evaluation.mask_prompt_invariance_refinement.pooled_branch.final_mask_a,
            pooled_final_b=evaluation.mask_prompt_invariance_refinement.pooled_branch.final_mask_b,
            random_final_a=evaluation.mask_prompt_invariance_refinement.random_to_pooled_aligned_final_mask_a,
            random_final_b=evaluation.mask_prompt_invariance_refinement.random_to_pooled_aligned_final_mask_b,
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.boundary_refine_sweep_refinement is not None:
        return render_boundary_refine_sweep_panel(
            sample=sample,
            sweep=evaluation.boundary_refine_sweep_refinement,
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    if evaluation.feature_cluster_coarse_to_fine_global_refinement is not None:
        if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
            return render_feature_cluster_positionality_panel(
                sample=sample,
                raw_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.raw_diagnostic_label_map,
                baseline_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.baseline_branch_label_map,
                projection_removed_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.projection_removed_label_map
                ),
                flip_avg_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_label_map,
                null_constant_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.null_constant_label_map
                ),
                null_noise_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.null_noise_label_map,
                null_blur_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.null_blur_label_map,
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            )
        if variant == "flip_avg_plus_edge_debias":
            return render_feature_cluster_edge_bias_panel(
                sample=sample,
                raw_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.raw_diagnostic_label_map,
                baseline_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.baseline_branch_label_map,
                flip_avg_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_label_map,
                edge_debiased_label_map=evaluation.feature_cluster_coarse_to_fine_global_refinement.edge_debiased_label_map,
                flip_avg_plus_edge_debiased_label_map=(
                    evaluation.feature_cluster_coarse_to_fine_global_refinement.flip_avg_plus_edge_debiased_label_map
                ),
                protocol=f"sam3_auto:{variant}",
                metric_summary=evaluation.metric_summary,
            )
        return render_feature_cluster_coarse_to_fine_global_panel(
            sample=sample,
            level_label_maps=evaluation.feature_cluster_coarse_to_fine_global_refinement.level_label_maps,
            level_names=evaluation.feature_cluster_coarse_to_fine_global_refinement.level_names,
            rough_mask_a=evaluation.feature_cluster_coarse_to_fine_global_refinement.rough_mask_a,
            rough_mask_b=evaluation.feature_cluster_coarse_to_fine_global_refinement.rough_mask_b,
            refined_mask_a=evaluation.feature_cluster_coarse_to_fine_global_refinement.refined_mask_a,
            refined_mask_b=evaluation.feature_cluster_coarse_to_fine_global_refinement.refined_mask_b,
            chosen_prediction_a=evaluation.prediction_masks[0],
            chosen_prediction_b=evaluation.prediction_masks[1],
            assignment_used=str(evaluation.row.get("assignment_used", "")),
            multiscale_refinement_applied=(
                evaluation.feature_cluster_coarse_to_fine_global_refinement.multiscale_refinement_applied
            ),
            sam_refinement_applied=(
                evaluation.feature_cluster_coarse_to_fine_global_refinement.sam_refinement_applied
            ),
            protocol=f"sam3_auto:{variant}",
            metric_summary=evaluation.metric_summary,
        )
    return render_feature_mask_panel(
        sample=sample,
        dense_prediction_masks=evaluation.feature_mask_refinement.dense_prediction_masks,
        selection_mode=evaluation.feature_mask_refinement.selection_mode,
        selected_pair_masks=evaluation.feature_mask_refinement.selected_pair_masks,
        selected_pair_ids=evaluation.feature_mask_refinement.selected_pair_ids,
        coarse_margin=evaluation.feature_mask_refinement.coarse_margin,
        final_prediction_a=evaluation.prediction_masks[0],
        final_prediction_b=evaluation.prediction_masks[1],
        protocol=f"sam3_auto:{variant}",
        metric_summary=evaluation.metric_summary,
    )


def resolve_sam3_auto_variants(variant: str) -> tuple[str, ...]:
    if variant == "both":
        return ("default", "dense")
    return (variant,)


def resolve_sam3_auto_output_dir(output_dir: str | None, variant: str, split: str) -> Path:
    """Resolve the output directory for a SAM-3 automatic-mask run."""

    if output_dir:
        requested_dir = Path(output_dir)
        if _normalize_output_path(requested_dir) == _normalize_output_path(SAM3_AUTO_OUTPUT_ROOT):
            safe_dir = requested_dir / f"{split}_{variant}"
            LOGGER.warning(
                "Requested shared SAM-3 auto output root %s; redirecting this run to %s to avoid mixed artifacts.",
                requested_dir,
                safe_dir,
            )
            return safe_dir
        return requested_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"sam3-auto-{variant}-{split}-{timestamp}"
    return SAM3_AUTO_OUTPUT_ROOT / run_name


def prepare_sam3_auto_output_dir(
    output_dir: str | None,
    *,
    variant: str,
    split: str,
    run_kind: str,
    sample_index: int | None = None,
) -> Path:
    """Resolve and validate a SAM-3 automatic-mask output directory.

    The shared ``outputs/rwtd_sam3_auto`` root is treated as a namespace, not a
    real run directory. Passing it explicitly redirects to a variant-specific
    child. Existing directories are checked so predict/eval artifacts and
    conflicting variants cannot be mixed silently.
    """

    if run_kind not in {"predict", "eval"}:
        raise ValueError(f"Unsupported run_kind '{run_kind}'. Expected 'predict' or 'eval'.")

    resolved_dir = resolve_sam3_auto_output_dir(output_dir, variant=variant, split=split)
    if output_dir and _normalize_output_path(Path(output_dir)) == _normalize_output_path(SAM3_AUTO_OUTPUT_ROOT):
        if run_kind == "predict":
            resolved_dir = resolved_dir.with_name(f"{split}_{variant}_index{sample_index}")
        _validate_sam3_auto_output_dir(
            resolved_dir,
            requested_variant=variant,
            run_kind=run_kind,
        )
        return resolved_dir

    _validate_sam3_auto_output_dir(
        resolved_dir,
        requested_variant=variant,
        run_kind=run_kind,
    )
    return resolved_dir


def _normalize_output_path(path: Path) -> Path:
    """Resolve a path without requiring it to exist."""

    return path.expanduser().resolve(strict=False)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _validate_sam3_auto_output_dir(output_dir: Path, *, requested_variant: str, run_kind: str) -> None:
    """Fail loudly when a run directory would mix incompatible SAM-3 auto artifacts."""

    if not output_dir.exists():
        return

    has_prediction_artifacts = (output_dir / "prediction.json").exists()
    has_eval_artifacts = any(
        path.exists()
        for path in (
            output_dir / "per_sample_metrics.csv",
            output_dir / "summary.json",
            output_dir / "summary.md",
            output_dir / "visuals",
            output_dir / "visuals_manifest.jsonl",
        )
    )
    if has_prediction_artifacts and has_eval_artifacts:
        raise RuntimeError(
            "Refusing to reuse SAM-3 auto output directory "
            f"'{output_dir}' because it already mixes predict-one and eval artifacts. "
            "Use a fresh variant-specific run directory."
        )
    if run_kind == "predict" and has_eval_artifacts:
        raise RuntimeError(
            "predict-sam3-auto refuses to write into an existing eval-sam3-auto directory: "
            f"'{output_dir}'. Use a fresh variant-specific directory."
        )
    if run_kind == "eval" and has_prediction_artifacts:
        raise RuntimeError(
            "eval-sam3-auto refuses to write into an existing predict-sam3-auto directory: "
            f"'{output_dir}'. Use a fresh variant-specific directory."
        )

    for metadata_path in (output_dir / "config.json", output_dir / "summary.json", output_dir / "prediction.json"):
        metadata = _read_json_if_exists(metadata_path)
        if metadata is None:
            continue
        existing_variant = metadata.get("variant")
        if isinstance(existing_variant, str) and existing_variant != requested_variant:
            raise RuntimeError(
                "Refusing to reuse SAM-3 auto output directory "
                f"'{output_dir}' because it already belongs to variant '{existing_variant}', "
                f"not '{requested_variant}'."
            )
