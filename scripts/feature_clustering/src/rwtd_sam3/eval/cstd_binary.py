"""CSTD adapter for registered cross-dataset SAM-feature experiments.

This module exposes the local CSTD image/region/edge assets behind
``predict-cstd-binary`` and ``eval-cstd-binary``. It reuses the same
registered SAM-feature experiments used on RWTD, STLD, CAID, and DeTexture
ADE20K, applies permutation-invariant binary scoring, and writes the standard
run artifacts under ``outputs/cstd_binary/``.
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

from rwtd_sam3.data.cstd_binary import (
    CSTD_DATASET_ID,
    CSTDBinaryOverview,
    CSTDBinarySample,
    get_cstd_binary_sample,
    iter_cstd_binary_samples,
    load_cstd_binary_overview,
)
from rwtd_sam3.eval.metrics import (
    aggregate_masks_by_regions,
    build_canonical_evaluation_fields,
    CANONICAL_EVALUATION_CONTRACT,
    CANONICAL_PRIMARY_METRIC,
    CANONICAL_SECONDARY_METRIC,
    compute_binary_metrics,
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
    DEFAULT_CROSS_DATASET_EXPERIMENT,
    build_cross_dataset_experiment_runner,
    get_cross_dataset_experiment_spec,
)
from rwtd_sam3.eval.sam3_auto import select_best_binary_assignment
from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    FeatureClusterCoarseToFineGlobalRefinement,
)
from rwtd_sam3.models.sam3_boundary_refine_sweep_runner import (
    BOUNDARY_REFINE_SWEEP_SETTINGS,
    BOUNDARY_REFINE_SWEEP_VARIANT_IDS,
    BOUNDARY_REFINE_SWEEP_VARIANT_LABELS,
    BoundaryRefineSweepResult,
    build_boundary_refine_sweep_summary,
)
from rwtd_sam3.utils.visualization import (
    build_visual_caption,
    build_visual_footer_lines,
    render_boundary_refine_sweep_panel,
    render_feature_cluster_positionality_panel,
    render_feature_cluster_coarse_to_fine_global_panel,
    save_boundary_refine_sweep_panel,
    save_feature_cluster_positionality_panel,
    save_feature_cluster_coarse_to_fine_global_panel,
)
from rwtd_sam3.eval.pooled_feature_pca_overlay import save_named_sample_pooled_feature_pca_overlay


LOGGER = logging.getLogger(__name__)

CSTD_BINARY_DEFAULT_VARIANT = DEFAULT_CROSS_DATASET_EXPERIMENT
CSTD_BINARY_SUPPORTED_VARIANTS = CROSS_DATASET_EXPERIMENT_VARIANTS
CSTD_BINARY_OUTPUT_ROOT = Path("outputs") / "cstd_binary"
CSTD_HARDWARE_COMPATIBILITY_STANDARD = "cstd_streaming_eval_v1"
CSTD_HARDWARE_COMPATIBILITY_DESCRIPTION = (
    "CSTD evaluation must stream decoded image/region/edge triples instead of preloading the dataset, "
    "must resolve the dataset root once per run, and must persist the active sample-loading mode into run artifacts."
)
CSTD_BINARY_SCALAR_FIELDS = (
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
class CSTDBinarySampleResult:
    """Per-sample CSTD benchmark payload."""

    row: dict[str, Any]
    metric_summary: str
    prediction_masks: tuple[np.ndarray, np.ndarray]
    aggregated_prediction_a: np.ndarray
    aggregated_prediction_b: np.ndarray
    refinement: FeatureClusterCoarseToFineGlobalRefinement
    boundary_refine_sweep: BoundaryRefineSweepResult | None = None


def run_cstd_binary_predict_one(args) -> dict[str, Any]:
    """Run one registered experiment on one CSTD sample."""

    sample = get_cstd_binary_sample(dataset_root=args.dataset_root, index=args.index)
    output_dir = prepare_cstd_binary_output_dir(
        args.output_dir,
        variant=args.variant,
        run_kind="predict",
        sample_index=args.index,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = load_cstd_binary_overview(args.dataset_root)
    config_payload = build_cstd_binary_run_config(
        args,
        overview=overview,
        dataset_partition=None,
        selected_sample_count=1,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_cstd_binary_experiment_terms_markdown(
            args,
            dataset_partition=None,
            selected_sample_count=1,
        ),
    )

    runner = build_cross_dataset_experiment_runner(
        args.variant,
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
    )
    evaluation = evaluate_cstd_binary_sample(
        sample=sample,
        variant=args.variant,
        runner=runner,
    )

    prediction_path = output_dir / "prediction.json"
    write_json(prediction_path, evaluation.row)
    panel_path = None
    if args.save_visuals:
        panel_path = save_cstd_binary_panel(
            output_path=output_dir / "prediction.png",
            sample=sample,
            evaluation=evaluation,
            variant=args.variant,
        )
        write_jsonl(
            output_dir / "visuals_manifest.jsonl",
            [
                build_cstd_binary_visual_record(
                    sample=sample,
                    evaluation=evaluation,
                    variant=args.variant,
                    visual_path=Path("prediction.png"),
                )
            ],
        )
        if args.save_pooled_feature_pca_overlay and hasattr(runner, "extract_pooled_feature_map_for_visualization"):
            save_named_sample_pooled_feature_pca_overlay(
                output_dir / "visuals",
                sample=sample,
                runner=runner,
            )
    return {
        args.variant: {
            "prediction_json": str(prediction_path),
            "visualization": str(panel_path) if panel_path is not None else None,
            "metrics": evaluation.row,
        }
    }


def run_cstd_binary_evaluation(args) -> dict[str, Any]:
    """Run one registered experiment over the local CSTD assets."""

    overview = load_cstd_binary_overview(args.dataset_root)
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_total_samples = resolve_eval_sample_count(args.limit, overview.num_examples, dataset_partition)
    if num_total_samples < 1:
        raise RuntimeError("The CSTD root did not yield any samples to evaluate.")

    output_dir = prepare_cstd_binary_output_dir(
        args.output_dir,
        variant=args.variant,
        run_kind="eval",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = build_cstd_binary_run_config(
        args,
        overview=overview,
        dataset_partition=dataset_partition,
        selected_sample_count=num_total_samples,
    )
    write_json(output_dir / "config.json", config_payload)
    write_text(
        output_dir / "experiment_terms.md",
        build_cstd_binary_experiment_terms_markdown(
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
    runner = build_cross_dataset_experiment_runner(
        args.variant,
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
    )
    LOGGER.info(
        "CSTD eval uses streamed sample decoding under hardware standard %s; limit=%s.",
        CSTD_HARDWARE_COMPATIBILITY_STANDARD,
        args.limit,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    visual_records: list[dict[str, Any]] = []

    try:
        progress = tqdm(total=num_total_samples, desc="eval:cstd", unit="sample")
        for step, sample in enumerate(
            iter_cstd_binary_samples(
                dataset_root=overview.dataset_root,
                limit=num_total_samples,
                start_index=dataset_partition.start_index if dataset_partition is not None else 0,
            )
        ):
            try:
                evaluation = evaluate_cstd_binary_sample(
                    sample=sample,
                    variant=args.variant,
                    runner=runner,
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
            progress.update(1)
            progress.set_postfix({"crop": sample.crop_name, "eval_miou": f"{evaluation.row['eval_miou']:.3f}"})

            if args.save_visuals:
                visual_path = Path("visuals") / f"{sample.crop_name}.png"
                save_cstd_binary_panel(
                    output_path=output_dir / visual_path,
                    sample=sample,
                    evaluation=evaluation,
                    variant=args.variant,
                )
                visual_records.append(
                    build_cstd_binary_visual_record(
                        sample=sample,
                        evaluation=evaluation,
                        variant=args.variant,
                        visual_path=visual_path,
                    )
                )
                if args.save_pooled_feature_pca_overlay and hasattr(runner, "extract_pooled_feature_map_for_visualization"):
                    save_named_sample_pooled_feature_pca_overlay(
                        output_dir / "visuals",
                        sample=sample,
                        runner=runner,
                    )

            if args.wandb:
                wandb_session.log(
                    {
                        "dataset_id": CSTD_DATASET_ID,
                        "variant": args.variant,
                        "eval_miou": evaluation.row["eval_miou"],
                        "eval_ari": evaluation.row["eval_ari"],
                        "miou": evaluation.row["miou"],
                        "ari": evaluation.row["ari"],
                        "miou_agg": evaluation.row["miou_agg"],
                    },
                    step=step,
                )
                if step % args.log_every == 0:
                    preview_image = render_cstd_binary_panel(
                        sample=sample,
                        evaluation=evaluation,
                        variant=args.variant,
                    )
                    caption = build_visual_caption(
                        sample=sample,
                        protocol=_cstd_protocol_label(args.variant),
                        metric_summary=evaluation.metric_summary,
                    )
                    wandb_session.log_preview(preview_image, caption=caption, step=step)
        progress.close()

        if not rows:
            raise RuntimeError(
                "The CSTD benchmark produced no successful evaluations. "
                "Check the recorded failures for details."
            )

        write_csv(output_dir / "per_sample_metrics.csv", rows)
        if args.save_visuals:
            write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
        summary = build_cstd_binary_summary(
            overview=overview,
            variant=args.variant,
            model_id=args.model_id,
            rows=rows,
            failures=failures,
            num_total_samples=num_total_samples,
            dataset_partition=dataset_partition,
        )
        write_json(output_dir / "summary.json", summary)
        write_text(output_dir / "summary.md", build_cstd_binary_markdown_summary(summary))
    finally:
        wandb_session.finish()

    return {args.variant: summary}


def evaluate_cstd_binary_sample(
    sample: CSTDBinarySample,
    variant: str,
    runner,
) -> CSTDBinarySampleResult:
    """Evaluate one registered experiment on one CSTD sample."""

    if variant == "boundary_refine_sweep":
        if not hasattr(runner, "generate_feature_cluster_sweep"):
            raise RuntimeError("boundary_refine_sweep requires a runner with generate_feature_cluster_sweep().")
        sweep = runner.generate_feature_cluster_sweep(sample.image)
        baseline_output = sweep.variants["v0_no_refine"]
        assignment = select_best_binary_assignment(
            prediction_a=baseline_output.mask_a,
            prediction_b=baseline_output.mask_b,
            target_a=sample.texture_a_mask,
            target_b=sample.texture_b_mask,
            source_a="cluster_a",
            source_b="cluster_b",
        )
        chosen_prediction_a = assignment.chosen_prediction_a
        chosen_prediction_b = assignment.chosen_prediction_b
        aggregated_masks, overlap_counts = aggregate_masks_by_regions(
            [chosen_prediction_a, chosen_prediction_b],
            (sample.texture_a_mask, sample.texture_b_mask),
        )
        texture_a_iou = compute_binary_metrics(chosen_prediction_a, sample.texture_a_mask).iou
        texture_b_iou = compute_binary_metrics(chosen_prediction_b, sample.texture_b_mask).iou
        texture_a_agg_iou = compute_binary_metrics(aggregated_masks[0], sample.texture_a_mask).iou
        texture_b_agg_iou = compute_binary_metrics(aggregated_masks[1], sample.texture_b_mask).iou
        row = {
            "dataset_id": CSTD_DATASET_ID,
            "variant": variant,
            "split": sample.split,
            "sample_index": sample.index,
            "crop_name": sample.crop_name,
            "num_predicted_masks": 2,
            "evaluation_view": sample.evaluation_view,
            "miou": assignment.chosen_miou,
            "ari": assignment.chosen_ari,
            "miou_agg": float(np.mean([texture_a_agg_iou, texture_b_agg_iou])),
            "texture_a_best_iou": texture_a_iou,
            "texture_b_best_iou": texture_b_iou,
            "texture_a_agg_iou": texture_a_agg_iou,
            "texture_b_agg_iou": texture_b_agg_iou,
            "texture_a_overlap_mask_count": overlap_counts[0],
            "texture_b_overlap_mask_count": overlap_counts[1],
            "mask_score_mean": 0.0,
            "mask_score_median": 0.0,
            "boundary_refine_sweep_status": "ok",
            "label_permutation_invariant": True,
            "assignment_used": assignment.assignment_used,
            "assignment_source_for_texture_a": assignment.chosen_source_a,
            "assignment_source_for_texture_b": assignment.chosen_source_b,
            "assignment_direct_miou": assignment.direct_miou,
            "assignment_direct_ari": assignment.direct_ari,
            "assignment_swapped_miou": assignment.swapped_miou,
            "assignment_swapped_ari": assignment.swapped_ari,
            "evaluation_selector": "best_partition_assignment",
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
        }
        row.update(
            build_canonical_evaluation_fields(
                miou=assignment.chosen_miou,
                ari=assignment.chosen_ari,
                evaluation_view=sample.evaluation_view,
            )
        )
        best_variant_id = "v0_no_refine"
        best_tuple = (float("-inf"), float("inf"))
        for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
            variant_output = sweep.variants[variant_id]
            variant_assignment = select_best_binary_assignment(
                prediction_a=variant_output.mask_a,
                prediction_b=variant_output.mask_b,
                target_a=sample.texture_a_mask,
                target_b=sample.texture_b_mask,
                source_a="cluster_a",
                source_b="cluster_b",
            )
            row[f"{variant_id}_eval_miou"] = variant_assignment.chosen_miou
            row[f"{variant_id}_eval_ari"] = variant_assignment.chosen_ari
            row[f"{variant_id}_percent_pixels_changed"] = variant_output.percent_pixels_changed
            row[f"{variant_id}_num_changed_components"] = variant_output.num_changed_components
            row[f"{variant_id}_average_changed_margin"] = variant_output.average_changed_margin
            rank_tuple = (variant_assignment.chosen_miou, -variant_output.percent_pixels_changed)
            if rank_tuple > best_tuple:
                best_variant_id = variant_id
                best_tuple = rank_tuple
        row["recommended_variant"] = best_variant_id
        row["recommended_variant_label"] = BOUNDARY_REFINE_SWEEP_VARIANT_LABELS[best_variant_id]
        return CSTDBinarySampleResult(
            row=row,
            metric_summary=f"boundary_refine_sweep | best={best_variant_id}",
            prediction_masks=(chosen_prediction_a, chosen_prediction_b),
            aggregated_prediction_a=aggregated_masks[0],
            aggregated_prediction_b=aggregated_masks[1],
            refinement=FeatureClusterCoarseToFineGlobalRefinement(
                coarsest_label_map=sweep.coarsest_label_map,
                level_label_maps=(sweep.coarsest_label_map,),
                level_names=(sweep.level_names[0],),
                level_resolutions=(sweep.level_resolutions[0],),
                rough_mask_a=baseline_output.mask_a,
                rough_mask_b=baseline_output.mask_b,
                refined_mask_a=baseline_output.mask_a,
                refined_mask_b=baseline_output.mask_b,
                refined_score_a=0.0,
                refined_score_b=0.0,
                refined_pair_selection_score=0.0,
                refined_pair_overlap_iou=0.0,
                cluster_pixel_count_a=int(baseline_output.mask_a.sum()),
                cluster_pixel_count_b=int(baseline_output.mask_b.sum()),
                multiscale_refinement_applied=False,
                sam_refinement_applied=False,
            ),
            boundary_refine_sweep=sweep,
        )

    refinement = runner.generate_feature_clusters(sample.image)
    prediction_a = np.asarray(refinement.refined_mask_a, dtype=bool)
    prediction_b = np.asarray(refinement.refined_mask_b, dtype=bool)
    assignment = select_best_binary_assignment(
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        target_a=sample.texture_a_mask,
        target_b=sample.texture_b_mask,
        source_a="cluster_a",
        source_b="cluster_b",
    )
    chosen_prediction_a = assignment.chosen_prediction_a
    chosen_prediction_b = assignment.chosen_prediction_b
    aggregated_masks, overlap_counts = aggregate_masks_by_regions(
        [chosen_prediction_a, chosen_prediction_b],
        (sample.texture_a_mask, sample.texture_b_mask),
    )
    texture_a_iou = compute_binary_metrics(chosen_prediction_a, sample.texture_a_mask).iou
    texture_b_iou = compute_binary_metrics(chosen_prediction_b, sample.texture_b_mask).iou
    texture_a_agg_iou = compute_binary_metrics(aggregated_masks[0], sample.texture_a_mask).iou
    texture_b_agg_iou = compute_binary_metrics(aggregated_masks[1], sample.texture_b_mask).iou
    refined_scores = [
        score
        for score in (refinement.refined_score_a, refinement.refined_score_b)
        if refinement.sam_refinement_applied or score not in {0.0, None}
    ]

    row = {
        "dataset_id": CSTD_DATASET_ID,
        "variant": variant,
        "split": sample.split,
        "sample_index": sample.index,
        "crop_name": sample.crop_name,
        "num_predicted_masks": 2,
        "evaluation_view": sample.evaluation_view,
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
        f"{variant}_status": "ok",
        "label_permutation_invariant": True,
        "assignment_used": assignment.assignment_used,
        "assignment_source_for_texture_a": assignment.chosen_source_a,
        "assignment_source_for_texture_b": assignment.chosen_source_b,
        "assignment_direct_miou": assignment.direct_miou,
        "assignment_direct_ari": assignment.direct_ari,
        "assignment_swapped_miou": assignment.swapped_miou,
        "assignment_swapped_ari": assignment.swapped_ari,
        "evaluation_selector": "best_partition_assignment",
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
            "coarsest_pooled_projection_removed_direct"
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
            evaluation_view=sample.evaluation_view,
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
        row["active_coarsest_component_stats_json"] = json.dumps(refinement.coarsest_component_stats, sort_keys=True)
    if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
        row["recommended_mitigation"] = refinement.recommended_mitigation
        row["recommendation_reason"] = refinement.recommendation_reason

    metric_summary = (
        f"{variant} | CSTD invariant "
        f"mIoU={row['miou']:.3f} "
        f"ARI={row['ari']:.3f} "
        f"Aggr mIoU={row['miou_agg']:.3f}"
    )
    return CSTDBinarySampleResult(
        row=row,
        metric_summary=metric_summary,
        prediction_masks=(chosen_prediction_a, chosen_prediction_b),
        aggregated_prediction_a=aggregated_masks[0],
        aggregated_prediction_b=aggregated_masks[1],
        refinement=refinement,
    )


def build_cstd_binary_run_config(
    args,
    overview: CSTDBinaryOverview | None,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> dict[str, Any]:
    """Build a stable run-config payload for CSTD."""

    spec = get_cross_dataset_experiment_spec(args.variant)
    config = {
        "command": getattr(args, "command", None),
        "dataset_id": CSTD_DATASET_ID,
        "dataset_root": str(Path(args.dataset_root).expanduser()),
        "variant": args.variant,
        "cross_dataset_registered": True,
        "experiment_summary": spec.summary,
        "supported_dataset_ids": list(spec.supported_datasets),
        "visual_contract": spec.visual_contract,
        "promotion_note": spec.promotion_note,
        "model_id": args.model_id,
        "device": args.device,
        "hardware_compatibility_standard": CSTD_HARDWARE_COMPATIBILITY_STANDARD,
        "hardware_compatibility_description": CSTD_HARDWARE_COMPATIBILITY_DESCRIPTION,
        "sample_loading_mode": "streamed_iter",
        "official_checkpoint_path": getattr(args, "official_checkpoint_path", None),
        "limit": getattr(args, "limit", None),
        "failure_policy": getattr(args, "failure_policy", "abort"),
        "save_visuals": args.save_visuals,
        "wandb": getattr(args, "wandb", False),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_cluster_coarse_to_fine_global_pooled_init_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_debiased_settings": (
            FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS
        ),
        "boundary_refine_sweep_settings": BOUNDARY_REFINE_SWEEP_SETTINGS,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "primary_metric_reason": (
            "CSTD exposes one binary region mask per image, so the adapter evaluates the region and its complement "
            "under both cluster assignments and reports the better partition-invariant result."
        ),
        "versions": discovered_package_versions(),
    }
    if overview is not None:
        config["resolved_dataset_root"] = str(overview.dataset_root)
        config["resolved_image_dir"] = str(overview.image_dir)
        config["resolved_region_dir"] = str(overview.region_dir)
        config["resolved_edge_dir"] = str(overview.edge_dir)
        config["num_examples"] = overview.num_examples
        config["evaluation_view"] = overview.evaluation_view
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_cstd_binary_experiment_terms_markdown(
    args,
    *,
    dataset_partition: DatasetPartitionSelection | None,
    selected_sample_count: int | None,
) -> str:
    """Render a dataset-specific protocol-spec document for CSTD."""

    spec = get_cross_dataset_experiment_spec(args.variant)
    lines = [
        "# CSTD Benchmark Terms",
        "",
        "## Run Scope",
        "",
        f"- Command family: `{getattr(args, 'command', 'cstd-binary')}`",
        f"- Dataset id: `{CSTD_DATASET_ID}`",
        f"- Requested dataset root: `{args.dataset_root}`",
        "- Expected local root form: `<root>/images`, `<root>/regions`, and `<root>/edges`.",
        f"- Executed variant: `{args.variant}`",
        f"- Variant summary: {spec.summary}",
        f"- Registered supported datasets: {', '.join(f'`{dataset}`' for dataset in spec.supported_datasets)}",
        f"- Model: `{args.model_id}`",
        f"- Device request: `{args.device}`",
        f"- Hardware compatibility standard: `{CSTD_HARDWARE_COMPATIBILITY_STANDARD}`",
        "- Sample loading mode: `streamed_iter` (the adapter does not preload all images into memory).",
        f"- Save visuals: `{getattr(args, 'save_visuals', True)}`",
        (
            f"- Dataset partition: `{dataset_partition.raw_spec}` -> indices `{dataset_partition.start_index}:{dataset_partition.end_index}` "
            f"({selected_sample_count} selected sample(s) from {dataset_partition.partition_size} partition items, full dataset size {dataset_partition.total_size})."
            if dataset_partition is not None
            else "- Dataset partition: full dataset order (no partitioning)."
        ),
        "",
        "## Dataset Semantics",
        "",
        "- Each sample is one image from `images/`, one binary region mask from `regions/`, and one binary edge map from `edges/` with the same natural-sorted stem.",
        "- The adapter uses the provided `regions/` mask as texture A, its complement as texture B, and preserves the provided `edges/` mask as the boundary visualization input.",
        "- The task is scored as a two-region binary partition with permutation-invariant evaluation.",
        "- Cross-dataset experiment contract: every current-method experiment promoted into this adapter must register itself in `src/rwtd_sam3/eval/experiment_registry.py`, declare support for all registered binary datasets, and be documented in the root README before merge.",
        "",
        "## Protocol Standard",
        "",
        f"1. Run `{args.variant}` exactly as registered in the cross-dataset experiment registry.",
        "2. The dataset adapter reuses the registered runner unchanged and applies only CSTD-specific GT decoding, permutation-invariant assignment, and output-directory handling.",
        "3. Preserve disconnected same-texture regions unless the registered experiment explicitly changes that behavior.",
        f"4. Shared pooled-init settings available to the registered current-method family: feature source=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['feature_source']}`, pooled init kernel=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_kernel_size']}`, pooled init stride=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['coarsest_init_pool_stride']}`, k-means max iterations=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_max_iterations']}`, k-means convergence tolerance=`{FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS['kmeans_convergence_tolerance']}`.",
        "",
        "## Metric Standard",
        "",
        f"- `{CANONICAL_PRIMARY_METRIC}` / `{CANONICAL_SECONDARY_METRIC}` are the repo-wide default metrics here and are identical to partition-invariant `miou` / `ari`.",
        "- Both cluster-to-region assignments are scored and the better one is reported.",
        "- `miou_agg` is still exported for consistency with the repo-wide artifact contract, but it is not the primary dataset metric here.",
        "",
        "## Output Contract",
        "",
        "- `prediction.json` / `prediction.png` for `predict-cstd-binary`.",
        "- `per_sample_metrics.csv`, `summary.json`, `summary.md`, and `visuals_manifest.jsonl` for `eval-cstd-binary`.",
        f"- The preview contract comes from the registry entry for `{args.variant}` and is currently `{spec.visual_contract}`.",
        "",
        "## Failure Semantics",
        "",
        "- There is no silent fallback to another dataset or refinement stage.",
        "- Missing dataset roots, unmatched image/region/edge triples, empty regions, shape mismatches, or missing official Meta `sam3` features are explicit failures.",
        "- `failure_policy=skip` applies only to evaluation runs and records failed samples in the summary instead of aborting immediately.",
    ]
    return "\n".join(lines) + "\n"


def build_cstd_binary_summary(
    overview: CSTDBinaryOverview,
    variant: str,
    model_id: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    num_total_samples: int,
    dataset_partition: DatasetPartitionSelection | None,
) -> dict[str, Any]:
    """Aggregate one local CSTD run."""

    spec = get_cross_dataset_experiment_spec(variant)
    mean_metrics = {name: mean(float(row[name]) for row in rows) for name in CSTD_BINARY_SCALAR_FIELDS}
    median_metrics = {name: median(float(row[name]) for row in rows) for name in CSTD_BINARY_SCALAR_FIELDS}
    summary = {
        "dataset_id": CSTD_DATASET_ID,
        "dataset_root": str(overview.dataset_root),
        "evaluation_view": overview.evaluation_view,
        "variant": variant,
        "cross_dataset_registered": True,
        "supported_dataset_ids": list(spec.supported_datasets),
        "visual_contract": spec.visual_contract,
        "experiment_summary": spec.summary,
        "model_id": model_id,
        "hardware_compatibility_standard": CSTD_HARDWARE_COMPATIBILITY_STANDARD,
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "primary_metric_name": CANONICAL_PRIMARY_METRIC,
        "primary_metric_value": mean_metrics[CANONICAL_PRIMARY_METRIC],
        "secondary_metric_name": CANONICAL_SECONDARY_METRIC,
        "secondary_metric_value": mean_metrics[CANONICAL_SECONDARY_METRIC],
        "primary_metric_reason": (
            "CSTD exposes one binary region mask per image, so this run scores the region and its complement "
            "under both cluster assignments and reports the better partition-invariant result."
        ),
        "num_total_samples": num_total_samples,
        "full_dataset_num_examples": overview.num_examples,
        "num_evaluated_samples": len(rows),
        "num_failed_samples": len(failures),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "failures": failures,
    }
    if variant == "boundary_refine_sweep":
        sweep_summary = build_boundary_refine_sweep_summary(rows)
        summary["sweep_variant_metrics"] = sweep_summary["variant_metrics"]
        summary["recommended_variant"] = sweep_summary["recommended_variant"]
        summary["recommended_variant_reason"] = sweep_summary["recommended_variant_reason"]
    append_dataset_partition_fields(summary, dataset_partition, selected_sample_count=num_total_samples)
    return summary


def build_cstd_binary_markdown_summary(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for CSTD."""

    lines = [
        "# CSTD Summary",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Evaluation view: `{summary['evaluation_view']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Registered supported datasets: {', '.join(f'`{name}`' for name in summary.get('supported_dataset_ids', []))}",
        f"- Model: `{summary['model_id']}`",
        f"- Hardware compatibility standard: `{summary['hardware_compatibility_standard']}`",
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
    for metric_name in CSTD_BINARY_SCALAR_FIELDS:
        lines.append(
            f"| `{metric_name}` | {summary['mean_metrics'][metric_name]:.6f} | "
            f"{summary['median_metrics'][metric_name]:.6f} |"
        )
    if "sweep_variant_metrics" in summary:
        lines.extend(
            [
                "",
                "## Boundary Refine Sweep",
                "",
                f"- Recommended variant: `{summary['recommended_variant']}`",
                f"- Reason: {summary['recommended_variant_reason']}",
                "",
                "| Variant | Mean mIoU | Mean ARI | Mean % Changed | Mean Changed CCs | Mean Changed Margin | Improved vs V0 | Worsened vs V0 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
            metrics = summary["sweep_variant_metrics"][variant_id]
            lines.append(
                f"| `{variant_id}` | {metrics['eval_miou']:.6f} | {metrics['eval_ari']:.6f} | "
                f"{metrics['percent_pixels_changed']:.6f} | {metrics['num_changed_components']:.6f} | "
                f"{metrics['average_changed_margin']:.6f} | {metrics['improved_vs_v0_count']} | "
                f"{metrics['worsened_vs_v0_count']} |"
            )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(f"- `{failure['crop_name']}`: {failure['error']}")
    return "\n".join(lines) + "\n"


def save_cstd_binary_panel(
    output_path: str | Path,
    sample: CSTDBinarySample,
    evaluation: CSTDBinarySampleResult,
    variant: str,
) -> Path:
    """Save the registered preview panel used for CSTD."""

    if evaluation.boundary_refine_sweep is not None:
        return save_boundary_refine_sweep_panel(
            output_path=output_path,
            sample=sample,
            sweep=evaluation.boundary_refine_sweep,
            protocol=_cstd_protocol_label(variant),
            metric_summary=evaluation.metric_summary,
        )

    if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
        return save_feature_cluster_positionality_panel(
            output_path=output_path,
            sample=sample,
            raw_label_map=evaluation.refinement.raw_diagnostic_label_map,
            baseline_label_map=evaluation.refinement.baseline_branch_label_map,
            projection_removed_label_map=evaluation.refinement.projection_removed_label_map,
            flip_avg_label_map=evaluation.refinement.flip_avg_label_map,
            null_constant_label_map=evaluation.refinement.null_constant_label_map,
            null_noise_label_map=evaluation.refinement.null_noise_label_map,
            null_blur_label_map=evaluation.refinement.null_blur_label_map,
            protocol=_cstd_protocol_label(variant),
            metric_summary=evaluation.metric_summary,
        )

    return save_feature_cluster_coarse_to_fine_global_panel(
        output_path=output_path,
        sample=sample,
        level_label_maps=evaluation.refinement.level_label_maps,
        level_names=evaluation.refinement.level_names,
        rough_mask_a=evaluation.refinement.rough_mask_a,
        rough_mask_b=evaluation.refinement.rough_mask_b,
        refined_mask_a=evaluation.refinement.refined_mask_a,
        refined_mask_b=evaluation.refinement.refined_mask_b,
        chosen_prediction_a=evaluation.prediction_masks[0],
        chosen_prediction_b=evaluation.prediction_masks[1],
        assignment_used=str(evaluation.row.get("assignment_used", "")),
        multiscale_refinement_applied=evaluation.refinement.multiscale_refinement_applied,
        sam_refinement_applied=evaluation.refinement.sam_refinement_applied,
        protocol=_cstd_protocol_label(variant),
        metric_summary=evaluation.metric_summary,
    )


def render_cstd_binary_panel(
    sample: CSTDBinarySample,
    evaluation: CSTDBinarySampleResult,
    variant: str,
):
    """Render the registered preview panel for previews and exports."""

    if evaluation.boundary_refine_sweep is not None:
        return render_boundary_refine_sweep_panel(
            sample=sample,
            sweep=evaluation.boundary_refine_sweep,
            protocol=_cstd_protocol_label(variant),
            metric_summary=evaluation.metric_summary,
        )

    if variant == "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only":
        return render_feature_cluster_positionality_panel(
            sample=sample,
            raw_label_map=evaluation.refinement.raw_diagnostic_label_map,
            baseline_label_map=evaluation.refinement.baseline_branch_label_map,
            projection_removed_label_map=evaluation.refinement.projection_removed_label_map,
            flip_avg_label_map=evaluation.refinement.flip_avg_label_map,
            null_constant_label_map=evaluation.refinement.null_constant_label_map,
            null_noise_label_map=evaluation.refinement.null_noise_label_map,
            null_blur_label_map=evaluation.refinement.null_blur_label_map,
            protocol=_cstd_protocol_label(variant),
            metric_summary=evaluation.metric_summary,
        )

    return render_feature_cluster_coarse_to_fine_global_panel(
        sample=sample,
        level_label_maps=evaluation.refinement.level_label_maps,
        level_names=evaluation.refinement.level_names,
        rough_mask_a=evaluation.refinement.rough_mask_a,
        rough_mask_b=evaluation.refinement.rough_mask_b,
        refined_mask_a=evaluation.refinement.refined_mask_a,
        refined_mask_b=evaluation.refinement.refined_mask_b,
        chosen_prediction_a=evaluation.prediction_masks[0],
        chosen_prediction_b=evaluation.prediction_masks[1],
        assignment_used=str(evaluation.row.get("assignment_used", "")),
        multiscale_refinement_applied=evaluation.refinement.multiscale_refinement_applied,
        sam_refinement_applied=evaluation.refinement.sam_refinement_applied,
        protocol=_cstd_protocol_label(variant),
        metric_summary=evaluation.metric_summary,
    )


def build_cstd_binary_visual_record(
    sample: CSTDBinarySample,
    evaluation: CSTDBinarySampleResult,
    variant: str,
    visual_path: Path,
) -> dict[str, Any]:
    """Build one machine-readable visual record for CSTD."""

    spec = get_cross_dataset_experiment_spec(variant)
    record = dict(evaluation.row)
    record.update(
        {
            "visual_path": str(visual_path),
            "visual_contract": spec.visual_contract,
            "supported_dataset_ids": list(spec.supported_datasets),
            "caption": build_visual_caption(
                sample=sample,
                protocol=_cstd_protocol_label(variant),
                metric_summary=evaluation.metric_summary,
            ),
            "footer_lines": build_visual_footer_lines(
                sample=sample,
                protocol=_cstd_protocol_label(variant),
                metric_summary=evaluation.metric_summary,
            ),
            "original_texture_a": sample.original_texture_a,
            "original_texture_b": sample.original_texture_b,
            "oracle_points_a_count": len(sample.oracle_points_a),
            "oracle_points_b_count": len(sample.oracle_points_b),
        }
    )
    return record


def resolve_cstd_binary_output_dir(output_dir: str | None, variant: str) -> Path:
    """Resolve the output directory for a CSTD run."""

    if output_dir:
        requested_dir = Path(output_dir)
        if _normalize_output_path(requested_dir) == _normalize_output_path(CSTD_BINARY_OUTPUT_ROOT):
            safe_dir = requested_dir / variant
            LOGGER.warning(
                "Requested shared CSTD output root %s; redirecting this run to %s to avoid mixed artifacts.",
                requested_dir,
                safe_dir,
            )
            return safe_dir
        return requested_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return CSTD_BINARY_OUTPUT_ROOT / f"cstd-{variant}-{timestamp}"


def prepare_cstd_binary_output_dir(
    output_dir: str | None,
    *,
    variant: str,
    run_kind: str,
    sample_index: int | None = None,
) -> Path:
    """Resolve and validate a CSTD output directory."""

    if run_kind not in {"predict", "eval"}:
        raise ValueError(f"Unsupported run_kind '{run_kind}'. Expected 'predict' or 'eval'.")
    resolved_dir = resolve_cstd_binary_output_dir(output_dir, variant=variant)
    if output_dir and _normalize_output_path(Path(output_dir)) == _normalize_output_path(CSTD_BINARY_OUTPUT_ROOT):
        if run_kind == "predict":
            resolved_dir = resolved_dir.with_name(f"{variant}_index{sample_index}")
        _validate_cstd_binary_output_dir(resolved_dir, requested_variant=variant, run_kind=run_kind)
        return resolved_dir
    _validate_cstd_binary_output_dir(resolved_dir, requested_variant=variant, run_kind=run_kind)
    return resolved_dir


def _cstd_protocol_label(variant: str) -> str:
    return f"cstd_binary:{variant}"


def _normalize_output_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate_cstd_binary_output_dir(
    output_dir: Path,
    *,
    requested_variant: str,
    run_kind: str,
) -> None:
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
            "Refusing to reuse CSTD output directory "
            f"'{output_dir}' because it already mixes predict and eval artifacts."
        )
    if run_kind == "predict" and has_eval_artifacts:
        raise RuntimeError(
            "predict-cstd-binary refuses to write into an existing eval directory: "
            f"'{output_dir}'. Use a fresh variant-specific directory."
        )
    if run_kind == "eval" and has_prediction_artifacts:
        raise RuntimeError(
            "eval-cstd-binary refuses to write into an existing predict directory: "
            f"'{output_dir}'. Use a fresh variant-specific directory."
        )

    for metadata_path in (output_dir / "config.json", output_dir / "summary.json", output_dir / "prediction.json"):
        metadata = _read_json_if_exists(metadata_path)
        if metadata is None:
            continue
        existing_variant = metadata.get("variant")
        if isinstance(existing_variant, str) and existing_variant != requested_variant:
            raise RuntimeError(
                f"Refusing to reuse CSTD output directory '{output_dir}' because it belongs to variant "
                f"'{existing_variant}', not '{requested_variant}'."
            )
