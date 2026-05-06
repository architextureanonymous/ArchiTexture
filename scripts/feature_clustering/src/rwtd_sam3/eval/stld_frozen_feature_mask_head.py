"""Dense-supervised STLD baseline on frozen SAM multiscale features.

This module implements a minimal supervised baseline for STLD:

- the SAM backbone stays frozen
- one tiny multiscale mask head is trained on dense foreground masks
- evaluation reports direct foreground IoU / Dice plus auxiliary
  partition-invariant binary metrics from the same predictions

The implementation intentionally reuses the existing frozen-SAM feature
extractor, the shared tiny head family, and the repository's standard
run-artifact writers instead of introducing a separate training stack.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.data.architexture_binary import (
    ArchiTextureBinarySample as GlasBinarySample,
    iter_architexture_binary_samples as iter_architexture_binary_samples,
    load_architexture_binary_overview as load_architexture_binary_overview,
)
from rwtd_sam3.eval.experiment_terms import (
    ExperimentTermsSection,
    format_shape_mapping,
    render_experiment_terms_markdown,
    render_relative_path,
)
from rwtd_sam3.eval.few_shot_subsets import (
    FewShotSubsetManifest,
    load_few_shot_subset_manifest,
    select_items_by_few_shot_manifest,
)
from rwtd_sam3.eval.core import build_quality_summary_record, render_quality_summary_markdown
from rwtd_sam3.eval.frozen_mask_head_augmentations import (
    apply_train_augmentation_to_binary_sample,
    describe_train_augmentation_policy,
    resolve_train_augmentation_policy,
)
from rwtd_sam3.eval.metrics import (
    CANONICAL_EVALUATION_CONTRACT,
    boundary_from_region_masks,
    build_canonical_evaluation_fields,
    compute_binary_metrics,
)
from rwtd_sam3.eval.runner import (
    SampleEvaluationResult,
    append_dataset_partition_fields,
    build_visual_record,
    discovered_package_versions,
    resolve_dataset_partition,
    resolve_eval_sample_count,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from rwtd_sam3.eval.sam3_auto import select_best_binary_assignment
from rwtd_sam3.models.autosam_upstream import load_upstream_autosam_resize_longest_side_class
from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import Sam3CoarseVsFineScaleFeatureExtractor
from rwtd_sam3.models.sam3_frozen_multiscale_mask_head import (
    FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS,
    FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS,
    FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS,
    FROZEN_MASK_HEAD_LOSS_VARIANTS,
    FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS,
    FROZEN_MASK_HEAD_SETTINGS,
    FROZEN_MASK_HEAD_VARIANT_SPECS,
    GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS,
    GLAS_FROZEN_FEATURE_PROBE_SETTINGS,
    GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS,
    FrozenForegroundProbeOutput,
    FrozenMaskHeadOutput,
    FrozenResidualMaskHeadOutput,
    FrozenMaskHeadRuntimeError,
    FrozenSamCoarsePlusAttentionRefineMaskHead,
    FrozenSamCoarsePlusCrossAttentionRefineMaskHead,
    FrozenSamCoarsePlusFineResidualMaskHead,
    FrozenSamForegroundProbe,
    FrozenSamMemoryAttentionMaskHead,
    FrozenSamMultiscaleMaskHead,
    MaskHeadLossResult,
    ResidualHeadCombinationResult,
    ResidualHeadTrainingLossResult,
    compute_bce_dice_loss,
    build_residual_head_settings_payload,
    combine_residual_head_logits,
    compute_residual_head_training_loss,
    count_trainable_parameters,
    is_coarse_plus_residual_variant,
    is_attention_refine_variant,
    is_cross_attention_refine_variant,
    is_memory_attention_variant,
    is_memory_control_variant,
    is_memory_head_variant,
    resolve_coarse_plus_residual_levels,
    resolve_frozen_mask_head_head_family,
    resolve_glas_frozen_feature_probe_variant,
    resolve_frozen_mask_head_variant_levels,
    resolve_residual_head_gate_mode,
    resolve_residual_head_gate_threshold,
    summarize_residual_head_combination,
    threshold_foreground_logits,
)
from rwtd_sam3.utils.visualization import save_prediction_panel


LOGGER = logging.getLogger(__name__)

STLD_DATASET_ID = "architexture:stld"
STLD_ROUTE = "stld"
DATASET_DISPLAY_NAME = "STLD"
FOREGROUND_LABEL = "foreground"
BACKGROUND_LABEL = "background"
GLAS_DATASET_ID = STLD_DATASET_ID
GLAS_FROZEN_MASK_HEAD_OUTPUT_ROOT = Path("outputs") / "architexture_binary" / "frozen_sam_mask_head" / "stld"
GLAS_FROZEN_MASK_HEAD_FEATURE_CACHE_MAX_BYTES = 5 << 30
GLAS_FROZEN_MASK_HEAD_BACKBONE_PREPROCESSING_MODES = ("native_resolution", "autosam_style_1024")
AUTOSAM_STYLE_BACKBONE_LONG_SIDE = 1024
GLAS_FROZEN_MASK_HEAD_SCALAR_FIELDS = (
    "direct_foreground_iou",
    "direct_foreground_dice",
    "direct_foreground_precision",
    "direct_foreground_recall",
    "eval_miou",
    "eval_ari",
    "predicted_positive_fraction",
    "target_positive_fraction",
)
GLAS_FROZEN_MASK_HEAD_EXPERIMENT_CONTRACT_PATH = Path("experiments") / "coarse_vs_fine_sam_scales" / "experiment_contract.md"


@dataclass(frozen=True)
class CachedGlasFeatureSample:
    """One decoded STLD sample plus cached raw frozen SAM pyramid features."""

    sample: GlasBinarySample
    image_size: tuple[int, int]
    backbone_input_size: tuple[int, int]
    raw_level_names: tuple[str, ...]
    raw_level_shapes: dict[str, tuple[int, int, int]]
    selected_level_names: tuple[str, ...]
    selected_level_shapes: dict[str, tuple[int, int, int]]
    selected_feature_levels: dict[str, np.ndarray]
    supervision_foreground_mask: np.ndarray
    supervision_background_mask: np.ndarray
    supervision_boundary_mask: np.ndarray


@dataclass(frozen=True)
class GlasFeatureCache:
    """Cached frozen SAM features for one STLD split and one scale selection."""

    dataset_name: str
    cache_dir: str | None
    split: str
    extractor: Sam3CoarseVsFineScaleFeatureExtractor
    selected_level_names: tuple[str, ...]
    reference_raw_level_shapes: dict[str, tuple[int, int, int]]
    reference_selected_level_shapes: dict[str, tuple[int, int, int]]
    samples: tuple[GlasBinarySample, ...]
    cached_feature_samples: tuple[CachedGlasFeatureSample, ...] | None
    materialization_mode: str
    resize_hw: tuple[int, int] | None
    backbone_preprocessing_mode: str


@dataclass(frozen=True)
class GlasMaskHeadTrainingArtifacts:
    """Persistent artifacts from one frozen-mask-head training run."""

    checkpoint_path: Path
    trainable_parameter_count: int
    level_input_dims: dict[str, int]
    training_history_rows: list[dict[str, Any]]
    epoch_eval_history_rows: list[dict[str, Any]]
    best_checkpoint_path: Path | None
    best_epoch_by_eval_dice: int | None
    best_eval_dice: float | None


@dataclass(frozen=True)
class GlasMaskHeadEvalBundle:
    """Per-sample eval payload stored in the standard run directory."""

    row: dict[str, Any]
    metric_summary: str
    foreground_prediction: np.ndarray
    background_prediction: np.ndarray
    foreground_probability: np.ndarray


def iter_glas_binary_samples(
    *,
    split: str,
    dataset_name: str,
    cache_dir: str | None,
    limit: int | None = None,
    start_index: int = 0,
):
    del cache_dir
    yield from iter_architexture_binary_samples(
        route=STLD_ROUTE,
        benchmark_root=dataset_name,
        split=split,
        require_official_split=True,
        limit=limit,
        start_index=start_index,
    )


def load_glas_binary_overview(*, split: str, dataset_name: str, cache_dir: str | None):
    del cache_dir
    return load_architexture_binary_overview(
        route=STLD_ROUTE,
        benchmark_root=dataset_name,
        split=split,
        require_official_split=True,
    )


def resolve_mask_head_projection_dim(args) -> int:
    return int(getattr(args, "projection_dim", FROZEN_MASK_HEAD_SETTINGS["projection_dim"]))


def resolve_mask_head_decoder_dim(args) -> int:
    return int(getattr(args, "decoder_dim", FROZEN_MASK_HEAD_SETTINGS["decoder_dim"]))


def resolve_mask_head_group_norm_groups(args) -> int:
    return int(getattr(args, "group_norm_groups", FROZEN_MASK_HEAD_SETTINGS["group_norm_groups"]))


def resolve_mask_head_memory_token_count(args) -> int:
    return int(getattr(args, "memory_token_count", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"]))


def resolve_mask_head_attention_heads(args) -> int:
    return int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"]))


def resolve_mask_head_attention_blocks(args) -> int:
    return int(getattr(args, "attention_blocks", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"]))


def resolve_train_subset_manifest(args) -> FewShotSubsetManifest | None:
    manifest_path = getattr(args, "train_subset_manifest", None)
    if manifest_path in (None, ""):
        return None
    return load_few_shot_subset_manifest(
        manifest_path,
        expected_dataset_id=STLD_DATASET_ID,
        expected_source_split=str(getattr(args, "train_split", "train")),
    )


def resolve_train_selection_policy(args) -> str:
    if resolve_train_subset_manifest(args) is None:
        return "random_50_50_seed0_prepared_root"
    return "few_shot_subset_manifest_over_random_50_50_seed0_prepared_root"


def resolve_mask_head_resize_hw(args) -> tuple[int, int] | None:
    resize_height = getattr(args, "resize_height", None)
    resize_width = getattr(args, "resize_width", None)
    if resize_height is None and resize_width is None:
        return None
    if resize_height is None or resize_width is None:
        raise FrozenMaskHeadRuntimeError(
            "resize_height and resize_width must either both be set or both be omitted.",
            diagnostics={
                "resize_height": resize_height,
                "resize_width": resize_width,
            },
        )
    resolved = (int(resize_height), int(resize_width))
    if resolved[0] < 1 or resolved[1] < 1:
        raise FrozenMaskHeadRuntimeError(
            "resize_height and resize_width must both be positive.",
            diagnostics={"resize_hw": resolved},
        )
    return resolved


def resolve_mask_head_backbone_preprocessing_mode(args) -> str:
    mode = str(getattr(args, "backbone_preprocessing_mode", "native_resolution"))
    if mode not in GLAS_FROZEN_MASK_HEAD_BACKBONE_PREPROCESSING_MODES:
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported frozen-mask-head backbone_preprocessing_mode '{mode}'.",
            diagnostics={
                "supported_backbone_preprocessing_modes": list(GLAS_FROZEN_MASK_HEAD_BACKBONE_PREPROCESSING_MODES)
            },
        )
    if mode != "native_resolution" and resolve_mask_head_resize_hw(args) is not None:
        raise FrozenMaskHeadRuntimeError(
            "Explicit resize_height/resize_width cannot be combined with non-native backbone_preprocessing_mode.",
            diagnostics={
                "backbone_preprocessing_mode": mode,
                "resize_hw": resolve_mask_head_resize_hw(args),
            },
        )
    return mode


def resolve_mask_head_preprocessing_summary(args) -> str:
    backbone_preprocessing_mode = resolve_mask_head_backbone_preprocessing_mode(args)
    resize_hw = resolve_mask_head_resize_hw(args)
    if resize_hw is None:
        if backbone_preprocessing_mode == "native_resolution":
            return "native_resolution"
        return (
            f"{backbone_preprocessing_mode}:"
            f"ResizeLongestSide({AUTOSAM_STYLE_BACKBONE_LONG_SIDE}) image before frozen SAM feature extraction; "
            "nearest-neighbor mask resize for supervision; predictions restored to original resolution for eval"
        )
    return (
        f"resize_to_{resize_hw[0]}x{resize_hw[1]}:"
        "image=bilinear,mask=nearest,before_feature_extraction_and_eval"
    )


def resolve_mask_head_eval_every_epochs(args) -> int | None:
    value = getattr(args, "eval_every_epochs", None)
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise FrozenMaskHeadRuntimeError(
            "eval_every_epochs must be positive when provided.",
            diagnostics={"eval_every_epochs": value},
        )
    return resolved


def resolve_mask_head_loss_variant(args) -> str:
    variant = str(getattr(args, "loss_variant", "bce_dice"))
    if variant not in FROZEN_MASK_HEAD_LOSS_VARIANTS:
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported frozen-mask-head loss variant '{variant}'.",
            diagnostics={"supported_loss_variants": list(FROZEN_MASK_HEAD_LOSS_VARIANTS)},
        )
    return variant


def resolve_mask_head_boundary_weight(args) -> float:
    value = float(getattr(args, "boundary_weight", FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"]))
    if value < 0.0:
        raise FrozenMaskHeadRuntimeError(
            "boundary_weight must be non-negative when provided.",
            diagnostics={"boundary_weight": value},
        )
    return value


def resolve_mask_head_loss_name(args) -> str:
    loss_variant = resolve_mask_head_loss_variant(args)
    if loss_variant == "boundary_weighted_bce_dice":
        return (
            "Boundary-weighted BCEWithLogits + Dice "
            f"(boundary_weight={resolve_mask_head_boundary_weight(args):.3f})"
        )
    return "BCEWithLogits + Dice"


def inspect_and_enforce_frozen_extractor_backbone(extractor: Any) -> dict[str, Any]:
    """Freeze the extractor SAM backend explicitly and report parameter counts."""

    stage1_runner = getattr(extractor, "_stage1_runner", None)
    sam3_runner = getattr(stage1_runner, "_sam3_runner", None)
    ensure_backend = getattr(sam3_runner, "_ensure_official_backend", None)
    if not callable(ensure_backend):
        return {
            "backbone_parameter_count": None,
            "backbone_frozen_parameter_count": None,
            "backbone_trainable_parameter_count": None,
            "backbone_requires_grad_was_zero_before_enforcement": None,
            "backbone_frozen_enforced": False,
            "backbone_contract_source": None,
        }
    backend = ensure_backend()
    model = getattr(backend, "model", None)
    if model is None or not hasattr(model, "parameters"):
        return {
            "backbone_parameter_count": None,
            "backbone_frozen_parameter_count": None,
            "backbone_trainable_parameter_count": None,
            "backbone_requires_grad_was_zero_before_enforcement": None,
            "backbone_frozen_enforced": False,
            "backbone_contract_source": None,
        }
    parameters = tuple(model.parameters())
    total_parameters = int(sum(int(parameter.numel()) for parameter in parameters))
    initial_trainable = int(sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad))
    for parameter in parameters:
        parameter.requires_grad = False
    final_trainable = int(sum(int(parameter.numel()) for parameter in parameters if parameter.requires_grad))
    model.eval()
    return {
        "backbone_parameter_count": total_parameters,
        "backbone_frozen_parameter_count": total_parameters - final_trainable,
        "backbone_trainable_parameter_count": final_trainable,
        "backbone_requires_grad_was_zero_before_enforcement": initial_trainable == 0,
        "backbone_frozen_enforced": True,
        "backbone_contract_source": "sam3_feature_extractor_official_backend.model",
    }


def resize_glas_binary_sample(sample: GlasBinarySample, *, resize_hw: tuple[int, int] | None) -> GlasBinarySample:
    """Resize one decoded STLD sample for the mask-head path, if requested."""

    if resize_hw is None:
        return sample
    target_height, target_width = (int(resize_hw[0]), int(resize_hw[1]))
    resized_image = sample.image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    nucleus_mask_uint8 = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    resized_foreground_mask = (
        np.asarray(
            Image.fromarray(nucleus_mask_uint8, mode="L").resize(
                (target_width, target_height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )
    resized_background_mask = np.logical_not(resized_foreground_mask)
    if not resized_foreground_mask.any() or not resized_background_mask.any():
        raise FrozenMaskHeadRuntimeError(
            "Resizing the STLD sample produced an empty foreground or background mask.",
            diagnostics={
                "crop_name": sample.crop_name,
                "resize_hw": (target_height, target_width),
                "foreground_positive_pixels": int(resized_foreground_mask.sum()),
                "background_positive_pixels": int(resized_background_mask.sum()),
            },
        )
    resized_boundary_mask = boundary_from_region_masks(resized_foreground_mask, resized_background_mask)
    return GlasBinarySample(
        index=sample.index,
        route=sample.route,
        split=sample.split,
        crop_name=sample.crop_name,
        image=resized_image,
        label_mask=np.asarray(resized_foreground_mask, dtype=np.uint8) * 255,
        label_values=sample.label_values,
        boundary_mask=np.asarray(resized_boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(resized_foreground_mask, dtype=bool),
        texture_b_mask=np.asarray(resized_background_mask, dtype=bool),
        texture_a=sample.texture_a,
        texture_b=sample.texture_b,
        original_texture_a=sample.original_texture_a,
        original_texture_b=sample.original_texture_b,
        oracle_points_a=sample.oracle_points_a,
        oracle_points_b=sample.oracle_points_b,
        evaluation_view=sample.evaluation_view,
    )


def apply_backbone_preprocessing_to_glas_binary_sample(
    sample: GlasBinarySample,
    *,
    backbone_preprocessing_mode: str,
) -> GlasBinarySample:
    if backbone_preprocessing_mode == "native_resolution":
        return sample
    if backbone_preprocessing_mode != "autosam_style_1024":
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported backbone preprocessing mode '{backbone_preprocessing_mode}'.",
            diagnostics={"backbone_preprocessing_mode": backbone_preprocessing_mode},
        )

    resize_longest_side_class = load_upstream_autosam_resize_longest_side_class()
    target_height, target_width = resize_longest_side_class.get_preprocess_shape(
        int(sample.height),
        int(sample.width),
        AUTOSAM_STYLE_BACKBONE_LONG_SIDE,
    )
    resized_image_array = resize_longest_side_class(AUTOSAM_STYLE_BACKBONE_LONG_SIDE).apply_image(
        np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
    )
    resized_image = Image.fromarray(np.asarray(resized_image_array, dtype=np.uint8), mode="RGB")
    if resized_image.size != (int(target_width), int(target_height)):
        raise FrozenMaskHeadRuntimeError(
            "AutoSAM-style backbone preprocessing produced an unexpected resized image size.",
            diagnostics={
                "expected_size_wh": (int(target_width), int(target_height)),
                "actual_size_wh": resized_image.size,
                "crop_name": sample.crop_name,
            },
        )

    foreground_mask_uint8 = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    resized_foreground_mask = (
        np.asarray(
            Image.fromarray(foreground_mask_uint8, mode="L").resize(
                (int(target_width), int(target_height)),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )
    resized_background_mask = np.logical_not(resized_foreground_mask)
    if not resized_foreground_mask.any() or not resized_background_mask.any():
        raise FrozenMaskHeadRuntimeError(
            "AutoSAM-style backbone preprocessing produced an empty foreground or background mask.",
            diagnostics={
                "crop_name": sample.crop_name,
                "target_height": int(target_height),
                "target_width": int(target_width),
                "foreground_positive_pixels": int(resized_foreground_mask.sum()),
                "background_positive_pixels": int(resized_background_mask.sum()),
                "backbone_preprocessing_mode": backbone_preprocessing_mode,
            },
        )
    resized_boundary_mask = boundary_from_region_masks(resized_foreground_mask, resized_background_mask)
    return GlasBinarySample(
        index=sample.index,
        route=sample.route,
        split=sample.split,
        crop_name=sample.crop_name,
        image=resized_image,
        label_mask=np.asarray(resized_foreground_mask, dtype=np.uint8) * 255,
        label_values=sample.label_values,
        boundary_mask=np.asarray(resized_boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(resized_foreground_mask, dtype=bool),
        texture_b_mask=np.asarray(resized_background_mask, dtype=bool),
        texture_a=sample.texture_a,
        texture_b=sample.texture_b,
        original_texture_a=sample.original_texture_a,
        original_texture_b=sample.original_texture_b,
        oracle_points_a=sample.oracle_points_a,
        oracle_points_b=sample.oracle_points_b,
        evaluation_view=sample.evaluation_view,
    )


def resize_foreground_probability_to_sample_resolution(
    foreground_probability: np.ndarray,
    *,
    sample: GlasBinarySample,
) -> np.ndarray:
    probability_uint8 = np.clip(np.asarray(foreground_probability, dtype=np.float32), 0.0, 1.0)
    resized = Image.fromarray(np.rint(probability_uint8 * 255.0).astype(np.uint8), mode="L").resize(
        (int(sample.width), int(sample.height)),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def prepare_glas_frozen_mask_head_output_dir(
    output_dir: str | None,
    *,
    variant: str,
    run_kind: str,
    split: str | None,
) -> Path:
    if output_dir is None:
        suffix = f"{run_kind}_{variant}"
        if split is not None:
            suffix = f"{suffix}_{split}"
        return GLAS_FROZEN_MASK_HEAD_OUTPUT_ROOT / suffix
    return Path(output_dir)


def run_glas_frozen_mask_head_train(args) -> dict[str, Any]:
    """Train the tiny supervised mask head on frozen SAM features and evaluate on STLD."""

    import gc

    if getattr(args, "train_limit", None) is not None and getattr(args, "train_subset_manifest", None) not in (None, ""):
        raise FrozenMaskHeadRuntimeError(
            "train_limit and train_subset_manifest are mutually exclusive for STLD frozen-mask-head training.",
            diagnostics={
                "train_limit": getattr(args, "train_limit", None),
                "train_subset_manifest": getattr(args, "train_subset_manifest", None),
            },
        )

    augmentation_policy = resolve_train_augmentation_policy(args)
    eval_every_epochs = resolve_mask_head_eval_every_epochs(args)
    train_subset_manifest = resolve_train_subset_manifest(args)
    setattr(
        args,
        "_resolved_train_subset_manifest",
        train_subset_manifest.to_json_dict() if train_subset_manifest is not None else None,
    )
    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )
    train_cache = build_glas_feature_cache(
        extractor=extractor,
        dataset_name=args.benchmark_root,
        cache_dir=None,
        split=args.train_split,
        limit=args.train_limit,
        variant=args.variant,
        subset_manifest=train_subset_manifest,
        resize_hw=resolve_mask_head_resize_hw(args),
        backbone_preprocessing_mode=resolve_mask_head_backbone_preprocessing_mode(args),
        skip_feature_materialization=(augmentation_policy != "none"),
    )
    setattr(args, "_backbone_contract", inspect_and_enforce_frozen_extractor_backbone(extractor))

    output_dir = prepare_glas_frozen_mask_head_output_dir(
        args.output_dir,
        variant=args.variant,
        run_kind="train",
        split=args.eval_split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_subset_manifest is not None:
        local_manifest_path = output_dir / "train_subset_manifest.json"
        write_json(local_manifest_path, train_subset_manifest.to_json_dict())
        setattr(args, "_resolved_train_subset_manifest_output_path", str(local_manifest_path))
    else:
        setattr(args, "_resolved_train_subset_manifest_output_path", None)

    eval_cache: GlasFeatureCache | None = None
    if eval_every_epochs is not None:
        eval_cache = build_glas_feature_cache(
            extractor=extractor,
            dataset_name=args.benchmark_root,
            cache_dir=None,
            split=args.eval_split,
            limit=args.eval_limit,
            variant=args.variant,
            expected_selected_level_names=train_cache.selected_level_names,
            resize_hw=resolve_mask_head_resize_hw(args),
            backbone_preprocessing_mode=resolve_mask_head_backbone_preprocessing_mode(args),
        )

    head_module, training_artifacts = train_glas_frozen_mask_head(
        train_cache=train_cache,
        output_dir=output_dir,
        args=args,
        eval_cache_for_tracking=eval_cache,
    )
    train_summary = evaluate_glas_frozen_mask_head_on_cache(
        head_module=head_module,
        cache=train_cache,
        variant=args.variant,
        args=args,
        checkpoint_path=training_artifacts.checkpoint_path,
        run_kind="train_internal_eval",
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=None,
        save_artifacts=False,
    )
    write_json(output_dir / "train_set_summary.json", train_summary)
    train_sample_count = len(train_cache.samples)
    selected_level_names = train_cache.selected_level_names
    reference_raw_level_shapes = train_cache.reference_raw_level_shapes
    reference_selected_level_shapes = train_cache.reference_selected_level_shapes
    train_materialization_mode = train_cache.materialization_mode
    del train_cache
    gc.collect()
    if eval_cache is None:
        eval_cache = build_glas_feature_cache(
            extractor=extractor,
            dataset_name=args.benchmark_root,
            cache_dir=None,
            split=args.eval_split,
            limit=args.eval_limit,
            variant=args.variant,
            expected_selected_level_names=selected_level_names,
            resize_hw=resolve_mask_head_resize_hw(args),
            backbone_preprocessing_mode=resolve_mask_head_backbone_preprocessing_mode(args),
        )
    config = build_glas_frozen_mask_head_run_config(
        args,
        run_kind="train",
        selected_level_names=selected_level_names,
        reference_raw_level_shapes=reference_raw_level_shapes,
        reference_selected_level_shapes=reference_selected_level_shapes,
        materialization_mode=f"train={train_materialization_mode};eval={eval_cache.materialization_mode}",
    )
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_glas_frozen_mask_head_experiment_terms_markdown(
            args,
            run_kind="train",
            selected_level_names=selected_level_names,
            reference_raw_level_shapes=reference_raw_level_shapes,
            reference_selected_level_shapes=reference_selected_level_shapes,
            train_sample_count=train_sample_count,
            eval_sample_count=len(eval_cache.samples),
            checkpoint_path=None,
        ),
    )
    summary = evaluate_glas_frozen_mask_head_on_cache(
        head_module=head_module,
        cache=eval_cache,
        variant=args.variant,
        args=args,
        checkpoint_path=training_artifacts.checkpoint_path,
        run_kind="train",
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=output_dir,
        save_artifacts=True,
    )
    summary["train_set_mean_metrics"] = dict(train_summary["mean_metrics"])
    summary["validation_set_mean_metrics"] = None
    summary["train_set_summary_path"] = str(output_dir / "train_set_summary.json")
    summary["train_augmentation_policy"] = augmentation_policy
    summary["train_augmentation_summary"] = describe_train_augmentation_policy(augmentation_policy)
    summary["epoch_eval_history_path"] = (
        str(output_dir / "epoch_eval_history.csv")
        if training_artifacts.epoch_eval_history_rows
        else None
    )
    summary["best_checkpoint_path"] = (
        str(training_artifacts.best_checkpoint_path)
        if training_artifacts.best_checkpoint_path is not None
        else None
    )
    summary["best_epoch_by_eval_dice"] = training_artifacts.best_epoch_by_eval_dice
    summary["best_eval_dice"] = training_artifacts.best_eval_dice
    summary["best_epoch_selection_metric"] = (
        "direct_foreground_dice" if training_artifacts.best_epoch_by_eval_dice is not None else None
    )
    summary["best_epoch_selection_caveat"] = (
        "analysis_only_when_eval_split_is_test"
        if training_artifacts.best_epoch_by_eval_dice is not None and str(args.eval_split) == "test"
        else None
    )
    summary["selection_split"] = str(getattr(args, "eval_split", "test"))
    summary["headline_split"] = str(getattr(args, "eval_split", "test"))
    summary["quality_summary"] = build_quality_summary_record(summary)
    rewrite_glas_frozen_mask_head_summary(output_dir=output_dir, summary=summary)
    return summary


def run_glas_frozen_mask_head_eval(args) -> dict[str, Any]:
    """Evaluate one saved frozen-mask-head checkpoint on an STLD split."""

    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen-mask-head checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_variant = str(checkpoint["variant"])
    setattr(args, "variant", checkpoint_variant)
    setattr(args, "learning_rate", float(checkpoint.get("learning_rate", FROZEN_MASK_HEAD_SETTINGS["learning_rate"])))
    setattr(args, "weight_decay", float(checkpoint.get("weight_decay", FROZEN_MASK_HEAD_SETTINGS["weight_decay"])))
    setattr(args, "num_epochs", int(checkpoint.get("num_epochs", FROZEN_MASK_HEAD_SETTINGS["num_epochs"])))
    setattr(args, "bce_weight", float(checkpoint.get("bce_weight", FROZEN_MASK_HEAD_SETTINGS["bce_weight"])))
    setattr(args, "dice_weight", float(checkpoint.get("dice_weight", FROZEN_MASK_HEAD_SETTINGS["dice_weight"])))
    setattr(args, "train_augmentation_policy", str(checkpoint.get("train_augmentation_policy", "none")))
    setattr(args, "loss_variant", str(checkpoint.get("loss_variant", "bce_dice")))
    setattr(
        args,
        "boundary_weight",
        float(checkpoint.get("boundary_weight", FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS["boundary_weight"])),
    )
    setattr(
        args,
        "backbone_preprocessing_mode",
        str(checkpoint.get("backbone_preprocessing_mode", "native_resolution")),
    )
    if checkpoint.get("model_family", "multiscale_mask_head") != "foreground_probe":
        setattr(args, "projection_dim", int(checkpoint.get("projection_dim", FROZEN_MASK_HEAD_SETTINGS["projection_dim"])))
        setattr(args, "decoder_dim", int(checkpoint.get("decoder_dim", FROZEN_MASK_HEAD_SETTINGS["decoder_dim"])))
        setattr(
            args,
            "group_norm_groups",
            int(checkpoint.get("group_norm_groups", FROZEN_MASK_HEAD_SETTINGS["group_norm_groups"])),
        )
    setattr(
        args,
        "memory_token_count",
        int(checkpoint.get("memory_token_count", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"])),
    )
    setattr(
        args,
        "attention_heads",
        int(checkpoint.get("attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])),
    )
    setattr(
        args,
        "attention_blocks",
        int(checkpoint.get("attention_blocks", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"])),
    )
    setattr(
        args,
        "_backbone_contract",
        {
            "backbone_parameter_count": checkpoint.get("backbone_parameter_count"),
            "backbone_frozen_parameter_count": checkpoint.get("backbone_frozen_parameter_count"),
            "backbone_trainable_parameter_count": checkpoint.get("backbone_trainable_parameter_count"),
            "backbone_requires_grad_was_zero_before_enforcement": checkpoint.get(
                "backbone_requires_grad_was_zero_before_enforcement"
            ),
            "backbone_frozen_enforced": checkpoint.get("backbone_frozen_enforced"),
            "backbone_contract_source": checkpoint.get("backbone_contract_source"),
        },
    )
    if getattr(args, "foreground_threshold", None) == 0.5 and "foreground_threshold" in checkpoint:
        setattr(args, "foreground_threshold", float(checkpoint["foreground_threshold"]))
    checkpoint_resize_height = checkpoint.get("resize_height")
    checkpoint_resize_width = checkpoint.get("resize_width")
    checkpoint_resize_hw = (
        (int(checkpoint_resize_height), int(checkpoint_resize_width))
        if checkpoint_resize_height is not None and checkpoint_resize_width is not None
        else None
    )
    cli_resize_hw = resolve_mask_head_resize_hw(args)
    if checkpoint_resize_hw is not None:
        if cli_resize_hw is not None and tuple(cli_resize_hw) != tuple(checkpoint_resize_hw):
            raise FrozenMaskHeadRuntimeError(
                "Eval resize arguments do not match the checkpoint preprocessing protocol.",
                diagnostics={
                    "checkpoint_resize_hw": checkpoint_resize_hw,
                    "cli_resize_hw": cli_resize_hw,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
        setattr(args, "resize_height", int(checkpoint_resize_hw[0]))
        setattr(args, "resize_width", int(checkpoint_resize_hw[1]))

    overview = load_glas_binary_overview(
        split=args.split,
        dataset_name=args.benchmark_root,
        cache_dir=None,
    )
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_samples = resolve_eval_sample_count(args.limit, overview.num_examples, dataset_partition)
    if num_samples < 1:
        raise FrozenMaskHeadRuntimeError("The requested STLD eval split produced zero samples.")

    extractor = Sam3CoarseVsFineScaleFeatureExtractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )
    setattr(args, "_backbone_contract", inspect_and_enforce_frozen_extractor_backbone(extractor))
    eval_cache = build_glas_feature_cache(
        extractor=extractor,
        dataset_name=args.benchmark_root,
        cache_dir=None,
        split=args.split,
        limit=num_samples,
        variant=checkpoint_variant,
        expected_selected_level_names=tuple(checkpoint["selected_level_names"]),
        start_index=dataset_partition.start_index if dataset_partition is not None else 0,
        resize_hw=resolve_mask_head_resize_hw(args),
        backbone_preprocessing_mode=str(checkpoint.get("backbone_preprocessing_mode", "native_resolution")),
    )

    output_dir = prepare_glas_frozen_mask_head_output_dir(
        args.output_dir,
        variant=checkpoint_variant,
        run_kind="eval",
        split=args.split,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_glas_frozen_mask_head_run_config(
        args,
        run_kind="eval",
        selected_level_names=eval_cache.selected_level_names,
        reference_raw_level_shapes=eval_cache.reference_raw_level_shapes,
        reference_selected_level_shapes=eval_cache.reference_selected_level_shapes,
        materialization_mode=eval_cache.materialization_mode,
        dataset_partition=dataset_partition,
        selected_sample_count=num_samples,
    )
    config["checkpoint_path"] = str(checkpoint_path)
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_glas_frozen_mask_head_experiment_terms_markdown(
            args,
            run_kind="eval",
            selected_level_names=eval_cache.selected_level_names,
            reference_raw_level_shapes=eval_cache.reference_raw_level_shapes,
            reference_selected_level_shapes=eval_cache.reference_selected_level_shapes,
            train_sample_count=None,
            eval_sample_count=len(eval_cache.samples),
            checkpoint_path=checkpoint_path,
        ),
    )

    head_module = build_frozen_mask_head_from_checkpoint(checkpoint)
    head_module.load_state_dict(checkpoint["head_state_dict"])
    summary = evaluate_glas_frozen_mask_head_on_cache(
        head_module=head_module,
        cache=eval_cache,
        variant=checkpoint_variant,
        args=args,
        checkpoint_path=checkpoint_path,
        run_kind="eval",
        trainable_parameter_count=int(checkpoint["trainable_parameter_count"]),
        output_dir=output_dir,
        save_artifacts=True,
    )
    append_dataset_partition_fields(summary, dataset_partition, selected_sample_count=num_samples)
    summary["selection_split"] = str(getattr(args, "split", "test"))
    summary["headline_split"] = str(getattr(args, "split", "test"))
    summary["quality_summary"] = build_quality_summary_record(summary)
    rewrite_glas_frozen_mask_head_summary(output_dir=output_dir, summary=summary)
    return summary


def build_glas_feature_cache(
    *,
    extractor: Sam3CoarseVsFineScaleFeatureExtractor,
    dataset_name: str,
    cache_dir: str | None,
    split: str,
    limit: int | None,
    variant: str,
    subset_manifest: FewShotSubsetManifest | None = None,
    expected_selected_level_names: tuple[str, ...] | None = None,
    start_index: int = 0,
    resize_hw: tuple[int, int] | None = None,
    backbone_preprocessing_mode: str = "native_resolution",
    skip_feature_materialization: bool = False,
) -> GlasFeatureCache:
    """Decode STLD samples and cache the raw frozen SAM pyramid levels."""

    raw_samples = tuple(
        resize_glas_binary_sample(
            sample,
            resize_hw=resize_hw,
        )
        for sample in iter_glas_binary_samples(
            split=split,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            limit=limit,
            start_index=start_index,
        )
    )
    raw_samples = select_items_by_few_shot_manifest(
        raw_samples,
        subset_manifest,
        item_id_getter=lambda sample: sample.crop_name,
    )
    if not raw_samples:
        raise FrozenMaskHeadRuntimeError(
            f"No STLD samples were loaded for split '{split}' with limit={limit} and start_index={start_index}."
        )

    reference_sample = prepare_cached_glas_feature_sample(
        sample=raw_samples[0],
        extractor=extractor,
        variant=variant,
        backbone_preprocessing_mode=backbone_preprocessing_mode,
        expected_selected_level_names=expected_selected_level_names,
        reference_selected_level_names=None,
        reference_channels=None,
    )
    selected_level_names = reference_sample.selected_level_names
    reference_raw_level_shapes = reference_sample.raw_level_shapes
    reference_selected_level_shapes = reference_sample.selected_level_shapes
    reference_channels = {
        level_name: int(reference_selected_level_shapes[level_name][0])
        for level_name in selected_level_names
    }
    estimated_bundle_bytes = estimate_cached_glas_feature_sample_nbytes(reference_sample)
    estimated_total_bytes = int(len(raw_samples)) * estimated_bundle_bytes
    if skip_feature_materialization:
        materialization_mode = "train_augmented_streamed_reextract"
    else:
        materialization_mode = (
            "streamed_reextract"
            if estimated_total_bytes > GLAS_FROZEN_MASK_HEAD_FEATURE_CACHE_MAX_BYTES
            else "in_memory"
        )

    cached_feature_samples: list[CachedGlasFeatureSample] | None = (
        [reference_sample] if materialization_mode == "in_memory" else None
    )
    if cached_feature_samples is not None:
        for sample in raw_samples[1:]:
            cached_feature_samples.append(
                prepare_cached_glas_feature_sample(
                    sample=sample,
                    extractor=extractor,
                    variant=variant,
                    backbone_preprocessing_mode=backbone_preprocessing_mode,
                    expected_selected_level_names=expected_selected_level_names,
                    reference_selected_level_names=selected_level_names,
                    reference_channels=reference_channels,
                )
            )

    if selected_level_names is None or reference_raw_level_shapes is None or reference_selected_level_shapes is None:
        raise FrozenMaskHeadRuntimeError("Failed to build a non-empty STLD frozen-feature cache.")

    LOGGER.info(
        "Prepared %d STLD samples for split=%s | levels=%s | materialization_mode=%s | estimated_bundle_mb=%.2f | estimated_total_gb=%.2f | reference_raw_shapes=%s | reference_selected_shapes=%s | resize_hw=%s | backbone_preprocessing_mode=%s",
        len(raw_samples),
        split,
        list(selected_level_names),
        materialization_mode,
        float(estimated_bundle_bytes / (1024 * 1024)),
        float(estimated_total_bytes / (1024 * 1024 * 1024)),
        reference_raw_level_shapes,
        reference_selected_level_shapes,
        resize_hw,
        backbone_preprocessing_mode,
    )
    return GlasFeatureCache(
        dataset_name=str(dataset_name),
        cache_dir=cache_dir,
        split=split,
        extractor=extractor,
        selected_level_names=selected_level_names,
        reference_raw_level_shapes=reference_raw_level_shapes,
        reference_selected_level_shapes=reference_selected_level_shapes,
        samples=tuple(raw_samples),
        cached_feature_samples=tuple(cached_feature_samples) if cached_feature_samples is not None else None,
        materialization_mode=materialization_mode,
        resize_hw=resize_hw,
        backbone_preprocessing_mode=backbone_preprocessing_mode,
    )


def build_frozen_mask_head_from_cache(cache: GlasFeatureCache, *, variant: str, args=None) -> Any:
    level_input_dims = {
        level_name: int(cache.reference_selected_level_shapes[level_name][0])
        for level_name in cache.selected_level_names
    }
    if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS:
        variant_spec = GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]
        return FrozenSamForegroundProbe(
            level_input_dims=level_input_dims,
            head_kind=str(variant_spec["head_kind"]),
            projection_dim=int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["projection_dim"]),
            hidden_dim=int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["hidden_dim"]),
        ).module
    if is_coarse_plus_residual_variant(variant):
        coarse_level_name, fine_level_name = resolve_coarse_plus_residual_levels(variant)
        if is_cross_attention_refine_variant(variant):
            return FrozenSamCoarsePlusCrossAttentionRefineMaskHead(
                level_input_dims=level_input_dims,
                coarse_level_name=coarse_level_name,
                fine_level_name=fine_level_name,
                projection_dim=resolve_mask_head_projection_dim(args),
                decoder_dim=resolve_mask_head_decoder_dim(args),
                group_norm_groups=resolve_mask_head_group_norm_groups(args),
                residual_projection_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
                residual_hidden_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
                attention_hidden_dim=int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
                attention_heads=int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])),
                cross_attn_query_stride=int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])),
                residual_scale_init=float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
            ).module
        if is_attention_refine_variant(variant):
            return FrozenSamCoarsePlusAttentionRefineMaskHead(
                level_input_dims=level_input_dims,
                coarse_level_name=coarse_level_name,
                fine_level_name=fine_level_name,
                projection_dim=resolve_mask_head_projection_dim(args),
                decoder_dim=resolve_mask_head_decoder_dim(args),
                group_norm_groups=resolve_mask_head_group_norm_groups(args),
                residual_projection_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
                residual_hidden_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
                attention_hidden_dim=int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
                residual_scale_init=float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
            ).module
        return FrozenSamCoarsePlusFineResidualMaskHead(
            level_input_dims=level_input_dims,
            coarse_level_name=coarse_level_name,
            fine_level_name=fine_level_name,
            projection_dim=resolve_mask_head_projection_dim(args),
            decoder_dim=resolve_mask_head_decoder_dim(args),
            group_norm_groups=resolve_mask_head_group_norm_groups(args),
            residual_projection_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
            residual_hidden_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
            residual_scale_init=float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
        ).module
    if is_memory_head_variant(variant):
        return FrozenSamMemoryAttentionMaskHead(
            level_input_dims=level_input_dims,
            projection_dim=resolve_mask_head_projection_dim(args),
            decoder_dim=resolve_mask_head_decoder_dim(args),
            group_norm_groups=resolve_mask_head_group_norm_groups(args),
            memory_token_count=resolve_mask_head_memory_token_count(args),
            attention_heads=resolve_mask_head_attention_heads(args),
            attention_blocks=resolve_mask_head_attention_blocks(args),
            memory_init_std=float(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_init_std"]),
            mixing_kind=("cross_attention" if is_memory_attention_variant(variant) else "global_memory_control"),
        ).module
    return FrozenSamMultiscaleMaskHead(
        level_input_dims=level_input_dims,
        projection_dim=resolve_mask_head_projection_dim(args),
        decoder_dim=resolve_mask_head_decoder_dim(args),
        group_norm_groups=resolve_mask_head_group_norm_groups(args),
    ).module


def build_frozen_mask_head_from_checkpoint(checkpoint: dict[str, Any]) -> Any:
    model_family = str(checkpoint.get("model_family", "multiscale_mask_head"))
    if model_family == "foreground_probe":
        return FrozenSamForegroundProbe(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            head_kind=str(checkpoint["head_kind"]),
            projection_dim=int(checkpoint["projection_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).module
    if model_family == "coarse_plus_fine_residual_mask_head":
        return FrozenSamCoarsePlusFineResidualMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            fine_level_name=str(checkpoint.get("fine_level_name", "fpn_0")),
            projection_dim=int(checkpoint["projection_dim"]),
            decoder_dim=int(checkpoint["decoder_dim"]),
            group_norm_groups=int(checkpoint["group_norm_groups"]),
            residual_projection_dim=int(
                checkpoint.get("residual_projection_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            ),
            residual_hidden_dim=int(
                checkpoint.get("residual_hidden_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            ),
            residual_scale_init=float(
                checkpoint.get("residual_scale_init", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            ),
        ).module
    if model_family == "coarse_plus_attention_refine_mask_head":
        return FrozenSamCoarsePlusAttentionRefineMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            fine_level_name=str(checkpoint.get("fine_level_name", "fpn_1")),
            projection_dim=int(checkpoint["projection_dim"]),
            decoder_dim=int(checkpoint["decoder_dim"]),
            group_norm_groups=int(checkpoint["group_norm_groups"]),
            residual_projection_dim=int(
                checkpoint.get("residual_projection_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            ),
            residual_hidden_dim=int(
                checkpoint.get("residual_hidden_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            ),
            attention_hidden_dim=int(
                checkpoint.get("attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])
            ),
            residual_scale_init=float(
                checkpoint.get("residual_scale_init", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            ),
        ).module
    if model_family == "coarse_plus_cross_attention_refine_mask_head":
        return FrozenSamCoarsePlusCrossAttentionRefineMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            fine_level_name=str(checkpoint.get("fine_level_name", "fpn_1")),
            projection_dim=int(checkpoint["projection_dim"]),
            decoder_dim=int(checkpoint["decoder_dim"]),
            group_norm_groups=int(checkpoint["group_norm_groups"]),
            residual_projection_dim=int(
                checkpoint.get("residual_projection_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            ),
            residual_hidden_dim=int(
                checkpoint.get("residual_hidden_dim", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            ),
            attention_hidden_dim=int(
                checkpoint.get("attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])
            ),
            attention_heads=int(
                checkpoint.get("attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])
            ),
            cross_attn_query_stride=int(
                checkpoint.get("cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])
            ),
            residual_scale_init=float(
                checkpoint.get("residual_scale_init", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            ),
        ).module
    if model_family in {"memory_attention_mask_head", "memory_control_mask_head"}:
        return FrozenSamMemoryAttentionMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            projection_dim=int(checkpoint["projection_dim"]),
            decoder_dim=int(checkpoint["decoder_dim"]),
            group_norm_groups=int(checkpoint["group_norm_groups"]),
            memory_token_count=int(
                checkpoint.get("memory_token_count", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_token_count"])
            ),
            attention_heads=int(
                checkpoint.get("attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])
            ),
            attention_blocks=int(
                checkpoint.get("attention_blocks", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_blocks"])
            ),
            memory_init_std=float(
                checkpoint.get("memory_init_std", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_init_std"])
            ),
            mixing_kind=("cross_attention" if model_family == "memory_attention_mask_head" else "global_memory_control"),
        ).module
    return FrozenSamMultiscaleMaskHead(
        level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
        projection_dim=int(checkpoint["projection_dim"]),
        decoder_dim=int(checkpoint["decoder_dim"]),
        group_norm_groups=int(checkpoint["group_norm_groups"]),
    ).module


def prepare_cached_glas_feature_sample(
    *,
    sample: GlasBinarySample,
    extractor: Sam3CoarseVsFineScaleFeatureExtractor,
    variant: str,
    backbone_preprocessing_mode: str,
    expected_selected_level_names: tuple[str, ...] | None,
    reference_selected_level_names: tuple[str, ...] | None,
    reference_channels: dict[str, int] | None,
) -> CachedGlasFeatureSample:
    """Extract and validate one STLD sample's raw frozen SAM multiscale features."""

    backbone_sample = apply_backbone_preprocessing_to_glas_binary_sample(
        sample,
        backbone_preprocessing_mode=backbone_preprocessing_mode,
    )
    image_size, pyramid = extractor.extract_sam_pyramid(backbone_sample.image)
    expected_image_size = (int(backbone_sample.height), int(backbone_sample.width))
    if tuple(int(value) for value in image_size) != expected_image_size:
        raise FrozenMaskHeadRuntimeError(
            "Frozen-mask-head feature extraction returned an image size that does not match the decoded STLD sample.",
            diagnostics={
                "reported_image_size": tuple(int(value) for value in image_size),
                "expected_image_size": expected_image_size,
                "crop_name": sample.crop_name,
                "backbone_preprocessing_mode": backbone_preprocessing_mode,
            },
        )
    raw_level_names = tuple(pyramid.keys())
    raw_level_shapes = {
        level_name: tuple(int(value) for value in feature_map.shape)
        for level_name, feature_map in pyramid.items()
    }
    resolved_selected = expected_selected_level_names or resolve_glas_frozen_variant_levels(
        variant=variant,
        available_level_names=raw_level_names,
    )
    missing_levels = [name for name in resolved_selected if name not in raw_level_names]
    if missing_levels:
        raise FrozenMaskHeadRuntimeError(
            "The requested frozen-mask-head levels were missing from the discovered SAM pyramid.",
            diagnostics={
                "missing_levels": missing_levels,
                "available_level_names": raw_level_names,
                "crop_name": sample.crop_name,
            },
        )
    if reference_selected_level_names is not None and tuple(resolved_selected) != tuple(reference_selected_level_names):
        raise FrozenMaskHeadRuntimeError(
            "Frozen-mask-head cache resolved inconsistent selected levels across samples.",
            diagnostics={
                "reference_selected_level_names": reference_selected_level_names,
                "current_selected_level_names": tuple(resolved_selected),
                "crop_name": sample.crop_name,
            },
        )

    selected_features = {
        level_name: np.asarray(pyramid[level_name], dtype=np.float16)
        for level_name in resolved_selected
    }
    selected_level_shapes = {
        level_name: tuple(int(value) for value in feature_map.shape)
        for level_name, feature_map in selected_features.items()
    }
    current_channels = {level_name: int(shape[0]) for level_name, shape in selected_level_shapes.items()}
    if reference_channels is not None and current_channels != reference_channels:
        raise FrozenMaskHeadRuntimeError(
            "Discovered inconsistent SAM channel counts across STLD samples.",
            diagnostics={
                "reference_channels": reference_channels,
                "current_channels": current_channels,
                "crop_name": sample.crop_name,
            },
        )
    return CachedGlasFeatureSample(
        sample=sample,
        image_size=(int(image_size[0]), int(image_size[1])),
        backbone_input_size=(int(image_size[0]), int(image_size[1])),
        raw_level_names=raw_level_names,
        raw_level_shapes=raw_level_shapes,
        selected_level_names=tuple(resolved_selected),
        selected_level_shapes=selected_level_shapes,
        selected_feature_levels=selected_features,
        supervision_foreground_mask=np.asarray(backbone_sample.texture_a_mask, dtype=bool),
        supervision_background_mask=np.asarray(backbone_sample.texture_b_mask, dtype=bool),
        supervision_boundary_mask=np.asarray(backbone_sample.boundary_mask, dtype=bool),
    )


def estimate_cached_glas_feature_sample_nbytes(sample: CachedGlasFeatureSample) -> int:
    """Estimate the raw CPU memory footprint of one cached STLD feature sample."""

    return int(sum(int(feature_map.nbytes) for feature_map in sample.selected_feature_levels.values()))


def iter_cached_glas_feature_samples(cache: GlasFeatureCache):
    """Yield prepared STLD feature samples, streaming re-extraction when needed."""

    if cache.cached_feature_samples is not None:
        yield from cache.cached_feature_samples
        return
    reference_channels = {
        level_name: int(cache.reference_selected_level_shapes[level_name][0])
        for level_name in cache.selected_level_names
    }
    for sample in cache.samples:
        yield prepare_cached_glas_feature_sample(
            sample=sample,
            extractor=cache.extractor,
            variant="all_scales",
            backbone_preprocessing_mode=cache.backbone_preprocessing_mode,
            expected_selected_level_names=cache.selected_level_names,
            reference_selected_level_names=cache.selected_level_names,
            reference_channels=reference_channels,
        )


def materialize_cached_feature_levels_on_device(
    sample: CachedGlasFeatureSample,
    *,
    device: Any,
) -> dict[str, Any]:
    """Move one cached STLD sample's selected raw feature pyramid onto a torch device."""

    import torch

    materialized: dict[str, Any] = {}
    for level_name in sample.selected_level_names:
        feature_map = np.asarray(sample.selected_feature_levels[level_name], dtype=np.float32)
        if feature_map.ndim != 3:
            raise FrozenMaskHeadRuntimeError(
                f"Cached STLD feature map must have shape [C,H,W], got {feature_map.shape} for {level_name}."
            )
        tensor = torch.as_tensor(feature_map[None], dtype=torch.float32, device=device)
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise FrozenMaskHeadRuntimeError(
                f"Materialized STLD feature tensor must have shape [1,C,H,W], got {tuple(int(value) for value in tensor.shape)}."
            )
        materialized[level_name] = tensor
    return materialized


def train_glas_frozen_mask_head(
    *,
    train_cache: GlasFeatureCache,
    output_dir: Path,
    args,
    eval_cache_for_tracking: GlasFeatureCache | None = None,
) -> tuple[Any, GlasMaskHeadTrainingArtifacts]:
    """Optimize only the tiny supervised mask head on cached frozen STLD features."""

    import torch

    if not train_cache.samples:
        raise FrozenMaskHeadRuntimeError("Frozen-mask-head training received an empty cached train split.")

    head_module = build_frozen_mask_head_from_cache(train_cache, variant=args.variant, args=args)
    device = resolve_torch_device(args.device)
    head_module.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head_module.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    trainable_parameter_count = count_trainable_parameters(head_module)
    training_history_rows: list[dict[str, Any]] = []
    epoch_eval_history_rows: list[dict[str, Any]] = []
    permutation_rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    augmentation_rng = np.random.default_rng(int(getattr(args, "seed", 0)) + 1)
    augmentation_policy = resolve_train_augmentation_policy(args)
    eval_every_epochs = resolve_mask_head_eval_every_epochs(args)
    loss_variant = resolve_mask_head_loss_variant(args)
    boundary_weight = resolve_mask_head_boundary_weight(args)
    backbone_contract = dict(getattr(args, "_backbone_contract", {}) or {})
    best_epoch_by_eval_dice: int | None = None
    best_eval_dice: float | None = None
    best_state_dict: dict[str, Any] | None = None

    LOGGER.info(
        "Frozen-mask-head training start | variant=%s levels=%s materialization_mode=%s trainable_params=%d backbone_params=%s backbone_trainable=%s lr=%.6f wd=%.6f epochs=%d train_aug_policy=%s loss_variant=%s boundary_weight=%.3f eval_every_epochs=%s",
        args.variant,
        list(train_cache.selected_level_names),
        train_cache.materialization_mode,
        trainable_parameter_count,
        backbone_contract.get("backbone_parameter_count"),
        backbone_contract.get("backbone_trainable_parameter_count"),
        float(args.learning_rate),
        float(args.weight_decay),
        int(args.num_epochs),
        augmentation_policy,
        loss_variant,
        boundary_weight,
        eval_every_epochs,
    )

    for epoch in range(int(args.num_epochs)):
        head_module.train()
        epoch_losses: list[float] = []
        epoch_bce: list[float] = []
        epoch_dice: list[float] = []
        epoch_pred_fraction: list[float] = []
        epoch_coarse_loss: list[float] = []
        epoch_residual_l1_loss: list[float] = []
        iteration_order = permutation_rng.permutation(len(train_cache.samples)).tolist()
        reference_channels = {
            level_name: int(train_cache.reference_selected_level_shapes[level_name][0])
            for level_name in train_cache.selected_level_names
        }
        for sample_index in iteration_order:
            if augmentation_policy != "none":
                augmented_sample = apply_train_augmentation_to_binary_sample(
                    train_cache.samples[int(sample_index)],
                    rng=augmentation_rng,
                    policy=augmentation_policy,
                )
                cached_sample = prepare_cached_glas_feature_sample(
                    sample=augmented_sample,
                    extractor=train_cache.extractor,
                    variant=args.variant,
                    backbone_preprocessing_mode=train_cache.backbone_preprocessing_mode,
                    expected_selected_level_names=train_cache.selected_level_names,
                    reference_selected_level_names=train_cache.selected_level_names,
                    reference_channels=reference_channels,
                )
            elif train_cache.cached_feature_samples is not None:
                cached_sample = train_cache.cached_feature_samples[int(sample_index)]
            else:
                cached_sample = prepare_cached_glas_feature_sample(
                    sample=train_cache.samples[int(sample_index)],
                    extractor=train_cache.extractor,
                    variant=args.variant,
                    backbone_preprocessing_mode=train_cache.backbone_preprocessing_mode,
                    expected_selected_level_names=train_cache.selected_level_names,
                    reference_selected_level_names=train_cache.selected_level_names,
                    reference_channels=reference_channels,
                )
            feature_levels = materialize_cached_feature_levels_on_device(cached_sample, device=device)
            optimizer.zero_grad(set_to_none=True)
            output: FrozenMaskHeadOutput | FrozenForegroundProbeOutput | FrozenResidualMaskHeadOutput = head_module(
                feature_levels,
                image_size=cached_sample.image_size,
            )
            coarse_loss_result = None
            residual_l1_loss = None
            if is_coarse_plus_residual_variant(args.variant):
                loss_bundle = compute_residual_head_training_loss(
                    output,
                    np.asarray(cached_sample.supervision_foreground_mask, dtype=np.float32),
                    bce_weight=float(args.bce_weight),
                    dice_weight=float(args.dice_weight),
                    boundary_mask=(
                        np.asarray(cached_sample.supervision_boundary_mask, dtype=np.float32)
                        if loss_variant == "boundary_weighted_bce_dice"
                        else None
                    ),
                    boundary_weight=(boundary_weight if loss_variant == "boundary_weighted_bce_dice" else 0.0),
                    coarse_loss_weight=float(getattr(args, "coarse_loss_weight", 0.0)),
                    residual_l1_weight=float(getattr(args, "residual_l1_weight", 0.0)),
                    attention_sparsity_weight=float(getattr(args, "attention_sparsity_weight", 0.0)),
                )
                loss_result = loss_bundle.final_loss_result
                coarse_loss_result = loss_bundle.coarse_loss_result
                residual_l1_loss = loss_bundle.residual_l1_loss
                total_loss = loss_bundle.total_loss
            else:
                loss_result = compute_bce_dice_loss(
                    output.logits,
                    np.asarray(cached_sample.supervision_foreground_mask, dtype=np.float32),
                    bce_weight=float(args.bce_weight),
                    dice_weight=float(args.dice_weight),
                    boundary_mask=(
                        np.asarray(cached_sample.supervision_boundary_mask, dtype=np.float32)
                        if loss_variant == "boundary_weighted_bce_dice"
                        else None
                    ),
                    boundary_weight=(boundary_weight if loss_variant == "boundary_weighted_bce_dice" else 0.0),
                )
                total_loss = loss_result.loss
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(float(total_loss.detach().cpu().item()))
            epoch_bce.append(float(loss_result.bce_loss.detach().cpu().item()))
            epoch_dice.append(float(loss_result.dice_loss.detach().cpu().item()))
            epoch_pred_fraction.append(float(loss_result.predicted_positive_fraction))
            if coarse_loss_result is not None:
                epoch_coarse_loss.append(float(coarse_loss_result.loss.detach().cpu().item()))
            if residual_l1_loss is not None:
                epoch_residual_l1_loss.append(float(residual_l1_loss.detach().cpu().item()))
        if not epoch_losses:
            raise FrozenMaskHeadRuntimeError("Frozen-mask-head training produced no optimization steps.")
        history_row = {
            "epoch": epoch + 1,
            "mean_train_loss": float(mean(epoch_losses)),
            "mean_bce_loss": float(mean(epoch_bce)),
            "mean_dice_loss": float(mean(epoch_dice)),
            "mean_predicted_positive_fraction": float(mean(epoch_pred_fraction)),
        }
        if epoch_coarse_loss:
            history_row["mean_coarse_loss"] = float(mean(epoch_coarse_loss))
        if epoch_residual_l1_loss:
            history_row["mean_residual_l1_loss"] = float(mean(epoch_residual_l1_loss))
        training_history_rows.append(history_row)
        LOGGER.info(
            "Frozen-mask-head epoch=%d/%d | mean_loss=%.6f mean_bce=%.6f mean_dice=%.6f mean_pred_pos=%.4f%s%s",
            epoch + 1,
            int(args.num_epochs),
            history_row["mean_train_loss"],
            history_row["mean_bce_loss"],
            history_row["mean_dice_loss"],
            history_row["mean_predicted_positive_fraction"],
            (f" mean_coarse={history_row['mean_coarse_loss']:.6f}" if "mean_coarse_loss" in history_row else ""),
            (f" mean_residual_l1={history_row['mean_residual_l1_loss']:.6f}" if "mean_residual_l1_loss" in history_row else ""),
        )
        should_run_epoch_eval = (
            eval_cache_for_tracking is not None
            and eval_every_epochs is not None
            and (((epoch + 1) % int(eval_every_epochs)) == 0 or (epoch + 1) == int(args.num_epochs))
        )
        if should_run_epoch_eval:
            epoch_checkpoint_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
            epoch_summary = evaluate_glas_frozen_mask_head_on_cache(
                head_module=head_module,
                cache=eval_cache_for_tracking,
                variant=args.variant,
                args=args,
                checkpoint_path=epoch_checkpoint_path,
                run_kind="epoch_tracking_eval",
                trainable_parameter_count=trainable_parameter_count,
                output_dir=None,
                save_artifacts=False,
            )
            epoch_eval_row = {
                "epoch": epoch + 1,
                "direct_foreground_iou": float(epoch_summary["mean_metrics"]["direct_foreground_iou"]),
                "direct_foreground_dice": float(epoch_summary["mean_metrics"]["direct_foreground_dice"]),
                "eval_miou": float(epoch_summary["mean_metrics"]["eval_miou"]),
                "eval_ari": float(epoch_summary["mean_metrics"]["eval_ari"]),
            }
            epoch_eval_history_rows.append(epoch_eval_row)
            if best_eval_dice is None or float(epoch_eval_row["direct_foreground_dice"]) > float(best_eval_dice):
                best_eval_dice = float(epoch_eval_row["direct_foreground_dice"])
                best_epoch_by_eval_dice = int(epoch + 1)
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in head_module.state_dict().items()
                }
            LOGGER.info(
                "Frozen-mask-head epoch-eval=%d | eval_fg_iou=%.6f eval_dice=%.6f eval_miou=%.6f eval_ari=%.6f",
                epoch + 1,
                epoch_eval_row["direct_foreground_iou"],
                epoch_eval_row["direct_foreground_dice"],
                epoch_eval_row["eval_miou"],
                epoch_eval_row["eval_ari"],
            )

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_payload = {
        "variant": args.variant,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[args.variant]["summary"],
        "model_id": args.model_id,
        "selected_level_names": list(train_cache.selected_level_names),
        "level_input_dims": {
            level_name: int(train_cache.reference_selected_level_shapes[level_name][0])
            for level_name in train_cache.selected_level_names
        },
        "head_state_dict": head_module.state_dict(),
        "trainable_parameter_count": int(trainable_parameter_count),
        "seed": int(getattr(args, "seed", 0)),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "loss_variant": loss_variant,
        "loss_name": resolve_mask_head_loss_name(args),
        "boundary_weight": boundary_weight,
        "foreground_threshold": float(args.foreground_threshold),
        "reference_raw_level_shapes": train_cache.reference_raw_level_shapes,
        "reference_selected_level_shapes": train_cache.reference_selected_level_shapes,
        "training_history_rows": training_history_rows,
        "epoch_eval_history_rows": epoch_eval_history_rows,
        "best_epoch_by_eval_dice": best_epoch_by_eval_dice,
        "best_eval_dice": best_eval_dice,
        "train_augmentation_policy": augmentation_policy,
        "train_augmentation_summary": describe_train_augmentation_policy(augmentation_policy),
        "resize_height": int(train_cache.resize_hw[0]) if train_cache.resize_hw is not None else None,
        "resize_width": int(train_cache.resize_hw[1]) if train_cache.resize_hw is not None else None,
        "backbone_preprocessing_mode": train_cache.backbone_preprocessing_mode,
        "backbone_preprocessing_long_side": (
            int(AUTOSAM_STYLE_BACKBONE_LONG_SIDE)
            if train_cache.backbone_preprocessing_mode == "autosam_style_1024"
            else None
        ),
        "preprocessing_summary": resolve_mask_head_preprocessing_summary(args),
        "backbone_parameter_count": backbone_contract.get("backbone_parameter_count"),
        "backbone_frozen_parameter_count": backbone_contract.get("backbone_frozen_parameter_count"),
        "backbone_trainable_parameter_count": backbone_contract.get("backbone_trainable_parameter_count"),
        "backbone_requires_grad_was_zero_before_enforcement": backbone_contract.get(
            "backbone_requires_grad_was_zero_before_enforcement"
        ),
        "backbone_frozen_enforced": backbone_contract.get("backbone_frozen_enforced"),
        "backbone_contract_source": backbone_contract.get("backbone_contract_source"),
    }
    if args.variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS:
        checkpoint_payload.update(
            {
                "model_family": "foreground_probe",
                "head_kind": str(GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[args.variant]["head_kind"]),
                "projection_dim": int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["projection_dim"]),
                "hidden_dim": int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["hidden_dim"]),
            }
        )
    elif is_coarse_plus_residual_variant(args.variant):
        coarse_level_name, fine_level_name = resolve_coarse_plus_residual_levels(args.variant)
        residual_model_family = (
            "coarse_plus_cross_attention_refine_mask_head"
            if is_cross_attention_refine_variant(args.variant)
            else (
                "coarse_plus_attention_refine_mask_head"
                if is_attention_refine_variant(args.variant)
                else "coarse_plus_fine_residual_mask_head"
            )
        )
        residual_head_kind = (
            "coarse_plus_cross_attention_refine"
            if is_cross_attention_refine_variant(args.variant)
            else (
                "coarse_plus_attention_refine"
                if is_attention_refine_variant(args.variant)
                else "coarse_plus_fine_residual"
            )
        )
        checkpoint_payload.update(
            {
                "model_family": residual_model_family,
                "head_kind": residual_head_kind,
                "projection_dim": resolve_mask_head_projection_dim(args),
                "decoder_dim": resolve_mask_head_decoder_dim(args),
                "group_norm_groups": resolve_mask_head_group_norm_groups(args),
                "coarse_level_name": coarse_level_name,
                "fine_level_name": fine_level_name,
                "residual_projection_dim": int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
                "residual_hidden_dim": int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
                "attention_hidden_dim": int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
                "attention_heads": int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])),
                "cross_attn_query_stride": int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])),
                "residual_scale_init": float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
            }
        )
    elif is_memory_head_variant(args.variant):
        checkpoint_payload.update(
            {
                "model_family": resolve_frozen_mask_head_head_family(args.variant),
                "head_kind": ("memory_attention" if is_memory_attention_variant(args.variant) else "memory_control"),
                "projection_dim": resolve_mask_head_projection_dim(args),
                "decoder_dim": resolve_mask_head_decoder_dim(args),
                "group_norm_groups": resolve_mask_head_group_norm_groups(args),
                "memory_token_count": resolve_mask_head_memory_token_count(args),
                "attention_heads": resolve_mask_head_attention_heads(args),
                "attention_blocks": resolve_mask_head_attention_blocks(args),
                "memory_init_std": float(FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["memory_init_std"]),
            }
        )
    else:
        checkpoint_payload.update(
            {
                "model_family": "multiscale_mask_head",
                "projection_dim": resolve_mask_head_projection_dim(args),
                "decoder_dim": resolve_mask_head_decoder_dim(args),
                "group_norm_groups": resolve_mask_head_group_norm_groups(args),
            }
        )
    checkpoint_payload.update(build_residual_head_settings_payload(args, include_eval_alpha_override=False))
    torch.save(checkpoint_payload, checkpoint_path)
    best_checkpoint_path: Path | None = None
    if best_state_dict is not None and best_epoch_by_eval_dice is not None:
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        best_checkpoint_payload = dict(checkpoint_payload)
        best_checkpoint_payload["head_state_dict"] = best_state_dict
        best_checkpoint_payload["checkpoint_kind"] = "best_eval_dice_analysis"
        torch.save(best_checkpoint_payload, best_checkpoint_path)
    write_csv(output_dir / "train_history.csv", training_history_rows)
    if epoch_eval_history_rows:
        write_csv(output_dir / "epoch_eval_history.csv", epoch_eval_history_rows)
    return head_module, GlasMaskHeadTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        trainable_parameter_count=trainable_parameter_count,
        level_input_dims={
            level_name: int(train_cache.reference_selected_level_shapes[level_name][0])
            for level_name in train_cache.selected_level_names
        },
        training_history_rows=training_history_rows,
        epoch_eval_history_rows=epoch_eval_history_rows,
        best_checkpoint_path=best_checkpoint_path,
        best_epoch_by_eval_dice=best_epoch_by_eval_dice,
        best_eval_dice=best_eval_dice,
    )


def evaluate_glas_frozen_mask_head_on_cache(
    *,
    head_module: Any,
    cache: GlasFeatureCache,
    variant: str,
    args,
    checkpoint_path: Path,
    run_kind: str,
    trainable_parameter_count: int,
    output_dir: Path | None,
    save_artifacts: bool,
) -> dict[str, Any]:
    """Evaluate one trained frozen mask head on one cached STLD split."""

    import torch

    head_module.eval()
    device = resolve_torch_device(args.device)
    head_module.to(device)
    rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []

    if save_artifacts and output_dir is None:
        raise FrozenMaskHeadRuntimeError("save_artifacts=True requires a concrete output_dir.")

    for cached_sample in iter_cached_glas_feature_samples(cache):
        with torch.no_grad():
            feature_levels = materialize_cached_feature_levels_on_device(cached_sample, device=device)
            output: FrozenMaskHeadOutput = head_module(
                feature_levels,
                image_size=cached_sample.image_size,
            )
        residual_combination: ResidualHeadCombinationResult | None = None
        if is_coarse_plus_residual_variant(variant):
            residual_combination = combine_residual_head_logits(
                output,
                residual_alpha_override=getattr(args, "residual_alpha_override", None),
                residual_gate_mode=getattr(args, "residual_gate_mode", "none"),
                residual_gate_threshold=float(
                    getattr(args, "residual_gate_threshold", resolve_residual_head_gate_threshold(None))
                ),
            )
            final_logits = residual_combination.final_logits
            coarse_logits = residual_combination.coarse_logits
        else:
            final_logits = output.logits
            coarse_logits = None
        foreground_probability = torch.sigmoid(final_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        foreground_probability = resize_foreground_probability_to_sample_resolution(
            foreground_probability,
            sample=cached_sample.sample,
        )
        foreground_prediction = np.asarray(
            foreground_probability >= float(args.foreground_threshold),
            dtype=bool,
        )
        background_prediction = np.logical_not(foreground_prediction)
        direct_foreground_metrics = compute_binary_metrics(
            np.asarray(foreground_prediction, dtype=bool),
            np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
        )
        assignment = select_best_binary_assignment(
            np.asarray(foreground_prediction, dtype=bool),
            np.asarray(background_prediction, dtype=bool),
            np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
            np.asarray(cached_sample.sample.texture_b_mask, dtype=bool),
            "foreground_head",
            "background_complement",
        )
        coarse_direct_foreground_metrics = None
        coarse_assignment = None
        residual_summary = None
        if residual_combination is not None:
            coarse_foreground_probability = torch.sigmoid(coarse_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            coarse_foreground_probability = resize_foreground_probability_to_sample_resolution(
                coarse_foreground_probability,
                sample=cached_sample.sample,
            )
            coarse_foreground_prediction = np.asarray(
                coarse_foreground_probability >= float(args.foreground_threshold),
                dtype=bool,
            )
            coarse_background_prediction = np.logical_not(coarse_foreground_prediction)
            coarse_direct_foreground_metrics = compute_binary_metrics(
                np.asarray(coarse_foreground_prediction, dtype=bool),
                np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
            )
            coarse_assignment = select_best_binary_assignment(
                np.asarray(coarse_foreground_prediction, dtype=bool),
                np.asarray(coarse_background_prediction, dtype=bool),
                np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
                np.asarray(cached_sample.sample.texture_b_mask, dtype=bool),
                "foreground_head",
                "background_complement",
            )
            residual_summary = summarize_residual_head_combination(
                residual_combination,
                boundary_mask=np.asarray(cached_sample.sample.boundary_mask, dtype=bool),
            )
        row = build_glas_frozen_mask_head_eval_row(
            cached_sample=cached_sample,
            output=output,
            variant=variant,
            checkpoint_path=checkpoint_path,
            trainable_parameter_count=trainable_parameter_count,
            direct_foreground_metrics=direct_foreground_metrics,
            foreground_threshold=float(args.foreground_threshold),
            assignment=assignment,
            residual_combination=residual_combination,
            coarse_direct_foreground_metrics=coarse_direct_foreground_metrics,
            coarse_assignment=coarse_assignment,
            residual_summary=residual_summary,
        )
        rows.append(row)
        metric_summary = (
            f"Fg IoU={row['direct_foreground_iou']:.3f} "
            f"Dice={row['direct_foreground_dice']:.3f} "
            f"Aux mIoU={row['eval_miou']:.3f} "
            f"Aux ARI={row['eval_ari']:.3f}"
        )
        if residual_combination is not None:
            metric_summary += (
                f" | Coarse IoU={row['coarse_direct_foreground_iou']:.3f}"
                f" ΔIoU={row['final_minus_coarse_direct_foreground_iou']:+.3f}"
                f" scale={row['effective_residual_scale']:.3f}"
            )
        if save_artifacts and output_dir is not None:
            result = GlasMaskHeadEvalBundle(
                row=row,
                metric_summary=metric_summary,
                foreground_prediction=foreground_prediction,
                background_prediction=background_prediction,
                foreground_probability=foreground_probability,
            )
            save_glas_frozen_mask_head_result_bundle(
                output_dir=output_dir,
                cached_sample=cached_sample,
                result=result,
                variant=variant,
                save_visuals=bool(args.save_visuals),
            )
            if args.save_visuals:
                visual_records.append(
                    build_visual_record(
                        sample=cached_sample.sample,
                        evaluation=SampleEvaluationResult(
                            row=row,
                            metric_summary=metric_summary,
                            prediction_a=foreground_prediction,
                            prediction_b=background_prediction,
                        ),
                        protocol=f"stld_frozen_mask_head:{variant}",
                        visual_path=Path("visuals") / f"{cached_sample.sample.index}.png",
                    )
                )

    if not rows:
        raise FrozenMaskHeadRuntimeError("Frozen-mask-head evaluation produced no STLD sample rows.")

    summary = build_glas_frozen_mask_head_summary(
        rows=rows,
        cache=cache,
        args=args,
        checkpoint_path=checkpoint_path,
        run_kind=run_kind,
        trainable_parameter_count=trainable_parameter_count,
        variant=variant,
    )
    if run_kind in {"train", "eval"}:
        summary.update(
            profile_glas_frozen_mask_head_inference(
                head_module=head_module,
                cache=cache,
                variant=variant,
                args=args,
            )
        )
    if save_artifacts and output_dir is not None:
        write_csv(output_dir / "per_sample_metrics.csv", rows)
        write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
        write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
        rewrite_glas_frozen_mask_head_summary(output_dir=output_dir, summary=summary)
    return summary


def build_glas_frozen_mask_head_eval_row(
    *,
    cached_sample: CachedGlasFeatureSample,
    output: FrozenMaskHeadOutput | FrozenForegroundProbeOutput | FrozenResidualMaskHeadOutput,
    variant: str,
    checkpoint_path: Path,
    trainable_parameter_count: int,
    direct_foreground_metrics,
    foreground_threshold: float,
    assignment,
    residual_combination: ResidualHeadCombinationResult | None = None,
    coarse_direct_foreground_metrics=None,
    coarse_assignment=None,
    residual_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_pixels = int(cached_sample.sample.height * cached_sample.sample.width)
    projected_level_shapes = (
        output.projected_level_shapes
        if hasattr(output, "projected_level_shapes")
        else output.level_shapes
    )
    row: dict[str, Any] = {
        "variant": variant,
        "dataset_id": GLAS_DATASET_ID,
        "split": cached_sample.sample.split,
        "sample_index": int(cached_sample.sample.index),
        "crop_name": cached_sample.sample.crop_name,
        "grade_label": getattr(cached_sample.sample, "grade_label", None),
        "checkpoint_path": str(checkpoint_path),
        "foreground_evaluation_view": "direct_foreground",
        "foreground_assignment_used": "foreground_head->foreground,background_complement->background",
        "foreground_threshold": float(foreground_threshold),
        "selected_level_names_json": json.dumps(list(cached_sample.selected_level_names)),
        "raw_level_names_json": json.dumps(list(cached_sample.raw_level_names)),
        "raw_level_shapes_json": json.dumps(cached_sample.raw_level_shapes, sort_keys=True),
        "selected_level_shapes_json": json.dumps(cached_sample.selected_level_shapes, sort_keys=True),
        "finest_level_name": output.finest_level_name,
        "finest_grid_height": int(output.finest_grid_size[0]),
        "finest_grid_width": int(output.finest_grid_size[1]),
        "projected_level_shapes_json": json.dumps(projected_level_shapes, sort_keys=True),
        "fused_feature_shape_json": json.dumps(output.fused_feature_shape),
        "image_height": int(cached_sample.sample.height),
        "image_width": int(cached_sample.sample.width),
        "backbone_input_height": int(cached_sample.backbone_input_size[0]),
        "backbone_input_width": int(cached_sample.backbone_input_size[1]),
        "trainable_parameter_count": int(trainable_parameter_count),
        "residual_scale": float(output.residual_scale) if hasattr(output, "residual_scale") else None,
        "direct_foreground_iou": float(direct_foreground_metrics.iou),
        "direct_foreground_dice": float(direct_foreground_metrics.dice),
        "direct_foreground_precision": float(direct_foreground_metrics.precision),
        "direct_foreground_recall": float(direct_foreground_metrics.recall),
        "direct_foreground_predicted_positive": int(direct_foreground_metrics.predicted_positive),
        "direct_foreground_target_positive": int(direct_foreground_metrics.target_positive),
        "direct_foreground_matched_positive": int(direct_foreground_metrics.matched_positive),
        "predicted_positive_fraction": float(direct_foreground_metrics.predicted_positive / total_pixels),
        "target_positive_fraction": float(direct_foreground_metrics.target_positive / total_pixels),
        "assignment_used": assignment.assignment_used,
        "direct_eval_miou": float(assignment.direct_miou),
        "direct_eval_ari": float(assignment.direct_ari),
        "swapped_eval_miou": float(assignment.swapped_miou),
        "swapped_eval_ari": float(assignment.swapped_ari),
    }
    if residual_combination is not None:
        coarse_primary_metrics = coarse_direct_foreground_metrics
        if coarse_primary_metrics is None:
            raise FrozenMaskHeadRuntimeError("Residual diagnostics require coarse metrics for residual variants.")
        if coarse_assignment is None:
            raise FrozenMaskHeadRuntimeError("Residual diagnostics require coarse assignment metrics for residual variants.")
        coarse_row = {
            "coarse_direct_foreground_iou": float(coarse_primary_metrics.iou),
            "coarse_direct_foreground_dice": float(coarse_primary_metrics.dice),
            "coarse_direct_foreground_precision": float(coarse_primary_metrics.precision),
            "coarse_direct_foreground_recall": float(coarse_primary_metrics.recall),
            "coarse_direct_foreground_predicted_positive": int(coarse_primary_metrics.predicted_positive),
            "coarse_direct_foreground_target_positive": int(coarse_primary_metrics.target_positive),
            "coarse_direct_foreground_matched_positive": int(coarse_primary_metrics.matched_positive),
            "coarse_predicted_positive_fraction": float(coarse_primary_metrics.predicted_positive / total_pixels),
            "coarse_target_positive_fraction": float(coarse_primary_metrics.target_positive / total_pixels),
            "coarse_assignment_used": coarse_assignment.assignment_used,
            "coarse_direct_eval_miou": float(coarse_assignment.direct_miou),
            "coarse_direct_eval_ari": float(coarse_assignment.direct_ari),
            "coarse_swapped_eval_miou": float(coarse_assignment.swapped_miou),
            "coarse_swapped_eval_ari": float(coarse_assignment.swapped_ari),
            "final_minus_coarse_direct_foreground_iou": float(direct_foreground_metrics.iou - coarse_primary_metrics.iou),
            "final_minus_coarse_direct_foreground_dice": float(direct_foreground_metrics.dice - coarse_primary_metrics.dice),
            "final_minus_coarse_eval_miou": float(assignment.chosen_miou - coarse_assignment.chosen_miou),
            "final_minus_coarse_eval_ari": float(assignment.chosen_ari - coarse_assignment.chosen_ari),
        }
        if residual_summary is not None:
            coarse_row.update(residual_summary)
        row.update(coarse_row)
    row.update(
        build_canonical_evaluation_fields(
            miou=float(assignment.chosen_miou),
            ari=float(assignment.chosen_ari),
            evaluation_view="partition_invariant",
        )
    )
    return row


def profile_glas_frozen_mask_head_inference(
    *,
    head_module: Any,
    cache: GlasFeatureCache,
    variant: str,
    args,
) -> dict[str, Any]:
    """Measure end-to-end per-image latency from resized image to final logits."""

    import torch

    device = resolve_torch_device(args.device)
    head_module.eval()
    head_module.to(device)
    reference_channels = {
        level_name: int(cache.reference_selected_level_shapes[level_name][0])
        for level_name in cache.selected_level_names
    }
    timings_seconds: list[float] = []
    peak_memory_bytes = 0
    for sample in cache.samples:
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.synchronize(device=device)
        start = time.perf_counter()
        cached_sample = prepare_cached_glas_feature_sample(
            sample=sample,
            extractor=cache.extractor,
            variant=variant,
            backbone_preprocessing_mode=cache.backbone_preprocessing_mode,
            expected_selected_level_names=cache.selected_level_names,
            reference_selected_level_names=cache.selected_level_names,
            reference_channels=reference_channels,
        )
        with torch.no_grad():
            feature_levels = materialize_cached_feature_levels_on_device(cached_sample, device=device)
            output = head_module(feature_levels, image_size=cached_sample.image_size)
            final_logits = (
                combine_residual_head_logits(
                    output,
                    residual_alpha_override=getattr(args, "residual_alpha_override", None),
                    residual_gate_mode=getattr(args, "residual_gate_mode", "none"),
                    residual_gate_threshold=float(
                        getattr(args, "residual_gate_threshold", resolve_residual_head_gate_threshold(None))
                    ),
                ).final_logits
                if is_coarse_plus_residual_variant(variant)
                else output.logits
            )
            _ = threshold_foreground_logits(final_logits, threshold=float(args.foreground_threshold))
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device=device)
            peak_memory_bytes = max(int(peak_memory_bytes), int(torch.cuda.max_memory_allocated(device=device)))
        timings_seconds.append(float(time.perf_counter() - start))
    if not timings_seconds:
        raise FrozenMaskHeadRuntimeError("Frozen-mask-head inference profiling received zero STLD samples.")
    return {
        "single_pass_inference": True,
        "sam_decoder_invoked_at_inference": False,
        "inference_profile_includes_feature_extraction": True,
        "inference_profile_num_samples": len(timings_seconds),
        "mean_inference_seconds_per_image": float(mean(timings_seconds)),
        "median_inference_seconds_per_image": float(median(timings_seconds)),
        "peak_inference_device_memory_bytes": int(peak_memory_bytes),
        "peak_inference_device_memory_megabytes": float(peak_memory_bytes / (1024.0 * 1024.0)),
    }


def build_glas_frozen_mask_head_summary(
    *,
    rows: list[dict[str, Any]],
    cache: GlasFeatureCache,
    args,
    checkpoint_path: Path,
    run_kind: str,
    trainable_parameter_count: int,
    variant: str,
) -> dict[str, Any]:
    mean_metrics = {name: float(mean(float(row[name]) for row in rows)) for name in GLAS_FROZEN_MASK_HEAD_SCALAR_FIELDS}
    median_metrics = {name: float(median(float(row[name]) for row in rows)) for name in GLAS_FROZEN_MASK_HEAD_SCALAR_FIELDS}
    coarse_level_name, fine_level_name = (
        resolve_coarse_plus_residual_levels(variant) if is_coarse_plus_residual_variant(variant) else (None, None)
    )
    residual_variant = is_coarse_plus_residual_variant(variant)
    def mean_optional(name: str) -> float | None:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        return float(mean(values)) if values else None
    def first_optional(name: str) -> Any | None:
        for row in rows:
            if row.get(name) is not None:
                return row[name]
        return None
    coarse_mean_metrics = None
    residual_diagnostics_mean = None
    if residual_variant:
        coarse_mean_metrics = {
            "coarse_direct_foreground_iou": mean_optional("coarse_direct_foreground_iou"),
            "coarse_direct_foreground_dice": mean_optional("coarse_direct_foreground_dice"),
            "coarse_direct_foreground_precision": mean_optional("coarse_direct_foreground_precision"),
            "coarse_direct_foreground_recall": mean_optional("coarse_direct_foreground_recall"),
            "coarse_predicted_positive_fraction": mean_optional("coarse_predicted_positive_fraction"),
            "coarse_target_positive_fraction": mean_optional("coarse_target_positive_fraction"),
            "coarse_direct_eval_miou": mean_optional("coarse_direct_eval_miou"),
            "coarse_direct_eval_ari": mean_optional("coarse_direct_eval_ari"),
            "coarse_swapped_eval_miou": mean_optional("coarse_swapped_eval_miou"),
            "coarse_swapped_eval_ari": mean_optional("coarse_swapped_eval_ari"),
            "final_minus_coarse_direct_foreground_iou": mean_optional("final_minus_coarse_direct_foreground_iou"),
            "final_minus_coarse_direct_foreground_dice": mean_optional("final_minus_coarse_direct_foreground_dice"),
            "final_minus_coarse_eval_miou": mean_optional("final_minus_coarse_eval_miou"),
            "final_minus_coarse_eval_ari": mean_optional("final_minus_coarse_eval_ari"),
        }
        residual_diagnostics_mean = {
            "learned_residual_scale": mean_optional("learned_residual_scale"),
            "effective_residual_scale": mean_optional("effective_residual_scale"),
            "residual_scale_source": first_optional("residual_scale_source"),
            "residual_gate_mode": first_optional("residual_gate_mode"),
            "residual_gate_threshold": mean_optional("residual_gate_threshold"),
            "residual_alpha_override": mean_optional("residual_alpha_override"),
            "residual_gate_mean": mean_optional("residual_gate_mean"),
            "residual_gate_fraction_ge_half": mean_optional("residual_gate_fraction_ge_half"),
            "attention_mean": mean_optional("attention_mean"),
            "attention_fraction_ge_half": mean_optional("attention_fraction_ge_half"),
            "mean_abs_residual_logits": mean_optional("mean_abs_residual_logits"),
            "mean_abs_residual_contribution": mean_optional("mean_abs_residual_contribution"),
            "residual_sign_flip_fraction": mean_optional("residual_sign_flip_fraction"),
            "final_minus_coarse_mean_abs": mean_optional("final_minus_coarse_mean_abs"),
        }
    backbone_contract = dict(getattr(args, "_backbone_contract", {}) or {})
    return {
        "run_kind": run_kind,
        "variant": variant,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[variant]["summary"],
        "dataset_id": GLAS_DATASET_ID,
        "route": STLD_ROUTE,
        "split": cache.split,
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "benchmark_root": str(cache.dataset_name),
        "cache_dir": cache.cache_dir,
        "train_selection_policy": resolve_train_selection_policy(args),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "model_id": args.model_id,
        "device": args.device,
        "official_checkpoint_path": args.official_checkpoint_path,
        "checkpoint_path": str(checkpoint_path),
        "num_evaluated_samples": len(rows),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "foreground_evaluation_view": "direct_foreground",
        "primary_metric_name": "direct_foreground_iou",
        "secondary_metric_name": "direct_foreground_dice",
        "aux_partition_primary_metric_name": "eval_miou",
        "aux_partition_secondary_metric_name": "eval_ari",
        "selected_level_names": list(cache.selected_level_names),
        "reference_raw_level_shapes": cache.reference_raw_level_shapes,
        "reference_selected_level_shapes": cache.reference_selected_level_shapes,
        "materialization_mode": cache.materialization_mode,
        "resize_height": int(cache.resize_hw[0]) if cache.resize_hw is not None else None,
        "resize_width": int(cache.resize_hw[1]) if cache.resize_hw is not None else None,
        "backbone_preprocessing_mode": cache.backbone_preprocessing_mode,
        "backbone_preprocessing_long_side": (
            int(AUTOSAM_STYLE_BACKBONE_LONG_SIDE)
            if cache.backbone_preprocessing_mode == "autosam_style_1024"
            else None
        ),
        "preprocessing_summary": resolve_mask_head_preprocessing_summary(args),
        "image_resize_resample": (
            "bilinear"
            if cache.resize_hw is not None
            else ("upstream_resize_longest_side" if cache.backbone_preprocessing_mode == "autosam_style_1024" else None)
        ),
        "mask_resize_resample": (
            "nearest"
            if cache.resize_hw is not None or cache.backbone_preprocessing_mode == "autosam_style_1024"
            else None
        ),
        "model_family": (
            "foreground_probe"
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "coarse_plus_attention_refine_mask_head"
                if is_attention_refine_variant(variant)
                else (
                    "coarse_plus_fine_residual_mask_head"
                    if is_coarse_plus_residual_variant(variant)
                    else (
                        resolve_frozen_mask_head_head_family(variant)
                        if is_memory_head_variant(variant)
                        else "multiscale_mask_head"
                    )
                )
            )
        ),
        "head_kind": (
            GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]["head_kind"]
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "coarse_plus_cross_attention_refine"
                if is_cross_attention_refine_variant(variant)
                else (
                    "coarse_plus_attention_refine"
                    if is_attention_refine_variant(variant)
                    else (
                        "coarse_plus_fine_residual"
                        if is_coarse_plus_residual_variant(variant)
                        else ("memory_attention" if is_memory_attention_variant(variant) else ("memory_control" if is_memory_control_variant(variant) else "multiscale_mask_head"))
                    )
                )
            )
        ),
        "projection_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["projection_dim"])
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_projection_dim(args)
        ),
        "decoder_dim": (
            None
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_decoder_dim(args)
        ),
        "hidden_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["hidden_dim"])
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else None
        ),
        "group_norm_groups": (
            None
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_group_norm_groups(args)
        ),
        "memory_token_count": (resolve_mask_head_memory_token_count(args) if is_memory_head_variant(variant) else None),
        "attention_heads": (resolve_mask_head_attention_heads(args) if is_memory_head_variant(variant) else None),
        "attention_blocks": (resolve_mask_head_attention_blocks(args) if is_memory_head_variant(variant) else None),
        "coarse_level_name": coarse_level_name,
        "fine_level_name": fine_level_name,
        "residual_projection_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "residual_hidden_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "attention_hidden_dim": (
            int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"]))
            if is_attention_refine_variant(variant)
            else None
        ),
        "residual_scale_init": (
            float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "backbone_frozen": True,
        "backbone_parameter_count": backbone_contract.get("backbone_parameter_count"),
        "backbone_frozen_parameter_count": backbone_contract.get("backbone_frozen_parameter_count"),
        "backbone_trainable_parameter_count": backbone_contract.get("backbone_trainable_parameter_count"),
        "backbone_requires_grad_was_zero_before_enforcement": backbone_contract.get(
            "backbone_requires_grad_was_zero_before_enforcement"
        ),
        "backbone_frozen_enforced": backbone_contract.get("backbone_frozen_enforced"),
        "backbone_contract_source": backbone_contract.get("backbone_contract_source"),
        "trainable_parameter_count": int(trainable_parameter_count),
        "head_trainable_parameter_count": int(trainable_parameter_count),
        "optimizer": "AdamW",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "loss_variant": resolve_mask_head_loss_variant(args),
        "loss_name": resolve_mask_head_loss_name(args),
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "boundary_weight": resolve_mask_head_boundary_weight(args),
        "foreground_threshold": float(args.foreground_threshold),
        "train_augmentation_policy": resolve_train_augmentation_policy(args),
        "train_augmentation_summary": describe_train_augmentation_policy(resolve_train_augmentation_policy(args)),
        "eval_every_epochs": resolve_mask_head_eval_every_epochs(args),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "coarse_mean_metrics": coarse_mean_metrics,
        "residual_diagnostics_mean": residual_diagnostics_mean,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def flatten_glas_frozen_mask_head_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flattened = {
        "variant": summary["variant"],
        "split": summary["split"],
        "num_evaluated_samples": summary["num_evaluated_samples"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "backbone_parameter_count": summary.get("backbone_parameter_count"),
        "backbone_trainable_parameter_count": summary.get("backbone_trainable_parameter_count"),
        "mean_inference_seconds_per_image": summary.get("mean_inference_seconds_per_image"),
        "peak_inference_device_memory_bytes": summary.get("peak_inference_device_memory_bytes"),
        "selected_level_names_json": json.dumps(summary["selected_level_names"]),
        "direct_foreground_iou": summary["mean_metrics"]["direct_foreground_iou"],
        "direct_foreground_dice": summary["mean_metrics"]["direct_foreground_dice"],
        "eval_miou": summary["mean_metrics"]["eval_miou"],
        "eval_ari": summary["mean_metrics"]["eval_ari"],
        "checkpoint_path": summary["checkpoint_path"],
    }
    coarse_mean_metrics = summary.get("coarse_mean_metrics")
    if coarse_mean_metrics is not None:
        flattened.update(
            {
                "coarse_direct_foreground_iou": coarse_mean_metrics["coarse_direct_foreground_iou"],
                "coarse_direct_foreground_dice": coarse_mean_metrics["coarse_direct_foreground_dice"],
                "coarse_direct_eval_miou": coarse_mean_metrics["coarse_direct_eval_miou"],
                "coarse_direct_eval_ari": coarse_mean_metrics["coarse_direct_eval_ari"],
                "final_minus_coarse_direct_foreground_iou": coarse_mean_metrics["final_minus_coarse_direct_foreground_iou"],
                "final_minus_coarse_direct_foreground_dice": coarse_mean_metrics["final_minus_coarse_direct_foreground_dice"],
                "final_minus_coarse_eval_miou": coarse_mean_metrics["final_minus_coarse_eval_miou"],
                "final_minus_coarse_eval_ari": coarse_mean_metrics["final_minus_coarse_eval_ari"],
            }
        )
    residual_diagnostics_mean = summary.get("residual_diagnostics_mean")
    if residual_diagnostics_mean is not None:
        flattened.update(
            {
                "learned_residual_scale": residual_diagnostics_mean["learned_residual_scale"],
                "effective_residual_scale": residual_diagnostics_mean["effective_residual_scale"],
                "residual_scale_source": residual_diagnostics_mean["residual_scale_source"],
                "residual_gate_mode": residual_diagnostics_mean["residual_gate_mode"],
                "residual_gate_threshold": residual_diagnostics_mean["residual_gate_threshold"],
                "residual_alpha_override": residual_diagnostics_mean["residual_alpha_override"],
                "residual_gate_mean": residual_diagnostics_mean["residual_gate_mean"],
                "residual_gate_fraction_ge_half": residual_diagnostics_mean["residual_gate_fraction_ge_half"],
                "attention_mean": residual_diagnostics_mean["attention_mean"],
                "attention_fraction_ge_half": residual_diagnostics_mean["attention_fraction_ge_half"],
                "mean_abs_residual_logits": residual_diagnostics_mean["mean_abs_residual_logits"],
                "mean_abs_residual_contribution": residual_diagnostics_mean["mean_abs_residual_contribution"],
                "residual_sign_flip_fraction": residual_diagnostics_mean["residual_sign_flip_fraction"],
                "final_minus_coarse_mean_abs": residual_diagnostics_mean["final_minus_coarse_mean_abs"],
            }
        )
    return flattened


def build_glas_frozen_mask_head_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# STLD Frozen SAM Mask Head Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Variant summary: {summary['variant_summary']}",
        f"- Split: `{summary['split']}`",
        f"- Train selection policy: `{summary.get('train_selection_policy')}`",
        f"- Train subset manifest: `{summary.get('train_subset_manifest_output_path') or summary.get('train_subset_manifest_path')}`",
        f"- Selected levels: {', '.join(f'`{name}`' for name in summary['selected_level_names'])}",
        f"- Model family: `{summary['model_family']}`",
        f"- Head kind: `{summary['head_kind']}`",
        f"- Trainable params: `{summary['trainable_parameter_count']}`",
        f"- Frozen SAM params: `{summary.get('backbone_frozen_parameter_count')}` / total `{summary.get('backbone_parameter_count')}`",
        f"- Preprocessing: `{summary.get('preprocessing_summary')}`",
        f"- Train augmentation: `{summary.get('train_augmentation_summary')}`",
        f"- Loss: `{summary.get('loss_name')}`",
        f"- Memory tokens: `{summary.get('memory_token_count')}`",
        f"- Attention heads: `{summary.get('attention_heads')}`",
        f"- Attention blocks: `{summary.get('attention_blocks')}`",
        f"- Single-pass inference: `{summary.get('single_pass_inference')}`",
        f"- SAM decoder invoked at inference: `{summary.get('sam_decoder_invoked_at_inference')}`",
        f"- Mean inference seconds/image: `{summary.get('mean_inference_seconds_per_image')}`",
        f"- Peak inference device memory bytes: `{summary.get('peak_inference_device_memory_bytes')}`",
        f"- Mean direct foreground IoU: `{summary['mean_metrics']['direct_foreground_iou']:.6f}`",
        f"- Mean Dice: `{summary['mean_metrics']['direct_foreground_dice']:.6f}`",
        f"- Mean auxiliary partition mIoU: `{summary['mean_metrics']['eval_miou']:.6f}`",
        f"- Mean auxiliary partition ARI: `{summary['mean_metrics']['eval_ari']:.6f}`",
    ]
    if summary.get("train_set_mean_metrics") is not None:
        lines.append(f"- Mean train-set foreground IoU: `{summary['train_set_mean_metrics']['direct_foreground_iou']:.6f}`")
        lines.append(f"- Mean train-set Dice: `{summary['train_set_mean_metrics']['direct_foreground_dice']:.6f}`")
    if summary.get("coarse_mean_metrics") is not None:
        lines.append(f"- Mean coarse direct foreground IoU: `{summary['coarse_mean_metrics']['coarse_direct_foreground_iou']:.6f}`")
        lines.append(f"- Mean coarse direct foreground Dice: `{summary['coarse_mean_metrics']['coarse_direct_foreground_dice']:.6f}`")
        lines.append(f"- Mean final-minus-coarse direct foreground IoU: `{summary['coarse_mean_metrics']['final_minus_coarse_direct_foreground_iou']:.6f}`")
        lines.append(f"- Mean final-minus-coarse direct foreground Dice: `{summary['coarse_mean_metrics']['final_minus_coarse_direct_foreground_dice']:.6f}`")
    if summary.get("residual_diagnostics_mean") is not None:
        lines.append(f"- Residual gate mode: `{summary['residual_diagnostics_mean']['residual_gate_mode']}`")
        lines.append(f"- Residual gate threshold: `{summary['residual_diagnostics_mean']['residual_gate_threshold']}`")
        lines.append(f"- Residual alpha override: `{summary['residual_diagnostics_mean']['residual_alpha_override']}`")
        lines.append(f"- Learned residual scale: `{summary['residual_diagnostics_mean']['learned_residual_scale']}`")
        lines.append(f"- Effective residual scale: `{summary['residual_diagnostics_mean']['effective_residual_scale']}`")
        lines.append(f"- Mean abs residual logits: `{summary['residual_diagnostics_mean']['mean_abs_residual_logits']}`")
        lines.append(f"- Mean abs residual contribution: `{summary['residual_diagnostics_mean']['mean_abs_residual_contribution']}`")
        lines.append(f"- Residual sign-flip fraction: `{summary['residual_diagnostics_mean']['residual_sign_flip_fraction']}`")
        lines.append(f"- Mean final-minus-coarse abs logit delta: `{summary['residual_diagnostics_mean']['final_minus_coarse_mean_abs']}`")
    lines.extend(
        [
            f"- Best checkpoint: `{summary.get('best_checkpoint_path')}`",
            f"- Best epoch by eval Dice: `{summary.get('best_epoch_by_eval_dice')}`",
            f"- Best eval Dice: `{summary.get('best_eval_dice')}`",
            f"- Best-epoch caveat: `{summary.get('best_epoch_selection_caveat')}`",
            f"- Validation set metrics: `{summary.get('validation_set_mean_metrics')}`",
            f"- Checkpoint: `{summary['checkpoint_path']}`",
            "",
        ]
    )
    lines.append(render_quality_summary_markdown(summary).rstrip())
    lines.append("")
    return "\n".join(lines) + "\n"


def rewrite_glas_frozen_mask_head_summary(*, output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [flatten_glas_frozen_mask_head_summary(summary)])
    write_text(output_dir / "summary.md", build_glas_frozen_mask_head_summary_markdown(summary))


def save_glas_frozen_mask_head_result_bundle(
    *,
    output_dir: Path,
    cached_sample: CachedGlasFeatureSample,
    result: GlasMaskHeadEvalBundle,
    variant: str,
    save_visuals: bool,
) -> None:
    sample_index = int(cached_sample.sample.index)
    (output_dir / "sample_rows").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sample_rows" / f"{sample_index}.json", result.row)
    np.savez_compressed(
        output_dir / "masks" / f"{sample_index}.npz",
        foreground_prediction=np.asarray(result.foreground_prediction, dtype=bool),
        background_prediction=np.asarray(result.background_prediction, dtype=bool),
        target_foreground=np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
        target_background=np.asarray(cached_sample.sample.texture_b_mask, dtype=bool),
        foreground_probability=np.asarray(result.foreground_probability, dtype=np.float32),
    )
    if save_visuals:
        (output_dir / "visuals").mkdir(parents=True, exist_ok=True)
        save_prediction_panel(
            output_path=output_dir / "visuals" / f"{sample_index}.png",
            sample=cached_sample.sample,
            prediction_a=result.foreground_prediction,
            prediction_b=result.background_prediction,
            protocol=f"stld_frozen_mask_head:{variant}",
            metric_summary=result.metric_summary,
        )


def build_glas_frozen_mask_head_run_config(
    args,
    *,
    run_kind: str,
    selected_level_names: tuple[str, ...],
    reference_raw_level_shapes: dict[str, tuple[int, int, int]],
    reference_selected_level_shapes: dict[str, tuple[int, int, int]],
    materialization_mode: str | None = None,
    dataset_partition=None,
    selected_sample_count: int | None = None,
) -> dict[str, Any]:
    coarse_level_name, fine_level_name = (
        resolve_coarse_plus_residual_levels(str(getattr(args, "variant", None)))
        if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
        else (None, None)
    )
    variant = str(getattr(args, "variant", None))
    backbone_contract = dict(getattr(args, "_backbone_contract", {}) or {})
    config = {
        "command": getattr(args, "command", None),
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(str(value)) for value in sys.argv),
        "run_kind": run_kind,
        "dataset_id": GLAS_DATASET_ID,
        "route": STLD_ROUTE,
        "benchmark_root": getattr(args, "benchmark_root", None),
        "cache_dir": None,
        "train_selection_policy": resolve_train_selection_policy(args),
        "split": getattr(args, "split", None),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "variant": variant,
        "model_id": args.model_id,
        "device": args.device,
        "official_checkpoint_path": args.official_checkpoint_path,
        "hf_token_supplied": args.hf_token is not None,
        "train_limit": getattr(args, "train_limit", None),
        "eval_limit": getattr(args, "eval_limit", None),
        "limit": getattr(args, "limit", None),
        "selected_level_names": list(selected_level_names),
        "reference_raw_level_shapes": reference_raw_level_shapes,
        "reference_selected_level_shapes": reference_selected_level_shapes,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[variant]["summary"],
        "model_family": (
            "foreground_probe"
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "coarse_plus_cross_attention_refine_mask_head"
                if is_cross_attention_refine_variant(variant)
                else (
                    "coarse_plus_attention_refine_mask_head"
                    if is_attention_refine_variant(variant)
                    else (
                        "coarse_plus_fine_residual_mask_head"
                        if is_coarse_plus_residual_variant(variant)
                        else (resolve_frozen_mask_head_head_family(variant) if is_memory_head_variant(variant) else "multiscale_mask_head")
                    )
                )
            )
        ),
        "head_kind": (
            GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]["head_kind"]
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "coarse_plus_cross_attention_refine"
                if is_cross_attention_refine_variant(variant)
                else (
                    "coarse_plus_attention_refine"
                    if is_attention_refine_variant(variant)
                    else (
                        "coarse_plus_fine_residual"
                        if is_coarse_plus_residual_variant(variant)
                        else ("memory_attention" if is_memory_attention_variant(variant) else ("memory_control" if is_memory_control_variant(variant) else "multiscale_mask_head"))
                    )
                )
            )
        ),
        "projection_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["projection_dim"])
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_projection_dim(args)
        ),
        "decoder_dim": (
            None
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_decoder_dim(args)
        ),
        "hidden_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["hidden_dim"])
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else None
        ),
        "group_norm_groups": (
            None
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_group_norm_groups(args)
        ),
        "memory_token_count": (resolve_mask_head_memory_token_count(args) if is_memory_head_variant(variant) else None),
        "attention_heads": (resolve_mask_head_attention_heads(args) if is_memory_head_variant(variant) else None),
        "attention_blocks": (resolve_mask_head_attention_blocks(args) if is_memory_head_variant(variant) else None),
        "coarse_level_name": coarse_level_name,
        "fine_level_name": fine_level_name,
        "residual_projection_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "residual_hidden_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "attention_hidden_dim": (
            int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"]))
            if is_attention_refine_variant(variant)
            else None
        ),
        "residual_scale_init": (
            float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            if is_coarse_plus_residual_variant(variant)
            else None
        ),
        "backbone_frozen": True,
        "backbone_parameter_count": backbone_contract.get("backbone_parameter_count"),
        "backbone_frozen_parameter_count": backbone_contract.get("backbone_frozen_parameter_count"),
        "backbone_trainable_parameter_count": backbone_contract.get("backbone_trainable_parameter_count"),
        "backbone_requires_grad_was_zero_before_enforcement": backbone_contract.get(
            "backbone_requires_grad_was_zero_before_enforcement"
        ),
        "backbone_frozen_enforced": backbone_contract.get("backbone_frozen_enforced"),
        "backbone_contract_source": backbone_contract.get("backbone_contract_source"),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "loss_variant": resolve_mask_head_loss_variant(args),
        "loss_name": resolve_mask_head_loss_name(args),
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "boundary_weight": resolve_mask_head_boundary_weight(args),
        "foreground_threshold": float(args.foreground_threshold),
        "train_augmentation_policy": resolve_train_augmentation_policy(args),
        "train_augmentation_summary": describe_train_augmentation_policy(resolve_train_augmentation_policy(args)),
        "eval_every_epochs": resolve_mask_head_eval_every_epochs(args),
        "resize_height": int(resolve_mask_head_resize_hw(args)[0]) if resolve_mask_head_resize_hw(args) is not None else None,
        "resize_width": int(resolve_mask_head_resize_hw(args)[1]) if resolve_mask_head_resize_hw(args) is not None else None,
        "backbone_preprocessing_mode": resolve_mask_head_backbone_preprocessing_mode(args),
        "backbone_preprocessing_long_side": (
            int(AUTOSAM_STYLE_BACKBONE_LONG_SIDE)
            if resolve_mask_head_backbone_preprocessing_mode(args) == "autosam_style_1024"
            else None
        ),
        "preprocessing_summary": resolve_mask_head_preprocessing_summary(args),
        "image_resize_resample": (
            "bilinear"
            if resolve_mask_head_resize_hw(args) is not None
            else (
                "upstream_resize_longest_side"
                if resolve_mask_head_backbone_preprocessing_mode(args) == "autosam_style_1024"
                else None
            )
        ),
        "mask_resize_resample": (
            "nearest"
            if resolve_mask_head_resize_hw(args) is not None
            or resolve_mask_head_backbone_preprocessing_mode(args) == "autosam_style_1024"
            else None
        ),
        "save_visuals": bool(args.save_visuals),
        "seed": int(getattr(args, "seed", 0)),
        "materialization_mode": materialization_mode,
        "variant_specs": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS,
        "settings": (
            GLAS_FROZEN_FEATURE_PROBE_SETTINGS if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS else FROZEN_MASK_HEAD_SETTINGS
        ),
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    config.update(build_residual_head_settings_payload(args, include_eval_alpha_override=True))
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_glas_frozen_mask_head_experiment_terms_markdown(
    args,
    *,
    run_kind: str,
    selected_level_names: tuple[str, ...],
    reference_raw_level_shapes: dict[str, tuple[int, int, int]],
    reference_selected_level_shapes: dict[str, tuple[int, int, int]],
    train_sample_count: int | None,
    eval_sample_count: int | None,
    checkpoint_path: Path | None,
) -> str:
    if not GLAS_FROZEN_MASK_HEAD_EXPERIMENT_CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing experiment contract file: {GLAS_FROZEN_MASK_HEAD_EXPERIMENT_CONTRACT_PATH}"
        )
    variant = str(getattr(args, "variant", "unknown"))
    variant_spec = GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[variant]
    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            bullets=(
                f"Command family: `{getattr(args, 'command', 'stld-frozen-mask-head')}`.",
                f"Run kind: `{run_kind}`.",
                f"Variant: `{variant}`.",
                    f"Variant summary: {variant_spec['summary']}",
                f"Dataset: `{GLAS_DATASET_ID}` with route `{STLD_ROUTE}` from benchmark root `{getattr(args, 'benchmark_root', None)}`.",
                f"Model: `{args.model_id}` on device request `{args.device}`.",
                f"Official checkpoint override for SAM feature extraction: `{args.official_checkpoint_path}`.",
                "SAM stays frozen for the entire run. Only the new mask head is optimized.",
            ),
        ),
        ExperimentTermsSection(
            title="Data And Split Separation",
            bullets=(
                f"Training split: `{getattr(args, 'train_split', None)}` with limit `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Evaluation split: `{getattr(args, 'eval_split', getattr(args, 'split', None))}` with limit `{getattr(args, 'eval_limit', getattr(args, 'limit', None))}`; loaded `{eval_sample_count}` sample(s).",
                "The repo-native STLD loader requires the prepared split-aware root with `ImageSets/Segmentation/{train,test}.txt` and uses the repository's random 50/50 seed-0 split artifact.",
                (
                    f"Train subset manifest: `{getattr(args, '_resolved_train_subset_manifest_output_path', getattr(args, 'train_subset_manifest', None))}` selecting `{getattr(args, '_resolved_train_subset_manifest', {}).get('shot_count')}` image(s) with subset seed `{getattr(args, '_resolved_train_subset_manifest', {}).get('subset_seed')}`."
                    if getattr(args, "_resolved_train_subset_manifest", None) is not None
                    else "No train subset manifest is applied. The full prepared STLD train split is used."
                ),
                "No separate validation split is introduced here. Validation metrics are therefore recorded as `null` rather than inventing a new split.",
                "Foreground polarity contract: the foreground region is the positive class and corresponds to `texture_a_mask` in the STLD adapter.",
                f"Train augmentation policy: `{describe_train_augmentation_policy(resolve_train_augmentation_policy(args))}`.",
                (
                    f"Explicit preprocessing resize: `{resolve_mask_head_resize_hw(args)[0]}x{resolve_mask_head_resize_hw(args)[1]}` with bilinear image resize and nearest-neighbor mask resize before SAM feature extraction, supervision, and evaluation."
                    if resolve_mask_head_resize_hw(args) is not None
                    else "No explicit resize is applied. Images and masks stay at native decoded STLD resolution."
                ),
            ),
        ),
        ExperimentTermsSection(
            title="Frozen Feature Source",
            bullets=tuple(
                [
                    f"Frozen SAM feature source: `{FROZEN_MASK_HEAD_SETTINGS['feature_source']}` from the main `backbone_fpn` pyramid.",
                    f"Selected levels for this run: {', '.join(f'`{name}`' for name in selected_level_names)}.",
                ]
                + [f"Reference raw level shape {item}." for item in format_shape_mapping(reference_raw_level_shapes)]
                + [f"Reference selected level shape {item}." for item in format_shape_mapping(reference_selected_level_shapes)]
            ),
        ),
        ExperimentTermsSection(
            title="Mask Head Architecture",
            paragraphs=(
                "Architecture is variant-dependent. Legacy mask-head variants keep the existing tiny FPN-style decoder, while the new probe variants keep only the smallest dense readout needed for the requested scale diagnosis.",
            ),
            bullets=_build_architecture_bullets(variant),
        ),
        ExperimentTermsSection(
            title="Objective And Evaluation",
            bullets=(
                f"Loss: `BCEWithLogits + Dice` with weights `{float(args.bce_weight)}` and `{float(args.dice_weight)}`.",
                f"Optimizer: `AdamW(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
                f"Foreground threshold at eval: `{float(args.foreground_threshold)}`.",
                f"Per-epoch eval tracking cadence: `{resolve_mask_head_eval_every_epochs(args)}`.",
                (
                    f"Mask-head width: projection dim `{resolve_mask_head_projection_dim(args)}`, "
                    f"decoder dim `{resolve_mask_head_decoder_dim(args)}`, "
                    f"GroupNorm groups `{resolve_mask_head_group_norm_groups(args)}`."
                    if variant not in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
                    else f"Probe width: projection dim `{int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS['projection_dim'])}`, hidden dim `{int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS['hidden_dim'])}`."
                ),
                (
                    f"Memory-attention settings: `{resolve_mask_head_memory_token_count(args)}` learned memory tokens, `{resolve_mask_head_attention_heads(args)}` heads, `{resolve_mask_head_attention_blocks(args)}` attention block(s)."
                    if is_memory_head_variant(variant)
                    else "No explicit memory-attention block is used in this run."
                ),
                (
                    f"Residual branch width: projection dim `{int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_projection_dim'])}`, hidden dim `{int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_hidden_dim'])}`, initial residual scale `{float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_scale_init'])}`."
                    if is_coarse_plus_residual_variant(variant)
                    else "No explicit fine residual branch is used in this run."
                ),
                f"Loss: `{resolve_mask_head_loss_name(args)}`.",
                "Primary supervised metrics: direct foreground IoU and Dice.",
                "Auxiliary repo-native metrics: permutation-invariant binary `eval_miou` and `eval_ari` computed from the same foreground prediction and its complement.",
            ),
        ),
        ExperimentTermsSection(
            title="Outputs In This Results Directory",
            bullets=(
                f"Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `{render_relative_path(GLAS_FROZEN_MASK_HEAD_OUTPUT_ROOT)}`.",
                "Training runs additionally write `checkpoint.pt`, `train_history.csv`, and `train_set_summary.json`.",
                "Per-sample artifacts: `sample_rows/<index>.json`, `masks/<index>.npz`, and `visuals/<index>.png` when visuals are enabled.",
                (
                    f"Checkpoint consumed by this run: `{checkpoint_path}`."
                    if checkpoint_path is not None
                    else "This run trains a fresh checkpoint and then evaluates it."
                ),
            ),
        ),
        ExperimentTermsSection(
            title="Failure Semantics",
            bullets=(
                "Hard-fail on missing or unreadable split-aware STLD assets, empty sample sets, or inconsistent SAM pyramid channels across samples.",
                "Hard-fail on missing requested SAM pyramid levels, invalid tensor ranks, or NaN/Inf feature/logit/loss values.",
                "Hard-fail on prediction/target shape mismatches instead of silently resizing targets beyond the explicit image-resolution decoder step.",
            ),
        ),
    ]
    return render_experiment_terms_markdown(
        title="STLD Frozen SAM Mask Head Experiment Terms",
        summary_lines=(
            "This file describes the exact frozen-feature supervised STLD run that produced this directory.",
            "It records the split separation, architecture, losses, metrics, outputs, and explicit failure rules for review.",
        ),
        sections=sections,
        related_paths=(
            render_relative_path(GLAS_FROZEN_MASK_HEAD_EXPERIMENT_CONTRACT_PATH),
            render_relative_path(Path("src/rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py")),
            render_relative_path(Path("src/rwtd_sam3/eval/stld_frozen_feature_mask_head.py")),
        ),
    )


def resolve_torch_device(requested_device: str) -> Any:
    """Resolve the requested torch device string into a torch device."""

    import torch

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise FrozenMaskHeadRuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_glas_frozen_variant_levels(
    *,
    variant: str,
    available_level_names: tuple[str, ...],
) -> tuple[str, ...]:
    if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS:
        _, selected = resolve_glas_frozen_feature_probe_variant(
            variant=variant,
            available_level_names=available_level_names,
        )
        return selected
    return resolve_frozen_mask_head_variant_levels(
        variant=variant,
        available_level_names=available_level_names,
    )


def _build_architecture_bullets(variant: str) -> tuple[str, ...]:
    if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS:
        head_kind = str(GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]["head_kind"])
        if head_kind == "linear":
            return (
                "Probe family: frozen-feature foreground probe.",
                "One selected SAM level feeds a single learnable `1x1` foreground classifier with no hidden layer.",
                "The low-resolution foreground logit map is upsampled directly to image resolution.",
                "This is the smallest supervised readout in the experiment matrix.",
            )
        if variant == "tiny_nonlinear_fpn0":
            return (
                "Probe family: frozen-feature foreground probe.",
                "One selected SAM level feeds `3x3 -> ReLU -> 1x1` with no multiscale fusion.",
                "The low-resolution foreground logit map is upsampled directly to image resolution.",
                "No prompt generator, transformer decoder, or UNet-style architecture is introduced.",
            )
        return (
            "Probe family: frozen-feature foreground probe.",
            "Each selected level is projected with `1x1`, resized to the finest selected grid, concatenated, then processed by `3x3 -> ReLU -> 1x1`.",
            "Fusion stays transparent and low-capacity.",
            "SAM remains completely frozen; only the tiny dense probe parameters are trained.",
        )
    if is_memory_attention_variant(variant):
        return (
            "Probe family: global-memory cross-attention frozen mask head.",
            "The first selected SAM level is projected to the working width and flattened into image tokens on the coarse grid.",
            "A small learned memory bank of texture tokens is shared across one or more cross-attention blocks, where image tokens attend to memory tokens without any prompt roundtrip back into SAM.",
            "If a finer skip is selected, it is projected and fused only after the coarse memory-attention commitment step.",
            "The attended map is decoded by the same shallow `3x3 -> GroupNorm -> GELU` stack and upsampled directly to image resolution in a single pass.",
        )
    if is_memory_control_variant(variant):
        return (
            "Probe family: matched-capacity memory control frozen mask head.",
            "The coarse selected SAM level is projected to the same working width and paired with the same learned memory bank as the attention variant.",
            "Instead of per-token cross-attention, one global pooled image descriptor mixes the memory bank once and broadcasts the result back to every spatial token.",
            "The output path is otherwise the same shallow decoder and direct image-resolution logits used by the attention variant.",
            "This control isolates whether any gains come from attention itself rather than from simply adding a few more trainable parameters.",
        )
    if is_coarse_plus_residual_variant(variant):
        _, fine_level_name = resolve_coarse_plus_residual_levels(variant)
        return (
            "Probe family: coarse-plus-fine residual frozen mask head.",
            "The coarse branch is the same tiny frozen mask head previously used for `fpn_2_only`.",
            f"A separate tiny `{fine_level_name}` branch projects the refinement feature map, applies two `3x3 -> GroupNorm -> GELU` blocks, and predicts an additive residual logit map.",
            "Final logits are `coarse_logits + residual_scale * residual_logits`, with the residual scale initialized near zero.",
        )
    return (
        "Probe family: tiny multiscale frozen mask head.",
        "Each selected SAM level is projected with a learnable `1x1` convolution to 64 channels.",
        "All projected maps are resized to the finest selected grid, concatenated, decoded with two `3x3 -> GroupNorm -> GELU` blocks, and mapped to one foreground logit map.",
        "This legacy baseline is retained for comparison against the smaller probe variants.",
    )


run_stld_frozen_mask_head_train = run_glas_frozen_mask_head_train
run_stld_frozen_mask_head_eval = run_glas_frozen_mask_head_eval
