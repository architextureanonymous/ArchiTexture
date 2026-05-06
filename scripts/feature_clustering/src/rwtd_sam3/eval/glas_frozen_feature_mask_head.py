"""Dense-supervised GlaS baseline on frozen SAM multiscale features.

This module implements a minimal supervised baseline for GlaS:

- the SAM backbone stays frozen
- one tiny multiscale mask head is trained on dense gland masks
- evaluation reports direct foreground IoU / Dice plus auxiliary
  partition-invariant GlaS metrics from the same predictions

The implementation intentionally reuses the existing GlaS loader, the existing
frozen-SAM feature extractor, and the repository's standard run-artifact
writers instead of introducing a separate training stack.
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
from PIL import Image, ImageDraw, ImageFont

from rwtd_sam3.data.glas_binary import (
    GLAS_DATASET_ID,
    GlasBinarySample,
    iter_glas_binary_samples,
    load_glas_binary_overview,
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
    WandbSession,
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
from rwtd_sam3.models.frozen_sam_pyramid_extractor import (
    FrozenSamPyramidExtractorProtocol,
    build_frozen_sam_pyramid_extractor,
    describe_frozen_sam_feature_source,
    resolve_frozen_sam_backbone_family,
)
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
    FrozenSamCoarsePlusMultibankNullRefineMaskHead,
    FrozenSamSerializedProgressiveRefinementMaskHead,
    FrozenSamSerializedLocalRoutedRefinementMaskHead,
    FrozenSamMemoryAttentionMaskHead,
    FrozenSamCoarsePlusFineResidualMaskHead,
    FrozenSamForegroundProbe,
    FrozenSamMultiscaleMaskHead,
    MaskHeadLossResult,
    RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD,
    ResidualHeadCombinationResult,
    ResidualHeadTrainingLossResult,
    build_residual_head_settings_payload,
    combine_residual_head_logits,
    compute_bce_dice_loss,
    compute_residual_head_training_loss,
    count_trainable_parameters,
    is_coarse_plus_residual_variant,
    is_serialized_refine_variant,
    is_serialized_local_routed_refine_variant,
    is_attention_refine_variant,
    is_cross_attention_refine_variant,
    is_multibank_null_refine_variant,
    is_memory_attention_variant,
    is_memory_control_variant,
    is_memory_head_variant,
    resolve_coarse_plus_residual_levels,
    resolve_frozen_mask_head_head_family,
    resolve_glas_frozen_feature_probe_variant,
    resolve_frozen_mask_head_variant_levels,
    resolve_residual_head_coarse_loss_weight,
    resolve_residual_head_gate_mode,
    resolve_residual_head_gate_threshold,
    summarize_residual_head_combination,
    threshold_foreground_logits,
)
from rwtd_sam3.utils.visualization import render_prediction_panel, save_prediction_panel


LOGGER = logging.getLogger(__name__)

GLAS_FROZEN_MASK_HEAD_OUTPUT_ROOT = Path("outputs") / "glas_binary" / "frozen_sam_mask_head"
GLAS_FROZEN_MASK_HEAD_FEATURE_CACHE_MAX_BYTES = 5 << 30
GLAS_FROZEN_MASK_HEAD_BACKBONE_PREPROCESSING_MODES = ("native_resolution", "autosam_style_1024")
AUTOSAM_STYLE_BACKBONE_LONG_SIDE = 1024
GLAS_FROZEN_MASK_HEAD_CONTRACT_PROFILE = {
    "profile_name": "sam3_historical_512_full_train_v1",
    "train_split_policy": "official_full_train",
    "resize_height": 512,
    "resize_width": 512,
    "backbone_preprocessing_mode": "native_resolution",
}
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
_WANDB_TILE_FONT = ImageFont.load_default()


def resolve_frozen_mask_head_wandb_run_name(*, dataset_tag: str, variant: str, explicit_run_name: str | None) -> str:
    if explicit_run_name not in (None, ""):
        return str(explicit_run_name)
    if str(variant) == "fpn_2_only":
        return f"{dataset_tag}_fpn2_only_reset"
    if is_serialized_local_routed_refine_variant(str(variant)):
        return f"{dataset_tag}_fpn2_local_fpn1_local_fpn0_refine_coarse05_zero_warm"
    sanitized = str(variant).replace("fpn_", "fpn").replace("__", "_")
    return f"{dataset_tag}_{sanitized}"


def _render_wandb_titled_tile(image: Image.Image, title: str, *, tile_size: tuple[int, int]) -> Image.Image:
    tile = image.convert("RGB").resize(tile_size, Image.BILINEAR)
    canvas = Image.new("RGB", (tile_size[0], tile_size[1] + 18), color=(255, 255, 255))
    canvas.paste(tile, (0, 18))
    ImageDraw.Draw(canvas).text((4, 4), title, fill=(0, 0, 0), font=_WANDB_TILE_FONT)
    return canvas


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    mask_uint8 = (np.asarray(mask, dtype=np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L").convert("RGB")


def _error_overlay_image(sample: GlasBinarySample, prediction: np.ndarray, target: np.ndarray) -> Image.Image:
    image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8).copy()
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    false_positive = np.logical_and(pred, np.logical_not(gt))
    false_negative = np.logical_and(np.logical_not(pred), gt)
    image[false_positive] = np.array([255, 0, 0], dtype=np.uint8)
    image[false_negative] = np.array([0, 128, 255], dtype=np.uint8)
    return Image.fromarray(image, mode="RGB")


def render_local_refine_wandb_panel(
    *,
    sample: GlasBinarySample,
    output: FrozenResidualMaskHeadOutput,
    metric_summary: str,
) -> Image.Image:
    import torch

    tile_size = (160, 160)
    coarse_probability = torch.sigmoid(output.coarse_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    stage1_probability = torch.sigmoid(output.stage1_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    final_probability = torch.sigmoid(output.logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    final_prediction = final_probability >= 0.5
    panels = [
        _render_wandb_titled_tile(sample.image.convert("RGB"), "Input", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(np.asarray(sample.texture_a_mask, dtype=np.float32)), "GT", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(coarse_probability), "Coarse", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(output.route1_mask[0, 0].detach().cpu().numpy()), "Route1", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(stage1_probability), "Stage1", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(output.route0_mask[0, 0].detach().cpu().numpy()), "Route0", tile_size=tile_size),
        _render_wandb_titled_tile(_mask_to_image(final_probability), "Final", tile_size=tile_size),
        _render_wandb_titled_tile(
            _error_overlay_image(sample, final_prediction, np.asarray(sample.texture_a_mask, dtype=bool)),
            "Error",
            tile_size=tile_size,
        ),
    ]
    top = np.concatenate([np.asarray(panel, dtype=np.uint8) for panel in panels[:4]], axis=1)
    bottom = np.concatenate([np.asarray(panel, dtype=np.uint8) for panel in panels[4:]], axis=1)
    canvas = Image.fromarray(np.concatenate([top, bottom], axis=0), mode="RGB")
    footer = Image.new("RGB", (canvas.width, canvas.height + 18), color=(255, 255, 255))
    footer.paste(canvas, (0, 18))
    ImageDraw.Draw(footer).text((4, 4), metric_summary, fill=(0, 0, 0), font=_WANDB_TILE_FONT)
    return footer


def build_frozen_mask_head_wandb_preview(
    *,
    head_module: Any,
    cached_sample,
    variant: str,
    device: Any,
    metric_summary: str,
) -> Image.Image:
    import torch

    feature_levels = materialize_cached_feature_levels_on_device(cached_sample, device=device)
    with torch.no_grad():
        output = head_module(feature_levels, image_size=cached_sample.image_size)
    final_prediction = threshold_foreground_logits(output.logits, threshold=0.5)
    background_prediction = np.logical_not(final_prediction)
    if is_serialized_local_routed_refine_variant(variant):
        return render_local_refine_wandb_panel(
            sample=cached_sample.sample,
            output=output,
            metric_summary=metric_summary,
        )
    return render_prediction_panel(
        sample=cached_sample.sample,
        prediction_a=final_prediction,
        prediction_b=background_prediction,
        protocol=f"glas_frozen_mask_head:{variant}",
        metric_summary=metric_summary,
    )


@dataclass(frozen=True)
class CachedGlasFeatureSample:
    """One decoded GlaS sample plus cached raw frozen SAM pyramid features."""

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
    """Cached frozen SAM features for one GlaS split and one scale selection."""

    dataset_root: str
    split: str
    extractor: FrozenSamPyramidExtractorProtocol
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


def validate_best_eval_requires_cadence(args) -> None:
    if str(getattr(args, "selection_checkpoint_mode", "final")) == "best_eval" and resolve_mask_head_eval_every_epochs(args) is None:
        raise FrozenMaskHeadRuntimeError(
            "selection_checkpoint_mode=best_eval requires eval_every_epochs to be a positive integer so a best checkpoint can actually be materialized."
        )


def validate_sam3_checkpoint_contract(args) -> None:
    official_checkpoint_path = getattr(args, "official_checkpoint_path", None)
    if official_checkpoint_path in (None, ""):
        return
    path_text = str(official_checkpoint_path).lower()
    if "sam_vit_h" in path_text or "vit_h" in path_text:
        raise FrozenMaskHeadRuntimeError(
            "SAM3 frozen-feature routes do not accept the AutoSAM ViT-H checkpoint path. Use a SAM3-native checkpoint/config path instead.",
            diagnostics={"official_checkpoint_path": official_checkpoint_path},
        )
    if "sam3" not in path_text:
        raise FrozenMaskHeadRuntimeError(
            "SAM3 frozen-feature routes require a SAM3-native checkpoint/config path.",
            diagnostics={"official_checkpoint_path": official_checkpoint_path},
        )


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


def resolve_mask_head_backbone_family(args) -> str:
    return resolve_frozen_sam_backbone_family(str(getattr(args, "model_id", "")))


def resolve_mask_head_feature_source_description(args) -> str:
    return describe_frozen_sam_feature_source(str(getattr(args, "model_id", "")))


def build_mask_head_settings_payload(args) -> dict[str, Any]:
    settings = dict(FROZEN_MASK_HEAD_SETTINGS)
    settings["feature_source"] = resolve_mask_head_feature_source_description(args)
    return settings


def resolve_train_subset_manifest(args) -> FewShotSubsetManifest | None:
    manifest_path = getattr(args, "train_subset_manifest", None)
    if manifest_path in (None, ""):
        return None
    return load_few_shot_subset_manifest(
        manifest_path,
        expected_dataset_id=GLAS_DATASET_ID,
        expected_source_split=str(getattr(args, "train_split", "train")),
    )


def resolve_train_selection_policy(args) -> str:
    if resolve_train_subset_manifest(args) is None:
        return "official_glas_train"
    return "few_shot_subset_manifest_over_official_glas_train"


def resize_glas_binary_sample(sample: GlasBinarySample, *, resize_hw: tuple[int, int] | None) -> GlasBinarySample:
    """Resize one decoded GlaS sample for the mask-head path, if requested."""

    if resize_hw is None:
        return sample
    target_height, target_width = (int(resize_hw[0]), int(resize_hw[1]))
    resized_image = sample.image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)
    gland_mask_uint8 = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    resized_gland_mask = (
        np.asarray(
            Image.fromarray(gland_mask_uint8, mode="L").resize(
                (target_width, target_height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )
        > 0
    )
    resized_background_mask = np.logical_not(resized_gland_mask)
    if not resized_gland_mask.any() or not resized_background_mask.any():
        raise FrozenMaskHeadRuntimeError(
            "Resizing the GlaS sample produced an empty gland or background mask.",
            diagnostics={
                "crop_name": sample.crop_name,
                "resize_hw": (target_height, target_width),
                "gland_positive_pixels": int(resized_gland_mask.sum()),
                "background_positive_pixels": int(resized_background_mask.sum()),
            },
        )
    resized_boundary_mask = boundary_from_region_masks(resized_gland_mask, resized_background_mask)
    return GlasBinarySample(
        index=sample.index,
        dataset_id=sample.dataset_id,
        split=sample.split,
        crop_name=sample.crop_name,
        image=resized_image,
        boundary_mask=np.asarray(resized_boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(resized_gland_mask, dtype=bool),
        texture_b_mask=np.asarray(resized_background_mask, dtype=bool),
        texture_a=sample.texture_a,
        texture_b=sample.texture_b,
        original_texture_a=sample.original_texture_a,
        original_texture_b=sample.original_texture_b,
        oracle_points_a=sample.oracle_points_a,
        oracle_points_b=sample.oracle_points_b,
        evaluation_view=sample.evaluation_view,
        grade_label=sample.grade_label,
    )


def apply_backbone_preprocessing_to_glas_binary_sample(
    sample: GlasBinarySample,
    *,
    backbone_preprocessing_mode: str,
) -> GlasBinarySample:
    """Apply an explicit backbone-input geometry transform while preserving the original sample for eval."""

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

    gland_mask_uint8 = np.asarray(sample.texture_a_mask, dtype=np.uint8) * 255
    resized_foreground_mask = (
        np.asarray(
            Image.fromarray(gland_mask_uint8, mode="L").resize(
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
            "AutoSAM-style backbone preprocessing produced an empty gland or background mask.",
            diagnostics={
                "crop_name": sample.crop_name,
                "backbone_preprocessing_mode": backbone_preprocessing_mode,
                "target_hw": (int(target_height), int(target_width)),
                "foreground_positive_pixels": int(resized_foreground_mask.sum()),
                "background_positive_pixels": int(resized_background_mask.sum()),
            },
        )
    resized_boundary_mask = boundary_from_region_masks(resized_foreground_mask, resized_background_mask)
    return GlasBinarySample(
        index=sample.index,
        dataset_id=sample.dataset_id,
        split=sample.split,
        crop_name=sample.crop_name,
        image=resized_image,
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
        grade_label=sample.grade_label,
    )


def resize_foreground_probability_to_sample_resolution(
    foreground_probability: np.ndarray,
    *,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize a foreground probability map to the requested eval/sample resolution with bilinear interpolation."""

    resolved_target_hw = (int(target_hw[0]), int(target_hw[1]))
    if tuple(int(value) for value in foreground_probability.shape) == resolved_target_hw:
        return np.asarray(foreground_probability, dtype=np.float32)
    probability_image = Image.fromarray(np.asarray(foreground_probability, dtype=np.float32), mode="F")
    resized_probability = np.asarray(
        probability_image.resize((resolved_target_hw[1], resolved_target_hw[0]), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return np.asarray(resized_probability, dtype=np.float32)


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
    """Train the tiny supervised mask head on frozen SAM features and evaluate on GlaS."""

    import gc

    if getattr(args, "train_limit", None) is not None and getattr(args, "train_subset_manifest", None) not in (None, ""):
        raise FrozenMaskHeadRuntimeError(
            "train_limit and train_subset_manifest are mutually exclusive for GlaS frozen-mask-head training.",
            diagnostics={
                "train_limit": getattr(args, "train_limit", None),
                "train_subset_manifest": getattr(args, "train_subset_manifest", None),
            },
        )

    augmentation_policy = resolve_train_augmentation_policy(args)
    eval_every_epochs = resolve_mask_head_eval_every_epochs(args)
    validate_best_eval_requires_cadence(args)
    validate_sam3_checkpoint_contract(args)
    train_subset_manifest = resolve_train_subset_manifest(args)
    setattr(
        args,
        "_resolved_train_subset_manifest",
        train_subset_manifest.to_json_dict() if train_subset_manifest is not None else None,
    )
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )
    train_cache = build_glas_feature_cache(
        extractor=extractor,
        dataset_root=args.dataset_root,
        split=args.train_split,
        limit=args.train_limit,
        variant=args.variant,
        subset_manifest=train_subset_manifest,
        resize_hw=resolve_mask_head_resize_hw(args),
        backbone_preprocessing_mode=resolve_mask_head_backbone_preprocessing_mode(args),
        skip_feature_materialization=(augmentation_policy != "none"),
    )

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
            dataset_root=args.dataset_root,
            split=args.eval_split,
            limit=args.eval_limit,
            variant=args.variant,
            expected_selected_level_names=train_cache.selected_level_names,
            resize_hw=resolve_mask_head_resize_hw(args),
            backbone_preprocessing_mode=resolve_mask_head_backbone_preprocessing_mode(args),
        )

    train_sample_count = len(train_cache.samples)
    selected_level_names = train_cache.selected_level_names
    reference_raw_level_shapes = train_cache.reference_raw_level_shapes
    reference_selected_level_shapes = train_cache.reference_selected_level_shapes
    train_materialization_mode = train_cache.materialization_mode
    if eval_cache is None:
        eval_cache = build_glas_feature_cache(
            extractor=extractor,
            dataset_root=args.dataset_root,
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
    config.update(build_residual_head_settings_payload(args, include_eval_alpha_override=False))
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
    wandb_session = WandbSession(
        enabled=bool(getattr(args, "wandb", False)),
        project=str(getattr(args, "wandb_project", "glas-frozen-sam3")),
        run_name=resolve_frozen_mask_head_wandb_run_name(
            dataset_tag="glas",
            variant=str(args.variant),
            explicit_run_name=getattr(args, "wandb_run_name", None),
        ),
        config=config,
    )
    try:
        head_module, training_artifacts = train_glas_frozen_mask_head(
            train_cache=train_cache,
            output_dir=output_dir,
            args=args,
            eval_cache_for_tracking=eval_cache,
            wandb_session=wandb_session,
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
        del train_cache
        gc.collect()
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
        wandb_session.log(
            {
                "final/fg_iou": summary["mean_metrics"]["direct_foreground_iou"],
                "final/dice": summary["mean_metrics"]["direct_foreground_dice"],
                "final/eval_miou": summary["mean_metrics"]["eval_miou"],
                "final/eval_ari": summary["mean_metrics"]["eval_ari"],
            }
        )
    finally:
        wandb_session.finish()
    summary["train_set_mean_metrics"] = dict(train_summary["mean_metrics"])
    summary["validation_set_mean_metrics"] = None
    summary["train_set_summary_path"] = str(output_dir / "train_set_summary.json")
    summary["test_set_mean_metrics"] = dict(summary["mean_metrics"])
    summary["test_set_summary_path"] = str(output_dir / "summary.json")
    summary["headline_metrics_source"] = "test_set_mean_metrics"
    summary["train_augmentation_policy"] = augmentation_policy
    summary["train_augmentation_summary"] = describe_train_augmentation_policy(augmentation_policy)
    summary["resolved_contract_profile"] = GLAS_FROZEN_MASK_HEAD_CONTRACT_PROFILE["profile_name"]
    summary["resolved_split_policy"] = GLAS_FROZEN_MASK_HEAD_CONTRACT_PROFILE["train_split_policy"]
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
    summary["checkpoint_selection_materialized"] = (
        "best_eval"
        if training_artifacts.best_checkpoint_path is not None and eval_every_epochs is not None
        else "final"
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
    """Evaluate one saved frozen-mask-head checkpoint on a GlaS split."""

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
    if checkpoint.get("model_family", "multiscale_mask_head") != "foreground_probe":
        setattr(args, "projection_dim", int(checkpoint.get("projection_dim", FROZEN_MASK_HEAD_SETTINGS["projection_dim"])))
        setattr(args, "decoder_dim", int(checkpoint.get("decoder_dim", FROZEN_MASK_HEAD_SETTINGS["decoder_dim"])))
        setattr(
            args,
            "group_norm_groups",
            int(checkpoint.get("group_norm_groups", FROZEN_MASK_HEAD_SETTINGS["group_norm_groups"])),
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
    setattr(
        args,
        "backbone_preprocessing_mode",
        str(checkpoint.get("backbone_preprocessing_mode", "native_resolution")),
    )
    setattr(args, "residual_gate_mode", str(checkpoint.get("residual_gate_mode", "none")))
    setattr(
        args,
        "residual_gate_threshold",
        float(checkpoint.get("residual_gate_threshold", RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)),
    )
    setattr(args, "coarse_loss_weight", float(checkpoint.get("coarse_loss_weight", 0.0)))
    setattr(args, "residual_l1_weight", float(checkpoint.get("residual_l1_weight", 0.0)))
    setattr(args, "joker_disable_fpn0", bool(checkpoint.get("joker_disable_fpn0", False)))
    setattr(args, "joker_disable_fpn1", bool(checkpoint.get("joker_disable_fpn1", False)))
    setattr(args, "joker_disable_null_token", bool(checkpoint.get("joker_disable_null_token", False)))
    setattr(args, "joker_use_learned_gate", bool(checkpoint.get("joker_use_learned_gate", False)))
    setattr(args, "joker_zero_init_residual_scale", bool(checkpoint.get("joker_zero_init_residual_scale", False)))
    setattr(args, "joker_zero_init_attn_qkv", bool(checkpoint.get("joker_zero_init_attn_qkv", False)))
    setattr(args, "joker_residual_warmup_epochs", int(checkpoint.get("joker_residual_warmup_epochs", 0)))
    setattr(args, "joker_residual_ramp_epochs", int(checkpoint.get("joker_residual_ramp_epochs", 0)))

    overview = load_glas_binary_overview(args.dataset_root, split=args.split)
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_samples = resolve_eval_sample_count(args.limit, overview.num_examples, dataset_partition)
    if num_samples < 1:
        raise FrozenMaskHeadRuntimeError("The requested GlaS eval split produced zero samples.")

    extractor = build_frozen_sam_pyramid_extractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        official_checkpoint_path=args.official_checkpoint_path,
    )
    eval_cache = build_glas_feature_cache(
        extractor=extractor,
        dataset_root=args.dataset_root,
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
    config.update(build_residual_head_settings_payload(args, include_eval_alpha_override=True))
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
    try:
        head_module.load_state_dict(checkpoint["head_state_dict"])
    except RuntimeError as exc:
        raise FrozenMaskHeadRuntimeError(
            "Loaded GlaS frozen-mask-head checkpoint state_dict is incompatible with the instantiated architecture.",
            diagnostics={"checkpoint_path": str(checkpoint_path), "variant": checkpoint_variant},
        ) from exc
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
    extractor: FrozenSamPyramidExtractorProtocol,
    dataset_root: str,
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
    """Decode GlaS samples and cache the raw frozen SAM pyramid levels."""

    raw_samples = tuple(
        resize_glas_binary_sample(
            sample,
            resize_hw=resize_hw,
        )
        for sample in iter_glas_binary_samples(
            dataset_root=dataset_root,
            split=split,
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
            f"No GlaS samples were loaded for split '{split}' with limit={limit} and start_index={start_index}."
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
        raise FrozenMaskHeadRuntimeError("Failed to build a non-empty GlaS frozen-feature cache.")

    LOGGER.info(
        "Prepared %d GlaS samples for split=%s | levels=%s | materialization_mode=%s | estimated_bundle_mb=%.2f | estimated_total_gb=%.2f | reference_raw_shapes=%s | reference_selected_shapes=%s | resize_hw=%s | backbone_preprocessing_mode=%s",
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
        dataset_root=str(dataset_root),
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
    if is_serialized_local_routed_refine_variant(variant):
        coarse_level_name = "fpn_2"
        mid_level_name = "fpn_1"
        fine_level_name = "fpn_0"
        return FrozenSamSerializedLocalRoutedRefinementMaskHead(
            level_input_dims=level_input_dims,
            coarse_level_name=coarse_level_name,
            mid_level_name=mid_level_name,
            fine_level_name=fine_level_name,
            projection_dim=resolve_mask_head_projection_dim(args),
            decoder_dim=resolve_mask_head_decoder_dim(args),
            group_norm_groups=resolve_mask_head_group_norm_groups(args),
            residual_projection_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
            residual_hidden_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
            residual_scale_init=float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
        ).module
    if is_coarse_plus_residual_variant(variant):
        coarse_level_name, fine_level_name = resolve_coarse_plus_residual_levels(variant)
        if is_multibank_null_refine_variant(variant):
            return FrozenSamCoarsePlusMultibankNullRefineMaskHead(
                level_input_dims=level_input_dims,
                coarse_level_name=coarse_level_name,
                mid_level_name="fpn_1",
                fine_level_name="fpn_0",
                projection_dim=resolve_mask_head_projection_dim(args),
                decoder_dim=resolve_mask_head_decoder_dim(args),
                group_norm_groups=resolve_mask_head_group_norm_groups(args),
                residual_projection_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
                residual_hidden_dim=int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
                attention_hidden_dim=int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
                attention_heads=int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])),
                cross_attn_query_stride=int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])),
                use_mid_level=not bool(getattr(args, "joker_disable_fpn1", False)),
                use_fine_level=not bool(getattr(args, "joker_disable_fpn0", False)),
                use_null_token=not bool(getattr(args, "joker_disable_null_token", False)),
                use_learned_gate=bool(getattr(args, "joker_use_learned_gate", False)),
                zero_init_residual_scale=bool(getattr(args, "joker_zero_init_residual_scale", False)),
                zero_init_attention_qkv=bool(getattr(args, "joker_zero_init_attn_qkv", False)),
                residual_warmup_epochs=int(getattr(args, "joker_residual_warmup_epochs", 0)),
                residual_ramp_epochs=int(getattr(args, "joker_residual_ramp_epochs", 0)),
                residual_scale_init=float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
            ).module
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
    if is_serialized_refine_variant(variant):
        return FrozenSamSerializedProgressiveRefinementMaskHead(
            level_input_dims=level_input_dims,
            projection_dim=resolve_mask_head_projection_dim(args),
            decoder_dim=resolve_mask_head_decoder_dim(args),
            group_norm_groups=resolve_mask_head_group_norm_groups(args),
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
    if model_family == "serialized_local_routed_refinement_mask_head":
        return FrozenSamSerializedLocalRoutedRefinementMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            mid_level_name=str(checkpoint.get("mid_level_name", "fpn_1")),
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
            zero_init_residual_scale=bool(checkpoint.get("joker_zero_init_residual_scale", False)),
            zero_init_attention_qkv=bool(checkpoint.get("joker_zero_init_attn_qkv", False)),
            residual_warmup_epochs=int(checkpoint.get("joker_residual_warmup_epochs", 0)),
            residual_ramp_epochs=int(checkpoint.get("joker_residual_ramp_epochs", 0)),
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
    if model_family == "coarse_plus_multibank_null_refine_mask_head":
        return FrozenSamCoarsePlusMultibankNullRefineMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            mid_level_name=str(checkpoint.get("mid_level_name", "fpn_1")),
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
            attention_hidden_dim=int(
                checkpoint.get("attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])
            ),
            attention_heads=int(
                checkpoint.get("attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])
            ),
            cross_attn_query_stride=int(
                checkpoint.get("cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])
            ),
            use_mid_level=not bool(checkpoint.get("joker_disable_fpn1", False)),
            use_fine_level=not bool(checkpoint.get("joker_disable_fpn0", False)),
            use_null_token=not bool(checkpoint.get("joker_disable_null_token", False)),
            use_learned_gate=bool(checkpoint.get("joker_use_learned_gate", False)),
            zero_init_residual_scale=bool(checkpoint.get("joker_zero_init_residual_scale", False)),
            zero_init_attention_qkv=bool(checkpoint.get("joker_zero_init_attn_qkv", False)),
            residual_warmup_epochs=int(checkpoint.get("joker_residual_warmup_epochs", 0)),
            residual_ramp_epochs=int(checkpoint.get("joker_residual_ramp_epochs", 0)),
            residual_scale_init=float(
                checkpoint.get("residual_scale_init", FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            ),
        ).module
    if model_family == "serialized_progressive_refinement_mask_head":
        return FrozenSamSerializedProgressiveRefinementMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            mid_level_name=str(checkpoint.get("mid_level_name", "fpn_1")),
            fine_level_name=str(checkpoint.get("fine_level_name", "fpn_0")),
            projection_dim=int(checkpoint["projection_dim"]),
            decoder_dim=int(checkpoint["decoder_dim"]),
            group_norm_groups=int(checkpoint["group_norm_groups"]),
        ).module
    if model_family == "serialized_local_routed_refinement_mask_head":
        return FrozenSamSerializedLocalRoutedRefinementMaskHead(
            level_input_dims={name: int(value) for name, value in checkpoint["level_input_dims"].items()},
            coarse_level_name=str(checkpoint.get("coarse_level_name", "fpn_2")),
            mid_level_name=str(checkpoint.get("mid_level_name", "fpn_1")),
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
            zero_init_residual_scale=bool(checkpoint.get("joker_zero_init_residual_scale", False)),
            zero_init_attention_qkv=bool(checkpoint.get("joker_zero_init_attn_qkv", False)),
            residual_warmup_epochs=int(checkpoint.get("joker_residual_warmup_epochs", 0)),
            residual_ramp_epochs=int(checkpoint.get("joker_residual_ramp_epochs", 0)),
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
    extractor: FrozenSamPyramidExtractorProtocol,
    variant: str,
    backbone_preprocessing_mode: str,
    expected_selected_level_names: tuple[str, ...] | None,
    reference_selected_level_names: tuple[str, ...] | None,
    reference_channels: dict[str, int] | None,
) -> CachedGlasFeatureSample:
    """Extract and validate one GlaS sample's raw frozen SAM multiscale features."""

    backbone_sample = apply_backbone_preprocessing_to_glas_binary_sample(
        sample,
        backbone_preprocessing_mode=backbone_preprocessing_mode,
    )
    image_size, pyramid = extractor.extract_sam_pyramid(backbone_sample.image)
    expected_image_size = (int(backbone_sample.height), int(backbone_sample.width))
    if tuple(int(value) for value in image_size) != expected_image_size:
        raise FrozenMaskHeadRuntimeError(
            "Frozen-mask-head feature extraction returned an image size that does not match the decoded GlaS sample.",
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
            "Discovered inconsistent SAM channel counts across GlaS samples.",
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
    """Estimate the raw CPU memory footprint of one cached GlaS feature sample."""

    return int(sum(int(feature_map.nbytes) for feature_map in sample.selected_feature_levels.values()))


def iter_cached_glas_feature_samples(cache: GlasFeatureCache):
    """Yield prepared GlaS feature samples, streaming re-extraction when needed."""

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
    """Move one cached GlaS sample's selected raw feature pyramid onto a torch device."""

    import torch

    materialized: dict[str, Any] = {}
    for level_name in sample.selected_level_names:
        feature_map = np.asarray(sample.selected_feature_levels[level_name], dtype=np.float32)
        if feature_map.ndim != 3:
            raise FrozenMaskHeadRuntimeError(
                f"Cached GlaS feature map must have shape [C,H,W], got {feature_map.shape} for {level_name}."
            )
        tensor = torch.as_tensor(feature_map[None], dtype=torch.float32, device=device)
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise FrozenMaskHeadRuntimeError(
                f"Materialized GlaS feature tensor must have shape [1,C,H,W], got {tuple(int(value) for value in tensor.shape)}."
            )
        materialized[level_name] = tensor
    return materialized


def train_glas_frozen_mask_head(
    *,
    train_cache: GlasFeatureCache,
    output_dir: Path,
    args,
    eval_cache_for_tracking: GlasFeatureCache | None = None,
    selection_cache_for_tracking: GlasFeatureCache | None = None,
    wandb_session: WandbSession | None = None,
) -> tuple[Any, GlasMaskHeadTrainingArtifacts]:
    """Optimize only the tiny supervised mask head on cached frozen GlaS features."""

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
    residual_gate_mode = resolve_residual_head_gate_mode(getattr(args, "residual_gate_mode", "none"))
    residual_gate_threshold = resolve_residual_head_gate_threshold(
        getattr(args, "residual_gate_threshold", RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)
    )
    coarse_loss_weight = resolve_residual_head_coarse_loss_weight(args)
    residual_l1_weight = float(getattr(args, "residual_l1_weight", 0.0))
    attention_sparsity_weight = float(getattr(args, "attention_sparsity_weight", 0.0))
    best_epoch_by_eval_dice: int | None = None
    best_eval_dice: float | None = None
    best_state_dict: dict[str, Any] | None = None

    LOGGER.info(
        "Frozen-mask-head training start | variant=%s levels=%s materialization_mode=%s trainable_params=%d lr=%.6f wd=%.6f epochs=%d train_aug_policy=%s loss_variant=%s boundary_weight=%.3f eval_every_epochs=%s residual_gate_mode=%s residual_gate_threshold=%.3f coarse_loss_weight=%.3f residual_l1_weight=%.3f attention_sparsity_weight=%.3f joker_zero_res=%s joker_zero_qkv=%s joker_warmup=%d joker_ramp=%d",
        args.variant,
        list(train_cache.selected_level_names),
        train_cache.materialization_mode,
        trainable_parameter_count,
        float(args.learning_rate),
        float(args.weight_decay),
        int(args.num_epochs),
        augmentation_policy,
        loss_variant,
        boundary_weight,
        eval_every_epochs,
        residual_gate_mode,
        residual_gate_threshold,
        coarse_loss_weight,
        residual_l1_weight,
        attention_sparsity_weight,
        bool(getattr(args, "joker_zero_init_residual_scale", False)),
        bool(getattr(args, "joker_zero_init_attn_qkv", False)),
        int(getattr(args, "joker_residual_warmup_epochs", 0)),
        int(getattr(args, "joker_residual_ramp_epochs", 0)),
    )

    for epoch in range(int(args.num_epochs)):
        head_module.train()
        if hasattr(head_module, "set_residual_training_progress"):
            head_module.set_residual_training_progress(epoch_index=int(epoch))
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
            residual_combination: ResidualHeadCombinationResult | None = None
            if is_serialized_local_routed_refine_variant(args.variant):
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
                    coarse_loss_weight=coarse_loss_weight,
                )
                final_logits = output.logits
                coarse_logits = output.coarse_logits
                loss_result = loss_bundle.final_loss_result
                coarse_loss_result = loss_bundle.coarse_loss_result
                residual_l1_loss = loss_bundle.residual_l1_loss
                total_loss = loss_bundle.total_loss
            elif is_coarse_plus_residual_variant(args.variant):
                residual_combination = combine_residual_head_logits(
                    output,
                    residual_gate_mode=residual_gate_mode,
                    residual_gate_threshold=residual_gate_threshold,
                )
                final_logits = residual_combination.final_logits
                coarse_logits = residual_combination.coarse_logits
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
                    coarse_loss_weight=coarse_loss_weight,
                    residual_l1_weight=residual_l1_weight,
                    attention_sparsity_weight=attention_sparsity_weight,
                )
                loss_result = loss_bundle.final_loss_result
                coarse_loss_result = loss_bundle.coarse_loss_result
                residual_l1_loss = loss_bundle.residual_l1_loss
                total_loss = loss_bundle.total_loss
            else:
                final_logits = output.logits
                coarse_logits = None
                coarse_loss_result = None
                residual_l1_loss = None
                total_loss = None
                loss_result = compute_bce_dice_loss(
                    final_logits,
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
        if wandb_session is not None:
            wandb_session.log(
                {
                    "train/loss": history_row["mean_train_loss"],
                    "train/bce_loss": history_row["mean_bce_loss"],
                    "train/dice_loss": history_row["mean_dice_loss"],
                    "train/predicted_positive_fraction": history_row["mean_predicted_positive_fraction"],
                    "train/coarse_loss": history_row.get("mean_coarse_loss"),
                    "train/residual_l1_loss": history_row.get("mean_residual_l1_loss"),
                },
                step=epoch + 1,
            )
        LOGGER.info(
            "Frozen-mask-head epoch=%d/%d | mean_loss=%.6f mean_bce=%.6f mean_dice=%.6f mean_pred_pos=%.4f%s%s",
            epoch + 1,
            int(args.num_epochs),
            history_row["mean_train_loss"],
            history_row["mean_bce_loss"],
            history_row["mean_dice_loss"],
            history_row["mean_predicted_positive_fraction"],
            (
                f" mean_coarse={history_row['mean_coarse_loss']:.6f}"
                if "mean_coarse_loss" in history_row
                else ""
            ),
            (
                f" mean_residual_l1={history_row['mean_residual_l1_loss']:.6f}"
                if "mean_residual_l1_loss" in history_row
                else ""
            ),
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
            if epoch_summary.get("coarse_mean_metrics") is not None:
                coarse_mean = epoch_summary["coarse_mean_metrics"]
                epoch_eval_row.update(
                    {
                        "coarse_direct_foreground_iou": coarse_mean["coarse_direct_foreground_iou"],
                        "coarse_direct_foreground_dice": coarse_mean["coarse_direct_foreground_dice"],
                        "coarse_direct_eval_miou": coarse_mean["coarse_direct_eval_miou"],
                        "coarse_direct_eval_ari": coarse_mean["coarse_direct_eval_ari"],
                        "final_minus_coarse_direct_foreground_iou": coarse_mean["final_minus_coarse_direct_foreground_iou"],
                        "final_minus_coarse_direct_foreground_dice": coarse_mean["final_minus_coarse_direct_foreground_dice"],
                        "final_minus_coarse_eval_miou": coarse_mean["final_minus_coarse_eval_miou"],
                        "final_minus_coarse_eval_ari": coarse_mean["final_minus_coarse_eval_ari"],
                    }
                )
            if epoch_summary.get("residual_diagnostics_mean") is not None:
                residual_mean = epoch_summary["residual_diagnostics_mean"]
                epoch_eval_row.update(
                    {
                        "learned_residual_scale": residual_mean["learned_residual_scale"],
                        "effective_residual_scale": residual_mean["effective_residual_scale"],
                        "residual_runtime_multiplier": residual_mean["residual_runtime_multiplier"],
                        "residual_scale_source": residual_mean["residual_scale_source"],
                        "residual_gate_mode": residual_mean["residual_gate_mode"],
                        "residual_gate_threshold": residual_mean["residual_gate_threshold"],
                        "residual_gate_mean": residual_mean["residual_gate_mean"],
                        "residual_gate_fraction_ge_half": residual_mean["residual_gate_fraction_ge_half"],
                        "attention_mean": residual_mean["attention_mean"],
                        "attention_real_mean": residual_mean["attention_real_mean"],
                        "attention_null_mean": residual_mean["attention_null_mean"],
                        "attention_fpn1_mean": residual_mean["attention_fpn1_mean"],
                        "attention_fpn0_mean": residual_mean["attention_fpn0_mean"],
                        "residual_scale_fpn1": residual_mean.get("residual_scale_fpn1"),
                        "residual_scale_fpn0": residual_mean.get("residual_scale_fpn0"),
                        "alpha_fpn1": residual_mean.get("alpha_fpn1"),
                        "alpha_fpn0": residual_mean.get("alpha_fpn0"),
                        "attention_fraction_ge_half": residual_mean["attention_fraction_ge_half"],
                        "mean_abs_residual_logits": residual_mean["mean_abs_residual_logits"],
                        "mean_abs_residual_contribution": residual_mean["mean_abs_residual_contribution"],
                        "residual_sign_flip_fraction": residual_mean["residual_sign_flip_fraction"],
                        "final_minus_coarse_mean_abs": residual_mean["final_minus_coarse_mean_abs"],
                    }
                )
            epoch_eval_history_rows.append(epoch_eval_row)
            if wandb_session is not None:
                wandb_session.log(
                    {
                        "eval/fg_iou": epoch_eval_row["direct_foreground_iou"],
                        "eval/dice": epoch_eval_row["direct_foreground_dice"],
                        "eval/eval_miou": epoch_eval_row["eval_miou"],
                        "eval/eval_ari": epoch_eval_row["eval_ari"],
                        "eval/coarse_fg_iou": epoch_eval_row.get("coarse_direct_foreground_iou"),
                        "eval/coarse_dice": epoch_eval_row.get("coarse_direct_foreground_dice"),
                        "model/alpha_fpn1": epoch_eval_row.get("alpha_fpn1"),
                        "model/alpha_fpn0": epoch_eval_row.get("alpha_fpn0"),
                        "model/scale_fpn1": epoch_eval_row.get("residual_scale_fpn1"),
                        "model/scale_fpn0": epoch_eval_row.get("residual_scale_fpn0"),
                    },
                    step=epoch + 1,
                )
                preview_sample = next(iter_cached_glas_feature_samples(eval_cache_for_tracking), None)
                if preview_sample is not None and bool(getattr(wandb_session, "enabled", False)):
                    preview_metric_summary = (
                        f"epoch={epoch + 1} fg_iou={epoch_eval_row['direct_foreground_iou']:.3f} "
                        f"dice={epoch_eval_row['direct_foreground_dice']:.3f} "
                        f"miou={epoch_eval_row['eval_miou']:.3f} ari={epoch_eval_row['eval_ari']:.3f}"
                    )
                    wandb_session.log(
                        {
                            "eval/preview": wandb_session._wandb.Image(
                                build_frozen_mask_head_wandb_preview(
                                    head_module=head_module,
                                    cached_sample=preview_sample,
                                    variant=args.variant,
                                    device=device,
                                    metric_summary=preview_metric_summary,
                                ),
                                caption=preview_metric_summary,
                            )
                        },
                        step=epoch + 1,
                    )
            if best_eval_dice is None or float(epoch_eval_row["direct_foreground_dice"]) > float(best_eval_dice):
                best_eval_dice = float(epoch_eval_row["direct_foreground_dice"])
                best_epoch_by_eval_dice = int(epoch + 1)
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in head_module.state_dict().items()
                }
            LOGGER.info(
                "Frozen-mask-head epoch-eval=%d | eval_fg_iou=%.6f eval_dice=%.6f eval_miou=%.6f eval_ari=%.6f s1=%.4f s0=%.4f a1=%.2f a0=%.2f attn_f1=%s attn_f0=%s",
                epoch + 1,
                epoch_eval_row["direct_foreground_iou"],
                epoch_eval_row["direct_foreground_dice"],
                epoch_eval_row["eval_miou"],
                epoch_eval_row["eval_ari"],
                float(epoch_eval_row.get("residual_scale_fpn1") or 0.0),
                float(epoch_eval_row.get("residual_scale_fpn0") or 0.0),
                float(epoch_eval_row.get("alpha_fpn1") or 0.0),
                float(epoch_eval_row.get("alpha_fpn0") or 0.0),
                ("n/a" if epoch_eval_row.get("attention_fpn1_mean") is None else f"{float(epoch_eval_row['attention_fpn1_mean']):.6f}"),
                ("n/a" if epoch_eval_row.get("attention_fpn0_mean") is None else f"{float(epoch_eval_row['attention_fpn0_mean']):.6f}"),
            )

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_payload = {
        "variant": args.variant,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[args.variant]["summary"],
        "model_id": args.model_id,
        "backbone_family": resolve_mask_head_backbone_family(args),
        "feature_source": resolve_mask_head_feature_source_description(args),
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
    }
    checkpoint_payload.update(build_residual_head_settings_payload(args, include_eval_alpha_override=False))
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
        if is_serialized_refine_variant(args.variant):
            residual_model_family = "serialized_progressive_refinement_mask_head"
            residual_head_kind = "serialized_progressive_refinement"
        elif is_serialized_local_routed_refine_variant(args.variant):
            residual_model_family = "serialized_local_routed_refinement_mask_head"
            residual_head_kind = "serialized_local_routed_refinement"
        elif is_multibank_null_refine_variant(args.variant):
            residual_model_family = "coarse_plus_multibank_null_refine_mask_head"
            residual_head_kind = "coarse_plus_multibank_null_refine"
        elif is_cross_attention_refine_variant(args.variant):
            residual_model_family = "coarse_plus_cross_attention_refine_mask_head"
            residual_head_kind = "coarse_plus_cross_attention_refine"
        elif is_attention_refine_variant(args.variant):
            residual_model_family = "coarse_plus_attention_refine_mask_head"
            residual_head_kind = "coarse_plus_attention_refine"
        else:
            residual_model_family = "coarse_plus_fine_residual_mask_head"
            residual_head_kind = "coarse_plus_fine_residual"
        checkpoint_payload.update(
            {
                "model_family": residual_model_family,
                "head_kind": residual_head_kind,
                "projection_dim": resolve_mask_head_projection_dim(args),
                "decoder_dim": resolve_mask_head_decoder_dim(args),
                "group_norm_groups": resolve_mask_head_group_norm_groups(args),
                "coarse_level_name": coarse_level_name,
                "fine_level_name": fine_level_name,
                "mid_level_name": "fpn_1",
                "residual_projection_dim": int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"]),
                "residual_hidden_dim": int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"]),
                "attention_hidden_dim": int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
                "attention_heads": int(getattr(args, "attention_heads", FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS["attention_heads"])),
                "cross_attn_query_stride": int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])),
                "residual_scale_init": float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"]),
                "joker_disable_fpn0": bool(getattr(args, "joker_disable_fpn0", False)),
                "joker_disable_fpn1": bool(getattr(args, "joker_disable_fpn1", False)),
                "joker_disable_null_token": bool(getattr(args, "joker_disable_null_token", False)),
                "joker_use_learned_gate": bool(getattr(args, "joker_use_learned_gate", False)),
                "joker_zero_init_residual_scale": bool(getattr(args, "joker_zero_init_residual_scale", False)),
                "joker_zero_init_attn_qkv": bool(getattr(args, "joker_zero_init_attn_qkv", False)),
                "joker_residual_warmup_epochs": int(getattr(args, "joker_residual_warmup_epochs", 0)),
                "joker_residual_ramp_epochs": int(getattr(args, "joker_residual_ramp_epochs", 0)),
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
    """Evaluate one trained frozen mask head on one cached GlaS split."""

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
        if is_serialized_local_routed_refine_variant(variant):
            final_logits = output.logits
            coarse_logits = output.coarse_logits
        elif is_coarse_plus_residual_variant(variant):
            residual_combination = combine_residual_head_logits(
                output,
                residual_alpha_override=getattr(args, "residual_alpha_override", None),
                residual_gate_mode=getattr(args, "residual_gate_mode", "none"),
                residual_gate_threshold=float(
                    getattr(args, "residual_gate_threshold", RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)
                ),
            )
            final_logits = residual_combination.final_logits
            coarse_logits = residual_combination.coarse_logits
        else:
            final_logits = output.logits
            coarse_logits = None
        final_foreground_probability = torch.sigmoid(final_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        final_foreground_probability = resize_foreground_probability_to_sample_resolution(
            final_foreground_probability,
            target_hw=(int(cached_sample.sample.height), int(cached_sample.sample.width)),
        )
        final_foreground_prediction = np.asarray(
            final_foreground_probability >= float(args.foreground_threshold),
            dtype=bool,
        )
        final_background_prediction = np.logical_not(final_foreground_prediction)
        if coarse_logits is not None:
            coarse_foreground_probability = torch.sigmoid(coarse_logits)[0, 0].detach().cpu().numpy().astype(np.float32)
            coarse_foreground_probability = resize_foreground_probability_to_sample_resolution(
                coarse_foreground_probability,
                target_hw=(int(cached_sample.sample.height), int(cached_sample.sample.width)),
            )
            coarse_foreground_prediction = np.asarray(
                coarse_foreground_probability >= float(args.foreground_threshold),
                dtype=bool,
            )
            coarse_background_prediction = np.logical_not(coarse_foreground_prediction)
        else:
            coarse_foreground_probability = None
            coarse_foreground_prediction = None
            coarse_background_prediction = None

        direct_foreground_metrics = compute_binary_metrics(
            np.asarray(final_foreground_prediction, dtype=bool),
            np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
        )
        assignment = select_best_binary_assignment(
            np.asarray(final_foreground_prediction, dtype=bool),
            np.asarray(final_background_prediction, dtype=bool),
            np.asarray(cached_sample.sample.texture_a_mask, dtype=bool),
            np.asarray(cached_sample.sample.texture_b_mask, dtype=bool),
            "foreground_head",
            "background_complement",
        )
        if residual_combination is not None or is_serialized_local_routed_refine_variant(variant):
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
            residual_summary = (
                summarize_residual_head_combination(
                    residual_combination,
                    boundary_mask=np.asarray(cached_sample.sample.boundary_mask, dtype=bool),
                )
                if residual_combination is not None
                else {
                    "learned_residual_scale": float(getattr(output, "residual_scale_fpn0", getattr(output, "residual_scale", 0.0))),
                    "effective_residual_scale": float(
                        float(getattr(output, "alpha_fpn0", 0.0))
                        * float(getattr(output, "residual_scale_fpn0", getattr(output, "residual_scale", 0.0)))
                    ),
                    "residual_scale_source": "learned",
                    "residual_runtime_multiplier": 1.0,
                    "residual_gate_mode": "none",
                    "residual_gate_threshold": None,
                    "residual_gate_mean": float(getattr(output, "routed_mask_fpn0", None).float().mean().cpu().item()) if getattr(output, "routed_mask_fpn0", None) is not None else None,
                    "residual_gate_fraction_ge_half": float((getattr(output, "routed_mask_fpn0", None) >= 0.5).float().mean().cpu().item()) if getattr(output, "routed_mask_fpn0", None) is not None else None,
                    "attention_mean": None,
                    "attention_real_mean": None,
                    "attention_null_mean": None,
                    "attention_fpn1_mean": None,
                    "attention_fpn0_mean": None,
                    "residual_scale_fpn1": float(getattr(output, "residual_scale_fpn1", 0.0)),
                    "residual_scale_fpn0": float(getattr(output, "residual_scale_fpn0", 0.0)),
                    "alpha_fpn1": float(getattr(output, "alpha_fpn1", 0.0)),
                    "alpha_fpn0": float(getattr(output, "alpha_fpn0", 0.0)),
                    "attention_fraction_ge_half": None,
                    "mean_abs_residual_logits": float(output.residual_logits.detach().abs().mean().cpu().item()),
                    "mean_abs_residual_contribution": float((output.logits - output.coarse_logits).detach().abs().mean().cpu().item()),
                    "residual_sign_flip_fraction": float(((output.coarse_logits >= 0) != (output.logits >= 0)).float().mean().cpu().item()),
                    "final_minus_coarse_mean_abs": float((output.logits - output.coarse_logits).abs().mean().cpu().item()),
                }
            )
        else:
            coarse_direct_foreground_metrics = None
            coarse_assignment = None
            residual_summary = None
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
                foreground_prediction=final_foreground_prediction,
                background_prediction=final_background_prediction,
                foreground_probability=final_foreground_probability,
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
                            prediction_a=final_foreground_prediction,
                            prediction_b=final_background_prediction,
                        ),
                        protocol=f"glas_frozen_mask_head:{variant}",
                        visual_path=Path("visuals") / f"{cached_sample.sample.index}.png",
                    )
                )

    if not rows:
        raise FrozenMaskHeadRuntimeError("Frozen-mask-head evaluation produced no GlaS sample rows.")

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
            logits_for_profile = output.logits
            if is_coarse_plus_residual_variant(variant):
                logits_for_profile = combine_residual_head_logits(
                    output,
                    residual_alpha_override=getattr(args, "residual_alpha_override", None),
                    residual_gate_mode=getattr(args, "residual_gate_mode", "none"),
                    residual_gate_threshold=float(
                        getattr(args, "residual_gate_threshold", RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)
                    ),
                ).final_logits
            _ = threshold_foreground_logits(logits_for_profile, threshold=float(args.foreground_threshold))
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device=device)
            peak_memory_bytes = max(int(peak_memory_bytes), int(torch.cuda.max_memory_allocated(device=device)))
        timings_seconds.append(float(time.perf_counter() - start))
    if not timings_seconds:
        raise FrozenMaskHeadRuntimeError("Frozen-mask-head inference profiling received zero GlaS samples.")
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
    total_pixels = int(cached_sample.image_size[0] * cached_sample.image_size[1])
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
        "grade_label": cached_sample.sample.grade_label,
        "checkpoint_path": str(checkpoint_path),
        "foreground_evaluation_view": "direct_gland_foreground",
        "foreground_assignment_used": "foreground_head->gland,background_complement->background",
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
        "attended_feature_shape_json": (json.dumps(output.attended_feature_shape) if getattr(output, "attended_feature_shape", None) is not None else None),
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
            "residual_runtime_multiplier": mean_optional("residual_runtime_multiplier"),
            "residual_gate_mean": mean_optional("residual_gate_mean"),
            "residual_gate_fraction_ge_half": mean_optional("residual_gate_fraction_ge_half"),
            "attention_mean": mean_optional("attention_mean"),
            "attention_fraction_ge_half": mean_optional("attention_fraction_ge_half"),
            "attention_real_mean": mean_optional("attention_real_mean"),
            "attention_null_mean": mean_optional("attention_null_mean"),
            "attention_fpn1_mean": mean_optional("attention_fpn1_mean"),
            "attention_fpn0_mean": mean_optional("attention_fpn0_mean"),
            "mean_abs_residual_logits": mean_optional("mean_abs_residual_logits"),
            "mean_abs_gated_residual_logits": mean_optional("mean_abs_gated_residual_logits"),
            "mean_abs_residual_contribution": mean_optional("mean_abs_residual_contribution"),
            "residual_sign_flip_fraction": mean_optional("residual_sign_flip_fraction"),
            "final_minus_coarse_mean_abs": mean_optional("final_minus_coarse_mean_abs"),
            "boundary_mean_abs_residual_contribution": mean_optional("boundary_mean_abs_residual_contribution"),
            "non_boundary_mean_abs_residual_contribution": mean_optional("non_boundary_mean_abs_residual_contribution"),
            "boundary_attention_mean": mean_optional("boundary_attention_mean"),
            "non_boundary_attention_mean": mean_optional("non_boundary_attention_mean"),
            "residual_gate_mode": first_optional("residual_gate_mode"),
            "residual_gate_threshold": mean_optional("residual_gate_threshold"),
            "residual_alpha_override": mean_optional("residual_alpha_override"),
            "residual_scale_source": first_optional("residual_scale_source"),
        }
    coarse_level_name, fine_level_name = (
        resolve_coarse_plus_residual_levels(variant) if is_coarse_plus_residual_variant(variant) else (None, None)
    )
    return {
        "run_kind": run_kind,
        "variant": variant,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[variant]["summary"],
        "dataset_id": GLAS_DATASET_ID,
        "split": cache.split,
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "dataset_root": str(cache.dataset_root),
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
        "foreground_evaluation_view": "direct_gland_foreground",
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
                "coarse_plus_multibank_null_refine_mask_head"
                if is_multibank_null_refine_variant(variant)
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
            )
        ),
        "head_kind": (
            GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]["head_kind"]
            if variant in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "coarse_plus_multibank_null_refine"
                if is_multibank_null_refine_variant(variant)
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
        "trainable_parameter_count": int(trainable_parameter_count),
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
                "residual_runtime_multiplier": residual_diagnostics_mean["residual_runtime_multiplier"],
                "residual_scale_source": residual_diagnostics_mean["residual_scale_source"],
                "residual_gate_mode": residual_diagnostics_mean["residual_gate_mode"],
                "residual_gate_threshold": residual_diagnostics_mean["residual_gate_threshold"],
                "residual_alpha_override": residual_diagnostics_mean["residual_alpha_override"],
                "residual_gate_mean": residual_diagnostics_mean["residual_gate_mean"],
                "residual_gate_fraction_ge_half": residual_diagnostics_mean["residual_gate_fraction_ge_half"],
                "attention_mean": residual_diagnostics_mean["attention_mean"],
                "attention_fraction_ge_half": residual_diagnostics_mean["attention_fraction_ge_half"],
                "attention_real_mean": residual_diagnostics_mean["attention_real_mean"],
                "attention_null_mean": residual_diagnostics_mean["attention_null_mean"],
                "attention_fpn1_mean": residual_diagnostics_mean["attention_fpn1_mean"],
                "attention_fpn0_mean": residual_diagnostics_mean["attention_fpn0_mean"],
                "mean_abs_residual_logits": residual_diagnostics_mean["mean_abs_residual_logits"],
                "mean_abs_gated_residual_logits": residual_diagnostics_mean["mean_abs_gated_residual_logits"],
                "mean_abs_residual_contribution": residual_diagnostics_mean["mean_abs_residual_contribution"],
                "residual_sign_flip_fraction": residual_diagnostics_mean["residual_sign_flip_fraction"],
                "final_minus_coarse_mean_abs": residual_diagnostics_mean["final_minus_coarse_mean_abs"],
                "boundary_mean_abs_residual_contribution": residual_diagnostics_mean["boundary_mean_abs_residual_contribution"],
                "non_boundary_mean_abs_residual_contribution": residual_diagnostics_mean["non_boundary_mean_abs_residual_contribution"],
                "boundary_attention_mean": residual_diagnostics_mean["boundary_attention_mean"],
                "non_boundary_attention_mean": residual_diagnostics_mean["non_boundary_attention_mean"],
            }
        )
    return flattened


def build_glas_frozen_mask_head_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GlaS Frozen SAM Mask Head Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Variant summary: {summary['variant_summary']}",
        f"- Split: `{summary['split']}`",
        f"- Train selection policy: `{summary.get('train_selection_policy')}`",
        f"- Resolved split policy: `{summary.get('resolved_split_policy')}`",
        f"- Resolved contract profile: `{summary.get('resolved_contract_profile')}`",
        f"- Train subset manifest: `{summary.get('train_subset_manifest_output_path') or summary.get('train_subset_manifest_path')}`",
        f"- Selected levels: {', '.join(f'`{name}`' for name in summary['selected_level_names'])}",
        f"- Model family: `{summary['model_family']}`",
        f"- Head kind: `{summary['head_kind']}`",
        f"- Trainable params: `{summary['trainable_parameter_count']}`",
        f"- Preprocessing: `{summary.get('preprocessing_summary')}`",
        f"- Train augmentation: `{summary.get('train_augmentation_summary')}`",
        f"- Loss: `{summary.get('loss_name')}`",
        f"- Memory tokens: `{summary.get('memory_token_count')}`",
        f"- Attention heads: `{summary.get('attention_heads')}`",
        f"- Attention blocks: `{summary.get('attention_blocks')}`",
        f"- Mean direct foreground IoU: `{summary['mean_metrics']['direct_foreground_iou']:.6f}`",
        f"- Mean Dice: `{summary['mean_metrics']['direct_foreground_dice']:.6f}`",
        f"- Mean auxiliary partition mIoU: `{summary['mean_metrics']['eval_miou']:.6f}`",
        f"- Mean auxiliary partition ARI: `{summary['mean_metrics']['eval_ari']:.6f}`",
    ]
    if summary.get("coarse_mean_metrics") is not None:
        coarse_mean = summary["coarse_mean_metrics"]
        lines.extend(
            [
                f"- Mean coarse direct foreground IoU: `{coarse_mean['coarse_direct_foreground_iou']}`",
                f"- Mean coarse Dice: `{coarse_mean['coarse_direct_foreground_dice']}`",
                f"- Mean final-minus-coarse direct foreground IoU delta: `{coarse_mean['final_minus_coarse_direct_foreground_iou']}`",
                f"- Mean final-minus-coarse Dice delta: `{coarse_mean['final_minus_coarse_direct_foreground_dice']}`",
                f"- Mean coarse partition mIoU: `{coarse_mean['coarse_direct_eval_miou']}`",
                f"- Mean coarse partition ARI: `{coarse_mean['coarse_direct_eval_ari']}`",
            ]
        )
    if summary.get("residual_diagnostics_mean") is not None:
        residual_mean = summary["residual_diagnostics_mean"]
        lines.extend(
            [
                f"- Residual scale source: `{residual_mean['residual_scale_source']}`",
                f"- Learned residual scale: `{residual_mean['learned_residual_scale']}`",
                f"- Effective residual scale: `{residual_mean['effective_residual_scale']}`",
                f"- Residual gate mode: `{residual_mean['residual_gate_mode']}`",
                f"- Residual gate threshold: `{residual_mean['residual_gate_threshold']}`",
                f"- Residual gate mean: `{residual_mean['residual_gate_mean']}`",
                f"- Attention mean: `{residual_mean.get('attention_mean')}`",
                f"- Attention fraction >= 0.5: `{residual_mean.get('attention_fraction_ge_half')}`",
                f"- Mean abs residual logits: `{residual_mean['mean_abs_residual_logits']}`",
                f"- Mean abs residual contribution: `{residual_mean['mean_abs_residual_contribution']}`",
                f"- Residual sign flip fraction: `{residual_mean['residual_sign_flip_fraction']}`",
            ]
        )
    if summary.get("train_set_mean_metrics") is not None:
        lines.append(f"- Mean train-set foreground IoU (auxiliary): `{summary['train_set_mean_metrics']['direct_foreground_iou']:.6f}`")
        lines.append(f"- Mean train-set Dice (auxiliary): `{summary['train_set_mean_metrics']['direct_foreground_dice']:.6f}`")
    lines.extend(
        [
            f"- Best checkpoint: `{summary.get('best_checkpoint_path')}`",
            f"- Checkpoint selection materialized as: `{summary.get('checkpoint_selection_materialized')}`",
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
            protocol=f"glas_frozen_mask_head:{variant}",
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
    config = {
        "command": getattr(args, "command", None),
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(str(value)) for value in sys.argv),
        "run_kind": run_kind,
        "dataset_id": GLAS_DATASET_ID,
        "dataset_root": getattr(args, "dataset_root", None),
        "train_selection_policy": resolve_train_selection_policy(args),
        "resolved_split_policy": GLAS_FROZEN_MASK_HEAD_CONTRACT_PROFILE["train_split_policy"],
        "split": getattr(args, "split", None),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "resolved_contract_profile": GLAS_FROZEN_MASK_HEAD_CONTRACT_PROFILE["profile_name"],
        "variant": getattr(args, "variant", None),
        "model_id": args.model_id,
        "backbone_family": resolve_mask_head_backbone_family(args),
        "feature_source": resolve_mask_head_feature_source_description(args),
        "device": args.device,
        "official_checkpoint_path": args.official_checkpoint_path,
        "hf_token_supplied": args.hf_token is not None,
        "train_limit": getattr(args, "train_limit", None),
        "eval_limit": getattr(args, "eval_limit", None),
        "limit": getattr(args, "limit", None),
        "selected_level_names": list(selected_level_names),
        "reference_raw_level_shapes": reference_raw_level_shapes,
        "reference_selected_level_shapes": reference_selected_level_shapes,
        "variant_summary": GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS[getattr(args, "variant", None)]["summary"],
        "model_family": (
            "foreground_probe"
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "serialized_local_routed_refinement_mask_head"
                if is_serialized_local_routed_refine_variant(str(getattr(args, "variant", None)))
                else (
                    "serialized_progressive_refinement_mask_head"
                    if is_serialized_refine_variant(str(getattr(args, "variant", None)))
                    else (
                "coarse_plus_multibank_null_refine_mask_head"
                if is_multibank_null_refine_variant(str(getattr(args, "variant", None)))
                else (
                    "coarse_plus_cross_attention_refine_mask_head"
                    if is_cross_attention_refine_variant(str(getattr(args, "variant", None)))
                    else (
                        "coarse_plus_attention_refine_mask_head"
                        if is_attention_refine_variant(str(getattr(args, "variant", None)))
                        else (
                            "coarse_plus_fine_residual_mask_head"
                            if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
                            else (
                                resolve_frozen_mask_head_head_family(str(getattr(args, "variant", None)))
                                if is_memory_head_variant(str(getattr(args, "variant", None)))
                                else "multiscale_mask_head"
                            )
                        )
                    )
                )
                    )
                )
            )
        ),
        "head_kind": (
            GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[getattr(args, "variant", None)]["head_kind"]
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else (
                "serialized_local_routed_refinement"
                if is_serialized_local_routed_refine_variant(str(getattr(args, "variant", None)))
                else (
                    "serialized_progressive_refinement"
                    if is_serialized_refine_variant(str(getattr(args, "variant", None)))
                    else (
                "coarse_plus_multibank_null_refine"
                if is_multibank_null_refine_variant(str(getattr(args, "variant", None)))
                else (
                    "coarse_plus_cross_attention_refine"
                    if is_cross_attention_refine_variant(str(getattr(args, "variant", None)))
                    else (
                        "coarse_plus_attention_refine"
                        if is_attention_refine_variant(str(getattr(args, "variant", None)))
                        else (
                            "coarse_plus_fine_residual"
                            if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
                            else (
                                "memory_attention"
                                if is_memory_attention_variant(str(getattr(args, "variant", None)))
                                else ("memory_control" if is_memory_control_variant(str(getattr(args, "variant", None))) else "multiscale_mask_head")
                            )
                        )
                    )
                )
                    )
                )
            )
        ),
        "projection_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["projection_dim"])
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_projection_dim(args)
        ),
        "decoder_dim": (
            None
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_decoder_dim(args)
        ),
        "hidden_dim": (
            int(GLAS_FROZEN_FEATURE_PROBE_SETTINGS["hidden_dim"])
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else None
        ),
        "group_norm_groups": (
            None
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else resolve_mask_head_group_norm_groups(args)
        ),
        "memory_token_count": (
            resolve_mask_head_memory_token_count(args)
            if is_memory_head_variant(str(getattr(args, "variant", None)))
            else None
        ),
        "attention_heads": (
            resolve_mask_head_attention_heads(args)
            if (
                is_memory_head_variant(str(getattr(args, "variant", None)))
                or is_cross_attention_refine_variant(str(getattr(args, "variant", None)))
                or is_multibank_null_refine_variant(str(getattr(args, "variant", None)))
            )
            else None
        ),
        "attention_blocks": (
            resolve_mask_head_attention_blocks(args)
            if is_memory_head_variant(str(getattr(args, "variant", None)))
            else None
        ),
        "coarse_level_name": coarse_level_name,
        "fine_level_name": fine_level_name,
        "residual_projection_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_projection_dim"])
            if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
            else None
        ),
        "residual_hidden_dim": (
            int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_hidden_dim"])
            if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
            else None
        ),
        "attention_hidden_dim": (
            int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"]))
            if (
                is_attention_refine_variant(str(getattr(args, "variant", None)))
                or is_multibank_null_refine_variant(str(getattr(args, "variant", None)))
            )
            else None
        ),
        "cross_attn_query_stride": (
            int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"]))
            if (
                is_cross_attention_refine_variant(str(getattr(args, "variant", None)))
                or is_multibank_null_refine_variant(str(getattr(args, "variant", None)))
            )
            else None
        ),
        "residual_scale_init": (
            float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS["residual_scale_init"])
            if is_coarse_plus_residual_variant(str(getattr(args, "variant", None)))
            else None
        ),
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
        "checkpoint_selection_materialized": (
            "best_eval"
            if str(getattr(args, "selection_checkpoint_mode", "final")) == "best_eval"
            and resolve_mask_head_eval_every_epochs(args) is not None
            else "final"
        ),
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
            GLAS_FROZEN_FEATURE_PROBE_SETTINGS
            if getattr(args, "variant", None) in GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS
            else build_mask_head_settings_payload(args)
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
                f"Command family: `{getattr(args, 'command', 'glas-frozen-mask-head')}`.",
                f"Run kind: `{run_kind}`.",
                f"Variant: `{variant}`.",
                    f"Variant summary: {variant_spec['summary']}",
                f"Dataset: `{GLAS_DATASET_ID}` from `{getattr(args, 'dataset_root', None)}`.",
                f"Model: `{args.model_id}` on device request `{args.device}`.",
                f"Resolved frozen backbone family: `{resolve_mask_head_backbone_family(args)}`.",
                f"Official checkpoint override for SAM feature extraction: `{args.official_checkpoint_path}`.",
                "SAM stays frozen for the entire run. Only the new mask head is optimized.",
            ),
        ),
        ExperimentTermsSection(
            title="Data And Split Separation",
            bullets=(
                f"Training split: `{getattr(args, 'train_split', None)}` with limit `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Evaluation split: `{getattr(args, 'eval_split', getattr(args, 'split', None))}` with limit `{getattr(args, 'eval_limit', getattr(args, 'limit', None))}`; loaded `{eval_sample_count}` sample(s).",
                "The existing repo-native GlaS loader exposes the official train/test split and binary gland-vs-background masks.",
                (
                    f"Train subset manifest: `{getattr(args, '_resolved_train_subset_manifest_output_path', getattr(args, 'train_subset_manifest', None))}` selecting `{getattr(args, '_resolved_train_subset_manifest', {}).get('shot_count')}` image(s) with subset seed `{getattr(args, '_resolved_train_subset_manifest', {}).get('subset_seed')}`."
                    if getattr(args, "_resolved_train_subset_manifest", None) is not None
                    else "No train subset manifest is applied. The full official GlaS train split is used."
                ),
                "No separate validation split is introduced here. Validation metrics are therefore recorded as `null` rather than inventing a new split.",
                "Foreground polarity contract: the gland mask is the positive class and corresponds to `texture_a_mask` in the GlaS adapter.",
                f"Train augmentation policy: `{describe_train_augmentation_policy(resolve_train_augmentation_policy(args))}`.",
                (
                    f"Explicit preprocessing resize: `{resolve_mask_head_resize_hw(args)[0]}x{resolve_mask_head_resize_hw(args)[1]}` with bilinear image resize and nearest-neighbor mask resize before SAM feature extraction, supervision, and evaluation."
                    if resolve_mask_head_resize_hw(args) is not None
                    else "No explicit resize is applied. Images and masks stay at native decoded GlaS resolution."
                ),
            ),
        ),
        ExperimentTermsSection(
            title="Frozen Feature Source",
            bullets=tuple(
                [
                    (
                        f"Frozen SAM feature source: `{resolve_mask_head_feature_source_description(args)}` from the native multiscale `backbone_fpn` pyramid."
                        if resolve_mask_head_backbone_family(args) == "sam3"
                        else "Frozen SAM feature source: `sam2_predictor_features(image_embed+high_res_feats)` mapped explicitly to repo level names `fpn_2` (coarsest), `fpn_1` (middle), `fpn_0` (finest)."
                    ),
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
                    f"Residual branch width: projection dim `{int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_projection_dim'])}`, hidden dim `{int(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_hidden_dim'])}`, initial residual scale `{float(FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS['residual_scale_init'])}`."
                    if is_coarse_plus_residual_variant(variant)
                    else "No explicit fine residual branch is used in this run."
                ),
                (
                    f"Residual settings: gate mode `{getattr(args, 'residual_gate_mode', 'none')}`, gate threshold `{getattr(args, 'residual_gate_threshold', RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)}`, coarse-loss weight `{getattr(args, 'coarse_loss_weight', 0.0)}`, residual-L1 weight `{getattr(args, 'residual_l1_weight', 0.0)}`."
                    if is_coarse_plus_residual_variant(variant)
                    else "Residual settings: not applicable because this is not a coarse-plus-residual variant."
                ),
                f"Loss: `{resolve_mask_head_loss_name(args)}`.",
                "Primary supervised metrics: direct gland foreground IoU and Dice.",
                "Auxiliary repo-native metrics: permutation-invariant GlaS `eval_miou` and `eval_ari` computed from the same foreground prediction and its complement.",
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
                "Hard-fail on missing GlaS roots, missing image/mask pairs, empty sample sets, or inconsistent SAM pyramid channels across samples.",
                "Hard-fail on missing requested SAM pyramid levels, invalid tensor ranks, or NaN/Inf feature/logit/loss values.",
                "Hard-fail on prediction/target shape mismatches instead of silently resizing targets beyond the explicit image-resolution decoder step.",
            ),
        ),
    ]
    return render_experiment_terms_markdown(
        title="GlaS Frozen SAM Mask Head Experiment Terms",
        summary_lines=(
            "This file describes the exact frozen-feature supervised GlaS run that produced this directory.",
            "It records the split separation, architecture, losses, metrics, outputs, and explicit failure rules for review.",
        ),
        sections=sections,
        related_paths=(
            render_relative_path(GLAS_FROZEN_MASK_HEAD_EXPERIMENT_CONTRACT_PATH),
            render_relative_path(Path("src/rwtd_sam3/models/sam3_frozen_multiscale_mask_head.py")),
            render_relative_path(Path("src/rwtd_sam3/eval/glas_frozen_feature_mask_head.py")),
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
    if is_coarse_plus_residual_variant(variant):
        if is_serialized_local_routed_refine_variant(variant):
            return (
                "Probe family: coarse-first serialized local residual refiner.",
                "Stage 0 predicts coarse logits from `fpn_2` using the existing tiny coarse head.",
                "Stage 1 predicts a full-image `fpn_1` residual but applies it only inside a deterministic top-20%-uncertainty route mask dilated with a 5x5 max-pool band.",
                "Stage 2 predicts a full-image `fpn_0` residual but applies it only inside a tighter top-10%-uncertainty route mask dilated with a 3x3 max-pool band.",
                "Outside the routed masks the logits are identity by construction, and both stage scales start at zero with staged warmup.",
            )
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
