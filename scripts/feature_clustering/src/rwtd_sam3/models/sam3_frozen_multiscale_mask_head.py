"""Tiny supervised dense probes on frozen SAM multiscale features.

This module implements a deliberately small FPN-style binary segmentation head:

- SAM features stay frozen and are provided as multiscale `[B,C,H,W]` tensors.
- Each selected scale is projected to 64 channels with a `1x1` convolution.
- All projected scales are resized to the finest selected grid and concatenated.
- Two `3x3 -> GroupNorm -> GELU` blocks decode the fused feature map.
- The decoder is upsampled to image resolution and mapped to one foreground logit.

The intent is a minimal dense-supervised readout baseline for GlaS rather than a
new prompt-generator or a heavy segmentation architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FROZEN_MASK_HEAD_SETTINGS: dict[str, float | int | str] = {
    "projection_dim": 64,
    "decoder_dim": 64,
    "group_norm_groups": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 20,
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "feature_source": "backbone_fpn",
}

FROZEN_MASK_HEAD_FINE_RESIDUAL_SETTINGS: dict[str, float | int | str] = {
    "residual_projection_dim": 32,
    "residual_hidden_dim": 32,
    "residual_scale_init": 0.0,
}

FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS: dict[str, float | int | str] = {
    "attention_hidden_dim": 32,
    "cross_attn_query_stride": 2,
    "joker_default_coarse_loss_weight": 0.5,
}

FROZEN_MASK_HEAD_MEMORY_ATTN_SETTINGS: dict[str, float | int | str] = {
    "memory_token_count": 16,
    "attention_heads": 4,
    "attention_blocks": 1,
    "memory_init_std": 0.02,
}

FROZEN_MASK_HEAD_LOSS_VARIANTS: tuple[str, ...] = (
    "bce_dice",
    "boundary_weighted_bce_dice",
)

RESIDUAL_HEAD_GATE_MODES: tuple[str, ...] = (
    "none",
    "hard_uncertainty",
    "soft_uncertainty",
)

RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD: float = 0.25

FROZEN_MASK_HEAD_BOUNDARY_LOSS_SETTINGS: dict[str, float] = {
    "boundary_weight": 2.0,
}


GLAS_FROZEN_FEATURE_PROBE_SETTINGS: dict[str, float | int | str | bool] = {
    "projection_dim": 32,
    "hidden_dim": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 20,
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "normalize_features": True,
    "foreground_threshold": 0.5,
}


GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "linear_fpn2": {
        "head_kind": "linear",
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "summary": "Single-scale linear probe using only the coarsest SAM level `fpn_2`.",
    },
    "linear_fpn1": {
        "head_kind": "linear",
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_1",
        "summary": "Single-scale linear probe using only the middle SAM level `fpn_1`.",
    },
    "linear_fpn0": {
        "head_kind": "linear",
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_0",
        "summary": "Single-scale linear probe using only the finest SAM level `fpn_0`.",
    },
    "tiny_nonlinear_fpn0": {
        "head_kind": "tiny_nonlinear",
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_0",
        "summary": "Single-scale tiny non-linear probe using only the finest SAM level `fpn_0`.",
    },
    "tiny_multiscale": {
        "head_kind": "tiny_nonlinear",
        "selected_level_policy": "all_scales",
        "summary": "Tiny multiscale probe that upsamples all discovered SAM levels to the finest grid, concatenates them, and predicts one gland logit map.",
    },
}


FROZEN_MASK_HEAD_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "all_scales": {
        "selected_level_policy": "all_scales",
        "summary": "Use every discovered main SAM pyramid level in the frozen multiscale mask head.",
    },
    "mid_plus_fine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_1", "fpn_0"),
        "summary": "Use only the middle and finest SAM levels (`fpn_1`, `fpn_0`) in the frozen multiscale mask head.",
    },
    "coarse_plus_next_finer": {
        "selected_level_policy": "coarse_plus_next_finer",
        "summary": "Use only the coarsest level and the next finer level in the frozen multiscale mask head.",
    },
    "fpn_2_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "summary": "Single-scale frozen mask head using only the coarsest SAM level `fpn_2`.",
    },
    "fpn_1_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_1",
        "summary": "Single-scale frozen mask head using only the middle SAM level `fpn_1`.",
    },
    "fpn_0_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_0",
        "summary": "Single-scale frozen mask head using only the finest SAM level `fpn_0`.",
    },
    "fpn_2_plus_fine_residual": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_0"),
        "summary": "Coarse-only frozen mask head on `fpn_2` plus a tiny `fpn_0` residual-refinement branch that predicts an additive foreground-logit correction.",
    },
    "fpn_2_plus_fpn_1_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1"),
        "summary": "Coarse-only frozen mask head on `fpn_2` plus a tiny `fpn_1` residual-refinement branch that predicts an additive foreground-logit correction.",
    },
    "fpn_2_plus_fpn_1_attn_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1"),
        "head_family": "coarse_plus_attention_refine_mask_head",
        "summary": "Coarse-only frozen mask head on `fpn_2` plus an uncertainty-guided `fpn_1` residual branch with a learned spatial attention gate.",
    },
    "fpn_2_plus_fpn_1_cross_attn_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1"),
        "head_family": "coarse_plus_cross_attention_refine_mask_head",
        "summary": "Coarse-only frozen mask head on `fpn_2` plus a coarse-guided `fpn_1` cross-attention refinement branch that retrieves fine evidence before decoding a gated residual correction.",
    },
    "fpn_2_plus_multibank_null_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1", "fpn_0"),
        "head_family": "coarse_plus_multibank_null_refine_mask_head",
        "summary": "Coarse-first frozen mask head on `fpn_2` plus a coarse-guided multi-bank refinement branch that can attend into `fpn_1`, `fpn_0`, or a learned null token before predicting an additive residual correction.",
    },
    "fpn_2_memory_attn": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "head_family": "memory_attention_mask_head",
        "summary": "Single-scale frozen mask head using `fpn_2` plus a tiny learned global memory cross-attention block before shallow dense decoding.",
    },
    "fpn_2_plus_fpn_1_memory_attn": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1"),
        "head_family": "memory_attention_mask_head",
        "summary": "Frozen mask head using `fpn_2` global-memory cross-attention with one projected `fpn_1` skip fused only after coarse texture commitment.",
    },
    "fpn_2_memory_control": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "head_family": "memory_control_mask_head",
        "summary": "Matched-capacity coarse-only control that keeps the learned memory bank and shallow decoder but replaces per-token cross-attention with one global memory-mixing pathway.",
    },
    "fpn_2_then_fpn_1_then_fpn_0_attn_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1", "fpn_0"),
        "head_family": "serialized_progressive_refinement_mask_head",
        "summary": "Coarse-first frozen mask head on `fpn_2` followed by a serialized chain of refinement stages using `fpn_1` and `fpn_0` features with progressive warmup.",
    },
    "fpn_2_then_fpn_1_then_fpn_0_local_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1", "fpn_0"),
        "head_family": "serialized_local_routed_refinement_mask_head",
        "summary": "Coarse-first frozen mask head on `fpn_2` followed by two routed local refinement stages using `fpn_1` and `fpn_0` uncertainty-band masks.",
    },
    "fpn_2_then_fpn_1_then_fpn_0_local_routed_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1", "fpn_0"),
        "head_family": "serialized_local_routed_refinement_mask_head",
        "summary": "Backward-compatible alias for `fpn_2_then_fpn_1_then_fpn_0_local_refine`.",
    },
}


GLAS_FROZEN_FEATURE_ALL_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    **FROZEN_MASK_HEAD_VARIANT_SPECS,
    **GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS,
}
FROZEN_MASK_HEAD_VARIANT_SPECS.update(GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS)


class FrozenMaskHeadRuntimeError(RuntimeError):
    """Raised when the frozen-feature mask-head path receives invalid inputs."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class FrozenMaskHeadOutput:
    """Forward-pass outputs from the tiny supervised mask head."""

    logits: Any
    finest_level_name: str
    finest_grid_size: tuple[int, int]
    projected_level_shapes: dict[str, tuple[int, int, int, int]]
    fused_feature_shape: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrozenResidualMaskHeadOutput:
    """Forward-pass outputs from the coarse-plus-fine residual mask head."""

    logits: Any
    coarse_logits: Any
    residual_logits: Any
    finest_level_name: str
    finest_grid_size: tuple[int, int]
    projected_level_shapes: dict[str, tuple[int, int, int, int]]
    fused_feature_shape: tuple[int, int, int, int]
    residual_scale: float
    attention_map: Any | None = None
    attended_feature_shape: tuple[int, int, int, int] | None = None
    residual_runtime_multiplier: float = 1.0
    attention_bank_means: dict[str, float] | None = None
    attention_real_mean: float | None = None
    attention_null_mean: float | None = None
    attention_fpn1_mean: float | None = None
    attention_fpn0_mean: float | None = None
    residual_scale_fpn1: float | None = None
    residual_scale_fpn0: float | None = None
    alpha_fpn1: float | None = None
    alpha_fpn0: float | None = None
    routed_mask_fpn1: Any | None = None
    routed_mask_fpn0: Any | None = None
    stage1_logits: Any | None = None
    stage2_logits: Any | None = None
    stage1_residual_logits: Any | None = None
    stage2_residual_logits: Any | None = None
    route1_mask: Any | None = None
    route0_mask: Any | None = None


@dataclass(frozen=True)
class ResidualHeadCombinationResult:
    """Runtime-only residual-head combination state used for diagnostics and eval overrides."""

    coarse_logits: Any
    residual_logits: Any
    final_logits: Any
    residual_contribution_logits: Any
    gate_map: Any
    attention_map: Any | None
    learned_residual_scale: float
    effective_residual_scale: float
    residual_scale_source: str
    residual_alpha_override: float | None
    residual_gate_mode: str
    residual_gate_threshold: float | None
    residual_runtime_multiplier: float
    attention_bank_means: dict[str, float] | None
    attention_real_mean: float | None
    attention_null_mean: float | None
    attention_fpn1_mean: float | None
    attention_fpn0_mean: float | None
    residual_scale_fpn1: float | None
    residual_scale_fpn0: float | None
    alpha_fpn1: float | None
    alpha_fpn0: float | None


@dataclass(frozen=True)
class ResidualHeadTrainingLossResult:
    """Scalar training-loss bundle for residual-head supervision and regularization."""

    total_loss: Any
    final_loss_result: MaskHeadLossResult
    coarse_loss_result: MaskHeadLossResult | None
    residual_l1_loss: Any | None
    attention_sparsity_loss: Any | None
    coarse_loss_weight: float
    residual_l1_weight: float
    attention_sparsity_weight: float


@dataclass(frozen=True)
class FrozenForegroundProbeOutput:
    """Forward-pass outputs from one frozen-feature foreground probe."""

    logits: Any
    finest_level_name: str
    finest_grid_size: tuple[int, int]
    level_shapes: dict[str, tuple[int, int, int, int]]
    fused_feature_shape: tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskHeadLossResult:
    """Scalar loss bundle for BCE plus Dice supervision."""

    loss: Any
    bce_loss: Any
    dice_loss: Any
    mean_probability: float
    predicted_positive_fraction: float
    target_positive_fraction: float


def is_coarse_plus_residual_variant(variant: str) -> bool:
    return str(variant) in {
        "fpn_2_plus_fine_residual",
        "fpn_2_plus_fpn_1_refine",
        "fpn_2_plus_fpn_1_attn_refine",
        "fpn_2_plus_fpn_1_cross_attn_refine",
        "fpn_2_plus_multibank_null_refine",
        "fpn_2_then_fpn_1_then_fpn_0_attn_refine",
        "fpn_2_then_fpn_1_then_fpn_0_local_routed_refine",
    }


def is_serialized_refine_variant(variant: str) -> bool:
    return str(variant) == "fpn_2_then_fpn_1_then_fpn_0_attn_refine"


def is_serialized_local_routed_refine_variant(variant: str) -> bool:
    return str(variant) in {
        "fpn_2_then_fpn_1_then_fpn_0_local_refine",
        "fpn_2_then_fpn_1_then_fpn_0_local_routed_refine",
    }


def is_attention_refine_variant(variant: str) -> bool:
    return str(variant) in {
        "fpn_2_plus_fpn_1_attn_refine",
        "fpn_2_plus_fpn_1_cross_attn_refine",
    }


def is_cross_attention_refine_variant(variant: str) -> bool:
    return str(variant) == "fpn_2_plus_fpn_1_cross_attn_refine"


def is_multibank_null_refine_variant(variant: str) -> bool:
    return str(variant) == "fpn_2_plus_multibank_null_refine"


def resolve_frozen_mask_head_head_family(variant: str) -> str:
    try:
        variant_spec = FROZEN_MASK_HEAD_VARIANT_SPECS[str(variant)]
    except KeyError as exc:
        expected = ", ".join(sorted(FROZEN_MASK_HEAD_VARIANT_SPECS))
        raise FrozenMaskHeadRuntimeError(
            f"Unknown frozen mask-head variant '{variant}'. Expected one of: {expected}."
        ) from exc
    return str(variant_spec.get("head_family", "multiscale_mask_head"))


def is_memory_attention_variant(variant: str) -> bool:
    return resolve_frozen_mask_head_head_family(str(variant)) == "memory_attention_mask_head"


def is_memory_control_variant(variant: str) -> bool:
    return resolve_frozen_mask_head_head_family(str(variant)) == "memory_control_mask_head"


def is_memory_head_variant(variant: str) -> bool:
    return is_memory_attention_variant(str(variant)) or is_memory_control_variant(str(variant))


def resolve_coarse_plus_residual_levels(variant: str) -> tuple[str, str]:
    variant_name = str(variant)
    if variant_name == "fpn_2_plus_fine_residual":
        return ("fpn_2", "fpn_0")
    if variant_name == "fpn_2_plus_fpn_1_refine":
        return ("fpn_2", "fpn_1")
    if variant_name == "fpn_2_plus_fpn_1_attn_refine":
        return ("fpn_2", "fpn_1")
    if variant_name == "fpn_2_plus_fpn_1_cross_attn_refine":
        return ("fpn_2", "fpn_1")
    if variant_name == "fpn_2_plus_multibank_null_refine":
        return ("fpn_2", "fpn_0")
    if variant_name == "fpn_2_then_fpn_1_then_fpn_0_attn_refine":
        return ("fpn_2", "fpn_0")
    if variant_name == "fpn_2_then_fpn_1_then_fpn_0_local_refine":
        return ("fpn_2", "fpn_0")
    if variant_name == "fpn_2_then_fpn_1_then_fpn_0_local_routed_refine":
        return ("fpn_2", "fpn_0")
    raise FrozenMaskHeadRuntimeError(
        f"Variant '{variant_name}' is not a supported coarse-plus-residual frozen mask head.",
        diagnostics={
            "supported_variants": [
                "fpn_2_plus_fine_residual",
                "fpn_2_plus_fpn_1_refine",
                "fpn_2_plus_fpn_1_attn_refine",
                "fpn_2_plus_fpn_1_cross_attn_refine",
                "fpn_2_plus_multibank_null_refine",
                "fpn_2_then_fpn_1_then_fpn_0_attn_refine",
                "fpn_2_then_fpn_1_then_fpn_0_local_refine",
                "fpn_2_then_fpn_1_then_fpn_0_local_routed_refine",
            ]
        },
    )


class FrozenSamMultiscaleMaskHead:
    """Small FPN-style decoder over selected frozen SAM pyramid levels."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        num_classes: int = 1,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not level_input_dims:
                    raise FrozenMaskHeadRuntimeError("FrozenSamMultiscaleMaskHead requires at least one input level.")
                self.level_names = tuple(level_input_dims.keys())
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.num_classes = int(num_classes)
                if self.projection_dim < 1 or self.decoder_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "projection_dim and decoder_dim must both be positive.",
                        diagnostics={
                            "projection_dim": self.projection_dim,
                            "decoder_dim": self.decoder_dim,
                        },
                    )
                if self.decoder_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "decoder_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "decoder_dim": self.decoder_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                self.projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(
                            int(input_dim),
                            self.projection_dim,
                            kernel_size=1,
                            bias=True,
                        )
                        for level_name, input_dim in self.level_input_dims.items()
                    }
                )
                fused_dim = len(self.level_names) * self.projection_dim
                self.decoder = nn.Sequential(
                    nn.Conv2d(fused_dim, self.decoder_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.decoder_dim),
                    nn.GELU(),
                    nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.decoder_dim),
                    nn.GELU(),
                )
                self.classifier = nn.Conv2d(self.decoder_dim, self.num_classes, kernel_size=1, bias=True)

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamMultiscaleMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                if image_height < 1 or image_width < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size values must be positive.",
                        diagnostics={"image_size": (image_height, image_width)},
                    )

                projected_levels: dict[str, Any] = {}
                reference_fine_name: str | None = None
                reference_fine_hw: tuple[int, int] | None = None
                projected_shapes: dict[str, tuple[int, int, int, int]] = {}

                for level_name in self.level_names:
                    tensor = feature_levels[level_name]
                    if tensor.ndim != 4:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamMultiscaleMaskHead expected [B,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                        )
                    if int(tensor.shape[0]) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            "FrozenSamMultiscaleMaskHead expects batch size 1 during local training/eval.",
                            diagnostics={"level_name": level_name, "shape": tuple(int(value) for value in tensor.shape)},
                        )
                    current_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                    if reference_fine_hw is None or (current_hw[0] * current_hw[1], level_name) > (
                        reference_fine_hw[0] * reference_fine_hw[1],
                        reference_fine_name or "",
                    ):
                        reference_fine_name = level_name
                        reference_fine_hw = current_hw
                    projected = self.projections[level_name](tensor)
                    if not torch.isfinite(projected).all():
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamMultiscaleMaskHead produced NaN/Inf values after projecting {level_name}."
                        )
                    projected_levels[level_name] = projected
                    projected_shapes[level_name] = tuple(int(value) for value in projected.shape)

                if reference_fine_hw is None or reference_fine_name is None:
                    raise FrozenMaskHeadRuntimeError("FrozenSamMultiscaleMaskHead could not resolve a finest feature grid.")

                resized_levels: list[Any] = []
                for level_name in self.level_names:
                    projected = projected_levels[level_name]
                    if tuple(int(value) for value in projected.shape[-2:]) != reference_fine_hw:
                        projected = F.interpolate(
                            projected,
                            size=reference_fine_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                    resized_levels.append(projected)

                fused = torch.cat(resized_levels, dim=1)
                decoded = self.decoder(fused)
                logits = self.classifier(
                    F.interpolate(
                        decoded,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                if tuple(int(value) for value in logits.shape) != (1, self.num_classes, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamMultiscaleMaskHead produced an unexpected logit shape.",
                        diagnostics={
                            "expected_shape": (1, self.num_classes, image_height, image_width),
                            "received_shape": tuple(int(value) for value in logits.shape),
                        },
                    )
                if not torch.isfinite(logits).all():
                    raise FrozenMaskHeadRuntimeError("FrozenSamMultiscaleMaskHead produced NaN/Inf logits.")

                return FrozenMaskHeadOutput(
                    logits=logits,
                    finest_level_name=reference_fine_name,
                    finest_grid_size=reference_fine_hw,
                    projected_level_shapes=projected_shapes,
                    fused_feature_shape=tuple(int(value) for value in fused.shape),
                )

        self.module = _Module()


class FrozenSamCoarsePlusFineResidualMaskHead:
    """Coarse frozen mask head with a tiny fine-scale residual correction branch."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        fine_level_name: str = "fpn_0",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        residual_scale_init: float = 0.0,
        num_classes: int = 1,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if coarse_level_name not in level_input_dims or fine_level_name not in level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusFineResidualMaskHead requires both coarse and fine input levels.",
                        diagnostics={
                            "coarse_level_name": coarse_level_name,
                            "fine_level_name": fine_level_name,
                            "available_level_names": tuple(level_input_dims.keys()),
                        },
                    )
                self.coarse_level_name = str(coarse_level_name)
                self.fine_level_name = str(fine_level_name)
                self.level_names = (self.coarse_level_name, self.fine_level_name)
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.residual_scale_init = float(residual_scale_init)
                self.num_classes = int(num_classes)
                if self.projection_dim < 1 or self.decoder_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "projection_dim and decoder_dim must both be positive.",
                        diagnostics={
                            "projection_dim": self.projection_dim,
                            "decoder_dim": self.decoder_dim,
                        },
                    )
                if self.residual_projection_dim < 1 or self.residual_hidden_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_projection_dim and residual_hidden_dim must both be positive.",
                        diagnostics={
                            "residual_projection_dim": self.residual_projection_dim,
                            "residual_hidden_dim": self.residual_hidden_dim,
                        },
                    )
                if self.decoder_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "decoder_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "decoder_dim": self.decoder_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.residual_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "residual_hidden_dim": self.residual_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )

                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: int(level_input_dims[self.coarse_level_name])},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=self.num_classes,
                ).module
                self.fine_projection = nn.Conv2d(
                    int(level_input_dims[self.fine_level_name]),
                    self.residual_projection_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(
                        self.residual_projection_dim,
                        self.residual_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(
                        self.residual_hidden_dim,
                        self.residual_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.residual_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1, bias=True)
                self.residual_scale = nn.Parameter(
                    torch.tensor(float(self.residual_scale_init), dtype=torch.float32)
                )

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenResidualMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusFineResidualMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                if image_height < 1 or image_width < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size values must be positive.",
                        diagnostics={"image_size": (image_height, image_width)},
                    )

                coarse_output: FrozenMaskHeadOutput = self.coarse_head(
                    {self.coarse_level_name: feature_levels[self.coarse_level_name]},
                    image_size=image_size,
                )
                fine_tensor = feature_levels[self.fine_level_name]
                if fine_tensor.ndim != 4 or int(fine_tensor.shape[0]) != 1:
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusFineResidualMaskHead expected [1,C,H,W] for {self.fine_level_name}, got {tuple(fine_tensor.shape)}."
                    )
                fine_hw = (int(fine_tensor.shape[-2]), int(fine_tensor.shape[-1]))
                projected = self.fine_projection(fine_tensor)
                if not torch.isfinite(projected).all():
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusFineResidualMaskHead produced NaN/Inf values after projecting {self.fine_level_name}."
                    )
                residual_features = self.residual_decoder(projected)
                residual_logits = self.residual_classifier(
                    F.interpolate(
                        residual_features,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                final_logits = coarse_output.logits + self.residual_scale.view(1, 1, 1, 1) * residual_logits
                if tuple(int(value) for value in final_logits.shape) != (1, self.num_classes, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusFineResidualMaskHead produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in final_logits.shape),
                        },
                    )
                if not torch.isfinite(final_logits).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusFineResidualMaskHead produced NaN/Inf logits."
                    )
                return FrozenResidualMaskHeadOutput(
                    logits=final_logits,
                    coarse_logits=coarse_output.logits,
                    residual_logits=residual_logits,
                    finest_level_name=self.fine_level_name,
                    finest_grid_size=fine_hw,
                    projected_level_shapes={
                        f"{self.coarse_level_name}::coarse": coarse_output.projected_level_shapes[self.coarse_level_name],
                        f"{self.fine_level_name}::residual": tuple(int(value) for value in projected.shape),
                    },
                    fused_feature_shape=tuple(int(value) for value in residual_features.shape),
                    residual_scale=float(self.residual_scale.detach().cpu().item()),
                )

        self.module = _Module()


class FrozenSamCoarsePlusAttentionRefineMaskHead:
    """Coarse-plus-fine residual head with learned spatial attention on residual corrections."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        fine_level_name: str = "fpn_1",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        attention_hidden_dim: int = 32,
        residual_scale_init: float = 0.0,
        num_classes: int = 1,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_names = tuple(level_input_dims.keys())
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                if not self.level_names:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead requires at least one feature level."
                    )
                self.coarse_level_name = str(coarse_level_name)
                self.fine_level_name = str(fine_level_name)
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.attention_hidden_dim = int(attention_hidden_dim)
                self.residual_scale_init = float(residual_scale_init)
                self.num_classes = int(num_classes)
                if self.coarse_level_name not in self.level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead missing coarse level.",
                        diagnostics={
                            "coarse_level_name": self.coarse_level_name,
                            "available_level_names": self.level_names,
                        },
                    )
                if self.fine_level_name not in self.level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead missing fine level.",
                        diagnostics={
                            "fine_level_name": self.fine_level_name,
                            "available_level_names": self.level_names,
                        },
                    )
                if self.residual_projection_dim < 1 or self.residual_hidden_dim < 1 or self.attention_hidden_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_projection_dim, residual_hidden_dim, and attention_hidden_dim must be positive.",
                        diagnostics={
                            "residual_projection_dim": self.residual_projection_dim,
                            "residual_hidden_dim": self.residual_hidden_dim,
                            "attention_hidden_dim": self.attention_hidden_dim,
                        },
                    )
                if self.decoder_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "decoder_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "decoder_dim": self.decoder_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.residual_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "residual_hidden_dim": self.residual_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.attention_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )

                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: int(level_input_dims[self.coarse_level_name])},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=self.num_classes,
                ).module
                self.fine_projection = nn.Conv2d(
                    int(level_input_dims[self.fine_level_name]),
                    self.residual_projection_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(
                        self.residual_projection_dim,
                        self.residual_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(
                        self.residual_hidden_dim,
                        self.residual_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.residual_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1, bias=True)

                self.attention_decoder = nn.Sequential(
                    nn.Conv2d(
                        self.residual_projection_dim + (3 * self.num_classes),
                        self.attention_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(
                        self.attention_hidden_dim,
                        self.attention_hidden_dim,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                    nn.GELU(),
                )
                self.attention_classifier = nn.Conv2d(self.attention_hidden_dim, 1, kernel_size=1, bias=True)
                self.residual_scale = nn.Parameter(
                    torch.tensor(float(self.residual_scale_init), dtype=torch.float32)
                )

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenResidualMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                if image_height < 1 or image_width < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size values must be positive.",
                        diagnostics={"image_size": (image_height, image_width)},
                    )

                coarse_output: FrozenMaskHeadOutput = self.coarse_head(
                    {self.coarse_level_name: feature_levels[self.coarse_level_name]},
                    image_size=image_size,
                )
                fine_tensor = feature_levels[self.fine_level_name]
                if fine_tensor.ndim != 4 or int(fine_tensor.shape[0]) != 1:
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusAttentionRefineMaskHead expected [1,C,H,W] for {self.fine_level_name}, got {tuple(fine_tensor.shape)}."
                    )
                fine_hw = (int(fine_tensor.shape[-2]), int(fine_tensor.shape[-1]))
                projected = self.fine_projection(fine_tensor)
                if not torch.isfinite(projected).all():
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusAttentionRefineMaskHead produced NaN/Inf values after projecting {self.fine_level_name}."
                    )
                residual_features = self.residual_decoder(projected)
                residual_logits = self.residual_classifier(
                    F.interpolate(
                        residual_features,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )

                coarse_logits_up = F.interpolate(
                    coarse_output.logits,
                    size=fine_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_probabilities = torch.sigmoid(coarse_logits_up)
                coarse_uncertainty = 4.0 * coarse_probabilities * (1.0 - coarse_probabilities)
                attention_input = torch.cat(
                    [projected, coarse_logits_up, coarse_probabilities, coarse_uncertainty],
                    dim=1,
                )
                attention_features = self.attention_decoder(attention_input)
                attention_map_fine = torch.sigmoid(self.attention_classifier(attention_features))
                attention_map = F.interpolate(
                    attention_map_fine,
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                )
                gated_residual_logits = attention_map * residual_logits
                final_logits = coarse_output.logits + self.residual_scale.view(1, 1, 1, 1) * gated_residual_logits
                if tuple(int(value) for value in final_logits.shape) != (1, self.num_classes, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in final_logits.shape),
                        },
                    )
                if not torch.isfinite(final_logits).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusAttentionRefineMaskHead produced NaN/Inf logits."
                    )
                return FrozenResidualMaskHeadOutput(
                    logits=final_logits,
                    coarse_logits=coarse_output.logits,
                    residual_logits=residual_logits,
                    finest_level_name=self.fine_level_name,
                    finest_grid_size=fine_hw,
                    projected_level_shapes={
                        f"{self.coarse_level_name}::coarse": coarse_output.projected_level_shapes[self.coarse_level_name],
                        f"{self.fine_level_name}::residual": tuple(int(value) for value in projected.shape),
                        f"{self.fine_level_name}::attention": tuple(int(value) for value in attention_features.shape),
                    },
                    fused_feature_shape=tuple(int(value) for value in residual_features.shape),
                    residual_scale=float(self.residual_scale.detach().cpu().item()),
                    attention_map=attention_map,
                )

        self.module = _Module()


class FrozenSamCoarsePlusCrossAttentionRefineMaskHead:
    """Coarse-plus-fine residual head that retrieves fine evidence with coarse-guided cross-attention."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        fine_level_name: str = "fpn_1",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        attention_hidden_dim: int = 32,
        attention_heads: int = 4,
        cross_attn_query_stride: int = 2,
        residual_scale_init: float = 0.0,
        num_classes: int = 1,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_names = tuple(level_input_dims.keys())
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                if not self.level_names:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead requires at least one feature level."
                    )
                self.coarse_level_name = str(coarse_level_name)
                self.fine_level_name = str(fine_level_name)
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.attention_hidden_dim = int(attention_hidden_dim)
                self.attention_heads = int(attention_heads)
                self.cross_attn_query_stride = int(cross_attn_query_stride)
                self.residual_scale_init = float(residual_scale_init)
                self.num_classes = int(num_classes)
                if self.coarse_level_name not in self.level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead missing coarse level.",
                        diagnostics={
                            "coarse_level_name": self.coarse_level_name,
                            "available_level_names": self.level_names,
                        },
                    )
                if self.fine_level_name not in self.level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead missing fine level.",
                        diagnostics={
                            "fine_level_name": self.fine_level_name,
                            "available_level_names": self.level_names,
                        },
                    )
                if self.residual_projection_dim < 1 or self.residual_hidden_dim < 1 or self.attention_hidden_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_projection_dim, residual_hidden_dim, and attention_hidden_dim must be positive.",
                        diagnostics={
                            "residual_projection_dim": self.residual_projection_dim,
                            "residual_hidden_dim": self.residual_hidden_dim,
                            "attention_hidden_dim": self.attention_hidden_dim,
                        },
                    )
                if self.attention_heads < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_heads must be positive for cross-attention refine heads.",
                        diagnostics={"attention_heads": self.attention_heads},
                    )
                if self.cross_attn_query_stride < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "cross_attn_query_stride must be positive.",
                        diagnostics={"cross_attn_query_stride": self.cross_attn_query_stride},
                    )
                if self.decoder_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "decoder_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "decoder_dim": self.decoder_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.residual_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "residual_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "residual_hidden_dim": self.residual_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.attention_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.attention_hidden_dim % self.attention_heads != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim must be divisible by attention_heads for cross-attention refine heads.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "attention_heads": self.attention_heads,
                        },
                    )

                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: int(level_input_dims[self.coarse_level_name])},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=self.num_classes,
                ).module
                self.fine_projection = nn.Conv2d(
                    int(level_input_dims[self.fine_level_name]),
                    self.residual_projection_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.query_seed_projection = nn.Conv2d(
                    self.residual_projection_dim + (3 * self.num_classes),
                    self.attention_hidden_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.key_projection = nn.Conv2d(
                    self.residual_projection_dim,
                    self.attention_hidden_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.value_projection = nn.Conv2d(
                    self.residual_projection_dim,
                    self.attention_hidden_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=self.attention_hidden_dim,
                    num_heads=self.attention_heads,
                    batch_first=True,
                )
                decoder_input_dim = self.attention_hidden_dim + self.residual_projection_dim + (3 * self.num_classes)
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(decoder_input_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.residual_hidden_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.residual_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1, bias=True)
                self.attention_decoder = nn.Sequential(
                    nn.Conv2d(decoder_input_dim, self.attention_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.attention_hidden_dim, self.attention_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                    nn.GELU(),
                )
                self.attention_classifier = nn.Conv2d(self.attention_hidden_dim, 1, kernel_size=1, bias=True)
                self.residual_scale = nn.Parameter(torch.tensor(float(self.residual_scale_init), dtype=torch.float32))

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenResidualMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                if image_height < 1 or image_width < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size values must be positive.",
                        diagnostics={"image_size": (image_height, image_width)},
                    )

                coarse_output: FrozenMaskHeadOutput = self.coarse_head(
                    {self.coarse_level_name: feature_levels[self.coarse_level_name]},
                    image_size=image_size,
                )
                fine_tensor = feature_levels[self.fine_level_name]
                if fine_tensor.ndim != 4 or int(fine_tensor.shape[0]) != 1:
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusCrossAttentionRefineMaskHead expected [1,C,H,W] for {self.fine_level_name}, got {tuple(fine_tensor.shape)}."
                    )
                fine_hw = (int(fine_tensor.shape[-2]), int(fine_tensor.shape[-1]))
                projected = self.fine_projection(fine_tensor)
                if not torch.isfinite(projected).all():
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamCoarsePlusCrossAttentionRefineMaskHead produced NaN/Inf values after projecting {self.fine_level_name}."
                    )
                coarse_logits_up = F.interpolate(
                    coarse_output.logits,
                    size=fine_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_probabilities = torch.sigmoid(coarse_logits_up)
                coarse_uncertainty = 4.0 * coarse_probabilities * (1.0 - coarse_probabilities)
                coarse_guidance = torch.cat(
                    [projected, coarse_logits_up, coarse_probabilities, coarse_uncertainty],
                    dim=1,
                )
                query_seed = self.query_seed_projection(coarse_guidance)
                if self.cross_attn_query_stride > 1:
                    query_seed = F.avg_pool2d(query_seed, kernel_size=self.cross_attn_query_stride, stride=self.cross_attn_query_stride)
                query_hw = (int(query_seed.shape[-2]), int(query_seed.shape[-1]))
                query_tokens = query_seed.flatten(2).transpose(1, 2).contiguous()
                key_tokens = self.key_projection(projected).flatten(2).transpose(1, 2).contiguous()
                value_tokens = self.value_projection(projected).flatten(2).transpose(1, 2).contiguous()
                attended_tokens, _ = self.cross_attention(query_tokens, key_tokens, value_tokens, need_weights=False)
                if not torch.isfinite(attended_tokens).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead produced NaN/Inf attended tokens."
                    )
                attended_features = attended_tokens.transpose(1, 2).reshape(
                    int(projected.shape[0]),
                    self.attention_hidden_dim,
                    query_hw[0],
                    query_hw[1],
                )
                attended_features = F.interpolate(
                    attended_features,
                    size=fine_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                decode_input = torch.cat(
                    [attended_features, projected, coarse_logits_up, coarse_probabilities, coarse_uncertainty],
                    dim=1,
                )
                residual_features = self.residual_decoder(decode_input)
                residual_logits = self.residual_classifier(
                    F.interpolate(
                        residual_features,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                attention_features = self.attention_decoder(decode_input)
                attention_map = torch.sigmoid(
                    self.attention_classifier(
                        F.interpolate(
                            attention_features,
                            size=(image_height, image_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                )
                gated_residual_logits = attention_map * residual_logits
                final_logits = coarse_output.logits + self.residual_scale.view(1, 1, 1, 1) * gated_residual_logits
                if tuple(int(value) for value in final_logits.shape) != (1, self.num_classes, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in final_logits.shape),
                        },
                    )
                if not torch.isfinite(final_logits).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusCrossAttentionRefineMaskHead produced NaN/Inf logits."
                    )
                return FrozenResidualMaskHeadOutput(
                    logits=final_logits,
                    coarse_logits=coarse_output.logits,
                    residual_logits=residual_logits,
                    finest_level_name=self.fine_level_name,
                    finest_grid_size=fine_hw,
                    projected_level_shapes={
                        f"{self.coarse_level_name}::coarse": coarse_output.projected_level_shapes[self.coarse_level_name],
                        f"{self.fine_level_name}::residual": tuple(int(value) for value in projected.shape),
                        f"{self.fine_level_name}::cross_attended": tuple(int(value) for value in attended_features.shape),
                    },
                    fused_feature_shape=tuple(int(value) for value in residual_features.shape),
                    residual_scale=float(self.residual_scale.detach().cpu().item()),
                    attention_map=attention_map,
                    attended_feature_shape=tuple(int(value) for value in attended_features.shape),
                )

        self.module = _Module()


class FrozenSamMemoryAttentionMaskHead:
    """Tiny coarse-first head with learned global memory tokens.

    The head keeps the feature contract intentionally small:

    - one coarse selected level is always the attention source
    - at most one finer skip is fused after coarse texture commitment
    - learned memory tokens are shared across the requested attention blocks
    - the head stays single-pass and never routes back into SAM
    """

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        memory_token_count: int = 16,
        attention_heads: int = 4,
        attention_blocks: int = 1,
        memory_init_std: float = 0.02,
        mixing_kind: str = "cross_attention",
    ) -> None:
        import torch
        import torch.nn as nn

        class _CrossAttentionBlock(nn.Module):
            def __init__(self, *, embed_dim: int, num_heads: int) -> None:
                super().__init__()
                self.query_norm = nn.LayerNorm(embed_dim)
                self.memory_norm = nn.LayerNorm(embed_dim)
                self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

            def forward(self, tokens, memory):
                attended, _ = self.attn(
                    self.query_norm(tokens),
                    self.memory_norm(memory),
                    self.memory_norm(memory),
                    need_weights=False,
                )
                return tokens + attended

        class _GlobalMemoryControlBlock(nn.Module):
            def __init__(self, *, embed_dim: int) -> None:
                super().__init__()
                self.token_norm = nn.LayerNorm(embed_dim)
                self.memory_norm = nn.LayerNorm(embed_dim)
                self.query_proj = nn.Linear(embed_dim, embed_dim, bias=True)
                self.key_proj = nn.Linear(embed_dim, embed_dim, bias=True)
                self.value_proj = nn.Linear(embed_dim, embed_dim, bias=True)
                self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)

            def forward(self, tokens, memory):
                norm_tokens = self.token_norm(tokens)
                norm_memory = self.memory_norm(memory)
                pooled_context = norm_tokens.mean(dim=1)
                query = self.query_proj(pooled_context)
                keys = self.key_proj(norm_memory)
                values = self.value_proj(norm_memory)
                scores = torch.einsum("bd,bmd->bm", query, keys) / math.sqrt(float(query.shape[-1]))
                weights = torch.softmax(scores, dim=-1)
                mixed = torch.einsum("bm,bmd->bd", weights, values)
                broadcast = self.output_proj(mixed).unsqueeze(1)
                return tokens + broadcast

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not level_input_dims:
                    raise FrozenMaskHeadRuntimeError("FrozenSamMemoryAttentionMaskHead requires at least one input level.")
                self.level_names = tuple(level_input_dims.keys())
                if len(self.level_names) > 2:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamMemoryAttentionMaskHead supports at most one coarse level plus one optional finer skip.",
                        diagnostics={"level_names": self.level_names},
                    )
                self.primary_level_name = str(self.level_names[0])
                self.skip_level_names = tuple(str(name) for name in self.level_names[1:])
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.memory_token_count = int(memory_token_count)
                self.attention_heads = int(attention_heads)
                self.attention_blocks = int(attention_blocks)
                self.memory_init_std = float(memory_init_std)
                self.mixing_kind = str(mixing_kind)
                if self.projection_dim < 1 or self.decoder_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "projection_dim and decoder_dim must both be positive.",
                        diagnostics={
                            "projection_dim": self.projection_dim,
                            "decoder_dim": self.decoder_dim,
                        },
                    )
                if self.memory_token_count < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "memory_token_count must be positive.",
                        diagnostics={"memory_token_count": self.memory_token_count},
                    )
                if self.attention_heads < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_heads must be positive.",
                        diagnostics={"attention_heads": self.attention_heads},
                    )
                if self.attention_blocks < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_blocks must be positive.",
                        diagnostics={"attention_blocks": self.attention_blocks},
                    )
                if self.projection_dim % self.attention_heads != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "projection_dim must be divisible by attention_heads for memory-attention heads.",
                        diagnostics={
                            "projection_dim": self.projection_dim,
                            "attention_heads": self.attention_heads,
                        },
                    )
                if self.decoder_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "decoder_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "decoder_dim": self.decoder_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.mixing_kind not in {"cross_attention", "global_memory_control"}:
                    raise FrozenMaskHeadRuntimeError(
                        f"Unsupported memory-head mixing_kind '{self.mixing_kind}'.",
                        diagnostics={"supported_mixing_kinds": ["cross_attention", "global_memory_control"]},
                    )

                self.primary_projection = nn.Conv2d(
                    int(level_input_dims[self.primary_level_name]),
                    self.projection_dim,
                    kernel_size=1,
                    bias=True,
                )
                self.skip_projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(int(level_input_dims[level_name]), self.projection_dim, kernel_size=1, bias=True)
                        for level_name in self.skip_level_names
                    }
                )
                self.memory_tokens = nn.Parameter(
                    torch.empty((self.memory_token_count, self.projection_dim), dtype=torch.float32)
                )
                nn.init.normal_(self.memory_tokens, mean=0.0, std=self.memory_init_std)
                block_factory = (
                    (lambda: _CrossAttentionBlock(embed_dim=self.projection_dim, num_heads=self.attention_heads))
                    if self.mixing_kind == "cross_attention"
                    else (lambda: _GlobalMemoryControlBlock(embed_dim=self.projection_dim))
                )
                self.blocks = nn.ModuleList(block_factory() for _ in range(self.attention_blocks))
                decoder_input_dim = self.projection_dim * (1 + len(self.skip_level_names))
                self.decoder = nn.Sequential(
                    nn.Conv2d(decoder_input_dim, self.decoder_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.decoder_dim),
                    nn.GELU(),
                    nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.decoder_dim),
                    nn.GELU(),
                )
                self.classifier = nn.Conv2d(self.decoder_dim, 1, kernel_size=1, bias=True)

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamMemoryAttentionMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                primary_tensor = feature_levels[self.primary_level_name]
                if primary_tensor.ndim != 4 or int(primary_tensor.shape[0]) != 1:
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamMemoryAttentionMaskHead expected [1,C,H,W] for {self.primary_level_name}, got {tuple(primary_tensor.shape)}."
                    )
                primary_projected = self.primary_projection(primary_tensor)
                if not torch.isfinite(primary_projected).all():
                    raise FrozenMaskHeadRuntimeError(
                        f"FrozenSamMemoryAttentionMaskHead produced NaN/Inf values after projecting {self.primary_level_name}."
                    )
                projected_level_shapes: dict[str, tuple[int, int, int, int]] = {
                    self.primary_level_name: tuple(int(value) for value in primary_projected.shape)
                }
                primary_hw = (int(primary_projected.shape[-2]), int(primary_projected.shape[-1]))
                tokens = primary_projected.flatten(2).transpose(1, 2).contiguous()
                memory = self.memory_tokens.unsqueeze(0).expand(int(tokens.shape[0]), -1, -1)
                for block in self.blocks:
                    tokens = block(tokens, memory)
                    if not torch.isfinite(tokens).all():
                        raise FrozenMaskHeadRuntimeError("FrozenSamMemoryAttentionMaskHead produced NaN/Inf token states.")
                coarse_features = tokens.transpose(1, 2).reshape(
                    int(primary_projected.shape[0]),
                    int(primary_projected.shape[1]),
                    int(primary_projected.shape[2]),
                    int(primary_projected.shape[3]),
                )
                reference_fine_name = self.primary_level_name
                reference_fine_hw = primary_hw
                fused_inputs: list[Any] = [coarse_features]
                for level_name in self.skip_level_names:
                    skip_tensor = feature_levels[level_name]
                    if skip_tensor.ndim != 4 or int(skip_tensor.shape[0]) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamMemoryAttentionMaskHead expected [1,C,H,W] for {level_name}, got {tuple(skip_tensor.shape)}."
                        )
                    skip_projected = self.skip_projections[level_name](skip_tensor)
                    if not torch.isfinite(skip_projected).all():
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamMemoryAttentionMaskHead produced NaN/Inf values after projecting {level_name}."
                        )
                    projected_level_shapes[level_name] = tuple(int(value) for value in skip_projected.shape)
                    skip_hw = (int(skip_projected.shape[-2]), int(skip_projected.shape[-1]))
                    if (skip_hw[0] * skip_hw[1], level_name) > (reference_fine_hw[0] * reference_fine_hw[1], reference_fine_name):
                        reference_fine_name = level_name
                        reference_fine_hw = skip_hw
                    fused_inputs.append(skip_projected)

                resized_fused_inputs: list[Any] = []
                for fused_input in fused_inputs:
                    if tuple(int(value) for value in fused_input.shape[-2:]) != reference_fine_hw:
                        fused_input = F.interpolate(
                            fused_input,
                            size=reference_fine_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                    resized_fused_inputs.append(fused_input)
                fused = torch.cat(resized_fused_inputs, dim=1)
                decoded = self.decoder(fused)
                logits = self.classifier(
                    F.interpolate(
                        decoded,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                if tuple(int(value) for value in logits.shape) != (1, 1, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamMemoryAttentionMaskHead produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in logits.shape),
                        },
                    )
                if not torch.isfinite(logits).all():
                    raise FrozenMaskHeadRuntimeError("FrozenSamMemoryAttentionMaskHead produced NaN/Inf logits.")
                return FrozenMaskHeadOutput(
                    logits=logits,
                    finest_level_name=reference_fine_name,
                    finest_grid_size=reference_fine_hw,
                    projected_level_shapes=projected_level_shapes,
                    fused_feature_shape=tuple(int(value) for value in fused.shape),
                )

        self.module = _Module()


class FrozenSamCoarsePlusMultibankNullRefineMaskHead:
    """Coarse-first residual refiner with optional `fpn_1`/`fpn_0` evidence banks and a learned null token."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        mid_level_name: str = "fpn_1",
        fine_level_name: str = "fpn_0",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        attention_hidden_dim: int = 32,
        attention_heads: int = 4,
        cross_attn_query_stride: int = 2,
        use_mid_level: bool = True,
        use_fine_level: bool = True,
        use_null_token: bool = True,
        use_learned_gate: bool = False,
        zero_init_residual_scale: bool = False,
        zero_init_attention_qkv: bool = False,
        residual_warmup_epochs: int = 0,
        residual_ramp_epochs: int = 0,
        residual_scale_init: float = 0.0,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.coarse_level_name = str(coarse_level_name)
                self.mid_level_name = str(mid_level_name)
                self.fine_level_name = str(fine_level_name)
                self.level_names = tuple(self.level_input_dims.keys())
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.attention_hidden_dim = int(attention_hidden_dim)
                self.attention_heads = int(attention_heads)
                self.cross_attn_query_stride = int(cross_attn_query_stride)
                self.num_classes = 1
                self.use_mid_level = bool(use_mid_level)
                self.use_fine_level = bool(use_fine_level)
                self.use_null_token = bool(use_null_token)
                self.use_learned_gate = bool(use_learned_gate)
                self.zero_init_residual_scale = bool(zero_init_residual_scale)
                self.zero_init_attention_qkv = bool(zero_init_attention_qkv)
                self.residual_warmup_epochs = int(residual_warmup_epochs)
                self.residual_ramp_epochs = int(residual_ramp_epochs)
                self.residual_scale_init = 0.0 if self.zero_init_residual_scale else float(residual_scale_init)
                self._runtime_epoch_index = 0
                self.active_bank_names = tuple(
                    level_name
                    for enabled, level_name in (
                        (self.use_mid_level, self.mid_level_name),
                        (self.use_fine_level, self.fine_level_name),
                    )
                    if enabled
                )
                if self.coarse_level_name not in self.level_input_dims:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead requires the coarse level in level_input_dims.",
                        diagnostics={"coarse_level_name": self.coarse_level_name, "level_input_dims": self.level_input_dims},
                    )
                if not self.active_bank_names:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead requires at least one enabled evidence bank.",
                        diagnostics={
                            "use_mid_level": self.use_mid_level,
                            "use_fine_level": self.use_fine_level,
                            "mid_level_name": self.mid_level_name,
                            "fine_level_name": self.fine_level_name,
                        },
                    )
                for level_name in self.active_bank_names:
                    if level_name not in self.level_input_dims:
                        raise FrozenMaskHeadRuntimeError(
                            "FrozenSamCoarsePlusMultibankNullRefineMaskHead requires enabled evidence-bank levels in level_input_dims.",
                            diagnostics={"missing_level_name": level_name, "level_input_dims": self.level_input_dims},
                        )
                if self.attention_hidden_dim < 1 or self.residual_projection_dim < 1 or self.residual_hidden_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim, residual_projection_dim, and residual_hidden_dim must be positive.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "residual_projection_dim": self.residual_projection_dim,
                            "residual_hidden_dim": self.residual_hidden_dim,
                        },
                    )
                if self.group_norm_groups < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "group_norm_groups must be positive.",
                        diagnostics={"group_norm_groups": self.group_norm_groups},
                    )
                if self.attention_hidden_dim % self.group_norm_groups != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim must be divisible by group_norm_groups.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "group_norm_groups": self.group_norm_groups,
                        },
                    )
                if self.attention_heads < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_heads must be positive.",
                        diagnostics={"attention_heads": self.attention_heads},
                    )
                if self.attention_hidden_dim % self.attention_heads != 0:
                    raise FrozenMaskHeadRuntimeError(
                        "attention_hidden_dim must be divisible by attention_heads for multibank null refine heads.",
                        diagnostics={
                            "attention_hidden_dim": self.attention_hidden_dim,
                            "attention_heads": self.attention_heads,
                        },
                    )
                if self.cross_attn_query_stride < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "cross_attn_query_stride must be positive.",
                        diagnostics={"cross_attn_query_stride": self.cross_attn_query_stride},
                    )
                if self.residual_warmup_epochs < 0 or self.residual_ramp_epochs < 0:
                    raise FrozenMaskHeadRuntimeError(
                        "joker residual warmup/ramp epochs must be non-negative.",
                        diagnostics={
                            "joker_residual_warmup_epochs": self.residual_warmup_epochs,
                            "joker_residual_ramp_epochs": self.residual_ramp_epochs,
                        },
                    )
                if self.residual_warmup_epochs > 0 and self.residual_ramp_epochs > 0:
                    raise FrozenMaskHeadRuntimeError(
                        "joker residual warmup and ramp schedules are mutually exclusive.",
                        diagnostics={
                            "joker_residual_warmup_epochs": self.residual_warmup_epochs,
                            "joker_residual_ramp_epochs": self.residual_ramp_epochs,
                        },
                    )

                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: self.level_input_dims[self.coarse_level_name]},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=self.num_classes,
                ).module
                self.bank_projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(
                            int(self.level_input_dims[level_name]),
                            self.residual_projection_dim,
                            kernel_size=1,
                            bias=True,
                        )
                        for level_name in self.active_bank_names
                    }
                )
                self.query_seed_projection = nn.Conv2d(3, self.attention_hidden_dim, kernel_size=1, bias=True)
                self.key_projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(self.residual_projection_dim, self.attention_hidden_dim, kernel_size=1, bias=True)
                        for level_name in self.active_bank_names
                    }
                )
                self.value_projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(self.residual_projection_dim, self.attention_hidden_dim, kernel_size=1, bias=True)
                        for level_name in self.active_bank_names
                    }
                )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=self.attention_hidden_dim,
                    num_heads=self.attention_heads,
                    batch_first=True,
                )
                if self.use_null_token:
                    self.null_key_token = nn.Parameter(torch.zeros(1, 1, self.attention_hidden_dim))
                    self.null_value_token = nn.Parameter(torch.zeros(1, 1, self.attention_hidden_dim))
                    nn.init.normal_(self.null_key_token, mean=0.0, std=0.02)
                    nn.init.normal_(self.null_value_token, mean=0.0, std=0.02)
                else:
                    self.register_parameter("null_key_token", None)
                    self.register_parameter("null_value_token", None)
                decoder_input_dim = self.attention_hidden_dim + (3 * self.num_classes)
                self.residual_decoder = nn.Sequential(
                    nn.Conv2d(decoder_input_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.residual_hidden_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.residual_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1, bias=True)
                if self.use_learned_gate:
                    self.attention_decoder = nn.Sequential(
                        nn.Conv2d(decoder_input_dim, self.attention_hidden_dim, kernel_size=3, padding=1, bias=True),
                        nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                        nn.GELU(),
                        nn.Conv2d(self.attention_hidden_dim, self.attention_hidden_dim, kernel_size=3, padding=1, bias=True),
                        nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                        nn.GELU(),
                    )
                    self.attention_classifier = nn.Conv2d(self.attention_hidden_dim, 1, kernel_size=1, bias=True)
                else:
                    self.attention_decoder = None
                    self.attention_classifier = None
                self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
                if self.zero_init_attention_qkv:
                    nn.init.zeros_(self.query_seed_projection.weight)
                    nn.init.zeros_(self.query_seed_projection.bias)
                    for projection in self.key_projections.values():
                        nn.init.zeros_(projection.weight)
                        nn.init.zeros_(projection.bias)
                    for projection in self.value_projections.values():
                        nn.init.zeros_(projection.weight)
                        nn.init.zeros_(projection.bias)

            def set_residual_training_progress(self, *, epoch_index: int) -> None:
                self._runtime_epoch_index = int(epoch_index)

            def get_residual_runtime_multiplier(self) -> float:
                return resolve_joker_residual_runtime_multiplier(
                    epoch_index=int(self._runtime_epoch_index),
                    residual_warmup_epochs=int(self.residual_warmup_epochs),
                    residual_ramp_epochs=int(self.residual_ramp_epochs),
                )

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenResidualMaskHeadOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                coarse_output: FrozenMaskHeadOutput = self.coarse_head(
                    {self.coarse_level_name: feature_levels[self.coarse_level_name]},
                    image_size=image_size,
                )

                projected_level_shapes: dict[str, tuple[int, int, int, int]] = {
                    f"{self.coarse_level_name}::coarse": coarse_output.projected_level_shapes[self.coarse_level_name]
                }
                target_level_name = max(
                    self.active_bank_names,
                    key=lambda level_name: (
                        int(feature_levels[level_name].shape[-2]) * int(feature_levels[level_name].shape[-1]),
                        level_name,
                    ),
                )
                target_hw = (
                    int(feature_levels[target_level_name].shape[-2]),
                    int(feature_levels[target_level_name].shape[-1]),
                )
                bank_key_tokens: list[Any] = []
                bank_value_tokens: list[Any] = []
                for level_name in self.active_bank_names:
                    tensor = feature_levels[level_name]
                    if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamCoarsePlusMultibankNullRefineMaskHead expected [1,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                        )
                    projected = self.bank_projections[level_name](tensor)
                    if not torch.isfinite(projected).all():
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamCoarsePlusMultibankNullRefineMaskHead produced NaN/Inf values after projecting {level_name}."
                        )
                    projected_level_shapes[f"{level_name}::bank"] = tuple(int(value) for value in projected.shape)
                    bank_key_tokens.append(self.key_projections[level_name](projected).flatten(2).transpose(1, 2).contiguous())
                    bank_value_tokens.append(self.value_projections[level_name](projected).flatten(2).transpose(1, 2).contiguous())

                key_tokens = torch.cat(bank_key_tokens, dim=1)
                value_tokens = torch.cat(bank_value_tokens, dim=1)
                if self.use_null_token:
                    key_tokens = torch.cat([key_tokens, self.null_key_token.expand(int(key_tokens.shape[0]), -1, -1)], dim=1)
                    value_tokens = torch.cat([value_tokens, self.null_value_token.expand(int(value_tokens.shape[0]), -1, -1)], dim=1)

                coarse_logits_up = F.interpolate(coarse_output.logits, size=target_hw, mode="bilinear", align_corners=False)
                coarse_probabilities = torch.sigmoid(coarse_logits_up)
                coarse_uncertainty = 4.0 * coarse_probabilities * (1.0 - coarse_probabilities)
                coarse_guidance = torch.cat([coarse_logits_up, coarse_probabilities, coarse_uncertainty], dim=1)
                query_seed = self.query_seed_projection(coarse_guidance)
                if self.cross_attn_query_stride > 1:
                    query_seed = F.avg_pool2d(query_seed, kernel_size=self.cross_attn_query_stride, stride=self.cross_attn_query_stride)
                query_hw = (int(query_seed.shape[-2]), int(query_seed.shape[-1]))
                query_tokens = query_seed.flatten(2).transpose(1, 2).contiguous()
                attended_tokens, attention_weights = self.cross_attention(
                    query_tokens,
                    key_tokens,
                    value_tokens,
                    need_weights=True,
                    average_attn_weights=False,
                )
                if not torch.isfinite(attended_tokens).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead produced NaN/Inf attended tokens."
                    )
                attended_features = attended_tokens.transpose(1, 2).reshape(
                    int(query_seed.shape[0]),
                    self.attention_hidden_dim,
                    query_hw[0],
                    query_hw[1],
                )
                attended_features = F.interpolate(attended_features, size=target_hw, mode="bilinear", align_corners=False)
                projected_level_shapes["multibank::attended"] = tuple(int(value) for value in attended_features.shape)
                attention_bank_means: dict[str, float] | None = None
                attention_real_mean: float | None = None
                attention_null_mean: float | None = None
                if attention_weights is not None:
                    attention_mass = attention_weights.detach().float().mean(dim=(0, 1, 2))
                    token_cursor = 0
                    attention_bank_means = {}
                    for level_name, tokens in zip(self.active_bank_names, bank_key_tokens, strict=False):
                        token_count = int(tokens.shape[1])
                        attention_bank_means[level_name] = float(
                            attention_mass[token_cursor : token_cursor + token_count].sum().cpu().item()
                        )
                        token_cursor += token_count
                    attention_real_mean = float(sum(attention_bank_means.values()))
                    if self.use_null_token:
                        attention_null_mean = float(attention_mass[token_cursor:].sum().cpu().item())

                decode_input = torch.cat([attended_features, coarse_guidance], dim=1)
                residual_features = self.residual_decoder(decode_input)
                residual_logits = self.residual_classifier(
                    F.interpolate(residual_features, size=(image_height, image_width), mode="bilinear", align_corners=False)
                )
                attention_map = None
                gated_residual_logits = residual_logits
                if self.use_learned_gate:
                    attention_features = self.attention_decoder(decode_input)
                    attention_map = torch.sigmoid(
                        self.attention_classifier(
                            F.interpolate(attention_features, size=(image_height, image_width), mode="bilinear", align_corners=False)
                        )
                    )
                    gated_residual_logits = attention_map * residual_logits
                residual_runtime_multiplier = float(self.get_residual_runtime_multiplier())
                final_logits = coarse_output.logits + (
                    float(residual_runtime_multiplier) * self.residual_scale.view(1, 1, 1, 1) * gated_residual_logits
                )
                if tuple(int(value) for value in final_logits.shape) != (1, self.num_classes, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in final_logits.shape),
                        },
                    )
                if not torch.isfinite(final_logits).all():
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamCoarsePlusMultibankNullRefineMaskHead produced NaN/Inf logits."
                    )
                return FrozenResidualMaskHeadOutput(
                    logits=final_logits,
                    coarse_logits=coarse_output.logits,
                    residual_logits=residual_logits,
                    finest_level_name=target_level_name,
                    finest_grid_size=target_hw,
                    projected_level_shapes=projected_level_shapes,
                    fused_feature_shape=tuple(int(value) for value in residual_features.shape),
                    residual_scale=float(self.residual_scale.detach().cpu().item()),
                    attention_map=attention_map,
                    attended_feature_shape=tuple(int(value) for value in attended_features.shape),
                    residual_runtime_multiplier=residual_runtime_multiplier,
                    attention_bank_means=attention_bank_means,
                    attention_real_mean=attention_real_mean,
                    attention_null_mean=attention_null_mean,
                )

        self.module = _Module()


class FrozenSamSerializedProgressiveRefinementMaskHead:
    """Coarse-first progressive refiner: fpn_2 -> fpn_1 refine -> fpn_0 refine."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        mid_level_name: str = "fpn_1",
        fine_level_name: str = "fpn_0",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        attention_hidden_dim: int = 32,
        num_classes: int = 1,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.coarse_level_name = str(coarse_level_name)
                self.mid_level_name = str(mid_level_name)
                self.fine_level_name = str(fine_level_name)
                self.level_names = (self.coarse_level_name, self.mid_level_name, self.fine_level_name)
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.attention_hidden_dim = int(attention_hidden_dim)
                self.num_classes = int(num_classes)
                self._runtime_epoch_index = 0
                self.alpha_fpn1 = 0.0
                self.alpha_fpn0 = 0.0

                for name in self.level_names:
                    if name not in self.level_input_dims:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamSerializedProgressiveRefinementMaskHead requires {name} in level_input_dims."
                        )

                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: self.level_input_dims[self.coarse_level_name]},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=self.num_classes,
                ).module

                # fpn_1 Refiner
                self.fpn1_projection = nn.Conv2d(self.level_input_dims[self.mid_level_name], self.residual_projection_dim, kernel_size=1)
                self.fpn1_decoder = self._build_refine_decoder()
                self.fpn1_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1)
                self.fpn1_attention = self._build_attention_block()
                self.fpn1_scale = nn.Parameter(torch.zeros(1))

                # fpn_0 Refiner
                self.fpn0_projection = nn.Conv2d(self.level_input_dims[self.fine_level_name], self.residual_projection_dim, kernel_size=1)
                self.fpn0_decoder = self._build_refine_decoder()
                self.fpn0_classifier = nn.Conv2d(self.residual_hidden_dim, self.num_classes, kernel_size=1)
                self.fpn0_attention = self._build_attention_block()
                self.fpn0_scale = nn.Parameter(torch.zeros(1))

                nn.init.zeros_(self.fpn1_classifier.weight)
                nn.init.zeros_(self.fpn1_classifier.bias)
                nn.init.zeros_(self.fpn0_classifier.weight)
                nn.init.zeros_(self.fpn0_classifier.bias)

            def _build_refine_decoder(self) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(self.residual_projection_dim, self.residual_hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.residual_hidden_dim, self.residual_hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )

            def _build_attention_block(self) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(self.residual_projection_dim + 3, self.attention_hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(self.group_norm_groups, self.attention_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.attention_hidden_dim, 1, kernel_size=1),
                    nn.Sigmoid(),
                )

            def set_residual_training_progress(self, epoch_index: int) -> None:
                self._runtime_epoch_index = int(epoch_index)
                t = float(self._runtime_epoch_index)
                # fpn_1: 0 -> 1 over first 20 epochs (0-19)
                self.alpha_fpn1 = min(max(t / 19.0, 0.0), 1.0)
                # fpn_0: stay at 0 for 20 epochs (0-19), then ramp 0 -> 1 over next 20 epochs (20-39)
                self.alpha_fpn0 = min(max((t - 20.0) / 19.0, 0.0), 1.0)

            def forward(self, feature_levels: Mapping[str, Any], *, image_size: Sequence[int]) -> FrozenResidualMaskHeadOutput:
                import torch
                import torch.nn.functional as F
                h, w = int(image_size[0]), int(image_size[1])

                coarse_output = self.coarse_head({self.coarse_level_name: feature_levels[self.coarse_level_name]}, image_size=image_size)
                L2 = coarse_output.logits

                # Stage 1: fpn_1
                f1_feat = feature_levels[self.mid_level_name]
                f1_proj = self.fpn1_projection(f1_feat)
                f1_res = self.fpn1_decoder(f1_proj)
                f1_delta = F.interpolate(self.fpn1_classifier(f1_res), size=(h, w), mode="bilinear", align_corners=False)

                L2_up = F.interpolate(L2, size=f1_feat.shape[-2:], mode="bilinear", align_corners=False)
                p2 = torch.sigmoid(L2_up)
                u2 = 4.0 * p2 * (1.0 - p2)
                attn1 = self.fpn1_attention(torch.cat([f1_proj, L2_up, p2, u2], dim=1))
                attn1_up = F.interpolate(attn1, size=(h, w), mode="bilinear", align_corners=False)

                # alpha1(t) * s1 * delta1
                L21 = L2 + self.alpha_fpn1 * self.fpn1_scale.view(1, 1, 1, 1) * attn1_up * f1_delta

                # Stage 2: fpn_0
                f0_feat = feature_levels[self.fine_level_name]
                f0_proj = self.fpn0_projection(f0_feat)
                f0_res = self.fpn0_decoder(f0_proj)
                f0_delta = F.interpolate(self.fpn0_classifier(f0_res), size=(h, w), mode="bilinear", align_corners=False)

                L21_up = F.interpolate(L21, size=f0_feat.shape[-2:], mode="bilinear", align_corners=False)
                p21 = torch.sigmoid(L21_up)
                u21 = 4.0 * p21 * (1.0 - p21)
                attn0 = self.fpn0_attention(torch.cat([f0_proj, L21_up, p21, u21], dim=1))
                attn0_up = F.interpolate(attn0, size=(h, w), mode="bilinear", align_corners=False)

                # alpha0(t) * s0 * delta0
                L210 = L21 + self.alpha_fpn0 * self.fpn0_scale.view(1, 1, 1, 1) * attn0_up * f0_delta

                projected_level_shapes = {
                    self.coarse_level_name: coarse_output.projected_level_shapes[self.coarse_level_name],
                    self.mid_level_name: tuple(int(v) for v in f1_proj.shape),
                    self.fine_level_name: tuple(int(v) for v in f0_proj.shape),
                }

                return FrozenResidualMaskHeadOutput(
                    logits=L210,
                    coarse_logits=L2,
                    residual_logits=f0_delta,  # Using fpn_0 delta as representative residual_logits for L1 if enabled
                    finest_level_name=self.fine_level_name,
                    finest_grid_size=(f0_feat.shape[-2], f0_feat.shape[-1]),
                    projected_level_shapes=projected_level_shapes,
                    fused_feature_shape=tuple(int(v) for v in f0_res.shape),
                    residual_scale=float(self.fpn0_scale.detach().cpu().item()),
                    attention_map=attn0_up,
                    residual_runtime_multiplier=float(max(self.alpha_fpn1, self.alpha_fpn0)),
                    attention_fpn1_mean=float(attn1.mean().detach().cpu().item()),
                    attention_fpn0_mean=float(attn0.mean().detach().cpu().item()),
                    residual_scale_fpn1=float(self.fpn1_scale.detach().cpu().item()),
                    residual_scale_fpn0=float(self.fpn0_scale.detach().cpu().item()),
                    alpha_fpn1=float(self.alpha_fpn1),
                    alpha_fpn0=float(self.alpha_fpn0),
                )

        self.module = _Module()


class FrozenSamCoarseThenMidThenFineLocalResidualMaskHead:
    """Coarse-first local routed refiner: `fpn_2 -> fpn_1 -> fpn_0`."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        coarse_level_name: str = "fpn_2",
        mid_level_name: str = "fpn_1",
        fine_level_name: str = "fpn_0",
        projection_dim: int = 64,
        decoder_dim: int = 64,
        group_norm_groups: int = 8,
        residual_projection_dim: int = 32,
        residual_hidden_dim: int = 32,
        zero_init_residual_scale: bool = False,
        zero_init_attention_qkv: bool = False,
        residual_warmup_epochs: int = 0,
        residual_ramp_epochs: int = 0,
        residual_scale_init: float = 0.0,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.coarse_level_name = str(coarse_level_name)
                self.mid_level_name = str(mid_level_name)
                self.fine_level_name = str(fine_level_name)
                self.level_names = (self.coarse_level_name, self.mid_level_name, self.fine_level_name)
                for name in self.level_names:
                    if name not in self.level_input_dims:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamCoarseThenMidThenFineLocalResidualMaskHead requires {name} in level_input_dims."
                        )
                self.projection_dim = int(projection_dim)
                self.decoder_dim = int(decoder_dim)
                self.group_norm_groups = int(group_norm_groups)
                self.residual_projection_dim = int(residual_projection_dim)
                self.residual_hidden_dim = int(residual_hidden_dim)
                self.zero_init_residual_scale = bool(zero_init_residual_scale)
                self.zero_init_attention_qkv = bool(zero_init_attention_qkv)
                self.residual_warmup_epochs = int(residual_warmup_epochs)
                self.residual_ramp_epochs = int(residual_ramp_epochs)
                if self.residual_warmup_epochs < 0 or self.residual_ramp_epochs < 0:
                    raise FrozenMaskHeadRuntimeError(
                        "joker residual warmup/ramp epochs must be non-negative.",
                        diagnostics={
                            "joker_residual_warmup_epochs": self.residual_warmup_epochs,
                            "joker_residual_ramp_epochs": self.residual_ramp_epochs,
                        },
                    )
                if self.residual_warmup_epochs > 0 and self.residual_ramp_epochs > 0:
                    raise FrozenMaskHeadRuntimeError(
                        "joker residual warmup and ramp schedules are mutually exclusive.",
                        diagnostics={
                            "joker_residual_warmup_epochs": self.residual_warmup_epochs,
                            "joker_residual_ramp_epochs": self.residual_ramp_epochs,
                        },
                    )
                self.residual_scale_init = 0.0 if self.zero_init_residual_scale else float(residual_scale_init)
                self.alpha_fpn1 = 0.0
                self.alpha_fpn0 = 0.0
                self._runtime_epoch_index = 0
                self.coarse_head = FrozenSamMultiscaleMaskHead(
                    level_input_dims={self.coarse_level_name: self.level_input_dims[self.coarse_level_name]},
                    projection_dim=self.projection_dim,
                    decoder_dim=self.decoder_dim,
                    group_norm_groups=self.group_norm_groups,
                    num_classes=1,
                ).module
                self.fpn1_projection = nn.Conv2d(self.level_input_dims[self.mid_level_name], self.residual_projection_dim, kernel_size=1, bias=True)
                self.fpn1_decoder = nn.Sequential(
                    nn.Conv2d(self.residual_projection_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.residual_hidden_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.fpn1_classifier = nn.Conv2d(self.residual_hidden_dim, 1, kernel_size=1, bias=True)
                self.scale_fpn1 = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))
                self.fpn0_projection = nn.Conv2d(self.level_input_dims[self.fine_level_name], self.residual_projection_dim, kernel_size=1, bias=True)
                self.fpn0_decoder = nn.Sequential(
                    nn.Conv2d(self.residual_projection_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(self.residual_hidden_dim, self.residual_hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.GroupNorm(self.group_norm_groups, self.residual_hidden_dim),
                    nn.GELU(),
                )
                self.fpn0_classifier = nn.Conv2d(self.residual_hidden_dim, 1, kernel_size=1, bias=True)
                self.scale_fpn0 = nn.Parameter(torch.tensor(self.residual_scale_init, dtype=torch.float32))

            def set_residual_training_progress(self, *, epoch_index: int) -> None:
                t = max(0, int(epoch_index))
                self._runtime_epoch_index = t
                w = int(self.residual_warmup_epochs)
                r = int(self.residual_ramp_epochs)
                if w > 0:
                    self.alpha_fpn1 = 0.0 if t < w else 1.0
                    self.alpha_fpn0 = 0.0 if t < 2 * w else 1.0
                elif r > 0:
                    self.alpha_fpn1 = float(min(max(float(t) / float(r), 0.0), 1.0))
                    self.alpha_fpn0 = 0.0 if t < r else float(min(max(float(t - r) / float(r), 0.0), 1.0))
                else:
                    self.alpha_fpn1 = 1.0
                    self.alpha_fpn0 = 1.0

            @staticmethod
            def _make_topk_routed_mask(logits: Any, *, top_fraction: float, dilation_kernel_size: int) -> Any:
                import torch
                import torch.nn.functional as F

                probabilities = torch.sigmoid(logits)
                local_uncertainty = 4.0 * probabilities * (1.0 - probabilities)
                batch_size = int(local_uncertainty.shape[0])
                flat_uncertainty = local_uncertainty.flatten(1)
                topk_count = max(1, int(math.ceil(float(flat_uncertainty.shape[1]) * float(top_fraction))))
                thresholds = torch.topk(flat_uncertainty, k=topk_count, dim=1, largest=True, sorted=True).values[:, -1]
                routed = (flat_uncertainty >= thresholds.unsqueeze(1)).reshape_as(local_uncertainty).to(dtype=logits.dtype)
                if dilation_kernel_size > 1:
                    routed = F.max_pool2d(
                        routed,
                        kernel_size=int(dilation_kernel_size),
                        stride=1,
                        padding=int(dilation_kernel_size) // 2,
                    )
                return routed.reshape(batch_size, 1, int(logits.shape[-2]), int(logits.shape[-1]))

            def forward(self, feature_levels: Mapping[str, Any], *, image_size: Sequence[int]) -> FrozenResidualMaskHeadOutput:
                import torch.nn.functional as F
                h, w = int(image_size[0]), int(image_size[1])
                coarse_output = self.coarse_head({self.coarse_level_name: feature_levels[self.coarse_level_name]}, image_size=image_size)
                coarse_logits = coarse_output.logits

                f1 = feature_levels[self.mid_level_name]
                f1_proj = self.fpn1_projection(f1)
                f1_res = self.fpn1_decoder(f1_proj)
                stage1_residual_logits = F.interpolate(
                    self.fpn1_classifier(f1_res), size=(h, w), mode="bilinear", align_corners=False
                )
                route1_mask = self._make_topk_routed_mask(
                    coarse_logits,
                    top_fraction=0.20,
                    dilation_kernel_size=5,
                ).detach()
                stage1_logits = coarse_logits + (
                    self.alpha_fpn1
                    * self.scale_fpn1.view(1, 1, 1, 1)
                    * route1_mask
                    * stage1_residual_logits
                )

                f0 = feature_levels[self.fine_level_name]
                f0_proj = self.fpn0_projection(f0)
                f0_res = self.fpn0_decoder(f0_proj)
                stage2_residual_logits = F.interpolate(
                    self.fpn0_classifier(f0_res), size=(h, w), mode="bilinear", align_corners=False
                )
                route0_mask = self._make_topk_routed_mask(
                    stage1_logits,
                    top_fraction=0.10,
                    dilation_kernel_size=3,
                ).detach()
                stage2_logits = stage1_logits + (
                    self.alpha_fpn0
                    * self.scale_fpn0.view(1, 1, 1, 1)
                    * route0_mask
                    * stage2_residual_logits
                )

                return FrozenResidualMaskHeadOutput(
                    logits=stage2_logits,
                    coarse_logits=coarse_logits,
                    residual_logits=stage2_residual_logits,
                    finest_level_name=self.fine_level_name,
                    finest_grid_size=(int(f0.shape[-2]), int(f0.shape[-1])),
                    projected_level_shapes={
                        f"{self.coarse_level_name}::coarse": coarse_output.projected_level_shapes[self.coarse_level_name],
                        f"{self.mid_level_name}::residual": tuple(int(v) for v in f1_proj.shape),
                        f"{self.fine_level_name}::residual": tuple(int(v) for v in f0_proj.shape),
                    },
                    fused_feature_shape=tuple(int(v) for v in f0_res.shape),
                    residual_scale=float(self.scale_fpn0.detach().cpu().item()),
                    residual_runtime_multiplier=1.0,
                    alpha_fpn1=float(self.alpha_fpn1),
                    alpha_fpn0=float(self.alpha_fpn0),
                    routed_mask_fpn1=route1_mask,
                    routed_mask_fpn0=route0_mask,
                    route1_mask=route1_mask,
                    route0_mask=route0_mask,
                    stage1_logits=stage1_logits,
                    stage2_logits=stage2_logits,
                    stage1_residual_logits=stage1_residual_logits,
                    stage2_residual_logits=stage2_residual_logits,
                    residual_scale_fpn1=float(self.scale_fpn1.detach().cpu().item()),
                    residual_scale_fpn0=float(self.scale_fpn0.detach().cpu().item()),
                )

        self.module = _Module()


class FrozenSamSerializedLocalRoutedRefinementMaskHead(FrozenSamCoarseThenMidThenFineLocalResidualMaskHead):
    """Backward-compatible alias for the deterministic local routed refiner."""


class FrozenSamForegroundProbe:
    """Small supervised foreground probe on top of frozen SAM feature levels."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        head_kind: str,
        projection_dim: int = 32,
        hidden_dim: int = 32,
    ) -> None:
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not level_input_dims:
                    raise FrozenMaskHeadRuntimeError("FrozenSamForegroundProbe requires at least one input level.")
                self.level_names = tuple(level_input_dims.keys())
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.head_kind = str(head_kind)
                self.projection_dim = int(projection_dim)
                self.hidden_dim = int(hidden_dim)
                if self.hidden_dim < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "hidden_dim must be positive.",
                        diagnostics={"hidden_dim": self.hidden_dim},
                    )
                if self.head_kind not in {"linear", "tiny_nonlinear"}:
                    raise FrozenMaskHeadRuntimeError(
                        f"Unsupported FrozenSamForegroundProbe head_kind '{self.head_kind}'."
                    )
                if self.head_kind == "linear":
                    if len(self.level_names) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            "Linear frozen foreground probes currently support exactly one selected level.",
                            diagnostics={"level_names": self.level_names},
                        )
                    level_name = self.level_names[0]
                    self.single_level_name = level_name
                    self.linear_classifier = nn.Conv2d(
                        int(self.level_input_dims[level_name]),
                        1,
                        kernel_size=1,
                        bias=True,
                    )
                    self.level_projections = None
                    self.head = None
                    return

                if len(self.level_names) == 1:
                    level_name = self.level_names[0]
                    self.single_level_name = level_name
                    self.level_projections = None
                    self.head = nn.Sequential(
                        nn.Conv2d(
                            int(self.level_input_dims[level_name]),
                            self.hidden_dim,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True),
                    )
                    return

                self.single_level_name = None
                self.level_projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(
                            int(input_dim),
                            self.projection_dim,
                            kernel_size=1,
                            bias=True,
                        )
                        for level_name, input_dim in self.level_input_dims.items()
                    }
                )
                fused_dim = len(self.level_names) * self.projection_dim
                self.head = nn.Sequential(
                    nn.Conv2d(fused_dim, self.hidden_dim, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True),
                )

            def forward(
                self,
                feature_levels: Mapping[str, Any],
                *,
                image_size: Sequence[int],
            ) -> FrozenForegroundProbeOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamForegroundProbe got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                if len(image_size) != 2:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size must contain exactly two integers: (height, width).",
                        diagnostics={"image_size": tuple(image_size)},
                    )
                image_height, image_width = (int(image_size[0]), int(image_size[1]))
                if image_height < 1 or image_width < 1:
                    raise FrozenMaskHeadRuntimeError(
                        "image_size values must be positive.",
                        diagnostics={"image_size": (image_height, image_width)},
                    )

                level_shapes: dict[str, tuple[int, int, int, int]] = {}
                if self.head_kind == "linear":
                    level_name = self.single_level_name
                    tensor = feature_levels[level_name]
                    if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamForegroundProbe expected [1,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                        )
                    logits = self.linear_classifier(tensor)
                    level_shapes[level_name] = tuple(int(value) for value in tensor.shape)
                    fused = tensor
                    finest_level_name = level_name
                    finest_grid_size = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                elif self.single_level_name is not None:
                    level_name = self.single_level_name
                    tensor = feature_levels[level_name]
                    if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                        raise FrozenMaskHeadRuntimeError(
                            f"FrozenSamForegroundProbe expected [1,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                        )
                    logits = self.head(tensor)
                    level_shapes[level_name] = tuple(int(value) for value in tensor.shape)
                    fused = tensor
                    finest_level_name = level_name
                    finest_grid_size = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                else:
                    projected_levels: list[Any] = []
                    finest_level_name: str | None = None
                    finest_grid_size: tuple[int, int] | None = None
                    for level_name in self.level_names:
                        tensor = feature_levels[level_name]
                        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
                            raise FrozenMaskHeadRuntimeError(
                                f"FrozenSamForegroundProbe expected [1,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                            )
                        level_shapes[level_name] = tuple(int(value) for value in tensor.shape)
                        current_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                        if finest_grid_size is None or (current_hw[0] * current_hw[1], level_name) > (
                            finest_grid_size[0] * finest_grid_size[1],
                            finest_level_name or "",
                        ):
                            finest_grid_size = current_hw
                            finest_level_name = level_name
                    if finest_level_name is None or finest_grid_size is None:
                        raise FrozenMaskHeadRuntimeError("FrozenSamForegroundProbe could not resolve a finest level.")
                    for level_name in self.level_names:
                        projected = self.level_projections[level_name](feature_levels[level_name])
                        if tuple(int(value) for value in projected.shape[-2:]) != finest_grid_size:
                            projected = F.interpolate(
                                projected,
                                size=finest_grid_size,
                                mode="bilinear",
                                align_corners=False,
                            )
                        projected_levels.append(projected)
                    fused = torch.cat(projected_levels, dim=1)
                    logits = self.head(fused)

                if not torch.isfinite(logits).all():
                    raise FrozenMaskHeadRuntimeError("FrozenSamForegroundProbe produced NaN/Inf logits.")
                if tuple(int(value) for value in logits.shape[-2:]) != (image_height, image_width):
                    logits = F.interpolate(
                        logits,
                        size=(image_height, image_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                if tuple(int(value) for value in logits.shape) != (1, 1, image_height, image_width):
                    raise FrozenMaskHeadRuntimeError(
                        "FrozenSamForegroundProbe produced an unexpected foreground logit shape.",
                        diagnostics={
                            "expected_shape": (1, 1, image_height, image_width),
                            "received_shape": tuple(int(value) for value in logits.shape),
                        },
                    )
                if not torch.isfinite(logits).all():
                    raise FrozenMaskHeadRuntimeError("FrozenSamForegroundProbe produced NaN/Inf logits.")
                return FrozenForegroundProbeOutput(
                    logits=logits,
                    finest_level_name=finest_level_name,
                    finest_grid_size=finest_grid_size,
                    level_shapes=level_shapes,
                    fused_feature_shape=tuple(int(value) for value in fused.shape),
                )

        self.module = _Module()


def resolve_frozen_mask_head_variant_levels(
    *,
    variant: str,
    available_level_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve the ordered frozen-mask-head scale subset for one run variant."""

    try:
        variant_spec = FROZEN_MASK_HEAD_VARIANT_SPECS[variant]
    except KeyError as exc:
        expected = ", ".join(sorted(FROZEN_MASK_HEAD_VARIANT_SPECS))
        raise FrozenMaskHeadRuntimeError(
            f"Unknown frozen mask-head variant '{variant}'. Expected one of: {expected}."
        ) from exc

    available = tuple(str(name) for name in available_level_names)
    if not available:
        raise FrozenMaskHeadRuntimeError("No SAM pyramid levels were available for the frozen mask head.")

    policy = str(variant_spec["selected_level_policy"])
    if policy == "all_scales":
        return available
    if policy == "coarse_plus_next_finer":
        if len(available) < 2:
            raise FrozenMaskHeadRuntimeError(
                "Frozen mask-head coarse-plus-next-finer runs require at least two discovered SAM levels.",
                diagnostics={"available_level_names": available},
            )
        return available[:2]
    if policy == "single_level_by_name":
        requested = str(variant_spec["requested_level_name"])
        if requested not in available:
            raise FrozenMaskHeadRuntimeError(
                "Frozen mask-head single-scale variant requested a level that was not discovered.",
                diagnostics={"requested_level_name": requested, "available_level_names": available},
            )
        return (requested,)
    if policy == "named_level_subset":
        requested_level_names = tuple(str(name) for name in variant_spec["requested_level_names"])
        missing = [name for name in requested_level_names if name not in available]
        if missing:
            raise FrozenMaskHeadRuntimeError(
                "Frozen mask-head named-level variant requested levels that were not discovered.",
                diagnostics={
                    "requested_level_names": requested_level_names,
                    "missing_levels": missing,
                    "available_level_names": available,
                },
            )
        return requested_level_names
    raise FrozenMaskHeadRuntimeError(f"Unsupported frozen mask-head selected-level policy '{policy}'.")


def resolve_glas_frozen_feature_probe_variant(
    *,
    variant: str,
    available_level_names: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Resolve the ordered scale subset and head spec for one GlaS probe variant."""

    try:
        variant_spec = GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS[variant]
    except KeyError as exc:
        expected = ", ".join(sorted(GLAS_FROZEN_FEATURE_PROBE_VARIANT_SPECS))
        raise FrozenMaskHeadRuntimeError(
            f"Unknown GlaS frozen-feature probe variant '{variant}'. Expected one of: {expected}."
        ) from exc
    policy = str(variant_spec["selected_level_policy"])
    available = tuple(str(name) for name in available_level_names)
    if not available:
        raise FrozenMaskHeadRuntimeError("No SAM pyramid levels were available for the GlaS frozen-feature probe.")
    if policy == "all_scales":
        return variant_spec, available
    if policy == "single_level_by_name":
        requested = str(variant_spec["requested_level_name"])
        if requested not in available:
            raise FrozenMaskHeadRuntimeError(
                "GlaS frozen-feature probe variant requested a level that was not discovered.",
                diagnostics={"requested_level_name": requested, "available_level_names": available},
            )
        return variant_spec, (requested,)
    raise FrozenMaskHeadRuntimeError(
        f"Unsupported GlaS frozen-feature probe selected-level policy '{policy}'."
    )


def compute_bce_dice_loss(
    logits: Any,
    target_mask: Any,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1.0,
    boundary_mask: Any | None = None,
    boundary_weight: float = 0.0,
) -> MaskHeadLossResult:
    """Compute a transparent BCE plus Dice loss for one binary foreground mask."""

    import torch
    import torch.nn.functional as F

    if logits.ndim != 4 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 1:
        raise FrozenMaskHeadRuntimeError(
            f"compute_bce_dice_loss expects logits [1,1,H,W], got {tuple(int(value) for value in logits.shape)}."
        )

    target_tensor = torch.as_tensor(target_mask, dtype=logits.dtype, device=logits.device)
    if target_tensor.ndim == 2:
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif target_tensor.ndim == 3:
        target_tensor = target_tensor.unsqueeze(0)
    if target_tensor.ndim != 4 or tuple(int(value) for value in target_tensor.shape) != tuple(
        int(value) for value in logits.shape
    ):
        raise FrozenMaskHeadRuntimeError(
            "Target mask shape does not match the predicted logit map.",
            diagnostics={
                "logit_shape": tuple(int(value) for value in logits.shape),
                "target_shape": tuple(int(value) for value in target_tensor.shape),
            },
        )

    boundary_weight_value = float(boundary_weight)
    if boundary_weight_value < 0.0:
        raise FrozenMaskHeadRuntimeError(
            "boundary_weight must be non-negative.",
            diagnostics={"boundary_weight": boundary_weight_value},
        )
    probabilities = torch.sigmoid(logits)
    if boundary_mask is not None:
        boundary_tensor = torch.as_tensor(boundary_mask, dtype=logits.dtype, device=logits.device)
        if boundary_tensor.ndim == 2:
            boundary_tensor = boundary_tensor.unsqueeze(0).unsqueeze(0)
        elif boundary_tensor.ndim == 3:
            boundary_tensor = boundary_tensor.unsqueeze(0)
        if boundary_tensor.ndim != 4 or tuple(int(value) for value in boundary_tensor.shape) != tuple(
            int(value) for value in logits.shape
        ):
            raise FrozenMaskHeadRuntimeError(
                "Boundary mask shape does not match the predicted logit map.",
                diagnostics={
                    "logit_shape": tuple(int(value) for value in logits.shape),
                    "boundary_shape": tuple(int(value) for value in boundary_tensor.shape),
                },
            )
        bce_pixel_weight = 1.0 + boundary_weight_value * boundary_tensor
        bce_loss = F.binary_cross_entropy_with_logits(logits, target_tensor, weight=bce_pixel_weight)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
    intersection = torch.sum(probabilities * target_tensor)
    denominator = torch.sum(probabilities) + torch.sum(target_tensor)
    dice_score = (2.0 * intersection + float(smooth)) / (denominator + float(smooth))
    dice_loss = 1.0 - dice_score
    total_loss = float(bce_weight) * bce_loss + float(dice_weight) * dice_loss
    if not torch.isfinite(total_loss):
        raise FrozenMaskHeadRuntimeError("BCE+Dice loss became NaN/Inf.")

    return MaskHeadLossResult(
        loss=total_loss,
        bce_loss=bce_loss,
        dice_loss=dice_loss,
        mean_probability=float(probabilities.detach().mean().cpu().item()),
        predicted_positive_fraction=float((probabilities.detach() >= 0.5).float().mean().cpu().item()),
        target_positive_fraction=float(target_tensor.detach().mean().cpu().item()),
    )


def threshold_foreground_logits(logits: Any, *, threshold: float = 0.5) -> Any:
    """Convert one `[1,1,H,W]` logit map into a boolean foreground prediction."""

    import torch

    if logits.ndim != 4 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 1:
        raise FrozenMaskHeadRuntimeError(
            f"threshold_foreground_logits expects logits [1,1,H,W], got {tuple(int(value) for value in logits.shape)}."
        )
    probabilities = torch.sigmoid(logits)
    return (probabilities[0, 0] >= float(threshold)).detach().cpu().numpy().astype(bool)


def count_trainable_parameters(module: Any) -> int:
    """Return the number of trainable parameters in one torch module."""

    return int(sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad))


def resolve_residual_head_gate_mode(value: str) -> str:
    """Validate the explicit residual gating mode."""

    mode = str(value)
    if mode not in RESIDUAL_HEAD_GATE_MODES:
        raise FrozenMaskHeadRuntimeError(
            f"Unsupported residual gate mode '{mode}'.",
            diagnostics={"supported_residual_gate_modes": list(RESIDUAL_HEAD_GATE_MODES)},
        )
    return mode


def resolve_residual_head_gate_threshold(value: float) -> float:
    """Validate the explicit residual gating threshold."""

    threshold = float(value)
    if not (0.0 < threshold <= 0.5):
        raise FrozenMaskHeadRuntimeError(
            "Residual gate threshold must lie in the open interval (0, 0.5].",
            diagnostics={"residual_gate_threshold": threshold},
        )
    return threshold


def resolve_residual_head_coarse_loss_weight(args) -> float:
    """Resolve the auxiliary coarse-loss weight, including the joker default contract."""

    raw_value = getattr(args, "coarse_loss_weight", 0.0)
    value = 0.0 if raw_value is None else float(raw_value)
    if is_multibank_null_refine_variant(str(getattr(args, "variant", ""))) and value <= 0.0:
        return float(FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["joker_default_coarse_loss_weight"])
    return value


def resolve_joker_residual_runtime_multiplier(
    *,
    epoch_index: int,
    residual_warmup_epochs: int,
    residual_ramp_epochs: int,
) -> float:
    """Resolve the joker residual-path multiplier for the current zero-based epoch index."""

    current_epoch_index = max(0, int(epoch_index))
    warmup_epochs = max(0, int(residual_warmup_epochs))
    ramp_epochs = max(0, int(residual_ramp_epochs))
    if warmup_epochs > 0 and ramp_epochs > 0:
        raise FrozenMaskHeadRuntimeError(
            "joker residual warmup and ramp schedules are mutually exclusive.",
            diagnostics={
                "joker_residual_warmup_epochs": warmup_epochs,
                "joker_residual_ramp_epochs": ramp_epochs,
            },
        )
    if warmup_epochs > 0:
        return 0.0 if current_epoch_index < warmup_epochs else 1.0
    if ramp_epochs > 0:
        if ramp_epochs == 1:
            return 1.0
        return float(min(max(current_epoch_index, 0) / float(ramp_epochs - 1), 1.0))
    return 1.0


def combine_residual_head_logits(
    output: FrozenResidualMaskHeadOutput,
    *,
    residual_alpha_override: float | None = None,
    residual_gate_mode: str = "none",
    residual_gate_threshold: float = RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD,
) -> ResidualHeadCombinationResult:
    """Combine coarse and residual logits with an explicit eval-time gate/alpha contract."""

    import torch

    if not isinstance(output, FrozenResidualMaskHeadOutput):
        raise FrozenMaskHeadRuntimeError(
            "combine_residual_head_logits requires a FrozenResidualMaskHeadOutput."
        )
    gate_mode = resolve_residual_head_gate_mode(residual_gate_mode)
    gate_threshold = None if gate_mode == "none" else resolve_residual_head_gate_threshold(residual_gate_threshold)
    learned_residual_scale = float(output.residual_scale)
    residual_runtime_multiplier = float(getattr(output, "residual_runtime_multiplier", 1.0))
    if residual_alpha_override is None:
        effective_residual_scale = float(learned_residual_scale) * float(residual_runtime_multiplier)
        residual_scale_source = "learned"
    else:
        effective_residual_scale = float(residual_alpha_override) * float(residual_runtime_multiplier)
        residual_scale_source = "override"

    coarse_logits = output.coarse_logits
    residual_logits = output.residual_logits
    if coarse_logits.ndim != 4 or residual_logits.ndim != 4:
        raise FrozenMaskHeadRuntimeError(
            "Residual head logits must both have shape [1,1,H,W].",
            diagnostics={
                "coarse_logits_shape": tuple(int(value) for value in coarse_logits.shape),
                "residual_logits_shape": tuple(int(value) for value in residual_logits.shape),
            },
        )
    if gate_mode == "none":
        uncertainty_gate_map = torch.ones_like(residual_logits)
    else:
        coarse_probabilities = torch.sigmoid(coarse_logits)
        uncertainty = torch.abs(coarse_probabilities - 0.5)
        if gate_mode == "hard_uncertainty":
            uncertainty_gate_map = (uncertainty <= float(gate_threshold)).to(dtype=residual_logits.dtype)
        elif gate_mode == "soft_uncertainty":
            uncertainty_gate_map = torch.clamp(
                (float(gate_threshold) - uncertainty) / float(gate_threshold),
                min=0.0,
                max=1.0,
            )
        else:  # pragma: no cover - validated above
            raise FrozenMaskHeadRuntimeError(
                f"Unsupported residual gate mode '{gate_mode}'.",
                diagnostics={"supported_residual_gate_modes": list(RESIDUAL_HEAD_GATE_MODES)},
            )
    attention_map = output.attention_map
    if attention_map is None:
        gate_map = uncertainty_gate_map
    else:
        if tuple(int(value) for value in attention_map.shape) != tuple(int(value) for value in residual_logits.shape):
            raise FrozenMaskHeadRuntimeError(
                "Residual attention map shape must match residual logits shape.",
                diagnostics={
                    "attention_shape": tuple(int(value) for value in attention_map.shape),
                    "residual_logits_shape": tuple(int(value) for value in residual_logits.shape),
                },
            )
        if not torch.isfinite(attention_map).all():
            raise FrozenMaskHeadRuntimeError("Residual attention map contains NaN/Inf.")
        gate_map = uncertainty_gate_map * attention_map
    residual_contribution_logits = gate_map * effective_residual_scale * residual_logits
    final_logits = coarse_logits + residual_contribution_logits
    if not torch.isfinite(final_logits).all():
        raise FrozenMaskHeadRuntimeError("Residual-head combination produced NaN/Inf logits.")
    return ResidualHeadCombinationResult(
        coarse_logits=coarse_logits,
        residual_logits=residual_logits,
        final_logits=final_logits,
        residual_contribution_logits=residual_contribution_logits,
        gate_map=gate_map,
        attention_map=attention_map,
        learned_residual_scale=learned_residual_scale,
        effective_residual_scale=effective_residual_scale,
        residual_scale_source=residual_scale_source,
        residual_alpha_override=(None if residual_alpha_override is None else float(residual_alpha_override)),
        residual_gate_mode=gate_mode,
        residual_gate_threshold=gate_threshold,
        residual_runtime_multiplier=residual_runtime_multiplier,
        attention_bank_means=(None if output.attention_bank_means is None else dict(output.attention_bank_means)),
        attention_real_mean=output.attention_real_mean,
        attention_null_mean=output.attention_null_mean,
        attention_fpn1_mean=output.attention_fpn1_mean,
        attention_fpn0_mean=output.attention_fpn0_mean,
        residual_scale_fpn1=output.residual_scale_fpn1,
        residual_scale_fpn0=output.residual_scale_fpn0,
        alpha_fpn1=output.alpha_fpn1,
        alpha_fpn0=output.alpha_fpn0,
    )


def summarize_residual_head_combination(
    combination: ResidualHeadCombinationResult,
    *,
    boundary_mask: Any | None = None,
) -> dict[str, Any]:
    """Summarize coarse-vs-final residual behavior at image resolution."""

    import torch

    coarse_logits = combination.coarse_logits.detach()
    final_logits = combination.final_logits.detach()
    residual_contribution_logits = combination.residual_contribution_logits.detach()
    gate_map = combination.gate_map.detach()
    gated_residual_logits = gate_map * combination.residual_logits.detach()
    sign_flip_fraction = float(((coarse_logits >= 0) != (final_logits >= 0)).float().mean().cpu().item())
    learned_attention_map = combination.attention_map
    summary: dict[str, Any] = {
        "learned_residual_scale": float(combination.learned_residual_scale),
        "effective_residual_scale": float(combination.effective_residual_scale),
        "residual_scale_source": combination.residual_scale_source,
        "residual_runtime_multiplier": float(combination.residual_runtime_multiplier),
        "residual_alpha_override": combination.residual_alpha_override,
        "residual_gate_mode": combination.residual_gate_mode,
        "residual_gate_threshold": combination.residual_gate_threshold,
        "residual_gate_mean": float(gate_map.float().mean().cpu().item()),
        "residual_gate_fraction_ge_half": float((gate_map >= 0.5).float().mean().cpu().item()),
        "mean_abs_residual_logits": float(combination.residual_logits.detach().abs().mean().cpu().item()),
        "mean_abs_gated_residual_logits": float(gated_residual_logits.abs().mean().cpu().item()),
        "mean_abs_residual_contribution": float(residual_contribution_logits.abs().mean().cpu().item()),
        "residual_sign_flip_fraction": sign_flip_fraction,
        "final_minus_coarse_mean_abs": float((final_logits - coarse_logits).abs().mean().cpu().item()),
        "attention_mean": None,
        "attention_fraction_ge_half": None,
        "attention_real_mean": combination.attention_real_mean,
        "attention_null_mean": combination.attention_null_mean,
        "attention_fpn1_mean": (
            combination.attention_fpn1_mean
            if combination.attention_fpn1_mean is not None
            else (None if combination.attention_bank_means is None else combination.attention_bank_means.get("fpn_1"))
        ),
        "attention_fpn0_mean": (
            combination.attention_fpn0_mean
            if combination.attention_fpn0_mean is not None
            else (None if combination.attention_bank_means is None else combination.attention_bank_means.get("fpn_0"))
        ),
        "residual_scale_fpn1": combination.residual_scale_fpn1,
        "residual_scale_fpn0": combination.residual_scale_fpn0,
        "alpha_fpn1": combination.alpha_fpn1,
        "alpha_fpn0": combination.alpha_fpn0,
        "boundary_attention_mean": None,
        "non_boundary_attention_mean": None,
    }
    if learned_attention_map is not None:
        attention_tensor = learned_attention_map.detach()
        summary["attention_mean"] = float(attention_tensor.float().mean().cpu().item())
        summary["attention_fraction_ge_half"] = float((attention_tensor >= 0.5).float().mean().cpu().item())
    if boundary_mask is not None:
        boundary_tensor = torch.as_tensor(boundary_mask, dtype=torch.bool, device=final_logits.device)
        if boundary_tensor.ndim == 2:
            boundary_tensor = boundary_tensor.unsqueeze(0).unsqueeze(0)
        elif boundary_tensor.ndim == 3:
            boundary_tensor = boundary_tensor.unsqueeze(0)
        if boundary_tensor.ndim != 4 or tuple(int(value) for value in boundary_tensor.shape) != tuple(
            int(value) for value in final_logits.shape
        ):
            raise FrozenMaskHeadRuntimeError(
                "Residual boundary diagnostics require a boundary mask with the same [1,1,H,W] shape as the logits.",
                diagnostics={
                    "logit_shape": tuple(int(value) for value in final_logits.shape),
                    "boundary_shape": tuple(int(value) for value in boundary_tensor.shape),
                },
            )
        boundary_mask_bool = boundary_tensor.bool()
        non_boundary_mask_bool = torch.logical_not(boundary_mask_bool)
        if bool(boundary_mask_bool.any()):
            summary["boundary_mean_abs_residual_contribution"] = float(
                residual_contribution_logits.abs()[boundary_mask_bool].mean().cpu().item()
            )
            if learned_attention_map is not None:
                summary["boundary_attention_mean"] = float(attention_tensor.float()[boundary_mask_bool].mean().cpu().item())
        else:
            summary["boundary_mean_abs_residual_contribution"] = None
            summary["boundary_attention_mean"] = None
        if bool(non_boundary_mask_bool.any()):
            summary["non_boundary_mean_abs_residual_contribution"] = float(
                residual_contribution_logits.abs()[non_boundary_mask_bool].mean().cpu().item()
            )
            if learned_attention_map is not None:
                summary["non_boundary_attention_mean"] = float(attention_tensor.float()[non_boundary_mask_bool].mean().cpu().item())
        else:
            summary["non_boundary_mean_abs_residual_contribution"] = None
            summary["non_boundary_attention_mean"] = None
    else:
        summary["boundary_mean_abs_residual_contribution"] = None
        summary["non_boundary_mean_abs_residual_contribution"] = None
        summary["boundary_attention_mean"] = None
        summary["non_boundary_attention_mean"] = None
    return summary


def build_residual_head_settings_payload(
    args,
    *,
    include_eval_alpha_override: bool,
) -> dict[str, Any]:
    """Return the explicit residual-head configuration knobs for configs and checkpoints."""

    if not is_coarse_plus_residual_variant(str(getattr(args, "variant", ""))):
        return {}
    payload = {
        "residual_gate_mode": resolve_residual_head_gate_mode(getattr(args, "residual_gate_mode", "none")),
        "residual_gate_threshold": resolve_residual_head_gate_threshold(
            getattr(args, "residual_gate_threshold", RESIDUAL_HEAD_DEFAULT_GATE_THRESHOLD)
        ),
        "coarse_loss_weight": resolve_residual_head_coarse_loss_weight(args),
        "residual_l1_weight": float(getattr(args, "residual_l1_weight", 0.0)),
        "attention_sparsity_weight": float(getattr(args, "attention_sparsity_weight", 0.0)),
        "attention_hidden_dim": int(getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])),
        "cross_attn_query_stride": int(getattr(args, "cross_attn_query_stride", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["cross_attn_query_stride"])),
        "joker_disable_fpn0": bool(getattr(args, "joker_disable_fpn0", False)),
        "joker_disable_fpn1": bool(getattr(args, "joker_disable_fpn1", False)),
        "joker_disable_null_token": bool(getattr(args, "joker_disable_null_token", False)),
        "joker_use_learned_gate": bool(getattr(args, "joker_use_learned_gate", False)),
        "joker_zero_init_residual_scale": bool(getattr(args, "joker_zero_init_residual_scale", False)),
        "joker_zero_init_attn_qkv": bool(getattr(args, "joker_zero_init_attn_qkv", False)),
        "joker_residual_warmup_epochs": int(getattr(args, "joker_residual_warmup_epochs", 0)),
        "joker_residual_ramp_epochs": int(getattr(args, "joker_residual_ramp_epochs", 0)),
    }
    if include_eval_alpha_override:
        residual_alpha_override = getattr(args, "residual_alpha_override", None)
        payload["residual_alpha_override"] = (
            None if residual_alpha_override is None else float(residual_alpha_override)
        )
    return payload


def compute_residual_head_training_loss(
    output: FrozenResidualMaskHeadOutput,
    target_foreground_mask: Any,
    *,
    bce_weight: float,
    dice_weight: float,
    boundary_mask: Any | None = None,
    boundary_weight: float = 0.0,
    coarse_loss_weight: float = 0.0,
    residual_l1_weight: float = 0.0,
    attention_sparsity_weight: float = 0.0,
) -> ResidualHeadTrainingLossResult:
    """Compute the final residual-head loss plus optional coarse supervision and residual L1."""

    import torch

    if not isinstance(output, FrozenResidualMaskHeadOutput):
        raise FrozenMaskHeadRuntimeError(
            "compute_residual_head_training_loss requires a FrozenResidualMaskHeadOutput."
        )
    final_loss_result = compute_bce_dice_loss(
        output.logits,
        target_foreground_mask,
        bce_weight=float(bce_weight),
        dice_weight=float(dice_weight),
        boundary_mask=boundary_mask,
        boundary_weight=float(boundary_weight),
    )
    total_loss = final_loss_result.loss
    coarse_loss_result: MaskHeadLossResult | None = None
    if float(coarse_loss_weight) > 0.0:
        coarse_loss_result = compute_bce_dice_loss(
            output.coarse_logits,
            target_foreground_mask,
            bce_weight=float(bce_weight),
            dice_weight=float(dice_weight),
            boundary_mask=boundary_mask,
            boundary_weight=float(boundary_weight),
        )
        total_loss = total_loss + float(coarse_loss_weight) * coarse_loss_result.loss
    residual_l1_loss = None
    if float(residual_l1_weight) > 0.0:
        residual_l1_loss = float(residual_l1_weight) * output.residual_logits.abs().mean()
        total_loss = total_loss + residual_l1_loss
    attention_sparsity_loss = None
    if float(attention_sparsity_weight) > 0.0:
        if output.attention_map is None:
            raise FrozenMaskHeadRuntimeError(
                "attention_sparsity_weight requires a residual-head output with attention_map."
            )
        attention_sparsity_loss = float(attention_sparsity_weight) * output.attention_map.mean()
        total_loss = total_loss + attention_sparsity_loss
    if not torch.isfinite(total_loss):
        raise FrozenMaskHeadRuntimeError("Residual-head training produced a non-finite total loss.")
    return ResidualHeadTrainingLossResult(
        total_loss=total_loss,
        final_loss_result=final_loss_result,
        coarse_loss_result=coarse_loss_result,
        residual_l1_loss=residual_l1_loss,
        attention_sparsity_loss=attention_sparsity_loss,
        coarse_loss_weight=float(coarse_loss_weight),
        residual_l1_weight=float(residual_l1_weight),
        attention_sparsity_weight=float(attention_sparsity_weight),
    )


def is_residual_head_variant(variant: str) -> bool:
    return is_coarse_plus_residual_variant(str(variant))
