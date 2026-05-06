"""Minimal frozen SAM multiscale heads kept for the current texture project.

Phase-1 cleanup scope:
- keep the simple coarse-only baseline (`fpn_2_only`)
- keep the simple coarse + `fpn_1` residual refiner (`fpn_2_plus_fpn_1_refine`)
- keep the serialized progressive refiner (`fpn_2_then_fpn_1_then_fpn_0_attn_refine`)
- drop legacy GlaS probes, memory-attention branches, cross-attention refine,
  and multibank/null-token refine from this minimal module

This file is intentionally not a drop-in replacement for every historical experiment.
It is a trimmed module meant to make the currently relevant model surface explicit.
"""

from __future__ import annotations

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

FROZEN_MASK_HEAD_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "fpn_2_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "summary": "Single-scale frozen mask head using only the coarsest SAM level `fpn_2`.",
    },
    "fpn_2_plus_fpn_1_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1"),
        "summary": "Coarse-only frozen mask head on `fpn_2` plus a tiny `fpn_1` residual-refinement branch that predicts an additive foreground-logit correction.",
    },
    "fpn_2_then_fpn_1_then_fpn_0_attn_refine": {
        "selected_level_policy": "named_level_subset",
        "requested_level_names": ("fpn_2", "fpn_1", "fpn_0"),
        "head_family": "serialized_progressive_refinement_mask_head",
        "summary": "Coarse-first frozen mask head on `fpn_2` followed by a serialized chain of refinement stages using `fpn_1` and `fpn_0` features with progressive warmup.",
    },
}

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
    }


def is_serialized_refine_variant(variant: str) -> bool:
    return str(variant) == "fpn_2_then_fpn_1_then_fpn_0_attn_refine"


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
    raise FrozenMaskHeadRuntimeError(
        f"Variant '{variant_name}' is not a supported coarse-plus-residual frozen mask head.",
        diagnostics={
            "supported_variants": [
                "fpn_2_plus_fine_residual",
                "fpn_2_plus_fpn_1_refine",
                "fpn_2_plus_fpn_1_attn_refine",
                "fpn_2_plus_fpn_1_cross_attn_refine",
                "fpn_2_plus_multibank_null_refine",
            ]
        },
    )



def is_coarse_plus_residual_variant(variant: str) -> bool:
    return str(variant) in {
        "fpn_2_plus_fpn_1_refine",
        "fpn_2_then_fpn_1_then_fpn_0_attn_refine",
    }


def is_serialized_refine_variant(variant: str) -> bool:
    return str(variant) == "fpn_2_then_fpn_1_then_fpn_0_attn_refine"


def resolve_frozen_mask_head_head_family(variant: str) -> str:
    try:
        variant_spec = FROZEN_MASK_HEAD_VARIANT_SPECS[str(variant)]
    except KeyError as exc:
        expected = ", ".join(sorted(FROZEN_MASK_HEAD_VARIANT_SPECS))
        raise FrozenMaskHeadRuntimeError(
            f"Unknown frozen mask-head variant '{variant}'. Expected one of: {expected}."
        ) from exc
    return str(variant_spec.get("head_family", "multiscale_mask_head"))


def resolve_coarse_plus_residual_levels(variant: str) -> tuple[str, str]:
    variant_name = str(variant)
    if variant_name == "fpn_2_plus_fpn_1_refine":
        return ("fpn_2", "fpn_1")
    if variant_name == "fpn_2_then_fpn_1_then_fpn_0_attn_refine":
        return ("fpn_2", "fpn_0")
    raise FrozenMaskHeadRuntimeError(
        f"Variant '{variant_name}' is not a supported coarse-plus-residual frozen mask head.",
        diagnostics={
            "supported_variants": [
                "fpn_2_plus_fpn_1_refine",
                "fpn_2_then_fpn_1_then_fpn_0_attn_refine",
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
        "coarse_loss_weight": float(getattr(args, "coarse_loss_weight", 0.0) or 0.0),
        "residual_l1_weight": float(getattr(args, "residual_l1_weight", 0.0)),
        "attention_sparsity_weight": float(getattr(args, "attention_sparsity_weight", 0.0)),
        "attention_hidden_dim": int(
            getattr(args, "attention_hidden_dim", FROZEN_MASK_HEAD_ATTN_REFINEMENT_SETTINGS["attention_hidden_dim"])
        ),
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
