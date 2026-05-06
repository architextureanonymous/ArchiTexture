"""Few-shot linear probing on frozen coarse-vs-fine SAM multiscale features.

This module adds a deliberately simple supervised baseline on top of the same
frozen aligned SAM pyramid features used by the Stage-2 scale-gated probe:

- SAM stays frozen.
- Selected main pyramid levels are aligned to a common coarsest grid.
- The selected levels are concatenated channel-wise.
- A single learnable `1x1` classifier predicts the binary texture partition.

The intent is a classic linear-probe baseline rather than another learned
embedding-plus-clustering experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import (
    ProbePartitionResult,
    Sam3CoarseVsFineScaleProbeRuntimeError,
    evaluate_probe_partition,
)


COARSE_VS_FINE_LINEAR_PROBE_SETTINGS: dict[str, float | int | str | bool] = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 5,
    "post_concat_normalization": True,
}


LINEAR_PROBE_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "concat_all_scales": {
        "selected_level_policy": "all_scales",
        "summary": "Concatenate all discovered aligned SAM pyramid levels and fit one linear 1x1 classifier.",
    },
    "concat_coarse_plus_next_finer": {
        "selected_level_policy": "coarse_plus_next_finer",
        "summary": "Concatenate only the coarsest level and the next finer level, then fit one linear 1x1 classifier.",
    },
    "fpn_2_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "summary": "Single-scale linear probe using only the coarsest SAM level `fpn_2`.",
    },
    "fpn_1_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_1",
        "summary": "Single-scale linear probe using only the middle SAM level `fpn_1`.",
    },
    "fpn_0_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_0",
        "summary": "Single-scale linear probe using only the finest SAM level `fpn_0`.",
    },
}


@dataclass(frozen=True)
class LinearProbeOutput:
    """Forward-pass outputs from the few-shot linear probe."""

    logits: Any
    concatenated_features: Any


@dataclass(frozen=True)
class LinearProbeLossResult:
    """Scalar supervised loss plus explicit valid-pixel diagnostics."""

    loss: Any
    num_valid_pixels: int
    present_texture_labels: tuple[int, ...]


class LinearSegmentationProbeModule:
    """One-layer linear segmentation probe over concatenated aligned SAM features."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        post_concat_normalization: bool = True,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_names = tuple(level_input_dims.keys())
                self.level_input_dims = {name: int(value) for name, value in level_input_dims.items()}
                self.total_input_dim = int(sum(self.level_input_dims.values()))
                self.post_concat_normalization = bool(post_concat_normalization)
                self.classifier = nn.Conv2d(self.total_input_dim, 2, kernel_size=1, bias=True)
                self._level_channel_slices: dict[str, tuple[int, int]] = {}
                channel_start = 0
                for level_name in self.level_names:
                    channel_stop = channel_start + int(self.level_input_dims[level_name])
                    self._level_channel_slices[level_name] = (channel_start, channel_stop)
                    channel_start = channel_stop

            def current_level_weight_norms(self) -> dict[str, float]:
                import torch

                weights = self.classifier.weight.detach()
                norms: dict[str, float] = {}
                for level_name, (start, stop) in self._level_channel_slices.items():
                    level_weights = weights[:, start:stop, :, :]
                    norms[level_name] = float(torch.linalg.vector_norm(level_weights).cpu().item())
                return norms

            def forward(self, feature_levels: Mapping[str, Any]) -> LinearProbeOutput:
                import torch
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise Sam3CoarseVsFineScaleProbeRuntimeError(
                        "LinearSegmentationProbeModule got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )
                tensors: list[Any] = []
                reference_hw: tuple[int, int] | None = None
                for level_name in self.level_names:
                    tensor = feature_levels[level_name]
                    if tensor.ndim != 4:
                        raise Sam3CoarseVsFineScaleProbeRuntimeError(
                            f"LinearSegmentationProbeModule expected [B,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
                        )
                    current_hw = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                    if reference_hw is None:
                        reference_hw = current_hw
                    elif current_hw != reference_hw:
                        raise Sam3CoarseVsFineScaleProbeRuntimeError(
                            "LinearSegmentationProbeModule received feature maps with mismatched aligned grids.",
                            diagnostics={
                                "expected_hw": reference_hw,
                                "received_level": level_name,
                                "received_hw": current_hw,
                            },
                        )
                    tensors.append(tensor)
                concatenated = torch.cat(tensors, dim=1)
                if self.post_concat_normalization:
                    concatenated = F.normalize(concatenated, dim=1, eps=1e-6)
                if not torch.isfinite(concatenated).all():
                    raise Sam3CoarseVsFineScaleProbeRuntimeError(
                        "LinearSegmentationProbeModule produced NaN/Inf concatenated features."
                    )
                logits = self.classifier(concatenated)
                if not torch.isfinite(logits).all():
                    raise Sam3CoarseVsFineScaleProbeRuntimeError(
                        "LinearSegmentationProbeModule produced NaN/Inf logits."
                    )
                return LinearProbeOutput(logits=logits, concatenated_features=concatenated)

        self.module = _Module()


def resolve_linear_probe_variant_levels(
    *,
    variant: str,
    available_level_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve the ordered level list for one linear-probe run variant."""

    try:
        variant_spec = LINEAR_PROBE_VARIANT_SPECS[variant]
    except KeyError as exc:
        expected = ", ".join(sorted(LINEAR_PROBE_VARIANT_SPECS))
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Unknown coarse-vs-fine linear-probe variant '{variant}'. Expected one of: {expected}."
        ) from exc

    available = tuple(str(name) for name in available_level_names)
    policy = str(variant_spec["selected_level_policy"])
    if policy == "all_scales":
        return available
    if policy == "coarse_plus_next_finer":
        if len(available) < 2:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Linear-probe coarse-plus-next-finer runs require at least two discovered SAM levels.",
                diagnostics={"available_level_names": available},
            )
        return available[:2]
    if policy == "single_level_by_name":
        requested = str(variant_spec["requested_level_name"])
        if requested not in available:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "Linear-probe single-scale variant requested a level that was not discovered.",
                diagnostics={"requested_level_name": requested, "available_level_names": available},
            )
        return (requested,)
    raise Sam3CoarseVsFineScaleProbeRuntimeError(
        f"Unsupported linear-probe selected-level policy '{policy}'."
    )


def build_linear_probe_targets(coarsest_grid_labels: Any, *, device: Any) -> Any:
    """Map coarsest-grid labels {0,1,2} to CE targets {ignore,0,1}."""

    import torch

    labels = torch.as_tensor(coarsest_grid_labels, dtype=torch.long, device=device)
    if labels.ndim != 2:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Linear probe expected 2D coarsest-grid labels, got {tuple(labels.shape)}."
        )
    targets = torch.full_like(labels, fill_value=255)
    targets = torch.where(labels == 1, torch.zeros_like(targets), targets)
    targets = torch.where(labels == 2, torch.ones_like(targets), targets)
    valid_pixels = int((targets != 255).sum().item())
    if valid_pixels < 1:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Linear probe found no valid texture-region pixels on the coarsest grid.",
            diagnostics={"unique_labels": tuple(int(value) for value in torch.unique(labels).tolist())},
        )
    return targets


def compute_linear_probe_loss(logits: Any, coarsest_grid_labels: Any) -> LinearProbeLossResult:
    """Compute per-pixel cross-entropy on the coarsest aligned grid."""

    import torch
    import torch.nn.functional as F

    if logits.ndim != 4 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 2:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Linear probe loss expects logits [1,2,H,W], got {tuple(logits.shape)}."
        )
    targets = build_linear_probe_targets(coarsest_grid_labels, device=logits.device)
    if tuple(targets.shape) != (int(logits.shape[-2]), int(logits.shape[-1])):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Linear probe loss got GT labels that do not match the coarsest feature grid.",
            diagnostics={
                "logit_hw": (int(logits.shape[-2]), int(logits.shape[-1])),
                "label_shape": tuple(int(value) for value in targets.shape),
            },
        )
    loss = F.cross_entropy(logits, targets.unsqueeze(0), ignore_index=255)
    if not torch.isfinite(loss):
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Linear probe loss became NaN/Inf.")
    present_texture_labels = tuple(
        int(value) for value in np.unique(np.asarray(coarsest_grid_labels, dtype=np.int32)) if int(value) in {1, 2}
    )
    return LinearProbeLossResult(
        loss=loss,
        num_valid_pixels=int((targets != 255).sum().item()),
        present_texture_labels=present_texture_labels,
    )


def predict_linear_probe_label_map(logits: Any) -> np.ndarray:
    """Convert linear-probe logits [1,2,H,W] into a binary class map [H,W]."""

    import torch

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim != 4 or int(logits.shape[0]) != 1 or int(logits.shape[1]) != 2:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Linear probe prediction expects logits [1,2,H,W], got {tuple(logits.shape)}."
        )
    if not torch.isfinite(logits).all():
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Linear probe prediction received NaN/Inf logits.")
    return np.asarray(torch.argmax(logits, dim=1)[0].detach().cpu().numpy(), dtype=np.int32)


def evaluate_linear_probe_partition(
    *,
    sample: Any,
    coarsest_grid_label_map: np.ndarray,
) -> ProbePartitionResult:
    """Evaluate the direct argmax linear-probe prediction permutation-invariantly."""

    unique_values = tuple(int(value) for value in np.unique(coarsest_grid_label_map))
    if any(value not in {0, 1} for value in unique_values):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Linear probe evaluation expected a binary class map with labels {0,1}.",
            diagnostics={"unique_labels": unique_values, "crop_name": sample.crop_name},
        )
    return evaluate_probe_partition(sample=sample, coarsest_grid_label_map=np.asarray(coarsest_grid_label_map, dtype=np.int32))
