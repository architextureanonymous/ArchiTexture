"""Tiny learned coarse-vs-fine probe on frozen SAM multiscale features.

This module implements Stage 2 of the coarse-vs-fine SAM-scale study:

- SAM stays frozen.
- The main ``backbone_fpn`` feature pyramid is extracted once per image.
- Selected levels are aligned to the coarsest selected grid.
- A tiny global scale-gated probe projects and fuses the aligned levels.
- The learned per-pixel embeddings are clustered with 2-way cosine k-means.

The implementation is intentionally small and explicit so the learned scale
weights remain interpretable and Stage 3 can later add leave-one-out and lesion
studies without refactoring the frozen-feature pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import label_map_from_regions
from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS,
    Sam3FeatureClusterCoarseToFineGlobalRunner,
)
from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID
from rwtd_sam3.utils.deterministic_clustering import deterministic_kmeans


LOGGER = logging.getLogger(__name__)

COARSE_VS_FINE_STAGE2_SETTINGS: dict[str, float | int | str] = {
    "feature_source": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS["feature_source"],
    "feature_alignment_mode": "bilinear",
    "per_level_normalization": "l2",
    "post_fusion_normalization": "l2",
    "projection_dim": 64,
    "embedding_dim": 32,
    "pair_samples_per_image": 512,
    "affinity_temperature": 0.10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_epochs": 5,
    "kmeans_metric": "cosine",
    "kmeans_num_clusters": 2,
    "kmeans_max_iterations": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS["kmeans_max_iterations"],
    "kmeans_convergence_tolerance": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS[
        "kmeans_convergence_tolerance"
    ],
}

STAGE2_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "learned_global_gates_all_scales": {
        "selected_level_policy": "all_scales",
        "learnable_gates": True,
        "summary": "Learn one global softmax scale gate over all discovered main SAM pyramid levels.",
    },
    "fixed_uniform_gates_all_scales": {
        "selected_level_policy": "all_scales",
        "learnable_gates": False,
        "summary": "Use the same tiny probe, but keep scale weights fixed uniform across all discovered levels.",
    },
    "learned_global_gates_coarse_plus_next_finer": {
        "selected_level_policy": "coarse_plus_next_finer",
        "learnable_gates": True,
        "summary": "Learn one global softmax scale gate using only the coarsest level and the next finer level.",
    },
    "fixed_uniform_gates_coarse_plus_next_finer": {
        "selected_level_policy": "coarse_plus_next_finer",
        "learnable_gates": False,
        "summary": "Use the same two-level tiny probe on the coarsest level plus the next finer level, but keep scale weights fixed uniform.",
    },
    "fpn_2_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_2",
        "learnable_gates": False,
        "summary": "Single-scale Stage-2 control using only the coarsest SAM pyramid level `fpn_2`.",
    },
    "fpn_1_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_1",
        "learnable_gates": False,
        "summary": "Single-scale Stage-2 control using only the middle SAM pyramid level `fpn_1`.",
    },
    "fpn_0_only": {
        "selected_level_policy": "single_level_by_name",
        "requested_level_name": "fpn_0",
        "learnable_gates": False,
        "summary": "Single-scale Stage-2 control using only the finest SAM pyramid level `fpn_0`.",
    },
}


class Sam3CoarseVsFineScaleProbeRuntimeError(RuntimeError):
    """Raised when the Stage-2 coarse-vs-fine probe path becomes invalid."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class CoarseVsFineScaleSample:
    """Frozen aligned multiscale SAM features plus the coarsest-grid GT partition."""

    sample: Any
    raw_level_names: tuple[str, ...]
    raw_level_shapes: dict[str, tuple[int, int, int]]
    selected_level_names: tuple[str, ...]
    coarsest_level_name: str
    coarsest_grid_size: tuple[int, int]
    aligned_level_shapes: dict[str, tuple[int, int, int]]
    aligned_feature_levels: dict[str, np.ndarray]
    coarsest_grid_labels: np.ndarray


@dataclass(frozen=True)
class ScaleGatedProbeOutput:
    """Forward-pass outputs from the tiny global scale-gated probe."""

    embedding: Any
    gate_logits: Any
    gate_weights: Any
    projected_levels: dict[str, Any]


@dataclass(frozen=True)
class AffinityLossResult:
    """Scalar loss plus explicit pair-sampling diagnostics."""

    loss: Any
    num_same_pairs: int
    num_different_pairs: int
    mean_same_similarity: float
    mean_different_similarity: float


@dataclass(frozen=True)
class ProbePartitionResult:
    """One evaluated partition built from learned coarsest-grid embeddings."""

    coarsest_grid_label_map: np.ndarray
    image_space_label_map: np.ndarray
    prediction_a: np.ndarray
    prediction_b: np.ndarray
    assignment_used: str
    direct_eval_miou: float
    direct_eval_ari: float
    swapped_eval_miou: float
    swapped_eval_ari: float
    eval_miou: float
    eval_ari: float


class ScaleGatedSamProbeModule:
    """Small interpretable scale-gated probe over frozen aligned SAM feature maps."""

    def __init__(
        self,
        *,
        level_input_dims: Mapping[str, int],
        projection_dim: int,
        embedding_dim: int,
        learnable_gates: bool,
    ) -> None:
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.level_names = tuple(level_input_dims.keys())
                self.projections = nn.ModuleDict(
                    {
                        level_name: nn.Conv2d(int(input_dim), int(projection_dim), kernel_size=1, bias=True)
                        for level_name, input_dim in level_input_dims.items()
                    }
                )
                if int(embedding_dim) == int(projection_dim):
                    self.output_head = nn.Identity()
                else:
                    self.output_head = nn.Conv2d(
                        int(projection_dim),
                        int(embedding_dim),
                        kernel_size=1,
                        bias=True,
                    )
                self.learnable_gates = bool(learnable_gates)
                if self.learnable_gates:
                    self.gate_logits = nn.Parameter(torch.zeros((len(self.level_names),), dtype=torch.float32))
                else:
                    self.register_buffer(
                        "fixed_gate_logits",
                        torch.zeros((len(self.level_names),), dtype=torch.float32),
                        persistent=True,
                    )

            def current_gate_logits(self):
                return self.gate_logits if self.learnable_gates else self.fixed_gate_logits

            def current_gate_weights(self):
                if self.learnable_gates:
                    return torch.softmax(self.current_gate_logits(), dim=0)
                count = len(self.level_names)
                return torch.full(
                    (count,),
                    fill_value=1.0 / float(count),
                    dtype=torch.float32,
                    device=self.current_gate_logits().device,
                )

            def forward(self, feature_levels: Mapping[str, Any]) -> ScaleGatedProbeOutput:
                import torch.nn.functional as F

                missing = [name for name in self.level_names if name not in feature_levels]
                extra = [name for name in feature_levels if name not in self.level_names]
                if missing or extra:
                    raise Sam3CoarseVsFineScaleProbeRuntimeError(
                        "ScaleGatedSamProbeModule got mismatched feature-level keys.",
                        diagnostics={"missing_levels": missing, "extra_levels": extra},
                    )

                weights = self.current_gate_weights()
                projected_levels: dict[str, Any] = {}
                fused = None
                reference_hw: tuple[int, int] | None = None
                for level_index, level_name in enumerate(self.level_names):
                    feature_tensor = feature_levels[level_name]
                    if feature_tensor.ndim != 4:
                        raise Sam3CoarseVsFineScaleProbeRuntimeError(
                            f"ScaleGatedSamProbeModule expected [B,C,H,W] for {level_name}, got {tuple(feature_tensor.shape)}."
                        )
                    if reference_hw is None:
                        reference_hw = (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1]))
                    elif reference_hw != (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])):
                        raise Sam3CoarseVsFineScaleProbeRuntimeError(
                            "ScaleGatedSamProbeModule received feature maps with mismatched aligned grids.",
                            diagnostics={
                                "expected_hw": reference_hw,
                                "received_level": level_name,
                                "received_hw": (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
                            },
                        )
                    projected = self.projections[level_name](feature_tensor)
                    if not torch.isfinite(projected).all():
                        raise Sam3CoarseVsFineScaleProbeRuntimeError(
                            f"ScaleGatedSamProbeModule produced NaN/Inf values after projecting {level_name}."
                        )
                    projected_levels[level_name] = projected
                    weighted = projected * weights[level_index].view(1, 1, 1, 1)
                    fused = weighted if fused is None else (fused + weighted)

                embedding = self.output_head(fused)
                embedding = F.normalize(embedding, dim=1, eps=1e-6)
                if not torch.isfinite(embedding).all():
                    raise Sam3CoarseVsFineScaleProbeRuntimeError(
                        "ScaleGatedSamProbeModule produced NaN/Inf values after output normalization."
                    )
                return ScaleGatedProbeOutput(
                    embedding=embedding,
                    gate_logits=self.current_gate_logits(),
                    gate_weights=weights,
                    projected_levels=projected_levels,
                )

        self.module = _Module()


class Sam3CoarseVsFineScaleFeatureExtractor:
    """Frozen SAM feature extractor that reuses the Stage-1 multiscale path."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.settings = dict(COARSE_VS_FINE_STAGE2_SETTINGS)
        if settings is not None:
            self.settings.update(settings)
        stage1_settings = {
            "feature_source": self.settings["feature_source"],
            "kmeans_max_iterations": self.settings["kmeans_max_iterations"],
            "kmeans_convergence_tolerance": self.settings["kmeans_convergence_tolerance"],
            "boundary_band_radius": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS["boundary_band_radius"],
            "refinement_iterations_per_level": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS[
                "refinement_iterations_per_level"
            ],
            "refine_confidence_threshold": FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS["refine_confidence_threshold"],
        }
        self._stage1_runner = Sam3FeatureClusterCoarseToFineGlobalRunner(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=stage1_settings,
        )
        self._logged_pyramid_signature = False

    def extract_sam_pyramid(self, image: Image.Image) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
        """Return the normalized main SAM pyramid levels at native resolutions."""

        official_backend = self._stage1_runner._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._stage1_runner._prepare_feature_levels_with_backend(
            image,
            official_backend,
        )
        pyramid = {
            level.name: np.asarray(level.feature_map.detach().cpu().numpy(), dtype=np.float32)
            for level in feature_levels
        }
        if not self._logged_pyramid_signature:
            signature = {
                level.name: [
                    int(level.feature_map.shape[0]),
                    int(level.feature_map.shape[1]),
                    int(level.feature_map.shape[2]),
                ]
                for level in feature_levels
            }
            LOGGER.info("Coarse-vs-fine Stage 2 discovered SAM pyramid levels: %s", signature)
            self._logged_pyramid_signature = True
        return image_size, pyramid

    def prepare_sample(
        self,
        sample: Any,
        *,
        selected_level_names: Sequence[str] | None = None,
        alignment_reference_level_names: Sequence[str] | None = None,
    ) -> CoarseVsFineScaleSample:
        """Extract, align, and package one sample for Stage-2 training or eval."""

        _, pyramid = self.extract_sam_pyramid(sample.image)
        raw_level_names = tuple(pyramid.keys())
        raw_level_shapes = {
            level_name: tuple(int(value) for value in feature_map.shape)
            for level_name, feature_map in pyramid.items()
        }
        resolved_selected_names = resolve_selected_level_names(
            available_level_names=raw_level_names,
            selected_level_names=selected_level_names,
        )
        aligned_levels, coarsest_level_name, coarsest_grid_size = align_feature_levels_to_coarsest(
            feature_levels=pyramid,
            selected_level_names=resolved_selected_names,
            interpolation_mode=str(self.settings["feature_alignment_mode"]),
            reference_level_names=alignment_reference_level_names,
        )
        normalized_levels = l2_normalize_feature_levels(aligned_levels)
        aligned_level_shapes = {
            level_name: tuple(int(value) for value in feature_map.shape)
            for level_name, feature_map in normalized_levels.items()
        }
        coarsest_labels = build_coarsest_grid_ground_truth(
            sample=sample,
            target_size=coarsest_grid_size,
        )
        return CoarseVsFineScaleSample(
            sample=sample,
            raw_level_names=raw_level_names,
            raw_level_shapes=raw_level_shapes,
            selected_level_names=resolved_selected_names,
            coarsest_level_name=coarsest_level_name,
            coarsest_grid_size=coarsest_grid_size,
            aligned_level_shapes=aligned_level_shapes,
            aligned_feature_levels=normalized_levels,
            coarsest_grid_labels=coarsest_labels,
        )


def resolve_selected_level_names(
    *,
    available_level_names: Sequence[str],
    selected_level_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve an ordered subset of feature levels and validate availability."""

    available = tuple(str(name) for name in available_level_names)
    if not available:
        raise Sam3CoarseVsFineScaleProbeRuntimeError("No SAM pyramid levels were available.")
    if selected_level_names is None:
        return available
    requested = tuple(str(name) for name in selected_level_names)
    missing = [name for name in requested if name not in available]
    if missing:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Requested feature levels are missing from the discovered SAM pyramid.",
            diagnostics={"requested_levels": requested, "available_levels": available, "missing_levels": missing},
        )
    return requested


def align_feature_levels_to_coarsest(
    *,
    feature_levels: Mapping[str, np.ndarray],
    selected_level_names: Sequence[str],
    interpolation_mode: str,
    reference_level_names: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], str, tuple[int, int]]:
    """Resize selected feature levels to the requested coarsest reference grid."""

    import torch

    if interpolation_mode not in {"nearest", "bilinear", "area"}:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Unsupported feature alignment mode '{interpolation_mode}'."
        )
    selected = {name: np.asarray(feature_levels[name], dtype=np.float32) for name in selected_level_names}
    for level_name, feature_map in selected.items():
        if feature_map.ndim != 3:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"Expected aligned Stage-2 feature map [C,H,W] for {level_name}, got {feature_map.shape}."
            )
    reference_names = tuple(reference_level_names or selected_level_names)
    if not reference_names:
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Stage-2 feature alignment received no reference levels.")
    missing_reference_names = [name for name in reference_names if name not in feature_levels]
    if missing_reference_names:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Stage-2 feature alignment requested reference levels that were missing from the SAM pyramid.",
            diagnostics={
                "missing_reference_level_names": tuple(missing_reference_names),
                "available_level_names": tuple(feature_levels.keys()),
            },
        )
    reference_levels = {name: np.asarray(feature_levels[name], dtype=np.float32) for name in reference_names}
    for level_name, feature_map in reference_levels.items():
        if feature_map.ndim != 3:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"Expected reference Stage-2 feature map [C,H,W] for {level_name}, got {feature_map.shape}."
            )
    coarsest_level_name, coarsest_feature_map = min(
        reference_levels.items(),
        key=lambda item: (int(item[1].shape[-2] * item[1].shape[-1]), item[0]),
    )
    coarsest_grid_size = (int(coarsest_feature_map.shape[-2]), int(coarsest_feature_map.shape[-1]))
    aligned: dict[str, np.ndarray] = {}
    for level_name in selected_level_names:
        feature_map = selected[level_name]
        if (int(feature_map.shape[-2]), int(feature_map.shape[-1])) == coarsest_grid_size:
            aligned[level_name] = feature_map.astype(np.float32, copy=True)
            continue
        tensor = torch.as_tensor(feature_map[None], dtype=torch.float32)
        kwargs = {}
        if interpolation_mode in {"bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        resized = torch.nn.functional.interpolate(
            tensor,
            size=coarsest_grid_size,
            mode=interpolation_mode,
            **kwargs,
        )[0]
        aligned[level_name] = np.asarray(resized.cpu().numpy(), dtype=np.float32)
    return aligned, coarsest_level_name, coarsest_grid_size


def l2_normalize_feature_levels(feature_levels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """L2-normalize each feature level independently per pixel across channels."""

    normalized: dict[str, np.ndarray] = {}
    for level_name, feature_map in feature_levels.items():
        array = np.asarray(feature_map, dtype=np.float32)
        if array.ndim != 3:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"Expected feature map [C,H,W] for {level_name}, got {array.shape}."
            )
        norms = np.linalg.norm(array, axis=0, keepdims=True)
        safe = np.where(norms > 1e-6, norms, 1.0).astype(np.float32)
        normalized_map = array / safe
        normalized_map = np.where(norms > 1e-6, normalized_map, 0.0)
        if not np.isfinite(normalized_map).all():
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"NaN/Inf detected after per-level L2 normalization for {level_name}."
            )
        normalized[level_name] = normalized_map.astype(np.float32)
    return normalized


def build_coarsest_grid_ground_truth(
    *,
    sample: Any,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Downsample the binary GT partition to the coarsest aligned feature grid."""

    import torch

    target_height, target_width = (int(target_size[0]), int(target_size[1]))
    if target_height < 1 or target_width < 1:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Invalid coarsest-grid target size for GT alignment.",
            diagnostics={"target_size": target_size},
        )
    label_map = label_map_from_regions((sample.texture_a_mask, sample.texture_b_mask))
    label_tensor = torch.as_tensor(label_map[None, None], dtype=torch.float32)
    resized = torch.nn.functional.interpolate(
        label_tensor,
        size=(target_height, target_width),
        mode="nearest",
    )[0, 0]
    coarse_labels = np.asarray(resized.cpu().numpy(), dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(coarse_labels))
    if any(value not in {0, 1, 2} for value in unique_values):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "GT alignment produced unexpected labels on the coarsest feature grid.",
            diagnostics={"unique_labels": unique_values},
        )
    if 1 not in unique_values and 2 not in unique_values:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "GT alignment removed both texture regions on the coarsest feature grid.",
            diagnostics={
                "unique_labels": unique_values,
                "target_size": (target_height, target_width),
                "crop_name": sample.crop_name,
            },
        )
    return coarse_labels


def materialize_feature_levels_on_device(
    feature_levels: Mapping[str, np.ndarray],
    *,
    device: Any,
) -> dict[str, Any]:
    """Move cached aligned feature levels onto the requested torch device."""

    import torch

    materialized: dict[str, Any] = {}
    for level_name, feature_map in feature_levels.items():
        tensor = torch.as_tensor(feature_map[None], dtype=torch.float32, device=device)
        if tensor.ndim != 4:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"Expected materialized feature tensor [1,C,H,W] for {level_name}, got {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"NaN/Inf detected while materializing aligned features for {level_name}."
            )
        materialized[level_name] = tensor
    return materialized


def compute_sampled_pairwise_affinity_loss(
    embedding_map: Any,
    coarsest_grid_labels: Any,
    *,
    pairs_per_image: int,
    affinity_temperature: float,
) -> AffinityLossResult:
    """Compute a BCE affinity loss on explicit same/different sampled pixel pairs.

    If nearest-neighbor GT alignment leaves only one texture region on the
    coarsest grid, the sample is still valid. In that case the loss falls back
    to same-region affinity pairs from the surviving texture region only.
    """

    import torch
    import torch.nn.functional as F

    embeddings = embedding_map
    if embeddings.ndim == 4:
        if int(embeddings.shape[0]) != 1:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"Pairwise affinity loss expects batch size 1, got {tuple(embeddings.shape)}."
            )
        embeddings = embeddings[0]
    if embeddings.ndim != 3:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Pairwise affinity loss expects embeddings [D,H,W], got {tuple(embeddings.shape)}."
        )
    if not torch.isfinite(embeddings).all():
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Pairwise affinity loss got NaN/Inf embeddings.")

    labels = coarsest_grid_labels
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=embeddings.device)
    else:
        labels = labels.to(device=embeddings.device, dtype=torch.long)
    if labels.ndim != 2 or tuple(labels.shape) != (int(embeddings.shape[-2]), int(embeddings.shape[-1])):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Pairwise affinity loss got GT labels that do not match the coarsest embedding grid.",
            diagnostics={
                "embedding_hw": (int(embeddings.shape[-2]), int(embeddings.shape[-1])),
                "label_shape": tuple(int(value) for value in labels.shape),
            },
        )

    flat_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose(0, 1)
    flat_labels = labels.reshape(-1)
    class_a = torch.nonzero(flat_labels == 1, as_tuple=False).flatten()
    class_b = torch.nonzero(flat_labels == 2, as_tuple=False).flatten()
    if int(class_a.numel()) == 0 and int(class_b.numel()) == 0:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Pairwise affinity loss found no valid texture-region pixels on the coarsest grid.",
            diagnostics={
                "num_class_a": int(class_a.numel()),
                "num_class_b": int(class_b.numel()),
            },
        )
    if int(pairs_per_image) < 2:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"pairs_per_image must be at least 2, got {pairs_per_image}."
        )

    num_same_pairs = max(1, int(pairs_per_image) // 2)
    num_different_pairs = max(1, int(pairs_per_image) - num_same_pairs)
    same_i_parts: list[Any] = []
    same_j_parts: list[Any] = []
    if int(class_a.numel()) > 0 and int(class_b.numel()) > 0:
        num_same_a = max(1, num_same_pairs // 2)
        num_same_b = max(1, num_same_pairs - num_same_a)
        same_i_parts.extend(
            [
                class_a[torch.randint(int(class_a.numel()), (num_same_a,), device=embeddings.device)],
                class_b[torch.randint(int(class_b.numel()), (num_same_b,), device=embeddings.device)],
            ]
        )
        same_j_parts.extend(
            [
                class_a[torch.randint(int(class_a.numel()), (num_same_a,), device=embeddings.device)],
                class_b[torch.randint(int(class_b.numel()), (num_same_b,), device=embeddings.device)],
            ]
        )
    elif int(class_a.numel()) > 0:
        same_i_parts.append(
            class_a[torch.randint(int(class_a.numel()), (num_same_pairs,), device=embeddings.device)]
        )
        same_j_parts.append(
            class_a[torch.randint(int(class_a.numel()), (num_same_pairs,), device=embeddings.device)]
        )
    else:
        same_i_parts.append(
            class_b[torch.randint(int(class_b.numel()), (num_same_pairs,), device=embeddings.device)]
        )
        same_j_parts.append(
            class_b[torch.randint(int(class_b.numel()), (num_same_pairs,), device=embeddings.device)]
        )

    same_i = torch.cat(same_i_parts, dim=0)
    same_j = torch.cat(same_j_parts, dim=0)
    same_logits = torch.sum(flat_embeddings[same_i] * flat_embeddings[same_j], dim=1) / float(affinity_temperature)
    if int(class_a.numel()) > 0 and int(class_b.numel()) > 0:
        diff_i = class_a[torch.randint(int(class_a.numel()), (num_different_pairs,), device=embeddings.device)]
        diff_j = class_b[torch.randint(int(class_b.numel()), (num_different_pairs,), device=embeddings.device)]
        diff_logits = torch.sum(flat_embeddings[diff_i] * flat_embeddings[diff_j], dim=1) / float(
            affinity_temperature
        )
    else:
        diff_logits = torch.empty((0,), dtype=same_logits.dtype, device=embeddings.device)
    if not torch.isfinite(same_logits).all() or (int(diff_logits.numel()) > 0 and not torch.isfinite(diff_logits).all()):
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Pairwise affinity loss produced NaN/Inf similarities.")

    logits = torch.cat([same_logits, diff_logits], dim=0)
    targets = torch.cat(
        [
            torch.ones((same_logits.shape[0],), dtype=torch.float32, device=embeddings.device),
            torch.zeros((int(diff_logits.shape[0]),), dtype=torch.float32, device=embeddings.device),
        ],
        dim=0,
    )
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    if not torch.isfinite(loss):
        raise Sam3CoarseVsFineScaleProbeRuntimeError("Pairwise affinity loss became NaN/Inf.")

    return AffinityLossResult(
        loss=loss,
        num_same_pairs=int(same_logits.shape[0]),
        num_different_pairs=int(diff_logits.shape[0]),
        mean_same_similarity=float(torch.sigmoid(same_logits).mean().detach().cpu().item()),
        mean_different_similarity=(
            float(torch.sigmoid(diff_logits).mean().detach().cpu().item()) if int(diff_logits.numel()) > 0 else 0.0
        ),
    )


def cluster_embedding_map_kmeans(
    embedding_map: Any,
    *,
    num_clusters: int,
    metric: str,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Cluster a learned embedding map ``[D,H,W]`` into a binary label map ``[H,W]``."""

    import torch

    embeddings = embedding_map
    if torch.is_tensor(embeddings):
        if embeddings.ndim == 4:
            if int(embeddings.shape[0]) != 1:
                raise Sam3CoarseVsFineScaleProbeRuntimeError(
                    f"K-means expects batch size 1 embeddings, got {tuple(embeddings.shape)}."
                )
            embeddings = embeddings[0]
        if embeddings.ndim != 3:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"K-means expects embeddings [D,H,W], got {tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise Sam3CoarseVsFineScaleProbeRuntimeError("K-means received NaN/Inf embeddings.")
        embedding_array = np.asarray(embeddings.detach().cpu().numpy(), dtype=np.float32)
    else:
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.ndim != 3:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"K-means expects embeddings [D,H,W], got {embedding_array.shape}."
            )
        if not np.isfinite(embedding_array).all():
            raise Sam3CoarseVsFineScaleProbeRuntimeError("K-means received NaN/Inf embeddings.")

    _, height, width = (int(value) for value in embedding_array.shape)
    vectors = embedding_array.reshape(embedding_array.shape[0], height * width).T
    if vectors.shape[0] < int(num_clusters):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "K-means received fewer spatial vectors than clusters.",
            diagnostics={"num_vectors": int(vectors.shape[0]), "num_clusters": int(num_clusters)},
        )
    result = deterministic_kmeans(
        vectors,
        num_clusters=int(num_clusters),
        metric=str(metric),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    )
    return result.labels.reshape(height, width).astype(np.int32)


def upsample_label_map_to_image(
    label_map: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor upsample from the coarsest grid to image resolution."""

    import torch

    labels = np.asarray(label_map, dtype=np.int32)
    if labels.ndim != 2:
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Expected 2D label map for upsampling, got {labels.shape}."
        )
    tensor = torch.as_tensor(labels[None, None], dtype=torch.float32)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(int(image_size[0]), int(image_size[1])),
        mode="nearest",
    )[0, 0]
    return np.asarray(resized.cpu().numpy(), dtype=np.int32)


def select_best_binary_assignment(
    *,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    source_a: str,
    source_b: str,
) -> dict[str, Any]:
    """Pick the better GT alignment for two unlabeled binary predictions."""

    from rwtd_sam3.eval.metrics import compute_partition_metrics

    direct = compute_partition_metrics(prediction_a, prediction_b, target_a, target_b)
    swapped = compute_partition_metrics(prediction_b, prediction_a, target_a, target_b)
    if (swapped.miou, swapped.ari) > (direct.miou, direct.ari):
        return {
            "assignment_used": f"{source_b}->texture_a,{source_a}->texture_b",
            "prediction_a": np.asarray(prediction_b, dtype=bool),
            "prediction_b": np.asarray(prediction_a, dtype=bool),
            "direct_eval_miou": float(direct.miou),
            "direct_eval_ari": float(direct.ari),
            "swapped_eval_miou": float(swapped.miou),
            "swapped_eval_ari": float(swapped.ari),
            "eval_miou": float(swapped.miou),
            "eval_ari": float(swapped.ari),
        }
    return {
        "assignment_used": f"{source_a}->texture_a,{source_b}->texture_b",
        "prediction_a": np.asarray(prediction_a, dtype=bool),
        "prediction_b": np.asarray(prediction_b, dtype=bool),
        "direct_eval_miou": float(direct.miou),
        "direct_eval_ari": float(direct.ari),
        "swapped_eval_miou": float(swapped.miou),
        "swapped_eval_ari": float(swapped.ari),
        "eval_miou": float(direct.miou),
        "eval_ari": float(direct.ari),
    }


def evaluate_probe_partition(
    *,
    sample: Any,
    coarsest_grid_label_map: np.ndarray,
) -> ProbePartitionResult:
    """Upsample a binary coarsest-grid label map and score it permutation-invariantly."""

    image_space_label_map = upsample_label_map_to_image(
        coarsest_grid_label_map,
        image_size=(sample.height, sample.width),
    )
    unique_values = tuple(int(value) for value in np.unique(image_space_label_map))
    if unique_values != (0, 1):
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            "Probe evaluation expected a binary 2-cluster label map after k-means.",
            diagnostics={"unique_labels": unique_values, "crop_name": sample.crop_name},
        )
    mask_a = np.asarray(image_space_label_map == 0, dtype=bool)
    mask_b = np.asarray(image_space_label_map == 1, dtype=bool)
    assignment = select_best_binary_assignment(
        prediction_a=mask_a,
        prediction_b=mask_b,
        target_a=sample.texture_a_mask,
        target_b=sample.texture_b_mask,
        source_a="cluster_0",
        source_b="cluster_1",
    )
    return ProbePartitionResult(
        coarsest_grid_label_map=np.asarray(coarsest_grid_label_map, dtype=np.int32),
        image_space_label_map=image_space_label_map,
        prediction_a=assignment["prediction_a"],
        prediction_b=assignment["prediction_b"],
        assignment_used=str(assignment["assignment_used"]),
        direct_eval_miou=float(assignment["direct_eval_miou"]),
        direct_eval_ari=float(assignment["direct_eval_ari"]),
        swapped_eval_miou=float(assignment["swapped_eval_miou"]),
        swapped_eval_ari=float(assignment["swapped_eval_ari"]),
        eval_miou=float(assignment["eval_miou"]),
        eval_ari=float(assignment["eval_ari"]),
    )


def resolve_probe_variant_levels(
    *,
    variant: str,
    available_level_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve the selected level list for one Stage-2 run variant."""

    try:
        variant_spec = STAGE2_VARIANT_SPECS[variant]
    except KeyError as exc:
        expected = ", ".join(sorted(STAGE2_VARIANT_SPECS))
        raise Sam3CoarseVsFineScaleProbeRuntimeError(
            f"Unknown Stage-2 probe variant '{variant}'. Expected one of: {expected}."
        ) from exc
    available = tuple(str(name) for name in available_level_names)
    policy = str(variant_spec["selected_level_policy"])
    if policy == "all_scales":
        return available
    if policy == "coarse_plus_next_finer":
        if len(available) < 2:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                "The coarse-plus-next-finer Stage-2 probe variant needs at least two SAM pyramid levels.",
                diagnostics={"available_level_names": available},
            )
        return available[:2]
    if policy == "single_level_by_name":
        requested_level_name = str(variant_spec["requested_level_name"])
        if requested_level_name not in available:
            raise Sam3CoarseVsFineScaleProbeRuntimeError(
                f"The Stage-2 single-scale variant '{variant}' requested level '{requested_level_name}', but the available SAM pyramid levels were {available}.",
                diagnostics={
                    "variant": variant,
                    "requested_level_name": requested_level_name,
                    "available_level_names": available,
                },
            )
        return (requested_level_name,)
    raise Sam3CoarseVsFineScaleProbeRuntimeError(
        f"Unsupported Stage-2 level-selection policy '{policy}'."
    )
