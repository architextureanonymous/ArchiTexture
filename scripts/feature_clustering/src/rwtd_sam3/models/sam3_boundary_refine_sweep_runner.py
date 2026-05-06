"""Stage-B boundary-only refinement sweep on top of the flip-averaged coarse partition.

This module keeps the existing strong coarse segmentation fixed, then compares a
small set of conservative local refinement variants that only update boundary
regions on finer SAM feature levels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    DEFAULT_MODEL_ID,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalRuntimeError,
    _FeatureLevel,
    _apply_image_transform,
    _cluster_pooled_label_map_from_numpy_feature_map,
    _compute_label_map_component_stats,
    _feature_map_to_numpy,
    _label_map_branch_from_native_map,
    _normalize_numpy_feature_map,
    _undo_feature_transform,
    _upsample_label_map,
)


BOUNDARY_REFINE_SWEEP_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS,
    "boundary_refine_margin_threshold": 0.035,
    "boundary_refine_local_blend_alpha": 0.7,
    "boundary_refine_interior_exclusion_radius": 1,
    "boundary_refine_local_support_radius_cells": 2,
}

BOUNDARY_REFINE_SWEEP_VARIANT_SPECS: tuple[tuple[str, str], ...] = (
    ("v0_no_refine", "no_refine"),
    ("v1_refine_global_anchored", "refine_global_anchored"),
    ("v2_refine_global_anchored_margin", "refine_global_anchored_margin"),
    ("v3_refine_global_local_blend_margin", "refine_global_local_blend_margin"),
    ("v4_refine_two_level_global_margin", "refine_two_level_global_margin"),
)
BOUNDARY_REFINE_SWEEP_VARIANT_IDS = tuple(name for name, _ in BOUNDARY_REFINE_SWEEP_VARIANT_SPECS)
BOUNDARY_REFINE_SWEEP_VARIANT_LABELS = dict(BOUNDARY_REFINE_SWEEP_VARIANT_SPECS)


@dataclass(frozen=True)
class BoundaryRefineVariantResult:
    """One boundary-only refinement branch emitted by the sweep."""

    variant_id: str
    label_map: np.ndarray
    native_label_map: np.ndarray
    mask_a: np.ndarray
    mask_b: np.ndarray
    changed_mask: np.ndarray
    percent_pixels_changed: float
    num_changed_components: int
    average_changed_margin: float


@dataclass(frozen=True)
class BoundaryRefineSweepResult:
    """Per-image sweep payload containing the fixed coarse partition and V0-V4 outputs."""

    coarsest_label_map: np.ndarray
    coarsest_native_label_map: np.ndarray
    boundary_cell_map: np.ndarray
    boundary_cell_map_native: np.ndarray
    level_names: tuple[str, ...]
    level_resolutions: tuple[tuple[int, int], ...]
    coarsest_pool_kernel_size: int
    coarsest_pool_stride: int
    pooled_grid_resolution: tuple[int, int]
    variants: dict[str, BoundaryRefineVariantResult]


@dataclass(frozen=True)
class _NumpyFeatureLevel:
    """One flip-restored SAM feature level kept in numpy form."""

    name: str
    resolution: tuple[int, int]
    feature_map: np.ndarray


class Sam3BoundaryRefineSweepRunner(Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner):
    """Run the stage-B boundary-only ablation sweep on top of the strong coarse partition."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(BOUNDARY_REFINE_SWEEP_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_feature_cluster_sweep(self, image: Image.Image) -> BoundaryRefineSweepResult:
        """Run V0-V4 on one image and return the full comparison payload."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, identity_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        flip_avg_levels = self._build_flip_average_feature_levels(image, official_backend, identity_levels)
        if len(flip_avg_levels) < 2:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "boundary_refine_sweep requires at least two feature levels after the coarse pooled initialization."
            )

        pooled_native_label_map, pooled_grid_resolution = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=flip_avg_levels[0].feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        base_branch = _label_map_branch_from_native_map(
            native_label_map=pooled_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        boundary_cell_map_native = _boundary_touching_cells(pooled_native_label_map)
        boundary_cell_map = _upsample_label_map(
            boundary_cell_map_native.astype(np.int32),
            target_size=image_size,
            torch_module=official_backend.torch,
        ).astype(bool)

        variants: dict[str, BoundaryRefineVariantResult] = {}
        variants["v0_no_refine"] = _build_variant_result(
            variant_id="v0_no_refine",
            native_label_map=pooled_native_label_map,
            image_space_label_map=base_branch["image_space_label_map"],
            baseline_image_space_label_map=base_branch["image_space_label_map"],
            margin_map=None,
            torch_module=official_backend.torch,
        )

        next_finer_level = flip_avg_levels[1]
        finest_level = flip_avg_levels[-1]
        margin_threshold = float(self.settings["boundary_refine_margin_threshold"])
        local_blend_alpha = float(self.settings["boundary_refine_local_blend_alpha"])
        support_radius_cells = int(self.settings["boundary_refine_local_support_radius_cells"])
        interior_radius = int(self.settings["boundary_refine_interior_exclusion_radius"])

        v1_native, v1_margin = _refine_pooled_boundary_cells_one_level(
            pooled_label_map=pooled_native_label_map,
            boundary_cell_map=boundary_cell_map_native,
            target_feature_map=next_finer_level.feature_map,
            margin_threshold=None,
            local_blend_alpha=None,
            local_support_radius_cells=support_radius_cells,
            interior_exclusion_radius=interior_radius,
        )
        variants["v1_refine_global_anchored"] = _build_variant_result(
            variant_id="v1_refine_global_anchored",
            native_label_map=v1_native,
            image_space_label_map=_upsample_label_map(
                v1_native,
                target_size=image_size,
                torch_module=official_backend.torch,
            ),
            baseline_image_space_label_map=base_branch["image_space_label_map"],
            margin_map=v1_margin,
            torch_module=official_backend.torch,
        )

        v2_native, v2_margin = _refine_pooled_boundary_cells_one_level(
            pooled_label_map=pooled_native_label_map,
            boundary_cell_map=boundary_cell_map_native,
            target_feature_map=next_finer_level.feature_map,
            margin_threshold=margin_threshold,
            local_blend_alpha=None,
            local_support_radius_cells=support_radius_cells,
            interior_exclusion_radius=interior_radius,
        )
        variants["v2_refine_global_anchored_margin"] = _build_variant_result(
            variant_id="v2_refine_global_anchored_margin",
            native_label_map=v2_native,
            image_space_label_map=_upsample_label_map(
                v2_native,
                target_size=image_size,
                torch_module=official_backend.torch,
            ),
            baseline_image_space_label_map=base_branch["image_space_label_map"],
            margin_map=v2_margin,
            torch_module=official_backend.torch,
        )

        v3_native, v3_margin = _refine_pooled_boundary_cells_one_level(
            pooled_label_map=pooled_native_label_map,
            boundary_cell_map=boundary_cell_map_native,
            target_feature_map=next_finer_level.feature_map,
            margin_threshold=margin_threshold,
            local_blend_alpha=local_blend_alpha,
            local_support_radius_cells=support_radius_cells,
            interior_exclusion_radius=interior_radius,
        )
        variants["v3_refine_global_local_blend_margin"] = _build_variant_result(
            variant_id="v3_refine_global_local_blend_margin",
            native_label_map=v3_native,
            image_space_label_map=_upsample_label_map(
                v3_native,
                target_size=image_size,
                torch_module=official_backend.torch,
            ),
            baseline_image_space_label_map=base_branch["image_space_label_map"],
            margin_map=v3_margin,
            torch_module=official_backend.torch,
        )

        v4_mid_native, _ = _refine_pooled_boundary_cells_one_level(
            pooled_label_map=pooled_native_label_map,
            boundary_cell_map=boundary_cell_map_native,
            target_feature_map=next_finer_level.feature_map,
            margin_threshold=margin_threshold,
            local_blend_alpha=None,
            local_support_radius_cells=support_radius_cells,
            interior_exclusion_radius=interior_radius,
        )
        if finest_level.resolution == next_finer_level.resolution:
            v4_native = v4_mid_native
            v4_margin = np.zeros_like(v4_mid_native, dtype=np.float32)
        else:
            upsampled_mid = _upsample_label_map(
                v4_mid_native,
                target_size=finest_level.resolution,
                torch_module=official_backend.torch,
            )
            v4_native, v4_margin = _refine_boundary_band_one_level(
                current_label_map=upsampled_mid,
                target_feature_map=finest_level.feature_map,
                margin_threshold=margin_threshold,
                interior_exclusion_radius=interior_radius,
            )
        variants["v4_refine_two_level_global_margin"] = _build_variant_result(
            variant_id="v4_refine_two_level_global_margin",
            native_label_map=v4_native,
            image_space_label_map=_upsample_label_map(
                v4_native,
                target_size=image_size,
                torch_module=official_backend.torch,
            ),
            baseline_image_space_label_map=base_branch["image_space_label_map"],
            margin_map=v4_margin,
            torch_module=official_backend.torch,
        )

        return BoundaryRefineSweepResult(
            coarsest_label_map=base_branch["image_space_label_map"],
            coarsest_native_label_map=pooled_native_label_map,
            boundary_cell_map=boundary_cell_map,
            boundary_cell_map_native=boundary_cell_map_native,
            level_names=tuple(level.name for level in flip_avg_levels),
            level_resolutions=tuple(level.resolution for level in flip_avg_levels),
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            variants=variants,
        )

    def _build_flip_average_feature_levels(
        self,
        image: Image.Image,
        official_backend: Any,
        identity_levels: list[_FeatureLevel],
    ) -> list[_NumpyFeatureLevel]:
        transform_names = ("identity", "hflip", "vflip", "hvflip")
        collected: list[list[np.ndarray]] = [[] for _ in identity_levels]

        for transform_name in transform_names:
            transformed_image = _apply_image_transform(image, transform_name)
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(transformed_image, official_backend)
            if len(feature_levels) != len(identity_levels):
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "boundary_refine_sweep got inconsistent feature-level counts across flip averaging."
                )
            for level_index, feature_level in enumerate(feature_levels):
                if feature_level.resolution != identity_levels[level_index].resolution:
                    raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                        "boundary_refine_sweep got inconsistent feature-level resolutions across flip averaging."
                    )
                restored = _undo_feature_transform(_feature_map_to_numpy(feature_level.feature_map), transform_name)
                collected[level_index].append(restored)

        averaged_levels: list[_NumpyFeatureLevel] = []
        for identity_level, feature_maps in zip(identity_levels, collected):
            averaged_levels.append(
                _NumpyFeatureLevel(
                    name=identity_level.name,
                    resolution=identity_level.resolution,
                    feature_map=_normalize_numpy_feature_map(np.mean(np.stack(feature_maps, axis=0), axis=0)),
                )
            )
        return averaged_levels


def build_boundary_refine_sweep_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-variant metrics and change statistics over one sweep run."""

    if not rows:
        raise ValueError("Boundary refine sweep summary requires at least one row.")

    summary: dict[str, Any] = {"variant_metrics": {}, "baseline_variant": "v0_no_refine"}
    baseline_field = "v0_no_refine_eval_miou"
    best_variant = "v0_no_refine"
    best_tuple = (-float("inf"), float("inf"))

    for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
        eval_miou_field = f"{variant_id}_eval_miou"
        eval_ari_field = f"{variant_id}_eval_ari"
        changed_pct_field = f"{variant_id}_percent_pixels_changed"
        changed_components_field = f"{variant_id}_num_changed_components"
        changed_margin_field = f"{variant_id}_average_changed_margin"
        metrics = {
            "eval_miou": float(np.mean([float(row[eval_miou_field]) for row in rows])),
            "eval_ari": float(np.mean([float(row[eval_ari_field]) for row in rows])),
            "percent_pixels_changed": float(np.mean([float(row[changed_pct_field]) for row in rows])),
            "num_changed_components": float(np.mean([float(row[changed_components_field]) for row in rows])),
            "average_changed_margin": float(np.mean([float(row[changed_margin_field]) for row in rows])),
            "improved_vs_v0_count": int(
                sum(float(row[eval_miou_field]) > float(row[baseline_field]) for row in rows)
            ),
            "worsened_vs_v0_count": int(
                sum(float(row[eval_miou_field]) < float(row[baseline_field]) for row in rows)
            ),
        }
        summary["variant_metrics"][variant_id] = metrics
        rank_tuple = (metrics["eval_miou"], metrics["percent_pixels_changed"])
        if rank_tuple[0] > best_tuple[0] or (rank_tuple[0] == best_tuple[0] and rank_tuple[1] < best_tuple[1]):
            best_variant = variant_id
            best_tuple = rank_tuple

    summary["recommended_variant"] = best_variant
    summary["recommended_variant_reason"] = (
        f"Highest mean eval_miou with percent-pixels-changed tie-break: {best_variant} "
        f"({summary['variant_metrics'][best_variant]['eval_miou']:.4f}, "
        f"{summary['variant_metrics'][best_variant]['percent_pixels_changed']:.4f})."
    )
    return summary


def _build_variant_result(
    *,
    variant_id: str,
    native_label_map: np.ndarray,
    image_space_label_map: np.ndarray,
    baseline_image_space_label_map: np.ndarray,
    margin_map: np.ndarray | None,
    torch_module: Any,
) -> BoundaryRefineVariantResult:
    mask_a, mask_b = _label_map_to_masks(image_space_label_map)
    changed_mask = np.asarray(image_space_label_map != baseline_image_space_label_map, dtype=bool)
    changed_pixel_count = int(changed_mask.sum())
    total_pixels = int(changed_mask.size)
    percent_pixels_changed = float(changed_pixel_count / total_pixels) if total_pixels else 0.0
    changed_margin = 0.0
    if margin_map is not None and changed_pixel_count > 0:
        image_space_margin = _upsample_scalar_map_nearest(margin_map, image_space_label_map.shape, torch_module)
        changed_margin = float(np.mean(np.abs(image_space_margin[changed_mask])))
    return BoundaryRefineVariantResult(
        variant_id=variant_id,
        label_map=np.asarray(image_space_label_map, dtype=np.int32),
        native_label_map=np.asarray(native_label_map, dtype=np.int32),
        mask_a=mask_a,
        mask_b=mask_b,
        changed_mask=changed_mask,
        percent_pixels_changed=percent_pixels_changed,
        num_changed_components=_count_connected_components(changed_mask),
        average_changed_margin=changed_margin,
    )


def _label_map_to_masks(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(label_map, dtype=np.int32)
    return labels == 0, labels == 1


def _boundary_touching_cells(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def _refine_pooled_boundary_cells_one_level(
    *,
    pooled_label_map: np.ndarray,
    boundary_cell_map: np.ndarray,
    target_feature_map: np.ndarray,
    margin_threshold: float | None,
    local_blend_alpha: float | None,
    local_support_radius_cells: int,
    interior_exclusion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_height, target_width = int(target_feature_map.shape[1]), int(target_feature_map.shape[2])
    current_labels = _upsample_numpy_label_map(pooled_label_map, (target_height, target_width))
    refined_labels = current_labels.copy()
    margin_map = np.zeros((target_height, target_width), dtype=np.float32)
    support_mask = _confident_support_mask(current_labels, interior_exclusion_radius)
    global_a, global_b = _compute_global_prototypes_numpy(target_feature_map, current_labels, support_mask)
    source_height, source_width = pooled_label_map.shape

    for cell_y, cell_x in np.argwhere(boundary_cell_map):
        block_y = _map_index_range(cell_y, source_height, target_height)
        block_x = _map_index_range(cell_x, source_width, target_width)
        y0, y1 = block_y
        x0, x1 = block_x
        if y1 <= y0 or x1 <= x0:
            continue

        prototype_a = global_a
        prototype_b = global_b
        if local_blend_alpha is not None:
            local_pair = _compute_local_prototypes_for_pooled_cell(
                feature_map=target_feature_map,
                current_labels=current_labels,
                support_mask=support_mask,
                cell_y=int(cell_y),
                cell_x=int(cell_x),
                source_shape=pooled_label_map.shape,
                target_shape=(target_height, target_width),
                radius_cells=local_support_radius_cells,
            )
            if local_pair is not None:
                local_a, local_b = local_pair
                prototype_a = _normalize_vector(local_blend_alpha * global_a + (1.0 - local_blend_alpha) * local_a)
                prototype_b = _normalize_vector(local_blend_alpha * global_b + (1.0 - local_blend_alpha) * local_b)

        block_features = target_feature_map[:, y0:y1, x0:x1].reshape(target_feature_map.shape[0], -1).T
        similarity_a = block_features @ prototype_a
        similarity_b = block_features @ prototype_b
        margin = (similarity_a - similarity_b).reshape(y1 - y0, x1 - x0).astype(np.float32)
        proposal = (similarity_b > similarity_a).astype(np.int32).reshape(y1 - y0, x1 - x0)
        updated_block = refined_labels[y0:y1, x0:x1].copy()
        if margin_threshold is None:
            updated_block = proposal
        else:
            reassign_mask = np.abs(margin) > float(margin_threshold)
            updated_block[reassign_mask] = proposal[reassign_mask]
        refined_labels[y0:y1, x0:x1] = updated_block
        margin_map[y0:y1, x0:x1] = margin

    return refined_labels.astype(np.int32), margin_map


def _refine_boundary_band_one_level(
    *,
    current_label_map: np.ndarray,
    target_feature_map: np.ndarray,
    margin_threshold: float,
    interior_exclusion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(current_label_map, dtype=np.int32)
    support_mask = _confident_support_mask(labels, interior_exclusion_radius)
    prototype_a, prototype_b = _compute_global_prototypes_numpy(target_feature_map, labels, support_mask)
    boundary_mask = _boundary_touching_cells(labels)
    if not np.any(boundary_mask):
        return labels.astype(np.int32), np.zeros_like(labels, dtype=np.float32)

    refined = labels.copy()
    margin_map = np.zeros_like(labels, dtype=np.float32)
    uncertain_vectors = target_feature_map[:, boundary_mask].T
    similarity_a = uncertain_vectors @ prototype_a
    similarity_b = uncertain_vectors @ prototype_b
    margin = (similarity_a - similarity_b).astype(np.float32)
    proposal = (similarity_b > similarity_a).astype(np.int32)
    update_mask = np.abs(margin) > float(margin_threshold)
    flat_refined = refined.reshape(-1)
    flat_boundary = boundary_mask.reshape(-1)
    boundary_indices = np.flatnonzero(flat_boundary)
    flat_refined[boundary_indices[update_mask]] = proposal[update_mask]
    refined = flat_refined.reshape(labels.shape)
    margin_map.reshape(-1)[boundary_indices] = margin
    return refined.astype(np.int32), margin_map


def _confident_support_mask(label_map: np.ndarray, interior_exclusion_radius: int) -> np.ndarray:
    boundary_mask = _boundary_touching_cells(label_map)
    dilated = _dilate_mask(boundary_mask, radius=max(0, int(interior_exclusion_radius)))
    support_mask = ~dilated
    if np.any(np.logical_and(support_mask, label_map == 0)) and np.any(np.logical_and(support_mask, label_map == 1)):
        return support_mask
    fallback_mask = ~boundary_mask
    if np.any(np.logical_and(fallback_mask, label_map == 0)) and np.any(np.logical_and(fallback_mask, label_map == 1)):
        return fallback_mask
    return np.ones_like(label_map, dtype=bool)


def _compute_global_prototypes_numpy(
    feature_map: np.ndarray,
    label_map: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_vectors = feature_map.reshape(feature_map.shape[0], -1).T
    flat_labels = label_map.reshape(-1)
    flat_support = support_mask.reshape(-1)
    support_a = np.logical_and(flat_support, flat_labels == 0)
    support_b = np.logical_and(flat_support, flat_labels == 1)
    if not np.any(support_a) or not np.any(support_b):
        support_a = flat_labels == 0
        support_b = flat_labels == 1
    if not np.any(support_a) or not np.any(support_b):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "boundary_refine_sweep could not build both anchored prototypes."
        )
    proto_a = _normalize_vector(flat_vectors[support_a].mean(axis=0))
    proto_b = _normalize_vector(flat_vectors[support_b].mean(axis=0))
    return proto_a, proto_b


def _compute_local_prototypes_for_pooled_cell(
    *,
    feature_map: np.ndarray,
    current_labels: np.ndarray,
    support_mask: np.ndarray,
    cell_y: int,
    cell_x: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    radius_cells: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    source_height, source_width = source_shape
    y0, y1 = _map_index_range(max(0, cell_y - radius_cells), source_height, target_shape[0])
    _, y2 = _map_index_range(min(source_height - 1, cell_y + radius_cells), source_height, target_shape[0])
    x0, x1 = _map_index_range(max(0, cell_x - radius_cells), source_width, target_shape[1])
    _, x2 = _map_index_range(min(source_width - 1, cell_x + radius_cells), source_width, target_shape[1])
    local_slice = (slice(y0, y2), slice(x0, x2))
    local_labels = current_labels[local_slice]
    local_support = support_mask[local_slice]
    if local_labels.size == 0:
        return None

    support_a = np.logical_and(local_support, local_labels == 0)
    support_b = np.logical_and(local_support, local_labels == 1)
    if not np.any(support_a) or not np.any(support_b):
        return None

    local_vectors = feature_map[:, local_slice[0], local_slice[1]].reshape(feature_map.shape[0], -1).T
    proto_a = _normalize_vector(local_vectors[support_a.reshape(-1)].mean(axis=0))
    proto_b = _normalize_vector(local_vectors[support_b.reshape(-1)].mean(axis=0))
    return proto_a, proto_b


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32)


def _upsample_numpy_label_map(label_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    source_height, source_width = label_map.shape
    target_height, target_width = target_size
    row_indices = np.minimum(
        (np.arange(target_height, dtype=np.int64) * source_height) // max(target_height, 1),
        source_height - 1,
    )
    col_indices = np.minimum(
        (np.arange(target_width, dtype=np.int64) * source_width) // max(target_width, 1),
        source_width - 1,
    )
    return np.asarray(label_map[row_indices[:, None], col_indices[None, :]], dtype=np.int32)


def _map_index_range(index: int, source_size: int, target_size: int) -> tuple[int, int]:
    start = (int(index) * int(target_size)) // int(source_size)
    end = ((int(index) + 1) * int(target_size)) // int(source_size)
    return start, max(start + 1, end)


def _upsample_scalar_map_nearest(scalar_map: np.ndarray, target_size: tuple[int, int], torch_module: Any) -> np.ndarray:
    scalar_tensor = torch_module.as_tensor(np.asarray(scalar_map, dtype=np.float32))[None, None]
    upsampled = torch_module.nn.functional.interpolate(
        scalar_tensor,
        size=target_size,
        mode="nearest",
    )[0, 0]
    return np.asarray(upsampled.detach().cpu().numpy(), dtype=np.float32)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return boolean_mask
    padded = np.pad(boolean_mask, pad_width=radius, mode="constant", constant_values=False)
    height, width = boolean_mask.shape
    windows: list[np.ndarray] = []
    for dy in range((2 * radius) + 1):
        for dx in range((2 * radius) + 1):
            windows.append(padded[dy : dy + height, dx : dx + width])
    return np.logical_or.reduce(windows)


def _count_connected_components(mask: np.ndarray) -> int:
    boolean_mask = np.asarray(mask, dtype=bool)
    height, width = boolean_mask.shape
    visited = np.zeros_like(boolean_mask, dtype=bool)
    component_count = 0
    for row in range(height):
        for col in range(width):
            if not boolean_mask[row, col] or visited[row, col]:
                continue
            component_count += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                current_row, current_col = queue.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + dy
                    next_col = current_col + dx
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if not boolean_mask[next_row, next_col] or visited[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
    return component_count
