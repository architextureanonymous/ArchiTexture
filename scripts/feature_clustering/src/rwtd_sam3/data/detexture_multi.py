"""Local DeTexture ADE20K multi-region benchmark adapter.

This module resolves the public ``detecture_data`` asset layout, matches each
RGB crop to its JPEG-compressed multi-region mask, and decodes that mask into a
single integer label map for prompt-free multi-way partitioning experiments.

Primary entrypoints:
- ``load_detexture_multi_overview()``: inspect the local multi-region asset root.
- ``iter_detexture_multi_samples()``: stream decoded image/labelmap pairs.
- ``get_detexture_multi_sample()``: decode one sample by natural-sorted index.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import boundary_from_label_map
from rwtd_sam3.utils.deterministic_clustering import select_num_clusters_by_bic


LOGGER = logging.getLogger(__name__)

DETEXTURE_MULTI_DATASET_ID = "detexture_ade20k_multi"
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DETEXTURE_MULTI_JOKER_LABEL = -1
DETEXTURE_MULTI_GT_DECODE_SETTINGS: dict[str, int | float | str] = {
    "max_decode_clusters": 8,
    "kmeans_max_iterations": 40,
    "kmeans_convergence_tolerance": 1e-4,
    "decode_metric": "euclidean",
    "selection_criterion": "bic",
}
DETEXTURE_MULTI_REGION_FILTER_SETTINGS: dict[str, int | float] = {
    "minimum_region_pixel_fraction": 0.12,
    "joker_label_value": DETEXTURE_MULTI_JOKER_LABEL,
}


@dataclass(frozen=True)
class DeTextureMultiOverview:
    """Resolved metadata for one local DeTexture multi-region asset root."""

    dataset_id: str
    dataset_root: Path
    detecture_data_root: Path
    image_dir: Path
    mask_dir: Path
    overlay_dir: Path | None
    split: str
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str
    gt_decode_settings: dict[str, int | float | str]
    gt_region_filter_settings: dict[str, int | float]


@dataclass(frozen=True)
class DeTextureMultiSample:
    """One decoded multi-region DeTexture crop."""

    index: int
    dataset_id: str
    split: str
    crop_name: str
    image: Image.Image
    gt_label_map: np.ndarray
    valid_pixel_mask: np.ndarray
    boundary_mask: np.ndarray
    oracle_num_regions: int
    raw_oracle_num_regions: int
    joker_region_count: int
    joker_pixel_count: int
    joker_pixel_fraction: float
    joker_label_value: int
    evaluation_view: str
    gt_decode_method: str
    gt_decode_diagnostics: dict[str, Any]

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def width(self) -> int:
        return self.image.size[0]


def load_detexture_multi_overview(
    dataset_root: str | Path,
    *,
    split: str = "all",
) -> DeTextureMultiOverview:
    """Inspect a local DeTexture ADE20K multi-region dataset root."""

    if split not in {"all", "training", "validation"}:
        raise ValueError(f"Unsupported DeTexture multi split '{split}'.")
    resolved_root, detecture_data_root, image_dir, mask_dir, overlay_dir = resolve_detexture_multi_dirs(dataset_root)
    image_ids = _discover_multi_image_ids(image_dir=image_dir, mask_dir=mask_dir, split=split)
    return DeTextureMultiOverview(
        dataset_id=DETEXTURE_MULTI_DATASET_ID,
        dataset_root=resolved_root,
        detecture_data_root=detecture_data_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        overlay_dir=overlay_dir,
        split=split,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view="multi_partition_invariant",
        gt_decode_settings=dict(DETEXTURE_MULTI_GT_DECODE_SETTINGS),
        gt_region_filter_settings=dict(DETEXTURE_MULTI_REGION_FILTER_SETTINGS),
    )


def iter_detexture_multi_samples(
    dataset_root: str | Path,
    *,
    split: str = "all",
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[DeTextureMultiSample]:
    """Yield decoded DeTexture multi-region samples in natural filename order."""

    overview = load_detexture_multi_overview(dataset_root, split=split)
    max_items = max(0, overview.num_examples - start_index) if limit is None else min(
        limit,
        max(0, overview.num_examples - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_detexture_multi_sample_from_overview(overview=overview, index=index)


def get_detexture_multi_sample(
    dataset_root: str | Path,
    *,
    index: int,
    split: str = "all",
) -> DeTextureMultiSample:
    """Load one DeTexture multi-region crop by natural-sorted index."""

    overview = load_detexture_multi_overview(dataset_root, split=split)
    return _decode_detexture_multi_sample_from_overview(overview=overview, index=index)


def resolve_detexture_multi_dirs(
    dataset_root: str | Path,
) -> tuple[Path, Path, Path, Path, Path | None]:
    """Resolve either the dataset root or the embedded ``detecture_data`` directory."""

    requested_root = Path(dataset_root).expanduser().resolve()
    candidates = (
        requested_root,
        requested_root / "detecture_data",
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        image_dir = candidate / "images"
        mask_dir = candidate / "masks"
        overlay_dir = candidate / "overlays"
        if image_dir.is_dir() and mask_dir.is_dir():
            dataset_dir = candidate.parent if candidate.name == "detecture_data" else candidate
            resolved_overlay_dir = overlay_dir if overlay_dir.is_dir() else None
            return dataset_dir, candidate, image_dir, mask_dir, resolved_overlay_dir
    raise FileNotFoundError(
        "Could not resolve a DeTexture multi-region dataset root. Expected either the directory containing "
        "`detecture_data/images` and `detecture_data/masks`, or `detecture_data/` itself. "
        f"Tried from '{requested_root}'."
    )


def _decode_detexture_multi_sample_from_overview(
    overview: DeTextureMultiOverview,
    index: int,
) -> DeTextureMultiSample:
    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for dataset '{overview.dataset_id}' with "
            f"{overview.num_examples} matched image/mask pairs."
        )
    image_id = overview.image_ids[index]
    image_path = _find_asset_path(overview.image_dir, image_id)
    mask_path = _find_asset_path(overview.mask_dir, image_id)
    image = Image.open(image_path).convert("RGB")
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    if mask_rgb.shape[:2] != image.size[::-1]:
        raise ValueError(
            f"DeTexture multi sample '{image_id}' has mismatched image/mask sizes: "
            f"image={image.size}, mask={mask_rgb.shape[1]}x{mask_rgb.shape[0]}."
        )

    raw_label_map, gt_decode_diagnostics = _decode_gt_label_map(mask_rgb)
    filtered_label_map, valid_pixel_mask, region_filter_diagnostics = _apply_joker_region_filter(raw_label_map)
    gt_decode_diagnostics = {**gt_decode_diagnostics, **region_filter_diagnostics}
    boundary_mask = boundary_from_label_map(filtered_label_map, valid_mask=valid_pixel_mask)
    oracle_num_regions = int(np.unique(filtered_label_map[valid_pixel_mask]).size)
    return DeTextureMultiSample(
        index=index,
        dataset_id=overview.dataset_id,
        split=_split_from_image_id(image_id),
        crop_name=image_id,
        image=image,
        gt_label_map=filtered_label_map.astype(np.int32),
        valid_pixel_mask=valid_pixel_mask,
        boundary_mask=boundary_mask,
        oracle_num_regions=oracle_num_regions,
        raw_oracle_num_regions=int(gt_decode_diagnostics["raw_oracle_num_regions"]),
        joker_region_count=int(gt_decode_diagnostics["joker_region_count"]),
        joker_pixel_count=int(gt_decode_diagnostics["joker_pixel_count"]),
        joker_pixel_fraction=float(gt_decode_diagnostics["joker_pixel_fraction"]),
        joker_label_value=int(DETEXTURE_MULTI_REGION_FILTER_SETTINGS["joker_label_value"]),
        evaluation_view=overview.evaluation_view,
        gt_decode_method="jpeg_color_clustering_bic",
        gt_decode_diagnostics=gt_decode_diagnostics,
    )


def _decode_gt_label_map(mask_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width, channels = mask_rgb.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB mask image, got shape {mask_rgb.shape}.")
    flat_colors = mask_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    unique_colors, inverse, counts = np.unique(flat_colors, axis=0, return_inverse=True, return_counts=True)
    max_decode_clusters = min(int(DETEXTURE_MULTI_GT_DECODE_SETTINGS["max_decode_clusters"]), int(unique_colors.shape[0]))
    if max_decode_clusters < 1:
        raise ValueError("DeTexture multi GT decode found no unique colors.")

    selection = select_num_clusters_by_bic(
        unique_colors,
        range(1, max_decode_clusters + 1),
        metric=str(DETEXTURE_MULTI_GT_DECODE_SETTINGS["decode_metric"]),
        max_iterations=int(DETEXTURE_MULTI_GT_DECODE_SETTINGS["kmeans_max_iterations"]),
        tolerance=float(DETEXTURE_MULTI_GT_DECODE_SETTINGS["kmeans_convergence_tolerance"]),
        sample_weights=counts.astype(np.float32),
    )
    flat_labels = selection.clustering.labels[inverse]
    label_map = flat_labels.reshape(height, width).astype(np.int32)
    label_map = _reindex_label_map_by_region_size(label_map)
    region_sizes = [int((label_map == label_id).sum()) for label_id in range(int(label_map.max()) + 1)]
    diagnostics = {
        "decode_method": "jpeg_color_clustering_bic",
        "selection_criterion": selection.criterion_name,
        "selection_value": float(selection.criterion_value),
        "scores_by_k": {int(key): float(value) for key, value in selection.scores_by_k.items()},
        "num_unique_rgb_colors": int(unique_colors.shape[0]),
        "raw_oracle_num_regions": int(np.unique(label_map).size),
        "raw_region_sizes": region_sizes,
        "max_decode_clusters": int(DETEXTURE_MULTI_GT_DECODE_SETTINGS["max_decode_clusters"]),
    }
    return label_map, diagnostics


def _apply_joker_region_filter(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels = np.asarray(label_map, dtype=np.int32)
    total_pixels = int(labels.size)
    if total_pixels < 1:
        raise ValueError("DeTexture multi GT label map is empty.")

    label_values, counts = np.unique(labels, return_counts=True)
    fractions = counts.astype(np.float64) / float(total_pixels)
    minimum_region_pixel_fraction = float(DETEXTURE_MULTI_REGION_FILTER_SETTINGS["minimum_region_pixel_fraction"])
    joker_label_values = tuple(int(value) for value in label_values[fractions < minimum_region_pixel_fraction])
    joker_region_sizes = tuple(int(size) for size in counts[fractions < minimum_region_pixel_fraction])
    joker_pixel_count = int(np.sum(counts[fractions < minimum_region_pixel_fraction]))

    valid_pixel_mask = np.ones(labels.shape, dtype=bool)
    if joker_label_values:
        valid_pixel_mask &= ~np.isin(labels, np.asarray(joker_label_values, dtype=np.int32))
    if int(valid_pixel_mask.sum()) < 1:
        raise ValueError(
            "DeTexture multi joker filtering removed every GT region. "
            f"minimum_region_pixel_fraction={minimum_region_pixel_fraction}"
        )

    filtered_label_map = np.full(labels.shape, fill_value=int(DETEXTURE_MULTI_REGION_FILTER_SETTINGS["joker_label_value"]), dtype=np.int32)
    valid_label_values, valid_counts = np.unique(labels[valid_pixel_mask], return_counts=True)
    order = np.argsort(-valid_counts, kind="stable")
    effective_region_sizes: list[int] = []
    for new_label, old_index in enumerate(order):
        old_label_value = int(valid_label_values[old_index])
        region = labels == old_label_value
        filtered_label_map[region] = int(new_label)
        effective_region_sizes.append(int(region[valid_pixel_mask].sum()))

    diagnostics = {
        "minimum_region_pixel_fraction": minimum_region_pixel_fraction,
        "joker_label_value": int(DETEXTURE_MULTI_REGION_FILTER_SETTINGS["joker_label_value"]),
        "joker_region_count": len(joker_label_values),
        "joker_region_labels": list(joker_label_values),
        "joker_region_sizes": list(joker_region_sizes),
        "joker_pixel_count": joker_pixel_count,
        "joker_pixel_fraction": float(joker_pixel_count / float(total_pixels)),
        "effective_oracle_num_regions": int(np.unique(filtered_label_map[valid_pixel_mask]).size),
        "effective_region_sizes": effective_region_sizes,
        "valid_pixel_count": int(valid_pixel_mask.sum()),
        "invalid_pixel_count": int((~valid_pixel_mask).sum()),
    }
    return filtered_label_map.astype(np.int32), valid_pixel_mask, diagnostics


def _reindex_label_map_by_region_size(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    label_values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remapped = np.empty_like(labels, dtype=np.int32)
    for new_label, old_index in enumerate(order):
        remapped[labels == int(label_values[old_index])] = int(new_label)
    return remapped


def _discover_multi_image_ids(image_dir: Path, mask_dir: Path, split: str) -> tuple[str, ...]:
    image_ids = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    mask_ids = {
        path.stem
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    shared_ids = sorted(image_ids & mask_ids, key=_natural_sort_key)
    if split != "all":
        shared_ids = [image_id for image_id in shared_ids if _split_from_image_id(image_id) == split]
    if not shared_ids:
        raise FileNotFoundError(
            f"No matched DeTexture multi image/mask pairs were found in '{image_dir}' and '{mask_dir}' for split '{split}'."
        )
    missing_images = sorted(mask_ids - image_ids, key=_natural_sort_key)
    missing_masks = sorted(image_ids - mask_ids, key=_natural_sort_key)
    if missing_images or missing_masks:
        LOGGER.warning(
            "DeTexture multi has unmatched image/mask files. Missing images=%s missing masks=%s",
            missing_images[:5],
            missing_masks[:5],
        )
    return tuple(shared_ids)


def _find_asset_path(root: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = root / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find asset '{image_id}' under '{root}'.")


def _split_from_image_id(image_id: str) -> str:
    if image_id.startswith("training_"):
        return "training"
    if image_id.startswith("validation_"):
        return "validation"
    return "all"


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
