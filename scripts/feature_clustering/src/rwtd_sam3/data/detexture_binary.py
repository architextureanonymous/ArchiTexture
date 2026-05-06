"""Local DeTexture ADE20K binary benchmark adapter.

This module resolves the public ``detexture_ADE20K`` asset layout, matches crop
images to their paired ``mask_a`` / ``mask_b`` files, and decodes each example
into a repository-native binary segmentation sample.

Primary entrypoints:
- ``load_detexture_binary_overview()``: inspect the local benchmark root.
- ``iter_detexture_binary_samples()``: stream decoded crop/mask triples.
- ``get_detexture_binary_sample()``: decode one sample by natural-sorted index.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import boundary_from_region_masks


LOGGER = logging.getLogger(__name__)

DETEXTURE_DATASET_ID = "detexture_ade20k"
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DeTextureBinaryOverview:
    """Resolved metadata for one local DeTexture ADE20K asset root."""

    dataset_id: str
    dataset_root: Path
    assets_root: Path
    image_dir: Path
    mask_dir: Path
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str


@dataclass(frozen=True)
class DeTextureBinarySample:
    """One decoded DeTexture ADE20K crop with paired binary masks."""

    index: int
    dataset_id: str
    split: str
    crop_name: str
    image: Image.Image
    boundary_mask: np.ndarray
    texture_a_mask: np.ndarray
    texture_b_mask: np.ndarray
    texture_a: str
    texture_b: str
    original_texture_a: str
    original_texture_b: str
    oracle_points_a: tuple[tuple[int, int], ...]
    oracle_points_b: tuple[tuple[int, int], ...]
    evaluation_view: str

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def width(self) -> int:
        return self.image.size[0]


def load_detexture_binary_overview(dataset_root: str | Path) -> DeTextureBinaryOverview:
    """Inspect a local DeTexture ADE20K dataset root."""

    resolved_root, assets_root, image_dir, mask_dir = resolve_detexture_binary_dirs(dataset_root)
    image_ids = _discover_image_ids(image_dir=image_dir, mask_dir=mask_dir)
    return DeTextureBinaryOverview(
        dataset_id=DETEXTURE_DATASET_ID,
        dataset_root=resolved_root,
        assets_root=assets_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view="partition_invariant",
    )


def iter_detexture_binary_samples(
    dataset_root: str | Path,
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[DeTextureBinarySample]:
    """Yield decoded DeTexture ADE20K samples in natural filename order."""

    overview = load_detexture_binary_overview(dataset_root)
    max_items = max(0, overview.num_examples - start_index) if limit is None else min(
        limit,
        max(0, overview.num_examples - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_detexture_binary_sample_from_overview(overview=overview, index=index)


def get_detexture_binary_sample(
    dataset_root: str | Path,
    index: int,
) -> DeTextureBinarySample:
    """Load one local DeTexture ADE20K crop by natural-sorted index."""

    overview = load_detexture_binary_overview(dataset_root)
    return _decode_detexture_binary_sample_from_overview(overview=overview, index=index)


def resolve_detexture_binary_dirs(
    dataset_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve the dataset root and the canonical ``assets/crops`` / ``assets/masks`` directories."""

    requested_root = Path(dataset_root).expanduser().resolve()
    candidates = (
        requested_root,
        requested_root / "assets",
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        assets_root = candidate if candidate.name == "assets" else candidate / "assets"
        image_dir = assets_root / "crops"
        mask_dir = assets_root / "masks"
        if image_dir.is_dir() and mask_dir.is_dir():
            dataset_dir = candidate.parent if candidate.name == "assets" else candidate
            return dataset_dir, assets_root, image_dir, mask_dir
    raise FileNotFoundError(
        "Could not resolve a DeTexture ADE20K dataset root. Expected either the dataset root containing "
        "`assets/crops` and `assets/masks`, or the `assets/` directory itself. "
        f"Tried from '{requested_root}'."
    )


def _decode_detexture_binary_sample_from_overview(
    overview: DeTextureBinaryOverview,
    index: int,
) -> DeTextureBinarySample:
    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for dataset '{overview.dataset_id}' with "
            f"{overview.num_examples} matched crop/mask triples."
        )
    image_id = overview.image_ids[index]
    image_path = _find_image_path(overview.image_dir, image_id)
    mask_a_path = overview.mask_dir / f"{image_id}_mask_a.png"
    mask_b_path = overview.mask_dir / f"{image_id}_mask_b.png"
    image = Image.open(image_path).convert("RGB")
    texture_a_mask = np.asarray(Image.open(mask_a_path).convert("L"), dtype=np.uint8) > 127
    texture_b_mask = np.asarray(Image.open(mask_b_path).convert("L"), dtype=np.uint8) > 127
    _validate_sample_shapes(image=image, mask_a=texture_a_mask, mask_b=texture_b_mask, image_id=image_id)
    boundary_mask = boundary_from_region_masks(texture_a_mask, texture_b_mask)
    return DeTextureBinarySample(
        index=index,
        dataset_id=overview.dataset_id,
        split="benchmark",
        crop_name=image_id,
        image=image,
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(texture_a_mask, dtype=bool),
        texture_b_mask=np.asarray(texture_b_mask, dtype=bool),
        texture_a="region_a",
        texture_b="region_b",
        original_texture_a="mask_a",
        original_texture_b="mask_b",
        oracle_points_a=(),
        oracle_points_b=(),
        evaluation_view=overview.evaluation_view,
    )


def _discover_image_ids(image_dir: Path, mask_dir: Path) -> tuple[str, ...]:
    image_ids = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    shared_ids: list[str] = []
    missing_mask_a: list[str] = []
    missing_mask_b: list[str] = []
    for image_id in sorted(image_ids, key=_natural_sort_key):
        has_mask_a = (mask_dir / f"{image_id}_mask_a.png").is_file()
        has_mask_b = (mask_dir / f"{image_id}_mask_b.png").is_file()
        if has_mask_a and has_mask_b:
            shared_ids.append(image_id)
            continue
        if not has_mask_a:
            missing_mask_a.append(image_id)
        if not has_mask_b:
            missing_mask_b.append(image_id)
    if missing_mask_a or missing_mask_b:
        LOGGER.warning(
            "DeTexture ADE20K has unmatched crop/mask files. Missing mask_a=%s, missing mask_b=%s",
            missing_mask_a[:5],
            missing_mask_b[:5],
        )
    if not shared_ids:
        raise FileNotFoundError(
            f"No matched crop/mask triples were found in '{image_dir}' and '{mask_dir}'."
        )
    return tuple(shared_ids)


def _find_image_path(image_dir: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find crop image '{image_id}' under '{image_dir}'.")


def _validate_sample_shapes(
    image: Image.Image,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    image_id: str,
) -> None:
    expected_shape = image.size[::-1]
    if mask_a.shape != expected_shape or mask_b.shape != expected_shape:
        raise ValueError(
            f"DeTexture sample '{image_id}' has mismatched image/mask sizes: "
            f"image={image.size}, mask_a={mask_a.shape[::-1]}, mask_b={mask_b.shape[::-1]}."
        )
    if not np.any(mask_a) and not np.any(mask_b):
        raise ValueError(f"DeTexture sample '{image_id}' has both mask_a and mask_b empty.")


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
