"""Local loader for the official TextureSAM ``Kaust256`` RWTD release.

This module adapts the paper-faithful local RWTD directory used by
``eval-sam2-official`` and ``predict-sam2-official``. It resolves the dataset
root, validates the expected image/label pairing, and decodes each sample into
the lightweight ``Kaust256Sample`` structure consumed by the official
TextureSAM reproduction path.

Primary entrypoints:
- ``load_kaust256_overview()``: inspect a resolved dataset root.
- ``iter_kaust256_samples()``: stream decoded samples in natural filename order.
- ``get_kaust256_sample()``: load one sample by natural-sorted index.

Each sample contains one RGB image of size ``(256, 256)`` and one integer label
map of shape ``(256, 256)`` with exactly two labels. Missing directories,
unmatched image/label pairs, invalid sizes, and non-binary label maps are
reported as explicit exceptions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

EXPECTED_IMAGE_SIZE = (256, 256)


@dataclass(frozen=True)
class Kaust256Overview:
    """Metadata discovered for an official TextureSAM RWTD directory.

    Attributes:
        root_dir: Resolved dataset root directory containing ``images`` and ``labeles``.
        image_dir: Directory containing ``*.jpg`` RGB inputs.
        label_dir: Directory containing binary ``*.png`` label maps.
        num_examples: Number of matched image/label pairs.
        image_ids: Natural-sorted image identifiers without extensions.
    """

    root_dir: Path
    image_dir: Path
    label_dir: Path
    num_examples: int
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class Kaust256Sample:
    """One official TextureSAM RWTD sample from the Kaust256 release.

    Attributes:
        index: Zero-based sample index after natural sorting.
        image_id: Filename stem shared by the image and label map.
        image: RGB image of size ``(width, height)``.
        label_mask: Integer label map of shape ``(height, width)``.
        label_values: Sorted unique label values present in ``label_mask``.
    """

    index: int
    image_id: str
    image: Image.Image
    label_mask: np.ndarray
    label_values: tuple[int, ...]

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def region_masks(self) -> tuple[np.ndarray, ...]:
        """Return one boolean mask per unique label value."""

        return tuple(self.label_mask == label_value for label_value in self.label_values)


def load_kaust256_overview(root_dir: str | Path) -> Kaust256Overview:
    """Inspect a local Kaust256 directory and return discovered metadata."""

    resolved_root, image_dir, label_dir = resolve_kaust256_dirs(root_dir)
    image_ids = _discover_image_ids(image_dir=image_dir, label_dir=label_dir)
    return Kaust256Overview(
        root_dir=resolved_root,
        image_dir=image_dir,
        label_dir=label_dir,
        num_examples=len(image_ids),
        image_ids=image_ids,
    )


def iter_kaust256_samples(root_dir: str | Path, limit: int | None = None) -> Iterator[Kaust256Sample]:
    """Yield decoded Kaust256 samples in natural filename order."""

    overview = load_kaust256_overview(root_dir)
    max_items = overview.num_examples if limit is None else min(limit, overview.num_examples)
    for index in range(max_items):
        yield get_kaust256_sample(root_dir=overview.root_dir, index=index)


def get_kaust256_sample(root_dir: str | Path, index: int) -> Kaust256Sample:
    """Load one Kaust256 sample by natural-sorted row index."""

    overview = load_kaust256_overview(root_dir)
    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for Kaust256 with {overview.num_examples} images."
        )

    image_id = overview.image_ids[index]
    image_path = overview.image_dir / f"{image_id}.jpg"
    label_path = overview.label_dir / f"{image_id}.png"
    image = Image.open(image_path).convert("RGB")
    label_mask = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
    _validate_sample_shapes(image=image, label_mask=label_mask, image_id=image_id)
    label_values = tuple(int(value) for value in np.unique(label_mask))
    if len(label_values) != 2:
        raise ValueError(
            f"Expected Kaust256 sample '{image_id}' to contain exactly 2 labels, got {label_values}."
        )
    return Kaust256Sample(
        index=index,
        image_id=image_id,
        image=image,
        label_mask=label_mask,
        label_values=label_values,
    )


def resolve_kaust256_dirs(root_dir: str | Path) -> tuple[Path, Path, Path]:
    """Resolve the official TextureSAM Kaust256 dataset directories.

    Inputs:
        root_dir: Either the ``Kaust256`` directory itself or the parent repo/root
            directory that contains it.

    Returns:
        Tuple ``(dataset_root, image_dir, label_dir)``.
    """

    requested_root = Path(root_dir).expanduser().resolve()
    candidates = (requested_root, requested_root / "Kaust256")
    for candidate in candidates:
        image_dir = candidate / "images"
        label_dir = candidate / "labeles"
        if image_dir.is_dir() and label_dir.is_dir():
            return candidate, image_dir, label_dir
        fallback_label_dir = candidate / "labels"
        if image_dir.is_dir() and fallback_label_dir.is_dir():
            return candidate, image_dir, fallback_label_dir

    raise FileNotFoundError(
        "Could not resolve a Kaust256 dataset root. Expected either "
        "`<root>/images` with `<root>/labeles` or `<root>/labels`, or `<root>/Kaust256/...`."
    )


def _discover_image_ids(image_dir: Path, label_dir: Path) -> tuple[str, ...]:
    image_ids = {
        image_path.stem
        for image_path in image_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    label_ids = {
        label_path.stem
        for label_path in label_dir.iterdir()
        if label_path.is_file() and label_path.suffix.lower() == ".png"
    }
    shared_ids = tuple(sorted(image_ids & label_ids, key=_natural_sort_key))
    if not shared_ids:
        raise FileNotFoundError(
            f"No matched image/label pairs were found in '{image_dir}' and '{label_dir}'."
        )

    missing_images = sorted(label_ids - image_ids, key=_natural_sort_key)
    missing_labels = sorted(image_ids - label_ids, key=_natural_sort_key)
    if missing_images or missing_labels:
        LOGGER.warning(
            "Kaust256 root has unmatched files. Missing images for labels=%s, missing labels for images=%s",
            missing_images[:5],
            missing_labels[:5],
        )
    return shared_ids


def _validate_sample_shapes(image: Image.Image, label_mask: np.ndarray, image_id: str) -> None:
    if image.size != label_mask.shape[::-1]:
        raise ValueError(
            f"Kaust256 sample '{image_id}' has mismatched image and label sizes: "
            f"{image.size} vs {label_mask.shape[::-1]}."
        )
    if image.size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Kaust256 sample '{image_id}' has unexpected size {image.size}. "
            f"Expected {EXPECTED_IMAGE_SIZE}."
        )


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
