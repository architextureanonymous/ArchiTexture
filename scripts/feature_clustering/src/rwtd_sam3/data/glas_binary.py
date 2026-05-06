"""Local GlaS binary benchmark adapter.

This module resolves the AutoSAM-style GlaS layout, matches flat image files to
their paired ``*_anno.bmp`` masks, and decodes each example into a
repository-native binary segmentation sample. The resolver is intentionally
flexible: it accepts either a flat AutoSAM-style directory or an extracted
official Warwick archive that contains the flat files in a descendant folder.

Primary entrypoints:
- ``load_glas_binary_overview()``: inspect one local GlaS root.
- ``iter_glas_binary_samples()``: stream decoded GlaS samples.
- ``get_glas_binary_sample()``: decode one sample by natural-sorted index.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import boundary_from_region_masks


LOGGER = logging.getLogger(__name__)

GLAS_DATASET_ID = "glas"
SUPPORTED_IMAGE_SUFFIXES = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
GLAS_SUPPORTED_SPLITS = ("train", "test", "all")
_IMAGE_STEM_PATTERN = re.compile(r"^(train|testA|testB)_(\d+)$")
_MASK_STEM_PATTERN = re.compile(r"^(train|testA|testB)_(\d+)_anno$")


@dataclass(frozen=True)
class GlasBinaryOverview:
    """Resolved metadata for one local GlaS root and one selected split."""

    dataset_id: str
    dataset_root: Path
    data_dir: Path
    split: str
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str
    grade_csv_path: Path | None


@dataclass(frozen=True)
class GlasBinarySample:
    """One decoded GlaS image with the gland mask and its complement."""

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
    grade_label: str | None

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def width(self) -> int:
        return self.image.size[0]


def load_glas_binary_overview(dataset_root: str | Path, split: str = "test") -> GlasBinaryOverview:
    """Inspect a local GlaS dataset root for one requested split."""

    if split not in GLAS_SUPPORTED_SPLITS:
        expected = ", ".join(GLAS_SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported GlaS split '{split}'. Expected one of: {expected}.")
    resolved_root, data_dir = resolve_glas_binary_dirs(dataset_root)
    image_ids = _discover_image_ids(data_dir=data_dir, split=split)
    grade_csv_path = _resolve_grade_csv_path(resolved_root=resolved_root, data_dir=data_dir)
    return GlasBinaryOverview(
        dataset_id=GLAS_DATASET_ID,
        dataset_root=resolved_root,
        data_dir=data_dir,
        split=split,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view="partition_invariant",
        grade_csv_path=grade_csv_path,
    )


def iter_glas_binary_samples(
    dataset_root: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[GlasBinarySample]:
    """Yield decoded GlaS samples in natural order for one requested split."""

    overview = load_glas_binary_overview(dataset_root=dataset_root, split=split)
    max_items = max(0, overview.num_examples - start_index) if limit is None else min(
        limit,
        max(0, overview.num_examples - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_glas_binary_sample_from_overview(overview=overview, index=index)


def get_glas_binary_sample(
    dataset_root: str | Path,
    *,
    split: str = "test",
    index: int,
) -> GlasBinarySample:
    """Load one local GlaS sample by natural-sorted index."""

    overview = load_glas_binary_overview(dataset_root=dataset_root, split=split)
    return _decode_glas_binary_sample_from_overview(overview=overview, index=index)


def resolve_glas_binary_dirs(dataset_root: str | Path) -> tuple[Path, Path]:
    """Resolve the dataset root and the flat directory that holds GlaS files."""

    requested_root = Path(dataset_root).expanduser().resolve()
    if not requested_root.is_dir():
        raise FileNotFoundError(f"Could not resolve a GlaS dataset root from '{requested_root}'.")

    candidate_dirs: list[Path] = [requested_root]
    candidate_dirs.extend(path for path in requested_root.rglob("*") if path.is_dir())
    best_dir: Path | None = None
    best_count = 0
    for candidate in candidate_dirs:
        shared_count = _count_matched_pairs(candidate)
        if shared_count > best_count:
            best_dir = candidate
            best_count = shared_count
    if best_dir is None or best_count < 1:
        raise FileNotFoundError(
            "Could not resolve a GlaS dataset directory. Expected either a flat AutoSAM-style root "
            "containing `train_*`, `testA_*`, `testB_*`, and `*_anno.bmp` files, or an extracted "
            f"Warwick archive that contains such a directory somewhere below '{requested_root}'."
        )
    return requested_root, best_dir


def _decode_glas_binary_sample_from_overview(
    overview: GlasBinaryOverview,
    index: int,
) -> GlasBinarySample:
    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for dataset '{overview.dataset_id}' split '{overview.split}' "
            f"with {overview.num_examples} matched image/mask pairs."
        )
    image_id = overview.image_ids[index]
    image_path = _find_image_path(overview.data_dir, image_id)
    mask_path = _find_mask_path(overview.data_dir, image_id)
    image = Image.open(image_path).convert("RGB")
    gland_mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    background_mask = ~gland_mask
    _validate_sample_shapes(
        image=image,
        gland_mask=gland_mask,
        background_mask=background_mask,
        image_id=image_id,
    )
    boundary_mask = boundary_from_region_masks(gland_mask, background_mask)
    return GlasBinarySample(
        index=index,
        dataset_id=overview.dataset_id,
        split=_sample_split_name(image_id),
        crop_name=image_id,
        image=image,
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(gland_mask, dtype=bool),
        texture_b_mask=np.asarray(background_mask, dtype=bool),
        texture_a="gland",
        texture_b="background",
        original_texture_a="anno_positive",
        original_texture_b="anno_zero",
        oracle_points_a=(),
        oracle_points_b=(),
        evaluation_view=overview.evaluation_view,
        grade_label=_load_grade_label(overview.grade_csv_path, image_id),
    )


def _count_matched_pairs(data_dir: Path) -> int:
    if not data_dir.is_dir():
        return 0
    image_stems: set[str] = set()
    mask_stems: set[str] = set()
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        stem = path.stem
        if suffix in SUPPORTED_IMAGE_SUFFIXES and _IMAGE_STEM_PATTERN.match(stem):
            image_stems.add(stem)
        elif suffix == ".bmp":
            match = _MASK_STEM_PATTERN.match(stem)
            if match:
                mask_stems.add(f"{match.group(1)}_{match.group(2)}")
    return len(image_stems & mask_stems)


def _discover_image_ids(data_dir: Path, split: str) -> tuple[str, ...]:
    image_ids: set[str] = set()
    missing_masks: list[str] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        stem = path.stem
        match = _IMAGE_STEM_PATTERN.match(stem)
        if match is None:
            continue
        if not _split_matches_prefix(split=split, prefix=match.group(1)):
            continue
        mask_path = data_dir / f"{stem}_anno.bmp"
        if mask_path.is_file():
            image_ids.add(stem)
        else:
            missing_masks.append(stem)
    if missing_masks:
        LOGGER.warning("GlaS has unmatched image/mask files. Missing masks=%s", missing_masks[:5])
    if not image_ids:
        raise FileNotFoundError(
            f"No matched GlaS image/mask pairs were found in '{data_dir}' for split '{split}'."
        )
    return tuple(sorted(image_ids, key=_natural_sort_key))


def _find_image_path(data_dir: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = data_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find GlaS image '{image_id}' under '{data_dir}'.")


def _find_mask_path(data_dir: Path, image_id: str) -> Path:
    candidate = data_dir / f"{image_id}_anno.bmp"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find GlaS mask '{image_id}_anno.bmp' under '{data_dir}'.")


def _validate_sample_shapes(
    image: Image.Image,
    gland_mask: np.ndarray,
    background_mask: np.ndarray,
    image_id: str,
) -> None:
    expected_shape = image.size[::-1]
    if gland_mask.shape != expected_shape or background_mask.shape != expected_shape:
        raise ValueError(
            f"GlaS sample '{image_id}' has mismatched image/mask sizes: "
            f"image={image.size}, gland={gland_mask.shape[::-1]}, background={background_mask.shape[::-1]}."
        )
    if not np.any(gland_mask):
        raise ValueError(f"GlaS sample '{image_id}' has an empty gland region.")
    if not np.any(background_mask):
        raise ValueError(f"GlaS sample '{image_id}' has an empty background region.")


def _split_matches_prefix(*, split: str, prefix: str) -> bool:
    if split == "all":
        return prefix in {"train", "testA", "testB"}
    if split == "train":
        return prefix == "train"
    return prefix in {"testA", "testB"}


def _sample_split_name(image_id: str) -> str:
    prefix = image_id.split("_", 1)[0]
    return "train" if prefix == "train" else "test"


def _resolve_grade_csv_path(*, resolved_root: Path, data_dir: Path) -> Path | None:
    for candidate in (data_dir / "Grade.csv", resolved_root / "Grade.csv"):
        if candidate.is_file():
            return candidate
    return None


def _load_grade_label(grade_csv_path: Path | None, image_id: str) -> str | None:
    if grade_csv_path is None:
        return None
    try:
        with grade_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == image_id and len(row) > 1:
                    return row[1].strip()
    except OSError:
        return None
    return None


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
