"""Local STLD and CAID route adapter for the ArchiTexture binary benchmarks.

This module resolves local ArchiTexture benchmark roots, discovers compatible
image and label directories, and decodes each example into an
``ArchiTextureBinarySample`` that matches the repository's existing
visualization and metric interfaces. It exists so the pooled coarse-only
SAM-feature ablation and the learned coarse-vs-fine probe can run on STLD and
CAID without changing the rest of the evaluation stack.

Primary entrypoints:
- ``load_architexture_binary_overview()``: resolve one route or official split
  and inspect counts.
- ``iter_architexture_binary_samples()``: stream decoded samples in natural or
  split-file order.
- ``get_architexture_binary_sample()``: load one sample by route-local index.

Decoded outputs contain one RGB image, one raw label map, and derived boolean
region and boundary masks of shape ``(height, width)``. The resolver accepts
several common image/label directory names and raises explicit errors for
missing roots, unmatched files, invalid shapes, or unsupported routes.
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

SUPPORTED_ARCHITEXTURE_ROUTES = ("stld", "caid")
SUPPORTED_ARCHITEXTURE_SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_LABEL_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IMAGE_DIR_CANDIDATES = ("images", "image", "imgs", "rgb")
LABEL_DIR_CANDIDATES = ("labels", "labeles", "masks", "mask", "gt", "ground_truth", "groundtruth")


@dataclass(frozen=True)
class ArchiTextureBinaryOverview:
    """Resolved metadata for one local ArchiTexture binary benchmark.

    Attributes:
        route: Benchmark route name, either ``stld`` or ``caid``.
        benchmark_root: Resolved benchmark directory containing the image/label dirs.
        image_dir: Directory containing RGB inputs.
        label_dir: Directory containing the binary or 2-label GT masks.
        num_examples: Number of matched image/label pairs.
        image_ids: Shared file stems in natural order or official split-file order.
        evaluation_view: Paper-facing primary metric view for the route.
        split: Active split name. ``benchmark`` is the legacy flat local view.
        split_layout: Concrete layout that produced this overview.
    """

    route: str
    benchmark_root: Path
    image_dir: Path
    label_dir: Path
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str
    split: str
    split_layout: str


@dataclass(frozen=True)
class ArchiTextureBinarySample:
    """One decoded ArchiTexture binary benchmark sample.

    This mirrors the visualization/evaluation surface already used by RWTD so the
    coarse-only SAM-feature ablation can reuse the existing panel writers.
    """

    index: int
    route: str
    split: str
    crop_name: str
    image: Image.Image
    label_mask: np.ndarray
    label_values: tuple[int, ...]
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


def load_architexture_binary_overview(
    route: str,
    benchmark_root: str | Path,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> ArchiTextureBinaryOverview:
    """Inspect one local ArchiTexture CAID/STLD route or official split."""

    resolved_route = _validate_route(route)
    if split is None:
        if require_official_split:
            raise ValueError("require_official_split=True requires an explicit split.")
        resolved_root, image_dir, label_dir = resolve_architexture_binary_dirs(
            route=resolved_route,
            benchmark_root=benchmark_root,
        )
        image_ids = _discover_image_ids(image_dir=image_dir, label_dir=label_dir)
        resolved_split = "benchmark"
        split_layout = "flat_benchmark_dirs"
    else:
        resolved_split = _validate_split(split)
        resolved_root, image_dir, label_dir, image_ids, split_layout = resolve_architexture_binary_split(
            route=resolved_route,
            benchmark_root=benchmark_root,
            split=resolved_split,
        )
    return ArchiTextureBinaryOverview(
        route=resolved_route,
        benchmark_root=resolved_root,
        image_dir=image_dir,
        label_dir=label_dir,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view=_route_evaluation_view(resolved_route),
        split=resolved_split,
        split_layout=split_layout,
    )


def iter_architexture_binary_samples(
    route: str,
    benchmark_root: str | Path,
    limit: int | None = None,
    start_index: int = 0,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> Iterator[ArchiTextureBinarySample]:
    """Yield decoded ArchiTexture binary samples in natural filename or split-file order."""

    overview = load_architexture_binary_overview(
        route=route,
        benchmark_root=benchmark_root,
        split=split,
        require_official_split=require_official_split,
    )
    max_items = max(0, overview.num_examples - start_index) if limit is None else min(
        limit,
        max(0, overview.num_examples - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_architexture_binary_sample_from_overview(overview=overview, index=index)


def get_architexture_binary_sample(
    benchmark_root: str | Path,
    route: str,
    index: int,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> ArchiTextureBinarySample:
    """Load one local ArchiTexture CAID/STLD benchmark sample by route-local index."""

    overview = load_architexture_binary_overview(
        route=route,
        benchmark_root=benchmark_root,
        split=split,
        require_official_split=require_official_split,
    )
    return _decode_architexture_binary_sample_from_overview(overview=overview, index=index)


def _decode_architexture_binary_sample_from_overview(
    overview: ArchiTextureBinaryOverview,
    index: int,
) -> ArchiTextureBinarySample:
    """Decode one ArchiTexture sample using a pre-resolved overview."""

    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for route '{overview.route}' with "
            f"{overview.num_examples} matched image/label pairs."
        )

    image_id = overview.image_ids[index]
    image_path = _find_sample_path(overview.image_dir, image_id, SUPPORTED_IMAGE_SUFFIXES)
    label_path = _find_sample_path(overview.label_dir, image_id, SUPPORTED_LABEL_SUFFIXES)
    image = Image.open(image_path).convert("RGB")
    label_mask = np.asarray(Image.open(label_path).convert("L"))
    _validate_sample_shapes(image=image, label_mask=label_mask, image_id=image_id)
    label_values = tuple(int(value) for value in np.unique(label_mask))
    decoded = _decode_route_masks(route=overview.route, label_mask=label_mask, label_values=label_values, image_id=image_id)
    return ArchiTextureBinarySample(
        index=index,
        route=overview.route,
        split=overview.split,
        crop_name=image_id,
        image=image,
        label_mask=label_mask,
        label_values=label_values,
        boundary_mask=decoded["boundary_mask"],
        texture_a_mask=decoded["texture_a_mask"],
        texture_b_mask=decoded["texture_b_mask"],
        texture_a=str(decoded["texture_a"]),
        texture_b=str(decoded["texture_b"]),
        original_texture_a=str(decoded["original_texture_a"]),
        original_texture_b=str(decoded["original_texture_b"]),
        oracle_points_a=(),
        oracle_points_b=(),
        evaluation_view=overview.evaluation_view,
    )


def resolve_architexture_binary_dirs(
    route: str,
    benchmark_root: str | Path,
) -> tuple[Path, Path, Path]:
    """Resolve benchmark/image/label directories for the public ArchiTexture routes."""

    resolved_route = _validate_route(route)
    requested_root = Path(benchmark_root).expanduser().resolve()
    candidates = [requested_root]
    if resolved_route == "stld":
        candidates.append(requested_root / "benchmark")
    else:
        candidates.extend(
            [
                requested_root / "benchmarks" / "caid_test",
                requested_root / "caid_test",
            ]
        )

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        discovered = _discover_image_and_label_dirs(candidate)
        if discovered is not None:
            image_dir, label_dir = discovered
            benchmark_dir = candidate
            if image_dir.parent == label_dir.parent:
                benchmark_dir = image_dir.parent
            return benchmark_dir, image_dir, label_dir

    raise FileNotFoundError(
        "Could not resolve an ArchiTexture benchmark root for route "
        f"'{resolved_route}'. Expected either the benchmark directory itself "
        f"or the documented experiment root that contains it. Tried from '{requested_root}'."
    )


def resolve_architexture_binary_split(
    route: str,
    benchmark_root: str | Path,
    split: str,
) -> tuple[Path, Path, Path, tuple[str, ...], str]:
    """Resolve one official ArchiTexture split without falling back to flat benchmarks."""

    resolved_route = _validate_route(route)
    resolved_split = _validate_split(split)
    requested_root = Path(benchmark_root).expanduser().resolve()

    for candidate_root in _iter_architexture_official_root_candidates(resolved_route, requested_root):
        if not candidate_root.is_dir():
            continue
        pascal_like = _try_pascal_voc_split_layout(candidate_root, resolved_split)
        if pascal_like is not None:
            image_dir, label_dir, image_ids = pascal_like
            return candidate_root, image_dir, label_dir, image_ids, "pascal_voc_split_file"
        split_subdirs = _try_split_subdir_layout(candidate_root, resolved_split)
        if split_subdirs is not None:
            image_dir, label_dir, image_ids = split_subdirs
            return candidate_root, image_dir, label_dir, image_ids, "split_subdirs"

    raise FileNotFoundError(
        "Could not resolve an official ArchiTexture split-aware root for route "
        f"'{resolved_route}' split '{resolved_split}' from '{requested_root}'. "
        "Expected either a Pascal-VOC-style layout with "
        "`ImageSets/Segmentation/<split>.txt` plus image/label dirs, or a split "
        "subdirectory layout like `<root>/<split>/images` and `<root>/<split>/labels`."
    )


def _validate_route(route: str) -> str:
    normalized = route.strip().lower()
    if normalized not in SUPPORTED_ARCHITEXTURE_ROUTES:
        expected = ", ".join(SUPPORTED_ARCHITEXTURE_ROUTES)
        raise ValueError(f"Unsupported ArchiTexture route '{route}'. Expected one of: {expected}.")
    return normalized


def _validate_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SUPPORTED_ARCHITEXTURE_SPLITS:
        expected = ", ".join(SUPPORTED_ARCHITEXTURE_SPLITS)
        raise ValueError(f"Unsupported ArchiTexture split '{split}'. Expected one of: {expected}.")
    return normalized


def _route_evaluation_view(route: str) -> str:
    return "direct_foreground" if route == "stld" else "partition_invariant"


def _iter_architexture_official_root_candidates(route: str, requested_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = [requested_root]
    if route == "stld":
        candidates.extend(
            [
                requested_root / "STLD",
                requested_root / "stld",
                requested_root / "benchmark",
            ]
        )
    else:
        candidates.extend(
            [
                requested_root / "CAID",
                requested_root / "caid",
                requested_root / "caid_test",
                requested_root / "benchmarks" / "caid_test",
            ]
        )
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _discover_image_and_label_dirs(root: Path) -> tuple[Path, Path] | None:
    direct_image_dir = _first_named_dir(root, IMAGE_DIR_CANDIDATES)
    direct_label_dir = _first_named_dir(root, LABEL_DIR_CANDIDATES)
    if direct_image_dir is not None and direct_label_dir is not None:
        return direct_image_dir, direct_label_dir

    image_dirs = _collect_named_dirs(root, IMAGE_DIR_CANDIDATES)
    label_dirs = _collect_named_dirs(root, LABEL_DIR_CANDIDATES)
    if not image_dirs or not label_dirs:
        return None

    image_dirs.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    label_dirs.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    return image_dirs[0], label_dirs[0]


def _try_split_subdir_layout(
    root: Path,
    split: str,
) -> tuple[Path, Path, tuple[str, ...]] | None:
    split_root = root / split
    if not split_root.is_dir():
        return None
    discovered = _discover_image_and_label_dirs(split_root)
    if discovered is None:
        return None
    image_dir, label_dir = discovered
    return image_dir, label_dir, _discover_image_ids(image_dir=image_dir, label_dir=label_dir)


def _try_pascal_voc_split_layout(
    root: Path,
    split: str,
) -> tuple[Path, Path, tuple[str, ...]] | None:
    split_file_candidates = (
        root / "ImageSets" / "Segmentation" / f"{split}.txt",
        root / "ImageSets" / f"{split}.txt",
    )
    split_file = next((candidate for candidate in split_file_candidates if candidate.is_file()), None)
    if split_file is None:
        return None

    image_dir = _first_existing_dir(
        (
            root / "JPEGImages",
            root / "images",
            root / "Images",
            root / "rgb",
        )
    )
    label_dir = _first_existing_dir(
        (
            root / "SegmentationClass",
            root / "labels",
            root / "Labels",
            root / "masks",
        )
    )
    if image_dir is None or label_dir is None:
        return None

    image_ids = _load_split_file_image_ids(split_file)
    _validate_split_membership(image_ids=image_ids, image_dir=image_dir, label_dir=label_dir, split_file=split_file)
    return image_dir, label_dir, image_ids


def _first_named_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _first_existing_dir(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _collect_named_dirs(root: Path, candidates: tuple[str, ...]) -> list[Path]:
    results: list[Path] = []
    for name in candidates:
        results.extend(path for path in root.rglob(name) if path.is_dir())
    return results


def _discover_image_ids(image_dir: Path, label_dir: Path) -> tuple[str, ...]:
    image_ids = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    label_ids = {
        path.stem
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_LABEL_SUFFIXES
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
            "ArchiTexture benchmark root has unmatched files. Missing images for labels=%s, missing labels for images=%s",
            missing_images[:5],
            missing_labels[:5],
        )
    return shared_ids


def _load_split_file_image_ids(split_file: Path) -> tuple[str, ...]:
    image_ids = tuple(line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not image_ids:
        raise FileNotFoundError(f"Split file '{split_file}' did not contain any sample ids.")
    return image_ids


def _validate_split_membership(
    *,
    image_ids: tuple[str, ...],
    image_dir: Path,
    label_dir: Path,
    split_file: Path,
) -> None:
    missing_images = [image_id for image_id in image_ids if not _sample_path_exists(image_dir, image_id, SUPPORTED_IMAGE_SUFFIXES)]
    missing_labels = [image_id for image_id in image_ids if not _sample_path_exists(label_dir, image_id, SUPPORTED_LABEL_SUFFIXES)]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"Official split file '{split_file}' referenced ids that were missing from the resolved image/label dirs. "
            f"Missing images={missing_images[:5]}, missing labels={missing_labels[:5]}."
        )


def _find_sample_path(directory: Path, image_id: str, suffixes: tuple[str, ...]) -> Path:
    for suffix in suffixes:
        candidate = directory / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find sample '{image_id}' under '{directory}'.")


def _sample_path_exists(directory: Path, image_id: str, suffixes: tuple[str, ...]) -> bool:
    return any((directory / f"{image_id}{suffix}").exists() for suffix in suffixes)


def _validate_sample_shapes(image: Image.Image, label_mask: np.ndarray, image_id: str) -> None:
    if image.size != label_mask.shape[::-1]:
        raise ValueError(
            f"ArchiTexture sample '{image_id}' has mismatched image and label sizes: "
            f"{image.size} vs {label_mask.shape[::-1]}."
        )


def _decode_route_masks(
    route: str,
    label_mask: np.ndarray,
    label_values: tuple[int, ...],
    image_id: str,
) -> dict[str, np.ndarray | str]:
    labels = np.asarray(label_mask)
    if route == "stld":
        if len(label_values) < 2:
            raise ValueError(
                f"STLD sample '{image_id}' must contain at least background and foreground labels, got {label_values}."
            )
        foreground_value = next((value for value in label_values if value != 0), label_values[-1])
        texture_a_mask = labels == foreground_value
        if not np.any(texture_a_mask):
            texture_a_mask = labels != label_values[0]
        texture_b_mask = np.logical_not(texture_a_mask)
        if not np.any(texture_a_mask) or not np.any(texture_b_mask):
            raise ValueError(
                f"STLD sample '{image_id}' collapsed into an empty foreground/background split after decoding {label_values}."
            )
        boundary_mask = boundary_from_region_masks(texture_a_mask, texture_b_mask)
        return {
            "texture_a_mask": np.asarray(texture_a_mask, dtype=bool),
            "texture_b_mask": np.asarray(texture_b_mask, dtype=bool),
            "boundary_mask": np.asarray(boundary_mask, dtype=bool),
            "texture_a": "foreground",
            "texture_b": "background",
            "original_texture_a": f"label_{foreground_value}",
            "original_texture_b": "background",
        }

    if len(label_values) > 2:
        raise ValueError(
            f"CAID sample '{image_id}' must contain at most 2 labels in the local binary shoreline benchmark, "
            f"got {label_values}."
        )
    texture_a_mask = labels == label_values[0]
    if len(label_values) == 2:
        texture_b_mask = labels == label_values[1]
        original_texture_b = f"label_{label_values[1]}"
    else:
        texture_b_mask = np.logical_not(texture_a_mask)
        original_texture_b = "empty_complement"
    boundary_mask = boundary_from_region_masks(texture_a_mask, texture_b_mask)
    return {
        "texture_a_mask": np.asarray(texture_a_mask, dtype=bool),
        "texture_b_mask": np.asarray(texture_b_mask, dtype=bool),
        "boundary_mask": np.asarray(boundary_mask, dtype=bool),
        "texture_a": "region_a",
        "texture_b": "region_b",
        "original_texture_a": f"label_{label_values[0]}",
        "original_texture_b": original_texture_b,
    }


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
