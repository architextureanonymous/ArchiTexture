"""Local CSTD binary benchmark adapter.

This module resolves local ``CSTD`` layouts in two modes:

- legacy flat benchmark roots with `images/`, `regions/`, and `edges/`
- official split-aware roots used by the Stage-2 coarse-vs-fine probe study

Primary entrypoints:
- ``load_cstd_binary_overview()``: inspect one flat benchmark root or one
  split-aware local root.
- ``iter_cstd_binary_samples()``: stream decoded image/region/edge triples.
- ``get_cstd_binary_sample()``: decode one sample by index.
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

CSTD_DATASET_ID = "cstd"
SUPPORTED_CSTD_SPLITS = ("train", "val", "test")
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class CSTDBinaryOverview:
    """Resolved metadata for one local CSTD root."""

    dataset_id: str
    dataset_root: Path
    image_dir: Path
    region_dir: Path
    edge_dir: Path
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str
    split: str
    split_layout: str


@dataclass(frozen=True)
class CSTDBinarySample:
    """One decoded CSTD image with a binary region mask and its complement."""

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


def load_cstd_binary_overview(
    dataset_root: str | Path,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> CSTDBinaryOverview:
    """Inspect a local CSTD dataset root."""

    if split is None:
        if require_official_split:
            raise ValueError("require_official_split=True requires an explicit split.")
        resolved_root, image_dir, region_dir, edge_dir = resolve_cstd_binary_dirs(dataset_root)
        image_ids = _discover_image_ids(image_dir=image_dir, region_dir=region_dir, edge_dir=edge_dir)
        resolved_split = "benchmark"
        split_layout = "flat_benchmark_dirs"
    else:
        resolved_split = _validate_split(split)
        resolved_root, image_dir, region_dir, edge_dir, image_ids, split_layout = resolve_cstd_binary_split(
            dataset_root=dataset_root,
            split=resolved_split,
        )
    return CSTDBinaryOverview(
        dataset_id=CSTD_DATASET_ID,
        dataset_root=resolved_root,
        image_dir=image_dir,
        region_dir=region_dir,
        edge_dir=edge_dir,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view="partition_invariant",
        split=resolved_split,
        split_layout=split_layout,
    )


def iter_cstd_binary_samples(
    dataset_root: str | Path,
    limit: int | None = None,
    start_index: int = 0,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> Iterator[CSTDBinarySample]:
    """Yield decoded CSTD samples in natural filename order."""

    overview = load_cstd_binary_overview(
        dataset_root,
        split=split,
        require_official_split=require_official_split,
    )
    max_items = max(0, overview.num_examples - start_index) if limit is None else min(
        limit,
        max(0, overview.num_examples - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_cstd_binary_sample_from_overview(overview=overview, index=index)


def get_cstd_binary_sample(
    dataset_root: str | Path,
    index: int,
    *,
    split: str | None = None,
    require_official_split: bool = False,
) -> CSTDBinarySample:
    """Load one local CSTD sample by natural-sorted index."""

    overview = load_cstd_binary_overview(
        dataset_root,
        split=split,
        require_official_split=require_official_split,
    )
    return _decode_cstd_binary_sample_from_overview(overview=overview, index=index)


def resolve_cstd_binary_dirs(
    dataset_root: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve the dataset root and the canonical ``images`` / ``regions`` / ``edges`` directories."""

    requested_root = Path(dataset_root).expanduser().resolve()
    if not requested_root.is_dir():
        raise FileNotFoundError(f"Could not resolve a CSTD dataset root from '{requested_root}'.")
    image_dir = requested_root / "images"
    region_dir = requested_root / "regions"
    edge_dir = requested_root / "edges"
    if image_dir.is_dir() and region_dir.is_dir() and edge_dir.is_dir():
        return requested_root, image_dir, region_dir, edge_dir
    raise FileNotFoundError(
        "Could not resolve a CSTD dataset root. Expected a directory containing "
        "`images/`, `regions/`, and `edges/`. "
        f"Tried from '{requested_root}'."
    )


def resolve_cstd_binary_split(
    dataset_root: str | Path,
    split: str,
) -> tuple[Path, Path, Path, Path, tuple[str, ...], str]:
    """Resolve one official CSTD split without falling back to flat benchmarks."""

    requested_root = Path(dataset_root).expanduser().resolve()
    resolved_split = _validate_split(split)
    for candidate_root in _iter_cstd_official_root_candidates(requested_root):
        if not candidate_root.is_dir():
            continue
        split_file_layout = _try_cstd_split_file_layout(candidate_root, resolved_split)
        if split_file_layout is not None:
            image_dir, region_dir, edge_dir, image_ids = split_file_layout
            return candidate_root, image_dir, region_dir, edge_dir, image_ids, "split_file"
        split_subdir_layout = _try_cstd_split_subdir_layout(candidate_root, resolved_split)
        if split_subdir_layout is not None:
            image_dir, region_dir, edge_dir, image_ids = split_subdir_layout
            return candidate_root, image_dir, region_dir, edge_dir, image_ids, "split_subdirs"
    raise FileNotFoundError(
        "Could not resolve an official CSTD split-aware root for "
        f"split '{resolved_split}' from '{requested_root}'. Expected either "
        "`ImageSets/Segmentation/<split>.txt` plus flat `images/`, `regions/`, `edges/`, "
        "or split subdirectories like `<root>/<split>/images`, `<root>/<split>/regions`, and `<root>/<split>/edges`."
    )


def _decode_cstd_binary_sample_from_overview(
    overview: CSTDBinaryOverview,
    index: int,
) -> CSTDBinarySample:
    if index < 0 or index >= overview.num_examples:
        raise IndexError(
            f"Sample index {index} is out of range for dataset '{overview.dataset_id}' with "
            f"{overview.num_examples} matched image/region/edge triples."
        )
    image_id = overview.image_ids[index]
    image_path = _find_image_path(overview.image_dir, image_id)
    region_path = overview.region_dir / f"{image_id}.png"
    edge_path = overview.edge_dir / f"{image_id}.png"
    image = Image.open(image_path).convert("RGB")
    region_mask = np.asarray(Image.open(region_path).convert("L"), dtype=np.uint8) > 127
    edge_mask = np.asarray(Image.open(edge_path).convert("L"), dtype=np.uint8) > 127
    complement_mask = ~region_mask
    _validate_sample_shapes(
        image=image,
        region_mask=region_mask,
        complement_mask=complement_mask,
        edge_mask=edge_mask,
        image_id=image_id,
    )
    return CSTDBinarySample(
        index=index,
        dataset_id=overview.dataset_id,
        split=overview.split,
        crop_name=image_id,
        image=image,
        boundary_mask=np.asarray(edge_mask, dtype=bool),
        texture_a_mask=np.asarray(region_mask, dtype=bool),
        texture_b_mask=np.asarray(complement_mask, dtype=bool),
        texture_a="region",
        texture_b="complement",
        original_texture_a="region_255",
        original_texture_b="region_0",
        oracle_points_a=(),
        oracle_points_b=(),
        evaluation_view=overview.evaluation_view,
    )


def _discover_image_ids(image_dir: Path, region_dir: Path, edge_dir: Path) -> tuple[str, ...]:
    image_ids = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    shared_ids: list[str] = []
    missing_region: list[str] = []
    missing_edge: list[str] = []
    for image_id in sorted(image_ids, key=_natural_sort_key):
        has_region = (region_dir / f"{image_id}.png").is_file()
        has_edge = (edge_dir / f"{image_id}.png").is_file()
        if has_region and has_edge:
            shared_ids.append(image_id)
            continue
        if not has_region:
            missing_region.append(image_id)
        if not has_edge:
            missing_edge.append(image_id)
    if missing_region or missing_edge:
        LOGGER.warning(
            "CSTD has unmatched image/region/edge files. Missing region=%s, missing edge=%s",
            missing_region[:5],
            missing_edge[:5],
        )
    if not shared_ids:
        raise FileNotFoundError(
            f"No matched image/region/edge triples were found in '{image_dir}', '{region_dir}', and '{edge_dir}'."
        )
    return tuple(shared_ids)


def _load_split_file_image_ids(split_file: Path) -> tuple[str, ...]:
    image_ids = tuple(line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not image_ids:
        raise FileNotFoundError(f"Split file '{split_file}' did not contain any sample ids.")
    return image_ids


def _validate_split_membership(
    *,
    image_ids: tuple[str, ...],
    image_dir: Path,
    region_dir: Path,
    edge_dir: Path,
    split_file: Path,
) -> None:
    missing_images = [image_id for image_id in image_ids if not _image_path_exists(image_dir, image_id)]
    missing_regions = [image_id for image_id in image_ids if not (region_dir / f"{image_id}.png").is_file()]
    missing_edges = [image_id for image_id in image_ids if not (edge_dir / f"{image_id}.png").is_file()]
    if missing_images or missing_regions or missing_edges:
        raise FileNotFoundError(
            f"Official CSTD split file '{split_file}' referenced ids that were missing from the resolved image/region/edge dirs. "
            f"Missing images={missing_images[:5]}, missing regions={missing_regions[:5]}, missing edges={missing_edges[:5]}."
        )


def _validate_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SUPPORTED_CSTD_SPLITS:
        expected = ", ".join(SUPPORTED_CSTD_SPLITS)
        raise ValueError(f"Unsupported CSTD split '{split}'. Expected one of: {expected}.")
    return normalized


def _iter_cstd_official_root_candidates(requested_root: Path) -> tuple[Path, ...]:
    candidates = [requested_root, requested_root / "CSTD", requested_root / "cstd"]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _try_cstd_split_subdir_layout(
    root: Path,
    split: str,
) -> tuple[Path, Path, Path, tuple[str, ...]] | None:
    split_root = root / split
    if not split_root.is_dir():
        return None
    try:
        _, image_dir, region_dir, edge_dir = resolve_cstd_binary_dirs(split_root)
    except FileNotFoundError:
        return None
    return image_dir, region_dir, edge_dir, _discover_image_ids(
        image_dir=image_dir,
        region_dir=region_dir,
        edge_dir=edge_dir,
    )


def _try_cstd_split_file_layout(
    root: Path,
    split: str,
) -> tuple[Path, Path, Path, tuple[str, ...]] | None:
    split_file_candidates = (
        root / "ImageSets" / "Segmentation" / f"{split}.txt",
        root / "ImageSets" / f"{split}.txt",
        root / "splits" / f"{split}.txt",
    )
    split_file = next((candidate for candidate in split_file_candidates if candidate.is_file()), None)
    if split_file is None:
        return None
    try:
        _, image_dir, region_dir, edge_dir = resolve_cstd_binary_dirs(root)
    except FileNotFoundError:
        return None
    image_ids = _load_split_file_image_ids(split_file)
    _validate_split_membership(
        image_ids=image_ids,
        image_dir=image_dir,
        region_dir=region_dir,
        edge_dir=edge_dir,
        split_file=split_file,
    )
    return image_dir, region_dir, edge_dir, image_ids


def _find_image_path(image_dir: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image '{image_id}' under '{image_dir}'.")


def _image_path_exists(image_dir: Path, image_id: str) -> bool:
    return any((image_dir / f"{image_id}{suffix}").is_file() for suffix in SUPPORTED_IMAGE_SUFFIXES)


def _validate_sample_shapes(
    image: Image.Image,
    region_mask: np.ndarray,
    complement_mask: np.ndarray,
    edge_mask: np.ndarray,
    image_id: str,
) -> None:
    expected_shape = image.size[::-1]
    if region_mask.shape != expected_shape or complement_mask.shape != expected_shape or edge_mask.shape != expected_shape:
        raise ValueError(
            f"CSTD sample '{image_id}' has mismatched image/mask sizes: "
            f"image={image.size}, region={region_mask.shape[::-1]}, complement={complement_mask.shape[::-1]}, "
            f"edge={edge_mask.shape[::-1]}."
        )
    if not np.any(region_mask):
        raise ValueError(f"CSTD sample '{image_id}' has an empty foreground region.")
    if not np.any(complement_mask):
        raise ValueError(f"CSTD sample '{image_id}' has an empty complement region.")


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]
