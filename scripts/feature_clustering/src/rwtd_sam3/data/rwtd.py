"""RWTD dataset loading, caching, and row decoding helpers.

This module is the canonical RWTD data adapter for the repository. It resolves
the repository-default curated RWTD bundle, supports the legacy public Hugging
Face RWTD dataset as an explicit override, handles a synthetic ``all`` split,
and decodes raw rows into strongly typed ``RwtdSample`` objects used by every
RWTD-backed evaluation path.

Primary entrypoints:
- ``load_split_overview()``: inspect one RWTD view without decoding all rows.
- ``iter_rwtd_samples()``: stream decoded ``RwtdSample`` objects.
- ``get_rwtd_sample()``: load and decode one sample by split-local index.

Decoded outputs contain one RGB ``PIL.Image.Image`` plus boolean masks of shape
``(height, width)`` and oracle-point tuples in ``(x, y)`` pixel coordinates.
This module depends on ``datasets`` at runtime and falls back to cached Arrow
artifacts when live dataset loading fails but local cache files already exist.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED_RWTD_DATASET_ID = "anonymous/rwtd"
CURATED_RWTD_LOCAL_ALIAS = "rwtd"
LEGACY_PUBLIC_RWTD_DATASET_ID = "aviadcohz/RWTD"
DEFAULT_DATASET_ID = CURATED_RWTD_DATASET_ID
DEFAULT_EVAL_SPLIT = "test"
COMBINED_SPLIT = "all"
DEFAULT_CACHE_DIR = Path(".cache") / "huggingface"
HF_GLOBAL_DATASETS_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"
PUBLIC_RWTD_SPLITS = ("train", "test")
SUPPORTED_SPLITS = ("train", "test", COMBINED_SPLIT)
SOURCE_SPLIT_COLUMN = "_rwtd_source_split"
CURATED_LOCAL_BUNDLE_DIR = REPO_ROOT / "datasets" / "rwtd"
CURATED_FLAT_SOURCE_DIR = REPO_ROOT / "outputs" / "rwtd_manual_curation_2026-04-09" / "curated_dataset"
CURATED_SPLIT_DIR = REPO_ROOT / "experiments" / "dataset_splits" / "rwtd_curated_train32_test215_seed0"
CURATED_LOCAL_DATASET_IDS = frozenset({CURATED_RWTD_DATASET_ID, CURATED_RWTD_LOCAL_ALIAS})
EXPECTED_COLUMNS = (
    "image",
    "boundary_mask",
    "texture_a_mask",
    "texture_b_mask",
    "texture_a",
    "texture_b",
    "original_texture_a",
    "original_texture_b",
    "crop_name",
    "oracle_points_a",
    "oracle_points_b",
)


class RwtdDependencyError(RuntimeError):
    """Raised when the Hugging Face datasets dependency is unavailable."""


@dataclass(frozen=True)
class DatasetSplitOverview:
    """Metadata discovered for one requested RWTD dataset view."""

    dataset_id: str
    split: str
    num_examples: int
    features: tuple[str, ...]
    split_sizes: dict[str, int]


@dataclass(frozen=True)
class RwtdSample:
    """A fully decoded RWTD sample ready for evaluation.

    Attributes:
        index: Zero-based row index within the requested dataset view.
        split: Source dataset split name such as ``train`` or ``test``.
        crop_name: Stable sample identifier from the dataset.
        image: RGB image with size ``(width, height)``.
        boundary_mask: Boolean mask of shape ``(height, width)``.
        texture_a_mask: Boolean mask of shape ``(height, width)``.
        texture_b_mask: Boolean mask of shape ``(height, width)``.
        texture_a: Natural-language description for texture A.
        texture_b: Natural-language description for texture B.
        original_texture_a: Short source label for texture A.
        original_texture_b: Short source label for texture B.
        oracle_points_a: Positive prompt points for texture A in ``(x, y)`` pixel coordinates.
        oracle_points_b: Positive prompt points for texture B in ``(x, y)`` pixel coordinates.
    """

    index: int
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

    @property
    def height(self) -> int:
        """Return the sample height in pixels."""

        return self.image.size[1]

    @property
    def width(self) -> int:
        """Return the sample width in pixels."""

        return self.image.size[0]


class RwtdDecodedDataset:
    """Lazy map-style RWTD dataset suitable for ``torch.utils.data.DataLoader``.

    This adapter stores only split-selection metadata up front so worker
    processes can lazily open the backing dataset on first access. Each
    ``__getitem__`` call decodes one row into an ``RwtdSample``.
    """

    def __init__(
        self,
        split: str,
        dataset_id: str = DEFAULT_DATASET_ID,
        cache_dir: str | None = None,
        limit: int | None = None,
        length: int | None = None,
        start_index: int = 0,
    ) -> None:
        if limit is not None and limit < 1:
            raise ValueError(f"--limit must be positive when provided, got {limit}")
        if start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {start_index}")
        self.split = split
        self.dataset_id = dataset_id
        self.cache_dir = cache_dir
        self.limit = limit
        self._length = length
        self.start_index = start_index
        self._dataset = None

    def __len__(self) -> int:
        if self._length is None:
            dataset = self._ensure_dataset()
            available = max(0, len(dataset) - self.start_index)
            self._length = available if self.limit is None else min(self.limit, available)
        return self._length

    def __getitem__(self, index: int) -> RwtdSample:
        if index < 0 or index >= len(self):
            raise IndexError(f"RWTD sample index {index} is out of range for dataset length {len(self)}.")
        dataset = self._ensure_dataset()
        raw_index = self.start_index + index
        row = dataset[raw_index]
        source_split = str(row.get(SOURCE_SPLIT_COLUMN, self.split))
        return decode_sample(row, index=raw_index, split=source_split)

    def _ensure_dataset(self):
        if self._dataset is None:
            self._dataset = load_split_dataset(
                split=self.split,
                dataset_id=self.dataset_id,
                cache_dir=self.cache_dir,
            )
        return self._dataset


def _require_datasets():
    try:
        from datasets import concatenate_datasets, load_dataset, load_dataset_builder
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime
        raise RwtdDependencyError(
            "RWTD loading requires the 'datasets' package. Install project dependencies first."
        ) from exc
    return load_dataset, load_dataset_builder, concatenate_datasets


def load_split_overview(
    split: str,
    dataset_id: str = DEFAULT_DATASET_ID,
    cache_dir: str | None = None,
) -> DatasetSplitOverview:
    """Return metadata for one requested RWTD dataset view."""

    validate_split_name(split)
    local_root = _resolve_local_rwtd_root(dataset_id)
    if local_root is not None:
        if _is_raw_rwtd_root(local_root):
            return _load_raw_split_overview(split=split, dataset_id=dataset_id, raw_root=local_root)
        return _load_local_split_overview(split=split, dataset_id=dataset_id, bundle_dir=local_root)

    _, load_dataset_builder, _ = _require_datasets()
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    try:
        builder = load_dataset_builder(dataset_id, cache_dir=resolved_cache_dir)
    except Exception as exc:
        cached_overview = _load_cached_split_overview(split=split, dataset_id=dataset_id, cache_dir=resolved_cache_dir)
        if cached_overview is not None:
            LOGGER.warning(
                "Falling back to cached RWTD overview from Arrow files because load_dataset_builder failed: %s",
                exc,
            )
            return cached_overview
        raise

    split_sizes = {
        split_name: int(split_info.num_examples)
        for split_name, split_info in builder.info.splits.items()
    }
    return _build_overview(
        dataset_id=str(dataset_id),
        split=split,
        split_sizes=split_sizes,
        features=tuple(builder.info.features.keys()),
    )


def load_split_dataset(
    split: str,
    dataset_id: str = DEFAULT_DATASET_ID,
    cache_dir: str | None = None,
):
    """Load a raw RWTD dataset view.

    The special split name ``all`` concatenates the public or curated ``train``
    and ``test`` splits in that order and adds a private source-split column
    used only during local decoding.
    """

    validate_split_name(split)
    local_root = _resolve_local_rwtd_root(dataset_id)
    if local_root is not None:
        if _is_raw_rwtd_root(local_root):
            return _load_raw_split_dataset(split=split, raw_root=local_root)
        return _load_local_split_dataset(split=split, bundle_dir=local_root)

    load_dataset, _, concatenate_datasets = _require_datasets()
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    try:
        if split == COMBINED_SPLIT:
            datasets = []
            for source_split in PUBLIC_RWTD_SPLITS:
                dataset = load_dataset(dataset_id, split=source_split, cache_dir=resolved_cache_dir)
                dataset = _attach_source_split(dataset, source_split, overwrite=True)
                datasets.append(dataset)
            return concatenate_datasets(datasets)

        dataset = load_dataset(dataset_id, split=split, cache_dir=resolved_cache_dir)
        return _attach_source_split(dataset, split, overwrite=True)
    except Exception as exc:
        cached_dataset = _load_cached_split_dataset(
            split=split,
            dataset_id=dataset_id,
            cache_dir=resolved_cache_dir,
        )
        if cached_dataset is not None:
            LOGGER.warning(
                "Falling back to cached RWTD Arrow files because load_dataset failed: %s",
                exc,
            )
            return cached_dataset
        raise


def iter_rwtd_samples(
    split: str,
    dataset_id: str = DEFAULT_DATASET_ID,
    cache_dir: str | None = None,
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[RwtdSample]:
    """Yield decoded RWTD samples for the requested dataset view."""

    if limit is not None and limit < 1:
        raise ValueError(f"--limit must be positive when provided, got {limit}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")

    dataset = load_split_dataset(split=split, dataset_id=dataset_id, cache_dir=cache_dir)
    max_items = max(0, len(dataset) - start_index) if limit is None else min(limit, max(0, len(dataset) - start_index))
    for index in range(start_index, start_index + max_items):
        row = dataset[index]
        source_split = str(row.get(SOURCE_SPLIT_COLUMN, split))
        yield decode_sample(row, index=index, split=source_split)


def get_rwtd_sample(
    split: str,
    index: int,
    dataset_id: str = DEFAULT_DATASET_ID,
    cache_dir: str | None = None,
) -> RwtdSample:
    """Load and decode one RWTD sample by row index within the requested view."""

    dataset = load_split_dataset(split=split, dataset_id=dataset_id, cache_dir=cache_dir)
    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"Sample index {index} is out of range for dataset view '{split}' with {len(dataset)} rows."
        )
    row = dataset[index]
    source_split = str(row.get(SOURCE_SPLIT_COLUMN, split))
    return decode_sample(row, index=index, split=source_split)


def validate_split_name(split: str) -> None:
    """Reject unsupported dataset view names up front."""

    if split not in SUPPORTED_SPLITS:
        expected = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {expected}.")


def _attach_source_split(dataset, split: str, overwrite: bool = False):
    """Attach a stable source-split column used when decoding combined views."""

    if SOURCE_SPLIT_COLUMN in dataset.column_names and not overwrite:
        return dataset
    if SOURCE_SPLIT_COLUMN in dataset.column_names:
        dataset = dataset.remove_columns(SOURCE_SPLIT_COLUMN)
    return dataset.add_column(SOURCE_SPLIT_COLUMN, [split] * len(dataset))


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Return a writable datasets cache directory path for this repository."""

    resolved = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _build_overview(
    dataset_id: str,
    split: str,
    split_sizes: dict[str, int],
    features: tuple[str, ...],
) -> DatasetSplitOverview:
    if split == COMBINED_SPLIT:
        num_examples = int(sum(split_sizes.get(split_name, 0) for split_name in PUBLIC_RWTD_SPLITS))
    else:
        num_examples = int(split_sizes.get(split, 0))
    return DatasetSplitOverview(
        dataset_id=dataset_id,
        split=split,
        num_examples=num_examples,
        features=features,
        split_sizes=split_sizes,
    )


def _resolve_local_rwtd_bundle_dir(dataset_id: str) -> Path | None:
    root = _resolve_local_rwtd_root(dataset_id)
    if root is None or not _is_saved_dataset_bundle(root):
        return None
    return root


def _resolve_local_rwtd_root(dataset_id: str) -> Path | None:
    candidate = str(dataset_id)
    candidate_path = Path(candidate)
    if candidate_path.exists() and (_is_saved_dataset_bundle(candidate_path) or _is_raw_rwtd_root(candidate_path)):
        return candidate_path
    if candidate in CURATED_LOCAL_DATASET_IDS and CURATED_LOCAL_BUNDLE_DIR.exists() and (
        _is_saved_dataset_bundle(CURATED_LOCAL_BUNDLE_DIR) or _is_raw_rwtd_root(CURATED_LOCAL_BUNDLE_DIR)
    ):
        return CURATED_LOCAL_BUNDLE_DIR
    return None


def _is_saved_dataset_bundle(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_dict.json").exists()


def _is_raw_rwtd_root(path: Path) -> bool:
    return path.is_dir() and (path / "image").is_dir() and (path / "edge").is_dir() and (path / "splits").is_dir()


def _load_local_split_overview(
    *,
    split: str,
    dataset_id: str,
    bundle_dir: Path,
) -> DatasetSplitOverview:
    dataset_dict = _load_local_dataset_dict(bundle_dir)
    split_sizes = {split_name: int(len(dataset_dict[split_name])) for split_name in PUBLIC_RWTD_SPLITS}
    features = tuple(dataset_dict["train"].features.keys())
    return _build_overview(
        dataset_id=str(dataset_id),
        split=split,
        split_sizes=split_sizes,
        features=features,
    )


def _load_raw_split_overview(
    *,
    split: str,
    dataset_id: str,
    raw_root: Path,
) -> DatasetSplitOverview:
    split_sizes = {
        "train": len(_read_raw_rwtd_split_ids(raw_root, "train")),
        "test": len(_read_raw_rwtd_split_ids(raw_root, "test")),
    }
    features = tuple(_build_raw_rwtd_row(raw_root, _read_raw_rwtd_split_ids(raw_root, "all")[0], "all").keys())
    return _build_overview(
        dataset_id=str(dataset_id),
        split=split,
        split_sizes=split_sizes,
        features=features,
    )


def _load_local_split_dataset(
    *,
    split: str,
    bundle_dir: Path,
):
    _, _, concatenate_datasets = _require_datasets()
    dataset_dict = _load_local_dataset_dict(bundle_dir)
    if split == COMBINED_SPLIT:
        datasets = []
        for source_split in PUBLIC_RWTD_SPLITS:
            datasets.append(_attach_source_split(dataset_dict[source_split], source_split, overwrite=True))
        return concatenate_datasets(datasets)
    return _attach_source_split(dataset_dict[split], split, overwrite=True)


def _load_local_dataset_dict(bundle_dir: Path):
    from datasets import load_from_disk

    dataset = load_from_disk(str(bundle_dir))
    if not hasattr(dataset, "keys"):
        raise RuntimeError(f"RWTD curated bundle at {bundle_dir} is not a DatasetDict.")
    missing_splits = [split_name for split_name in PUBLIC_RWTD_SPLITS if split_name not in dataset]
    if missing_splits:
        raise RuntimeError(f"RWTD curated bundle at {bundle_dir} is missing splits: {missing_splits}")
    return dataset


def _load_raw_split_dataset(*, split: str, raw_root: Path):
    from datasets import Dataset

    rows = [_build_raw_rwtd_row(raw_root, sample_id, split) for sample_id in _read_raw_rwtd_split_ids(raw_root, split)]
    if not rows:
        raise RuntimeError(f"RWTD raw root at {raw_root} did not yield any rows for split '{split}'.")
    return Dataset.from_list(rows)


def _read_raw_rwtd_split_ids(raw_root: Path, split: str) -> list[str]:
    split_dir = raw_root / "splits"
    if split == COMBINED_SPLIT:
        all_path = split_dir / "all.txt"
        if all_path.exists():
            return _read_id_list(all_path)
        return _read_id_list(split_dir / "train.txt") + _read_id_list(split_dir / "val.txt")
    if split == "train":
        return _read_id_list(split_dir / "train.txt")
    if split == "test":
        test_path = split_dir / "test.txt"
        if test_path.exists():
            return _read_id_list(test_path)
        return _read_id_list(split_dir / "val.txt")
    raise ValueError(f"Unsupported raw RWTD split: {split}")


def _read_id_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing RWTD split manifest: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_raw_rwtd_row(raw_root: Path, sample_id: str, split: str) -> dict[str, Any]:
    from scipy.io import loadmat

    source_split = _resolve_raw_rwtd_source_split(raw_root, sample_id, split)
    image_path = raw_root / "image" / source_split / f"{sample_id}.jpg"
    edge_path = raw_root / "edge" / source_split / f"{sample_id}.mat"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing RWTD image: {image_path}")
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing RWTD edge annotation: {edge_path}")

    image = Image.open(image_path).convert("RGB")
    mat = loadmat(edge_path)
    ground_truth = mat["groundTruth"][0, 0]
    segmentation = np.asarray(ground_truth["Segmentation"][0, 0], dtype=np.uint8)
    boundaries = np.asarray(ground_truth["Boundaries"][0, 0], dtype=np.uint8)

    texture_a_mask = segmentation == 1
    texture_b_mask = segmentation == 2
    boundary_mask = boundaries > 0

    if not np.any(texture_a_mask) or not np.any(texture_b_mask):
        raise RuntimeError(f"RWTD sample '{sample_id}' has an empty binary partition.")

    oracle_points_a = _raw_rwtd_oracle_points(texture_a_mask)
    oracle_points_b = _raw_rwtd_oracle_points(texture_b_mask)

    return {
        "image": image,
        "boundary_mask": Image.fromarray(np.asarray(boundary_mask, dtype=np.uint8) * 255, mode="L"),
        "texture_a_mask": Image.fromarray(np.asarray(texture_a_mask, dtype=np.uint8) * 255, mode="L"),
        "texture_b_mask": Image.fromarray(np.asarray(texture_b_mask, dtype=np.uint8) * 255, mode="L"),
        "texture_a": "region A",
        "texture_b": "region B",
        "original_texture_a": "label_1",
        "original_texture_b": "label_2",
        "crop_name": str(sample_id),
        "oracle_points_a": json.dumps(oracle_points_a),
        "oracle_points_b": json.dumps(oracle_points_b),
        SOURCE_SPLIT_COLUMN: source_split,
    }


def _resolve_raw_rwtd_source_split(raw_root: Path, sample_id: str, requested_split: str) -> str:
    if requested_split == "train":
        return "train"
    if requested_split == "test":
        test_dir = raw_root / "image" / "test"
        if test_dir.exists() and (test_dir / f"{sample_id}.jpg").exists():
            return "test"
        return "val"

    train_path = raw_root / "image" / "train" / f"{sample_id}.jpg"
    if train_path.exists():
        return "train"
    test_path = raw_root / "image" / "test" / f"{sample_id}.jpg"
    if test_path.exists():
        return "test"
    return "val"


def _raw_rwtd_oracle_points(mask: np.ndarray) -> list[list[int]]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        raise RuntimeError("Cannot derive oracle points from an empty RWTD mask.")
    first = coords[0]
    middle = coords[len(coords) // 2]
    return [[int(first[1]), int(first[0])], [int(middle[1]), int(middle[0])]]


def _load_cached_split_overview(
    split: str,
    dataset_id: str,
    cache_dir: str,
) -> DatasetSplitOverview | None:
    cached_root = _find_cached_rwtd_root(dataset_id=dataset_id, cache_dir=cache_dir)
    if cached_root is None:
        return None

    dataset_info = json.loads((cached_root / "dataset_info.json").read_text(encoding="utf-8"))
    split_sizes = {
        split_name: int(split_info["num_examples"])
        for split_name, split_info in dataset_info["splits"].items()
    }
    features = tuple(dataset_info["features"].keys())
    return _build_overview(
        dataset_id=dataset_id,
        split=split,
        split_sizes=split_sizes,
        features=features,
    )


def _load_cached_split_dataset(
    split: str,
    dataset_id: str,
    cache_dir: str,
):
    cached_root = _find_cached_rwtd_root(dataset_id=dataset_id, cache_dir=cache_dir)
    if cached_root is None:
        return None

    _, _, concatenate_datasets = _require_datasets()
    from datasets import Dataset

    if split == COMBINED_SPLIT:
        datasets = []
        for source_split in PUBLIC_RWTD_SPLITS:
            dataset = Dataset.from_file(str(cached_root / f"rwtd-{source_split}.arrow"))
            datasets.append(_attach_source_split(dataset, source_split, overwrite=True))
        return concatenate_datasets(datasets)

    dataset = Dataset.from_file(str(cached_root / f"rwtd-{split}.arrow"))
    return _attach_source_split(dataset, split, overwrite=True)


def _find_cached_rwtd_root(dataset_id: str, cache_dir: str) -> Path | None:
    if dataset_id != LEGACY_PUBLIC_RWTD_DATASET_ID:
        return None

    dataset_slug = dataset_id.replace("/", "___").lower()
    candidate_parents = (
        Path(cache_dir) / "datasets" / dataset_slug,
        Path(cache_dir) / dataset_slug,
        HF_GLOBAL_DATASETS_CACHE / dataset_slug,
    )
    for parent in candidate_parents:
        if not parent.exists():
            continue
        for candidate in sorted(parent.glob("default/*/*"), reverse=True):
            if _is_cached_rwtd_root(candidate):
                return candidate
    return None


def _is_cached_rwtd_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "dataset_info.json").exists()
        and (path / "rwtd-train.arrow").exists()
        and (path / "rwtd-test.arrow").exists()
    )


def decode_sample(row: dict[str, Any], index: int, split: str) -> RwtdSample:
    """Decode a raw Hugging Face row into validated RWTD tensors and metadata."""

    crop_name = str(row.get("crop_name", index))
    image = _coerce_rgb_image(row["image"], field_name="image", crop_name=crop_name)
    boundary_mask = _coerce_binary_mask(
        row["boundary_mask"],
        field_name="boundary_mask",
        crop_name=crop_name,
        expected_size=image.size,
    )
    texture_a_mask = _coerce_binary_mask(
        row["texture_a_mask"],
        field_name="texture_a_mask",
        crop_name=crop_name,
        expected_size=image.size,
    )
    texture_b_mask = _coerce_binary_mask(
        row["texture_b_mask"],
        field_name="texture_b_mask",
        crop_name=crop_name,
        expected_size=image.size,
    )

    if texture_a_mask.shape != texture_b_mask.shape:
        raise ValueError(
            f"Sample '{crop_name}' has mismatched texture mask shapes: "
            f"{texture_a_mask.shape} vs {texture_b_mask.shape}."
        )

    width, height = image.size
    oracle_points_a = parse_oracle_points(
        raw_points=row["oracle_points_a"],
        crop_name=crop_name,
        field_name="oracle_points_a",
        image_size=(width, height),
    )
    oracle_points_b = parse_oracle_points(
        raw_points=row["oracle_points_b"],
        crop_name=crop_name,
        field_name="oracle_points_b",
        image_size=(width, height),
    )

    return RwtdSample(
        index=index,
        split=split,
        crop_name=crop_name,
        image=image,
        boundary_mask=boundary_mask,
        texture_a_mask=texture_a_mask,
        texture_b_mask=texture_b_mask,
        texture_a=_require_non_empty_string(row["texture_a"], crop_name, "texture_a"),
        texture_b=_require_non_empty_string(row["texture_b"], crop_name, "texture_b"),
        original_texture_a=_require_non_empty_string(
            row["original_texture_a"], crop_name, "original_texture_a"
        ),
        original_texture_b=_require_non_empty_string(
            row["original_texture_b"], crop_name, "original_texture_b"
        ),
        oracle_points_a=oracle_points_a,
        oracle_points_b=oracle_points_b,
    )


def parse_oracle_points(
    raw_points: Any,
    crop_name: str,
    field_name: str,
    image_size: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Parse the JSON-encoded RWTD oracle point list into validated pixel coordinates."""

    width, height = image_size
    if raw_points is None:
        raise ValueError(f"Sample '{crop_name}' is missing '{field_name}'.")

    if isinstance(raw_points, str):
        try:
            parsed = json.loads(raw_points)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Sample '{crop_name}' has malformed JSON in '{field_name}': {raw_points!r}"
            ) from exc
    else:
        parsed = raw_points

    if not isinstance(parsed, list):
        raise ValueError(
            f"Sample '{crop_name}' has invalid '{field_name}': expected a JSON list of points."
        )

    points: list[tuple[int, int]] = []
    for point_index, point in enumerate(parsed):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(
                f"Sample '{crop_name}' has invalid point #{point_index} in '{field_name}': {point!r}"
            )
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(
                f"Sample '{crop_name}' has non-numeric point #{point_index} in '{field_name}': {point!r}"
            )
        x_int = int(x)
        y_int = int(y)
        if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
            raise ValueError(
                f"Sample '{crop_name}' has out-of-bounds point #{point_index} in '{field_name}': {point!r}"
            )
        points.append((x_int, y_int))
    return tuple(points)


def _require_non_empty_string(value: Any, crop_name: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Sample '{crop_name}' has invalid '{field_name}': expected a non-empty string.")
    return value


def _coerce_rgb_image(value: Any, field_name: str, crop_name: str) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value)
    else:
        raise ValueError(f"Sample '{crop_name}' has unsupported image type for '{field_name}': {type(value)!r}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _coerce_binary_mask(
    value: Any,
    field_name: str,
    crop_name: str,
    expected_size: tuple[int, int],
) -> np.ndarray:
    if isinstance(value, Image.Image):
        mask = np.asarray(value)
    elif isinstance(value, np.ndarray):
        mask = value
    else:
        raise ValueError(f"Sample '{crop_name}' has unsupported mask type for '{field_name}': {type(value)!r}")

    if mask.ndim != 2:
        raise ValueError(
            f"Sample '{crop_name}' has invalid '{field_name}' shape {mask.shape}; expected a 2D mask."
        )
    expected_width, expected_height = expected_size
    if mask.shape != (expected_height, expected_width):
        raise ValueError(
            f"Sample '{crop_name}' has '{field_name}' shape {mask.shape} but expected {(expected_height, expected_width)}."
        )
    return np.asarray(mask > 0, dtype=bool)
