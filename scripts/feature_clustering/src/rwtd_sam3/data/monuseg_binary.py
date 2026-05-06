"""MoNuSeg binary benchmark adapter backed by the Hugging Face mirror.

This module exposes the official AutoSAM-comparable MoNuSeg challenge split as
binary foreground/background samples:

- training uses only the 30 official challenge images
- the 7 extra Hugging Face `tissue == 0` train rows are excluded
- testing uses the 14 official challenge test images
- each binary foreground mask is the union of all annotated nucleus instances

The implementation intentionally stays small and transparent. It relies on the
`RationAI/MoNuSeg` dataset mirror and converts the per-instance masks into one
foreground nucleus mask plus its complement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rwtd_sam3.eval.metrics import boundary_from_region_masks


MONUSEG_DATASET_ID = "monuseg"
MONUSEG_HF_DATASET_NAME = "RationAI/MoNuSeg"
MONUSEG_SUPPORTED_SPLITS = ("train", "test", "all")
MONUSEG_TISSUE_LABELS: dict[int, str] = {
    0: "unknown",
    1: "breast",
    2: "kidney",
    3: "liver",
    4: "prostate",
    5: "bladder",
    6: "colon",
    7: "stomach",
}


@dataclass(frozen=True)
class MonusegBinaryOverview:
    """Resolved metadata for one official-split MoNuSeg view."""

    dataset_id: str
    dataset_name: str
    split: str
    num_examples: int
    patient_ids: tuple[str, ...]
    evaluation_view: str
    train_selection_policy: str
    cache_dir: str | None


@dataclass(frozen=True)
class MonusegBinarySample:
    """One decoded MoNuSeg image with the nucleus mask and its complement."""

    index: int
    dataset_id: str
    split: str
    crop_name: str
    image: Any
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
        return int(self.image.size[1])

    @property
    def width(self) -> int:
        return int(self.image.size[0])


@lru_cache(maxsize=8)
def _load_monuseg_dataset(dataset_name: str, cache_dir: str | None):
    from datasets import Dataset, DatasetDict, load_dataset

    cache_path = _resolve_monuseg_arrow_cache_dir(cache_dir)
    if cache_path is not None:
        train_arrow = cache_path / "mo_nu_seg-train.arrow"
        test_arrow = cache_path / "mo_nu_seg-test.arrow"
        if train_arrow.exists() and test_arrow.exists():
            return DatasetDict(
                {
                    "train": Dataset.from_file(str(train_arrow)),
                    "test": Dataset.from_file(str(test_arrow)),
                }
            )

    return load_dataset(dataset_name, cache_dir=cache_dir)


def _resolve_monuseg_arrow_cache_dir(cache_dir: str | None) -> Path | None:
    candidates: list[Path] = []
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        candidates.append(
            cache_root
            / "RationAI___mo_nu_seg"
            / "default"
            / "0.0.0"
            / "4812ea499ff8f6fad2df7af0c9aa459773e22459"
        )
        candidates.append(
            cache_root
            / "datasets"
            / "RationAI___mo_nu_seg"
            / "default"
            / "0.0.0"
            / "4812ea499ff8f6fad2df7af0c9aa459773e22459"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_monuseg_binary_overview(
    *,
    split: str = "test",
    dataset_name: str = MONUSEG_HF_DATASET_NAME,
    cache_dir: str | None = None,
) -> MonusegBinaryOverview:
    """Inspect one official-split MoNuSeg dataset view."""

    records = _resolve_monuseg_records(
        split=split,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    return MonusegBinaryOverview(
        dataset_id=MONUSEG_DATASET_ID,
        dataset_name=dataset_name,
        split=split,
        num_examples=len(records),
        patient_ids=tuple(str(record["patient"]) for record in records),
        evaluation_view="direct_foreground",
        train_selection_policy="official_challenge_train_excludes_tissue_0_unknown",
        cache_dir=cache_dir,
    )


def iter_monuseg_binary_samples(
    *,
    split: str = "test",
    dataset_name: str = MONUSEG_HF_DATASET_NAME,
    cache_dir: str | None = None,
    limit: int | None = None,
    start_index: int = 0,
) -> Iterator[MonusegBinarySample]:
    """Yield decoded MoNuSeg samples in official split order."""

    records = _resolve_monuseg_records(
        split=split,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    max_items = max(0, len(records) - start_index) if limit is None else min(
        int(limit),
        max(0, len(records) - start_index),
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_monuseg_binary_sample(record=records[index], split_index=index)


def get_monuseg_binary_sample(
    *,
    split: str = "test",
    index: int,
    dataset_name: str = MONUSEG_HF_DATASET_NAME,
    cache_dir: str | None = None,
) -> MonusegBinarySample:
    """Load one official-split MoNuSeg sample by index."""

    records = _resolve_monuseg_records(
        split=split,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    if index < 0 or index >= len(records):
        raise IndexError(
            f"Sample index {index} is out of range for dataset '{MONUSEG_DATASET_ID}' split '{split}' "
            f"with {len(records)} samples."
        )
    return _decode_monuseg_binary_sample(record=records[index], split_index=index)


def _resolve_monuseg_records(
    *,
    split: str,
    dataset_name: str,
    cache_dir: str | None,
) -> tuple[dict[str, Any], ...]:
    if split not in MONUSEG_SUPPORTED_SPLITS:
        expected = ", ".join(MONUSEG_SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported MoNuSeg split '{split}'. Expected one of: {expected}.")
    dataset = _load_monuseg_dataset(dataset_name, cache_dir)
    train_split = dataset["train"]
    test_split = dataset["test"]
    train_patients = _split_column(train_split, "patient")
    train_tissues = _split_column(train_split, "tissue")
    test_patients = _split_column(test_split, "patient")
    train_records = tuple(
        {
            "split": "train",
            "row_index": int(index),
            "patient": str(train_patients[int(index)]),
            "tissue": int(train_tissues[int(index)]),
            "dataset_name": str(dataset_name),
            "cache_dir": cache_dir,
        }
        for index in range(len(train_patients))
        if int(train_tissues[int(index)]) != 0
    )
    test_records = tuple(
        {
            "split": "test",
            "row_index": int(index),
            "patient": str(test_patients[int(index)]),
            "tissue": 0,
            "dataset_name": str(dataset_name),
            "cache_dir": cache_dir,
        }
        for index in range(len(test_patients))
    )
    if split == "train":
        return train_records
    if split == "test":
        return test_records
    return train_records + test_records


def _decode_monuseg_binary_sample(*, record: dict[str, Any], split_index: int) -> MonusegBinarySample:
    dataset = _load_monuseg_dataset(str(record["dataset_name"]), record.get("cache_dir"))
    row = _split_row(dataset[str(record["split"])], int(record["row_index"]))
    image = row["image"].convert("RGB")
    image_height = int(image.size[1])
    image_width = int(image.size[0])
    nucleus_mask = np.zeros((image_height, image_width), dtype=bool)
    for instance in row["instances"]:
        nucleus_mask |= np.asarray(instance, dtype=np.uint8) > 0
    background_mask = np.logical_not(nucleus_mask)
    if not nucleus_mask.any():
        raise ValueError(
            f"MoNuSeg sample '{row['patient']}' did not contain any positive nucleus pixels after raster union."
        )
    if not background_mask.any():
        raise ValueError(f"MoNuSeg sample '{row['patient']}' does not contain any background pixels.")
    boundary_mask = boundary_from_region_masks(nucleus_mask, background_mask)
    tissue_index = int(row["tissue"])
    tissue_label = MONUSEG_TISSUE_LABELS.get(tissue_index, f"tissue_{tissue_index}")
    return MonusegBinarySample(
        index=int(split_index),
        dataset_id=MONUSEG_DATASET_ID,
        split=str(record["split"]),
        crop_name=str(row["patient"]),
        image=image,
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(nucleus_mask, dtype=bool),
        texture_b_mask=np.asarray(background_mask, dtype=bool),
        texture_a="nucleus",
        texture_b="background",
        original_texture_a="instance_union_positive",
        original_texture_b="instance_union_zero",
        oracle_points_a=(),
        oracle_points_b=(),
        evaluation_view="direct_foreground",
        grade_label=tissue_label,
    )


def _split_column(split: Any, name: str) -> list[Any]:
    try:
        column = split[name]
    except Exception:
        return [row[name] for row in split]
    return list(column)


def _split_row(split: Any, index: int) -> dict[str, Any]:
    return split[int(index)]
