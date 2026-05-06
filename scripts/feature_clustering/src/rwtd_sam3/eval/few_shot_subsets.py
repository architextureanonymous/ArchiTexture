"""Helpers for deterministic few-shot subset manifests.

This module keeps few-shot subset selection explicit and reproducible:

- manifests are stored as JSON with the selected image IDs
- nested subsets are prefixes of a single seeded permutation
- dataset consumers can validate and apply a manifest without inventing
  dataset-specific semantics in each trainer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FewShotSubsetManifest:
    """One deterministic few-shot subset manifest."""

    dataset_id: str
    source_split: str
    shot_count: int
    full_split_size: int
    subset_seed: int
    selected_image_ids: tuple[str, ...]
    selection_policy: str
    source_id_order: tuple[str, ...]
    sampled_prefix_ids: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_image_ids"] = list(self.selected_image_ids)
        payload["source_id_order"] = list(self.source_id_order)
        payload["sampled_prefix_ids"] = list(self.sampled_prefix_ids)
        return payload


def build_nested_few_shot_manifest(
    *,
    dataset_id: str,
    source_split: str,
    source_id_order: Sequence[str],
    shot_count: int,
    subset_seed: int,
) -> FewShotSubsetManifest:
    """Build one nested few-shot manifest from a seeded permutation."""

    source_ids = tuple(str(image_id) for image_id in source_id_order)
    full_split_size = len(source_ids)
    resolved_shot_count = int(shot_count)
    if resolved_shot_count < 1:
        raise ValueError(f"shot_count must be positive, got {shot_count}.")
    if resolved_shot_count > full_split_size:
        raise ValueError(
            f"shot_count={resolved_shot_count} exceeds source split size {full_split_size} for dataset '{dataset_id}'."
        )
    permutation = np.random.default_rng(int(subset_seed)).permutation(full_split_size).tolist()
    sampled_prefix_ids = tuple(source_ids[int(index)] for index in permutation[:resolved_shot_count])
    sampled_prefix_set = set(sampled_prefix_ids)
    selected_image_ids = tuple(image_id for image_id in source_ids if image_id in sampled_prefix_set)
    if len(selected_image_ids) != resolved_shot_count:
        raise RuntimeError(
            "Few-shot subset manifest construction produced a size mismatch after reordering to source order."
        )
    return FewShotSubsetManifest(
        dataset_id=str(dataset_id),
        source_split=str(source_split),
        shot_count=resolved_shot_count,
        full_split_size=full_split_size,
        subset_seed=int(subset_seed),
        selected_image_ids=selected_image_ids,
        selection_policy=(
            "nested_seeded_prefix_over_source_split_then_reordered_to_source_order:"
            "np.random.default_rng(subset_seed).permutation(source_ids)[:shot_count]"
        ),
        source_id_order=source_ids,
        sampled_prefix_ids=sampled_prefix_ids,
    )


def write_few_shot_subset_manifest(path: str | Path, manifest: FewShotSubsetManifest) -> Path:
    """Write one few-shot subset manifest to disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_few_shot_subset_manifest(
    path: str | Path,
    *,
    expected_dataset_id: str | None = None,
    expected_source_split: str | None = None,
) -> FewShotSubsetManifest:
    """Load and validate one few-shot subset manifest from disk."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = FewShotSubsetManifest(
        dataset_id=str(payload["dataset_id"]),
        source_split=str(payload["source_split"]),
        shot_count=int(payload["shot_count"]),
        full_split_size=int(payload["full_split_size"]),
        subset_seed=int(payload["subset_seed"]),
        selected_image_ids=tuple(str(value) for value in payload["selected_image_ids"]),
        selection_policy=str(payload["selection_policy"]),
        source_id_order=tuple(str(value) for value in payload["source_id_order"]),
        sampled_prefix_ids=tuple(str(value) for value in payload["sampled_prefix_ids"]),
    )
    if expected_dataset_id is not None and manifest.dataset_id != str(expected_dataset_id):
        raise ValueError(
            f"Few-shot manifest dataset_id '{manifest.dataset_id}' does not match expected '{expected_dataset_id}'."
        )
    if expected_source_split is not None and manifest.source_split != str(expected_source_split):
        raise ValueError(
            f"Few-shot manifest source_split '{manifest.source_split}' does not match expected '{expected_source_split}'."
        )
    if manifest.shot_count != len(manifest.selected_image_ids):
        raise ValueError(
            f"Few-shot manifest shot_count={manifest.shot_count} does not match "
            f"{len(manifest.selected_image_ids)} selected_image_ids."
        )
    if len(set(manifest.selected_image_ids)) != len(manifest.selected_image_ids):
        raise ValueError("Few-shot manifest selected_image_ids must be unique.")
    missing_from_source = [image_id for image_id in manifest.selected_image_ids if image_id not in manifest.source_id_order]
    if missing_from_source:
        raise ValueError(
            f"Few-shot manifest selected_image_ids contain IDs outside source_id_order: {missing_from_source}"
        )
    return manifest


def select_items_by_few_shot_manifest(
    items: Sequence[Any],
    manifest: FewShotSubsetManifest | None,
    *,
    item_id_getter,
) -> tuple[Any, ...]:
    """Filter a sequence of items to the manifest's selected IDs, preserving source order."""

    if manifest is None:
        return tuple(items)
    selected_id_set = set(manifest.selected_image_ids)
    filtered = tuple(item for item in items if str(item_id_getter(item)) in selected_id_set)
    filtered_ids = tuple(str(item_id_getter(item)) for item in filtered)
    if filtered_ids != manifest.selected_image_ids:
        raise ValueError(
            "Few-shot manifest IDs do not match the filtered source-order IDs.",
        )
    return filtered


def write_few_shot_manifest_index(
    *,
    path: str | Path,
    manifests: Iterable[FewShotSubsetManifest],
) -> Path:
    """Write a machine-readable index over all manifests in one sweep."""

    payload = [manifest.to_json_dict() for manifest in manifests]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
