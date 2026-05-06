"""Deterministic MoNuSeg train/val/test split helpers for repo evaluation.

The upstream public MoNuSeg route uses the official 30/14 train/test split and
selects the best checkpoint on the benchmark test split. The repo-wide
evaluation contract forbids that. This module introduces a deterministic held-
out validation subset over the official 30-image training split so MoNuSeg
trainers can select checkpoints on ``val`` while keeping ``test`` headline-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rwtd_sam3.eval.few_shot_subsets import (
    FewShotSubsetManifest,
    build_nested_few_shot_manifest,
    select_items_by_few_shot_manifest,
)


MONUSEG_CONTRACT_SUPPORTED_SPLITS = ("train", "val", "test", "all")
MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT = 6
MONUSEG_CONTRACT_VAL_SUBSET_SEED = 0


@dataclass(frozen=True)
class MonusegValidationSplit:
    """Deterministic partition of the official MoNuSeg train split."""

    train_samples: tuple[Any, ...]
    val_samples: tuple[Any, ...]
    validation_manifest: FewShotSubsetManifest


def build_monuseg_validation_manifest(
    *,
    source_ids: Sequence[str],
    holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> FewShotSubsetManifest:
    """Build the deterministic repo validation manifest over official train IDs."""

    resolved_holdout_count = int(holdout_count)
    if resolved_holdout_count == 0:
        raise ValueError("MoNuSeg validation holdout_count must be zero only when validation is disabled.")
    if resolved_holdout_count < 1:
        raise ValueError("MoNuSeg validation holdout_count must be positive.")
    if resolved_holdout_count >= len(source_ids):
        raise ValueError(
            "MoNuSeg validation holdout_count must leave at least one optimization sample. "
            f"Got holdout_count={resolved_holdout_count} for {len(source_ids)} train IDs."
        )
    return build_nested_few_shot_manifest(
        dataset_id="monuseg",
        source_split="official_train",
        source_id_order=tuple(str(value) for value in source_ids),
        shot_count=resolved_holdout_count,
        subset_seed=int(subset_seed),
    )


def partition_official_monuseg_train_split(
    train_samples: Sequence[Any],
    *,
    item_id_getter: Callable[[Any], str],
    holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> MonusegValidationSplit:
    """Split official-train samples into repo-contract train and val subsets."""

    resolved_holdout_count = int(holdout_count)
    if resolved_holdout_count < 1:
        raise ValueError("MoNuSeg validation holdout_count must be positive for held-out validation.")
    source_ids = tuple(str(item_id_getter(sample)) for sample in train_samples)
    manifest = build_monuseg_validation_manifest(
        source_ids=source_ids,
        holdout_count=resolved_holdout_count,
        subset_seed=subset_seed,
    )
    val_samples = select_items_by_few_shot_manifest(
        tuple(train_samples),
        manifest,
        item_id_getter=item_id_getter,
    )
    val_ids = set(manifest.selected_image_ids)
    train_partition = tuple(
        sample for sample in train_samples if str(item_id_getter(sample)) not in val_ids
    )
    if not train_partition:
        raise ValueError("MoNuSeg validation split consumed every training sample.")
    return MonusegValidationSplit(
        train_samples=train_partition,
        val_samples=val_samples,
        validation_manifest=manifest,
    )


def resolve_monuseg_contract_split_samples(
    *,
    requested_split: str,
    official_train_samples: Sequence[Any],
    official_test_samples: Sequence[Any],
    item_id_getter: Callable[[Any], str],
    holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> tuple[tuple[Any, ...], FewShotSubsetManifest]:
    """Resolve repo-contract samples for ``train``/``val``/``test``/``all``."""

    if requested_split not in MONUSEG_CONTRACT_SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported MoNuSeg contract split '{requested_split}'. "
            f"Expected one of {MONUSEG_CONTRACT_SUPPORTED_SPLITS}."
        )
    if requested_split == "train":
        if int(holdout_count) < 1:
            return tuple(official_train_samples), FewShotSubsetManifest(
                dataset_id="monuseg",
                source_split="official_train",
                shot_count=0,
                full_split_size=len(official_train_samples),
                subset_seed=int(subset_seed),
                selected_image_ids=(),
                selection_policy="validation_disabled_full_official_train",
                source_id_order=tuple(str(item_id_getter(sample)) for sample in official_train_samples),
                sampled_prefix_ids=(),
            )
        partition = partition_official_monuseg_train_split(
            official_train_samples,
            item_id_getter=item_id_getter,
            holdout_count=holdout_count,
            subset_seed=subset_seed,
        )
        return partition.train_samples, partition.validation_manifest
    if requested_split == "val":
        if int(holdout_count) < 1:
            raise ValueError(
                "MoNuSeg validation split was requested but validation is disabled by holdout_count=0."
            )
        partition = partition_official_monuseg_train_split(
            official_train_samples,
            item_id_getter=item_id_getter,
            holdout_count=holdout_count,
            subset_seed=subset_seed,
        )
        return partition.val_samples, partition.validation_manifest
    if requested_split == "test":
        validation_manifest = (
            partition_official_monuseg_train_split(
                official_train_samples,
                item_id_getter=item_id_getter,
                holdout_count=holdout_count,
                subset_seed=subset_seed,
            ).validation_manifest
            if int(holdout_count) >= 1
            else FewShotSubsetManifest(
                dataset_id="monuseg",
                source_split="official_train",
                shot_count=0,
                full_split_size=len(official_train_samples),
                subset_seed=int(subset_seed),
                selected_image_ids=(),
                selection_policy="validation_disabled_full_official_train",
                source_id_order=tuple(str(item_id_getter(sample)) for sample in official_train_samples),
                sampled_prefix_ids=(),
            )
        )
        return tuple(official_test_samples), validation_manifest
    if int(holdout_count) < 1:
        return tuple(official_train_samples) + tuple(official_test_samples), FewShotSubsetManifest(
            dataset_id="monuseg",
            source_split="official_train",
            shot_count=0,
            full_split_size=len(official_train_samples),
            subset_seed=int(subset_seed),
            selected_image_ids=(),
            selection_policy="validation_disabled_full_official_train",
            source_id_order=tuple(str(item_id_getter(sample)) for sample in official_train_samples),
            sampled_prefix_ids=(),
        )
    partition = partition_official_monuseg_train_split(
        official_train_samples,
        item_id_getter=item_id_getter,
        holdout_count=holdout_count,
        subset_seed=subset_seed,
    )
    return tuple(partition.train_samples) + tuple(partition.val_samples) + tuple(official_test_samples), partition.validation_manifest


def describe_monuseg_validation_policy(
    *,
    holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> str:
    """Describe the repo-contract MoNuSeg validation derivation."""

    if int(holdout_count) < 1:
        return "validation_disabled_full_official_train"
    return (
        "repo_monuseg_val_v1:"
        f"hold_out_{int(holdout_count)}_official_train_images"
        f"_with_nested_seed_{int(subset_seed)}"
        "_over_source_order"
    )
