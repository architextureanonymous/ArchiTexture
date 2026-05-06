"""Faithful MoNuSeg AutoSAM reproduction route.

This module implements a MoNuSeg-specific AutoSAM runner whose primary goal is
reproduction fidelity rather than convenience:

- it keeps the upstream 2D AutoSAM prompt-generator path over frozen SAM
- it exposes paper/profile defaults explicitly instead of hiding them in code
- it writes repo-native artifacts so runs are auditable and reproducible

The headline metrics reported by this route follow the upstream AutoSAM-style
MoNuSeg evaluation view:

- resize the binary prediction and ground truth to ``Idim x Idim`` after SAM
  postprocessing
- threshold the resized prediction at ``foreground_threshold``
- compute Dice and IoU on that resized binary view

The route also reports repo-native direct-foreground and partition-invariant
metrics on the original image resolution so the same runs remain comparable to
the rest of this repository.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np

from rwtd_sam3.data.monuseg_binary import (
    MONUSEG_DATASET_ID,
    MONUSEG_HF_DATASET_NAME,
    MonusegBinaryOverview,
    MonusegBinarySample,
    iter_monuseg_binary_samples,
    load_monuseg_binary_overview,
)
from rwtd_sam3.eval.experiment_terms import ExperimentTermsSection, render_experiment_terms_markdown
from rwtd_sam3.eval.few_shot_subsets import (
    FewShotSubsetManifest,
    load_few_shot_subset_manifest,
    select_items_by_few_shot_manifest,
)
from rwtd_sam3.eval.core import evaluate_run, rewrite_contract_summary
from rwtd_sam3.eval.evaluation_contract import UPSTREAM_FAITHFUL_PRIMARY
from rwtd_sam3.eval.metrics import (
    CANONICAL_EVALUATION_CONTRACT,
    build_canonical_evaluation_fields,
    compute_binary_metrics,
)
from rwtd_sam3.eval.monuseg_split_contract import (
    MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    MONUSEG_CONTRACT_VAL_SUBSET_SEED,
    describe_monuseg_validation_policy,
    resolve_monuseg_contract_split_samples,
)
from rwtd_sam3.eval.runner import (
    SampleEvaluationResult,
    append_dataset_partition_fields,
    build_visual_record,
    discovered_package_versions,
    resolve_dataset_partition,
    resolve_eval_sample_count,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from rwtd_sam3.eval.sam3_auto import select_best_binary_assignment
from rwtd_sam3.models.autosam_upstream import (
    AUTOSAM_REPO_ROOT,
    build_upstream_autosam_prompt_generator,
    build_upstream_autosam_sam,
    load_upstream_autosam_resize_longest_side_class,
    load_upstream_autosam_transforms_shir,
)
from rwtd_sam3.utils.visualization import save_prediction_panel


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]

MONUSEG_AUTOSAM_ROUTE = "monuseg"
MONUSEG_AUTOSAM_OUTPUT_ROOT = Path("outputs") / "monuseg_binary" / "autosam"
MONUSEG_AUTOSAM_SCALAR_FIELDS = (
    "upstream_eval_iou",
    "upstream_eval_dice",
    "direct_foreground_iou",
    "direct_foreground_dice",
    "direct_foreground_precision",
    "direct_foreground_recall",
    "eval_miou",
    "eval_ari",
    "predicted_positive_fraction",
    "target_positive_fraction",
)
MONUSEG_AUTOSAM_AUGMENTATION_POLICIES = ("none", "monuseg_paper_v1")
MONUSEG_AUTOSAM_SELECTION_METRICS = ("upstream_eval_iou", "upstream_eval_dice")
MONUSEG_AUTOSAM_REPRODUCTION_PROFILE = {
    "profile_name": "monuseg_faithful_v1",
    "paper_anchor_url": "https://arxiv.org/abs/2306.06370",
    "dataset_name": MONUSEG_HF_DATASET_NAME,
    "sam_model_type": "vit_h",
    "prompt_image_size": 512,
    "autosam_backbone_order": 85,
    "autosam_depth_wise": False,
    "batch_size": 10,
    "gradient_accumulation_steps": 1,
    "drop_incomplete_accumulation": False,
    "num_workers": 0,
    "train_repeat_factor": 5,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "num_epochs": 200,
    "foreground_threshold": 0.5,
    "train_augmentation_policy": "monuseg_paper_v1",
    "eval_every_epochs": 1,
    "selection_metric": "upstream_eval_iou",
    "selection_split": "val",
    "selection_checkpoint_mode": "best_eval",
}


@dataclass(frozen=True)
class AutoSamMonusegBatchItem:
    """One transformed MoNuSeg sample prepared for AutoSAM.

    Attributes:
        sample: Source MoNuSeg record.
        processed_image: ``(3, 1024, 1024)`` float tensor after upstream resize
            and preprocess semantics.
        processed_mask: ``(1024, 1024)`` float tensor after the same spatial
            transforms plus mask thresholding and preprocess padding.
        original_size_hw: Spatial size after paired train/eval augmentation and
            before SAM square padding, in ``(H, W)`` order.
        sam_input_size_hw: Spatial size after ``ResizeLongestSide`` and before
            square padding, in ``(H, W)`` order.
    """

    sample: MonusegBinarySample
    processed_image: Any
    processed_mask: Any
    original_size_hw: tuple[int, int]
    sam_input_size_hw: tuple[int, int]


@dataclass(frozen=True)
class AutoSamMonusegTrainingArtifacts:
    """Persistent outputs from one MoNuSeg AutoSAM training run."""

    final_checkpoint_path: Path
    best_checkpoint_path: Path | None
    trainable_parameter_count: int
    training_history_rows: list[dict[str, Any]]
    epoch_eval_history_rows: list[dict[str, Any]]
    best_epoch_by_selection_metric: int | None
    best_selection_metric_value: float | None


@dataclass(frozen=True)
class AutoSamMonusegForwardOutput:
    """Forward outputs retained for loss and eval bookkeeping."""

    normalized_low_res_masks: Any
    postprocessed_masks: list[Any]
    dense_embeddings: Any


@dataclass(frozen=True)
class AutoSamMonusegEvalBundle:
    """Per-sample eval payload saved into the run directory."""

    row: dict[str, Any]
    metric_summary: str
    foreground_prediction: np.ndarray
    background_prediction: np.ndarray
    foreground_probability: np.ndarray


class AutoSamMonusegRuntimeError(RuntimeError):
    """Raised for explicit MoNuSeg AutoSAM contract violations."""


def resolve_monuseg_autosam_profile(args) -> dict[str, Any]:
    profile_name = str(getattr(args, "reproduction_profile", MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["profile_name"]))
    if profile_name != MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["profile_name"]:
        raise AutoSamMonusegRuntimeError(
            f"Unsupported MoNuSeg AutoSAM reproduction_profile '{profile_name}'.",
        )
    return dict(MONUSEG_AUTOSAM_REPRODUCTION_PROFILE)


def resolve_train_subset_manifest(args) -> FewShotSubsetManifest | None:
    manifest_path = getattr(args, "train_subset_manifest", None)
    if manifest_path in (None, ""):
        return None
    return load_few_shot_subset_manifest(
        manifest_path,
        expected_dataset_id=MONUSEG_DATASET_ID,
        expected_source_split=str(getattr(args, "train_split", "train")),
    )


def resolve_train_selection_policy(args) -> str:
    if resolve_train_subset_manifest(args) is None:
        return "official_challenge_train_excludes_tissue_0_unknown"
    return "few_shot_subset_manifest_over_official_challenge_train"


def resolve_train_augmentation_policy(args) -> str:
    policy = str(getattr(args, "train_augmentation_policy", MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["train_augmentation_policy"]))
    if policy not in MONUSEG_AUTOSAM_AUGMENTATION_POLICIES:
        raise AutoSamMonusegRuntimeError(
            f"Unsupported MoNuSeg AutoSAM augmentation policy '{policy}'.",
        )
    return policy


def describe_train_augmentation_policy(policy: str) -> str:
    if policy == "none":
        return "none"
    return (
        "monuseg_paper_v1:"
        "color_jitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)"
        "+hflip_p0.5"
        "+affine(angle_uniform[-20,20],scale_uniform[0.75,1.25],translate=0,shear=0)"
        "+upstream_paired_image_mask_ops"
    )


def build_monuseg_train_transform(*, policy: str):
    transforms = load_upstream_autosam_transforms_shir()
    if policy == "none":
        return transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    if policy != "monuseg_paper_v1":
        raise AutoSamMonusegRuntimeError(f"Unsupported MoNuSeg AutoSAM augmentation policy '{policy}'.")
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, scale=(0.75, 1.25)),
            transforms.ToTensor(),
        ]
    )


def build_monuseg_eval_transform():
    transforms = load_upstream_autosam_transforms_shir()
    return transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


class AutoSamMonusegDataset:
    """Wrap MoNuSeg samples in the tensor contract expected by AutoSAM."""

    def __init__(
        self,
        *,
        samples: Sequence[MonusegBinarySample],
        train: bool,
        repeat_factor: int,
        train_augmentation_policy: str,
        sam_transform,
    ) -> None:
        if not samples:
            raise AutoSamMonusegRuntimeError("AutoSAM MoNuSeg dataset wrapper received zero samples.")
        self.samples = tuple(samples)
        self.train = bool(train)
        self.repeat_factor = int(repeat_factor if train else 1)
        if self.repeat_factor < 1:
            raise AutoSamMonusegRuntimeError(f"repeat_factor must be >= 1, got {repeat_factor}.")
        self.sam_transform = sam_transform
        self.transform = (
            build_monuseg_train_transform(policy=train_augmentation_policy)
            if self.train
            else build_monuseg_eval_transform()
        )

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, index: int) -> AutoSamMonusegBatchItem:
        sample = self.samples[int(index) % len(self.samples)]
        image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
        mask = np.asarray(sample.texture_a_mask, dtype=np.uint8)
        transformed_image, transformed_mask = self.transform(image, mask)
        original_size = (int(transformed_image.shape[1]), int(transformed_image.shape[2]))
        transformed_image = self.sam_transform.apply_image_torch(transformed_image)
        transformed_mask = self.sam_transform.apply_image_torch(transformed_mask)
        transformed_mask = (transformed_mask > 0.5).to(dtype=transformed_image.dtype)
        processed_image = self.sam_transform.preprocess(transformed_image)
        processed_mask = self.sam_transform.preprocess(transformed_mask)
        sam_input_size = (int(transformed_image.shape[-2]), int(transformed_image.shape[-1]))
        return AutoSamMonusegBatchItem(
            sample=sample,
            processed_image=processed_image,
            processed_mask=processed_mask,
            original_size_hw=original_size,
            sam_input_size_hw=sam_input_size,
        )


def collate_autosam_monuseg_batch(batch: Sequence[AutoSamMonusegBatchItem]) -> dict[str, Any]:
    import torch

    if not batch:
        raise AutoSamMonusegRuntimeError("AutoSAM MoNuSeg collate received an empty batch.")
    return {
        "samples": [item.sample for item in batch],
        "images": torch.stack([item.processed_image for item in batch], dim=0),
        "masks": torch.stack([item.processed_mask for item in batch], dim=0),
        "original_sizes": torch.tensor([item.original_size_hw for item in batch], dtype=torch.int64),
        "image_sizes": torch.tensor([item.sam_input_size_hw for item in batch], dtype=torch.int64),
    }


def resolve_torch_device(requested_device: str):
    import torch

    if str(requested_device) == "cuda" and not torch.cuda.is_available():
        raise AutoSamMonusegRuntimeError("CUDA was requested for MoNuSeg AutoSAM but torch.cuda.is_available() is False.")
    return torch.device(str(requested_device))


def set_global_training_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def count_trainable_parameters(module) -> int:
    import torch

    total = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            total += int(torch.numel(parameter))
    return total


def norm_batch(x):
    import torch

    if x.ndim != 4:
        raise AutoSamMonusegRuntimeError(f"Expected a 4D tensor for norm_batch, got shape {tuple(x.shape)}.")
    batch_size = int(x.shape[0])
    height = int(x.shape[-2])
    width = int(x.shape[-1])
    min_value = x.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    max_value = x.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    return (x - min_value) / (max_value - min_value + 1e-6)


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    import torch

    alpha = 0.5
    beta = 0.5
    y_pred = y_pred.clamp(0.0, 1.0)
    y_true = y_true.clamp(0.0, 1.0)
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - torch.mean(tversky)


def compute_upstream_binary_iou_dice(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=np.uint8).reshape(-1) + 1
    target = np.asarray(target, dtype=np.uint8).reshape(-1) + 1
    true_positive = int(np.sum((prediction == 2) & (target == 2)))
    false_positive = int(np.sum((prediction == 2) & (target == 1)))
    false_negative = int(np.sum((prediction == 1) & (target == 2)))
    iou = float(np.nan_to_num(true_positive / (true_positive + false_positive + false_negative)))
    dice = float(np.nan_to_num((2 * true_positive) / (2 * true_positive + false_positive + false_negative)))
    return iou, dice


def build_batched_input(images, original_sizes, image_sizes) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    for image, original_size, image_size in zip(images, original_sizes, image_sizes, strict=True):
        batch.append(
            {
                "image": image,
                "original_size": tuple(int(value) for value in original_size.tolist()),
                "image_size": tuple(int(value) for value in image_size.tolist()),
                "point_coords": None,
                "point_labels": None,
            }
        )
    return batch


def autosam_dense_prompt_sam_call(*, batched_input: list[dict[str, Any]], sam_model, dense_embeddings):
    import torch

    with torch.no_grad():
        input_images = torch.stack([sam_model.preprocess(record["image"]) for record in batched_input], dim=0)
        image_embeddings = sam_model.image_encoder(input_images)
        sparse_embeddings_none, _dense_embeddings_none = sam_model.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, _iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def forward_autosam_batch(
    *,
    prompt_generator,
    sam_model,
    processed_images,
    original_sizes,
    image_sizes,
    prompt_image_size: int,
) -> AutoSamMonusegForwardOutput:
    import torch.nn.functional as F

    small_images = F.interpolate(
        processed_images,
        size=(int(prompt_image_size), int(prompt_image_size)),
        mode="bilinear",
        align_corners=True,
    )
    dense_embeddings = prompt_generator(small_images)
    batched_input = build_batched_input(processed_images, original_sizes, image_sizes)
    normalized_low_res_masks = norm_batch(
        autosam_dense_prompt_sam_call(
            batched_input=batched_input,
            sam_model=sam_model,
            dense_embeddings=dense_embeddings,
        )
    )
    postprocessed_masks = []
    for batch_index in range(int(normalized_low_res_masks.shape[0])):
        postprocessed_masks.append(
            sam_model.postprocess_masks(
                normalized_low_res_masks[batch_index : batch_index + 1],
                input_size=tuple(int(value) for value in image_sizes[batch_index].tolist()),
                original_size=tuple(int(value) for value in original_sizes[batch_index].tolist()),
            )
        )
    return AutoSamMonusegForwardOutput(
        normalized_low_res_masks=normalized_low_res_masks,
        postprocessed_masks=postprocessed_masks,
        dense_embeddings=dense_embeddings,
    )


def resolve_variant_name(args) -> str:
    return (
        f"autosam_{str(args.sam_model_type)}_order{int(args.autosam_backbone_order)}"
        f"_idim{int(args.prompt_image_size)}"
    )


def resolve_profile_deviations(args) -> dict[str, Any]:
    profile = resolve_monuseg_autosam_profile(args)
    current = {
        "sam_model_type": str(args.sam_model_type),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "drop_incomplete_accumulation": bool(args.drop_incomplete_accumulation),
        "num_workers": int(args.num_workers),
        "train_repeat_factor": int(args.train_repeat_factor),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "foreground_threshold": float(args.foreground_threshold),
        "train_augmentation_policy": str(args.train_augmentation_policy),
        "eval_every_epochs": int(args.eval_every_epochs),
        "selection_metric": str(args.selection_metric),
        "selection_split": str(args.selection_split),
        "selection_checkpoint_mode": str(args.selection_checkpoint_mode),
    }
    deviations = {}
    for key, value in current.items():
        if key in profile and profile[key] != value:
            deviations[key] = {"profile": profile[key], "resolved": value}
    return deviations


def prepare_output_dir(output_dir: str | None, *, run_kind: str, split: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    suffix = f"{run_kind}_autosam"
    if split is not None:
        suffix = f"{suffix}_{split}"
    return MONUSEG_AUTOSAM_OUTPUT_ROOT / suffix


def build_train_dataloader(
    *,
    samples: Sequence[MonusegBinarySample],
    batch_size: int,
    num_workers: int,
    repeat_factor: int,
    train_augmentation_policy: str,
    sam_transform,
    seed: int,
):
    import torch

    dataset = AutoSamMonusegDataset(
        samples=samples,
        train=True,
        repeat_factor=repeat_factor,
        train_augmentation_policy=train_augmentation_policy,
        sam_transform=sam_transform,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        collate_fn=collate_autosam_monuseg_batch,
        generator=generator,
    )


def build_eval_dataloader(
    *,
    samples: Sequence[MonusegBinarySample],
    num_workers: int,
    sam_transform,
):
    import torch

    dataset = AutoSamMonusegDataset(
        samples=samples,
        train=False,
        repeat_factor=1,
        train_augmentation_policy="none",
        sam_transform=sam_transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        collate_fn=collate_autosam_monuseg_batch,
    )


def load_monuseg_split_samples(
    *,
    split: str,
    dataset_name: str,
    cache_dir: str | None,
    limit: int | None = None,
    start_index: int = 0,
    subset_manifest: FewShotSubsetManifest | None = None,
    validation_holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    validation_subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> tuple[MonusegBinarySample, ...]:
    official_train_samples = tuple(
        iter_monuseg_binary_samples(
            split="train",
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )
    )
    official_test_samples = tuple(
        iter_monuseg_binary_samples(
            split="test",
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )
    )
    samples, _validation_manifest = resolve_monuseg_contract_split_samples(
        requested_split=split,
        official_train_samples=official_train_samples,
        official_test_samples=official_test_samples,
        item_id_getter=lambda sample: sample.crop_name,
        holdout_count=int(validation_holdout_count),
        subset_seed=int(validation_subset_seed),
    )
    if subset_manifest is not None:
        if split != "train":
            raise AutoSamMonusegRuntimeError(
                "train_subset_manifest may only be used with the repo-contract train split."
            )
        samples = select_items_by_few_shot_manifest(
            samples,
            subset_manifest,
            item_id_getter=lambda sample: sample.crop_name,
        )
    samples = samples[int(start_index) :]
    if limit is not None:
        samples = samples[: int(limit)]
    if not samples:
        raise AutoSamMonusegRuntimeError(
            f"No MoNuSeg samples were loaded for split '{split}' with limit={limit} and start_index={start_index}.",
        )
    return samples


def resolve_monuseg_validation_manifest_dict(
    *,
    dataset_name: str,
    cache_dir: str | None,
    validation_holdout_count: int = MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT,
    validation_subset_seed: int = MONUSEG_CONTRACT_VAL_SUBSET_SEED,
) -> dict[str, Any]:
    official_train_samples = tuple(
        iter_monuseg_binary_samples(
            split="train",
            dataset_name=dataset_name,
            cache_dir=cache_dir,
        )
    )
    _validation_samples, validation_manifest = resolve_monuseg_contract_split_samples(
        requested_split="val",
        official_train_samples=official_train_samples,
        official_test_samples=(),
        item_id_getter=lambda sample: sample.crop_name,
        holdout_count=int(validation_holdout_count),
        subset_seed=int(validation_subset_seed),
    )
    del _validation_samples
    return validation_manifest.to_json_dict()


def build_checkpoint_payload(
    *,
    prompt_generator,
    args,
    training_history_rows,
    epoch_eval_history_rows,
    trainable_parameter_count,
    checkpoint_role: str,
    selected_epoch: int | None,
    selected_metric_value: float | None,
) -> dict[str, Any]:
    return {
        "variant": resolve_variant_name(args),
        "dataset_id": MONUSEG_DATASET_ID,
        "route": MONUSEG_AUTOSAM_ROUTE,
        "model_family": "autosam_prompt_generator",
        "head_kind": "upstream_model_emb_dense_prompt",
        "prompt_generator_state_dict": prompt_generator.state_dict(),
        "trainable_parameter_count": int(trainable_parameter_count),
        "reproduction_profile": str(args.reproduction_profile),
        "sam_model_type": str(args.sam_model_type),
        "sam_checkpoint_path": str(args.sam_checkpoint_path),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
        "drop_incomplete_accumulation": bool(args.drop_incomplete_accumulation),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "foreground_threshold": float(args.foreground_threshold),
        "train_repeat_factor": int(args.train_repeat_factor),
        "train_augmentation_policy": resolve_train_augmentation_policy(args),
        "train_augmentation_summary": describe_train_augmentation_policy(resolve_train_augmentation_policy(args)),
        "selection_metric": str(args.selection_metric),
        "selection_split": str(args.selection_split),
        "selection_checkpoint_mode": str(args.selection_checkpoint_mode),
        "eval_every_epochs": int(args.eval_every_epochs),
        "training_history_rows": training_history_rows,
        "epoch_eval_history_rows": epoch_eval_history_rows,
        "checkpoint_role": checkpoint_role,
        "selected_epoch": selected_epoch,
        "selected_metric_value": selected_metric_value,
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "profile_deviations": resolve_profile_deviations(args),
    }


def compute_upstream_eval_metrics_for_sample(
    *,
    postprocessed_mask,
    processed_mask,
    original_size,
    image_size,
    prompt_image_size: int,
    foreground_threshold: float,
    sam_model,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    import torch.nn.functional as F

    resized_prediction = F.interpolate(
        postprocessed_mask,
        size=(int(prompt_image_size), int(prompt_image_size)),
        mode="bilinear",
        align_corners=True,
    )
    resized_target = sam_model.postprocess_masks(
        processed_mask.unsqueeze(0).unsqueeze(0),
        input_size=tuple(int(value) for value in image_size.tolist()),
        original_size=tuple(int(value) for value in original_size.tolist()),
    )
    resized_target = F.interpolate(
        resized_target,
        size=(int(prompt_image_size), int(prompt_image_size)),
        mode="nearest",
    )
    prediction_binary = np.asarray(
        resized_prediction[0, 0].detach().cpu().numpy() > float(foreground_threshold),
        dtype=bool,
    )
    target_binary = np.asarray(
        resized_target[0, 0].detach().cpu().numpy() > 0.5,
        dtype=bool,
    )
    iou, dice = compute_upstream_binary_iou_dice(prediction_binary, target_binary)
    return prediction_binary, target_binary, iou, dice


def build_eval_row(
    *,
    sample: MonusegBinarySample,
    checkpoint_path: Path,
    args,
    trainable_parameter_count: int,
    dense_embedding_shape: Sequence[int],
    low_res_shape: Sequence[int],
    direct_foreground_metrics,
    foreground_threshold: float,
    assignment,
    upstream_eval_iou: float,
    upstream_eval_dice: float,
) -> dict[str, Any]:
    total_pixels = int(sample.height * sample.width)
    row = {
        "variant": resolve_variant_name(args),
        "dataset_id": MONUSEG_DATASET_ID,
        "route": MONUSEG_AUTOSAM_ROUTE,
        "split": sample.split,
        "sample_index": int(sample.index),
        "crop_name": sample.crop_name,
        "grade_label": getattr(sample, "grade_label", None),
        "checkpoint_path": str(checkpoint_path),
        "foreground_evaluation_view": "direct_foreground",
        "foreground_threshold": float(foreground_threshold),
        "trainable_parameter_count": int(trainable_parameter_count),
        "dense_embedding_shape_json": json.dumps([int(value) for value in dense_embedding_shape]),
        "low_res_shape_json": json.dumps([int(value) for value in low_res_shape]),
        "image_height": int(sample.height),
        "image_width": int(sample.width),
        "upstream_eval_resolution": f"{int(args.prompt_image_size)}x{int(args.prompt_image_size)}",
        "upstream_eval_iou": float(upstream_eval_iou),
        "upstream_eval_dice": float(upstream_eval_dice),
        "direct_foreground_iou": float(direct_foreground_metrics.iou),
        "direct_foreground_dice": float(direct_foreground_metrics.dice),
        "direct_foreground_precision": float(direct_foreground_metrics.precision),
        "direct_foreground_recall": float(direct_foreground_metrics.recall),
        "direct_foreground_predicted_positive": int(direct_foreground_metrics.predicted_positive),
        "direct_foreground_target_positive": int(direct_foreground_metrics.target_positive),
        "direct_foreground_matched_positive": int(direct_foreground_metrics.matched_positive),
        "predicted_positive_fraction": float(direct_foreground_metrics.predicted_positive / total_pixels),
        "target_positive_fraction": float(direct_foreground_metrics.target_positive / total_pixels),
        "assignment_used": assignment.assignment_used,
        "direct_eval_miou": float(assignment.direct_miou),
        "direct_eval_ari": float(assignment.direct_ari),
        "swapped_eval_miou": float(assignment.swapped_miou),
        "swapped_eval_ari": float(assignment.swapped_ari),
    }
    row.update(
        build_canonical_evaluation_fields(
            miou=float(assignment.chosen_miou),
            ari=float(assignment.chosen_ari),
            evaluation_view="partition_invariant",
        )
    )
    return row


def save_result_bundle(
    *,
    output_dir: Path,
    sample: MonusegBinarySample,
    result: AutoSamMonusegEvalBundle,
    variant: str,
    save_visuals: bool,
) -> None:
    sample_index = int(sample.index)
    (output_dir / "sample_rows").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sample_rows" / f"{sample_index}.json", result.row)
    np.savez_compressed(
        output_dir / "masks" / f"{sample_index}.npz",
        foreground_prediction=np.asarray(result.foreground_prediction, dtype=bool),
        background_prediction=np.asarray(result.background_prediction, dtype=bool),
        target_foreground=np.asarray(sample.texture_a_mask, dtype=bool),
        target_background=np.asarray(sample.texture_b_mask, dtype=bool),
        foreground_probability=np.asarray(result.foreground_probability, dtype=np.float32),
    )
    if save_visuals:
        (output_dir / "visuals").mkdir(parents=True, exist_ok=True)
        save_prediction_panel(
            output_path=output_dir / "visuals" / f"{sample_index}.png",
            sample=sample,
            prediction_a=result.foreground_prediction,
            prediction_b=result.background_prediction,
            protocol=f"monuseg_autosam:{variant}",
            metric_summary=result.metric_summary,
        )


def build_summary(
    *,
    rows: list[dict[str, Any]],
    split: str,
    args,
    checkpoint_path: Path,
    checkpoint_role: str,
    run_kind: str,
    trainable_parameter_count: int,
) -> dict[str, Any]:
    mean_metrics = {name: float(mean(float(row[name]) for row in rows)) for name in MONUSEG_AUTOSAM_SCALAR_FIELDS}
    median_metrics = {name: float(median(float(row[name]) for row in rows)) for name in MONUSEG_AUTOSAM_SCALAR_FIELDS}
    return {
        "run_kind": run_kind,
        "variant": resolve_variant_name(args),
        "variant_summary": (
            "Upstream AutoSAM prompt-generator (ModelEmb HarDNet order "
            f"{int(args.autosam_backbone_order)}) over frozen SAM {args.sam_model_type}, "
            f"prompt image size {int(args.prompt_image_size)}."
        ),
        "reproduction_profile": str(args.reproduction_profile),
        "profile_deviations": resolve_profile_deviations(args),
        "dataset_id": MONUSEG_DATASET_ID,
        "route": MONUSEG_AUTOSAM_ROUTE,
        "split": split,
        "train_split": getattr(args, "train_split", None),
        "val_split": "val",
        "eval_split": getattr(args, "eval_split", None),
        "dataset_name": str(getattr(args, "dataset_name", MONUSEG_HF_DATASET_NAME)),
        "cache_dir": getattr(args, "cache_dir", None),
        "train_selection_policy": resolve_train_selection_policy(args),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "validation_subset_manifest_path": getattr(args, "_resolved_validation_subset_manifest_output_path", None),
        "validation_subset_manifest": getattr(args, "_resolved_validation_subset_manifest", None),
        "validation_split_policy": describe_monuseg_validation_policy(),
        "device": args.device,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_role": checkpoint_role,
        "num_evaluated_samples": len(rows),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "headline_evaluation_view": "upstream_eval_resized_binary",
        "primary_metric_name": "upstream_eval_iou",
        "secondary_metric_name": "upstream_eval_dice",
        "repo_native_primary_metric_name": "direct_foreground_iou",
        "repo_native_secondary_metric_name": "direct_foreground_dice",
        "aux_partition_primary_metric_name": "eval_miou",
        "aux_partition_secondary_metric_name": "eval_ari",
        "model_family": "autosam_prompt_generator",
        "head_kind": "upstream_model_emb_dense_prompt",
        "backbone_frozen": True,
        "sam_model_type": str(args.sam_model_type),
        "sam_checkpoint_path": str(args.sam_checkpoint_path),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "train_repeat_factor": int(args.train_repeat_factor),
        "trainable_parameter_count": int(trainable_parameter_count),
        "optimizer": "Adam",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
        "drop_incomplete_accumulation": bool(args.drop_incomplete_accumulation),
        "loss_name": "BCELoss + Dice on normalized low-res masks",
        "foreground_threshold": float(args.foreground_threshold),
        "train_augmentation_policy": resolve_train_augmentation_policy(args),
        "train_augmentation_summary": describe_train_augmentation_policy(resolve_train_augmentation_policy(args)),
        "selection_metric": str(args.selection_metric),
        "selection_split": str(args.selection_split),
        "selection_checkpoint_mode": str(args.selection_checkpoint_mode),
        "eval_every_epochs": int(args.eval_every_epochs),
        "preprocessing_summary": (
            "native MoNuSeg image/mask -> upstream paired transform -> "
            "ResizeLongestSide(1024) -> upstream preprocess semantics -> "
            f"prompt-generator bilinear downsample to {int(args.prompt_image_size)}x{int(args.prompt_image_size)}"
        ),
        "headline_metric_note": (
            "upstream_eval_iou/dice follow the public AutoSAM MoNuSeg inference order: "
            "SAM postprocess to original size -> resize both prediction and target to Idim -> threshold prediction."
        ),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "upstream_protocol_lock_path": str(REPO_ROOT / "docs" / "autosam_upstream_protocol.md"),
        "upstream_protocol_mandatory_disclosure": (
            "Any summary or paper-facing comparison using this upstream AutoSAM setting must state that "
            "the pinned upstream public protocol is locked by docs/autosam_upstream_protocol.md and that "
            "the operative 2D upstream implementation is train_3d + inference.py at the pinned commit, "
            "not the current train.py."
        ),
        "batch_audit_note": (
            "For paper-faithful MoNuSeg, the target batch contract is effective batch 10. On Run:AI A5000, "
            "the canonical cluster configuration is micro-batch 5 with gradient accumulation 2; lower-shot "
            "few-shot runs may become data-limited and use min(requested_batch_size, shot_count) instead."
        ),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": summary["variant"],
        "split": summary["split"],
        "checkpoint_role": summary["checkpoint_role"],
        "num_evaluated_samples": summary["num_evaluated_samples"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "upstream_eval_iou": summary["mean_metrics"]["upstream_eval_iou"],
        "upstream_eval_dice": summary["mean_metrics"]["upstream_eval_dice"],
        "direct_foreground_iou": summary["mean_metrics"]["direct_foreground_iou"],
        "direct_foreground_dice": summary["mean_metrics"]["direct_foreground_dice"],
        "eval_miou": summary["mean_metrics"]["eval_miou"],
        "eval_ari": summary["mean_metrics"]["eval_ari"],
        "checkpoint_path": summary["checkpoint_path"],
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MoNuSeg AutoSAM Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Reproduction profile: `{summary['reproduction_profile']}`",
        f"- Split: `{summary['split']}`",
        f"- Checkpoint role: `{summary['checkpoint_role']}`",
        f"- Train selection policy: `{summary.get('train_selection_policy')}`",
        f"- Train subset manifest: `{summary.get('train_subset_manifest_output_path') or summary.get('train_subset_manifest_path')}`",
        f"- Trainable params: `{summary['trainable_parameter_count']}`",
        f"- SAM model type: `{summary['sam_model_type']}`",
        f"- Prompt image size: `{summary['prompt_image_size']}`",
        f"- HarDNet order: `{summary['autosam_backbone_order']}`",
        f"- Train repeat factor: `{summary['train_repeat_factor']}`",
        f"- Train augmentation: `{summary['train_augmentation_summary']}`",
        f"- Selection metric: `{summary['selection_metric']}` on split `{summary['selection_split']}`",
        f"- Upstream protocol lock: `{summary['upstream_protocol_lock_path']}`",
        f"- Batch audit: `{summary['batch_audit_note']}`",
        f"- Headline upstream IoU: `{summary['mean_metrics']['upstream_eval_iou']:.6f}`",
        f"- Headline upstream Dice: `{summary['mean_metrics']['upstream_eval_dice']:.6f}`",
        f"- Repo-native direct foreground IoU: `{summary['mean_metrics']['direct_foreground_iou']:.6f}`",
        f"- Repo-native direct foreground Dice: `{summary['mean_metrics']['direct_foreground_dice']:.6f}`",
        f"- Auxiliary partition mIoU: `{summary['mean_metrics']['eval_miou']:.6f}`",
        f"- Auxiliary partition ARI: `{summary['mean_metrics']['eval_ari']:.6f}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "",
        "## Mandatory Disclosure",
        "",
        f"- {summary['upstream_protocol_mandatory_disclosure']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def rewrite_summary(*, output_dir: Path, summary: dict[str, Any], basename: str = "summary") -> None:
    write_json(output_dir / f"{basename}.json", summary)
    write_csv(output_dir / f"{basename}.csv", [flatten_summary(summary)])
    write_text(output_dir / f"{basename}.md", build_summary_markdown(summary))


def build_run_config(args, *, run_kind: str, dataset_partition=None, selected_sample_count: int | None = None) -> dict[str, Any]:
    config = {
        "command": getattr(args, "command", None),
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(str(value)) for value in sys.argv),
        "run_kind": run_kind,
        "dataset_id": MONUSEG_DATASET_ID,
        "route": MONUSEG_AUTOSAM_ROUTE,
        "dataset_name": getattr(args, "dataset_name", MONUSEG_HF_DATASET_NAME),
        "cache_dir": getattr(args, "cache_dir", None),
        "train_selection_policy": resolve_train_selection_policy(args),
        "split": getattr(args, "split", None),
        "train_split": getattr(args, "train_split", None),
        "val_split": "val",
        "eval_split": getattr(args, "eval_split", None),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "validation_subset_manifest_output_path": getattr(args, "_resolved_validation_subset_manifest_output_path", None),
        "validation_subset_manifest": getattr(args, "_resolved_validation_subset_manifest", None),
        "validation_split_policy": describe_monuseg_validation_policy(),
        "variant": resolve_variant_name(args),
        "reproduction_profile": str(args.reproduction_profile),
        # 1. Profile Resolution & Deviation Tracking
        "paper_faithful_monuseg": bool(getattr(args, "paper_faithful_monuseg", False)),
        "profile_defaults": resolve_monuseg_autosam_profile(args),
        "profile_deviations": resolve_profile_deviations(args),
        "device": args.device,
        "sam_model_type": str(args.sam_model_type),
        "sam_checkpoint_path": str(args.sam_checkpoint_path),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
        "drop_incomplete_accumulation": bool(args.drop_incomplete_accumulation),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "foreground_threshold": float(args.foreground_threshold),
        "train_repeat_factor": int(args.train_repeat_factor),
        "train_augmentation_policy": resolve_train_augmentation_policy(args),
        "train_augmentation_summary": describe_train_augmentation_policy(resolve_train_augmentation_policy(args)),
        "selection_metric": str(args.selection_metric),
        "selection_split": str(args.selection_split),
        "selection_checkpoint_mode": str(args.selection_checkpoint_mode),
        "eval_every_epochs": int(args.eval_every_epochs),
        "num_workers": int(args.num_workers),
        "save_visuals": bool(args.save_visuals),
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_experiment_terms_markdown(
    args,
    *,
    run_kind: str,
    train_sample_count: int | None,
    eval_sample_count: int | None,
    checkpoint_path: Path | None,
) -> str:
    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            bullets=(
                f"Command family: `{getattr(args, 'command', 'monuseg-autosam-faithful')}`.",
                f"Run kind: `{run_kind}`.",
                f"Dataset: `{MONUSEG_DATASET_ID}` from `{getattr(args, 'dataset_name', MONUSEG_HF_DATASET_NAME)}`.",
                f"Upstream AutoSAM checkout: `{AUTOSAM_REPO_ROOT}`.",
                f"Reproduction profile: `{getattr(args, 'reproduction_profile', None)}`.",
                f"SAM checkpoint: `{getattr(args, 'sam_checkpoint_path', None)}` with model type `{getattr(args, 'sam_model_type', None)}`.",
            ),
        ),
        ExperimentTermsSection(
            title="Data And Split Separation",
            bullets=(
                f"Training split: `{getattr(args, 'train_split', None)}` with limit `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Validation split: `val` derived from the official train split via `{describe_monuseg_validation_policy()}`.",
                f"Evaluation split: `{getattr(args, 'eval_split', getattr(args, 'split', None))}` with limit `{getattr(args, 'eval_limit', getattr(args, 'limit', None))}`; loaded `{eval_sample_count}` sample(s).",
                (
                    f"Train subset manifest: `{getattr(args, '_resolved_train_subset_manifest_output_path', getattr(args, 'train_subset_manifest', None))}` selecting `{getattr(args, '_resolved_train_subset_manifest', {}).get('shot_count')}` image(s) with subset seed `{getattr(args, '_resolved_train_subset_manifest', {}).get('subset_seed')}`."
                    if getattr(args, "_resolved_train_subset_manifest", None) is not None
                    else "No train subset manifest is applied. The repo-contract MoNuSeg train split is the official train split minus the deterministic validation holdout."
                ),
                f"Validation subset manifest: `{getattr(args, '_resolved_validation_subset_manifest_output_path', None)}`.",
            ),
        ),
        ExperimentTermsSection(
            title="Faithful Training Path",
            bullets=(
                f"Prompt generator: upstream `ModelEmb` with HarDNet order `{int(args.autosam_backbone_order)}` and depth_wise `{bool(args.autosam_depth_wise)}`.",
                f"Prompt image size: `{int(args.prompt_image_size)}`.",
                f"Optimizer: `Adam(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
                f"Mini-batch size: `{int(args.batch_size)}` with gradient accumulation steps `{int(args.gradient_accumulation_steps)}` (effective batch `{int(args.batch_size) * int(args.gradient_accumulation_steps)}`) and train repeat factor `{int(args.train_repeat_factor)}`.",
                f"Train augmentation: `{describe_train_augmentation_policy(resolve_train_augmentation_policy(args))}`.",
                f"Checkpoint selection: `{str(args.selection_checkpoint_mode)}` using `{str(args.selection_metric)}` on split `{str(args.selection_split)}` every `{int(args.eval_every_epochs)}` epoch(s).",
            ),
        ),
        ExperimentTermsSection(
            title="Outputs",
            bullets=(
                "Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `protocol.json`, `paper_row.json`, `fairness.md`, `benchmark_manifest.csv`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl`.",
                "Training runs additionally write `checkpoint.pt`, `best_checkpoint.pt`, `train_history.csv`, and `epoch_eval_history.csv`.",
                (
                    f"Checkpoint consumed by this run: `{checkpoint_path}`."
                    if checkpoint_path is not None
                    else "This run trains a fresh AutoSAM prompt-generator checkpoint and evaluates both the headline checkpoint and the final checkpoint."
                ),
            ),
        ),
    ]
    return render_experiment_terms_markdown(
        title="MoNuSeg AutoSAM Experiment Terms",
        summary_lines=(
            "This file records the exact MoNuSeg AutoSAM faithful-reproduction run that produced this directory.",
            "Headline metrics in `summary.json` use the upstream-style resized-binary evaluation view.",
        ),
        sections=sections,
        related_paths=(
            str(AUTOSAM_REPO_ROOT / "models" / "model_single.py"),
            str(AUTOSAM_REPO_ROOT / "dataset" / "MoNuBrain.py"),
            str(AUTOSAM_REPO_ROOT / "inference.py"),
            str(Path("src/rwtd_sam3/eval/monuseg_autosam.py")),
        ),
    )


def evaluate_monuseg_autosam(
    *,
    prompt_generator,
    checkpoint_path: Path,
    samples: Sequence[MonusegBinarySample],
    args,
    run_kind: str,
    checkpoint_role: str,
    trainable_parameter_count: int,
    output_dir: Path | None,
    save_artifacts: bool,
) -> dict[str, Any]:
    import torch

    if save_artifacts and output_dir is None:
        raise AutoSamMonusegRuntimeError("save_artifacts=True requires a concrete output_dir.")
    device = resolve_torch_device(args.device)
    sam_transform = load_upstream_autosam_resize_longest_side_class()(1024)
    sam_model = build_upstream_autosam_sam(
        model_type=str(args.sam_model_type),
        checkpoint_path=str(args.sam_checkpoint_path),
    ).to(device)
    for parameter in sam_model.parameters():
        parameter.requires_grad = False
    sam_model.eval()
    prompt_generator.to(device)
    prompt_generator.eval()

    eval_loader = build_eval_dataloader(
        samples=samples,
        num_workers=int(args.num_workers),
        sam_transform=sam_transform,
    )
    rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []

    for batch in eval_loader:
        processed_images = batch["images"].to(device)
        processed_masks = batch["masks"].to(device)
        original_sizes = batch["original_sizes"].to(device)
        image_sizes = batch["image_sizes"].to(device)
        source_sample = batch["samples"][0]
        with torch.no_grad():
            output = forward_autosam_batch(
                prompt_generator=prompt_generator,
                sam_model=sam_model,
                processed_images=processed_images,
                original_sizes=original_sizes,
                image_sizes=image_sizes,
                prompt_image_size=int(args.prompt_image_size),
            )
        foreground_probability = output.postprocessed_masks[0][0, 0].detach().cpu().numpy().astype(np.float32)
        foreground_prediction = np.asarray(foreground_probability > float(args.foreground_threshold), dtype=bool)
        background_prediction = np.logical_not(foreground_prediction)
        direct_foreground_metrics = compute_binary_metrics(
            np.asarray(foreground_prediction, dtype=bool),
            np.asarray(source_sample.texture_a_mask, dtype=bool),
        )
        assignment = select_best_binary_assignment(
            np.asarray(foreground_prediction, dtype=bool),
            np.asarray(background_prediction, dtype=bool),
            np.asarray(source_sample.texture_a_mask, dtype=bool),
            np.asarray(source_sample.texture_b_mask, dtype=bool),
            "autosam_foreground",
            "autosam_background_complement",
        )
        _upstream_pred_binary, _upstream_target_binary, upstream_eval_iou, upstream_eval_dice = compute_upstream_eval_metrics_for_sample(
            postprocessed_mask=output.postprocessed_masks[0],
            processed_mask=processed_masks[0],
            original_size=original_sizes[0],
            image_size=image_sizes[0],
            prompt_image_size=int(args.prompt_image_size),
            foreground_threshold=float(args.foreground_threshold),
            sam_model=sam_model,
        )
        row = build_eval_row(
            sample=source_sample,
            checkpoint_path=checkpoint_path,
            args=args,
            trainable_parameter_count=trainable_parameter_count,
            dense_embedding_shape=tuple(int(value) for value in output.dense_embeddings.shape),
            low_res_shape=tuple(int(value) for value in output.normalized_low_res_masks.shape),
            direct_foreground_metrics=direct_foreground_metrics,
            foreground_threshold=float(args.foreground_threshold),
            assignment=assignment,
            upstream_eval_iou=upstream_eval_iou,
            upstream_eval_dice=upstream_eval_dice,
        )
        rows.append(row)
        metric_summary = (
            f"Upstream IoU={row['upstream_eval_iou']:.3f} "
            f"Dice={row['upstream_eval_dice']:.3f} "
            f"Direct IoU={row['direct_foreground_iou']:.3f} "
            f"Direct Dice={row['direct_foreground_dice']:.3f}"
        )
        if save_artifacts and output_dir is not None:
            result = AutoSamMonusegEvalBundle(
                row=row,
                metric_summary=metric_summary,
                foreground_prediction=foreground_prediction,
                background_prediction=background_prediction,
                foreground_probability=foreground_probability,
            )
            save_result_bundle(
                output_dir=output_dir,
                sample=source_sample,
                result=result,
                variant=resolve_variant_name(args),
                save_visuals=bool(args.save_visuals),
            )
            if args.save_visuals:
                visual_records.append(
                    build_visual_record(
                        sample=source_sample,
                        evaluation=SampleEvaluationResult(
                            row=row,
                            metric_summary=metric_summary,
                            prediction_a=foreground_prediction,
                            prediction_b=background_prediction,
                        ),
                        protocol=f"monuseg_autosam:{resolve_variant_name(args)}",
                        visual_path=Path("visuals") / f"{source_sample.index}.png",
                    )
                )

    if not rows:
        raise AutoSamMonusegRuntimeError("MoNuSeg AutoSAM evaluation produced no sample rows.")

    summary = build_summary(
        rows=rows,
        split=str(getattr(args, "eval_split", getattr(args, "split", "test"))),
        args=args,
        checkpoint_path=checkpoint_path,
        checkpoint_role=checkpoint_role,
        run_kind=run_kind,
        trainable_parameter_count=trainable_parameter_count,
    )
    if save_artifacts and output_dir is not None:
        summary = evaluate_run(
            output_dir=output_dir,
            route_name="monuseg_autosam_reproduction",
            split_role=str(getattr(args, "eval_split", getattr(args, "split", "test"))),
            rows=rows,
            summary=summary,
            protocol_family="upstream_faithful",
            primary_metric_pair=UPSTREAM_FAITHFUL_PRIMARY,
            selection_split=str(args.selection_split),
            headline_split="test",
            safety_tag="paper_safe",
            summary_markdown_builder=build_summary_markdown,
            summary_csv_flattener=flatten_summary,
            deviations=tuple(sorted(resolve_profile_deviations(args).keys())),
            visual_records=visual_records,
        )
    return summary


def build_prompt_generator_from_checkpoint(checkpoint: dict[str, Any]):
    prompt_generator = build_upstream_autosam_prompt_generator(
        order=int(checkpoint["autosam_backbone_order"]),
        depth_wise=bool(checkpoint["autosam_depth_wise"]),
    )
    prompt_generator.load_state_dict(checkpoint["prompt_generator_state_dict"])
    return prompt_generator


def _build_epoch_eval_row(epoch: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "upstream_eval_iou": float(summary["mean_metrics"]["upstream_eval_iou"]),
        "upstream_eval_dice": float(summary["mean_metrics"]["upstream_eval_dice"]),
        "direct_foreground_iou": float(summary["mean_metrics"]["direct_foreground_iou"]),
        "direct_foreground_dice": float(summary["mean_metrics"]["direct_foreground_dice"]),
        "eval_miou": float(summary["mean_metrics"]["eval_miou"]),
        "eval_ari": float(summary["mean_metrics"]["eval_ari"]),
        "selection_metric_name": str(summary["selection_metric"]),
        "selection_metric_value": float(summary["mean_metrics"][str(summary["selection_metric"])]),
        "checkpoint_role": str(summary["checkpoint_role"]),
    }


def train_monuseg_autosam(
    *,
    train_samples: Sequence[MonusegBinarySample],
    selection_samples: Sequence[MonusegBinarySample],
    output_dir: Path,
    args,
) -> AutoSamMonusegTrainingArtifacts:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = resolve_torch_device(args.device)
    set_global_training_seed(int(getattr(args, "seed", 0)))
    sam_transform = load_upstream_autosam_resize_longest_side_class()(1024)
    prompt_generator = build_upstream_autosam_prompt_generator(
        order=int(args.autosam_backbone_order),
        depth_wise=bool(args.autosam_depth_wise),
    ).to(device)
    sam_model = build_upstream_autosam_sam(
        model_type=str(args.sam_model_type),
        checkpoint_path=str(args.sam_checkpoint_path),
    ).to(device)
    for parameter in sam_model.parameters():
        parameter.requires_grad = False
    sam_model.eval()

    train_loader = build_train_dataloader(
        samples=train_samples,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        repeat_factor=int(args.train_repeat_factor),
        train_augmentation_policy=resolve_train_augmentation_policy(args),
        sam_transform=sam_transform,
        seed=int(getattr(args, "seed", 0)),
    )
    num_batches_per_epoch = len(train_loader)
    if num_batches_per_epoch < 1:
        raise AutoSamMonusegRuntimeError(
            "MoNuSeg AutoSAM training resolved zero optimization batches. Increase train_repeat_factor or lower batch_size.",
        )
    optimizer = torch.optim.Adam(
        [parameter for parameter in prompt_generator.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.BCELoss()
    trainable_parameter_count = count_trainable_parameters(prompt_generator)
    training_history_rows: list[dict[str, Any]] = []
    epoch_eval_history_rows: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_metric_value: float | None = None
    best_checkpoint_path = output_dir / "best_checkpoint.pt"

    LOGGER.info(
        "MoNuSeg AutoSAM training start | variant=%s lr=%.6f wd=%.6f epochs=%d batch=%d accum=%d repeat=%d aug=%s trainable_params=%d",
        resolve_variant_name(args),
        float(args.learning_rate),
        float(args.weight_decay),
        int(args.num_epochs),
        int(args.batch_size),
        int(args.gradient_accumulation_steps),
        int(args.train_repeat_factor),
        resolve_train_augmentation_policy(args),
        trainable_parameter_count,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(args.num_epochs)):
        prompt_generator.train()
        if getattr(args, "freeze_prompt_bn", False):
            for module in prompt_generator.modules():
                if isinstance(module, (nn.modules.batchnorm._BatchNorm)):
                    module.eval()
        epoch_losses: list[float] = []
        epoch_bce: list[float] = []
        epoch_dice: list[float] = []
        epoch_pred_fraction: list[float] = []
        steps_since_update = 0
        for step_index, batch in enumerate(train_loader):
            processed_images = batch["images"].to(device)
            processed_masks = batch["masks"].to(device)
            original_sizes = batch["original_sizes"].to(device)
            image_sizes = batch["image_sizes"].to(device)
            output = forward_autosam_batch(
                prompt_generator=prompt_generator,
                sam_model=sam_model,
                processed_images=processed_images,
                original_sizes=original_sizes,
                image_sizes=image_sizes,
                prompt_image_size=int(args.prompt_image_size),
            )
            target_low_res = F.interpolate(
                processed_masks.unsqueeze(1),
                size=output.normalized_low_res_masks.shape[-2:],
                mode="nearest",
            ).float().clamp(0.0, 1.0)
            bce = criterion(output.normalized_low_res_masks, target_low_res)
            dice = dice_loss(target_low_res, output.normalized_low_res_masks)
            loss = bce + dice
            backward_loss = loss / float(args.gradient_accumulation_steps)
            backward_loss.backward()
            steps_since_update += 1
            if steps_since_update >= int(args.gradient_accumulation_steps):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_bce.append(float(bce.detach().cpu().item()))
            epoch_dice.append(float(dice.detach().cpu().item()))
            epoch_pred_fraction.append(float(output.normalized_low_res_masks.detach().mean().cpu().item()))
        if steps_since_update > 0 and not bool(args.drop_incomplete_accumulation):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        elif steps_since_update > 0:
            optimizer.zero_grad(set_to_none=True)
        history_row = {
            "epoch": epoch + 1,
            "mean_train_loss": float(mean(epoch_losses)) if epoch_losses else None,
            "mean_bce_loss": float(mean(epoch_bce)) if epoch_bce else None,
            "mean_dice_loss": float(mean(epoch_dice)) if epoch_dice else None,
            "mean_predicted_positive_fraction": float(mean(epoch_pred_fraction)) if epoch_pred_fraction else None,
            "num_batches": int(len(epoch_losses)),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "drop_incomplete_accumulation": bool(args.drop_incomplete_accumulation),
        }
        training_history_rows.append(history_row)
        write_csv(output_dir / "train_history.csv", training_history_rows)
        LOGGER.info(
            "MoNuSeg AutoSAM epoch=%d/%d | mean_loss=%.6f mean_bce=%.6f mean_dice=%.6f mean_pred=%.4f batches=%d",
            epoch + 1,
            int(args.num_epochs),
            float(history_row["mean_train_loss"] or 0.0),
            float(history_row["mean_bce_loss"] or 0.0),
            float(history_row["mean_dice_loss"] or 0.0),
            float(history_row["mean_predicted_positive_fraction"] or 0.0),
            int(history_row["num_batches"]),
        )

        if ((epoch + 1) % int(args.eval_every_epochs)) != 0:
            continue
        selection_summary = evaluate_monuseg_autosam(
            prompt_generator=copy.deepcopy(prompt_generator).to(device),
            checkpoint_path=Path("<in_memory_epoch_selection>"),
            samples=selection_samples,
            args=args,
            run_kind="epoch_eval",
            checkpoint_role="epoch_selection",
            trainable_parameter_count=trainable_parameter_count,
            output_dir=None,
            save_artifacts=False,
        )
        epoch_eval_row = _build_epoch_eval_row(epoch + 1, selection_summary)
        epoch_eval_history_rows.append(epoch_eval_row)
        write_csv(output_dir / "epoch_eval_history.csv", epoch_eval_history_rows)
        selection_metric_value = float(selection_summary["mean_metrics"][str(args.selection_metric)])
        if best_metric_value is None or selection_metric_value > best_metric_value:
            best_metric_value = selection_metric_value
            best_epoch = epoch + 1
            LOGGER.info(
                "MoNuSeg AutoSAM new best checkpoint | epoch=%d selection_metric=%s value=%.6f",
                best_epoch,
                str(args.selection_metric),
                best_metric_value,
            )
            best_payload = build_checkpoint_payload(
                prompt_generator=prompt_generator,
                args=args,
                training_history_rows=training_history_rows,
                epoch_eval_history_rows=epoch_eval_history_rows,
                trainable_parameter_count=trainable_parameter_count,
                checkpoint_role="best_eval",
                selected_epoch=best_epoch,
                selected_metric_value=best_metric_value,
            )
            torch.save(best_payload, best_checkpoint_path)

    final_checkpoint_path = output_dir / "checkpoint.pt"
    final_payload = build_checkpoint_payload(
        prompt_generator=prompt_generator,
        args=args,
        training_history_rows=training_history_rows,
        epoch_eval_history_rows=epoch_eval_history_rows,
        trainable_parameter_count=trainable_parameter_count,
        checkpoint_role="final",
        selected_epoch=int(args.num_epochs),
        selected_metric_value=best_metric_value,
    )
    torch.save(final_payload, final_checkpoint_path)
    if best_epoch is None and best_checkpoint_path.exists():
        best_checkpoint_path.unlink()
        best_checkpoint_path = None
    return AutoSamMonusegTrainingArtifacts(
        final_checkpoint_path=final_checkpoint_path,
        best_checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        trainable_parameter_count=trainable_parameter_count,
        training_history_rows=training_history_rows,
        epoch_eval_history_rows=epoch_eval_history_rows,
        best_epoch_by_selection_metric=best_epoch,
        best_selection_metric_value=best_metric_value,
    )


def run_monuseg_autosam_train(args) -> dict[str, Any]:
    import torch

    if getattr(args, "train_limit", None) is not None and getattr(args, "train_subset_manifest", None) not in (None, ""):
        raise AutoSamMonusegRuntimeError(
            "train_limit and train_subset_manifest are mutually exclusive for MoNuSeg AutoSAM training."
        )
    if str(getattr(args, "selection_split", "val")) != "val":
        raise AutoSamMonusegRuntimeError(
            "Repo contract requires MoNuSeg AutoSAM checkpoint selection on the deterministic validation split ('val')."
        )
    resolved_checkpoint_path = Path(str(args.sam_checkpoint_path))
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint path does not exist: {resolved_checkpoint_path}")

    # 1. Profile Resolution & Deviation Tracking
    paper_faithful_monuseg = getattr(args, "paper_faithful_monuseg", False)
    profile = resolve_monuseg_autosam_profile(args)
    if paper_faithful_monuseg:
        # Override with exact paper recipe
        profile.update({
            "prompt_image_size": 512,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "num_epochs": 200,
            "train_repeat_factor": 5,
            "train_augmentation_policy": "monuseg_paper_v1",
        })
        # Force args to match profile for consistent behavior
        args.prompt_image_size = 512
        args.learning_rate = 3e-4
        args.weight_decay = 1e-5
        args.num_epochs = 200
        args.train_repeat_factor = 5
        args.train_augmentation_policy = "monuseg_paper_v1"
        LOGGER.info("[PAPER-FAITHFUL] Forcing MoNuSeg paper-identical hyperparameters.")
    
    deviations = resolve_profile_deviations(args)

    train_subset_manifest = resolve_train_subset_manifest(args)
    setattr(
        args,
        "_resolved_train_subset_manifest",
        train_subset_manifest.to_json_dict() if train_subset_manifest is not None else None,
    )

    # 2. Batch Labeling & Validation
    batch_size = int(args.batch_size)
    acc_steps = int(args.gradient_accumulation_steps)
    effective_bs = batch_size * acc_steps
    
    if paper_faithful_monuseg:
        if effective_bs != 10:
             raise AutoSamMonusegRuntimeError(f"Paper-faithful mode requires effective batch size 10, got {effective_bs}")
        
        if batch_size == 10 and acc_steps == 1:
            batch_label = "paperfaithful_truebs10"
            LOGGER.info(f"[BATCH-AUDIT] Running in TRUE PAPER BATCH mode (BS10).")
        else:
            batch_label = f"paperfaithful_effbs10_micro{batch_size}acc{acc_steps}"
            LOGGER.warning(f"[BATCH-AUDIT] Running in HARDWARE FALLBACK mode: effective BS10 via BS{batch_size}xAcc{acc_steps}.")
    else:
        batch_label = f"bs{batch_size}_acc{acc_steps}"

    # 3. Output Directory Preparation
    output_dir = prepare_output_dir(args.output_dir, run_kind=f"train_{batch_label}", split=args.eval_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_subset_manifest is not None:
        local_manifest_path = output_dir / "train_subset_manifest.json"
        write_json(local_manifest_path, train_subset_manifest.to_json_dict())
        setattr(args, "_resolved_train_subset_manifest_output_path", str(local_manifest_path))
    else:
        setattr(args, "_resolved_train_subset_manifest_output_path", None)
    validation_manifest_payload = resolve_monuseg_validation_manifest_dict(
        dataset_name=str(args.dataset_name),
        cache_dir=getattr(args, "cache_dir", None),
        validation_holdout_count=int(getattr(args, "validation_holdout_count", MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT)),
        validation_subset_seed=int(getattr(args, "validation_subset_seed", MONUSEG_CONTRACT_VAL_SUBSET_SEED)),
    )
    validation_manifest_path = output_dir / "validation_subset_manifest.json"
    write_json(validation_manifest_path, validation_manifest_payload)
    setattr(args, "_resolved_validation_subset_manifest", validation_manifest_payload)
    setattr(args, "_resolved_validation_subset_manifest_output_path", str(validation_manifest_path))

    train_samples = load_monuseg_split_samples(
        split=str(args.train_split),
        dataset_name=str(args.dataset_name),
        cache_dir=getattr(args, "cache_dir", None),
        limit=getattr(args, "train_limit", None),
        subset_manifest=train_subset_manifest,
        validation_holdout_count=int(getattr(args, "validation_holdout_count", MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT)),
        validation_subset_seed=int(getattr(args, "validation_subset_seed", MONUSEG_CONTRACT_VAL_SUBSET_SEED)),
    )
    selection_samples = load_monuseg_split_samples(
        split=str(args.selection_split),
        dataset_name=str(args.dataset_name),
        cache_dir=getattr(args, "cache_dir", None),
        limit=getattr(args, "eval_limit", None),
        validation_holdout_count=int(getattr(args, "validation_holdout_count", MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT)),
        validation_subset_seed=int(getattr(args, "validation_subset_seed", MONUSEG_CONTRACT_VAL_SUBSET_SEED)),
    )

    config = build_run_config(args, run_kind="train")
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_experiment_terms_markdown(
            args,
            run_kind="train",
            train_sample_count=len(train_samples),
            eval_sample_count=len(selection_samples),
            checkpoint_path=None,
        ),
    )

    if getattr(args, "monuseg_frozen_sam_head", False):
        from rwtd_sam3.eval.monuseg_frozen_feature_mask_head import run_glas_frozen_mask_head_train
        # Prepare arguments for the frozen head route
        # Note: we reuse the same args but some fields have different defaults/names in the other module
        # The other module is generic but we call it for MoNuSeg here.
        args.variant = "f2+refine_f1" # target variant for the closure audit
        LOGGER.info("[FROZEN-HEAD] Redirecting to MoNuSeg Frozen SAM Head (f2+refine f1) training.")
        return run_glas_frozen_mask_head_train(args)

    training_artifacts = train_monuseg_autosam(
        train_samples=train_samples,
        selection_samples=selection_samples,
        output_dir=output_dir,
        args=args,
    )
    final_checkpoint = torch.load(training_artifacts.final_checkpoint_path, map_location="cpu")
    final_prompt_generator = build_prompt_generator_from_checkpoint(final_checkpoint)
    final_summary = evaluate_monuseg_autosam(
        prompt_generator=final_prompt_generator,
        checkpoint_path=training_artifacts.final_checkpoint_path,
        samples=selection_samples,
        args=args,
        run_kind="train_final_checkpoint",
        checkpoint_role="final",
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=None,
        save_artifacts=False,
    )
    write_json(output_dir / "final_checkpoint_summary.json", final_summary)
    write_text(output_dir / "final_checkpoint_summary.md", build_summary_markdown(final_summary))

    if training_artifacts.best_checkpoint_path is not None:
        best_checkpoint = torch.load(training_artifacts.best_checkpoint_path, map_location="cpu")
        headline_prompt_generator = build_prompt_generator_from_checkpoint(best_checkpoint)
        headline_checkpoint_path = training_artifacts.best_checkpoint_path
        headline_checkpoint_role = "best_eval"
    else:
        headline_prompt_generator = build_prompt_generator_from_checkpoint(final_checkpoint)
        headline_checkpoint_path = training_artifacts.final_checkpoint_path
        headline_checkpoint_role = "final"

    summary = evaluate_monuseg_autosam(
        prompt_generator=headline_prompt_generator,
        checkpoint_path=headline_checkpoint_path,
        samples=selection_samples,
        args=args,
        run_kind="train",
        checkpoint_role=headline_checkpoint_role,
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=output_dir,
        save_artifacts=True,
    )
    summary["headline_checkpoint_path"] = str(headline_checkpoint_path)
    summary["headline_checkpoint_role"] = headline_checkpoint_role
    summary["best_checkpoint_path"] = str(training_artifacts.best_checkpoint_path) if training_artifacts.best_checkpoint_path is not None else None
    summary["best_epoch_by_selection_metric"] = training_artifacts.best_epoch_by_selection_metric
    summary["best_selection_metric_value"] = training_artifacts.best_selection_metric_value
    summary["final_checkpoint_path"] = str(training_artifacts.final_checkpoint_path)
    summary["final_checkpoint_mean_metrics"] = dict(final_summary["mean_metrics"])
    summary["epoch_eval_history_path"] = str(output_dir / "epoch_eval_history.csv")
    summary["final_checkpoint_summary_path"] = str(output_dir / "final_checkpoint_summary.json")
    rewrite_contract_summary(
        output_dir=output_dir,
        summary=summary,
        summary_markdown_builder=build_summary_markdown,
        summary_csv_flattener=flatten_summary,
    )
    return summary


def run_monuseg_autosam_eval(args) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MoNuSeg AutoSAM checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    setattr(args, "sam_model_type", str(checkpoint["sam_model_type"]))
    setattr(args, "sam_checkpoint_path", str(checkpoint["sam_checkpoint_path"]))
    setattr(args, "prompt_image_size", int(checkpoint["prompt_image_size"]))
    setattr(args, "autosam_backbone_order", int(checkpoint["autosam_backbone_order"]))
    setattr(args, "autosam_depth_wise", bool(checkpoint["autosam_depth_wise"]))
    setattr(args, "batch_size", int(checkpoint["batch_size"]))
    setattr(args, "gradient_accumulation_steps", int(checkpoint.get("gradient_accumulation_steps", 1)))
    setattr(args, "drop_incomplete_accumulation", bool(checkpoint.get("drop_incomplete_accumulation", False)))
    setattr(args, "learning_rate", float(checkpoint["learning_rate"]))
    setattr(args, "weight_decay", float(checkpoint["weight_decay"]))
    setattr(args, "num_epochs", int(checkpoint["num_epochs"]))
    setattr(args, "train_repeat_factor", int(checkpoint.get("train_repeat_factor", 5)))
    setattr(args, "train_augmentation_policy", str(checkpoint.get("train_augmentation_policy", "monuseg_paper_v1")))
    setattr(args, "selection_metric", str(checkpoint.get("selection_metric", "upstream_eval_iou")))
    setattr(args, "selection_split", str(checkpoint.get("selection_split", "val")))
    setattr(args, "selection_checkpoint_mode", str(checkpoint.get("selection_checkpoint_mode", "best_eval")))
    setattr(args, "eval_every_epochs", int(checkpoint.get("eval_every_epochs", 1)))
    setattr(args, "reproduction_profile", str(checkpoint.get("reproduction_profile", MONUSEG_AUTOSAM_REPRODUCTION_PROFILE["profile_name"])))
    prompt_generator = build_prompt_generator_from_checkpoint(checkpoint)

    overview = load_monuseg_binary_overview(
        split=args.split,
        dataset_name=str(args.dataset_name),
        cache_dir=getattr(args, "cache_dir", None),
    )
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_samples = resolve_eval_sample_count(getattr(args, "limit", None), overview.num_examples, dataset_partition)
    eval_samples = load_monuseg_split_samples(
        split=str(args.split),
        dataset_name=str(args.dataset_name),
        cache_dir=getattr(args, "cache_dir", None),
        limit=num_samples,
        start_index=dataset_partition.start_index if dataset_partition is not None else 0,
        validation_holdout_count=int(getattr(args, "validation_holdout_count", MONUSEG_CONTRACT_VAL_HOLDOUT_COUNT)),
        validation_subset_seed=int(getattr(args, "validation_subset_seed", MONUSEG_CONTRACT_VAL_SUBSET_SEED)),
    )

    output_dir = prepare_output_dir(args.output_dir, run_kind="eval", split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_run_config(
        args,
        run_kind="eval",
        dataset_partition=dataset_partition,
        selected_sample_count=len(eval_samples),
    )
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_experiment_terms_markdown(
            args,
            run_kind="eval",
            train_sample_count=None,
            eval_sample_count=len(eval_samples),
            checkpoint_path=checkpoint_path,
        ),
    )

    summary = evaluate_monuseg_autosam(
        prompt_generator=prompt_generator,
        checkpoint_path=checkpoint_path,
        samples=eval_samples,
        args=args,
        run_kind="eval",
        checkpoint_role=str(checkpoint.get("checkpoint_role", "loaded")),
        trainable_parameter_count=int(checkpoint["trainable_parameter_count"]),
        output_dir=output_dir,
        save_artifacts=True,
    )
    return summary
