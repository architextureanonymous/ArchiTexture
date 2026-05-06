"""STLD AutoSAM training and evaluation with the repo-native results contract.

This module keeps the upstream AutoSAM architecture intact:

- upstream ``ModelEmb`` prompt generator from ``third_party/AutoSAM``
- official SAM ViT checkpoint loaded through the vendored SAM build
- frozen SAM parameters throughout training
- BCE + Dice on normalized low-resolution SAM masks as in the upstream 2D path

What changes relative to the upstream repo is the orchestration only:

- the dataset path uses the repository's prepared split-aware STLD root
- train subsets are controlled by explicit few-shot manifests
- evaluation uses the repo-native direct-foreground and auxiliary
  partition-invariant metrics at original image resolution
- outputs are written in the same directory contract as the existing sweeps
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np
from PIL import Image

from rwtd_sam3.data.architexture_binary import (
    ArchiTextureBinaryOverview,
    ArchiTextureBinarySample,
    iter_architexture_binary_samples,
    load_architexture_binary_overview,
)
from rwtd_sam3.eval.experiment_terms import ExperimentTermsSection, render_experiment_terms_markdown
from rwtd_sam3.eval.few_shot_subsets import (
    FewShotSubsetManifest,
    load_few_shot_subset_manifest,
    select_items_by_few_shot_manifest,
)
from rwtd_sam3.eval.metrics import (
    CANONICAL_EVALUATION_CONTRACT,
    build_canonical_evaluation_fields,
    compute_binary_metrics,
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

SUPPORTED_ARCHITEXTURE_AUTOSAM_ROUTES = ("stld", "caid")
ARCHITEXTURE_AUTOSAM_SCALAR_FIELDS = (
    "direct_foreground_iou",
    "direct_foreground_dice",
    "direct_foreground_precision",
    "direct_foreground_recall",
    "eval_miou",
    "eval_ari",
    "predicted_positive_fraction",
    "target_positive_fraction",
)
ARCHITEXTURE_AUTOSAM_AUGMENTATION_POLICIES = ("none", "autosam_dense_v1")
STLD_AUTOSAM_AUGMENTATION_POLICIES = ARCHITEXTURE_AUTOSAM_AUGMENTATION_POLICIES


def resolve_architexture_autosam_route(args) -> str:
    route = str(getattr(args, "route", "stld")).strip().lower()
    if route not in SUPPORTED_ARCHITEXTURE_AUTOSAM_ROUTES:
        raise AutoSamStldRuntimeError(
            f"Unsupported ArchiTexture AutoSAM route '{route}'. Expected one of {SUPPORTED_ARCHITEXTURE_AUTOSAM_ROUTES}.",
        )
    return route


def architexture_autosam_dataset_id(route: str) -> str:
    return f"architexture:{route}"


def architexture_autosam_output_root(route: str) -> Path:
    return Path("outputs") / "architexture_binary" / "autosam" / route


def architexture_autosam_display_name(route: str) -> str:
    return str(route).upper()


@dataclass(frozen=True)
class AutoSamStldBatchItem:
    """One transformed STLD sample prepared for AutoSAM."""

    sample: ArchiTextureBinarySample
    processed_image: Any
    processed_mask: Any
    original_size_hw: tuple[int, int]
    sam_input_size_hw: tuple[int, int]


@dataclass(frozen=True)
class AutoSamStldTrainingArtifacts:
    """Persistent outputs from one STLD AutoSAM training run."""

    checkpoint_path: Path
    trainable_parameter_count: int
    training_history_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class AutoSamStldForwardOutput:
    """Forward outputs retained for metrics and artifact writing."""

    normalized_low_res_masks: Any
    postprocessed_masks: Any
    dense_embeddings: Any


@dataclass(frozen=True)
class AutoSamStldEvalBundle:
    """Per-sample eval payload stored in the standard run directory."""

    row: dict[str, Any]
    metric_summary: str
    foreground_prediction: np.ndarray
    background_prediction: np.ndarray
    foreground_probability: np.ndarray


class AutoSamStldRuntimeError(RuntimeError):
    """Raised for explicit STLD AutoSAM contract violations."""


def iter_architexture_autosam_samples(
    *,
    route: str,
    split: str,
    benchmark_root: str,
    limit: int | None = None,
    start_index: int = 0,
):
    yield from iter_architexture_binary_samples(
        route=route,
        benchmark_root=benchmark_root,
        split=split,
        require_official_split=True,
        limit=limit,
        start_index=start_index,
    )


def load_architexture_autosam_overview(*, route: str, split: str, benchmark_root: str) -> ArchiTextureBinaryOverview:
    return load_architexture_binary_overview(
        route=route,
        benchmark_root=benchmark_root,
        split=split,
        require_official_split=True,
    )


def resolve_train_subset_manifest(args) -> FewShotSubsetManifest | None:
    manifest_path = getattr(args, "train_subset_manifest", None)
    if manifest_path in (None, ""):
        return None
    return load_few_shot_subset_manifest(
        manifest_path,
        expected_dataset_id=architexture_autosam_dataset_id(resolve_architexture_autosam_route(args)),
        expected_source_split=str(getattr(args, "train_split", "train")),
    )


def resolve_train_selection_policy(args) -> str:
    route = resolve_architexture_autosam_route(args)
    if resolve_train_subset_manifest(args) is None:
        return f"{route}_prepared_root"
    return f"few_shot_subset_manifest_over_{route}_prepared_root"


def resolve_autosam_augmentation_policy(args) -> str:
    policy = str(getattr(args, "train_augmentation_policy", "autosam_dense_v1"))
    if policy not in ARCHITEXTURE_AUTOSAM_AUGMENTATION_POLICIES:
        raise AutoSamStldRuntimeError(
            f"Unsupported ArchiTexture AutoSAM augmentation policy '{policy}'.",
        )
    return policy


def describe_autosam_augmentation_policy(policy: str) -> str:
    if policy == "none":
        return "none"
    return (
        "autosam_dense_v1:"
        "color_jitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)"
        "+hflip_p0.5"
        "+affine(angle_uniform[-20,20],scale_uniform[0.75,1.25],translate=0,shear=0)"
        "+upstream_paired_image_mask_ops"
    )


def build_autosam_train_transform(*, policy: str):
    transforms = load_upstream_autosam_transforms_shir()
    if policy == "none":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
    if policy != "autosam_dense_v1":
        raise AutoSamStldRuntimeError(f"Unsupported STLD AutoSAM augmentation policy '{policy}'.")
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, scale=(0.75, 1.25)),
            transforms.ToTensor(),
        ]
    )


def build_autosam_eval_transform():
    transforms = load_upstream_autosam_transforms_shir()
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )


class AutoSamStldDataset:
    """Thin STLD dataset wrapper matching the upstream AutoSAM tensor contract."""

    def __init__(
        self,
        *,
        samples: Sequence[ArchiTextureBinarySample],
        train: bool,
        repeat_factor: int,
        train_augmentation_policy: str,
        sam_transform,
    ) -> None:
        if not samples:
            raise AutoSamStldRuntimeError("AutoSAM STLD dataset wrapper received zero samples.")
        self.samples = tuple(samples)
        self.train = bool(train)
        self.repeat_factor = int(repeat_factor if train else 1)
        if self.repeat_factor < 1:
            raise AutoSamStldRuntimeError(f"repeat_factor must be >= 1, got {repeat_factor}.")
        self.sam_transform = sam_transform
        self.transform = (
            build_autosam_train_transform(policy=train_augmentation_policy)
            if self.train
            else build_autosam_eval_transform()
        )

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, index: int) -> AutoSamStldBatchItem:
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
        return AutoSamStldBatchItem(
            sample=sample,
            processed_image=processed_image,
            processed_mask=processed_mask,
            original_size_hw=original_size,
            sam_input_size_hw=sam_input_size,
        )


def collate_autosam_stld_batch(batch: Sequence[AutoSamStldBatchItem]) -> dict[str, Any]:
    import torch

    if not batch:
        raise AutoSamStldRuntimeError("AutoSAM STLD collate received an empty batch.")
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
        raise AutoSamStldRuntimeError("CUDA was requested for STLD AutoSAM but torch.cuda.is_available() is False.")
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
        raise AutoSamStldRuntimeError(f"Expected a 4D tensor for norm_batch, got shape {tuple(x.shape)}.")
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
) -> AutoSamStldForwardOutput:
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
    return AutoSamStldForwardOutput(
        normalized_low_res_masks=normalized_low_res_masks,
        postprocessed_masks=postprocessed_masks,
        dense_embeddings=dense_embeddings,
    )


def resolve_autosam_variant_name(args) -> str:
    return (
        f"autosam_{str(args.sam_model_type)}_order{int(args.autosam_backbone_order)}"
        f"_idim{int(args.prompt_image_size)}"
    )


def prepare_stld_autosam_output_dir(output_dir: str | None, *, route: str, run_kind: str, split: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    suffix = f"{run_kind}_{'autosam'}"
    if split is not None:
        suffix = f"{suffix}_{split}"
    return architexture_autosam_output_root(route) / suffix


def build_stld_autosam_train_dataloader(
    *,
    samples: Sequence[ArchiTextureBinarySample],
    batch_size: int,
    num_workers: int,
    repeat_factor: int,
    train_augmentation_policy: str,
    sam_transform,
    seed: int,
):
    import torch

    dataset = AutoSamStldDataset(
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
        collate_fn=collate_autosam_stld_batch,
        generator=generator,
    )


def build_stld_autosam_eval_dataloader(
    *,
    samples: Sequence[ArchiTextureBinarySample],
    num_workers: int,
    sam_transform,
):
    import torch

    dataset = AutoSamStldDataset(
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
        collate_fn=collate_autosam_stld_batch,
    )


def load_stld_split_samples(
    *,
    route: str,
    benchmark_root: str,
    split: str,
    limit: int | None = None,
    start_index: int = 0,
    subset_manifest: FewShotSubsetManifest | None = None,
) -> tuple[ArchiTextureBinarySample, ...]:
    samples = tuple(
        iter_architexture_autosam_samples(
            route=route,
            split=split,
            benchmark_root=benchmark_root,
            limit=limit,
            start_index=start_index,
        )
    )
    samples = select_items_by_few_shot_manifest(
        samples,
        subset_manifest,
        item_id_getter=lambda sample: sample.crop_name,
    )
    if not samples:
        raise AutoSamStldRuntimeError(
            f"No {architexture_autosam_display_name(route)} samples were loaded for split '{split}' with limit={limit} and start_index={start_index}."
        )
    return samples


def build_stld_autosam_checkpoint_payload(*, prompt_generator, args, training_history_rows, trainable_parameter_count):
    route = resolve_architexture_autosam_route(args)
    return {
        "variant": resolve_autosam_variant_name(args),
        "dataset_id": architexture_autosam_dataset_id(route),
        "route": route,
        "prompt_generator_state_dict": prompt_generator.state_dict(),
        "trainable_parameter_count": int(trainable_parameter_count),
        "model_family": "autosam_prompt_generator",
        "head_kind": "upstream_model_emb_dense_prompt",
        "sam_model_type": str(args.sam_model_type),
        "sam_checkpoint_path": str(args.sam_checkpoint_path),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "foreground_threshold": float(args.foreground_threshold),
        "train_repeat_factor": int(args.train_repeat_factor),
        "train_augmentation_policy": resolve_autosam_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_autosam_augmentation_policy(args)),
        "training_history_rows": training_history_rows,
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "upstream_alignment_notes": (
            "Prompt generator and SAM forward mirror the upstream 2D AutoSAM path "
            "from train_3d/inference.py. Dataset loading, few-shot manifests, "
            "and evaluation artifacts are repo-native."
        ),
    }


def train_stld_autosam(
    *,
    train_samples: Sequence[ArchiTextureBinarySample],
    output_dir: Path,
    args,
) -> AutoSamStldTrainingArtifacts:
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

    train_loader = build_stld_autosam_train_dataloader(
        samples=train_samples,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        repeat_factor=int(args.train_repeat_factor),
        train_augmentation_policy=resolve_autosam_augmentation_policy(args),
        sam_transform=sam_transform,
        seed=int(getattr(args, "seed", 0)),
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in prompt_generator.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.BCELoss()
    trainable_parameter_count = count_trainable_parameters(prompt_generator)
    accumulation_steps = 4
    training_history_rows: list[dict[str, Any]] = []

    LOGGER.info(
        "STLD AutoSAM training start | variant=%s lr=%.6f wd=%.6f epochs=%d batch=%d prompt_idim=%d repeat=%d aug=%s trainable_params=%d",
        resolve_autosam_variant_name(args),
        float(args.learning_rate),
        float(args.weight_decay),
        int(args.num_epochs),
        int(args.batch_size),
        int(args.prompt_image_size),
        int(args.train_repeat_factor),
        resolve_autosam_augmentation_policy(args),
        trainable_parameter_count,
    )

    for epoch in range(int(args.num_epochs)):
        prompt_generator.train()
        epoch_losses: list[float] = []
        epoch_bce: list[float] = []
        epoch_dice: list[float] = []
        epoch_pred_fraction: list[float] = []
        optimizer.zero_grad(set_to_none=True)
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
            loss.backward()
            steps_since_update += 1
            should_step = ((step_index + 1) % accumulation_steps) == 0
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_bce.append(float(bce.detach().cpu().item()))
            epoch_dice.append(float(dice.detach().cpu().item()))
            epoch_pred_fraction.append(float(output.normalized_low_res_masks.detach().mean().cpu().item()))
        if steps_since_update > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        history_row = {
            "epoch": epoch + 1,
            "mean_train_loss": float(mean(epoch_losses)) if epoch_losses else None,
            "mean_bce_loss": float(mean(epoch_bce)) if epoch_bce else None,
            "mean_dice_loss": float(mean(epoch_dice)) if epoch_dice else None,
            "mean_predicted_positive_fraction": float(mean(epoch_pred_fraction)) if epoch_pred_fraction else None,
            "num_batches": int(len(epoch_losses)),
        }
        training_history_rows.append(history_row)
        LOGGER.info(
            "STLD AutoSAM epoch=%d/%d | mean_loss=%.6f mean_bce=%.6f mean_dice=%.6f mean_pred=%.4f batches=%d",
            epoch + 1,
            int(args.num_epochs),
            float(history_row["mean_train_loss"] or 0.0),
            float(history_row["mean_bce_loss"] or 0.0),
            float(history_row["mean_dice_loss"] or 0.0),
            float(history_row["mean_predicted_positive_fraction"] or 0.0),
            int(history_row["num_batches"]),
        )

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_payload = build_stld_autosam_checkpoint_payload(
        prompt_generator=prompt_generator,
        args=args,
        training_history_rows=training_history_rows,
        trainable_parameter_count=trainable_parameter_count,
    )
    torch.save(checkpoint_payload, checkpoint_path)
    write_csv(output_dir / "train_history.csv", training_history_rows)
    return AutoSamStldTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        trainable_parameter_count=trainable_parameter_count,
        training_history_rows=training_history_rows,
    )


def build_stld_autosam_eval_row(
    *,
    sample: ArchiTextureBinarySample,
    variant: str,
    checkpoint_path: Path,
    trainable_parameter_count: int,
    dense_embedding_shape: Sequence[int],
    low_res_shape: Sequence[int],
    direct_foreground_metrics,
    foreground_threshold: float,
    assignment,
) -> dict[str, Any]:
    route = str(sample.route)
    total_pixels = int(sample.height * sample.width)
    row = {
        "variant": variant,
        "dataset_id": architexture_autosam_dataset_id(route),
        "route": route,
        "split": sample.split,
        "sample_index": int(sample.index),
        "crop_name": sample.crop_name,
        "grade_label": getattr(sample, "grade_label", None),
        "checkpoint_path": str(checkpoint_path),
        "foreground_evaluation_view": "direct_foreground",
        "foreground_assignment_used": "autosam_foreground->foreground,complement->background",
        "foreground_threshold": float(foreground_threshold),
        "trainable_parameter_count": int(trainable_parameter_count),
        "dense_embedding_shape_json": json.dumps([int(value) for value in dense_embedding_shape]),
        "low_res_shape_json": json.dumps([int(value) for value in low_res_shape]),
        "image_height": int(sample.height),
        "image_width": int(sample.width),
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


def save_stld_autosam_result_bundle(
    *,
    output_dir: Path,
    sample: ArchiTextureBinarySample,
    result: AutoSamStldEvalBundle,
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
            protocol=f"{sample.route}_autosam:{variant}",
            metric_summary=result.metric_summary,
        )


def build_stld_autosam_summary(
    *,
    rows: list[dict[str, Any]],
    split: str,
    benchmark_root: str,
    args,
    checkpoint_path: Path,
    run_kind: str,
    trainable_parameter_count: int,
) -> dict[str, Any]:
    route = resolve_architexture_autosam_route(args)
    mean_metrics = {name: float(mean(float(row[name]) for row in rows)) for name in ARCHITEXTURE_AUTOSAM_SCALAR_FIELDS}
    median_metrics = {name: float(median(float(row[name]) for row in rows)) for name in ARCHITEXTURE_AUTOSAM_SCALAR_FIELDS}
    return {
        "run_kind": run_kind,
        "variant": resolve_autosam_variant_name(args),
        "variant_summary": (
            "Upstream AutoSAM prompt-generator (ModelEmb HarDNet order "
            f"{int(args.autosam_backbone_order)}) over frozen SAM {args.sam_model_type}, "
            f"prompt image size {int(args.prompt_image_size)}."
        ),
        "dataset_id": architexture_autosam_dataset_id(route),
        "route": route,
        "split": split,
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "benchmark_root": str(benchmark_root),
        "train_selection_policy": resolve_train_selection_policy(args),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "device": args.device,
        "checkpoint_path": str(checkpoint_path),
        "num_evaluated_samples": len(rows),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "foreground_evaluation_view": "direct_foreground",
        "primary_metric_name": "direct_foreground_iou",
        "secondary_metric_name": "direct_foreground_dice",
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
        "loss_name": "BCELoss + Dice on normalized low-res masks",
        "foreground_threshold": float(args.foreground_threshold),
        "train_augmentation_policy": resolve_autosam_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_autosam_augmentation_policy(args)),
        "preprocessing_summary": (
            "native STLD image/mask -> upstream paired transform -> "
            "ResizeLongestSide(1024) -> upstream preprocess semantics -> "
            f"prompt-generator bilinear downsample to {int(args.prompt_image_size)}x{int(args.prompt_image_size)}"
        ),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "upstream_alignment_notes": (
            "Training path mirrors the upstream 2D AutoSAM code path conceptually "
            "while replacing its dataset wrappers and Google-Drive outputs with "
            "repo-native STLD loaders, manifests, metrics, and artifact writing."
        ),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def flatten_stld_autosam_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": summary["variant"],
        "split": summary["split"],
        "num_evaluated_samples": summary["num_evaluated_samples"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "direct_foreground_iou": summary["mean_metrics"]["direct_foreground_iou"],
        "direct_foreground_dice": summary["mean_metrics"]["direct_foreground_dice"],
        "eval_miou": summary["mean_metrics"]["eval_miou"],
        "eval_ari": summary["mean_metrics"]["eval_ari"],
        "checkpoint_path": summary["checkpoint_path"],
    }


def build_stld_autosam_summary_markdown(summary: dict[str, Any]) -> str:
    route_name = architexture_autosam_display_name(str(summary["route"]))
    lines = [
        f"# {route_name} AutoSAM Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Split: `{summary['split']}`",
        f"- Train selection policy: `{summary.get('train_selection_policy')}`",
        f"- Train subset manifest: `{summary.get('train_subset_manifest_output_path') or summary.get('train_subset_manifest_path')}`",
        f"- Trainable params: `{summary['trainable_parameter_count']}`",
        f"- SAM model type: `{summary['sam_model_type']}`",
        f"- Prompt image size: `{summary['prompt_image_size']}`",
        f"- HarDNet order: `{summary['autosam_backbone_order']}`",
        f"- Depth-wise backbone: `{summary['autosam_depth_wise']}`",
        f"- Train repeat factor: `{summary['train_repeat_factor']}`",
        f"- Train augmentation: `{summary['train_augmentation_summary']}`",
        f"- Loss: `{summary['loss_name']}`",
        f"- Mean direct foreground IoU: `{summary['mean_metrics']['direct_foreground_iou']:.6f}`",
        f"- Mean Dice: `{summary['mean_metrics']['direct_foreground_dice']:.6f}`",
        f"- Mean auxiliary partition mIoU: `{summary['mean_metrics']['eval_miou']:.6f}`",
        f"- Mean auxiliary partition ARI: `{summary['mean_metrics']['eval_ari']:.6f}`",
        f"- Checkpoint: `{summary['checkpoint_path']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def rewrite_stld_autosam_summary(*, output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [flatten_stld_autosam_summary(summary)])
    write_text(output_dir / "summary.md", build_stld_autosam_summary_markdown(summary))


def build_stld_autosam_run_config(
    args,
    *,
    run_kind: str,
    dataset_partition=None,
    selected_sample_count: int | None = None,
) -> dict[str, Any]:
    route = resolve_architexture_autosam_route(args)
    config = {
        "command": getattr(args, "command", None),
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(str(value)) for value in sys.argv),
        "run_kind": run_kind,
        "dataset_id": architexture_autosam_dataset_id(route),
        "route": route,
        "benchmark_root": getattr(args, "benchmark_root", None),
        "train_selection_policy": resolve_train_selection_policy(args),
        "split": getattr(args, "split", None),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "variant": resolve_autosam_variant_name(args),
        "device": args.device,
        "sam_model_type": str(args.sam_model_type),
        "sam_checkpoint_path": str(args.sam_checkpoint_path),
        "prompt_image_size": int(args.prompt_image_size),
        "autosam_backbone_order": int(args.autosam_backbone_order),
        "autosam_depth_wise": bool(args.autosam_depth_wise),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "foreground_threshold": float(args.foreground_threshold),
        "train_repeat_factor": int(args.train_repeat_factor),
        "train_augmentation_policy": resolve_autosam_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_autosam_augmentation_policy(args)),
        "num_workers": int(args.num_workers),
        "save_visuals": bool(args.save_visuals),
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "preprocessing_summary": (
            "native STLD -> upstream paired transform -> ResizeLongestSide(1024) -> "
            f"upstream preprocess semantics -> bilinear downsample to prompt_image_size={int(args.prompt_image_size)}"
        ),
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_stld_autosam_experiment_terms_markdown(args, *, run_kind: str, train_sample_count: int | None, eval_sample_count: int | None, checkpoint_path: Path | None) -> str:
    route = resolve_architexture_autosam_route(args)
    route_name = architexture_autosam_display_name(route)
    output_root = architexture_autosam_output_root(route)
    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            bullets=(
                f"Command family: `{getattr(args, 'command', 'stld-autosam')}`.",
                f"Run kind: `{run_kind}`.",
                f"Dataset: `{architexture_autosam_dataset_id(route)}` with route `{route}` from benchmark root `{getattr(args, 'benchmark_root', None)}`.",
                f"Upstream AutoSAM checkout: `{AUTOSAM_REPO_ROOT}`.",
                f"SAM checkpoint: `{getattr(args, 'sam_checkpoint_path', None)}` with model type `{getattr(args, 'sam_model_type', None)}`.",
                "SAM parameters stay frozen. Only the upstream AutoSAM prompt generator is optimized.",
            ),
        ),
        ExperimentTermsSection(
            title="Data And Split Separation",
            bullets=(
                f"Training split: `{getattr(args, 'train_split', None)}` with limit `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Evaluation split: `{getattr(args, 'eval_split', getattr(args, 'split', None))}` with limit `{getattr(args, 'eval_limit', getattr(args, 'limit', None))}`; loaded `{eval_sample_count}` sample(s).",
                (
                    f"Train subset manifest: `{getattr(args, '_resolved_train_subset_manifest_output_path', getattr(args, 'train_subset_manifest', None))}` selecting `{getattr(args, '_resolved_train_subset_manifest', {}).get('shot_count')}` image(s) with subset seed `{getattr(args, '_resolved_train_subset_manifest', {}).get('subset_seed')}`."
                    if getattr(args, "_resolved_train_subset_manifest", None) is not None
                    else "No train subset manifest is applied. The full prepared STLD train split is used."
                ),
                f"Evaluation uses the repo-native {route_name} direct-foreground IoU/Dice and auxiliary partition-invariant mIoU/ARI at original image resolution.",
            ),
        ),
        ExperimentTermsSection(
            title="AutoSAM Training Path",
            bullets=(
                f"Prompt generator: upstream `ModelEmb` with HarDNet order `{int(args.autosam_backbone_order)}` and depth_wise `{bool(args.autosam_depth_wise)}`.",
                f"Prompt image size: `{int(args.prompt_image_size)}`.",
                f"Optimizer: `Adam(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
                f"Batch size: `{int(args.batch_size)}` with train repeat factor `{int(args.train_repeat_factor)}`.",
                f"Train augmentation: `{describe_autosam_augmentation_policy(resolve_autosam_augmentation_policy(args))}`.",
                "Loss: upstream-style `BCELoss + Dice` on normalized low-resolution SAM masks.",
            ),
        ),
        ExperimentTermsSection(
            title="Outputs",
            bullets=(
                f"Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `{output_root}`.",
                "Training runs additionally write `checkpoint.pt` and `train_history.csv`.",
                (
                    f"Checkpoint consumed by this run: `{checkpoint_path}`."
                    if checkpoint_path is not None
                    else "This run trains a fresh AutoSAM prompt-generator checkpoint and then evaluates it."
                ),
            ),
        ),
    ]
    return render_experiment_terms_markdown(
        title=f"{route_name} AutoSAM Experiment Terms",
        summary_lines=(
            f"This file records the exact {route_name} AutoSAM run that produced this directory.",
            "The upstream model path is preserved while the dataset loader, metrics, and outputs are repo-native.",
        ),
        sections=sections,
        related_paths=(
            str(AUTOSAM_REPO_ROOT / "models" / "model_single.py"),
            str(AUTOSAM_REPO_ROOT / "train_3d"),
            str(Path("src/rwtd_sam3/eval/stld_autosam.py")),
        ),
    )


def evaluate_stld_autosam(
    *,
    prompt_generator,
    checkpoint_path: Path,
    samples: Sequence[ArchiTextureBinarySample],
    args,
    run_kind: str,
    trainable_parameter_count: int,
    output_dir: Path | None,
    save_artifacts: bool,
) -> dict[str, Any]:
    import torch

    if save_artifacts and output_dir is None:
        raise AutoSamStldRuntimeError("save_artifacts=True requires a concrete output_dir.")
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

    eval_loader = build_stld_autosam_eval_dataloader(
        samples=samples,
        num_workers=int(args.num_workers),
        sam_transform=sam_transform,
    )
    rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []

    for batch in eval_loader:
        processed_images = batch["images"].to(device)
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
        row = build_stld_autosam_eval_row(
            sample=source_sample,
            variant=resolve_autosam_variant_name(args),
            checkpoint_path=checkpoint_path,
            trainable_parameter_count=trainable_parameter_count,
            dense_embedding_shape=tuple(int(value) for value in output.dense_embeddings.shape),
            low_res_shape=tuple(int(value) for value in output.normalized_low_res_masks.shape),
            direct_foreground_metrics=direct_foreground_metrics,
            foreground_threshold=float(args.foreground_threshold),
            assignment=assignment,
        )
        rows.append(row)
        metric_summary = (
            f"Fg IoU={row['direct_foreground_iou']:.3f} "
            f"Dice={row['direct_foreground_dice']:.3f} "
            f"Aux mIoU={row['eval_miou']:.3f} "
            f"Aux ARI={row['eval_ari']:.3f}"
        )
        if save_artifacts and output_dir is not None:
            result = AutoSamStldEvalBundle(
                row=row,
                metric_summary=metric_summary,
                foreground_prediction=foreground_prediction,
                background_prediction=background_prediction,
                foreground_probability=foreground_probability,
            )
            save_stld_autosam_result_bundle(
                output_dir=output_dir,
                sample=source_sample,
                result=result,
                variant=resolve_autosam_variant_name(args),
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
                        protocol=f"{source_sample.route}_autosam:{resolve_autosam_variant_name(args)}",
                        visual_path=Path("visuals") / f"{source_sample.index}.png",
                    )
                )

    if not rows:
        raise AutoSamStldRuntimeError("STLD AutoSAM evaluation produced no sample rows.")

    summary = build_stld_autosam_summary(
        rows=rows,
        split=str(getattr(args, "eval_split", getattr(args, "split", "test"))),
        benchmark_root=str(args.benchmark_root),
        args=args,
        checkpoint_path=checkpoint_path,
        run_kind=run_kind,
        trainable_parameter_count=trainable_parameter_count,
    )
    if save_artifacts and output_dir is not None:
        write_csv(output_dir / "per_sample_metrics.csv", rows)
        write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
        write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
        rewrite_stld_autosam_summary(output_dir=output_dir, summary=summary)
    return summary


def build_prompt_generator_from_checkpoint(checkpoint: dict[str, Any]):
    prompt_generator = build_upstream_autosam_prompt_generator(
        order=int(checkpoint["autosam_backbone_order"]),
        depth_wise=bool(checkpoint["autosam_depth_wise"]),
    )
    prompt_generator.load_state_dict(checkpoint["prompt_generator_state_dict"])
    return prompt_generator


def run_stld_autosam_train(args) -> dict[str, Any]:
    import torch

    route = resolve_architexture_autosam_route(args)
    if getattr(args, "train_limit", None) is not None and getattr(args, "train_subset_manifest", None) not in (None, ""):
        raise AutoSamStldRuntimeError(
            f"train_limit and train_subset_manifest are mutually exclusive for {architexture_autosam_display_name(route)} AutoSAM training."
        )
    resolved_checkpoint_path = Path(str(args.sam_checkpoint_path))
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint path does not exist: {resolved_checkpoint_path}")

    train_subset_manifest = resolve_train_subset_manifest(args)
    setattr(
        args,
        "_resolved_train_subset_manifest",
        train_subset_manifest.to_json_dict() if train_subset_manifest is not None else None,
    )
    output_dir = prepare_stld_autosam_output_dir(args.output_dir, route=route, run_kind="train", split=args.eval_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_subset_manifest is not None:
        local_manifest_path = output_dir / "train_subset_manifest.json"
        write_json(local_manifest_path, train_subset_manifest.to_json_dict())
        setattr(args, "_resolved_train_subset_manifest_output_path", str(local_manifest_path))
    else:
        setattr(args, "_resolved_train_subset_manifest_output_path", None)

    train_samples = load_stld_split_samples(
        route=route,
        benchmark_root=str(args.benchmark_root),
        split=str(args.train_split),
        limit=getattr(args, "train_limit", None),
        subset_manifest=train_subset_manifest,
    )
    eval_samples = load_stld_split_samples(
        route=route,
        benchmark_root=str(args.benchmark_root),
        split=str(args.eval_split),
        limit=getattr(args, "eval_limit", None),
    )

    config = build_stld_autosam_run_config(args, run_kind="train")
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_stld_autosam_experiment_terms_markdown(
            args,
            run_kind="train",
            train_sample_count=len(train_samples),
            eval_sample_count=len(eval_samples),
            checkpoint_path=None,
        ),
    )

    training_artifacts = train_stld_autosam(
        train_samples=train_samples,
        output_dir=output_dir,
        args=args,
    )
    checkpoint = torch.load(training_artifacts.checkpoint_path, map_location="cpu")
    prompt_generator = build_prompt_generator_from_checkpoint(checkpoint)
    train_summary = evaluate_stld_autosam(
        prompt_generator=build_prompt_generator_from_checkpoint(copy.deepcopy(checkpoint)),
        checkpoint_path=training_artifacts.checkpoint_path,
        samples=train_samples,
        args=args,
        run_kind="train_internal_eval",
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=None,
        save_artifacts=False,
    )
    write_json(output_dir / "train_set_summary.json", train_summary)
    summary = evaluate_stld_autosam(
        prompt_generator=prompt_generator,
        checkpoint_path=training_artifacts.checkpoint_path,
        samples=eval_samples,
        args=args,
        run_kind="train",
        trainable_parameter_count=training_artifacts.trainable_parameter_count,
        output_dir=output_dir,
        save_artifacts=True,
    )
    summary["train_set_mean_metrics"] = dict(train_summary["mean_metrics"])
    summary["validation_set_mean_metrics"] = None
    summary["train_set_summary_path"] = str(output_dir / "train_set_summary.json")
    rewrite_stld_autosam_summary(output_dir=output_dir, summary=summary)
    return summary


def run_stld_autosam_eval(args) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{architexture_autosam_display_name(resolve_architexture_autosam_route(args))} AutoSAM checkpoint path does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    setattr(args, "sam_model_type", str(checkpoint["sam_model_type"]))
    setattr(args, "sam_checkpoint_path", str(checkpoint["sam_checkpoint_path"]))
    setattr(args, "prompt_image_size", int(checkpoint["prompt_image_size"]))
    setattr(args, "autosam_backbone_order", int(checkpoint["autosam_backbone_order"]))
    setattr(args, "autosam_depth_wise", bool(checkpoint["autosam_depth_wise"]))
    setattr(args, "batch_size", int(checkpoint["batch_size"]))
    setattr(args, "learning_rate", float(checkpoint["learning_rate"]))
    setattr(args, "weight_decay", float(checkpoint["weight_decay"]))
    setattr(args, "num_epochs", int(checkpoint["num_epochs"]))
    setattr(args, "train_repeat_factor", int(checkpoint.get("train_repeat_factor", 3)))
    setattr(args, "train_augmentation_policy", str(checkpoint.get("train_augmentation_policy", "autosam_dense_v1")))
    prompt_generator = build_prompt_generator_from_checkpoint(checkpoint)

    route = resolve_architexture_autosam_route(args)
    overview = load_architexture_autosam_overview(route=route, split=args.split, benchmark_root=str(args.benchmark_root))
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_samples = resolve_eval_sample_count(getattr(args, "limit", None), overview.num_examples, dataset_partition)
    eval_samples = load_stld_split_samples(
        route=route,
        benchmark_root=str(args.benchmark_root),
        split=str(args.split),
        limit=num_samples,
        start_index=dataset_partition.start_index if dataset_partition is not None else 0,
    )

    output_dir = prepare_stld_autosam_output_dir(args.output_dir, route=route, run_kind="eval", split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_stld_autosam_run_config(
        args,
        run_kind="eval",
        dataset_partition=dataset_partition,
        selected_sample_count=num_samples,
    )
    config["checkpoint_path"] = str(checkpoint_path)
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_stld_autosam_experiment_terms_markdown(
            args,
            run_kind="eval",
            train_sample_count=None,
            eval_sample_count=len(eval_samples),
            checkpoint_path=checkpoint_path,
        ),
    )
    summary = evaluate_stld_autosam(
        prompt_generator=prompt_generator,
        checkpoint_path=checkpoint_path,
        samples=eval_samples,
        args=args,
        run_kind="eval",
        trainable_parameter_count=int(checkpoint["trainable_parameter_count"]),
        output_dir=output_dir,
        save_artifacts=True,
    )
    append_dataset_partition_fields(summary, dataset_partition, selected_sample_count=num_samples)
    rewrite_stld_autosam_summary(output_dir=output_dir, summary=summary)
    return summary
