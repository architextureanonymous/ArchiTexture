"""RWTD AutoSAM training and evaluation with fixed texture_a foreground targets.

This adapter keeps the upstream AutoSAM prompt-generator and SAM path intact,
but adapts the repository's curated RWTD split into a binary-foreground view:

- training target is always ``texture_a_mask``
- direct foreground metrics are computed against ``texture_a_mask``
- auxiliary partition metrics use Hungarian-equivalent 2-way assignment over
  ``{pred_fg, pred_bg}`` and ``{texture_a_mask, texture_b_mask}``
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np
from rwtd_sam3.data.rwtd import (
    CURATED_RWTD_LOCAL_ALIAS,
    DatasetSplitOverview,
    RwtdSample,
    iter_rwtd_samples,
    load_split_overview,
)
from rwtd_sam3.eval.experiment_terms import ExperimentTermsSection, render_experiment_terms_markdown
from rwtd_sam3.eval.few_shot_subsets import FewShotSubsetManifest, load_few_shot_subset_manifest, select_items_by_few_shot_manifest
from rwtd_sam3.eval.metrics import CANONICAL_EVALUATION_CONTRACT, build_canonical_evaluation_fields, compute_binary_metrics
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
from rwtd_sam3.eval.stld_autosam import (
    AutoSamStldRuntimeError,
    STLD_AUTOSAM_AUGMENTATION_POLICIES,
    build_autosam_eval_transform,
    build_autosam_train_transform,
    build_prompt_generator_from_checkpoint,
    count_trainable_parameters,
    describe_autosam_augmentation_policy,
    dice_loss,
    forward_autosam_batch,
    resolve_torch_device,
    set_global_training_seed,
)
from rwtd_sam3.models.autosam_upstream import (
    AUTOSAM_REPO_ROOT,
    build_upstream_autosam_prompt_generator,
    build_upstream_autosam_sam,
    load_upstream_autosam_resize_longest_side_class,
)
from rwtd_sam3.utils.visualization import save_prediction_panel


LOGGER = logging.getLogger(__name__)

RWTD_AUTOSAM_DATASET_ID = "rwtd"
RWTD_AUTOSAM_OUTPUT_ROOT = Path("outputs") / "rwtd" / "autosam"
RWTD_AUTOSAM_SCALAR_FIELDS = (
    "direct_foreground_iou",
    "direct_foreground_dice",
    "direct_foreground_precision",
    "direct_foreground_recall",
    "eval_miou",
    "eval_ari",
    "predicted_positive_fraction",
    "target_positive_fraction",
)


@dataclass(frozen=True)
class AutoSamRwtdBatchItem:
    sample: RwtdSample
    processed_image: Any
    processed_mask: Any
    original_size_hw: tuple[int, int]
    sam_input_size_hw: tuple[int, int]


@dataclass(frozen=True)
class AutoSamRwtdTrainingArtifacts:
    checkpoint_path: Path
    trainable_parameter_count: int
    training_history_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class AutoSamRwtdEvalBundle:
    row: dict[str, Any]
    metric_summary: str
    foreground_prediction: np.ndarray
    background_prediction: np.ndarray
    foreground_probability: np.ndarray


def resolve_rwtd_augmentation_policy(args) -> str:
    policy = str(getattr(args, "train_augmentation_policy", "autosam_dense_v1"))
    if policy not in STLD_AUTOSAM_AUGMENTATION_POLICIES:
        raise AutoSamStldRuntimeError(f"Unsupported RWTD AutoSAM augmentation policy '{policy}'.")
    return policy


class AutoSamRwtdDataset:
    def __init__(
        self,
        *,
        samples: Sequence[RwtdSample],
        train: bool,
        repeat_factor: int,
        train_augmentation_policy: str,
        sam_transform,
    ) -> None:
        if not samples:
            raise AutoSamStldRuntimeError("AutoSAM RWTD dataset wrapper received zero samples.")
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

    def __getitem__(self, index: int) -> AutoSamRwtdBatchItem:
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
        return AutoSamRwtdBatchItem(
            sample=sample,
            processed_image=processed_image,
            processed_mask=processed_mask,
            original_size_hw=original_size,
            sam_input_size_hw=sam_input_size,
        )


def collate_autosam_rwtd_batch(batch: Sequence[AutoSamRwtdBatchItem]) -> dict[str, Any]:
    import torch

    if not batch:
        raise AutoSamStldRuntimeError("AutoSAM RWTD collate received an empty batch.")
    return {
        "samples": [item.sample for item in batch],
        "images": torch.stack([item.processed_image for item in batch], dim=0),
        "masks": torch.stack([item.processed_mask for item in batch], dim=0),
        "original_sizes": torch.tensor([item.original_size_hw for item in batch], dtype=torch.int64),
        "image_sizes": torch.tensor([item.sam_input_size_hw for item in batch], dtype=torch.int64),
    }


def resolve_rwtd_variant_name(args) -> str:
    return (
        f"autosam_{str(args.sam_model_type)}_order{int(args.autosam_backbone_order)}"
        f"_idim{int(args.prompt_image_size)}_fg_texture_a"
    )


def prepare_rwtd_autosam_output_dir(output_dir: str | None, *, run_kind: str, split: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    suffix = f"{run_kind}_autosam"
    if split is not None:
        suffix = f"{suffix}_{split}"
    return RWTD_AUTOSAM_OUTPUT_ROOT / suffix


def resolve_train_subset_manifest(args) -> FewShotSubsetManifest | None:
    manifest_path = getattr(args, "train_subset_manifest", None)
    if manifest_path in (None, ""):
        return None
    return load_few_shot_subset_manifest(
        manifest_path,
        expected_dataset_id=RWTD_AUTOSAM_DATASET_ID,
        expected_source_split=str(getattr(args, "train_split", "train")),
    )


def load_rwtd_split_samples(
    *,
    dataset_id: str,
    split: str,
    limit: int | None = None,
    start_index: int = 0,
    subset_manifest: FewShotSubsetManifest | None = None,
) -> tuple[RwtdSample, ...]:
    samples = tuple(
        iter_rwtd_samples(
            split=split,
            dataset_id=dataset_id,
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
            f"No RWTD samples were loaded for split '{split}' with limit={limit} and start_index={start_index}.",
        )
    return samples


def build_rwtd_autosam_train_dataloader(
    *,
    samples: Sequence[RwtdSample],
    batch_size: int,
    num_workers: int,
    repeat_factor: int,
    train_augmentation_policy: str,
    sam_transform,
    seed: int,
):
    import torch

    dataset = AutoSamRwtdDataset(
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
        collate_fn=collate_autosam_rwtd_batch,
        generator=generator,
    )


def build_rwtd_autosam_eval_dataloader(*, samples: Sequence[RwtdSample], num_workers: int, sam_transform):
    import torch

    dataset = AutoSamRwtdDataset(
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
        collate_fn=collate_autosam_rwtd_batch,
    )


def build_rwtd_checkpoint_payload(*, prompt_generator, args, training_history_rows, trainable_parameter_count):
    return {
        "variant": resolve_rwtd_variant_name(args),
        "dataset_id": RWTD_AUTOSAM_DATASET_ID,
        "train_dataset_id": str(args.dataset_id),
        "foreground_target_name": "texture_a_mask",
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
        "train_augmentation_policy": resolve_rwtd_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_rwtd_augmentation_policy(args)),
        "training_history_rows": training_history_rows,
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "upstream_alignment_notes": (
            "Prompt generator and SAM forward mirror the upstream 2D AutoSAM path. "
            "RWTD is adapted into a binary foreground task by treating texture_a_mask "
            "as foreground during training and direct-foreground reporting."
        ),
    }


def train_rwtd_autosam(*, train_samples: Sequence[RwtdSample], output_dir: Path, args) -> AutoSamRwtdTrainingArtifacts:
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

    train_loader = build_rwtd_autosam_train_dataloader(
        samples=train_samples,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        repeat_factor=int(args.train_repeat_factor),
        train_augmentation_policy=resolve_rwtd_augmentation_policy(args),
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
            if ((step_index + 1) % accumulation_steps) == 0:
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
        training_history_rows.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(mean(epoch_losses)) if epoch_losses else None,
                "mean_bce_loss": float(mean(epoch_bce)) if epoch_bce else None,
                "mean_dice_loss": float(mean(epoch_dice)) if epoch_dice else None,
                "mean_predicted_positive_fraction": float(mean(epoch_pred_fraction)) if epoch_pred_fraction else None,
                "num_batches": int(len(epoch_losses)),
            }
        )

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        build_rwtd_checkpoint_payload(
            prompt_generator=prompt_generator,
            args=args,
            training_history_rows=training_history_rows,
            trainable_parameter_count=trainable_parameter_count,
        ),
        checkpoint_path,
    )
    write_csv(output_dir / "train_history.csv", training_history_rows)
    return AutoSamRwtdTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        trainable_parameter_count=trainable_parameter_count,
        training_history_rows=training_history_rows,
    )


def build_rwtd_eval_row(
    *,
    sample: RwtdSample,
    variant: str,
    checkpoint_path: Path,
    trainable_parameter_count: int,
    dense_embedding_shape: Sequence[int],
    low_res_shape: Sequence[int],
    direct_foreground_metrics,
    foreground_threshold: float,
    assignment,
) -> dict[str, Any]:
    total_pixels = int(sample.height * sample.width)
    row = {
        "variant": variant,
        "dataset_id": RWTD_AUTOSAM_DATASET_ID,
        "split": sample.split,
        "sample_index": int(sample.index),
        "crop_name": sample.crop_name,
        "checkpoint_path": str(checkpoint_path),
        "foreground_evaluation_view": "direct_foreground",
        "foreground_target_name": "texture_a_mask",
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


def save_rwtd_result_bundle(*, output_dir: Path, sample: RwtdSample, result: AutoSamRwtdEvalBundle, variant: str, save_visuals: bool) -> None:
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
            protocol=f"rwtd_autosam:{variant}",
            metric_summary=result.metric_summary,
        )


def build_rwtd_summary(*, rows: list[dict[str, Any]], split: str, args, checkpoint_path: Path, run_kind: str, trainable_parameter_count: int) -> dict[str, Any]:
    mean_metrics = {name: float(mean(float(row[name]) for row in rows)) for name in RWTD_AUTOSAM_SCALAR_FIELDS}
    median_metrics = {name: float(median(float(row[name]) for row in rows)) for name in RWTD_AUTOSAM_SCALAR_FIELDS}
    return {
        "run_kind": run_kind,
        "variant": resolve_rwtd_variant_name(args),
        "variant_summary": (
            "Upstream AutoSAM prompt-generator over frozen SAM, trained on RWTD "
            "with fixed foreground target texture_a_mask."
        ),
        "dataset_id": RWTD_AUTOSAM_DATASET_ID,
        "train_dataset_id": str(args.dataset_id),
        "split": split,
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "train_selection_policy": "curated_rwtd_default" if resolve_train_subset_manifest(args) is None else "few_shot_subset_manifest_over_curated_rwtd",
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "device": args.device,
        "checkpoint_path": str(checkpoint_path),
        "num_evaluated_samples": len(rows),
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "foreground_evaluation_view": "direct_foreground",
        "foreground_target_name": "texture_a_mask",
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
        "train_augmentation_policy": resolve_rwtd_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_rwtd_augmentation_policy(args)),
        "preprocessing_summary": (
            "native RWTD image/mask_a -> upstream paired transform -> "
            f"ResizeLongestSide(1024) -> upstream preprocess semantics -> prompt-generator bilinear downsample to {int(args.prompt_image_size)}x{int(args.prompt_image_size)}"
        ),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "upstream_alignment_notes": (
            "Training foreground is fixed to texture_a_mask. "
            "Auxiliary partition metrics use Hungarian-equivalent 2-way assignment over foreground/background and texture_a/texture_b."
        ),
        "mean_metrics": mean_metrics,
        "median_metrics": median_metrics,
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def flatten_rwtd_summary(summary: dict[str, Any]) -> dict[str, Any]:
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


def build_rwtd_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RWTD AutoSAM Summary",
        "",
        f"- Run kind: `{summary['run_kind']}`",
        f"- Variant: `{summary['variant']}`",
        f"- Split: `{summary['split']}`",
        f"- Foreground target: `{summary['foreground_target_name']}`",
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


def rewrite_rwtd_summary(*, output_dir: Path, summary: dict[str, Any]) -> None:
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", [flatten_rwtd_summary(summary)])
    write_text(output_dir / "summary.md", build_rwtd_summary_markdown(summary))


def build_rwtd_run_config(args, *, run_kind: str, dataset_partition=None, selected_sample_count: int | None = None) -> dict[str, Any]:
    config = {
        "command": getattr(args, "command", None),
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(str(value)) for value in sys.argv),
        "run_kind": run_kind,
        "dataset_id": RWTD_AUTOSAM_DATASET_ID,
        "train_dataset_id": str(args.dataset_id),
        "train_selection_policy": "curated_rwtd_default" if resolve_train_subset_manifest(args) is None else "few_shot_subset_manifest_over_curated_rwtd",
        "split": getattr(args, "split", None),
        "train_split": getattr(args, "train_split", None),
        "eval_split": getattr(args, "eval_split", None),
        "train_subset_manifest_path": getattr(args, "train_subset_manifest", None),
        "train_subset_manifest_output_path": getattr(args, "_resolved_train_subset_manifest_output_path", None),
        "train_subset_manifest": getattr(args, "_resolved_train_subset_manifest", None),
        "variant": resolve_rwtd_variant_name(args),
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
        "train_augmentation_policy": resolve_rwtd_augmentation_policy(args),
        "train_augmentation_summary": describe_autosam_augmentation_policy(resolve_rwtd_augmentation_policy(args)),
        "num_workers": int(args.num_workers),
        "save_visuals": bool(args.save_visuals),
        "seed": int(getattr(args, "seed", 0)),
        "upstream_autosam_repo_root": str(AUTOSAM_REPO_ROOT),
        "foreground_target_name": "texture_a_mask",
        "versions": discovered_package_versions(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    append_dataset_partition_fields(config, dataset_partition, selected_sample_count=selected_sample_count)
    return config


def build_rwtd_experiment_terms_markdown(args, *, run_kind: str, train_sample_count: int | None, eval_sample_count: int | None, checkpoint_path: Path | None) -> str:
    sections = [
        ExperimentTermsSection(
            title="Experiment Scope",
            bullets=(
                f"Command family: `{getattr(args, 'command', 'rwtd-autosam')}`.",
                f"Run kind: `{run_kind}`.",
                f"Dataset: `{RWTD_AUTOSAM_DATASET_ID}` from loader dataset id `{getattr(args, 'dataset_id', None)}`.",
                f"Upstream AutoSAM checkout: `{AUTOSAM_REPO_ROOT}`.",
                f"SAM checkpoint: `{getattr(args, 'sam_checkpoint_path', None)}` with model type `{getattr(args, 'sam_model_type', None)}`.",
                "SAM parameters stay frozen. Only the upstream AutoSAM prompt generator is optimized.",
            ),
        ),
        ExperimentTermsSection(
            title="Data And Target Semantics",
            bullets=(
                f"Training split: `{getattr(args, 'train_split', None)}` with limit `{getattr(args, 'train_limit', None)}`; loaded `{train_sample_count}` sample(s).",
                f"Evaluation split: `{getattr(args, 'eval_split', getattr(args, 'split', None))}` with limit `{getattr(args, 'eval_limit', getattr(args, 'limit', None))}`; loaded `{eval_sample_count}` sample(s).",
                (
                    f"Train subset manifest: `{getattr(args, '_resolved_train_subset_manifest_output_path', getattr(args, 'train_subset_manifest', None))}` selecting `{getattr(args, '_resolved_train_subset_manifest', {}).get('shot_count')}` image(s) with subset seed `{getattr(args, '_resolved_train_subset_manifest', {}).get('subset_seed')}`."
                    if getattr(args, "_resolved_train_subset_manifest", None) is not None
                    else "No train subset manifest is applied. The curated RWTD train split is used."
                ),
                "Training foreground target is always `texture_a_mask`.",
                "Direct foreground metrics score the predicted foreground against `texture_a_mask`.",
                "Auxiliary partition metrics use Hungarian-equivalent 2-way assignment between `{pred_fg,pred_bg}` and `{texture_a,texture_b}`.",
            ),
        ),
        ExperimentTermsSection(
            title="AutoSAM Training Path",
            bullets=(
                f"Prompt generator: upstream `ModelEmb` with HarDNet order `{int(args.autosam_backbone_order)}` and depth_wise `{bool(args.autosam_depth_wise)}`.",
                f"Prompt image size: `{int(args.prompt_image_size)}`.",
                f"Optimizer: `Adam(lr={float(args.learning_rate)}, wd={float(args.weight_decay)})` for `{int(args.num_epochs)}` epoch(s).",
                f"Batch size: `{int(args.batch_size)}` with train repeat factor `{int(args.train_repeat_factor)}`.",
                f"Train augmentation: `{describe_autosam_augmentation_policy(resolve_rwtd_augmentation_policy(args))}`.",
                "Loss: upstream-style `BCELoss + Dice` on normalized low-resolution SAM masks.",
            ),
        ),
        ExperimentTermsSection(
            title="Outputs",
            bullets=(
                f"Common run files: `config.json`, `experiment_terms.md`, `summary.json`, `summary.csv`, `summary.md`, `per_sample_metrics.csv`, `per_sample_metrics.jsonl`, and `visuals_manifest.jsonl` under `{RWTD_AUTOSAM_OUTPUT_ROOT}`.",
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
        title="RWTD AutoSAM Experiment Terms",
        summary_lines=(
            "This file records the exact RWTD AutoSAM run that produced this directory.",
            "RWTD is adapted into a fixed-foreground AutoSAM task by training against texture_a_mask.",
        ),
        sections=sections,
        related_paths=(
            str(AUTOSAM_REPO_ROOT / "models" / "model_single.py"),
            str(AUTOSAM_REPO_ROOT / "train_3d"),
            str(Path("src/rwtd_sam3/eval/rwtd_autosam.py")),
        ),
    )


def evaluate_rwtd_autosam(
    *,
    prompt_generator,
    checkpoint_path: Path,
    samples: Sequence[RwtdSample],
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

    eval_loader = build_rwtd_autosam_eval_dataloader(
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
        row = build_rwtd_eval_row(
            sample=source_sample,
            variant=resolve_rwtd_variant_name(args),
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
            result = AutoSamRwtdEvalBundle(
                row=row,
                metric_summary=metric_summary,
                foreground_prediction=foreground_prediction,
                background_prediction=background_prediction,
                foreground_probability=foreground_probability,
            )
            save_rwtd_result_bundle(
                output_dir=output_dir,
                sample=source_sample,
                result=result,
                variant=resolve_rwtd_variant_name(args),
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
                        protocol=f"rwtd_autosam:{resolve_rwtd_variant_name(args)}",
                        visual_path=Path("visuals") / f"{source_sample.index}.png",
                    )
                )

    if not rows:
        raise AutoSamStldRuntimeError("RWTD AutoSAM evaluation produced no sample rows.")

    summary = build_rwtd_summary(
        rows=rows,
        split=str(getattr(args, "eval_split", getattr(args, "split", "test"))),
        args=args,
        checkpoint_path=checkpoint_path,
        run_kind=run_kind,
        trainable_parameter_count=trainable_parameter_count,
    )
    if save_artifacts and output_dir is not None:
        write_csv(output_dir / "per_sample_metrics.csv", rows)
        write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
        write_jsonl(output_dir / "visuals_manifest.jsonl", visual_records)
        rewrite_rwtd_summary(output_dir=output_dir, summary=summary)
    return summary


def run_rwtd_autosam_train(args) -> dict[str, Any]:
    import torch

    if getattr(args, "train_limit", None) is not None and getattr(args, "train_subset_manifest", None) not in (None, ""):
        raise AutoSamStldRuntimeError("train_limit and train_subset_manifest are mutually exclusive for RWTD AutoSAM training.")
    resolved_checkpoint_path = Path(str(args.sam_checkpoint_path))
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint path does not exist: {resolved_checkpoint_path}")

    train_subset_manifest = resolve_train_subset_manifest(args)
    setattr(args, "_resolved_train_subset_manifest", train_subset_manifest.to_json_dict() if train_subset_manifest is not None else None)
    output_dir = prepare_rwtd_autosam_output_dir(args.output_dir, run_kind="train", split=args.eval_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_subset_manifest is not None:
        local_manifest_path = output_dir / "train_subset_manifest.json"
        write_json(local_manifest_path, train_subset_manifest.to_json_dict())
        setattr(args, "_resolved_train_subset_manifest_output_path", str(local_manifest_path))
    else:
        setattr(args, "_resolved_train_subset_manifest_output_path", None)

    train_samples = load_rwtd_split_samples(
        dataset_id=str(args.dataset_id),
        split=str(args.train_split),
        limit=getattr(args, "train_limit", None),
        subset_manifest=train_subset_manifest,
    )
    eval_samples = load_rwtd_split_samples(
        dataset_id=str(args.dataset_id),
        split=str(args.eval_split),
        limit=getattr(args, "eval_limit", None),
    )

    config = build_rwtd_run_config(args, run_kind="train")
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_rwtd_experiment_terms_markdown(
            args,
            run_kind="train",
            train_sample_count=len(train_samples),
            eval_sample_count=len(eval_samples),
            checkpoint_path=None,
        ),
    )

    training_artifacts = train_rwtd_autosam(train_samples=train_samples, output_dir=output_dir, args=args)
    checkpoint = torch.load(training_artifacts.checkpoint_path, map_location="cpu")
    train_summary = evaluate_rwtd_autosam(
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
    summary = evaluate_rwtd_autosam(
        prompt_generator=build_prompt_generator_from_checkpoint(checkpoint),
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
    rewrite_rwtd_summary(output_dir=output_dir, summary=summary)
    return summary


def run_rwtd_autosam_eval(args) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RWTD AutoSAM checkpoint path does not exist: {checkpoint_path}")
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
    setattr(args, "dataset_id", str(checkpoint.get("train_dataset_id", getattr(args, "dataset_id", CURATED_RWTD_LOCAL_ALIAS))))
    prompt_generator = build_prompt_generator_from_checkpoint(checkpoint)

    overview = load_split_overview(split=args.split, dataset_id=str(args.dataset_id))
    dataset_partition = resolve_dataset_partition(overview.num_examples, getattr(args, "dataset_partition", None))
    num_samples = resolve_eval_sample_count(getattr(args, "limit", None), overview.num_examples, dataset_partition)
    eval_samples = load_rwtd_split_samples(
        dataset_id=str(args.dataset_id),
        split=str(args.split),
        limit=num_samples,
        start_index=dataset_partition.start_index if dataset_partition is not None else 0,
    )

    output_dir = prepare_rwtd_autosam_output_dir(args.output_dir, run_kind="eval", split=args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_rwtd_run_config(args, run_kind="eval", dataset_partition=dataset_partition, selected_sample_count=num_samples)
    write_json(output_dir / "config.json", config)
    write_text(
        output_dir / "experiment_terms.md",
        build_rwtd_experiment_terms_markdown(
            args,
            run_kind="eval",
            train_sample_count=None,
            eval_sample_count=len(eval_samples),
            checkpoint_path=checkpoint_path,
        ),
    )
    summary = evaluate_rwtd_autosam(
        prompt_generator=prompt_generator,
        checkpoint_path=checkpoint_path,
        samples=eval_samples,
        args=args,
        run_kind="eval",
        trainable_parameter_count=int(checkpoint["trainable_parameter_count"]),
        output_dir=output_dir,
        save_artifacts=True,
    )
    rewrite_rwtd_summary(output_dir=output_dir, summary=summary)
    return summary
