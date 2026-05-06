"""Faithful GlaS AutoSAM reproduction route.

This module implements a GlaS-specific AutoSAM runner whose primary goal is
reproduction fidelity, mirroring the MoNuSeg faithful route.
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from rwtd_sam3.data.glas_binary import (
    GLAS_DATASET_ID,
    GlasBinaryOverview,
    GlasBinarySample,
    iter_glas_binary_samples,
    load_glas_binary_overview,
)
from rwtd_sam3.eval.experiment_terms import ExperimentTermsSection, render_experiment_terms_markdown
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

GLAS_AUTOSAM_ROUTE = "glas"
GLAS_AUTOSAM_OUTPUT_ROOT = Path("outputs") / "glas_binary" / "autosam"
GLAS_AUTOSAM_SCALAR_FIELDS = (
    "upstream_eval_iou",
    "upstream_eval_dice",
    "direct_foreground_iou",
    "direct_foreground_dice",
    "eval_miou",
    "eval_ari",
)

GLAS_AUTOSAM_REPRODUCTION_PROFILE = {
    "profile_name": "glas_faithful_v1",
    "sam_model_type": "vit_h",
    "prompt_image_size": 256,
    "autosam_backbone_order": 85,
    "autosam_depth_wise": False,
    "batch_size": 10,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "num_epochs": 200,
    "train_repeat_factor": 3,
    "eval_every_epochs": 1,
    "selection_metric": "upstream_eval_iou",
    "selection_split": "test",
    "selection_checkpoint_mode": "best_eval",
    "foreground_threshold": 0.5,
    "num_workers": 4,
}

@dataclass(frozen=True)
class AutoSamGlasBatchItem:
    sample: GlasBinarySample
    processed_image: torch.Tensor
    processed_mask: torch.Tensor
    original_size_hw: tuple[int, int]
    sam_input_size_hw: tuple[int, int]

@dataclass(frozen=True)
class AutoSamGlasTrainingArtifacts:
    best_checkpoint_path: Path | None
    final_checkpoint_path: Path
    trainable_parameter_count: int
    training_history_rows: list[dict[str, Any]]
    best_epoch_by_selection_metric: int | None
    best_selection_metric_value: float | None

def build_autosam_train_transform():
    transforms = load_upstream_autosam_transforms_shir()
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, scale=(0.75, 1.25)),
        transforms.ToTensor(),
    ])

def build_autosam_eval_transform():
    transforms = load_upstream_autosam_transforms_shir()
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

class AutoSamGlasDataset(torch.utils.data.Dataset):
    def __init__(self, samples: Sequence[GlasBinarySample], sam_transform, train: bool = False, repeat_factor: int = 1):
        self.samples = tuple(samples)
        self.sam_transform = sam_transform
        self.train = train
        self.repeat_factor = repeat_factor
        self.transform = build_autosam_train_transform() if self.train else build_autosam_eval_transform()

    def __len__(self):
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        image_np = np.asarray(sample.image.convert("RGB"))
        mask_np = np.asarray(sample.texture_a_mask, dtype=np.uint8)
        
        # AutoSAM upstream transform takes (image, mask)
        image_tensor, mask_tensor = self.transform(image_np, mask_np)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]
        
        orig_hw = (image_tensor.shape[1], image_tensor.shape[2])
        
        image_tensor = self.sam_transform.apply_image_torch(image_tensor)
        mask_tensor = self.sam_transform.apply_image_torch(mask_tensor)
        mask_tensor = (mask_tensor > 0.5).to(dtype=image_tensor.dtype)
        
        proc_image = self.sam_transform.preprocess(image_tensor)
        mask_h, mask_w = mask_tensor.shape[-2:]
        proc_mask = F.pad(
            mask_tensor,
            (0, self.sam_transform.target_length - mask_w, 0, self.sam_transform.target_length - mask_h),
        )
        
        sam_input_hw = (image_tensor.shape[-2], image_tensor.shape[-1])
        
        return AutoSamGlasBatchItem(
            sample=sample,
            processed_image=proc_image,
            processed_mask=proc_mask,
            original_size_hw=orig_hw,
            sam_input_size_hw=sam_input_hw,
        )

def collate_autosam_glas_batch(batch: Sequence[AutoSamGlasBatchItem]):
    return {
        "samples": [item.sample for item in batch],
        "images": torch.stack([item.processed_image for item in batch]),
        "masks": torch.stack([item.processed_mask for item in batch]),
        "original_sizes": torch.tensor([item.original_size_hw for item in batch]),
        "image_sizes": torch.tensor([item.sam_input_size_hw for item in batch]),
    }

def dice_loss(y_true, y_pred, smooth=1.0):
    y_pred = y_pred.clamp(0.0, 1.0)
    y_true = y_true.clamp(0.0, 1.0)
    intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
    union = y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def run_glas_autosam_train(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else GLAS_AUTOSAM_OUTPUT_ROOT / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_accumulation_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))

    sam_transform = load_upstream_autosam_resize_longest_side_class()(1024)
    prompt_generator = build_upstream_autosam_prompt_generator(
        order=args.autosam_backbone_order,
        depth_wise=args.autosam_depth_wise,
    ).to(device)
    
    sam_model = build_upstream_autosam_sam(
        model_type=args.sam_model_type,
        checkpoint_path=args.sam_checkpoint_path,
    ).to(device)
    sam_model.eval()
    for p in sam_model.parameters():
        p.requires_grad = False

    overview = load_glas_binary_overview(args.dataset_root, split="train")
    train_samples = list(iter_glas_binary_samples(args.dataset_root, split="train"))
    
    test_overview = load_glas_binary_overview(args.dataset_root, split="test")
    test_samples = list(iter_glas_binary_samples(args.dataset_root, split="test"))

    train_ds = AutoSamGlasDataset(train_samples, sam_transform, train=True, repeat_factor=args.train_repeat_factor)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_autosam_glas_batch, num_workers=args.num_workers
    )

    optimizer = torch.optim.Adam(prompt_generator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    best_selection_val = -1.0
    best_epoch = -1
    history = []
    write_json(
        output_dir / "config.json",
        {
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "dataset_id": GLAS_DATASET_ID,
            "dataset_root": str(args.dataset_root),
            "device": str(args.device),
            "sam_checkpoint_path": str(args.sam_checkpoint_path),
            "sam_model_type": str(args.sam_model_type),
            "prompt_image_size": int(args.prompt_image_size),
            "autosam_backbone_order": int(args.autosam_backbone_order),
            "autosam_depth_wise": bool(args.autosam_depth_wise),
            "batch_size": int(args.batch_size),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": int(args.batch_size) * gradient_accumulation_steps,
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_epochs": int(args.num_epochs),
            "train_repeat_factor": int(args.train_repeat_factor),
            "eval_every_epochs": int(args.eval_every_epochs),
            "num_workers": int(args.num_workers),
        },
    )

    for epoch in range(1, args.num_epochs + 1):
        prompt_generator.train()
        if getattr(args, "freeze_prompt_bn", False):
            for m in prompt_generator.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        epoch_loss = 0.0
        num_optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            
            # Simple faithful re-impl of forward
            small_images = F.interpolate(images, size=(args.prompt_image_size, args.prompt_image_size), mode="bilinear", align_corners=True)
            dense_prompts = prompt_generator(small_images)
            
            # Sam call placeholder logic
            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(images)
                sparse_embeddings, _ = sam_model.prompt_encoder(points=None, boxes=None, masks=None)
            
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_prompts,
                multimask_output=False,
            )
            
            # Normalize as per AutoSAM
            m_min = low_res_masks.view(low_res_masks.size(0), -1).min(1)[0].view(-1, 1, 1, 1)
            m_max = low_res_masks.view(low_res_masks.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
            norm_masks = (low_res_masks - m_min) / (m_max - m_min + 1e-6)
            
            target_low_res = F.interpolate(masks, size=norm_masks.shape[-2:], mode="nearest")
            
            loss = criterion(norm_masks, target_low_res) + dice_loss(target_low_res, norm_masks)
            (loss / float(gradient_accumulation_steps)).backward()
            if batch_index % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                num_optimizer_steps += 1
            epoch_loss += loss.item()
        if len(train_loader) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            num_optimizer_steps += 1
        
        avg_loss = epoch_loss / len(train_loader)
        row = {
            "epoch": epoch,
            "mean_train_loss": avg_loss,
            "num_batches": len(train_loader),
            "num_optimizer_steps": num_optimizer_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
        history.append(row)
        write_csv(output_dir / "train_history.csv", history)
        LOGGER.info(
            "GlaS AutoSAM epoch=%s/%s | mean_loss=%.4f batches=%s optimizer_steps=%s accum=%s",
            epoch,
            args.num_epochs,
            avg_loss,
            len(train_loader),
            num_optimizer_steps,
            gradient_accumulation_steps,
        )
        
        if epoch % args.eval_every_epochs == 0:
            # Add eval call here if needed for selection
            pass

    # Save final
    torch.save(prompt_generator.state_dict(), output_dir / "final.pth")
    return {"status": "success", "output_dir": str(output_dir), "train_history_path": str(output_dir / "train_history.csv")}

def run_glas_autosam_eval(args):
    # Integration for eval-glas-autosam-faithful
    return {"status": "not_implemented_fully"}
