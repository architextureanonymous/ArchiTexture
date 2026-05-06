"""YKND multiclass baseline on frozen SAM multiscale features.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rwtd_sam3.data.yknd import (
    YKND_DATASET_ID,
    YkndSample,
    iter_yknd_samples,
)
from rwtd_sam3.eval.runner import (
    write_csv,
    write_json,
)
from rwtd_sam3.models.frozen_sam_pyramid_extractor import (
    build_frozen_sam_pyramid_extractor,
)
from rwtd_sam3.models.sam3_frozen_multiscale_mask_head import (
    FROZEN_MASK_HEAD_VARIANT_SPECS,
    FrozenMaskHeadOutput,
    FrozenResidualMaskHeadOutput,
    FrozenSamMultiscaleMaskHead,
    FrozenSamCoarsePlusAttentionRefineMaskHead,
    FrozenSamCoarsePlusCrossAttentionRefineMaskHead,
    FrozenSamCoarsePlusFineResidualMaskHead,
    is_coarse_plus_residual_variant,
    resolve_coarse_plus_residual_levels,
    resolve_frozen_mask_head_head_family,
    resolve_frozen_mask_head_variant_levels,
)
from rwtd_sam3.eval.metrics import compute_binary_metrics

LOGGER = logging.getLogger(__name__)

YKND_FROZEN_MASK_HEAD_OUTPUT_ROOT = Path("outputs") / "yknd" / "frozen_sam_mask_head"

# Default test split as requested: 2-images
YKND_DEFAULT_TEST_STEMS = [
    "bulk_770958_9542817",
    "bulk_73740_9593218"
]

YKND_NUM_CLASSES = 6

def run_yknd_frozen_mask_head_train(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir or YKND_FROZEN_MASK_HEAD_OUTPUT_ROOT / f"train_multiclass_{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
    )
    
    # Load data
    train_samples = list(iter_yknd_samples(
        args.dataset_root,
        split="train",
        test_stems=YKND_DEFAULT_TEST_STEMS,
        limit=getattr(args, "train_limit", None)
    ))
    
    eval_samples = list(iter_yknd_samples(
        args.dataset_root,
        split="test",
        test_stems=YKND_DEFAULT_TEST_STEMS,
        limit=getattr(args, "eval_limit", None)
    ))
    
    # Feature caching
    feature_cache: dict[str, dict[str, np.ndarray]] = {}
    
    def get_cached_pyramid(sample: YkndSample) -> dict[str, np.ndarray]:
        if sample.crop_name not in feature_cache:
            with torch.no_grad():
                _, pyramid = extractor.extract_sam_pyramid(sample.image)
                feature_cache[sample.crop_name] = pyramid
        return feature_cache[sample.crop_name]

    # Pre-extract for first sample to get level dims
    if not train_samples:
        raise ValueError("No train samples found.")
    
    pyramid = get_cached_pyramid(train_samples[0])
    level_input_dims = {name: feat.shape[0] for name, feat in pyramid.items()}
    
    # Resolve variant levels
    try:
        selected_levels = resolve_frozen_mask_head_variant_levels(
            variant=args.variant,
            available_level_names=list(pyramid.keys())
        )
        head_input_dims = {name: level_input_dims[name] for name in selected_levels}
    except Exception:
        # Fallback for complex variants
        if is_coarse_plus_residual_variant(args.variant):
            c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
            head_input_dims = {
                c_name: level_input_dims[c_name],
                f_name: level_input_dims[f_name]
            }
        else:
            head_input_dims = level_input_dims

    # Build head with num_classes
    head_family = resolve_frozen_mask_head_head_family(args.variant)
    head_kwargs = {
        "level_input_dims": head_input_dims,
        "projection_dim": args.projection_dim,
        "decoder_dim": args.decoder_dim,
        "num_classes": YKND_NUM_CLASSES,
    }
    
    if head_family == "multiscale_mask_head":
        head = FrozenSamMultiscaleMaskHead(**head_kwargs).module.to(device)
    elif head_family == "coarse_plus_attention_refine_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusAttentionRefineMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
            attention_hidden_dim=args.attention_hidden_dim,
        ).module.to(device)
    elif head_family == "coarse_plus_cross_attention_refine_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusCrossAttentionRefineMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
            attention_hidden_dim=args.attention_hidden_dim,
            cross_attn_query_stride=args.cross_attn_query_stride,
        ).module.to(device)
    elif head_family == "coarse_plus_fine_residual_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusFineResidualMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
        ).module.to(device)
    else:
        raise ValueError(f"Unsupported head family: {head_family}")

    optimizer = optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Pre-warm cache
    LOGGER.info("Pre-extracting features for all samples...")
    for sample in train_samples:
        get_cached_pyramid(sample)
    for sample in eval_samples:
        get_cached_pyramid(sample)
    LOGGER.info("Feature extraction complete.")

    history = []
    best_miou = -1.0
    
    for epoch in range(args.num_epochs):
        head.train()
        epoch_losses = []
        
        for sample in train_samples:
            optimizer.zero_grad()
            
            pyramid = get_cached_pyramid(sample)
            active_features = {
                k: torch.from_numpy(pyramid[k]).to(device).unsqueeze(0)
                for k in head_input_dims
            }
                
            output = head(active_features, image_size=(sample.height, sample.width))
            target = torch.from_numpy(sample.label_mask).long().unsqueeze(0).to(device)
            
            # Categorical CE Loss
            if is_coarse_plus_residual_variant(args.variant):
                loss_main = F.cross_entropy(output.logits, target)
                loss_coarse = F.cross_entropy(output.coarse_logits, target)
                loss = loss_main + args.coarse_loss_weight * loss_coarse
            else:
                loss = F.cross_entropy(output.logits, target)
                
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        
        # Eval
        eval_metrics = evaluate_yknd_multiclass(head, get_cached_pyramid, eval_samples, device, head_input_dims)
        curr_miou = eval_metrics["mean_iou"]
        
        history_row = {
            "epoch": epoch,
            "loss": avg_loss,
            **eval_metrics
        }
        history.append(history_row)
        LOGGER.info(f"Epoch {epoch}: loss={avg_loss:.4f}, miou={curr_miou:.4f}")
        
        if curr_miou > best_miou:
            best_miou = curr_miou
            torch.save(head.state_dict(), output_dir / "best_checkpoint.pt")
            
    # Final save
    torch.save(head.state_dict(), output_dir / "checkpoint.pt")
    write_csv(output_dir / "train_history.csv", history)
    write_json(output_dir / "summary.json", {
        "best_miou": best_miou,
        "final_miou": history[-1]["mean_iou"] if history else -1.0,
        "variant": args.variant,
        "num_classes": YKND_NUM_CLASSES,
    })
    # Post-training: save visuals using the best model
    if (output_dir / "best_checkpoint.pt").exists():
        LOGGER.info("Generating final evaluation visuals...")
        head.load_state_dict(torch.load(output_dir / "best_checkpoint.pt", map_location=device))
        head.eval()
        with torch.no_grad():
            for sample in eval_samples:
                pyramid = get_cached_pyramid(sample)
                active_features = {
                    k: torch.from_numpy(pyramid[k]).to(device).unsqueeze(0)
                    for k in head_input_dims
                }
                output = head(active_features, image_size=(sample.height, sample.width))
                pred_labels = torch.argmax(output.logits, dim=1).squeeze(0).cpu().numpy()
                save_multiclass_visuals(sample, pred_labels, output_dir)
    
    return history[-1] if history else {}

def evaluate_yknd_multiclass(head, get_cached_pyramid, eval_samples, device, head_input_dims) -> dict[str, float]:
    head.eval()
    all_ious = []
    
    with torch.no_grad():
        for sample in eval_samples:
            pyramid = get_cached_pyramid(sample)
            active_features = {
                k: torch.from_numpy(pyramid[k]).to(device).unsqueeze(0)
                for k in head_input_dims
            }
            output = head(active_features, image_size=(sample.height, sample.width))
            # [1, num_classes, H, W] -> [H, W]
            pred_labels = torch.argmax(output.logits, dim=1).squeeze(0).cpu().numpy()
            
            target_labels = sample.label_mask
            
            # Compute IoU per class
            ious = []
            for c in range(YKND_NUM_CLASSES):
                pred_c = pred_labels == c
                target_c = target_labels == c
                if not np.any(target_c) and not np.any(pred_c):
                    continue # Skip classes not present in both
                
                intersection = np.logical_and(pred_c, target_c).sum()
                union = np.logical_or(pred_c, target_c).sum()
                ious.append(intersection / union if union > 0 else 1.0)
            
            if ious:
                all_ious.append(np.mean(ious))
            
    return {
        "mean_iou": np.mean(all_ious) if all_ious else 0.0,
    }

def run_yknd_frozen_mask_head_eval(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir or YKND_FROZEN_MASK_HEAD_OUTPUT_ROOT / f"eval_multiclass_{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    extractor = build_frozen_sam_pyramid_extractor(
        model_id=args.model_id,
        device=args.device,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        official_checkpoint_path=getattr(args, "official_checkpoint_path", None),
    )
    
    # Load data
    eval_samples = list(iter_yknd_samples(
        args.dataset_root,
        split="test",
        test_stems=YKND_DEFAULT_TEST_STEMS,
        limit=getattr(args, "limit", None)
    ))
    
    if not eval_samples:
        raise ValueError("No eval samples found.")
    
    # Get level dims from one forward pass
    _, pyramid = extractor.extract_sam_pyramid(eval_samples[0].image)
    level_input_dims = {name: feat.shape[0] for name, feat in pyramid.items()}
    
    # Resolve variant levels
    try:
        selected_levels = resolve_frozen_mask_head_variant_levels(
            variant=args.variant,
            available_level_names=list(pyramid.keys())
        )
        head_input_dims = {name: level_input_dims[name] for name in selected_levels}
    except Exception:
        if is_coarse_plus_residual_variant(args.variant):
            c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
            head_input_dims = {c_name: level_input_dims[c_name], f_name: level_input_dims[f_name]}
        else:
            head_input_dims = level_input_dims

    # Build head
    head_family = resolve_frozen_mask_head_head_family(args.variant)
    head_kwargs = {
        "level_input_dims": head_input_dims,
        "projection_dim": args.projection_dim,
        "decoder_dim": args.decoder_dim,
        "num_classes": YKND_NUM_CLASSES,
    }
    
    if head_family == "multiscale_mask_head":
        head = FrozenSamMultiscaleMaskHead(**head_kwargs).module.to(device)
    elif head_family == "coarse_plus_attention_refine_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusAttentionRefineMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
            attention_hidden_dim=args.attention_hidden_dim,
        ).module.to(device)
    elif head_family == "coarse_plus_cross_attention_refine_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusCrossAttentionRefineMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
            attention_hidden_dim=args.attention_hidden_dim,
            cross_attn_query_stride=args.cross_attn_query_stride,
        ).module.to(device)
    elif head_family == "coarse_plus_fine_residual_mask_head":
        c_name, f_name = resolve_coarse_plus_residual_levels(args.variant)
        head = FrozenSamCoarsePlusFineResidualMaskHead(
            **head_kwargs,
            coarse_level_name=c_name,
            fine_level_name=f_name,
        ).module.to(device)
    else:
        raise ValueError(f"Unsupported head family: {head_family}")

    # Load checkpoint
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        raise ValueError("Evaluation requires a --checkpoint-path argument.")
    
    head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    head.eval()
    
    all_ious = []
    LOGGER.info(f"Starting multiclass evaluation and visual saving to {output_dir}...")
    with torch.no_grad():
        for sample in eval_samples:
            _, pyramid = extractor.extract_sam_pyramid(sample.image)
            active_features = {
                k: torch.from_numpy(pyramid[k]).to(device).unsqueeze(0)
                for k in head_input_dims
            }
            output = head(active_features, image_size=(sample.height, sample.width))
            pred_labels = torch.argmax(output.logits, dim=1).squeeze(0).cpu().numpy()
            
            target_labels = sample.label_mask
            
            # Compute mIoU
            ious = []
            for c in range(YKND_NUM_CLASSES):
                pred_c = pred_labels == c
                target_c = target_labels == c
                if not np.any(target_c) and not np.any(pred_c):
                    continue
                intersection = np.logical_and(pred_c, target_c).sum()
                union = np.logical_or(pred_c, target_c).sum()
                ious.append(intersection / union if union > 0 else 1.0)
            
            curr_miou = np.mean(ious) if ious else 0.0
            all_ious.append(curr_miou)
            
            # Save visual
            save_multiclass_visuals(sample, pred_labels, output_dir)
            LOGGER.info(f"  Processed {sample.crop_name}: mIoU={curr_miou:.4f}")
            
    mean_iou = np.mean(all_ious)
    LOGGER.info(f"Evaluation complete. Global mIoU: {mean_iou:.4f}")
    
    summary = {
        "mean_iou": mean_iou,
        "variant": args.variant,
        "checkpoint": str(checkpoint_path),
        "num_classes": YKND_NUM_CLASSES,
    }
    write_json(output_dir / "eval_summary.json", summary)
    return summary

def save_multiclass_visuals(sample: YkndSample, pred_labels: np.ndarray, output_dir: Path):
    img_np = np.array(sample.image.convert("RGB"))
    
    # Simple colormap for 6 classes
    # 0: Black (BG), 1: Red, 2: Green, 3: Blue, 4: Yellow, 5: Cyan
    COLORS = np.array([
        [0, 0, 0],       # 0: BG
        [255, 0, 0],     # 1: Red
        [0, 255, 0],     # 2: Green
        [0, 0, 255],     # 3: Blue
        [255, 255, 0],   # 4: Yellow
        [0, 255, 255],   # 5: Cyan
    ], dtype=np.uint8)
    
    pred_color = COLORS[pred_labels]
    gt_color = COLORS[sample.label_mask]
    
    # Overlay with 0.5 alpha
    overlay_pred = (img_np * 0.5 + pred_color * 0.5).astype(np.uint8)
    overlay_gt = (img_np * 0.5 + gt_color * 0.5).astype(np.uint8)
    
    combined = np.concatenate([img_np, overlay_gt, overlay_pred], axis=1)
    Image.fromarray(combined).save(output_dir / f"{sample.crop_name}_multiclass.png")
