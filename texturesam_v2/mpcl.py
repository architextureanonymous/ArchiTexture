from __future__ import annotations

import torch

EPS = 1e-6


def soft_iou(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute SoftIoU for probabilistic masks.

    Args:
        mask_a: [H, W] or [N, H, W]
        mask_b: same shape as mask_a
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask_a and mask_b must have the same shape")

    a = mask_a.reshape(mask_a.shape[0], -1) if mask_a.ndim == 3 else mask_a.reshape(1, -1)
    b = mask_b.reshape(mask_b.shape[0], -1) if mask_b.ndim == 3 else mask_b.reshape(1, -1)

    inter = (a * b).sum(dim=1)
    union = a.sum(dim=1) + b.sum(dim=1) - inter
    return inter / (union + eps)


def mpcl_loss(pred_masks: torch.Tensor) -> torch.Tensor:
    """
    Multi-Prompt Consistency Loss.

    Args:
        pred_masks: tensor [K, H, W] with probabilities in [0,1]

    Returns:
        scalar tensor: average of (1 - SoftIoU(p_i, p_j)) over i != j
    """
    if pred_masks.ndim != 3:
        raise ValueError("pred_masks must have shape [K, H, W]")

    k = pred_masks.shape[0]
    if k <= 1:
        return pred_masks.new_tensor(0.0)

    total = pred_masks.new_tensor(0.0)
    count = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            sij = soft_iou(pred_masks[i], pred_masks[j]).mean()
            total = total + (1.0 - sij)
            count += 1

    return total / max(count, 1)
