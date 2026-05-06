"""Segmentation metric helpers shared across every evaluation track.

This module defines the metric dataclasses and pure NumPy scoring functions used
by prompt-conditioned SAM 3, SAM-2 baselines, SAM-3 automatic-mask variants,
the official TextureSAM reproduction path, and the ArchiTexture route adapter.

Primary entrypoints:
- ``compute_binary_metrics()``: score one boolean prediction/target mask pair.
- ``compute_partition_metrics()``: score an RWTD two-region partition.
- ``compute_mask_set_metrics()`` and ``compute_official_texturesam_metrics()``:
  score raw automatic-mask sets against texture-region labels.

Inputs are boolean or integer label arrays, typically with shape ``(height,
width)``. Outputs are small dataclasses and helper arrays consumed by the
evaluation runners. The module is intentionally side-effect free and depends
only on ``numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# This is the legacy shared comparison view used by many existing evaluators.
# It is kept for backward compatibility and common binary reporting, but it is
# not the full repo-wide evaluation policy surface. See
# ``rwtd_sam3.eval.evaluation_contract`` and ``EVALUATION_CONTRACT.md``.
CANONICAL_EVALUATION_CONTRACT = "architexture_binary_v1"
CANONICAL_PRIMARY_METRIC = "eval_miou"
CANONICAL_SECONDARY_METRIC = "eval_ari"


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary segmentation metrics for one mask pair.

    Attributes:
        iou: Intersection over union.
        dice: Dice score / F1 score.
        precision: Positive predictive value.
        recall: True positive rate.
        predicted_positive: Number of positive pixels in the prediction.
        target_positive: Number of positive pixels in the target.
        matched_positive: Number of matched positive pixels used in the score.
    """

    iou: float
    dice: float
    precision: float
    recall: float
    predicted_positive: int
    target_positive: int
    matched_positive: int

    def to_prefixed_dict(self, prefix: str) -> dict[str, float | int]:
        """Flatten this metric group with a stable key prefix."""

        return {
            f"{prefix}_iou": self.iou,
            f"{prefix}_dice": self.dice,
            f"{prefix}_precision": self.precision,
            f"{prefix}_recall": self.recall,
            f"{prefix}_predicted_positive": self.predicted_positive,
            f"{prefix}_target_positive": self.target_positive,
            f"{prefix}_matched_positive": self.matched_positive,
        }


@dataclass(frozen=True)
class PartitionMetrics:
    """Partition-level segmentation metrics for one RWTD sample.

    Attributes:
        miou: Mean IoU across the texture-A and texture-B regions.
        ari: Adjusted Rand Index on the per-pixel partition labels
            ``{background, A-only, B-only, overlap}``.
    """

    miou: float
    ari: float

    def to_prefixed_dict(self, prefix: str) -> dict[str, float]:
        """Flatten this metric group with a stable key prefix."""

        return {
            f"{prefix}_miou": self.miou,
            f"{prefix}_ari": self.ari,
        }


@dataclass(frozen=True)
class MaskSetMetrics:
    """Metrics for a raw automatic-mask set against texture regions.

    Attributes:
        miou: Non-aggregated mean IoU using one-to-one IoU matching between
            ground-truth regions and raw predicted masks.
        ari: Adjusted Rand Index between ground-truth region labels and the
            pixelwise partition induced by raw predicted mask memberships.
        aggregated_miou: Mean IoU after unioning all masks that overlap each ground-truth region.
        region_best_ious: Matched non-aggregated IoU per ground-truth region.
        region_aggregated_ious: Aggregated IoU per ground-truth region.
        region_overlap_counts: Number of predicted masks contributing to each aggregated region.
        num_predicted_masks: Total number of predicted masks in the set.
    """

    miou: float
    ari: float
    aggregated_miou: float
    region_best_ious: tuple[float, ...]
    region_aggregated_ious: tuple[float, ...]
    region_overlap_counts: tuple[int, ...]
    num_predicted_masks: int


@dataclass(frozen=True)
class OfficialTextureSamMetrics:
    """Metrics that mirror the released TextureSAM RWTD evaluation scripts.

    Attributes:
        miou: Non-aggregated mean IoU from the official raw-mask script.
        ari: Non-aggregated ARI from the official raw-mask script.
        aggregated_miou: Aggregated mean IoU from the official aggregation script.
        aggregated_miou_original: Aggregated mean IoU under the original binary labels.
        aggregated_miou_inverted: Aggregated mean IoU under inverted binary labels.
        region_average_ious: Per-label average IoU over all overlapping raw masks.
        region_average_aris: Per-label average ARI over all overlapping raw masks.
        region_overlap_counts: Number of raw predicted masks that overlap each GT label.
        label_values: Sorted label values found in the GT map.
        num_counted_regions: Number of GT labels that contributed to the raw dataset average.
        num_predicted_masks: Total number of raw masks produced for the sample.
    """

    miou: float
    ari: float
    aggregated_miou: float
    aggregated_miou_original: float
    aggregated_miou_inverted: float
    region_average_ious: tuple[float, ...]
    region_average_aris: tuple[float, ...]
    region_overlap_counts: tuple[int, ...]
    label_values: tuple[int, ...]
    num_counted_regions: int
    num_predicted_masks: int


def build_canonical_evaluation_fields(
    *,
    miou: float,
    ari: float,
    evaluation_view: str,
) -> dict[str, str | float]:
    """Build the repo-wide default evaluation-contract fields."""

    return {
        "evaluation_contract": CANONICAL_EVALUATION_CONTRACT,
        "evaluation_view": evaluation_view,
        CANONICAL_PRIMARY_METRIC: float(miou),
        CANONICAL_SECONDARY_METRIC: float(ari),
    }


def compute_binary_metrics(prediction: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    """Compute IoU, Dice, precision, and recall for two boolean masks."""

    pred = _validate_boolean_mask(prediction, "prediction")
    gt = _validate_boolean_mask(target, "target")
    _validate_same_shape(pred, gt)

    matched = int(np.logical_and(pred, gt).sum())
    pred_positive = int(pred.sum())
    target_positive = int(gt.sum())
    union = pred_positive + target_positive - matched

    if pred_positive == 0 and target_positive == 0:
        return BinaryMetrics(
            iou=1.0,
            dice=1.0,
            precision=1.0,
            recall=1.0,
            predicted_positive=0,
            target_positive=0,
            matched_positive=0,
        )

    precision = matched / pred_positive if pred_positive else 0.0
    recall = matched / target_positive if target_positive else 0.0
    dice = (2.0 * matched) / (pred_positive + target_positive) if (pred_positive + target_positive) else 1.0
    iou = matched / union if union else 1.0
    return BinaryMetrics(
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        predicted_positive=pred_positive,
        target_positive=target_positive,
        matched_positive=matched,
    )


def average_binary_metrics(metrics_list: Iterable[BinaryMetrics]) -> BinaryMetrics:
    """Macro-average a sequence of metric groups across the scalar fields."""

    metrics = tuple(metrics_list)
    if not metrics:
        raise ValueError("Cannot average an empty metric list.")

    return BinaryMetrics(
        iou=float(np.mean([item.iou for item in metrics])),
        dice=float(np.mean([item.dice for item in metrics])),
        precision=float(np.mean([item.precision for item in metrics])),
        recall=float(np.mean([item.recall for item in metrics])),
        predicted_positive=int(sum(item.predicted_positive for item in metrics)),
        target_positive=int(sum(item.target_positive for item in metrics)),
        matched_positive=int(sum(item.matched_positive for item in metrics)),
    )


def compute_partition_metrics(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> PartitionMetrics:
    """Compute mIoU and ARI for a predicted RWTD texture partition.

    Inputs:
        prediction_a: Boolean mask of shape ``(height, width)`` for texture A.
        prediction_b: Boolean mask of shape ``(height, width)`` for texture B.
        target_a: Boolean mask of shape ``(height, width)`` for texture A.
        target_b: Boolean mask of shape ``(height, width)`` for texture B.

    Returns:
        ``PartitionMetrics`` where ``miou`` is the mean IoU of the texture-A and
        texture-B masks and ``ari`` is computed on 4-state per-pixel labels
        ``{background, A-only, B-only, overlap}``.
    """

    texture_a_metrics = compute_binary_metrics(prediction_a, target_a)
    texture_b_metrics = compute_binary_metrics(prediction_b, target_b)
    prediction_labels = encode_partition_labels(prediction_a, prediction_b)
    target_labels = encode_partition_labels(target_a, target_b)
    return PartitionMetrics(
        miou=float(np.mean([texture_a_metrics.iou, texture_b_metrics.iou])),
        ari=compute_adjusted_rand_index(prediction_labels, target_labels),
    )


def compute_mask_set_metrics(
    prediction_masks: Sequence[np.ndarray],
    ground_truth_regions: Sequence[np.ndarray],
) -> MaskSetMetrics:
    """Compute TextureSAM-style metrics for a raw SAM mask set.

    Inputs:
        prediction_masks: Ordered boolean masks of shape ``(height, width)``.
        ground_truth_regions: Boolean region masks of shape ``(height, width)``.

    Returns:
        ``MaskSetMetrics`` with non-aggregated mIoU, aggregated mIoU, and ARI.
    """

    gt_regions = _validate_mask_sequence(ground_truth_regions, "ground_truth_regions")
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=gt_regions[0].shape)

    matched_ious = match_masks_one_to_one(pred_masks, gt_regions)

    aggregated_masks, overlap_counts = aggregate_masks_by_regions(pred_masks, gt_regions)
    aggregated_ious = tuple(
        compute_binary_metrics(aggregated_mask, region_mask).iou
        for aggregated_mask, region_mask in zip(aggregated_masks, gt_regions)
    )

    target_labels = label_map_from_regions(gt_regions)
    prediction_labels = label_map_from_mask_memberships(pred_masks, shape=gt_regions[0].shape)
    return MaskSetMetrics(
        miou=float(np.mean(matched_ious)) if matched_ious else 0.0,
        ari=compute_adjusted_rand_index(prediction_labels, target_labels),
        aggregated_miou=float(np.mean(aggregated_ious)) if aggregated_ious else 0.0,
        region_best_ious=tuple(float(value) for value in matched_ious),
        region_aggregated_ious=tuple(float(value) for value in aggregated_ious),
        region_overlap_counts=overlap_counts,
        num_predicted_masks=len(pred_masks),
    )


def compute_canonical_partition_metrics_for_mask_set(
    prediction_masks: Sequence[np.ndarray],
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[PartitionMetrics, tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Collapse a raw mask set into the canonical two-mask evaluation view.

    The default repository-wide evaluator follows the ArchiTexture-style binary
    comparison contract. For raw automatic-mask sets, this means unioning every
    predicted mask with non-empty overlap against each GT region and then
    scoring the resulting two-mask partition with the shared partition metric.
    """

    aggregated_masks, overlap_counts = aggregate_masks_by_regions(
        prediction_masks,
        (target_a, target_b),
    )
    partition = compute_partition_metrics(
        aggregated_masks[0],
        aggregated_masks[1],
        target_a,
        target_b,
    )
    return partition, (aggregated_masks[0], aggregated_masks[1]), (overlap_counts[0], overlap_counts[1])


def compute_canonical_partition_metrics_for_official_label_map(
    prediction_masks: Sequence[np.ndarray],
    label_mask: np.ndarray,
) -> tuple[PartitionMetrics, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Build the canonical binary-partition view for the official Kaust256 path."""

    indexed_label_map, label_values = reindex_label_map(label_mask)
    if len(label_values) != 2:
        raise ValueError(
            "The canonical ArchiTexture-style evaluator expects binary official labels, "
            f"got {label_values}."
        )
    pred_label_map_original = rasterize_matched_masks(prediction_masks, indexed_label_map)
    pred_label_map_inverted = rasterize_matched_masks(prediction_masks, invert_binary_label_map(indexed_label_map))
    prediction_a = pred_label_map_original == 1
    prediction_b = pred_label_map_inverted == 1
    target_a = indexed_label_map == 1
    target_b = indexed_label_map == 0
    partition = compute_partition_metrics(prediction_a, prediction_b, target_a, target_b)
    return partition, (prediction_a, prediction_b), (target_a, target_b), label_values


def compute_official_texturesam_metrics(
    prediction_masks: Sequence[np.ndarray],
    label_mask: np.ndarray,
) -> OfficialTextureSamMetrics:
    """Compute the released TextureSAM RWTD metrics for one binary label map.

    Inputs:
        prediction_masks: Ordered boolean SAM masks of shape ``(height, width)``.
        label_mask: Integer binary GT label map of shape ``(height, width)``.

    Returns:
        ``OfficialTextureSamMetrics`` matching the public TextureSAM scripts:
        raw ``mIoU``/``ARI`` average over all overlapping raw masks per GT label,
        plus aggregated ``mIoU`` with binary-label inversion averaging.
    """

    indexed_label_map, label_values = reindex_label_map(label_mask)
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=indexed_label_map.shape)
    gt_regions = tuple(indexed_label_map == label_index for label_index in range(len(label_values)))

    region_average_ious: list[float] = []
    region_average_aris: list[float] = []
    region_overlap_counts: list[int] = []
    total_iou = 0.0
    total_ari = 0.0
    num_counted_regions = 0

    for region_mask in gt_regions:
        overlapping_masks = [
            pred_mask
            for pred_mask in pred_masks
            if np.logical_and(pred_mask, region_mask).any()
        ]
        region_overlap_counts.append(len(overlapping_masks))
        iou_values = [compute_binary_metrics(pred_mask, region_mask).iou for pred_mask in overlapping_masks]
        ari_values = [
            compute_adjusted_rand_index(pred_mask.astype(np.int32), region_mask.astype(np.int32))
            for pred_mask in overlapping_masks
        ]
        avg_iou = float(np.mean(iou_values)) if iou_values else 0.0
        avg_ari = float(np.mean(ari_values)) if ari_values else 0.0
        region_average_ious.append(avg_iou)
        region_average_aris.append(avg_ari)
        if iou_values:
            total_iou += avg_iou
            total_ari += avg_ari
            num_counted_regions += 1

    aggregated_original = compute_official_aggregated_miou(pred_masks, indexed_label_map)
    if len(label_values) == 2:
        aggregated_inverted = compute_official_aggregated_miou(pred_masks, invert_binary_label_map(indexed_label_map))
        aggregated_miou = float((aggregated_original + aggregated_inverted) / 2.0)
    else:
        aggregated_inverted = aggregated_original
        aggregated_miou = aggregated_original

    return OfficialTextureSamMetrics(
        miou=float(total_iou / num_counted_regions) if num_counted_regions else 0.0,
        ari=float(total_ari / num_counted_regions) if num_counted_regions else 0.0,
        aggregated_miou=aggregated_miou,
        aggregated_miou_original=aggregated_original,
        aggregated_miou_inverted=aggregated_inverted,
        region_average_ious=tuple(region_average_ious),
        region_average_aris=tuple(region_average_aris),
        region_overlap_counts=tuple(region_overlap_counts),
        label_values=label_values,
        num_counted_regions=num_counted_regions,
        num_predicted_masks=len(pred_masks),
    )


def aggregate_masks_by_regions(
    prediction_masks: Sequence[np.ndarray],
    ground_truth_regions: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Union all predicted masks with non-empty overlap against each ground-truth region."""

    gt_regions = _validate_mask_sequence(ground_truth_regions, "ground_truth_regions")
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=gt_regions[0].shape)

    aggregated_masks: list[np.ndarray] = []
    overlap_counts: list[int] = []
    for region_mask in gt_regions:
        overlapping_masks = [
            pred_mask
            for pred_mask in pred_masks
            if np.logical_and(pred_mask, region_mask).any()
        ]
        overlap_counts.append(len(overlapping_masks))
        if not overlapping_masks:
            aggregated_masks.append(np.zeros_like(region_mask, dtype=bool))
            continue
        aggregated_masks.append(np.logical_or.reduce(overlapping_masks))
    return tuple(aggregated_masks), tuple(overlap_counts)


def compute_official_aggregated_miou(
    prediction_masks: Sequence[np.ndarray],
    indexed_label_map: np.ndarray,
) -> float:
    """Mirror the released TextureSAM aggregated RWTD evaluation script.

    Inputs:
        prediction_masks: Ordered boolean masks of shape ``(height, width)``.
        indexed_label_map: Integer label map with classes remapped to ``0..N-1``.

    Returns:
        The exact scalar produced by the released ``eval_agg_masks.py`` path after
        assigning each predicted mask to the GT label with maximum pixel overlap,
        sorting assigned masks by area, rasterizing only the strictly positive label
        IDs, and then calling the pinned ``torchmetrics==1.6.1`` ``mean_iou`` API
        with the released tensor shape and default arguments.
    """

    label_map = np.asarray(indexed_label_map, dtype=np.int32)
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=label_map.shape)
    pred_label_map = rasterize_matched_masks(pred_masks, label_map)
    return _compute_released_agg_script_mean_iou(pred_label_map, label_map)


def rasterize_matched_masks(
    prediction_masks: Sequence[np.ndarray],
    indexed_label_map: np.ndarray,
) -> np.ndarray:
    """Rasterize best-label mask assignments exactly as in TextureSAM's script."""

    label_map = np.asarray(indexed_label_map, dtype=np.int32)
    matched_masks = match_masks_to_label_map(prediction_masks, label_map)
    pred_label_map = np.zeros_like(label_map, dtype=np.int32)
    for matched_mask in matched_masks:
        positive_pixels = matched_mask > 0
        pred_label_map[positive_pixels] = matched_mask[positive_pixels]
    return pred_label_map


def _compute_released_agg_script_mean_iou(
    prediction_label_map: np.ndarray,
    target_label_map: np.ndarray,
) -> float:
    """Replicate the released TextureSAM aggregated-score call exactly.

    The public ``eval_agg_masks.py`` script pins ``torchmetrics==1.6.1`` and calls
    ``mean_iou(pred_label_tensor.unsqueeze(0), label_tensor.unsqueeze(0), num_classes=n_classes)``
    without overriding ``input_format``. Under that pinned API, tensors with shape
    ``(1, height, width)`` are treated as one-hot encoded, so the reduction happens
    row-wise instead of class-wise. This helper intentionally preserves that behavior
    so the official RWTD reproduction path is byte-for-byte compatible with the
    released script output rather than a corrected interpretation of the metric.
    """

    pred = np.asarray(prediction_label_map, dtype=np.int64)
    gt = np.asarray(target_label_map, dtype=np.int64)
    _validate_same_shape(pred, gt)

    batched_pred = pred[None, ...]
    batched_gt = gt[None, ...]
    intersection = np.bitwise_and(batched_pred, batched_gt).sum(axis=2, dtype=np.int64)
    target_sum = batched_gt.sum(axis=2, dtype=np.int64)
    pred_sum = batched_pred.sum(axis=2, dtype=np.int64)
    union = target_sum + pred_sum - intersection

    per_row_values = np.divide(
        intersection.astype(np.float64),
        union.astype(np.float64),
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union != 0,
    )
    reduced = per_row_values.mean(axis=1)
    return float(reduced[0]) if reduced.size else 0.0


def match_masks_to_label_map(
    prediction_masks: Sequence[np.ndarray],
    indexed_label_map: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Assign each predicted mask to the GT label with maximum pixel overlap.

    Inputs:
        prediction_masks: Ordered boolean masks of shape ``(height, width)``.
        indexed_label_map: Integer label map of shape ``(height, width)``.

    Returns:
        Area-sorted integer masks where positive pixels hold the best-matching GT label
        index. Class ``0`` is intentionally left as zeros to mirror the released script.
    """

    label_map = np.asarray(indexed_label_map, dtype=np.int32)
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=label_map.shape)
    label_ids = tuple(int(value) for value in np.unique(label_map))
    matched_masks: list[np.ndarray] = []
    for pred_mask in pred_masks:
        best_label = 0
        max_overlap = 0
        for label_id in label_ids:
            overlap = int(np.logical_and(pred_mask, label_map == label_id).sum())
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = label_id

        matched_mask = np.zeros_like(label_map, dtype=np.int32)
        matched_mask[pred_mask] = best_label
        matched_masks.append(matched_mask)
    matched_masks.sort(key=lambda mask: int(np.count_nonzero(mask)), reverse=True)
    return tuple(matched_masks)


def reindex_label_map(label_map: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Map an arbitrary integer label image to contiguous indices ``0..N-1``."""

    raw_label_map = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(raw_label_map))
    indexed_label_map = np.zeros_like(raw_label_map, dtype=np.int32)
    for new_index, label_value in enumerate(unique_values):
        indexed_label_map[raw_label_map == label_value] = new_index
    return indexed_label_map, unique_values


def invert_binary_label_map(label_map: np.ndarray) -> np.ndarray:
    """Invert a binary indexed label map while keeping labels in ``{0, 1}``."""

    indexed_label_map = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(indexed_label_map))
    if unique_values != (0, 1):
        raise ValueError(
            f"Binary label inversion expects indexed labels (0, 1), got {unique_values}."
        )
    return 1 - indexed_label_map


def boundary_from_region_masks(texture_a_mask: np.ndarray, texture_b_mask: np.ndarray) -> np.ndarray:
    """Derive a boundary map from two region masks using 8-neighbour adjacency."""

    mask_a = _validate_boolean_mask(texture_a_mask, "texture_a_mask")
    mask_b = _validate_boolean_mask(texture_b_mask, "texture_b_mask")
    _validate_same_shape(mask_a, mask_b)

    boundary = np.zeros_like(mask_a, dtype=bool)
    height, width = mask_a.shape
    padded_a = np.pad(mask_a, pad_width=1, mode="constant", constant_values=False)
    padded_b = np.pad(mask_b, pad_width=1, mode="constant", constant_values=False)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted_b = padded_b[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            shifted_a = padded_a[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            boundary |= np.logical_or(mask_a & shifted_b, mask_b & shifted_a)
    return boundary


def compute_boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: int = 0,
) -> BinaryMetrics:
    """Compute relaxed boundary metrics with a configurable pixel tolerance."""

    pred = _validate_boolean_mask(prediction, "prediction")
    gt = _validate_boolean_mask(target, "target")
    _validate_same_shape(pred, gt)
    if tolerance_px < 0:
        raise ValueError(f"Boundary tolerance must be non-negative, got {tolerance_px}.")

    if tolerance_px == 0:
        return compute_binary_metrics(pred, gt)

    pred_dilated = binary_dilate(pred, radius=tolerance_px)
    gt_dilated = binary_dilate(gt, radius=tolerance_px)
    matched_pred = int(np.logical_and(pred, gt_dilated).sum())
    matched_gt = int(np.logical_and(gt, pred_dilated).sum())
    pred_positive = int(pred.sum())
    target_positive = int(gt.sum())

    if pred_positive == 0 and target_positive == 0:
        return BinaryMetrics(
            iou=1.0,
            dice=1.0,
            precision=1.0,
            recall=1.0,
            predicted_positive=0,
            target_positive=0,
            matched_positive=0,
        )

    matched = min(matched_pred, matched_gt)
    precision = matched_pred / pred_positive if pred_positive else 0.0
    recall = matched_gt / target_positive if target_positive else 0.0
    union = pred_positive + target_positive - matched
    dice = (2.0 * matched) / (pred_positive + target_positive) if (pred_positive + target_positive) else 1.0
    iou = matched / union if union else 1.0
    return BinaryMetrics(
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        predicted_positive=pred_positive,
        target_positive=target_positive,
        matched_positive=matched,
    )


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask with a square 8-neighbourhood kernel."""

    result = _validate_boolean_mask(mask, "mask")
    if radius < 0:
        raise ValueError(f"Dilation radius must be non-negative, got {radius}.")
    if radius == 0:
        return result.copy()

    current = result.copy()
    height, width = current.shape
    for _ in range(radius):
        padded = np.pad(current, pad_width=1, mode="constant", constant_values=False)
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbours.append(padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width])
        current = np.logical_or.reduce(neighbours)
    return current


def encode_partition_labels(texture_a_mask: np.ndarray, texture_b_mask: np.ndarray) -> np.ndarray:
    """Encode two boolean masks into a 4-state integer label map.

    Inputs:
        texture_a_mask: Boolean mask of shape ``(height, width)`` for texture A.
        texture_b_mask: Boolean mask of shape ``(height, width)`` for texture B.

    Returns:
        Integer array of shape ``(height, width)`` with labels:
        ``0=background``, ``1=A-only``, ``2=B-only``, ``3=overlap``.
    """

    mask_a = _validate_boolean_mask(texture_a_mask, "texture_a_mask")
    mask_b = _validate_boolean_mask(texture_b_mask, "texture_b_mask")
    _validate_same_shape(mask_a, mask_b)
    return mask_a.astype(np.int64) + 2 * mask_b.astype(np.int64)


def label_map_from_regions(
    regions: Sequence[np.ndarray],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert an ordered region list into a single integer label map.

    Inputs:
        regions: Ordered boolean masks. Earlier masks take precedence when overlaps occur.
        shape: Optional fallback shape for the empty-region case.

    Returns:
        Integer label map of shape ``(height, width)`` with ``0`` as background and
        ``1..N`` as the ordered region IDs.
    """

    masks = _validate_mask_sequence(regions, "regions", shape=shape)
    if not masks:
        if shape is None:
            raise ValueError("A shape must be provided when converting an empty region list.")
        return np.zeros(shape, dtype=np.int32)

    label_map = np.zeros(masks[0].shape, dtype=np.int32)
    assigned = np.zeros(masks[0].shape, dtype=bool)
    for label_id, mask in enumerate(masks, start=1):
        new_pixels = np.logical_and(mask, np.logical_not(assigned))
        label_map[new_pixels] = label_id
        assigned |= new_pixels
    return label_map


def label_map_from_mask_memberships(
    regions: Sequence[np.ndarray],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert a mask set into a partition label map using full mask memberships.

    Inputs:
        regions: Boolean masks of shape ``(height, width)``.
        shape: Optional fallback shape for the empty-mask case.

    Returns:
        Integer label map of shape ``(height, width)`` with ``0`` reserved for
        pixels covered by no masks. Non-zero labels correspond to distinct
        membership signatures across the raw mask set, which preserves
        fragmentation caused by overlapping or nested predictions.
    """

    masks = _validate_mask_sequence(regions, "regions", shape=shape)
    if not masks:
        if shape is None:
            raise ValueError("A shape must be provided when converting an empty region list.")
        return np.zeros(shape, dtype=np.int32)

    membership = np.stack(masks, axis=0).reshape(len(masks), -1).T
    covered = np.any(membership, axis=1)
    labels = np.zeros(membership.shape[0], dtype=np.int32)
    if not np.any(covered):
        return labels.reshape(masks[0].shape)

    packed = np.packbits(membership[covered], axis=1, bitorder="little")
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    labels[covered] = inverse.astype(np.int32) + 1
    return labels.reshape(masks[0].shape)


def match_masks_one_to_one(
    prediction_masks: Sequence[np.ndarray],
    ground_truth_regions: Sequence[np.ndarray],
) -> tuple[float, ...]:
    """Match raw predicted masks to GT regions with a one-to-one IoU objective.

    Inputs:
        prediction_masks: Boolean masks of shape ``(height, width)``.
        ground_truth_regions: Boolean masks of shape ``(height, width)``.

    Returns:
        Tuple of IoU values, one per ground-truth region, where each predicted
        mask may contribute to at most one GT region. Unmatched GT regions
        receive ``0.0``.
    """

    gt_regions = _validate_mask_sequence(ground_truth_regions, "ground_truth_regions")
    pred_masks = _validate_mask_sequence(prediction_masks, "prediction_masks", shape=gt_regions[0].shape)
    if not gt_regions:
        return ()
    if not pred_masks:
        return tuple(0.0 for _ in gt_regions)

    iou_matrix = np.array(
        [
            [compute_binary_metrics(pred_mask, region_mask).iou for region_mask in gt_regions]
            for pred_mask in pred_masks
        ],
        dtype=np.float64,
    )
    num_gt = len(gt_regions)
    num_states = 1 << num_gt
    dp = np.full(num_states, -np.inf, dtype=np.float64)
    prev_state = np.full(num_states, -1, dtype=np.int32)
    prev_pred = np.full(num_states, -1, dtype=np.int32)
    prev_gt = np.full(num_states, -1, dtype=np.int32)
    dp[0] = 0.0

    for pred_index in range(len(pred_masks)):
        next_dp = dp.copy()
        next_prev_state = prev_state.copy()
        next_prev_pred = prev_pred.copy()
        next_prev_gt = prev_gt.copy()
        for mask in range(num_states):
            if not np.isfinite(dp[mask]):
                continue
            for gt_index in range(num_gt):
                if mask & (1 << gt_index):
                    continue
                next_mask = mask | (1 << gt_index)
                candidate = dp[mask] + iou_matrix[pred_index, gt_index]
                if candidate > next_dp[next_mask]:
                    next_dp[next_mask] = candidate
                    next_prev_state[next_mask] = mask
                    next_prev_pred[next_mask] = pred_index
                    next_prev_gt[next_mask] = gt_index
        dp = next_dp
        prev_state = next_prev_state
        prev_pred = next_prev_pred
        prev_gt = next_prev_gt

    best_mask = 0
    best_score = -np.inf
    for mask in range(num_states):
        if not np.isfinite(dp[mask]):
            continue
        score = dp[mask]
        assigned_count = mask.bit_count()
        if score > best_score or (score == best_score and assigned_count > best_mask.bit_count()):
            best_score = score
            best_mask = mask

    matched_ious = [0.0] * num_gt
    current_mask = best_mask
    while current_mask:
        gt_index = int(prev_gt[current_mask])
        pred_index = int(prev_pred[current_mask])
        if gt_index >= 0 and pred_index >= 0:
            matched_ious[gt_index] = float(iou_matrix[pred_index, gt_index])
        current_mask = int(prev_state[current_mask])
        if current_mask < 0:
            break
    return tuple(matched_ious)


def compute_adjusted_rand_index(prediction_labels: np.ndarray, target_labels: np.ndarray) -> float:
    """Compute Adjusted Rand Index for two same-shaped integer label maps.

    Inputs:
        prediction_labels: Integer array of shape ``(height, width)`` or ``(n,)``.
        target_labels: Integer array with the same shape as ``prediction_labels``.

    Returns:
        Scalar ARI in ``[-1, 1]``.
    """

    pred = np.asarray(prediction_labels, dtype=np.int64).reshape(-1)
    gt = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(
            "Prediction and target label maps must have the same flattened shape, "
            f"got {pred.shape} and {gt.shape}."
        )
    if pred.size < 2:
        return 1.0

    pred_ids, pred_inverse = np.unique(pred, return_inverse=True)
    gt_ids, gt_inverse = np.unique(gt, return_inverse=True)
    contingency = np.zeros((pred_ids.size, gt_ids.size), dtype=np.int64)
    np.add.at(contingency, (pred_inverse, gt_inverse), 1)

    sum_comb_c = _comb2(contingency).sum()
    sum_comb_pred = _comb2(contingency.sum(axis=1)).sum()
    sum_comb_gt = _comb2(contingency.sum(axis=0)).sum()
    sum_comb_total = _comb2(np.array([pred.size], dtype=np.int64))[0]

    if sum_comb_total == 0:
        return 1.0

    expected_index = (sum_comb_pred * sum_comb_gt) / sum_comb_total
    max_index = 0.5 * (sum_comb_pred + sum_comb_gt)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    ari = (sum_comb_c - expected_index) / denominator
    return float(ari)


def _comb2(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array * (array - 1.0) / 2.0


def _validate_mask_sequence(
    masks: Sequence[np.ndarray],
    name: str,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, ...]:
    normalized: list[np.ndarray] = []
    expected_shape = shape
    for index, mask in enumerate(masks):
        normalized_mask = _validate_boolean_mask(mask, f"{name}[{index}]")
        if expected_shape is None:
            expected_shape = normalized_mask.shape
        elif normalized_mask.shape != expected_shape:
            raise ValueError(
                f"All masks in {name} must share shape {expected_shape}, got {normalized_mask.shape}."
            )
        normalized.append(normalized_mask)
    if expected_shape is None and shape is not None:
        expected_shape = shape
    return tuple(normalized)


def _validate_boolean_mask(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {array.shape}.")
    return array


def _validate_same_shape(prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target masks must have the same shape, got "
            f"{prediction.shape} and {target.shape}."
        )


@dataclass(frozen=True)
class MultiPartitionMetrics:
    """Permutation-invariant metrics for a multi-region label partition.

    Attributes:
        miou: Hungarian-matched mean IoU between predicted and GT labels.
        ari: Adjusted Rand Index on valid pixels.
        nmi: Normalized mutual information on valid pixels.
        count_accuracy: ``1.0`` when the predicted and GT region counts match, else ``0.0``.
        coverage: Fraction of total pixels included in valid-pixel evaluation.
        valid_pixel_count: Number of pixels participating in evaluation.
        total_pixel_count: Total number of pixels in the original label map.
        num_predicted_regions: Number of distinct predicted labels on valid pixels.
        num_target_regions: Number of distinct GT labels on valid pixels.
        matched_ious: IoU values of the padded one-to-one assignment, length ``max(K_pred, K_gt)``.
        matched_pairs: Tuple of ``(pred_label, target_label, iou)`` entries. ``-1`` indicates a padded unmatched slot.
    """

    miou: float
    ari: float
    nmi: float
    count_accuracy: float
    coverage: float
    valid_pixel_count: int
    total_pixel_count: int
    num_predicted_regions: int
    num_target_regions: int
    matched_ious: tuple[float, ...]
    matched_pairs: tuple[tuple[int, int, float], ...]


def compute_multi_partition_metrics(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> MultiPartitionMetrics:
    """Compute permutation-invariant metrics for multi-region label maps.

    Inputs:
        prediction_labels: Integer label map of shape ``(height, width)`` or ``(n,)``.
        target_labels: Integer label map with the same shape as ``prediction_labels``.
        valid_mask: Optional boolean mask selecting evaluable pixels only.

    Returns:
        ``MultiPartitionMetrics`` containing Hungarian-matched mIoU, pixel ARI,
        NMI, and coverage/count diagnostics.
    """

    pred = np.asarray(prediction_labels, dtype=np.int64)
    gt = np.asarray(target_labels, dtype=np.int64)
    if pred.shape != gt.shape:
        raise ValueError(
            "Prediction and target label maps must have the same shape, "
            f"got {pred.shape} and {gt.shape}."
        )

    total_pixel_count = int(pred.size)
    if valid_mask is None:
        valid = np.ones(pred.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != pred.shape:
            raise ValueError(
                f"valid_mask must have shape {pred.shape}, got {valid.shape}."
            )

    flat_valid = valid.reshape(-1)
    valid_pixel_count = int(flat_valid.sum())
    if valid_pixel_count < 1:
        raise ValueError("compute_multi_partition_metrics requires at least one valid pixel.")

    pred_valid = pred.reshape(-1)[flat_valid]
    gt_valid = gt.reshape(-1)[flat_valid]
    pred_values, pred_inverse = np.unique(pred_valid, return_inverse=True)
    gt_values, gt_inverse = np.unique(gt_valid, return_inverse=True)

    contingency = np.zeros((pred_values.size, gt_values.size), dtype=np.int64)
    np.add.at(contingency, (pred_inverse, gt_inverse), 1)
    pred_sizes = contingency.sum(axis=1, keepdims=True)
    gt_sizes = contingency.sum(axis=0, keepdims=True)
    union = pred_sizes + gt_sizes - contingency
    iou_matrix = np.divide(
        contingency.astype(np.float64),
        np.clip(union.astype(np.float64), a_min=1.0, a_max=None),
        out=np.zeros_like(contingency, dtype=np.float64),
        where=union > 0,
    )

    assignment = _maximize_iou_assignment(iou_matrix)
    matched_ious: list[float] = []
    matched_pairs: list[tuple[int, int, float]] = []
    for pred_index, gt_index, score in assignment:
        pred_label = int(pred_values[pred_index]) if pred_index >= 0 and pred_index < pred_values.size else -1
        gt_label = int(gt_values[gt_index]) if gt_index >= 0 and gt_index < gt_values.size else -1
        matched_ious.append(float(score))
        matched_pairs.append((pred_label, gt_label, float(score)))

    miou = float(np.mean(matched_ious)) if matched_ious else 0.0
    ari = compute_adjusted_rand_index(pred_valid, gt_valid)
    nmi = compute_normalized_mutual_information(pred_valid, gt_valid)
    return MultiPartitionMetrics(
        miou=miou,
        ari=ari,
        nmi=nmi,
        count_accuracy=float(pred_values.size == gt_values.size),
        coverage=float(valid_pixel_count / max(total_pixel_count, 1)),
        valid_pixel_count=valid_pixel_count,
        total_pixel_count=total_pixel_count,
        num_predicted_regions=int(pred_values.size),
        num_target_regions=int(gt_values.size),
        matched_ious=tuple(float(value) for value in matched_ious),
        matched_pairs=tuple(matched_pairs),
    )


def compute_normalized_mutual_information(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    """Compute normalized mutual information for two integer label vectors/maps."""

    pred = np.asarray(prediction_labels, dtype=np.int64).reshape(-1)
    gt = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(
            "Prediction and target label maps must have the same flattened shape, "
            f"got {pred.shape} and {gt.shape}."
        )
    if pred.size < 1:
        return 1.0

    pred_values, pred_inverse = np.unique(pred, return_inverse=True)
    gt_values, gt_inverse = np.unique(gt, return_inverse=True)
    contingency = np.zeros((pred_values.size, gt_values.size), dtype=np.float64)
    np.add.at(contingency, (pred_inverse, gt_inverse), 1.0)
    total = float(contingency.sum())
    if total <= 0.0:
        return 1.0

    joint = contingency / total
    pred_marginal = joint.sum(axis=1, keepdims=True)
    gt_marginal = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = float(
        np.sum(joint[nonzero] * np.log(joint[nonzero] / (pred_marginal @ gt_marginal)[nonzero]))
    )
    pred_entropy = float(-np.sum(pred_marginal[pred_marginal > 0] * np.log(pred_marginal[pred_marginal > 0])))
    gt_entropy = float(-np.sum(gt_marginal[gt_marginal > 0] * np.log(gt_marginal[gt_marginal > 0])))
    denominator = pred_entropy + gt_entropy
    if denominator <= 1e-12:
        return 1.0
    return float((2.0 * mutual_information) / denominator)


def boundary_from_label_map(label_map: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute a thin multi-label boundary mask from integer labels."""

    labels = np.asarray(label_map, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(f"label_map must be 2D, got shape {labels.shape}.")
    if valid_mask is None:
        valid = np.ones(labels.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != labels.shape:
            raise ValueError(f"valid_mask must match label_map shape {labels.shape}, got {valid.shape}.")

    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal_valid = np.logical_and(valid[:, 1:], valid[:, :-1])
    horizontal_diff = np.logical_and(horizontal_valid, labels[:, 1:] != labels[:, :-1])
    boundary[:, 1:] |= horizontal_diff
    boundary[:, :-1] |= horizontal_diff

    vertical_valid = np.logical_and(valid[1:, :], valid[:-1, :])
    vertical_diff = np.logical_and(vertical_valid, labels[1:, :] != labels[:-1, :])
    boundary[1:, :] |= vertical_diff
    boundary[:-1, :] |= vertical_diff
    boundary &= valid
    return boundary


def _maximize_iou_assignment(score_matrix: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"score_matrix must be 2D, got shape {matrix.shape}.")
    num_rows, num_cols = matrix.shape
    size = max(int(num_rows), int(num_cols))
    if size == 0:
        return ()

    padded = np.zeros((size, size), dtype=np.float64)
    padded[:num_rows, :num_cols] = matrix
    dp: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row_index in range(size):
        next_dp: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, path) in dp.items():
            for col_index in range(size):
                if mask & (1 << col_index):
                    continue
                next_mask = mask | (1 << col_index)
                candidate_score = float(score + padded[row_index, col_index])
                candidate_path = path + (int(col_index),)
                current = next_dp.get(next_mask)
                if current is None:
                    next_dp[next_mask] = (candidate_score, candidate_path)
                    continue
                current_score, current_path = current
                if (candidate_score > current_score + 1e-12) or (
                    abs(candidate_score - current_score) <= 1e-12 and candidate_path < current_path
                ):
                    next_dp[next_mask] = (candidate_score, candidate_path)
        dp = next_dp

    full_mask = (1 << size) - 1
    best_score, best_path = dp[full_mask]
    del best_score
    assignment: list[tuple[int, int, float]] = []
    for row_index, col_index in enumerate(best_path):
        pred_index = row_index if row_index < num_rows else -1
        gt_index = col_index if col_index < num_cols else -1
        score = float(padded[row_index, col_index])
        assignment.append((pred_index, gt_index, score))
    return tuple(assignment)
