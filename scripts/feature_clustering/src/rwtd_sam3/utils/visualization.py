"""Visualization helpers for single-sample predictions and evaluation previews.

This module renders the PNG panels written by the prompt-conditioned SAM 3,
SAM-2, SAM-3 automatic-mask, and ArchiTexture evaluation paths. It is the shared
presentation layer that converts boolean masks, prompt metadata, and metric
summaries into human-readable preview images plus caption/footer text.

Primary entrypoints:
- ``render_prediction_panel()`` and ``save_prediction_panel()``: prompt-
  conditioned RWTD previews.
- ``render_automatic_mask_panel()`` and related save helpers: proposal-based
  previews for SAM-2 and SAM-3 automatic-mask runs.
- ``build_visual_caption()`` and ``build_visual_footer_lines()``: stable textual
  metadata reused by the JSONL visual manifests.

Inputs are decoded sample objects plus boolean masks of shape ``(height,
width)``. Outputs are ``PIL.Image.Image`` panels saved directly to disk by the
calling evaluation modules.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rwtd_sam3.data.rwtd import RwtdSample
from rwtd_sam3.eval.metrics import boundary_from_label_map, boundary_from_region_masks, label_map_from_regions
from rwtd_sam3.models.sam3_boundary_refine_sweep_runner import (
    BOUNDARY_REFINE_SWEEP_VARIANT_IDS,
    BOUNDARY_REFINE_SWEEP_VARIANT_LABELS,
    BoundaryRefineSweepResult,
)

_TEXT_PADDING_X = 10
_TEXT_PADDING_Y = 8
_TEXT_LINE_SPACING = 4
_FONT = ImageFont.load_default()
_SEGMENT_PALETTE = (
    (230, 57, 70),
    (29, 78, 216),
    (46, 125, 50),
    (244, 162, 97),
    (106, 76, 147),
    (20, 184, 166),
    (245, 158, 11),
    (236, 72, 153),
    (100, 116, 139),
    (132, 204, 22),
    (14, 165, 233),
    (249, 115, 22),
)


def _feature_map_to_numpy(feature_map: np.ndarray) -> np.ndarray:
    """Convert a torch-like tensor or array to a float32 numpy array."""

    if hasattr(feature_map, "detach"):
        feature_map = feature_map.detach()
    if hasattr(feature_map, "cpu"):
        feature_map = feature_map.cpu()
    if hasattr(feature_map, "numpy"):
        feature_map = feature_map.numpy()
    array = np.asarray(feature_map, dtype=np.float32)
    if array.ndim == 4 and int(array.shape[0]) == 1:
        array = array[0]
    return array


def _pca_to_rgb(
    vectors: np.ndarray,
    *,
    height: int,
    width: int,
    robust_percentiles: tuple[float, float],
) -> np.ndarray:
    """Project pooled feature vectors (N,C) to (H,W,3) uint8 via PCA (SVD)."""

    flat = np.asarray(vectors, dtype=np.float32)
    if flat.ndim != 2:
        raise ValueError(f"Expected (N,C) vectors, got shape={flat.shape}")
    if flat.shape[0] != int(height * width):
        raise ValueError(f"Expected N=H*W={height*width}, got N={flat.shape[0]}")

    centered = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:3].T

    lo, hi = robust_percentiles
    rgb = np.empty_like(projected, dtype=np.float32)
    for ch in range(3):
        channel = projected[:, ch]
        p_lo = float(np.percentile(channel, lo))
        p_hi = float(np.percentile(channel, hi))
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or abs(p_hi - p_lo) <= 1e-12:
            rgb[:, ch] = 0.5
        else:
            rgb[:, ch] = np.clip((channel - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    return (rgb.reshape(height, width, 3) * 255.0).round().astype(np.uint8)


def render_pooled_feature_pca_overlay(
    sample_image: Image.Image,
    pooled_feature_map: np.ndarray,
    *,
    alpha: float = 0.5,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> Image.Image:
    """Render pooled features as PCA-RGB and overlay the map on the input image."""

    feature_map = _feature_map_to_numpy(pooled_feature_map)
    if feature_map.ndim != 3:
        raise ValueError(f"Expected pooled feature map with shape (C,H,W), got {feature_map.shape}")

    channels, height, width = feature_map.shape
    pooled_vectors = feature_map.reshape(channels, height * width).T
    pca_rgb = _pca_to_rgb(
        pooled_vectors,
        height=int(height),
        width=int(width),
        robust_percentiles=robust_percentiles,
    )
    pca_image = Image.fromarray(pca_rgb, mode="RGB").resize(
        sample_image.size,
        resample=Image.Resampling.NEAREST,
    )
    return Image.blend(sample_image.convert("RGB"), pca_image.convert("RGB"), float(alpha))


def save_pooled_feature_pca_overlay(
    output_path: str | Path,
    sample_image: Image.Image,
    pooled_feature_map: np.ndarray,
    *,
    alpha: float = 0.5,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save the pooled feature PCA overlay to disk."""

    overlay = render_pooled_feature_pca_overlay(
        sample_image=sample_image,
        pooled_feature_map=pooled_feature_map,
        alpha=alpha,
        robust_percentiles=robust_percentiles,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)
    return path


def save_prediction_panel(
    output_path: str | Path,
    sample: RwtdSample,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save an Input/GT/Pred triptych for one RWTD sample."""

    panel = render_prediction_panel(
        sample=sample,
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def render_prediction_panel(
    sample: RwtdSample,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a single visualization panel with wrapped footer captions.

    Inputs:
        sample: RWTD sample with RGB image and prompt metadata.
        prediction_a: Boolean mask of shape ``(height, width)`` for texture A.
        prediction_b: Boolean mask of shape ``(height, width)`` for texture B.
        protocol: Prompting protocol label such as ``text`` or ``oracle_points``.
        metric_summary: Optional single-line metric summary rendered in the footer.

    Returns:
        ``PIL.Image.Image`` containing the input, ground-truth, and prediction panels plus
        a wrapped footer with full prompt text and metric context.
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), "Input")
    gt_overlay = render_overlay(
        sample.image,
        sample.texture_a_mask,
        sample.texture_b_mask,
        sample.boundary_mask,
    )
    gt_panel = _annotate_panel(gt_overlay, "GT | A=red B=blue boundary=white")

    pred_boundary = boundary_from_region_masks(prediction_a, prediction_b)
    pred_overlay = render_overlay(sample.image, prediction_a, prediction_b, pred_boundary)
    pred_panel = _annotate_panel(pred_overlay, f"Pred | protocol={protocol}")

    panel = _concatenate_horizontally([input_panel, gt_panel, pred_panel])
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def save_automatic_mask_panel(
    output_path: str | Path,
    sample: RwtdSample,
    prediction_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    protocol: str,
    aggregated_prediction_a: np.ndarray,
    aggregated_prediction_b: np.ndarray,
    metric_summary: str | None = None,
    raw_panel_title: str | None = None,
    aggregated_panel_title: str | None = None,
) -> Path:
    """Render and save an automatic-mask panel with both raw and aggregated predictions."""

    panel = render_automatic_mask_panel(
        sample=sample,
        prediction_masks=prediction_masks,
        protocol=protocol,
        aggregated_prediction_a=aggregated_prediction_a,
        aggregated_prediction_b=aggregated_prediction_b,
        metric_summary=metric_summary,
        raw_panel_title=raw_panel_title,
        aggregated_panel_title=aggregated_panel_title,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_feature_mask_panel(
    output_path: str | Path,
    sample: RwtdSample,
    dense_prediction_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    selection_mode: str,
    selected_pair_masks: tuple[np.ndarray, np.ndarray],
    selected_pair_ids: tuple[int, int | None],
    coarse_margin: np.ndarray,
    final_prediction_a: np.ndarray,
    final_prediction_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the extended feature-mask inspection panel."""

    panel = render_feature_mask_panel(
        sample=sample,
        dense_prediction_masks=dense_prediction_masks,
        selection_mode=selection_mode,
        selected_pair_masks=selected_pair_masks,
        selected_pair_ids=selected_pair_ids,
        coarse_margin=coarse_margin,
        final_prediction_a=final_prediction_a,
        final_prediction_b=final_prediction_b,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_feature_cluster_global_panel(
    output_path: str | Path,
    sample: RwtdSample,
    cluster_label_map: np.ndarray,
    rough_mask_a: np.ndarray,
    rough_mask_b: np.ndarray,
    refinement_applied: bool,
    refined_mask_a: np.ndarray,
    refined_mask_b: np.ndarray,
    chosen_prediction_a: np.ndarray,
    chosen_prediction_b: np.ndarray,
    assignment_used: str,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the global feature-clustering inspection panel."""

    panel = render_feature_cluster_global_panel(
        sample=sample,
        cluster_label_map=cluster_label_map,
        rough_mask_a=rough_mask_a,
        rough_mask_b=rough_mask_b,
        refinement_applied=refinement_applied,
        refined_mask_a=refined_mask_a,
        refined_mask_b=refined_mask_b,
        chosen_prediction_a=chosen_prediction_a,
        chosen_prediction_b=chosen_prediction_b,
        assignment_used=assignment_used,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_mask_prompt_invariance_panel(
    output_path: str | Path,
    sample: RwtdSample,
    raw_prompt_a: np.ndarray,
    raw_prompt_b: np.ndarray,
    pooled_prompt_a: np.ndarray,
    pooled_prompt_b: np.ndarray,
    random_prompt_a: np.ndarray,
    random_prompt_b: np.ndarray,
    raw_final_a: np.ndarray,
    raw_final_b: np.ndarray,
    pooled_final_a: np.ndarray,
    pooled_final_b: np.ndarray,
    random_final_a: np.ndarray,
    random_final_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the prompt-invariance side-experiment panel."""

    panel = render_mask_prompt_invariance_panel(
        sample=sample,
        raw_prompt_a=raw_prompt_a,
        raw_prompt_b=raw_prompt_b,
        pooled_prompt_a=pooled_prompt_a,
        pooled_prompt_b=pooled_prompt_b,
        random_prompt_a=random_prompt_a,
        random_prompt_b=random_prompt_b,
        raw_final_a=raw_final_a,
        raw_final_b=raw_final_b,
        pooled_final_a=pooled_final_a,
        pooled_final_b=pooled_final_b,
        random_final_a=random_final_a,
        random_final_b=random_final_b,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_feature_cluster_coarse_to_fine_global_panel(
    output_path: str | Path,
    sample: RwtdSample,
    level_label_maps: tuple[np.ndarray, ...],
    level_names: tuple[str, ...],
    rough_mask_a: np.ndarray,
    rough_mask_b: np.ndarray,
    refined_mask_a: np.ndarray,
    refined_mask_b: np.ndarray,
    chosen_prediction_a: np.ndarray,
    chosen_prediction_b: np.ndarray,
    assignment_used: str,
    protocol: str,
    multiscale_refinement_applied: bool = True,
    sam_refinement_applied: bool = True,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the coarse-to-fine global feature-clustering inspection panel."""

    panel = render_feature_cluster_coarse_to_fine_global_panel(
        sample=sample,
        level_label_maps=level_label_maps,
        level_names=level_names,
        rough_mask_a=rough_mask_a,
        rough_mask_b=rough_mask_b,
        refined_mask_a=refined_mask_a,
        refined_mask_b=refined_mask_b,
        chosen_prediction_a=chosen_prediction_a,
        chosen_prediction_b=chosen_prediction_b,
        assignment_used=assignment_used,
        multiscale_refinement_applied=multiscale_refinement_applied,
        sam_refinement_applied=sam_refinement_applied,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_feature_cluster_positionality_panel(
    output_path: str | Path,
    sample: RwtdSample,
    raw_label_map: np.ndarray,
    baseline_label_map: np.ndarray,
    projection_removed_label_map: np.ndarray,
    flip_avg_label_map: np.ndarray,
    null_constant_label_map: np.ndarray,
    null_noise_label_map: np.ndarray,
    null_blur_label_map: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the positionality diagnosis panel for the debiased coarse-only variant."""

    panel = render_feature_cluster_positionality_panel(
        sample=sample,
        raw_label_map=raw_label_map,
        baseline_label_map=baseline_label_map,
        projection_removed_label_map=projection_removed_label_map,
        flip_avg_label_map=flip_avg_label_map,
        null_constant_label_map=null_constant_label_map,
        null_noise_label_map=null_noise_label_map,
        null_blur_label_map=null_blur_label_map,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def save_feature_cluster_edge_bias_panel(
    output_path: str | Path,
    sample: RwtdSample,
    raw_label_map: np.ndarray,
    baseline_label_map: np.ndarray,
    flip_avg_label_map: np.ndarray,
    edge_debiased_label_map: np.ndarray,
    flip_avg_plus_edge_debiased_label_map: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the flip-symmetric edge-bias comparison panel."""

    panel = render_feature_cluster_edge_bias_panel(
        sample=sample,
        raw_label_map=raw_label_map,
        baseline_label_map=baseline_label_map,
        flip_avg_label_map=flip_avg_label_map,
        edge_debiased_label_map=edge_debiased_label_map,
        flip_avg_plus_edge_debiased_label_map=flip_avg_plus_edge_debiased_label_map,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def render_automatic_mask_panel(
    sample: RwtdSample,
    prediction_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    protocol: str,
    aggregated_prediction_a: np.ndarray,
    aggregated_prediction_b: np.ndarray,
    metric_summary: str | None = None,
    raw_panel_title: str | None = None,
    aggregated_panel_title: str | None = None,
) -> Image.Image:
    """Build a four-panel automatic-mask visualization.

    Inputs:
        sample: RWTD sample with the source RGB image and ground-truth masks.
        prediction_masks: Ordered boolean masks of shape ``(height, width)`` from the raw
            automatic mask generator output.
        protocol: Evaluation protocol label such as ``sam2_star`` or ``sam3_auto:dense``.
        aggregated_prediction_a: Boolean mask of shape ``(height, width)`` for the
            ground-truth-region aggregation mapped to texture A.
        aggregated_prediction_b: Boolean mask of shape ``(height, width)`` for the
            ground-truth-region aggregation mapped to texture B.
        metric_summary: Optional summary string rendered in the footer.

    Returns:
        ``PIL.Image.Image`` containing input, ground truth, raw categorical mask proposals,
        and aggregated A/B predictions plus a wrapped footer.
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), "Input")
    gt_overlay = render_overlay(
        sample.image,
        sample.texture_a_mask,
        sample.texture_b_mask,
        sample.boundary_mask,
    )
    gt_panel = _annotate_panel(gt_overlay, "GT | A=red B=blue boundary=white")

    raw_overlay = render_categorical_mask_overlay(sample.image, prediction_masks)
    raw_panel = _annotate_panel(
        raw_overlay,
        raw_panel_title or f"Pred raw masks | protocol={protocol} | count={len(prediction_masks)}",
    )

    pred_boundary = boundary_from_region_masks(aggregated_prediction_a, aggregated_prediction_b)
    aggregated_overlay = render_partition_overlay(
        sample.image,
        aggregated_prediction_a,
        aggregated_prediction_b,
        pred_boundary,
    )
    aggregated_panel = _annotate_panel(
        aggregated_overlay,
        aggregated_panel_title or "Pred aggregated | A-only=red B-only=blue overlap=purple contour=white",
    )

    panel = _concatenate_horizontally([input_panel, gt_panel, raw_panel, aggregated_panel])
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_feature_cluster_global_panel(
    sample: RwtdSample,
    cluster_label_map: np.ndarray,
    rough_mask_a: np.ndarray,
    rough_mask_b: np.ndarray,
    refinement_applied: bool,
    refined_mask_a: np.ndarray,
    refined_mask_b: np.ndarray,
    chosen_prediction_a: np.ndarray,
    chosen_prediction_b: np.ndarray,
    assignment_used: str,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build an eight-panel global feature-clustering visualization.

    When ``refinement_applied`` is ``False``, the "output" panels show the raw global
    cluster masks directly so the panel stays faithful to the executed protocol.
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), "Input")
    gt_overlay = render_overlay(
        sample.image,
        sample.texture_a_mask,
        sample.texture_b_mask,
        sample.boundary_mask,
    )
    gt_panel = _annotate_panel(gt_overlay, "GT | A=red B=blue boundary=white")

    cluster_overlay = render_cluster_label_overlay(sample.image, cluster_label_map)
    cluster_panel = _annotate_panel(cluster_overlay, "Global 2-cluster map | labels are unlabeled")

    rough_a_panel = _annotate_panel(
        render_single_mask_overlay(sample.image, rough_mask_a, color=(230, 56, 70)),
        "Rough cluster A",
    )
    rough_b_panel = _annotate_panel(
        render_single_mask_overlay(sample.image, rough_mask_b, color=(55, 120, 235)),
        "Rough cluster B",
    )
    output_a_title = "Refined cluster A" if refinement_applied else "Output cluster A | refine=off"
    output_b_title = "Refined cluster B" if refinement_applied else "Output cluster B | refine=off"
    refined_a_panel = _annotate_panel(
        render_single_mask_overlay(sample.image, refined_mask_a, color=(230, 56, 70)),
        output_a_title,
    )
    refined_b_panel = _annotate_panel(
        render_single_mask_overlay(sample.image, refined_mask_b, color=(55, 120, 235)),
        output_b_title,
    )
    chosen_panel = _annotate_panel(
        render_partition_overlay(
            sample.image,
            chosen_prediction_a,
            chosen_prediction_b,
            boundary_from_region_masks(chosen_prediction_a, chosen_prediction_b),
        ),
        f"Chosen result | {assignment_used}",
    )

    panel = _concatenate_horizontally(
        [
            input_panel,
            gt_panel,
            cluster_panel,
            rough_a_panel,
            rough_b_panel,
            refined_a_panel,
            refined_b_panel,
            chosen_panel,
        ]
    )
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_mask_prompt_invariance_panel(
    sample: RwtdSample,
    raw_prompt_a: np.ndarray,
    raw_prompt_b: np.ndarray,
    pooled_prompt_a: np.ndarray,
    pooled_prompt_b: np.ndarray,
    random_prompt_a: np.ndarray,
    random_prompt_b: np.ndarray,
    raw_final_a: np.ndarray,
    raw_final_b: np.ndarray,
    pooled_final_a: np.ndarray,
    pooled_final_b: np.ndarray,
    random_final_a: np.ndarray,
    random_final_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build an eight-panel prompt-invariance side-experiment visualization."""

    panels = [
        _annotate_panel(sample.image.convert("RGB"), "Input"),
        _annotate_panel(
            render_overlay(
                sample.image,
                sample.texture_a_mask,
                sample.texture_b_mask,
                sample.boundary_mask,
            ),
            "GT | A=red B=blue boundary=white",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                raw_prompt_a,
                raw_prompt_b,
                boundary_from_region_masks(raw_prompt_a, raw_prompt_b),
            ),
            "Dirty prompt | raw coarsest init",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                pooled_prompt_a,
                pooled_prompt_b,
                boundary_from_region_masks(pooled_prompt_a, pooled_prompt_b),
            ),
            "Cleaner prompt | pooled coarsest init",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                random_prompt_a,
                random_prompt_b,
                boundary_from_region_masks(random_prompt_a, random_prompt_b),
            ),
            "Random prompt | control",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                raw_final_a,
                raw_final_b,
                boundary_from_region_masks(raw_final_a, raw_final_b),
            ),
            "Raw-prompt final | aligned to pooled order",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                pooled_final_a,
                pooled_final_b,
                boundary_from_region_masks(pooled_final_a, pooled_final_b),
            ),
            "Pooled-prompt final | reference branch",
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                random_final_a,
                random_final_b,
                boundary_from_region_masks(random_final_a, random_final_b),
            ),
            "Random-prompt final | aligned to pooled order",
        ),
    ]
    panel = _concatenate_horizontally(panels)
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_feature_cluster_coarse_to_fine_global_panel(
    sample: RwtdSample,
    level_label_maps: tuple[np.ndarray, ...],
    level_names: tuple[str, ...],
    rough_mask_a: np.ndarray,
    rough_mask_b: np.ndarray,
    refined_mask_a: np.ndarray,
    refined_mask_b: np.ndarray,
    chosen_prediction_a: np.ndarray,
    chosen_prediction_b: np.ndarray,
    assignment_used: str,
    protocol: str,
    multiscale_refinement_applied: bool = True,
    sam_refinement_applied: bool = True,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a coarse-to-fine preview, with a 3-panel coarse-only shortcut."""

    if len(level_label_maps) != len(level_names):
        raise ValueError(
            "Coarse-to-fine panel expects one level name per level label map, "
            f"got {len(level_label_maps)} maps and {len(level_names)} names."
        )

    panels = [
        _annotate_panel(sample.image.convert("RGB"), "Input"),
        _annotate_panel(
            render_overlay(
                sample.image,
                sample.texture_a_mask,
                sample.texture_b_mask,
                sample.boundary_mask,
            ),
            "GT | A=red B=blue boundary=white",
        ),
    ]

    panels.append(
        _annotate_panel(
            render_cluster_label_overlay(sample.image, level_label_maps[0]),
            f"Coarsest partition | {level_names[0]}",
        )
    )

    if not multiscale_refinement_applied and not sam_refinement_applied:
        panel = _concatenate_horizontally(panels)
        return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))

    for level_index, (label_map, level_name) in enumerate(zip(level_label_maps[1:], level_names[1:]), start=1):
        panels.append(
            _annotate_panel(
                render_cluster_label_overlay(sample.image, label_map),
                f"Level refine {level_index + 1} | {level_name}",
            )
        )

    rough_boundary = boundary_from_region_masks(rough_mask_a, rough_mask_b)
    panels.append(
        _annotate_panel(
            render_partition_overlay(sample.image, rough_mask_a, rough_mask_b, rough_boundary),
            "Final rough partition | before SAM" if sam_refinement_applied else "Final rough partition",
        )
    )
    panels.append(
        _annotate_panel(
            render_single_mask_overlay(sample.image, refined_mask_a, color=(230, 56, 70)),
            "Refined cluster A" if sam_refinement_applied else "Output cluster A | no SAM",
        )
    )
    panels.append(
        _annotate_panel(
            render_single_mask_overlay(sample.image, refined_mask_b, color=(55, 120, 235)),
            "Refined cluster B" if sam_refinement_applied else "Output cluster B | no SAM",
        )
    )
    panels.append(
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                chosen_prediction_a,
                chosen_prediction_b,
                boundary_from_region_masks(chosen_prediction_a, chosen_prediction_b),
            ),
            f"Chosen result | {assignment_used}",
        )
    )

    panel = _concatenate_horizontally(panels)
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_feature_cluster_figure2_panel(
    sample: RwtdSample,
    pooled_feature_pca_overlay: Image.Image,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    *,
    input_title: str = "Input",
    pca_title: str = "Pooled features PCA",
    gt_title: str = "GT overlay",
    output_title: str = "Output",
) -> Image.Image:
    """Build a compact Figure 2-style panel.

    The layout is intentionally narrow and reviewer-friendly:
    input image, pooled-feature PCA overlay, ground-truth overlay, and final
    output overlay. No footer is added so the panel stays short.
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), input_title)
    pca_panel = _annotate_panel(pooled_feature_pca_overlay.convert("RGB"), pca_title)
    gt_overlay = render_overlay(
        sample.image,
        sample.texture_a_mask,
        sample.texture_b_mask,
        sample.boundary_mask,
    )
    gt_panel = _annotate_panel(gt_overlay, gt_title)
    output_boundary = boundary_from_region_masks(prediction_a, prediction_b)
    output_overlay = render_partition_overlay(sample.image, prediction_a, prediction_b, output_boundary)
    output_panel = _annotate_panel(output_overlay, output_title)
    return _concatenate_horizontally([input_panel, pca_panel, gt_panel, output_panel])


def render_feature_cluster_positionality_panel(
    sample: RwtdSample,
    raw_label_map: np.ndarray,
    baseline_label_map: np.ndarray,
    projection_removed_label_map: np.ndarray,
    flip_avg_label_map: np.ndarray,
    null_constant_label_map: np.ndarray,
    null_noise_label_map: np.ndarray,
    null_blur_label_map: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a one-row positionality diagnosis panel for the debiased coarse-only branch."""

    panels = [
        _annotate_panel(sample.image.convert("RGB"), "Input"),
        _annotate_panel(
            render_overlay(
                sample.image,
                sample.texture_a_mask,
                sample.texture_b_mask,
                sample.boundary_mask,
            ),
            "GT | A=red B=blue boundary=white",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, raw_label_map),
            "Raw coarsest cluster | pre-pooling",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, baseline_label_map),
            "Baseline pooled partition | current method",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, projection_removed_label_map),
            "Projection-removed partition | active",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, flip_avg_label_map),
            "Flip-averaged partition | comparison",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, null_constant_label_map),
            "Null test | constant gray",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, null_noise_label_map),
            "Null test | weak noise",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, null_blur_label_map),
            "Null test | strong blur",
        ),
    ]
    panel = _concatenate_horizontally(panels)
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def save_boundary_refine_sweep_panel(
    output_path: str | Path,
    sample,
    sweep: BoundaryRefineSweepResult,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Save the compact stage-B boundary-refine sweep panel."""

    panel = render_boundary_refine_sweep_panel(
        sample=sample,
        sweep=sweep,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


def render_boundary_refine_sweep_panel(
    sample,
    sweep: BoundaryRefineSweepResult,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Render one compact comparison panel for V0-V4 stage-B boundary refinement."""

    top_row = _concatenate_horizontally(
        [
            _annotate_panel(sample.image.convert("RGB"), "Input"),
            _annotate_panel(
                render_overlay(
                    sample.image,
                    sample.texture_a_mask,
                    sample.texture_b_mask,
                    sample.boundary_mask,
                ),
                "GT | A=red B=blue boundary=white",
            ),
            _annotate_panel(
                render_cluster_label_overlay(sample.image, sweep.coarsest_label_map),
                "Coarsest partition | baseline",
            ),
            _annotate_panel(
                render_boolean_mask_overlay(sample.image, sweep.boundary_cell_map, color=(245, 158, 11)),
                "Boundary-touching coarse cells",
            ),
        ]
    )
    variant_panels = []
    for variant_id in BOUNDARY_REFINE_SWEEP_VARIANT_IDS:
        variant = sweep.variants[variant_id]
        variant_panels.append(
            _annotate_panel(
                render_partition_overlay(
                    sample.image,
                    variant.mask_a,
                    variant.mask_b,
                    boundary_from_region_masks(variant.mask_a, variant.mask_b),
                ),
                (
                    f"{BOUNDARY_REFINE_SWEEP_VARIANT_LABELS[variant_id]} | "
                    f"{variant.percent_pixels_changed * 100.0:.1f}% changed"
                ),
            )
        )
    bottom_row = _concatenate_horizontally(variant_panels)
    panel = _concatenate_vertically([top_row, bottom_row])
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_feature_cluster_edge_bias_panel(
    sample: RwtdSample,
    raw_label_map: np.ndarray,
    baseline_label_map: np.ndarray,
    flip_avg_label_map: np.ndarray,
    edge_debiased_label_map: np.ndarray,
    flip_avg_plus_edge_debiased_label_map: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a one-row comparison panel for flip averaging plus symmetric edge de-biasing."""

    panels = [
        _annotate_panel(sample.image.convert("RGB"), "Input"),
        _annotate_panel(
            render_overlay(
                sample.image,
                sample.texture_a_mask,
                sample.texture_b_mask,
                sample.boundary_mask,
            ),
            "GT | A=red B=blue boundary=white",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, raw_label_map),
            "Raw coarsest cluster | no pooling",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, baseline_label_map),
            "Baseline pooled | no mitigation",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, flip_avg_label_map),
            "Flip-averaged pooled | comparison",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, edge_debiased_label_map),
            "Edge-debiased pooled | comparison",
        ),
        _annotate_panel(
            render_cluster_label_overlay(sample.image, flip_avg_plus_edge_debiased_label_map),
            "Flip+edge-debiased pooled | active",
        ),
    ]
    panel = _concatenate_horizontally(panels)
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_feature_cluster_coarse_to_fine_prompt_diagnostics_panel(
    sample: RwtdSample,
    old_prompt_mask_a: np.ndarray,
    old_prompt_mask_b: np.ndarray,
    old_candidates_a: tuple[dict[str, object], ...],
    old_candidates_b: tuple[dict[str, object], ...],
    old_selected_mask_a: np.ndarray,
    old_selected_mask_b: np.ndarray,
    old_selected_candidate_rank_a: int | None,
    old_selected_candidate_rank_b: int | None,
    old_selected_candidate_index_a: int | None,
    old_selected_candidate_index_b: int | None,
    new_prompt_mask_a: np.ndarray,
    new_prompt_mask_b: np.ndarray,
    new_candidates_a: tuple[dict[str, object], ...],
    new_candidates_b: tuple[dict[str, object], ...],
    new_selected_mask_a: np.ndarray,
    new_selected_mask_b: np.ndarray,
    new_selected_candidate_rank_a: int | None,
    new_selected_candidate_rank_b: int | None,
    new_selected_candidate_index_a: int | None,
    new_selected_candidate_index_b: int | None,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a side-by-side prompt-to-SAM diagnostics panel for pooled-init comparisons."""

    old_candidate_a = old_candidates_a[0]
    old_candidate_b = old_candidates_b[0]
    new_candidate_a = new_candidates_a[0]
    new_candidate_b = new_candidates_b[0]
    panels = [
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                old_prompt_mask_a,
                old_prompt_mask_b,
                boundary_from_region_masks(old_prompt_mask_a, old_prompt_mask_b),
            ),
            "Old prompt partition | raw init",
        ),
        _annotate_panel(
            render_single_mask_overlay(sample.image, np.asarray(old_candidate_a["mask"], dtype=bool), color=(230, 56, 70)),
            _format_prompt_candidate_title("Old prompt A top SAM", old_candidate_a),
        ),
        _annotate_panel(
            render_single_mask_overlay(sample.image, np.asarray(old_candidate_b["mask"], dtype=bool), color=(55, 120, 235)),
            _format_prompt_candidate_title("Old prompt B top SAM", old_candidate_b),
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                old_selected_mask_a,
                old_selected_mask_b,
                boundary_from_region_masks(old_selected_mask_a, old_selected_mask_b),
            ),
            _format_selected_prompt_title(
                "Old selected final | raw init",
                rank_a=old_selected_candidate_rank_a,
                rank_b=old_selected_candidate_rank_b,
                index_a=old_selected_candidate_index_a,
                index_b=old_selected_candidate_index_b,
            ),
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                new_prompt_mask_a,
                new_prompt_mask_b,
                boundary_from_region_masks(new_prompt_mask_a, new_prompt_mask_b),
            ),
            "New prompt partition | pooled init",
        ),
        _annotate_panel(
            render_single_mask_overlay(sample.image, np.asarray(new_candidate_a["mask"], dtype=bool), color=(230, 56, 70)),
            _format_prompt_candidate_title("New prompt A top SAM", new_candidate_a),
        ),
        _annotate_panel(
            render_single_mask_overlay(sample.image, np.asarray(new_candidate_b["mask"], dtype=bool), color=(55, 120, 235)),
            _format_prompt_candidate_title("New prompt B top SAM", new_candidate_b),
        ),
        _annotate_panel(
            render_partition_overlay(
                sample.image,
                new_selected_mask_a,
                new_selected_mask_b,
                boundary_from_region_masks(new_selected_mask_a, new_selected_mask_b),
            ),
            _format_selected_prompt_title(
                "New selected final | pooled init",
                rank_a=new_selected_candidate_rank_a,
                rank_b=new_selected_candidate_rank_b,
                index_a=new_selected_candidate_index_a,
                index_b=new_selected_candidate_index_b,
            ),
        ),
    ]
    panel = _concatenate_horizontally(panels)
    return _append_footer(panel, build_visual_footer_lines(sample, f"{protocol}:prompt-diagnostics", metric_summary))


def render_feature_mask_panel(
    sample: RwtdSample,
    dense_prediction_masks: list[np.ndarray] | tuple[np.ndarray, ...],
    selection_mode: str,
    selected_pair_masks: tuple[np.ndarray, np.ndarray],
    selected_pair_ids: tuple[int, int | None],
    coarse_margin: np.ndarray,
    final_prediction_a: np.ndarray,
    final_prediction_b: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build a six-panel feature-mask visualization.

    Inputs:
        sample: RWTD sample with the source RGB image and ground-truth masks.
        dense_prediction_masks: Dense automatic-mask proposals of shape ``(height, width)``.
        selection_mode: ``"pair"`` for dense-pair mode or ``"single_mask_complement"``
            when the second support region is synthesized locally.
        selected_pair_masks: Support masks used to build the coarse prompt.
        selected_pair_ids: Support-region ids aligned with ``selected_pair_masks``. The
            second id is ``None`` for synthetic local complements.
        coarse_margin: Dense feature-similarity margin for the first selected region.
        final_prediction_a: Final raw feature-mask prediction for texture A.
        final_prediction_b: Final raw feature-mask prediction for texture B.
        protocol: Evaluation protocol label such as ``sam3_auto:feature_mask``.
        metric_summary: Optional summary string rendered in the footer.

    Returns:
        ``PIL.Image.Image`` containing input, ground truth, dense proposals, selected
        pair, coarse feature prior, and final aggregated output plus a wrapped footer.
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), "Input")
    gt_overlay = render_overlay(
        sample.image,
        sample.texture_a_mask,
        sample.texture_b_mask,
        sample.boundary_mask,
    )
    gt_panel = _annotate_panel(gt_overlay, "GT | A=red B=blue boundary=white")

    raw_overlay = render_categorical_mask_overlay(sample.image, dense_prediction_masks)
    raw_panel = _annotate_panel(raw_overlay, f"Dense proposals | count={len(dense_prediction_masks)}")

    pair_overlay = render_pair_overlay(
        sample.image,
        selected_pair_masks[0],
        selected_pair_masks[1],
    )
    if selection_mode == "single_mask_complement":
        pair_title = f"Single-mask complement | seed={selected_pair_ids[0]} bg=local"
    else:
        pair_title = f"Selected pair | ids={selected_pair_ids[0]},{selected_pair_ids[1]}"
    pair_panel = _annotate_panel(
        pair_overlay,
        pair_title,
    )

    coarse_panel = _annotate_panel(
        render_margin_overlay(sample.image, coarse_margin),
        "Coarse feature prior | A=red B=blue",
    )

    pred_boundary = boundary_from_region_masks(final_prediction_a, final_prediction_b)
    aggregated_overlay = render_partition_overlay(
        sample.image,
        final_prediction_a,
        final_prediction_b,
        pred_boundary,
    )
    aggregated_panel = _annotate_panel(
        aggregated_overlay,
        "Refined output | raw final A/B partition",
    )

    panel = _concatenate_horizontally(
        [input_panel, gt_panel, raw_panel, pair_panel, coarse_panel, aggregated_panel]
    )
    return _append_footer(panel, build_visual_footer_lines(sample, protocol, metric_summary))


def render_overlay(
    image: Image.Image,
    texture_a_mask: np.ndarray,
    texture_b_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> Image.Image:
    """Overlay texture A, texture B, and boundary masks onto an RGB image."""

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    composed = _blend_mask(composed, texture_a_mask, color=(230, 56, 70), alpha=0.40)
    composed = _blend_mask(composed, texture_b_mask, color=(55, 120, 235), alpha=0.40)
    composed = _blend_mask(composed, boundary_mask, color=(255, 255, 255), alpha=0.80)
    return Image.fromarray(composed, mode="RGB")


def render_categorical_mask_overlay(
    image: Image.Image,
    masks: list[np.ndarray] | tuple[np.ndarray, ...],
    alpha: float = 0.55,
) -> Image.Image:
    """Overlay an ordered mask set with a distinct color per visible segment.

    Inputs:
        image: Base RGB image.
        masks: Ordered boolean masks of shape ``(height, width)``. Earlier masks take
            precedence where masks overlap.
        alpha: Blend factor for the categorical overlay.

    Returns:
        ``PIL.Image.Image`` with the raw automatic-mask proposals colorized by segment ID.
    """

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not masks:
        return Image.fromarray(base.copy(), mode="RGB")

    label_map = label_map_from_regions(masks, shape=(base.shape[0], base.shape[1]))
    composed = base.astype(np.float32)
    for label_id in range(1, int(label_map.max()) + 1):
        color = np.array(_segment_color(label_id - 1), dtype=np.float32)
        mask = label_map == label_id
        composed[mask] = (1.0 - alpha) * composed[mask] + alpha * color
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")


def render_cluster_label_overlay(image: Image.Image, label_map: np.ndarray) -> Image.Image:
    """Overlay a binary cluster label map with fixed red/blue colors."""

    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise ValueError(f"Binary cluster label overlay expects labels (0, 1), got {unique_values}.")

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    composed = _blend_mask(composed, labels == 0, color=(230, 56, 70), alpha=0.42)
    composed = _blend_mask(composed, labels == 1, color=(55, 120, 235), alpha=0.42)
    boundary = boundary_from_region_masks(labels == 0, labels == 1)
    composed = _blend_mask(composed, _mask_outline(boundary), color=(255, 255, 255), alpha=0.95)
    return Image.fromarray(composed, mode="RGB")


def render_native_cluster_label_map(label_map: np.ndarray, scale: int = 8) -> Image.Image:
    """Render a raw binary cluster label map without the underlying RGB image."""

    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise ValueError(f"Native cluster-map render expects labels (0, 1), got {unique_values}.")

    native = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    native[labels == 0] = (230, 56, 70)
    native[labels == 1] = (55, 120, 235)
    image = Image.fromarray(native, mode="RGB")
    if scale <= 1:
        return image
    return image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)


def render_partition_overlay(
    image: Image.Image,
    texture_a_mask: np.ndarray,
    texture_b_mask: np.ndarray,
    boundary_mask: np.ndarray,
) -> Image.Image:
    """Overlay an A/B partition with explicit overlap and a thin contour.

    Inputs:
        image: Base RGB image.
        texture_a_mask: Boolean mask of shape ``(height, width)`` for texture A.
        texture_b_mask: Boolean mask of shape ``(height, width)`` for texture B.
        boundary_mask: Boolean mask of shape ``(height, width)`` used only for a thin
            visual contour, not as a filled region.

    Returns:
        ``PIL.Image.Image`` where ``A-only`` is red, ``B-only`` is blue, ``A∩B`` is
        purple, and the contour is a thin white outline.
    """

    mask_a = np.asarray(texture_a_mask, dtype=bool)
    mask_b = np.asarray(texture_b_mask, dtype=bool)
    mask_boundary = np.asarray(boundary_mask, dtype=bool)
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)

    composed = base.copy()
    a_only = np.logical_and(mask_a, np.logical_not(mask_b))
    b_only = np.logical_and(mask_b, np.logical_not(mask_a))
    overlap = np.logical_and(mask_a, mask_b)
    contour = _mask_outline(mask_boundary)

    composed = _blend_mask(composed, a_only, color=(230, 56, 70), alpha=0.42)
    composed = _blend_mask(composed, b_only, color=(55, 120, 235), alpha=0.42)
    composed = _blend_mask(composed, overlap, color=(126, 87, 194), alpha=0.55)
    composed = _blend_mask(composed, contour, color=(255, 255, 255), alpha=0.95)
    return Image.fromarray(composed, mode="RGB")


def render_single_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    """Overlay one binary mask with a single accent color and white outline."""

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    boolean_mask = np.asarray(mask, dtype=bool)
    composed = _blend_mask(composed, boolean_mask, color=color, alpha=0.42)
    composed = _blend_mask(composed, _mask_outline(boolean_mask), color=(255, 255, 255), alpha=0.95)
    return Image.fromarray(composed, mode="RGB")


def render_boolean_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    """Overlay one boolean mask with a single accent color and white outline."""

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    boolean_mask = np.asarray(mask, dtype=bool)
    composed = _blend_mask(composed, boolean_mask, color=color, alpha=0.38)
    composed = _blend_mask(composed, _mask_outline(boolean_mask), color=(255, 255, 255), alpha=0.95)
    return Image.fromarray(composed, mode="RGB")


def render_pair_overlay(
    image: Image.Image,
    pair_mask_a: np.ndarray,
    pair_mask_b: np.ndarray,
) -> Image.Image:
    """Overlay the selected dense-mask pair with distinct colors."""

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    composed = _blend_mask(composed, pair_mask_a, color=(230, 56, 70), alpha=0.42)
    composed = _blend_mask(composed, pair_mask_b, color=(55, 120, 235), alpha=0.42)
    overlap = np.logical_and(pair_mask_a, pair_mask_b)
    composed = _blend_mask(composed, overlap, color=(255, 255, 255), alpha=0.75)
    return Image.fromarray(composed, mode="RGB")


def render_margin_overlay(image: Image.Image, margin: np.ndarray) -> Image.Image:
    """Overlay a signed feature margin as a red-vs-blue heatmap."""

    base = np.asarray(image.convert("RGB"), dtype=np.uint8).astype(np.float32)
    margin_array = np.asarray(margin, dtype=np.float32)
    max_abs = float(np.max(np.abs(margin_array))) if margin_array.size else 0.0
    if max_abs <= 1e-6:
        return Image.fromarray(base.astype(np.uint8), mode="RGB")

    normalized = np.clip(margin_array / max_abs, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    color = np.stack([positive * 230.0, np.zeros_like(positive), negative * 235.0], axis=-1)
    alpha = np.maximum(positive, negative)[..., None] * 0.65
    composed = ((1.0 - alpha) * base) + (alpha * color)
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="RGB")


def _format_prompt_candidate_title(prefix: str, candidate: dict[str, object]) -> str:
    rank = int(candidate.get("rank", 0))
    index = int(candidate.get("index", 0))
    prompt_iou = float(candidate.get("prompt_iou", 0.0))
    score = float(candidate.get("score", 0.0))
    return f"{prefix} | rank={rank} idx={index} pIoU={prompt_iou:.3f} score={score:.3f}"


def _format_selected_prompt_title(
    prefix: str,
    *,
    rank_a: int | None,
    rank_b: int | None,
    index_a: int | None,
    index_b: int | None,
) -> str:
    return (
        f"{prefix} | A(rank={rank_a},idx={index_a}) "
        f"B(rank={rank_b},idx={index_b})"
    )


def build_visual_footer_lines(
    sample: RwtdSample,
    protocol: str,
    metric_summary: str | None = None,
) -> list[str]:
    """Return wrapped-caption source lines for a saved RWTD visualization panel.

    Inputs:
        sample: RWTD sample containing prompt text, labels, and oracle points.
        protocol: Prompting protocol label used for the prediction.
        metric_summary: Optional summary string such as IoU/Dice values.

    Returns:
        List of logical caption lines before width-based wrapping.
    """

    lines = [
        f"crop={sample.crop_name} | split={sample.split} | protocol={protocol}",
        f"Texture A: {sample.texture_a}",
        f"Texture B: {sample.texture_b}",
        (
            f"Original labels: A={sample.original_texture_a} | B={sample.original_texture_b} | "
            f"Oracle points: A={len(sample.oracle_points_a)} | B={len(sample.oracle_points_b)}"
        ),
    ]
    if metric_summary:
        lines.append(f"Metrics: {metric_summary}")
    return lines


def build_visual_caption(
    sample: RwtdSample,
    protocol: str,
    metric_summary: str | None = None,
) -> str:
    """Return a single-line caption string for metadata exports and WandB previews."""

    return " | ".join(build_visual_footer_lines(sample, protocol, metric_summary))


def _blend_mask(
    image_array: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    color_array = np.array(color, dtype=np.float32)
    output = image_array.astype(np.float32)
    boolean_mask = np.asarray(mask, dtype=bool)
    output[boolean_mask] = (1.0 - alpha) * output[boolean_mask] + alpha * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def _segment_color(index: int) -> tuple[int, int, int]:
    if index < len(_SEGMENT_PALETTE):
        return _SEGMENT_PALETTE[index]

    hue = (0.6180339887498949 * (index - len(_SEGMENT_PALETTE) + 1)) % 1.0
    saturation = 0.68
    value = 0.92
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        int(round(red * 255)),
        int(round(green * 255)),
        int(round(blue * 255)),
    )


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    return np.logical_and(boolean_mask, np.logical_not(_erode_mask(boolean_mask)))


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    height, width = boolean_mask.shape
    padded = np.pad(boolean_mask, pad_width=1, mode="constant", constant_values=False)
    neighbours = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbours.append(padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width])
    return np.logical_and.reduce(neighbours)


def _save_native_label_map_sidecars(path_stem: Path, label_map: np.ndarray) -> None:
    np.save(path_stem.with_suffix(".npy"), np.asarray(label_map, dtype=np.int32))
    render_native_cluster_label_map(label_map).save(path_stem.with_suffix(".png"))


def _annotate_panel(image: Image.Image, title: str) -> Image.Image:
    lines = _wrap_text(title, max_width=image.width - 2 * _TEXT_PADDING_X)
    line_height = _line_height()
    bar_height = (
        2 * _TEXT_PADDING_Y
        + len(lines) * line_height
        + max(0, len(lines) - 1) * _TEXT_LINE_SPACING
    )
    annotated = Image.new("RGB", (image.width, image.height + bar_height), color=(248, 248, 248))
    annotated.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(annotated)
    y_offset = _TEXT_PADDING_Y
    for line in lines:
        draw.text((_TEXT_PADDING_X, y_offset), line, fill=(24, 24, 24), font=_FONT)
        y_offset += line_height + _TEXT_LINE_SPACING
    return annotated


def _concatenate_horizontally(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required for concatenation.")

    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def _concatenate_vertically(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required for concatenation.")

    max_width = max(image.width for image in images)
    total_height = sum(image.height for image in images)
    canvas = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

    y_offset = 0
    for image in images:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height
    return canvas


def _append_footer(image: Image.Image, lines: list[str]) -> Image.Image:
    wrapped_lines: list[str] = []
    max_width = image.width - 2 * _TEXT_PADDING_X
    for line in lines:
        wrapped_lines.extend(_wrap_text(line, max_width=max_width))

    line_height = _line_height()
    footer_height = (
        2 * _TEXT_PADDING_Y
        + len(wrapped_lines) * line_height
        + max(0, len(wrapped_lines) - 1) * _TEXT_LINE_SPACING
    )
    annotated = Image.new("RGB", (image.width, image.height + footer_height), color=(248, 248, 248))
    annotated.paste(image, (0, 0))

    draw = ImageDraw.Draw(annotated)
    y_offset = image.height + _TEXT_PADDING_Y
    for line in wrapped_lines:
        draw.text((_TEXT_PADDING_X, y_offset), line, fill=(24, 24, 24), font=_FONT)
        y_offset += line_height + _TEXT_LINE_SPACING
    return annotated


def _wrap_text(text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]
    for word in words[1:]:
        candidate = f"{current_line} {word}"
        if _text_width(candidate) <= max_width:
            current_line = candidate
            continue
        lines.append(current_line)
        current_line = word
    lines.append(current_line)
    return lines


def _line_height() -> int:
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), "Ag", font=_FONT)
    return bbox[3] - bbox[1]


def _text_width(text: str) -> int:
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=_FONT)
    return bbox[2] - bbox[0]


def save_detexture_multi_partition_panel(
    output_path: str | Path,
    sample,
    gt_label_map: np.ndarray,
    predicted_label_map: np.ndarray,
    valid_pixel_mask: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Path:
    """Render and save the standard DeTexture multi-region audit panel."""

    panel = render_detexture_multi_partition_panel(
        sample=sample,
        gt_label_map=gt_label_map,
        predicted_label_map=predicted_label_map,
        valid_pixel_mask=valid_pixel_mask,
        protocol=protocol,
        metric_summary=metric_summary,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def render_detexture_multi_partition_panel(
    sample,
    gt_label_map: np.ndarray,
    predicted_label_map: np.ndarray,
    valid_pixel_mask: np.ndarray,
    protocol: str,
    metric_summary: str | None = None,
) -> Image.Image:
    """Build the multi-region DeTexture audit panel.

    Layout:
    - Input image
    - GT label map overlay
    - Predicted label map overlay
    - Predicted boundaries over the RGB image
    """

    input_panel = _annotate_panel(sample.image.convert("RGB"), "Input")
    gt_panel = _annotate_panel(
        render_multilabel_partition_overlay(sample.image, gt_label_map, valid_pixel_mask),
        f"GT | K={int(np.unique(np.asarray(gt_label_map)[np.asarray(valid_pixel_mask, dtype=bool)]).size)}",
    )
    pred_panel = _annotate_panel(
        render_multilabel_partition_overlay(sample.image, predicted_label_map, valid_pixel_mask),
        f"Prediction | K={int(np.unique(np.asarray(predicted_label_map)[np.asarray(valid_pixel_mask, dtype=bool)]).size)}",
    )
    boundary_panel = _annotate_panel(
        render_multilabel_boundary_overlay(sample.image, predicted_label_map, valid_pixel_mask),
        "Pred boundaries",
    )
    panel = _concatenate_horizontally([input_panel, gt_panel, pred_panel, boundary_panel])
    return _append_footer(panel, _build_multilabel_visual_footer_lines(sample, protocol, metric_summary))


def render_multilabel_partition_overlay(
    image: Image.Image,
    label_map: np.ndarray,
    valid_pixel_mask: np.ndarray | None = None,
    alpha: float = 0.50,
) -> Image.Image:
    """Overlay an integer label map with one categorical color per visible region."""

    labels = np.asarray(label_map, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError(f"label_map must be 2D, got shape {labels.shape}.")
    if valid_pixel_mask is None:
        valid = np.ones(labels.shape, dtype=bool)
    else:
        valid = np.asarray(valid_pixel_mask, dtype=bool)
        if valid.shape != labels.shape:
            raise ValueError(f"valid_pixel_mask must match label_map shape {labels.shape}, got {valid.shape}.")

    base = np.asarray(image.convert("RGB"), dtype=np.uint8).astype(np.float32)
    composed = base.copy()
    visible_labels = [int(value) for value in np.unique(labels[valid])]
    for palette_index, label_value in enumerate(visible_labels):
        region = np.logical_and(valid, labels == label_value)
        if not np.any(region):
            continue
        color = np.array(_segment_color(palette_index), dtype=np.float32)
        composed[region] = (1.0 - alpha) * composed[region] + alpha * color
    boundary = boundary_from_label_map(labels, valid_mask=valid)
    composed = _blend_mask(np.clip(composed, 0, 255).astype(np.uint8), _mask_outline(boundary), color=(255, 255, 255), alpha=0.95)
    if np.any(~valid):
        invalid_overlay = _blend_mask(composed, ~valid, color=(180, 180, 180), alpha=0.35)
        composed = invalid_overlay
    return Image.fromarray(np.asarray(composed, dtype=np.uint8), mode="RGB")


def render_multilabel_boundary_overlay(
    image: Image.Image,
    label_map: np.ndarray,
    valid_pixel_mask: np.ndarray | None = None,
) -> Image.Image:
    """Overlay only the predicted multi-region boundaries on the RGB image."""

    labels = np.asarray(label_map, dtype=np.int32)
    if valid_pixel_mask is None:
        valid = np.ones(labels.shape, dtype=bool)
    else:
        valid = np.asarray(valid_pixel_mask, dtype=bool)
    boundary = boundary_from_label_map(labels, valid_mask=valid)
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    composed = base.copy()
    composed = _blend_mask(composed, _mask_outline(boundary), color=(255, 255, 255), alpha=0.95)
    if np.any(~valid):
        composed = _blend_mask(composed, ~valid, color=(180, 180, 180), alpha=0.35)
    return Image.fromarray(composed, mode="RGB")


def _build_multilabel_visual_footer_lines(sample, protocol: str, metric_summary: str | None = None) -> list[str]:
    lines = [
        f"crop={sample.crop_name} | split={sample.split} | protocol={protocol}",
        (
            f"GT decode: method={sample.gt_decode_method} | raw_K={sample.raw_oracle_num_regions} | "
            f"oracle_K={sample.oracle_num_regions} | joker_regions={sample.joker_region_count} | "
            f"joker_px={sample.joker_pixel_fraction:.3f}"
        ),
        (
            "GT decode diagnostics: "
            f"unique_rgb_colors={sample.gt_decode_diagnostics.get('num_unique_rgb_colors')} | "
            f"criterion={sample.gt_decode_diagnostics.get('selection_criterion')}"
        ),
    ]
    if metric_summary:
        lines.append(f"Metrics: {metric_summary}")
    return lines
