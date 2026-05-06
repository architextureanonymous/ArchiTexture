"""Shared helpers for pooled-feature PCA overlay exports.

The coarse-feature figure script projects pooled coarsest features to RGB with PCA
and overlays the image-space result on top of the input image. This module keeps
the same projection and blending logic available to the evaluation entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from rwtd_sam3.utils.visualization import save_pooled_feature_pca_overlay


def save_sample_pooled_feature_pca_overlay(
    output_path: str | Path,
    *,
    sample_image: Image.Image,
    runner: Any,
    alpha: float = 0.5,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save the pooled-feature PCA overlay for one sample using a runner-specific extractor."""

    if not hasattr(runner, "extract_pooled_feature_map_for_visualization"):
        raise AttributeError(
            "Runner does not expose extract_pooled_feature_map_for_visualization(), "
            "so pooled-feature PCA overlays cannot be generated."
        )
    pooled_feature_map = runner.extract_pooled_feature_map_for_visualization(sample_image)
    return save_pooled_feature_pca_overlay(
        output_path=output_path,
        sample_image=sample_image,
        pooled_feature_map=pooled_feature_map,
        alpha=float(alpha),
        robust_percentiles=robust_percentiles,
    )


def save_named_sample_pooled_feature_pca_overlay(
    output_dir: str | Path,
    *,
    sample: Any,
    runner: Any,
    file_name: str | None = None,
    alpha: float = 0.5,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),
) -> Path:
    """Save the PCA overlay for one sample into ``output_dir`` using a stable file name."""

    resolved_name = file_name or f"{sample.crop_name}_pooled_coarse_features_pca_rgb_overlay.png"
    output_path = Path(output_dir) / resolved_name
    return save_sample_pooled_feature_pca_overlay(
        output_path=output_path,
        sample_image=sample.image,
        runner=runner,
        alpha=alpha,
        robust_percentiles=robust_percentiles,
    )
