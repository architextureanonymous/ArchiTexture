from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rwtd_sam3.cli import build_parser
from rwtd_sam3.eval.experiment_registry import (
    CROSS_DATASET_EXPERIMENT_DATASETS,
    get_cross_dataset_experiment_spec,
)
from rwtd_sam3.utils.visualization import render_pooled_feature_pca_overlay
from rwtd_sam3.models.sam2_feature_cluster_runner import (
    _apply_image_transform_array,
    _undo_feature_transform,
)


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    @property
    def shape(self):
        return self._array.shape

    def clone(self):
        return _FakeTensor(np.array(self._array, copy=True))

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._array)


class _FakeTorch:
    @staticmethod
    def flip(feature_map: _FakeTensor, dims: tuple[int, ...]):
        array = feature_map.to_numpy()
        axes = tuple(dim if dim >= 0 else array.ndim + dim for dim in dims)
        return _FakeTensor(np.flip(array, axis=axes))


def _assert_cli_accepts_variant() -> None:
    parser = build_parser()
    for command in (
        ["eval-sam3-auto", "--variant", "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only"],
        [
            "eval-architexture-binary",
            "--route",
            "stld",
            "--benchmark-root",
            "/tmp/stld",
            "--variant",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
        ],
        [
            "eval-detexture-binary",
            "--dataset-root",
            "/tmp/detexture",
            "--variant",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
        ],
        [
            "eval-cstd-binary",
            "--dataset-root",
            "/tmp/cstd",
            "--variant",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
        ],
        [
            "eval-glas-binary",
            "--dataset-root",
            "/tmp/glas",
            "--variant",
            "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
            "--save-pooled-feature-pca-overlay",
        ],
    ):
        args = parser.parse_args(command)
        assert args.variant == "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only"
        if "--save-pooled-feature-pca-overlay" in command:
            assert args.save_pooled_feature_pca_overlay is True


def _assert_registry_covers_supported_datasets() -> None:
    spec = get_cross_dataset_experiment_spec("feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only")
    assert spec.supported_datasets == CROSS_DATASET_EXPERIMENT_DATASETS
    assert spec.default_model_id == "facebook/sam3"


def _assert_flip_helpers_round_trip() -> None:
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    assert np.array_equal(_apply_image_transform_array(image, "identity"), image)
    assert np.array_equal(_apply_image_transform_array(image, "hflip"), np.flip(image, axis=1))
    assert np.array_equal(_apply_image_transform_array(image, "vflip"), np.flip(image, axis=0))
    assert np.array_equal(_apply_image_transform_array(image, "hvflip"), np.flip(np.flip(image, axis=0), axis=1))

    feature = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    transformed_features = {
        "identity": feature,
        "hflip": np.flip(feature, axis=-1).copy(),
        "vflip": np.flip(feature, axis=-2).copy(),
        "hvflip": np.flip(np.flip(feature, axis=-2), axis=-1).copy(),
    }
    for name, transformed in transformed_features.items():
        restored = _undo_feature_transform(_FakeTensor(transformed), name, _FakeTorch())
        assert np.array_equal(restored.to_numpy(), feature)

    image = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
    pooled_feature_map = np.stack(
        [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32),
        ],
        axis=0,
    )
    overlay = render_pooled_feature_pca_overlay(image, pooled_feature_map, alpha=0.5)
    assert overlay.size == image.size
    assert np.any(np.asarray(overlay) != np.asarray(image))


def main() -> int:
    _assert_registry_covers_supported_datasets()
    _assert_cli_accepts_variant()
    _assert_flip_helpers_round_trip()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
