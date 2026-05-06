"""Prompt-free multi-region clustering on pooled coarsest SAM features.

This module extends the current binary coarse-only feature partitioning method to
K-way clustering while holding the rest of the method fixed:
- frozen SAM dense features
- true coarsest feature level
- L2 normalization
- average pooling before clustering
- deterministic clustering
- nearest-neighbor upsampling
- no post-processing or prompt refinement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    DEFAULT_MODEL_ID,
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    Sam3FeatureClusterCoarseToFineGlobalRunner,
    Sam3FeatureClusterCoarseToFineGlobalRuntimeError,
    _apply_image_transform,
    _feature_map_to_numpy,
    _normalize_feature_map,
    _normalize_numpy_feature_map,
    _undo_feature_transform,
    _upsample_label_map,
)
from rwtd_sam3.models.deepdpm_frozen_feature_head import (
    DEEPDPM_FROZEN_FEATURE_SETTINGS,
    fit_deepdpm_frozen_features,
)
from rwtd_sam3.models.hdbscan_frozen_feature_head import (
    HDBSCAN_FROZEN_FEATURE_SETTINGS,
    fit_hdbscan_frozen_features,
)
from rwtd_sam3.utils.deterministic_clustering import (
    deterministic_kmeans,
    select_num_clusters_by_silhouette,
)


FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    "predicted_k_min": 1,
    "predicted_k_max": 8,
}
FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    **DEEPDPM_FROZEN_FEATURE_SETTINGS,
}
FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS: dict[str, float | int | str | bool | None] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    **HDBSCAN_FROZEN_FEATURE_SETTINGS,
}

MULTI_ORACLE_K_VARIANT = "feature_cluster_coarse_global_pooled_oracle_k"
MULTI_ORACLE_K_FLIPAVG_VARIANT = "feature_cluster_coarse_global_pooled_oracle_k_flipavg"
MULTI_PREDICTED_K_VARIANT = "feature_cluster_coarse_global_pooled_predicted_k"
MULTI_DEEPDPM_VARIANT = "feature_cluster_coarse_global_pooled_deepdpm"
MULTI_DEEPDPM_FLIPAVG_VARIANT = "feature_cluster_coarse_global_pooled_deepdpm_flipavg"
MULTI_HDBSCAN_VARIANT = "feature_cluster_coarse_global_pooled_hdbscan"
MULTI_HDBSCAN_FLIPAVG_VARIANT = "feature_cluster_coarse_global_pooled_hdbscan_flipavg"
DETEXTURE_MULTI_SUPPORTED_VARIANTS = (
    MULTI_ORACLE_K_VARIANT,
    MULTI_ORACLE_K_FLIPAVG_VARIANT,
    MULTI_PREDICTED_K_VARIANT,
    MULTI_DEEPDPM_VARIANT,
    MULTI_DEEPDPM_FLIPAVG_VARIANT,
    MULTI_HDBSCAN_VARIANT,
    MULTI_HDBSCAN_FLIPAVG_VARIANT,
)
DETEXTURE_MULTI_DEFAULT_VARIANT = MULTI_ORACLE_K_VARIANT


@dataclass(frozen=True)
class FeatureClusterCoarseGlobalMultiPartition:
    """Artifacts from one prompt-free multi-region coarsest pooled partition run."""

    coarsest_level_name: str
    coarsest_level_resolution: tuple[int, int]
    coarsest_native_label_map: np.ndarray
    predicted_label_map: np.ndarray
    num_clusters: int
    cluster_pixel_counts: tuple[int, ...]
    coarsest_init_mode: str
    coarsest_pool_kernel_size: int
    coarsest_pool_stride: int
    pooled_grid_resolution: tuple[int, int]
    flip_averaged_features_used: bool
    predicted_k_selection_criterion: str | None = None
    predicted_k_criterion_value: float | None = None
    predicted_k_scores_by_k: dict[int, float] | None = None
    partition_head: str = "deterministic_kmeans"
    clustering_training_diagnostics: dict[str, Any] | None = None


class Sam3FeatureClusterCoarseGlobalPooledMultiRunner(Sam3FeatureClusterCoarseToFineGlobalRunner):
    """Shared prompt-free coarsest pooled K-way clustering runner."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_MULTI_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_multi_partition(
        self,
        image,
        *,
        num_clusters: int | None = None,
    ) -> FeatureClusterCoarseGlobalMultiPartition:
        """Cluster the pooled coarsest feature level into ``num_clusters`` regions."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        feature_map = self._build_clustering_feature_map(
            image=image,
            official_backend=official_backend,
            raw_feature_map=raw_feature_map,
            reference_shape=raw_feature_map.shape,
        )
        native_label_map, pooled_grid_resolution, diagnostics = _cluster_pooled_label_map_from_numpy_feature_map_kway(
            feature_map=feature_map,
            num_clusters=num_clusters,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        predicted_label_map = _upsample_label_map(
            native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        cluster_pixel_counts = tuple(
            int((predicted_label_map == label_id).sum()) for label_id in range(int(np.max(predicted_label_map)) + 1)
        )
        if any(pixel_count <= 0 for pixel_count in cluster_pixel_counts):
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_global_pooled produced an empty image-space cluster.",
                diagnostics={"cluster_pixel_counts": cluster_pixel_counts},
            )
        return FeatureClusterCoarseGlobalMultiPartition(
            coarsest_level_name=coarsest_level.name,
            coarsest_level_resolution=coarsest_level.resolution,
            coarsest_native_label_map=native_label_map.astype(np.int32),
            predicted_label_map=predicted_label_map.astype(np.int32),
            num_clusters=int(np.max(native_label_map)) + 1,
            cluster_pixel_counts=cluster_pixel_counts,
            coarsest_init_mode="pooled_avg_pool_multi_way",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            flip_averaged_features_used=False,
            predicted_k_selection_criterion=diagnostics.get("criterion_name"),
            predicted_k_criterion_value=diagnostics.get("criterion_value"),
            predicted_k_scores_by_k=diagnostics.get("scores_by_k"),
        )

    def _build_clustering_feature_map(
        self,
        *,
        image,
        official_backend: Any,
        raw_feature_map: np.ndarray,
        reference_shape: tuple[int, int, int],
    ) -> np.ndarray:
        del image, official_backend, reference_shape
        return raw_feature_map


class Sam3FeatureClusterCoarseGlobalPooledOracleKRunner(Sam3FeatureClusterCoarseGlobalPooledMultiRunner):
    """Oracle-K variant on the raw coarsest pooled SAM features."""

    def _build_clustering_feature_map(
        self,
        *,
        image,
        official_backend: Any,
        raw_feature_map: np.ndarray,
        reference_shape: tuple[int, int, int],
    ) -> np.ndarray:
        del image, official_backend, reference_shape
        return raw_feature_map


class Sam3FeatureClusterCoarseGlobalPooledOracleKFlipAvgRunner(Sam3FeatureClusterCoarseGlobalPooledMultiRunner):
    """Oracle-K variant using flip-averaged coarsest SAM features before clustering."""

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        partition = super().generate_multi_partition(image, num_clusters=num_clusters)
        return FeatureClusterCoarseGlobalMultiPartition(
            **{**partition.__dict__, "flip_averaged_features_used": True}
        )

    def _build_clustering_feature_map(
        self,
        *,
        image,
        official_backend: Any,
        raw_feature_map: np.ndarray,
        reference_shape: tuple[int, int, int],
    ) -> np.ndarray:
        feature_maps: list[np.ndarray] = [raw_feature_map]
        for transform_name in ("hflip", "vflip", "hvflip"):
            transformed_image = _apply_image_transform(image, transform_name)
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(transformed_image, official_backend)
            transformed_feature_map = _feature_map_to_numpy(feature_levels[0].feature_map)
            if transformed_feature_map.shape != reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_global_pooled_oracle_k_flipavg got inconsistent coarsest feature shapes "
                    f"across transforms: expected {reference_shape}, got {transformed_feature_map.shape} for {transform_name}."
                )
            feature_maps.append(_undo_feature_transform(transformed_feature_map, transform_name))
        return _normalize_numpy_feature_map(np.mean(np.stack(feature_maps, axis=0), axis=0))


class Sam3FeatureClusterCoarseGlobalPooledPredictedKRunner(Sam3FeatureClusterCoarseGlobalPooledOracleKRunner):
    """Predicted-K secondary ablation using deterministic silhouette selection."""

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        return super().generate_multi_partition(image, num_clusters=num_clusters)


class Sam3FeatureClusterCoarseGlobalPooledDeepDpmRunner(Sam3FeatureClusterCoarseGlobalPooledMultiRunner):
    """Stronger nonparametric deep clustering ablation on frozen pooled SAM features."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_DEEPDPM_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        del num_clusters
        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        feature_map = self._build_clustering_feature_map(
            image=image,
            official_backend=official_backend,
            raw_feature_map=raw_feature_map,
            reference_shape=raw_feature_map.shape,
        )
        native_label_map, pooled_grid_resolution, diagnostics = _cluster_pooled_label_map_from_numpy_feature_map_deepdpm(
            feature_map=feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        predicted_label_map = _upsample_label_map(
            native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        cluster_pixel_counts = tuple(
            int((predicted_label_map == label_id).sum()) for label_id in range(int(np.max(predicted_label_map)) + 1)
        )
        if any(pixel_count <= 0 for pixel_count in cluster_pixel_counts):
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_global_pooled_deepdpm produced an empty image-space cluster.",
                diagnostics={"cluster_pixel_counts": cluster_pixel_counts},
            )
        return FeatureClusterCoarseGlobalMultiPartition(
            coarsest_level_name=coarsest_level.name,
            coarsest_level_resolution=coarsest_level.resolution,
            coarsest_native_label_map=native_label_map.astype(np.int32),
            predicted_label_map=predicted_label_map.astype(np.int32),
            num_clusters=int(np.max(native_label_map)) + 1,
            cluster_pixel_counts=cluster_pixel_counts,
            coarsest_init_mode="pooled_avg_pool_deepdpm",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            flip_averaged_features_used=False,
            partition_head="deepdpm_frozen_features",
            clustering_training_diagnostics=diagnostics,
        )


class Sam3FeatureClusterCoarseGlobalPooledDeepDpmFlipAvgRunner(Sam3FeatureClusterCoarseGlobalPooledDeepDpmRunner):
    """Flip-averaged stronger nonparametric deep clustering ablation on frozen SAM features."""

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        partition = super().generate_multi_partition(image, num_clusters=num_clusters)
        return FeatureClusterCoarseGlobalMultiPartition(
            **{**partition.__dict__, "flip_averaged_features_used": True, "partition_head": "deepdpm_frozen_features_flipavg"}
        )

    def _build_clustering_feature_map(
        self,
        *,
        image,
        official_backend: Any,
        raw_feature_map: np.ndarray,
        reference_shape: tuple[int, int, int],
    ) -> np.ndarray:
        feature_maps: list[np.ndarray] = [raw_feature_map]
        for transform_name in ("hflip", "vflip", "hvflip"):
            transformed_image = _apply_image_transform(image, transform_name)
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(transformed_image, official_backend)
            transformed_feature_map = _feature_map_to_numpy(feature_levels[0].feature_map)
            if transformed_feature_map.shape != reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_global_pooled_deepdpm_flipavg got inconsistent coarsest feature shapes "
                    f"across transforms: expected {reference_shape}, got {transformed_feature_map.shape} for {transform_name}."
                )
            feature_maps.append(_undo_feature_transform(transformed_feature_map, transform_name))
        return _normalize_numpy_feature_map(np.mean(np.stack(feature_maps, axis=0), axis=0))


class Sam3FeatureClusterCoarseGlobalPooledHdbscanRunner(Sam3FeatureClusterCoarseGlobalPooledMultiRunner):
    """Simple HDBSCAN ablation on the same frozen pooled coarsest SAM features."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_GLOBAL_POOLED_HDBSCAN_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        del num_clusters
        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        feature_map = self._build_clustering_feature_map(
            image=image,
            official_backend=official_backend,
            raw_feature_map=raw_feature_map,
            reference_shape=raw_feature_map.shape,
        )
        native_label_map, pooled_grid_resolution, diagnostics = _cluster_pooled_label_map_from_numpy_feature_map_hdbscan(
            feature_map=feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        predicted_label_map = _upsample_label_map(
            native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        cluster_pixel_counts = tuple(
            int((predicted_label_map == label_id).sum()) for label_id in range(int(np.max(predicted_label_map)) + 1)
        )
        if any(pixel_count <= 0 for pixel_count in cluster_pixel_counts):
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_global_pooled_hdbscan produced an empty image-space cluster.",
                diagnostics={"cluster_pixel_counts": cluster_pixel_counts},
            )
        return FeatureClusterCoarseGlobalMultiPartition(
            coarsest_level_name=coarsest_level.name,
            coarsest_level_resolution=coarsest_level.resolution,
            coarsest_native_label_map=native_label_map.astype(np.int32),
            predicted_label_map=predicted_label_map.astype(np.int32),
            num_clusters=int(np.max(native_label_map)) + 1,
            cluster_pixel_counts=cluster_pixel_counts,
            coarsest_init_mode="pooled_avg_pool_hdbscan",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            flip_averaged_features_used=False,
            partition_head="hdbscan_frozen_features",
            clustering_training_diagnostics=diagnostics,
        )


class Sam3FeatureClusterCoarseGlobalPooledHdbscanFlipAvgRunner(Sam3FeatureClusterCoarseGlobalPooledHdbscanRunner):
    """Flip-averaged HDBSCAN ablation on frozen SAM pooled coarsest features."""

    def generate_multi_partition(self, image, *, num_clusters: int | None = None) -> FeatureClusterCoarseGlobalMultiPartition:
        partition = super().generate_multi_partition(image, num_clusters=num_clusters)
        return FeatureClusterCoarseGlobalMultiPartition(
            **{**partition.__dict__, "flip_averaged_features_used": True, "partition_head": "hdbscan_frozen_features_flipavg"}
        )

    def _build_clustering_feature_map(
        self,
        *,
        image,
        official_backend: Any,
        raw_feature_map: np.ndarray,
        reference_shape: tuple[int, int, int],
    ) -> np.ndarray:
        feature_maps: list[np.ndarray] = [raw_feature_map]
        for transform_name in ("hflip", "vflip", "hvflip"):
            transformed_image = _apply_image_transform(image, transform_name)
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(transformed_image, official_backend)
            transformed_feature_map = _feature_map_to_numpy(feature_levels[0].feature_map)
            if transformed_feature_map.shape != reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_global_pooled_hdbscan_flipavg got inconsistent coarsest feature shapes "
                    f"across transforms: expected {reference_shape}, got {transformed_feature_map.shape} for {transform_name}."
                )
            feature_maps.append(_undo_feature_transform(transformed_feature_map, transform_name))
        return _normalize_numpy_feature_map(np.mean(np.stack(feature_maps, axis=0), axis=0))


def build_detexture_multi_runner(
    variant: str,
    *,
    model_id: str,
    device: str,
    hf_token: str | None,
    official_checkpoint_path: str | None,
    settings: dict[str, float | int | str] | None = None,
):
    """Construct the multi-region coarsest pooled runner for the requested variant."""

    common_kwargs = {
        "model_id": model_id,
        "device": device,
        "hf_token": hf_token,
        "official_checkpoint_path": official_checkpoint_path,
        "settings": settings,
    }
    if variant == MULTI_ORACLE_K_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledOracleKRunner(**common_kwargs)
    if variant == MULTI_ORACLE_K_FLIPAVG_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledOracleKFlipAvgRunner(**common_kwargs)
    if variant == MULTI_PREDICTED_K_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledPredictedKRunner(**common_kwargs)
    if variant == MULTI_DEEPDPM_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledDeepDpmRunner(**common_kwargs)
    if variant == MULTI_DEEPDPM_FLIPAVG_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledDeepDpmFlipAvgRunner(**common_kwargs)
    if variant == MULTI_HDBSCAN_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledHdbscanRunner(**common_kwargs)
    if variant == MULTI_HDBSCAN_FLIPAVG_VARIANT:
        return Sam3FeatureClusterCoarseGlobalPooledHdbscanFlipAvgRunner(**common_kwargs)
    raise ValueError(f"Unsupported DeTexture multi variant '{variant}'.")


def _cluster_pooled_label_map_from_numpy_feature_map_kway(
    feature_map: np.ndarray,
    *,
    num_clusters: int | None,
    torch_module: Any,
    device: Any,
    settings: dict[str, float | int | str],
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    kernel_size = int(settings["coarsest_init_pool_kernel_size"])
    stride = int(settings["coarsest_init_pool_stride"])
    if kernel_size <= 0 or stride <= 0:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled requires positive pooling kernel/stride values.",
            diagnostics={
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    feature_tensor = torch_module.as_tensor(
        _normalize_numpy_feature_map(feature_map),
        dtype=torch_module.float32,
        device=device,
    )
    pooled_feature_map = torch_module.nn.functional.avg_pool2d(
        feature_tensor[None],
        kernel_size=kernel_size,
        stride=stride,
    )[0]
    if pooled_feature_map.shape[-2] < 1 or pooled_feature_map.shape[-1] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled collapsed the coarsest feature map during pooling.",
            diagnostics={
                "coarsest_resolution": (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    pooled_feature_map = _normalize_feature_map(pooled_feature_map.to(dtype=torch_module.float32), torch_module)
    pooled_feature_map_np = _feature_map_to_numpy(pooled_feature_map)
    pooled_vectors = pooled_feature_map_np.reshape(pooled_feature_map_np.shape[0], -1).T.astype(np.float32)
    if pooled_vectors.shape[0] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled found no pooled feature vectors to cluster."
        )

    if num_clusters is None:
        min_k = max(2, int(settings["predicted_k_min"]))
        max_k = max(min_k, int(settings["predicted_k_max"]))
        cluster_values = range(min_k, min(max_k, pooled_vectors.shape[0]) + 1)
        selection = select_num_clusters_by_silhouette(
            pooled_vectors,
            cluster_values,
            metric="cosine",
            max_iterations=int(settings["kmeans_max_iterations"]),
            tolerance=float(settings["kmeans_convergence_tolerance"]),
        )
        clustering = selection.clustering
        diagnostics = {
            "criterion_name": selection.criterion_name,
            "criterion_value": float(selection.criterion_value),
            "scores_by_k": {int(key): float(value) for key, value in selection.scores_by_k.items()},
        }
    else:
        clustering = deterministic_kmeans(
            pooled_vectors,
            num_clusters=int(num_clusters),
            metric="cosine",
            max_iterations=int(settings["kmeans_max_iterations"]),
            tolerance=float(settings["kmeans_convergence_tolerance"]),
        )
        diagnostics = {"criterion_name": None, "criterion_value": None, "scores_by_k": None}

    pooled_label_map = clustering.labels.reshape(pooled_feature_map_np.shape[1], pooled_feature_map_np.shape[2]).astype(np.int32)
    native_label_map = _upsample_label_map(
        pooled_label_map,
        target_size=(int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
        torch_module=torch_module,
    )
    native_label_map = _reindex_label_map_by_region_size(native_label_map)
    return native_label_map, (int(pooled_label_map.shape[0]), int(pooled_label_map.shape[1])), diagnostics


def _cluster_pooled_label_map_from_numpy_feature_map_hdbscan(
    feature_map: np.ndarray,
    *,
    torch_module: Any,
    device: Any,
    settings: dict[str, float | int | str | bool | None],
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    del device
    kernel_size = int(settings["coarsest_init_pool_kernel_size"])
    stride = int(settings["coarsest_init_pool_stride"])
    if kernel_size <= 0 or stride <= 0:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_hdbscan requires positive pooling kernel/stride values.",
            diagnostics={
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    feature_tensor = torch_module.as_tensor(
        _normalize_numpy_feature_map(feature_map),
        dtype=torch_module.float32,
        device="cpu",
    )
    pooled_feature_map = torch_module.nn.functional.avg_pool2d(
        feature_tensor[None],
        kernel_size=kernel_size,
        stride=stride,
    )[0]
    if pooled_feature_map.shape[-2] < 1 or pooled_feature_map.shape[-1] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_hdbscan collapsed the coarsest feature map during pooling.",
            diagnostics={
                "coarsest_resolution": (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    pooled_feature_map = _normalize_feature_map(pooled_feature_map.to(dtype=torch_module.float32), torch_module)
    pooled_feature_map_np = _feature_map_to_numpy(pooled_feature_map)
    pooled_vectors = pooled_feature_map_np.reshape(pooled_feature_map_np.shape[0], -1).T.astype(np.float32)
    if pooled_vectors.shape[0] < 2:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_hdbscan requires at least two pooled feature vectors."
        )

    fit = fit_hdbscan_frozen_features(pooled_vectors, settings=settings)
    pooled_label_map = fit.labels.reshape(pooled_feature_map_np.shape[1], pooled_feature_map_np.shape[2]).astype(np.int32)
    native_label_map = _upsample_label_map(
        pooled_label_map,
        target_size=(int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
        torch_module=torch_module,
    )
    native_label_map = _reindex_label_map_by_region_size(native_label_map)
    diagnostics = dict(fit.diagnostics)
    diagnostics.update(
        {
            "partition_head": "hdbscan_frozen_features",
            "final_num_clusters": int(fit.num_clusters),
            "cluster_sizes": [int(value) for value in fit.cluster_sizes],
            "noise_label_policy": "keep_noise_as_regular_region_if_present",
        }
    )
    return native_label_map, (int(pooled_label_map.shape[0]), int(pooled_label_map.shape[1])), diagnostics


def _cluster_pooled_label_map_from_numpy_feature_map_deepdpm(
    feature_map: np.ndarray,
    *,
    torch_module: Any,
    device: Any,
    settings: dict[str, float | int | str],
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    kernel_size = int(settings["coarsest_init_pool_kernel_size"])
    stride = int(settings["coarsest_init_pool_stride"])
    if kernel_size <= 0 or stride <= 0:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_deepdpm requires positive pooling kernel/stride values.",
            diagnostics={
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    feature_tensor = torch_module.as_tensor(
        _normalize_numpy_feature_map(feature_map),
        dtype=torch_module.float32,
        device=device,
    )
    pooled_feature_map = torch_module.nn.functional.avg_pool2d(
        feature_tensor[None],
        kernel_size=kernel_size,
        stride=stride,
    )[0]
    if pooled_feature_map.shape[-2] < 1 or pooled_feature_map.shape[-1] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_deepdpm collapsed the coarsest feature map during pooling.",
            diagnostics={
                "coarsest_resolution": (int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    pooled_feature_map = _normalize_feature_map(pooled_feature_map.to(dtype=torch_module.float32), torch_module)
    pooled_feature_map_np = _feature_map_to_numpy(pooled_feature_map)
    pooled_vectors = pooled_feature_map_np.reshape(pooled_feature_map_np.shape[0], -1).T.astype(np.float32)
    if pooled_vectors.shape[0] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_global_pooled_deepdpm found no pooled feature vectors to cluster."
        )

    fit = fit_deepdpm_frozen_features(
        pooled_vectors,
        device=device,
        settings=settings,
    )
    pooled_label_map = fit.labels.reshape(pooled_feature_map_np.shape[1], pooled_feature_map_np.shape[2]).astype(np.int32)
    native_label_map = _upsample_label_map(
        pooled_label_map,
        target_size=(int(feature_tensor.shape[-2]), int(feature_tensor.shape[-1])),
        torch_module=torch_module,
    )
    native_label_map = _reindex_label_map_by_region_size(native_label_map)
    diagnostics = dict(fit.diagnostics)
    diagnostics.update(
        {
            "partition_head": "deepdpm_frozen_features",
            "final_num_clusters": int(fit.num_clusters),
            "cluster_sizes": [int(value) for value in fit.cluster_sizes],
            "epochs_completed": int(fit.epochs_completed),
            "final_loss": float(fit.final_loss),
            "loss_history": [float(value) for value in fit.loss_history],
            "inferred_k_trace": [int(value) for value in fit.inferred_k_trace],
            "split_merge_events": list(fit.split_merge_events),
        }
    )
    return native_label_map, (int(pooled_label_map.shape[0]), int(pooled_label_map.shape[1])), diagnostics


def _reindex_label_map_by_region_size(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    label_values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remapped = np.empty_like(labels, dtype=np.int32)
    for new_label, old_index in enumerate(order):
        remapped[labels == int(label_values[old_index])] = int(new_label)
    return remapped
