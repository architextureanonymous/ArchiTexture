"""Vanilla binary CFC runners on a frozen SAM-2 backbone.

This module reuses the repository's current binary coarse feature clustering head
while swapping the frozen backbone from SAM-3 to Meta's official SAM-2 image
predictor. Supported SAM-2 ablations stay minimal:
- frozen coarsest image embedding only
- optional flip averaging of that same image embedding
- L2 normalization
- average pooling before clustering
- deterministic 2-way cosine-style clustering
- nearest-neighbor upsampling
- no prompts, no proposals, no spatial cleanup, no refinement
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID, Sam2RuntimeError
from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    FeatureClusterCoarseToFineGlobalRefinement,
    Sam3FeatureClusterCoarseToFineGlobalRuntimeError,
    _binary_iou,
    _cluster_pooled_coarsest_level_2way,
    _compute_label_map_component_stats,
    _label_map_to_cluster_masks,
    _normalize_feature_map,
    _upsample_label_map,
)


SAM2_CFC_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
}


class Sam2FeatureClusterCoarseOnlyRunner:
    """Run the vanilla binary CFC head on frozen SAM-2 image features."""

    def __init__(
        self,
        model_id: str = DEFAULT_SAM2_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
        fast_cuda: bool = False,
        compile_image_encoder: bool = False,
    ) -> None:
        resolved_settings = dict(SAM2_CFC_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self.fast_cuda = fast_cuda
        self.compile_image_encoder = compile_image_encoder
        self.settings = resolved_settings
        self._predictor_bundle: tuple[Any, Any] | None = None

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Extract one frozen SAM-2 coarsest embedding and run vanilla binary CFC."""

        torch_module, predictor = self._ensure_predictor_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        feature_map = self._extract_normalized_feature_map_for_array(
            image_array=image_array,
            torch_module=torch_module,
            predictor=predictor,
        )
        image_size = (int(image_array.shape[0]), int(image_array.shape[1]))
        return self._build_refinement_from_feature_map(
            feature_map=feature_map,
            image_size=image_size,
            torch_module=torch_module,
            coarsest_init_mode="pooled_avg_pool_coarse_only_sam2_backbone",
            empty_mask_variant="feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
            level_name="sam2_image_embed",
        )

    def _ensure_predictor_bundle(self) -> tuple[Any, Any]:
        if self._predictor_bundle is not None:
            return self._predictor_bundle
        if self.official_checkpoint_path is not None:
            raise Sam2RuntimeError(
                "The SAM-2 CFC backbone variants use Hugging Face model ids and do not support official checkpoint paths."
            )
        try:
            import torch
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - runtime path
            raise Sam2RuntimeError(
                "SAM-2 CFC evaluation requires Meta's official SAM-2 package. Install the sam2-baseline extra before running this variant."
            ) from exc

        device = _resolve_device(self.requested_device, torch)
        _configure_cuda_runtime(torch_module=torch, device=device, fast_cuda=self.fast_cuda)
        pretrained_kwargs: dict[str, Any] = {"device": device}
        if self.hf_token:
            pretrained_kwargs["token"] = self.hf_token
        if self.compile_image_encoder:
            pretrained_kwargs["hydra_overrides_extra"] = ["++model.compile_image_encoder=True"]
        try:
            sam_model = build_sam2_hf(self.model_id, **pretrained_kwargs)
            predictor = SAM2ImagePredictor(sam_model)
        except Exception as exc:  # pragma: no cover - runtime path
            raise Sam2RuntimeError(f"Failed to load SAM-2 CFC backbone {self.model_id}: {exc}") from exc

        self._predictor_bundle = (torch, predictor)
        return self._predictor_bundle

    def _extract_normalized_feature_map_for_array(self, *, image_array: np.ndarray, torch_module: Any, predictor: Any):
        autocast_context = _resolve_autocast_context(
            torch_module=torch_module,
            requested_device=self.requested_device,
            fast_cuda=self.fast_cuda,
        )
        try:
            with autocast_context:
                predictor.set_image(image_array)
        except Exception as exc:  # pragma: no cover - runtime path
            raise Sam2RuntimeError(f"SAM-2 feature extraction failed during set_image: {exc}") from exc
        return self._extract_coarsest_feature_map(predictor=predictor, torch_module=torch_module)

    def _build_refinement_from_feature_map(
        self,
        *,
        feature_map: Any,
        image_size: tuple[int, int],
        torch_module: Any,
        coarsest_init_mode: str,
        empty_mask_variant: str,
        level_name: str,
    ) -> FeatureClusterCoarseToFineGlobalRefinement:
        native_label_map, pooled_grid_resolution = _cluster_pooled_coarsest_level_2way(
            feature_map=feature_map,
            torch_module=torch_module,
            settings=self.settings,
        )
        image_space_label_map = _upsample_label_map(
            native_label_map,
            target_size=image_size,
            torch_module=torch_module,
        )
        rough_mask_a, rough_mask_b = _label_map_to_cluster_masks(image_space_label_map)
        cluster_pixel_count_a = int(rough_mask_a.sum())
        cluster_pixel_count_b = int(rough_mask_b.sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                f"{empty_mask_variant} produced an empty pooled coarse mask.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=image_space_label_map,
            level_label_maps=(image_space_label_map,),
            level_names=(level_name,),
            level_resolutions=((int(feature_map.shape[-2]), int(feature_map.shape[-1])),),
            rough_mask_a=rough_mask_a,
            rough_mask_b=rough_mask_b,
            refined_mask_a=rough_mask_a,
            refined_mask_b=rough_mask_b,
            refined_score_a=0.0,
            refined_score_b=0.0,
            refined_pair_selection_score=0.0,
            refined_pair_overlap_iou=float(_binary_iou(rough_mask_a, rough_mask_b)),
            cluster_pixel_count_a=cluster_pixel_count_a,
            cluster_pixel_count_b=cluster_pixel_count_b,
            coarsest_native_label_map=native_label_map,
            coarsest_init_mode=coarsest_init_mode,
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(native_label_map),
            multiscale_refinement_applied=False,
            sam_refinement_applied=False,
        )

    def extract_pooled_feature_map_for_visualization(self, image: Image.Image):
        """Expose the coarsest SAM-2 feature map used for pooled clustering."""

        torch_module, predictor = self._ensure_predictor_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        return self._extract_normalized_feature_map_for_array(
            image_array=image_array,
            torch_module=torch_module,
            predictor=predictor,
        )

    @staticmethod
    def _extract_coarsest_feature_map(*, predictor: Any, torch_module: Any):
        if hasattr(predictor, "get_image_embedding"):
            image_embed = predictor.get_image_embedding()
        else:  # pragma: no cover - compatibility path
            features = getattr(predictor, "_features", None)
            if not isinstance(features, dict) or "image_embed" not in features:
                raise Sam2RuntimeError("SAM-2 predictor did not expose an image embedding after set_image().")
            image_embed = features["image_embed"]
        if getattr(image_embed, "ndim", None) not in {4} or int(image_embed.shape[0]) < 1 or int(image_embed.shape[0]) > 1:
            raise Sam2RuntimeError(
                f"SAM-2 predictor returned an unexpected image_embed shape {getattr(image_embed, 'shape', None)}; expected 1xCxHxW."
            )
        feature_map = image_embed[0].to(dtype=torch_module.float32)
        return _normalize_feature_map(feature_map, torch_module)


class Sam2FeatureClusterFlipAvgCoarseOnlyRunner(Sam2FeatureClusterCoarseOnlyRunner):
    """Run the same vanilla CFC head on flip-averaged frozen SAM-2 image embeddings."""

    def extract_pooled_feature_map_for_visualization(self, image: Image.Image):
        """Expose the flip-averaged SAM-2 feature map used for pooled clustering."""

        torch_module, predictor = self._ensure_predictor_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        feature_maps = []
        reference_shape: tuple[int, int, int] | None = None
        for transform_name in ("identity", "hflip", "vflip", "hvflip"):
            transformed_array = _apply_image_transform_array(image_array, transform_name)
            transformed_feature_map = self._extract_normalized_feature_map_for_array(
                image_array=transformed_array,
                torch_module=torch_module,
                predictor=predictor,
            )
            restored_feature_map = _undo_feature_transform(transformed_feature_map, transform_name, torch_module)
            current_shape = (
                int(restored_feature_map.shape[0]),
                int(restored_feature_map.shape[1]),
                int(restored_feature_map.shape[2]),
            )
            if reference_shape is None:
                reference_shape = current_shape
            elif not current_shape == reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2 got inconsistent coarsest SAM-2 embedding shapes across flip averaging: "
                    f"expected {reference_shape}, got {current_shape}."
                )
            feature_maps.append(restored_feature_map)

        flip_averaged_feature_map = torch_module.stack(feature_maps, dim=0).mean(dim=0)
        return _normalize_feature_map(flip_averaged_feature_map, torch_module)

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Extract four flipped SAM-2 embeddings, average them in canonical coordinates, and cluster once."""

        torch_module, predictor = self._ensure_predictor_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        image_size = (int(image_array.shape[0]), int(image_array.shape[1]))
        feature_maps = []
        reference_shape: tuple[int, int, int] | None = None
        for transform_name in ("identity", "hflip", "vflip", "hvflip"):
            transformed_array = _apply_image_transform_array(image_array, transform_name)
            transformed_feature_map = self._extract_normalized_feature_map_for_array(
                image_array=transformed_array,
                torch_module=torch_module,
                predictor=predictor,
            )
            restored_feature_map = _undo_feature_transform(transformed_feature_map, transform_name, torch_module)
            current_shape = (
                int(restored_feature_map.shape[0]),
                int(restored_feature_map.shape[1]),
                int(restored_feature_map.shape[2]),
            )
            if reference_shape is None:
                reference_shape = current_shape
            elif not current_shape == reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2 got inconsistent coarsest SAM-2 embedding shapes across flip averaging: "
                    f"expected {reference_shape}, got {current_shape}."
                )
            feature_maps.append(restored_feature_map)

        flip_averaged_feature_map = torch_module.stack(feature_maps, dim=0).mean(dim=0)
        flip_averaged_feature_map = _normalize_feature_map(
            flip_averaged_feature_map.to(dtype=torch_module.float32),
            torch_module,
        )
        return self._build_refinement_from_feature_map(
            feature_map=flip_averaged_feature_map,
            image_size=image_size,
            torch_module=torch_module,
            coarsest_init_mode="pooled_avg_pool_flip_avg_coarse_only_sam2_backbone",
            empty_mask_variant="feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
            level_name="sam2_image_embed",
        )


def _apply_image_transform_array(image_array: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return np.array(image_array, copy=True)
    if transform_name == "hflip":
        return np.flip(image_array, axis=1).copy()
    if transform_name == "vflip":
        return np.flip(image_array, axis=0).copy()
    if transform_name == "hvflip":
        return np.flip(np.flip(image_array, axis=0), axis=1).copy()
    raise Sam2RuntimeError(f"Unsupported SAM-2 flip transform {transform_name}.")


def _undo_feature_transform(feature_map: Any, transform_name: str, torch_module: Any):
    if transform_name == "identity":
        return feature_map.clone()
    if transform_name == "hflip":
        return torch_module.flip(feature_map, dims=(-1,))
    if transform_name == "vflip":
        return torch_module.flip(feature_map, dims=(-2,))
    if transform_name == "hvflip":
        return torch_module.flip(feature_map, dims=(-2, -1))
    raise Sam2RuntimeError(f"Unsupported SAM-2 flip transform {transform_name}.")


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise Sam2RuntimeError("CUDA was requested explicitly but no CUDA device is available.")
    if requested_device not in {"cpu", "cuda"}:
        raise Sam2RuntimeError(f"Unsupported device {requested_device}. Expected one of: auto, cpu, cuda.")
    return requested_device


def _configure_cuda_runtime(torch_module: Any, device: str, fast_cuda: bool) -> None:
    if device == "cuda" and fast_cuda:
        if torch_module.cuda.get_device_properties(0).major >= 8:
            torch_module.backends.cuda.matmul.allow_tf32 = True
            torch_module.backends.cudnn.allow_tf32 = True


def _resolve_autocast_context(torch_module: Any, requested_device: str, fast_cuda: bool):
    device = _resolve_device(requested_device, torch_module)
    if device == "cuda" and fast_cuda:
        return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
    return nullcontext()
