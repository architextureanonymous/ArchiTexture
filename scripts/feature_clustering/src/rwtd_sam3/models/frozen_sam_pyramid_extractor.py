"""Backbone-swappable frozen SAM pyramid extractors for dense mask-head runs.

This adapter keeps the dense supervised mask-head trainers agnostic to whether
their frozen features come from SAM-3, SAM-2, or the original SAM ViT family.

- SAM-3 path reuses the existing native multiscale ``backbone_fpn`` extractor.
- SAM-2 path reuses the official image predictor and exposes its three frozen
  image feature maps under the repo's existing coarse-to-fine names:
  ``fpn_2`` (coarsest), ``fpn_1`` (middle), ``fpn_0`` (finest).
- Original SAM path reuses Hugging Face ``SamModel.get_image_embeddings()`` and
  exposes the single post-neck embedding grid as ``fpn_2`` only.

The module fails loudly when the model id cannot be mapped to a supported
backbone family, or when the discovered feature structure differs from the
expected runtime contract.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any, Protocol

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID, Sam2RuntimeError
from rwtd_sam3.models.sam3_coarse_vs_fine_scale_probe import Sam3CoarseVsFineScaleFeatureExtractor


LOGGER = logging.getLogger(__name__)


class FrozenSamPyramidExtractorRuntimeError(RuntimeError):
    """Raised when the frozen-backbone pyramid extractor path becomes invalid."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class FrozenSamPyramidExtractorProtocol(Protocol):
    """Minimal interface shared by the SAM-2 and SAM-3 frozen feature adapters."""

    def extract_sam_pyramid(self, image: Image.Image) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
        """Return ``(image_size_hw, pyramid_by_level_name)`` for one RGB image."""


def resolve_frozen_sam_backbone_family(model_id: str) -> str:
    """Infer the supported SAM backbone family from a model id.

    The mapping is explicit and intentionally narrow so unsupported ids fail
    rather than quietly taking the wrong runtime path.
    """

    lowered = str(model_id).strip().lower()
    if "sam3" in lowered:
        return "sam3"
    if "sam2" in lowered:
        return "sam2"
    if "sam-vit" in lowered or "sam_vit" in lowered:
        return "sam1"
    raise FrozenSamPyramidExtractorRuntimeError(
        f"Could not infer a supported frozen SAM backbone family from model id '{model_id}'.",
        diagnostics={
            "model_id": str(model_id),
            "supported_families": ["sam3", "sam2", "sam1"],
        },
    )


def describe_frozen_sam_feature_source(model_id: str) -> str:
    """Return a human-readable feature-source description for run metadata."""

    family = resolve_frozen_sam_backbone_family(model_id)
    if family == "sam3":
        return "backbone_fpn"
    if family == "sam1":
        return "sam1_get_image_embeddings"
    return "sam2_predictor_features(image_embed+high_res_feats)"


def build_frozen_sam_pyramid_extractor(
    *,
    model_id: str,
    device: str,
    hf_token: str | None,
    official_checkpoint_path: str | None,
) -> FrozenSamPyramidExtractorProtocol:
    """Create the correct frozen pyramid extractor for the requested backbone."""

    family = resolve_frozen_sam_backbone_family(model_id)
    if family == "sam3":
        _validate_sam3_checkpoint_contract(official_checkpoint_path)
        return Sam3CoarseVsFineScaleFeatureExtractor(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )
    if family == "sam1":
        return Sam1FrozenSingleScaleFeatureExtractor(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )
    return Sam2FrozenPyramidFeatureExtractor(
        model_id=model_id,
        device=device,
        hf_token=hf_token,
        official_checkpoint_path=official_checkpoint_path,
    )


def _validate_sam3_checkpoint_contract(official_checkpoint_path: str | None) -> None:
    if official_checkpoint_path in (None, ""):
        return
    lowered = str(official_checkpoint_path).lower()
    if "sam_vit_h" in lowered or "vit_h" in lowered:
        raise FrozenSamPyramidExtractorRuntimeError(
            "The SAM3 frozen-feature route does not accept the AutoSAM ViT-H checkpoint path. "
            "Use the AutoSAM faithful route for ViT-H and keep SAM3 runs on a SAM3-native checkpoint/config.",
            diagnostics={"official_checkpoint_path": official_checkpoint_path},
        )
    if "sam3" not in lowered:
        raise FrozenSamPyramidExtractorRuntimeError(
            "The SAM3 frozen-feature route requires a SAM3-native checkpoint/config path.",
            diagnostics={"official_checkpoint_path": official_checkpoint_path},
        )


class Sam1FrozenSingleScaleFeatureExtractor:
    """Expose original SAM image embeddings as a single `fpn_2` level."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self._model_bundle: tuple[Any, Any, Any] | None = None
        self._logged_signature = False

    def extract_sam_pyramid(self, image: Image.Image) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
        """Return the single original-SAM embedding grid as `fpn_2`."""

        torch_module, processor, model = self._ensure_model_bundle()
        rgb_image = image.convert("RGB")
        encoded_inputs = processor(images=rgb_image, return_tensors="pt")
        pixel_values = encoded_inputs.get("pixel_values")
        if pixel_values is None:
            raise FrozenSamPyramidExtractorRuntimeError(
                "Original SAM processor did not return `pixel_values`.",
                diagnostics={"model_id": self.model_id},
            )
        pixel_values = pixel_values.to(model.device)
        try:
            with torch_module.inference_mode():
                image_embeddings = model.get_image_embeddings(pixel_values)
        except Exception as exc:  # pragma: no cover - runtime path
            raise FrozenSamPyramidExtractorRuntimeError(
                f"Original SAM frozen feature extraction failed during get_image_embeddings(): {exc}",
                diagnostics={"model_id": self.model_id, "device": self.requested_device},
            ) from exc
        shape = getattr(image_embeddings, "shape", None)
        if shape is None or len(shape) != 4 or int(shape[0]) != 1:
            raise FrozenSamPyramidExtractorRuntimeError(
                "Original SAM `get_image_embeddings()` returned an unexpected tensor shape.",
                diagnostics={"shape": tuple(shape) if shape is not None else None},
            )
        feature_map = np.asarray(
            image_embeddings[0].detach().cpu().to(dtype=torch_module.float32).numpy(),
            dtype=np.float32,
        )
        if feature_map.ndim != 3:
            raise FrozenSamPyramidExtractorRuntimeError(
                "Original SAM image embeddings produced an unexpected feature-map rank.",
                diagnostics={"shape": tuple(int(value) for value in feature_map.shape)},
            )
        pyramid = {"fpn_2": feature_map}
        if not self._logged_signature:
            LOGGER.info(
                "Discovered original-SAM embedding grid mapped to repo names: %s",
                {"fpn_2": [int(feature_map.shape[0]), int(feature_map.shape[1]), int(feature_map.shape[2])]},
            )
            self._logged_signature = True
        return (int(rgb_image.height), int(rgb_image.width)), pyramid

    def _ensure_model_bundle(self) -> tuple[Any, Any, Any]:
        if self._model_bundle is not None:
            return self._model_bundle
        if self.official_checkpoint_path is not None:
            raise FrozenSamPyramidExtractorRuntimeError(
                "The original-SAM frozen mask-head extractor does not support official checkpoint override paths.",
                diagnostics={"official_checkpoint_path": self.official_checkpoint_path},
            )
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:  # pragma: no cover - runtime path
            raise FrozenSamPyramidExtractorRuntimeError(
                "Original SAM frozen feature extraction requires `transformers` with `SamModel` and `SamProcessor`."
            ) from exc
        device = _resolve_device(self.requested_device, torch)
        pretrained_kwargs: dict[str, Any] = {}
        if self.hf_token:
            pretrained_kwargs["token"] = self.hf_token
        try:
            processor = SamProcessor.from_pretrained(self.model_id, **pretrained_kwargs)
            model = SamModel.from_pretrained(self.model_id, **pretrained_kwargs).to(device)
            model.eval()
        except Exception as exc:  # pragma: no cover - runtime path
            raise FrozenSamPyramidExtractorRuntimeError(
                f"Failed to load original SAM frozen extractor for '{self.model_id}': {exc}",
                diagnostics={"model_id": self.model_id, "device": device},
            ) from exc
        self._model_bundle = (torch, processor, model)
        return self._model_bundle


class Sam2FrozenPyramidFeatureExtractor:
    """Expose the official SAM-2 predictor features as `fpn_2/fpn_1/fpn_0`."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SAM2_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        fast_cuda: bool = False,
        compile_image_encoder: bool = False,
    ) -> None:
        self.model_id = str(model_id)
        self.requested_device = str(device)
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self.fast_cuda = bool(fast_cuda)
        self.compile_image_encoder = bool(compile_image_encoder)
        self._predictor_bundle: tuple[Any, Any] | None = None
        self._logged_pyramid_signature = False

    def extract_sam_pyramid(self, image: Image.Image) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
        """Return the official SAM-2 image features as a named 3-level pyramid."""

        torch_module, predictor = self._ensure_predictor_bundle()
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        autocast_context = _resolve_autocast_context(
            torch_module=torch_module,
            requested_device=self.requested_device,
            fast_cuda=self.fast_cuda,
        )
        try:
            with autocast_context:
                predictor.set_image(image_array)
        except Exception as exc:  # pragma: no cover - runtime path
            raise FrozenSamPyramidExtractorRuntimeError(
                f"SAM-2 frozen pyramid feature extraction failed during set_image(): {exc}",
                diagnostics={"model_id": self.model_id, "device": self.requested_device},
            ) from exc

        features = getattr(predictor, "_features", None)
        if not isinstance(features, dict):
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 predictor did not expose the expected `_features` mapping after set_image().",
                diagnostics={"model_id": self.model_id},
            )
        if "image_embed" not in features or "high_res_feats" not in features:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 predictor features were missing `image_embed` and/or `high_res_feats`.",
                diagnostics={"available_feature_keys": sorted(features.keys())},
            )

        image_embed = features["image_embed"]
        high_res_feats = features["high_res_feats"]
        if not isinstance(high_res_feats, (list, tuple)):
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 predictor `high_res_feats` had an unexpected type.",
                diagnostics={"type": type(high_res_feats).__name__},
            )
        if len(high_res_feats) != 2:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 frozen mask-head path expects exactly two high-resolution feature levels.",
                diagnostics={"num_high_res_levels": len(high_res_feats)},
            )

        candidate_levels = [
            ("image_embed", image_embed),
            ("high_res_0", high_res_feats[0]),
            ("high_res_1", high_res_feats[1]),
        ]
        materialized_levels: list[tuple[int, int, int, np.ndarray, str]] = []
        for source_name, tensor in candidate_levels:
            shape = getattr(tensor, "shape", None)
            if shape is None or len(shape) != 4 or int(shape[0]) != 1:
                raise FrozenSamPyramidExtractorRuntimeError(
                    f"SAM-2 feature level `{source_name}` had unexpected shape {shape}; expected [1,C,H,W].",
                    diagnostics={"source_name": source_name, "shape": tuple(shape) if shape is not None else None},
                )
            feature_map = np.asarray(tensor[0].detach().cpu().to(dtype=torch_module.float32).numpy(), dtype=np.float32)
            if feature_map.ndim != 3:
                raise FrozenSamPyramidExtractorRuntimeError(
                    f"SAM-2 feature level `{source_name}` produced unexpected feature-map rank {feature_map.ndim}.",
                    diagnostics={"source_name": source_name, "shape": tuple(int(v) for v in feature_map.shape)},
                )
            channels, height, width = (int(feature_map.shape[0]), int(feature_map.shape[1]), int(feature_map.shape[2]))
            materialized_levels.append((height * width, height, width, feature_map, source_name))

        materialized_levels.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if len(materialized_levels) != 3:
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 frozen mask-head path expected exactly three feature levels after materialization.",
                diagnostics={"num_levels": len(materialized_levels)},
            )

        finest = materialized_levels[0][3]
        middle = materialized_levels[1][3]
        coarsest = materialized_levels[2][3]
        pyramid = {
            "fpn_2": coarsest,
            "fpn_1": middle,
            "fpn_0": finest,
        }
        if not self._logged_pyramid_signature:
            signature = {
                level_name: [int(feature_map.shape[0]), int(feature_map.shape[1]), int(feature_map.shape[2])]
                for level_name, feature_map in pyramid.items()
            }
            LOGGER.info("Discovered SAM-2 predictor pyramid levels mapped to repo names: %s", signature)
            self._logged_pyramid_signature = True
        return (int(image_array.shape[0]), int(image_array.shape[1])), pyramid

    def _ensure_predictor_bundle(self) -> tuple[Any, Any]:
        if self._predictor_bundle is not None:
            return self._predictor_bundle
        if self.official_checkpoint_path is not None:
            raise FrozenSamPyramidExtractorRuntimeError(
                "The SAM-2 frozen mask-head extractor does not support official checkpoint override paths.",
                diagnostics={"official_checkpoint_path": self.official_checkpoint_path},
            )
        try:
            import torch
            from sam2.build_sam import build_sam2_hf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - runtime path
            raise FrozenSamPyramidExtractorRuntimeError(
                "SAM-2 frozen mask-head extraction requires Meta's official `SAM-2` package. "
                "Install the `sam2-baseline` extra before running a SAM-2 backbone swap."
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
            raise FrozenSamPyramidExtractorRuntimeError(
                f"Failed to load SAM-2 frozen pyramid extractor for '{self.model_id}': {exc}",
                diagnostics={"model_id": self.model_id, "device": device},
            ) from exc

        self._predictor_bundle = (torch, predictor)
        return self._predictor_bundle


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise Sam2RuntimeError("CUDA was requested explicitly but no CUDA device is available.")
    if requested_device not in {"cpu", "cuda"}:
        raise Sam2RuntimeError(f"Unsupported device '{requested_device}'. Expected one of: auto, cpu, cuda.")
    return requested_device


def _configure_cuda_runtime(torch_module: Any, device: str, fast_cuda: bool) -> None:
    if device != "cuda" or not fast_cuda:
        return
    if torch_module.cuda.get_device_properties(0).major >= 8:
        torch_module.backends.cuda.matmul.allow_tf32 = True
        torch_module.backends.cudnn.allow_tf32 = True


def _resolve_autocast_context(torch_module: Any, requested_device: str, fast_cuda: bool):
    device = _resolve_device(requested_device, torch_module)
    if device == "cuda" and fast_cuda:
        return torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
    return nullcontext()
