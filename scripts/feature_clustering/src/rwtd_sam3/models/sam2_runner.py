"""Runtime wrapper for the official SAM-2 automatic mask generator.

This module encapsulates the Meta SAM-2 model loading path plus the small
variant preset table used by the repository's RWTD and official TextureSAM
baselines. Evaluation modules call ``Sam2BaselineRunner.generate_masks()`` to
obtain a sorted list of raw binary mask proposals for one image.

Primary entrypoints:
- ``Sam2BaselineRunner.generate_masks()``: run one SAM-2 baseline variant on one
  RGB image.
- ``SAM2_BASELINE_PRESETS``: repository-level settings for ``sam2`` and
  ``sam2_star``.

Inputs are one RGB ``PIL.Image.Image`` and a variant name. Outputs are sorted
``Sam2MaskPrediction`` records containing boolean masks, scores, areas, and box
metadata. Missing SAM-2 dependencies, invalid device requests, invalid batch
sizes, and generation failures are raised explicitly as ``Sam2RuntimeError``.
"""

from __future__ import annotations

from contextlib import nullcontext
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-small"

SAM2_BASELINE_PRESETS: dict[str, dict[str, float | int | str]] = {
    "sam2": {
        "points_per_side": 32,
        "stability_score_thresh": 0.95,
    },
    "sam2_star": {
        "points_per_side": 64,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.2,
        "mask_threshold": 0.0,
        "min_mask_region_area": 0,
        "multimask_output": False,
    },
}


class Sam2RuntimeError(RuntimeError):
    """Raised when SAM 2 loading or inference fails."""


@dataclass(frozen=True)
class Sam2MaskPrediction:
    """One raw SAM-2 automatic mask prediction."""

    segmentation: np.ndarray
    area: int
    predicted_iou: float
    stability_score: float
    bbox_xywh: tuple[float, float, float, float]


class Sam2BaselineRunner:
    """Thin wrapper around Meta's official SAM-2 automatic mask generator."""

    def __init__(
        self,
        model_id: str = DEFAULT_SAM2_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        points_per_batch: int | None = None,
        fast_cuda: bool = False,
        compile_image_encoder: bool = False,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self.requested_points_per_batch = points_per_batch
        self.fast_cuda = fast_cuda
        self.compile_image_encoder = compile_image_encoder
        self._model_bundle: tuple[Any, Any] | None = None
        self._generators: dict[str, Any] = {}

    def generate_masks(self, image: Image.Image, variant: str) -> list[Sam2MaskPrediction]:
        """Run SAM-2 AMG on one image for the selected baseline variant."""

        if variant not in SAM2_BASELINE_PRESETS:
            raise Sam2RuntimeError(
                f"Unsupported SAM-2 baseline variant '{variant}'. Expected one of: "
                f"{', '.join(sorted(SAM2_BASELINE_PRESETS))}."
            )

        torch_module, _ = self._ensure_model()
        generator = self._ensure_generator(variant)
        image_array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        autocast_context = _resolve_autocast_context(
            torch_module=torch_module,
            requested_device=self.requested_device,
            fast_cuda=self.fast_cuda,
        )
        try:
            with autocast_context:
                annotations = generator.generate(image_array)
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise Sam2RuntimeError(f"SAM-2 mask generation failed: {exc}") from exc

        predictions = [
            Sam2MaskPrediction(
                segmentation=np.asarray(annotation["segmentation"], dtype=bool),
                area=int(annotation["area"]),
                predicted_iou=float(annotation["predicted_iou"]),
                stability_score=float(annotation["stability_score"]),
                bbox_xywh=tuple(float(value) for value in annotation["bbox"]),
            )
            for annotation in annotations
        ]
        predictions.sort(key=lambda item: item.predicted_iou, reverse=True)
        return predictions

    def _ensure_generator(self, variant: str):
        generator = self._generators.get(variant)
        if generator is not None:
            return generator

        torch_module, sam_model = self._ensure_model()
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as exc:  # pragma: no cover - exercised via runtime
            raise Sam2RuntimeError(
                "SAM-2 baseline evaluation requires the official Meta 'SAM-2' package. "
                "Install the `sam2-baseline` extra before running this command."
            ) from exc

        generator_kwargs = dict(SAM2_BASELINE_PRESETS[variant])
        generator_kwargs["points_per_batch"] = _resolve_points_per_batch(
            requested_points_per_batch=self.requested_points_per_batch,
            requested_device=self.requested_device,
            torch_module=torch_module,
        )
        generator = SAM2AutomaticMaskGenerator(
            sam_model,
            output_mode="binary_mask",
            **generator_kwargs,
        )
        self._generators[variant] = generator
        LOGGER.info(
            "Loaded SAM-2 AMG variant %s from %s with %s on %s.",
            variant,
            self.model_id,
            generator_kwargs,
            _resolve_device(self.requested_device, torch_module),
        )
        return generator

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model_bundle is not None:
            return self._model_bundle

        try:
            import torch
            from sam2.build_sam import build_sam2_hf
        except ImportError as exc:  # pragma: no cover - exercised via runtime
            raise Sam2RuntimeError(
                "SAM-2 baseline evaluation requires Meta's official 'SAM-2' package plus its "
                "runtime dependencies. Install the `sam2-baseline` extra before running this command."
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
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise Sam2RuntimeError(f"Failed to load SAM-2 model '{self.model_id}': {exc}") from exc

        self._model_bundle = (torch, sam_model)
        return self._model_bundle


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise Sam2RuntimeError("CUDA was requested explicitly but no CUDA device is available.")
    if requested_device not in {"cpu", "cuda"}:
        raise Sam2RuntimeError(f"Unsupported device '{requested_device}'. Expected one of: auto, cpu, cuda.")
    return requested_device


def _resolve_points_per_batch(
    requested_points_per_batch: int | None,
    requested_device: str,
    torch_module: Any,
) -> int:
    if requested_points_per_batch is not None:
        if requested_points_per_batch < 1:
            raise Sam2RuntimeError(
                f"points_per_batch must be at least 1, got {requested_points_per_batch}."
            )
        return requested_points_per_batch

    device = _resolve_device(requested_device, torch_module)
    return 256 if device == "cuda" else 64


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
