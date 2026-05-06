"""Runtime wrapper for the Transformers SAM-3 automatic-mask pipeline.

This module provides the lightweight baseline used by the ``default`` and
``dense`` automatic-mask experiments. It loads the Hugging Face model and
processor once, runs the ``mask-generation`` pipeline, and converts the returned
outputs into sorted ``Sam3AutomaticMaskPrediction`` records.

Primary entrypoints:
- ``Sam3AutomaticMaskRunner.generate_masks()``: run one automatic-mask preset on
  one RGB image.
- ``SAM3_AUTO_PRESETS``: repository-level settings for ``default`` and
  ``dense`` proposal density.

Inputs are one RGB ``PIL.Image.Image`` and a preset name. Outputs are boolean
proposal masks with areas, scores, and bounding boxes. Missing dependencies,
gated-model access failures, unexpected mask shapes, and invalid device
requests are raised explicitly as ``Sam3AutomaticMaskRuntimeError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID, MODEL_CARD_URL


LOGGER = logging.getLogger(__name__)

DEFAULT_SAM3_AUTO_MODEL_ID = DEFAULT_MODEL_ID

SAM3_AUTO_PRESETS: dict[str, dict[str, float | int]] = {
    "default": {
        "points_per_crop": 32,
        "stability_score_thresh": 0.95,
    },
    "dense": {
        "points_per_crop": 64,
        "stability_score_thresh": 0.2,
    },
}


class Sam3AutomaticMaskRuntimeError(RuntimeError):
    """Raised when SAM-3 automatic mask generation cannot run."""


@dataclass(frozen=True)
class Sam3AutomaticMaskPrediction:
    """One raw SAM-3 automatic mask prediction."""

    segmentation: np.ndarray
    area: int
    score: float
    bbox_xyxy: tuple[float, float, float, float]


class Sam3AutomaticMaskRunner:
    """Thin wrapper around the Transformers SAM-3 mask-generation pipeline."""

    def __init__(
        self,
        model_id: str = DEFAULT_SAM3_AUTO_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self._pipeline: Any | None = None
        self._device_name: str | None = None

    def generate_masks(self, image: Image.Image, variant: str) -> list[Sam3AutomaticMaskPrediction]:
        """Run SAM-3 automatic mask generation on one image."""

        if variant not in SAM3_AUTO_PRESETS:
            raise Sam3AutomaticMaskRuntimeError(
                f"Unsupported SAM-3 automatic-mask variant '{variant}'. Expected one of: "
                f"{', '.join(sorted(SAM3_AUTO_PRESETS))}."
            )

        mask_pipeline = self._ensure_pipeline()
        preset = dict(SAM3_AUTO_PRESETS[variant])
        try:
            outputs = mask_pipeline(
                image,
                output_bboxes_mask=True,
                **preset,
            )
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise Sam3AutomaticMaskRuntimeError(f"SAM-3 automatic mask generation failed: {exc}") from exc

        masks = outputs.get("masks", [])
        scores = _coerce_score_list(outputs.get("scores", []))
        boxes = _coerce_box_list(outputs.get("bounding_boxes", []))

        predictions: list[Sam3AutomaticMaskPrediction] = []
        for index, mask in enumerate(masks):
            segmentation = _coerce_mask(mask, expected_shape=(image.size[1], image.size[0]))
            score = float(scores[index]) if index < len(scores) else 0.0
            bbox_xyxy = boxes[index] if index < len(boxes) else (0.0, 0.0, 0.0, 0.0)
            predictions.append(
                Sam3AutomaticMaskPrediction(
                    segmentation=segmentation,
                    area=int(segmentation.sum()),
                    score=score,
                    bbox_xyxy=bbox_xyxy,
                )
            )

        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import AutoModelForMaskGeneration, Sam3Processor, pipeline
        except ImportError as exc:  # pragma: no cover - exercised via runtime
            raise Sam3AutomaticMaskRuntimeError(
                "SAM-3 automatic mask generation requires 'torch', 'torchvision', and "
                "'transformers>=5.3.0'. Install the project dependencies before running this command."
            ) from exc

        device_name = _resolve_device(self.requested_device, torch)
        pipeline_device: int | str = 0 if device_name == "cuda" else "cpu"
        try:
            model = AutoModelForMaskGeneration.from_pretrained(
                self.model_id,
                token=self.hf_token,
            )
            processor = Sam3Processor.from_pretrained(
                self.model_id,
                token=self.hf_token,
            )
            self._pipeline = pipeline(
                task="mask-generation",
                model=model,
                image_processor=processor.image_processor,
                device=pipeline_device,
                dtype="auto",
            )
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise _translate_model_loading_error(self.model_id, exc) from exc

        self._device_name = device_name
        LOGGER.info("Loaded SAM-3 automatic mask pipeline from %s on %s.", self.model_id, device_name)
        return self._pipeline


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise Sam3AutomaticMaskRuntimeError("CUDA was requested explicitly but no CUDA device is available.")
    if requested_device not in {"cpu", "cuda"}:
        raise Sam3AutomaticMaskRuntimeError(
            f"Unsupported device '{requested_device}'. Expected one of: auto, cpu, cuda."
        )
    return requested_device


def _translate_model_loading_error(model_id: str, error: Exception) -> Sam3AutomaticMaskRuntimeError:
    message = str(error)
    lowered = message.lower()
    if "restricted" in lowered or "gated" in lowered or "401" in lowered or "403" in lowered:
        return Sam3AutomaticMaskRuntimeError(
            f"Access to '{model_id}' is restricted for automatic mask generation. "
            f"Request access at {MODEL_CARD_URL} and authenticate with Hugging Face before running inference."
        )
    if "can't load processor" in lowered or "can't load image processor" in lowered:
        return Sam3AutomaticMaskRuntimeError(
            f"Failed to initialize automatic mask generation for '{model_id}' because the required "
            "processor artifacts are not cached locally. Authenticate and run once with network access "
            f"to cache the gated files from {MODEL_CARD_URL}, then retry offline."
        )
    return Sam3AutomaticMaskRuntimeError(f"Failed to load SAM-3 model '{model_id}': {message}")


def _coerce_score_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    return [float(score) for score in value]


def _coerce_box_list(value: Any) -> list[tuple[float, float, float, float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    return [tuple(float(component) for component in box) for box in value]


def _coerce_mask(mask: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if hasattr(mask, "detach"):
        array = mask.detach().cpu().numpy()
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise Sam3AutomaticMaskRuntimeError(f"Unexpected mask array shape {array.shape} from SAM-3 automatic masks.")

    result = np.asarray(array, dtype=bool)
    if result.shape != expected_shape:
        raise Sam3AutomaticMaskRuntimeError(
            f"SAM-3 automatic mask returned shape {result.shape}, expected {expected_shape}."
        )
    return result
