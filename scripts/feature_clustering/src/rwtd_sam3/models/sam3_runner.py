"""Runtime wrapper for prompt-conditioned SAM 3 inference backends.

This module hides the split between the public Transformers SAM 3 path and the
official Meta SAM 3 image backend used by oracle-point prompting. The evaluation
entrypoints use ``Sam3Runner`` so they can request either text-prompt or
oracle-point masks through one small interface.

Primary entrypoints:
- ``Sam3Runner.segment_text()``: predict one boolean mask from a text prompt.
- ``Sam3Runner.segment_oracle_points()``: predict one boolean mask from positive
  point prompts.

Inputs are one RGB ``PIL.Image.Image`` plus prompt text or point coordinates.
Outputs are boolean masks of shape ``(height, width)`` plus ``PredictionMetadata``
describing backend choice, prompt form, and kept instance scores. Missing model
dependencies, gated-model access failures, invalid point inputs, and unexpected
backend tensor shapes are all raised explicitly as ``Sam3RuntimeError``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/sam3"
MODEL_CARD_URL = "https://huggingface.co/facebook/sam3"


class Sam3RuntimeError(RuntimeError):
    """Raised when SAM 3 loading or inference fails."""


@dataclass(frozen=True)
class PredictionMetadata:
    """Metadata captured alongside a predicted mask."""

    backend: str
    prompt: str
    num_instances: int
    kept_scores: tuple[float, ...]


@dataclass(frozen=True)
class _TextBackend:
    model: Any
    processor: Any
    torch: Any
    device: str


@dataclass(frozen=True)
class _OfficialBackend:
    model: Any
    processor: Any
    torch: Any
    device: str


class Sam3Runner:
    """Thin runtime wrapper over the available SAM 3 backends."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self._text_backend: _TextBackend | None = None
        self._official_backend: _OfficialBackend | None = None

    def segment_text(
        self,
        image: Image.Image,
        prompt: str,
        score_threshold: float,
        mask_threshold: float,
    ) -> tuple[np.ndarray, PredictionMetadata]:
        """Predict a binary mask for a single text prompt using the Transformers backend."""

        return self.segment_text_batch(
            images=[image],
            prompts=[prompt],
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )[0]

    def segment_text_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        score_threshold: float,
        mask_threshold: float,
    ) -> list[tuple[np.ndarray, PredictionMetadata]]:
        """Predict binary masks for a batch of text prompts using the Transformers backend."""

        if len(images) != len(prompts):
            raise Sam3RuntimeError(
                f"segment_text_batch received {len(images)} images and {len(prompts)} prompts; expected equal lengths."
            )
        if not images:
            return []

        backend = self._ensure_text_backend()
        inputs = backend.processor(images=list(images), text=list(prompts), return_tensors="pt")
        target_sizes = inputs["original_sizes"].tolist()
        inputs = _move_batch_encoding_to_device(inputs, backend.device)

        with backend.torch.inference_mode():
            outputs = backend.model(**inputs)

        results = backend.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )
        if len(results) != len(prompts):
            raise Sam3RuntimeError(
                f"Text backend returned {len(results)} post-processed results for {len(prompts)} prompts."
            )

        predictions: list[tuple[np.ndarray, PredictionMetadata]] = []
        for prompt, target_size, result in zip(prompts, target_sizes, results, strict=True):
            predictions.append(_build_text_prediction(result=result, prompt=prompt, target_size=target_size))
        return predictions

    def segment_oracle_points(
        self,
        image: Image.Image,
        points: Sequence[tuple[int, int]],
        score_threshold: float,
    ) -> tuple[np.ndarray, PredictionMetadata]:
        """Predict a binary mask for one oracle point set using the official SAM 3 backend."""

        if not points:
            raise Sam3RuntimeError("Oracle-point evaluation requires at least one positive point.")

        backend = self._ensure_official_backend()
        width, height = image.size

        state = backend.processor.set_image(image, state={})
        backend.processor.set_confidence_threshold(score_threshold)
        if "language_features" not in state["backbone_out"]:
            text_outputs = backend.model.backbone.forward_text(["visual"], device=backend.device)
            state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = backend.model._get_dummy_prompt()

        normalized_points = np.asarray(points, dtype=np.float32)
        normalized_points[:, 0] /= float(width)
        normalized_points[:, 1] /= float(height)

        point_tensor = backend.torch.tensor(
            normalized_points,
            device=backend.device,
            dtype=backend.torch.float32,
        ).view(len(points), 1, 2)
        label_tensor = backend.torch.ones(
            (len(points), 1),
            device=backend.device,
            dtype=backend.torch.long,
        )
        state["geometric_prompt"].append_points(point_tensor, label_tensor)
        state = backend.processor._forward_grounding(state)

        masks = state["masks"]
        scores = state["scores"]

        if hasattr(scores, "detach"):
            scores_cpu = scores.detach().cpu()
            scores_list = scores_cpu.tolist()
        else:
            scores_cpu = np.asarray(scores)
            scores_list = scores_cpu.tolist()

        if hasattr(masks, "detach"):
            masks = masks.detach().cpu()

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim != 3:
            raise Sam3RuntimeError(f"Unexpected oracle-point mask tensor shape: {tuple(masks.shape)}")

        if masks.shape[0] == 0:
            best_mask = np.zeros((height, width), dtype=bool)
        else:
            best_index = int(scores_cpu.argmax().item()) if len(scores_list) else 0
            best_mask = np.asarray(masks[best_index], dtype=bool)

        if best_mask.shape != (height, width):
            raise Sam3RuntimeError(
                f"Oracle-point backend returned shape {best_mask.shape}, expected {(height, width)}."
            )

        return best_mask, PredictionMetadata(
            backend="official-sam3",
            prompt=f"{len(points)} positive points",
            num_instances=int(masks.shape[0]),
            kept_scores=tuple(float(score) for score in scores_list),
        )

    def _ensure_text_backend(self) -> _TextBackend:
        if self._text_backend is not None:
            return self._text_backend

        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:  # pragma: no cover - exercised via runtime
            raise Sam3RuntimeError(
                "Text-prompt evaluation requires 'torch', 'torchvision', and "
                "'transformers>=5.3.0'. "
                "Install the text evaluation extras before running inference."
            ) from exc

        device = _resolve_device(self.requested_device, torch_module=torch)
        pretrained_kwargs = {"token": self.hf_token} if self.hf_token else {}
        try:
            model = Sam3Model.from_pretrained(self.model_id, **pretrained_kwargs)
            processor = Sam3Processor.from_pretrained(self.model_id, **pretrained_kwargs)
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise _translate_model_loading_error(self.model_id, exc, backend_name="transformers") from exc

        model = model.to(device)
        model.eval()
        self._text_backend = _TextBackend(model=model, processor=processor, torch=torch, device=device)
        LOGGER.info("Loaded text backend from %s on %s.", self.model_id, device)
        return self._text_backend

    def _ensure_official_backend(self) -> _OfficialBackend:
        if self._official_backend is not None:
            return self._official_backend

        try:
            import torch
            self._ensure_torchvision_nms_stub(torch)
            from sam3 import model_builder as sam3_model_builder
            from sam3.model import necks as sam3_necks
            from sam3.model import vitdet as sam3_vitdet
            from sam3.model.sam3_image_processor import Sam3Processor as OfficialSam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:  # pragma: no cover - exercised via runtime
            raise Sam3RuntimeError(_format_official_import_error(exc)) from exc

        device = _resolve_device(self.requested_device, torch_module=torch)
        if device == "cpu":
            _apply_official_sam3_cpu_builder_patch(model_builder_module=sam3_model_builder, torch_module=torch)
        load_from_hf = self.official_checkpoint_path is None
        try:
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=self.official_checkpoint_path,
                load_from_HF=load_from_hf,
                compile=False,
            )
            if hasattr(model, "float"):
                model = model.float()

            # Some official SAM3 package versions do not expose a backbone.processor object.
            # The fallback patch keeps every live path in float32 so coarse feature extraction
            # does not trip over mixed bfloat16/float weights on CUDA.
            if (
                hasattr(model, "backbone")
                and hasattr(model.backbone, "processor")
                and hasattr(model.backbone.processor, "transform")
            ):
                original_processor_transform = model.backbone.processor.transform

                def _processor_transform_float32_safe(image: Any, *args: Any, **kwargs: Any) -> Any:
                    transformed = original_processor_transform(image, *args, **kwargs)
                    if hasattr(transformed, "float"):
                        transformed = transformed.float()
                    return transformed

                model.backbone.processor.transform = _processor_transform_float32_safe
            original_neck_forward = sam3_necks.Sam3DualViTDetNeck.forward

            def _neck_forward_float32_safe(self, tensor_list: Any) -> Any:
                if isinstance(tensor_list, (list, tuple)):
                    tensor_list = [tensor.float() if hasattr(tensor, "float") else tensor for tensor in tensor_list]
                elif hasattr(tensor_list, "float"):
                    tensor_list = tensor_list.float()
                return original_neck_forward(self, tensor_list)

            sam3_necks.Sam3DualViTDetNeck.forward = _neck_forward_float32_safe
            def _mlp_forward_float32_safe(self, x: Any) -> Any:
                if hasattr(x, "float"):
                    x = x.float()
                x = sam3_vitdet.addmm_act(type(self.act), self.fc1, x)
                x = self.drop1(x)
                x = self.norm(x)
                if hasattr(x, "float"):
                    x = x.float()
                x = self.fc2(x)
                x = self.drop2(x)
                return x

            sam3_vitdet.Mlp.forward = _mlp_forward_float32_safe
            original_forward_image = model.backbone.forward_image

            def _forward_image_float32_safe(samples: Any, *args: Any, **kwargs: Any) -> Any:
                if hasattr(samples, "float"):
                    samples = samples.float()
                with torch.no_grad():
                    try:
                        autocast_context = torch.autocast(device_type=device, enabled=False)
                    except Exception:
                        autocast_context = nullcontext()
                    with autocast_context:
                        return original_forward_image(samples, *args, **kwargs)

            model.backbone.forward_image = _forward_image_float32_safe
        except Exception as exc:  # pragma: no cover - exercised via runtime
            raise _translate_model_loading_error(self.model_id, exc, backend_name="official-sam3") from exc

        processor = OfficialSam3Processor(model=model, device=device, confidence_threshold=0.5)
        self._official_backend = _OfficialBackend(model=model, processor=processor, torch=torch, device=device)
        LOGGER.info("Loaded official SAM 3 backend on %s.", device)
        return self._official_backend

    @staticmethod
    def _ensure_torchvision_nms_stub(torch_module: Any) -> None:
        """Define a minimal torchvision NMS operator stub when the wheel is mismatched.

        The official SAM3 package imports torchvision during module initialization.
        Some environments ship a torchvision build whose Python registration path
        expects the `torchvision::nms` operator to exist already, but the compiled
        extension is absent or incompatible. Defining the operator schema up front
        keeps the import path alive without changing any runtime SAM3 behavior for
        this repo, which does not call torchvision NMS directly.
        """

        if hasattr(torch_module.ops, "torchvision") and hasattr(torch_module.ops.torchvision, "nms"):
            return
        try:
            library = torch_module.library.Library("torchvision", "DEF")
            library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            return


def _resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise Sam3RuntimeError("CUDA was requested explicitly but no CUDA device is available.")
    if requested_device not in {"cpu", "cuda"}:
        raise Sam3RuntimeError(f"Unsupported device '{requested_device}'. Expected one of: auto, cpu, cuda.")
    return requested_device


def _apply_official_sam3_cpu_builder_patch(*, model_builder_module: Any, torch_module: Any) -> None:
    """Disable CUDA-only position-encoding precompute when SAM3 is built on CPU."""

    if getattr(model_builder_module, "_rwtd_cpu_builder_patch_applied", False):
        return
    original_create_position_encoding = model_builder_module._create_position_encoding

    def _create_position_encoding_cpu_safe(precompute_resolution=None):
        if precompute_resolution is not None and not torch_module.cuda.is_available():
            LOGGER.warning(
                "Official SAM3 CPU fallback disables position-encoding CUDA precompute during model construction."
            )
            precompute_resolution = None
        return original_create_position_encoding(precompute_resolution=precompute_resolution)

    def _create_transformer_decoder_cpu_safe(*args: Any, **kwargs: Any):
        LOGGER.warning(
            "Official SAM3 CPU fallback disables decoder CUDA cache precompute during model construction."
        )
        kwargs.pop("use_fa3", None)
        decoder_layer = model_builder_module.TransformerDecoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            cross_attention=model_builder_module.MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
            ),
            n_heads=8,
            use_text_cross_attention=True,
        )
        return model_builder_module.TransformerDecoder(
            layer=decoder_layer,
            num_layers=6,
            num_queries=200,
            return_intermediate=True,
            box_refine=True,
            num_o2m_queries=0,
            dac=True,
            boxRPB="log",
            d_model=256,
            frozen=False,
            interaction_layer=None,
            dac_use_selfatt_ln=True,
            resolution=None,
            stride=None,
            use_act_checkpoint=True,
            presence_token=True,
        )

    model_builder_module._create_position_encoding = _create_position_encoding_cpu_safe
    model_builder_module._create_transformer_decoder = _create_transformer_decoder_cpu_safe
    model_builder_module._rwtd_cpu_builder_patch_applied = True


def _translate_model_loading_error(model_id: str, error: Exception, backend_name: str) -> Sam3RuntimeError:
    message = str(error)
    lowered = message.lower()
    if "restricted" in lowered or "gated" in lowered or "401" in lowered or "403" in lowered:
        return Sam3RuntimeError(
            f"Access to '{model_id}' is restricted for the {backend_name} backend. "
            f"Request access at {MODEL_CARD_URL} and authenticate with Hugging Face "
            "before running inference."
        )
    if "does not appear to have a file named" in lowered and model_id == DEFAULT_MODEL_ID:
        return Sam3RuntimeError(
            f"Failed to resolve gated weights for '{model_id}' via the {backend_name} backend. "
            "The public SAM 3 repository exposes gated model files, and unauthenticated or unauthorized "
            "access can surface as a missing-file error. Request access at "
            f"{MODEL_CARD_URL} and authenticate with `hf auth login` or an `HF_TOKEN` before retrying."
        )
    return Sam3RuntimeError(
        f"Failed to load '{model_id}' for the {backend_name} backend: {message}"
    )


def _format_official_import_error(error: ImportError) -> str:
    missing_name = getattr(error, "name", None)
    if missing_name and missing_name != "sam3":
        return (
            "Oracle-point evaluation requires the official Meta 'sam3' package and its "
            f"runtime dependencies. Missing import: '{missing_name}'. Install the "
            "`oracle-eval` extra or add that dependency explicitly before retrying."
        )
    return (
        "Oracle-point evaluation requires the official Meta 'sam3' package and its runtime "
        "dependencies. Install the `oracle-eval` extra in a compatible environment, or run "
        "the text protocol instead."
    )


def _move_batch_encoding_to_device(batch_encoding: Any, device: str) -> Any:
    """Move a transformers ``BatchEncoding`` payload onto the selected device."""

    for key, value in batch_encoding.items():
        if hasattr(value, "to"):
            batch_encoding[key] = value.to(device, non_blocking=True)
    return batch_encoding


def _build_text_prediction(
    result: dict[str, Any],
    prompt: str,
    target_size: Sequence[int],
) -> tuple[np.ndarray, PredictionMetadata]:
    """Convert one post-processed SAM-3 text result into repo-native outputs."""

    masks = result["masks"]
    scores = result["scores"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().tolist()

    if len(masks) == 0:
        height, width = target_size
        merged_mask = np.zeros((height, width), dtype=bool)
    else:
        merged_mask = np.any(np.asarray(masks, dtype=bool), axis=0)

    return merged_mask, PredictionMetadata(
        backend="transformers",
        prompt=prompt,
        num_instances=len(scores),
        kept_scores=tuple(float(score) for score in scores),
    )
