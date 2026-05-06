from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.eval.metrics import boundary_from_region_masks
from rwtd_sam3.models.sam3_auto_runner import (
    DEFAULT_SAM3_AUTO_MODEL_ID,
    Sam3AutomaticMaskRunner,
)
from rwtd_sam3.models.sam3_runner import Sam3Runner


FEATURE_MASK_SETTINGS: dict[str, float | int | str] = {
    "base_variant": "dense",
    "prototype_erosion_radius": 2,
    "local_dilation_radius": 8,
    "single_mask_outer_dilation_radius": 24,
    "min_region_pixels": 128,
    "preferred_region_pixels": 1024,
    "min_prototype_pixels": 32,
    "min_boundary_contact": 16,
    "margin_threshold": 0.05,
    "refine_confidence_threshold": 0.0,
}


class Sam3FeatureMaskRuntimeError(RuntimeError):
    """Raised when the feature-mask protocol cannot produce a valid refinement."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
        dense_prediction_masks: tuple[np.ndarray, ...] | None = None,
        dense_prediction_scores: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.dense_prediction_masks = dense_prediction_masks or ()
        self.dense_prediction_scores = dense_prediction_scores or ()


@dataclass(frozen=True)
class FeatureMaskRefinement:
    """Artifacts emitted by one feature-mask refinement pass.

    Attributes:
        dense_prediction_masks: Dense automatic SAM masks of shape ``(height, width)``.
        dense_prediction_scores: Dense automatic SAM confidence scores aligned with
            ``dense_prediction_masks``.
        selection_mode: ``"pair"`` when both support regions come from dense proposals or
            ``"single_mask_complement"`` when the second support region is synthesized as
            a local complement around one dense mask.
        selected_pair_ids: Pair of support-region ids used for the local refinement.
            The second id is ``None`` when the background support region is synthetic.
        selected_pair_masks: Two support masks aligned with ``selected_pair_ids``. In
            single-mask-complement mode, the second mask is the synthetic local
            background support region rather than a dense SAM proposal.
        pair_score: Final support-selection score.
        pair_feature_distance: Cosine-distance term used in support scoring.
        pair_boundary_contact: Shared-boundary pixel count used in support scoring.
        pair_union_pixels: Pixel area of the selected support-region union.
        local_region_pixels: Pixel area of the dilated local refinement neighborhood.
        coarse_margin: Prototype-similarity margin for region A over region B.
        coarse_prompt_a: Binary coarse prompt for the first refined region.
        coarse_prompt_b: Binary coarse prompt for the second refined region.
        refined_mask_a: Refined binary mask for the first region.
        refined_mask_b: Refined binary mask for the second region.
        refined_score_a: Score of the selected refined mask for region A.
        refined_score_b: Score of the selected refined mask for region B. This is
            ``None`` when the second output is derived as the complement of the
            refined foreground mask rather than an independently refined SAM mask.
    """

    dense_prediction_masks: tuple[np.ndarray, ...]
    dense_prediction_scores: tuple[float, ...]
    selection_mode: str
    selected_pair_ids: tuple[int, int | None]
    selected_pair_masks: tuple[np.ndarray, np.ndarray]
    pair_score: float
    pair_feature_distance: float
    pair_boundary_contact: int
    pair_union_pixels: int
    local_region_pixels: int
    coarse_margin: np.ndarray
    coarse_prompt_a: np.ndarray
    coarse_prompt_b: np.ndarray
    refined_mask_a: np.ndarray
    refined_mask_b: np.ndarray
    refined_score_a: float
    refined_score_b: float | None


class Sam3FeatureMaskRunner:
    """Build a local coarse mask from dense SAM features and re-run SAM with it."""

    def __init__(
        self,
        model_id: str = DEFAULT_SAM3_AUTO_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self.settings = dict(FEATURE_MASK_SETTINGS)
        if settings is not None:
            self.settings.update(settings)

        self._dense_runner = Sam3AutomaticMaskRunner(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
        )
        self._sam3_runner = Sam3Runner(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )

    def generate_feature_masks(self, image: Image.Image) -> FeatureMaskRefinement:
        """Generate dense masks, choose a local pair, and refine with a mask prompt."""

        dense_predictions = self._dense_runner.generate_masks(
            image=image,
            variant=str(self.settings["base_variant"]),
        )
        dense_masks = tuple(np.asarray(prediction.segmentation, dtype=bool) for prediction in dense_predictions)
        dense_scores = tuple(float(prediction.score) for prediction in dense_predictions)
        if not dense_predictions:
            raise Sam3FeatureMaskRuntimeError(
                "feature_mask requires at least one dense automatic mask to seed the feature comparison; got 0.",
                diagnostics={"dense_num_predicted_masks": len(dense_predictions)},
                dense_prediction_masks=dense_masks,
                dense_prediction_scores=dense_scores,
            )

        official_backend = self._sam3_runner._ensure_official_backend()
        official_backend.processor.set_confidence_threshold(float(self.settings["refine_confidence_threshold"]))

        base_state = official_backend.processor.set_image(image, state={})
        if "language_features" not in base_state["backbone_out"]:
            text_outputs = official_backend.model.backbone.forward_text(["visual"], device=official_backend.device)
            base_state["backbone_out"].update(text_outputs)

        dense_feature_map = _extract_dense_feature_map(
            backbone_out=base_state["backbone_out"],
            target_size=(image.size[1], image.size[0]),
            torch_module=official_backend.torch,
        )
        try:
            support = _select_feature_support(
                dense_masks=dense_masks,
                dense_scores=dense_scores,
                dense_feature_map=dense_feature_map,
                torch_module=official_backend.torch,
                device=official_backend.device,
                settings=self.settings,
            )
            coarse_margin, coarse_prompt_a, coarse_prompt_b, local_region = _build_coarse_prompts(
                dense_feature_map=dense_feature_map,
                mask_a=support["mask_a"],
                mask_b=support["mask_b"],
                proto_mask_a=support["proto_mask_a"],
                proto_mask_b=support["proto_mask_b"],
                local_region=support["local_region"],
                torch_module=official_backend.torch,
                settings=self.settings,
            )

            refined_mask_a, refined_score_a = self._refine_with_mask_prompt(
                base_state=base_state,
                coarse_prompt=coarse_prompt_a,
                torch_module=official_backend.torch,
                official_backend=official_backend,
            )
            if support["selection_mode"] == "single_mask_complement":
                refined_mask_b = np.logical_not(refined_mask_a)
                refined_score_b = None
            else:
                refined_mask_b, refined_score_b = self._refine_with_mask_prompt(
                    base_state=base_state,
                    coarse_prompt=coarse_prompt_b,
                    torch_module=official_backend.torch,
                    official_backend=official_backend,
                )
        except Sam3FeatureMaskRuntimeError as exc:
            diagnostics = dict(exc.diagnostics)
            diagnostics.setdefault("dense_num_predicted_masks", len(dense_masks))
            raise Sam3FeatureMaskRuntimeError(
                str(exc),
                diagnostics=diagnostics,
                dense_prediction_masks=dense_masks,
                dense_prediction_scores=dense_scores,
            ) from exc

        return FeatureMaskRefinement(
            dense_prediction_masks=dense_masks,
            dense_prediction_scores=dense_scores,
            selection_mode=str(support["selection_mode"]),
            selected_pair_ids=(support["mask_index_a"], support["mask_index_b"]),
            selected_pair_masks=(
                np.asarray(support["mask_a"].detach().cpu().numpy(), dtype=bool),
                np.asarray(support["mask_b"].detach().cpu().numpy(), dtype=bool),
            ),
            pair_score=float(support["pair_score"]),
            pair_feature_distance=float(support["feature_distance"]),
            pair_boundary_contact=int(support["boundary_contact"]),
            pair_union_pixels=int(support["pair_union_pixels"]),
            local_region_pixels=int(local_region.sum().item()),
            coarse_margin=np.asarray(coarse_margin.detach().cpu().numpy(), dtype=np.float32),
            coarse_prompt_a=np.asarray(coarse_prompt_a.detach().cpu().numpy(), dtype=bool),
            coarse_prompt_b=np.asarray(coarse_prompt_b.detach().cpu().numpy(), dtype=bool),
            refined_mask_a=refined_mask_a,
            refined_mask_b=refined_mask_b,
            refined_score_a=float(refined_score_a),
            refined_score_b=float(refined_score_b) if refined_score_b is not None else None,
        )

    def _refine_with_mask_prompt(
        self,
        base_state: dict[str, Any],
        coarse_prompt,
        torch_module: Any,
        official_backend: Any,
    ) -> tuple[np.ndarray, float]:
        if int(coarse_prompt.sum().item()) == 0:
            raise Sam3FeatureMaskRuntimeError(
                "feature_mask produced an empty coarse prompt after feature-margin thresholding."
            )

        prompt = coarse_prompt.to(dtype=torch_module.float32, device=official_backend.device)[None, None, None]
        prompt_labels = torch_module.ones((1, 1), dtype=torch_module.long, device=official_backend.device)
        state = {
            "original_height": base_state["original_height"],
            "original_width": base_state["original_width"],
            "backbone_out": base_state["backbone_out"],
            "geometric_prompt": official_backend.model._get_dummy_prompt(),
        }
        state["geometric_prompt"].append_masks(prompt, labels=prompt_labels)
        state = official_backend.processor._forward_grounding(state)

        scores = state.get("scores")
        masks = state.get("masks")
        if scores is None or masks is None or int(scores.numel()) == 0 or int(masks.numel()) == 0:
            raise Sam3FeatureMaskRuntimeError(
                "feature_mask refinement returned no masks after applying the coarse prompt."
            )

        best_candidate = _select_prompt_aligned_mask(
            masks=masks,
            scores=scores,
            coarse_mask=np.asarray(coarse_prompt.detach().cpu().numpy(), dtype=bool),
            expected_shape=(int(base_state["original_height"]), int(base_state["original_width"])),
        )
        return best_candidate["mask"], float(best_candidate["score"])


def _extract_dense_feature_map(backbone_out: dict[str, Any], target_size: tuple[int, int], torch_module: Any):
    feature_levels = backbone_out.get("backbone_fpn")
    if not feature_levels:
        raise Sam3FeatureMaskRuntimeError("feature_mask could not find backbone_fpn features in the official SAM state.")

    highest_resolution = max(feature_levels, key=lambda tensor: int(tensor.shape[-2] * tensor.shape[-1]))
    if highest_resolution.ndim != 4 or highest_resolution.shape[0] < 1:
        raise Sam3FeatureMaskRuntimeError(
            f"feature_mask expected a 4D feature map with batch dimension, got {tuple(highest_resolution.shape)}."
        )

    upsampled = torch_module.nn.functional.interpolate(
        highest_resolution[:1].to(dtype=torch_module.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    norms = upsampled.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return upsampled / norms


def _select_feature_support(
    dense_masks: tuple[np.ndarray, ...],
    dense_scores: tuple[float, ...],
    dense_feature_map,
    torch_module: Any,
    device: str,
    settings: dict[str, float | int | str],
) -> dict[str, Any]:
    if len(dense_masks) >= 2:
        try:
            return _select_best_pair(
                dense_masks=dense_masks,
                dense_feature_map=dense_feature_map,
                torch_module=torch_module,
                device=device,
                settings=settings,
            )
        except Sam3FeatureMaskRuntimeError as exc:
            pair_error = str(exc)
    else:
        pair_error = None

    try:
        support = _select_best_single_mask_complement(
            dense_masks=dense_masks,
            dense_scores=dense_scores,
            dense_feature_map=dense_feature_map,
            torch_module=torch_module,
            device=device,
            settings=settings,
        )
    except Sam3FeatureMaskRuntimeError as exc:
        if pair_error is None:
            raise
        raise Sam3FeatureMaskRuntimeError(f"{pair_error} Fallback single-mask complement also failed: {exc}") from exc

    if pair_error is not None:
        support["pair_selection_error"] = pair_error
    return support


def _select_best_pair(
    dense_masks: tuple[np.ndarray, ...],
    dense_feature_map,
    torch_module: Any,
    device: str,
    settings: dict[str, float | int | str],
) -> dict[str, Any]:
    min_region_pixels = int(settings["min_region_pixels"])
    min_prototype_pixels = int(settings["min_prototype_pixels"])
    prototype_erosion_radius = int(settings["prototype_erosion_radius"])
    preferred_region_pixels = int(settings["preferred_region_pixels"])
    min_boundary_contact = int(settings["min_boundary_contact"])
    local_dilation_radius = int(settings["local_dilation_radius"])

    best_pair: dict[str, Any] | None = None
    candidate_errors = 0
    for mask_index_a in range(len(dense_masks)):
        for mask_index_b in range(mask_index_a + 1, len(dense_masks)):
            mask_a = torch_module.as_tensor(dense_masks[mask_index_a], device=device, dtype=torch_module.bool)
            mask_b = torch_module.as_tensor(dense_masks[mask_index_b], device=device, dtype=torch_module.bool)

            area_a = int(mask_a.sum().item())
            area_b = int(mask_b.sum().item())
            if min(area_a, area_b) < min_region_pixels:
                continue

            boundary_contact = int(
                boundary_from_region_masks(
                    np.asarray(mask_a.detach().cpu().numpy(), dtype=bool),
                    np.asarray(mask_b.detach().cpu().numpy(), dtype=bool),
                ).sum()
            )
            if boundary_contact < min_boundary_contact:
                continue

            proto_mask_a = _erode(mask_a, prototype_erosion_radius, torch_module)
            proto_mask_b = _erode(mask_b, prototype_erosion_radius, torch_module)
            if int(proto_mask_a.sum().item()) < min_prototype_pixels or int(proto_mask_b.sum().item()) < min_prototype_pixels:
                candidate_errors += 1
                continue

            proto_a = _mean_feature(dense_feature_map, proto_mask_a, torch_module)
            proto_b = _mean_feature(dense_feature_map, proto_mask_b, torch_module)
            feature_distance = float((1.0 - torch_module.dot(proto_a, proto_b)).item())
            size_factor = min(1.0, float(min(area_a, area_b)) / float(preferred_region_pixels))
            pair_score = feature_distance * float(boundary_contact) * size_factor
            pair_union_pixels = int(torch_module.logical_or(mask_a, mask_b).sum().item())
            local_region = _dilate(torch_module.logical_or(mask_a, mask_b), local_dilation_radius, torch_module)

            if best_pair is None or pair_score > float(best_pair["pair_score"]):
                best_pair = {
                    "selection_mode": "pair",
                    "mask_index_a": mask_index_a,
                    "mask_index_b": mask_index_b,
                    "mask_a": mask_a,
                    "mask_b": mask_b,
                    "proto_mask_a": proto_mask_a,
                    "proto_mask_b": proto_mask_b,
                    "local_region": local_region,
                    "feature_distance": feature_distance,
                    "boundary_contact": boundary_contact,
                    "pair_score": pair_score,
                    "pair_union_pixels": pair_union_pixels,
                }

    if best_pair is None:
        raise Sam3FeatureMaskRuntimeError(
            "feature_mask could not find a valid adjacent dense-mask pair "
            f"(pairs rejected after prototype/boundary checks: {candidate_errors})."
        )
    return best_pair


def _build_coarse_prompts(
    dense_feature_map,
    mask_a,
    mask_b,
    proto_mask_a,
    proto_mask_b,
    local_region,
    torch_module: Any,
    settings: dict[str, float | int | str],
):
    margin_threshold = float(settings["margin_threshold"])

    proto_a = _mean_feature(dense_feature_map, proto_mask_a, torch_module)
    proto_b = _mean_feature(dense_feature_map, proto_mask_b, torch_module)

    sim_a = (dense_feature_map * proto_a[:, None, None]).sum(dim=0)
    sim_b = (dense_feature_map * proto_b[:, None, None]).sum(dim=0)
    coarse_margin = torch_module.where(local_region, sim_a - sim_b, torch_module.zeros_like(sim_a))

    coarse_prompt_a = torch_module.logical_and(local_region, coarse_margin > margin_threshold)
    coarse_prompt_b = torch_module.logical_and(local_region, coarse_margin < -margin_threshold)
    if int(coarse_prompt_a.sum().item()) == 0 or int(coarse_prompt_b.sum().item()) == 0:
        raise Sam3FeatureMaskRuntimeError(
            "feature_mask could not build non-empty coarse prompts for both selected regions."
        )

    return coarse_margin, coarse_prompt_a, coarse_prompt_b, local_region


def _select_best_single_mask_complement(
    dense_masks: tuple[np.ndarray, ...],
    dense_scores: tuple[float, ...],
    dense_feature_map,
    torch_module: Any,
    device: str,
    settings: dict[str, float | int | str],
) -> dict[str, Any]:
    min_region_pixels = int(settings["min_region_pixels"])
    min_prototype_pixels = int(settings["min_prototype_pixels"])
    prototype_erosion_radius = int(settings["prototype_erosion_radius"])
    preferred_region_pixels = int(settings["preferred_region_pixels"])
    min_boundary_contact = int(settings["min_boundary_contact"])
    outer_radius = int(settings["single_mask_outer_dilation_radius"])

    best_support: dict[str, Any] | None = None
    candidate_errors = 0
    for mask_index, dense_mask in enumerate(dense_masks):
        mask_a = torch_module.as_tensor(dense_mask, device=device, dtype=torch_module.bool)
        area_a = int(mask_a.sum().item())
        if area_a < min_region_pixels:
            continue

        proto_mask_a = _erode(mask_a, prototype_erosion_radius, torch_module)
        if int(proto_mask_a.sum().item()) < min_prototype_pixels:
            candidate_errors += 1
            continue

        local_region = _dilate(mask_a, outer_radius, torch_module)
        mask_b = torch_module.logical_and(local_region, torch_module.logical_not(mask_a))
        proto_mask_b = _erode(mask_b, prototype_erosion_radius, torch_module)
        if int(proto_mask_b.sum().item()) < min_prototype_pixels:
            candidate_errors += 1
            continue

        boundary_contact = int(
            boundary_from_region_masks(
                np.asarray(mask_a.detach().cpu().numpy(), dtype=bool),
                np.asarray(mask_b.detach().cpu().numpy(), dtype=bool),
            ).sum()
        )
        if boundary_contact < min_boundary_contact:
            candidate_errors += 1
            continue

        proto_a = _mean_feature(dense_feature_map, proto_mask_a, torch_module)
        proto_b = _mean_feature(dense_feature_map, proto_mask_b, torch_module)
        feature_distance = float((1.0 - torch_module.dot(proto_a, proto_b)).item())
        size_factor = min(1.0, float(area_a) / float(preferred_region_pixels))
        dense_score = float(dense_scores[mask_index]) if mask_index < len(dense_scores) else 1.0
        pair_score = feature_distance * float(boundary_contact) * size_factor * max(dense_score, 1e-6)
        pair_union_pixels = int(torch_module.logical_or(mask_a, mask_b).sum().item())

        if best_support is None or pair_score > float(best_support["pair_score"]):
            best_support = {
                "selection_mode": "single_mask_complement",
                "mask_index_a": mask_index,
                "mask_index_b": None,
                "mask_a": mask_a,
                "mask_b": mask_b,
                "proto_mask_a": proto_mask_a,
                "proto_mask_b": proto_mask_b,
                "local_region": local_region,
                "feature_distance": feature_distance,
                "boundary_contact": boundary_contact,
                "pair_score": pair_score,
                "pair_union_pixels": pair_union_pixels,
            }

    if best_support is None:
        raise Sam3FeatureMaskRuntimeError(
            "feature_mask could not build a valid single-mask complement support region "
            f"(candidates rejected after prototype/boundary checks: {candidate_errors})."
        )
    return best_support


def _mean_feature(dense_feature_map, mask, torch_module: Any):
    if int(mask.sum().item()) == 0:
        raise Sam3FeatureMaskRuntimeError("feature_mask prototype extraction received an empty mask.")
    feature_values = dense_feature_map[:, mask]
    prototype = feature_values.mean(dim=1)
    prototype = prototype / prototype.norm().clamp_min(1e-6)
    return prototype.to(dtype=torch_module.float32)


def _coerce_refined_mask(mask_array: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask_array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise Sam3FeatureMaskRuntimeError(
            f"feature_mask refinement returned an unexpected mask shape {array.shape}; expected a 2D mask."
        )

    result = np.asarray(array, dtype=bool)
    if result.shape != expected_shape:
        raise Sam3FeatureMaskRuntimeError(
            f"feature_mask refinement returned shape {result.shape}, expected {expected_shape}."
        )
    return result


def _select_prompt_aligned_mask(
    masks: Any,
    scores: Any,
    coarse_mask: np.ndarray,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    masks_array = np.asarray(masks.detach().cpu().numpy())
    scores_array = np.asarray(scores.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if masks_array.ndim != 3:
        raise Sam3FeatureMaskRuntimeError(
            f"feature_mask refinement expected a candidate-mask tensor, got {masks_array.shape}."
        )

    best_candidate: dict[str, Any] | None = None
    prompt_mask = np.asarray(coarse_mask, dtype=bool)
    for index, candidate_mask in enumerate(masks_array):
        normalized_mask = _coerce_refined_mask(candidate_mask, expected_shape=expected_shape)
        intersection = int(np.logical_and(normalized_mask, prompt_mask).sum())
        union = int(np.logical_or(normalized_mask, prompt_mask).sum())
        prompt_iou = float(intersection / union) if union else 1.0
        score = float(scores_array[index]) if index < len(scores_array) else 0.0
        candidate = {
            "mask": normalized_mask,
            "score": score,
            "prompt_iou": prompt_iou,
            "index": index,
        }
        if best_candidate is None or (candidate["prompt_iou"], candidate["score"]) > (
            best_candidate["prompt_iou"],
            best_candidate["score"],
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise Sam3FeatureMaskRuntimeError(
            "feature_mask refinement produced no valid candidate masks after prompt alignment."
        )
    return best_candidate


def _dilate(mask, radius: int, torch_module: Any):
    if radius <= 0:
        return mask
    kernel_size = (radius * 2) + 1
    pooled = torch_module.nn.functional.max_pool2d(
        mask.to(dtype=torch_module.float32)[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    return pooled[0, 0] > 0.5


def _erode(mask, radius: int, torch_module: Any):
    if radius <= 0:
        return mask
    inverted = _dilate(torch_module.logical_not(mask), radius, torch_module)
    return torch_module.logical_not(inverted)
