from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID, Sam3Runner


FEATURE_CLUSTER_GLOBAL_SETTINGS: dict[str, float | int | str] = {
    "feature_source": "highest_resolution_backbone_fpn",
    "kmeans_max_iterations": 25,
    "kmeans_convergence_tolerance": 1e-4,
    "apply_refinement": True,
    "refine_confidence_threshold": 0.0,
}


class Sam3FeatureClusterGlobalRuntimeError(RuntimeError):
    """Raised when global SAM-feature clustering cannot produce valid masks."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class FeatureClusterGlobalRefinement:
    """Artifacts emitted by one global SAM-feature clustering pass.

    Attributes:
        cluster_label_map: Integer label map of shape ``(height, width)`` with values in
            ``{0, 1}``.
        rough_mask_a: Raw coarse mask for cluster A of shape ``(height, width)``.
        rough_mask_b: Raw coarse mask for cluster B of shape ``(height, width)``.
        refinement_applied: Whether official SAM mask refinement was applied after
            clustering. The current default setting is ``True``.
        refined_mask_a: Final output mask for cluster A. When refinement is disabled,
            this is the raw cluster-A mask.
        refined_mask_b: Final output mask for cluster B. When refinement is disabled,
            this is the raw cluster-B mask.
        refined_score_a: Official SAM confidence score for ``refined_mask_a`` or
            ``None`` when refinement is disabled.
        refined_score_b: Official SAM confidence score for ``refined_mask_b`` or
            ``None`` when refinement is disabled.
        refined_pair_selection_score: Joint score used to choose the final refined
            A/B pair from the two prompt-specific candidate sets.
        refined_pair_overlap_iou: IoU overlap between the chosen final A/B masks.
        cluster_pixel_count_a: Positive-pixel count for the raw cluster-A mask.
        cluster_pixel_count_b: Positive-pixel count for the raw cluster-B mask.
    """

    cluster_label_map: np.ndarray
    rough_mask_a: np.ndarray
    rough_mask_b: np.ndarray
    refinement_applied: bool
    refined_mask_a: np.ndarray
    refined_mask_b: np.ndarray
    refined_score_a: float | None
    refined_score_b: float | None
    refined_pair_selection_score: float | None
    refined_pair_overlap_iou: float | None
    cluster_pixel_count_a: int
    cluster_pixel_count_b: int


class Sam3FeatureClusterGlobalRunner:
    """Cluster global SAM dense features into two unlabeled texture groups."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.hf_token = hf_token
        self.official_checkpoint_path = official_checkpoint_path
        self.settings = dict(FEATURE_CLUSTER_GLOBAL_SETTINGS)
        if settings is not None:
            self.settings.update(settings)

        self._sam3_runner = Sam3Runner(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterGlobalRefinement:
        """Cluster normalized SAM dense features globally and optionally refine both masks."""

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
        label_map = _cluster_global_features_2way(
            dense_feature_map=dense_feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        rough_mask_a, rough_mask_b = _label_map_to_cluster_masks(label_map)
        cluster_pixel_count_a = int(rough_mask_a.sum())
        cluster_pixel_count_b = int(rough_mask_b.sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterGlobalRuntimeError(
                "feature_cluster_global produced an empty cluster mask after global feature grouping.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

        refinement_applied = bool(self.settings["apply_refinement"])
        if refinement_applied:
            candidates_a = self._collect_mask_prompt_candidates(
                base_state=base_state,
                coarse_mask=rough_mask_a,
                torch_module=official_backend.torch,
                official_backend=official_backend,
            )
            candidates_b = self._collect_mask_prompt_candidates(
                base_state=base_state,
                coarse_mask=rough_mask_b,
                torch_module=official_backend.torch,
                official_backend=official_backend,
            )
            selected_pair = _select_prompt_aligned_mask_pair(
                candidates_a=candidates_a,
                candidates_b=candidates_b,
                coarse_mask_a=np.asarray(rough_mask_a, dtype=bool),
                coarse_mask_b=np.asarray(rough_mask_b, dtype=bool),
            )
            refined_mask_a = selected_pair["candidate_a"]["mask"]
            refined_mask_b = selected_pair["candidate_b"]["mask"]
            refined_score_a = selected_pair["candidate_a"]["score"]
            refined_score_b = selected_pair["candidate_b"]["score"]
            refined_pair_selection_score = selected_pair["selection_score"]
            refined_pair_overlap_iou = selected_pair["overlap_iou"]
        else:
            refined_mask_a = np.asarray(rough_mask_a, dtype=bool)
            refined_mask_b = np.asarray(rough_mask_b, dtype=bool)
            refined_score_a = None
            refined_score_b = None
            refined_pair_selection_score = None
            refined_pair_overlap_iou = None

        return FeatureClusterGlobalRefinement(
            cluster_label_map=label_map,
            rough_mask_a=rough_mask_a,
            rough_mask_b=rough_mask_b,
            refinement_applied=refinement_applied,
            refined_mask_a=refined_mask_a,
            refined_mask_b=refined_mask_b,
            refined_score_a=float(refined_score_a) if refined_score_a is not None else None,
            refined_score_b=float(refined_score_b) if refined_score_b is not None else None,
            refined_pair_selection_score=(
                float(refined_pair_selection_score) if refined_pair_selection_score is not None else None
            ),
            refined_pair_overlap_iou=(
                float(refined_pair_overlap_iou) if refined_pair_overlap_iou is not None else None
            ),
            cluster_pixel_count_a=cluster_pixel_count_a,
            cluster_pixel_count_b=cluster_pixel_count_b,
        )

    def _collect_mask_prompt_candidates(
        self,
        base_state: dict[str, Any],
        coarse_mask: np.ndarray,
        torch_module: Any,
        official_backend: Any,
    ) -> list[dict[str, Any]]:
        if int(np.asarray(coarse_mask, dtype=bool).sum()) == 0:
            raise Sam3FeatureClusterGlobalRuntimeError(
                "feature_cluster_global produced an empty coarse cluster mask before SAM refinement."
            )

        prompt = (
            torch_module.as_tensor(coarse_mask, dtype=torch_module.float32, device=official_backend.device)[
                None, None, None
            ]
        )
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
            raise Sam3FeatureClusterGlobalRuntimeError(
                "feature_cluster_global refinement returned no masks after applying the cluster prompt."
            )

        return _collect_prompt_aligned_candidates(
            masks=masks,
            scores=scores,
            coarse_mask=np.asarray(coarse_mask, dtype=bool),
            expected_shape=(int(base_state["original_height"]), int(base_state["original_width"])),
        )


def _extract_dense_feature_map(backbone_out: dict[str, Any], target_size: tuple[int, int], torch_module: Any):
    feature_levels = backbone_out.get("backbone_fpn")
    if not feature_levels:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global could not find backbone_fpn features in the official SAM state."
        )

    highest_resolution = max(feature_levels, key=lambda tensor: int(tensor.shape[-2] * tensor.shape[-1]))
    if highest_resolution.ndim != 4 or highest_resolution.shape[0] < 1:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global expected a 4D feature map with a batch dimension, "
            f"got {tuple(highest_resolution.shape)}."
        )

    upsampled = torch_module.nn.functional.interpolate(
        highest_resolution[:1].to(dtype=torch_module.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    norms = upsampled.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return upsampled / norms


def _cluster_global_features_2way(dense_feature_map, torch_module: Any, settings: dict[str, float | int | str]) -> np.ndarray:
    channels, height, width = tuple(int(dimension) for dimension in dense_feature_map.shape)
    del channels
    vectors = dense_feature_map.reshape(dense_feature_map.shape[0], height * width).transpose(0, 1).contiguous()
    if vectors.shape[0] < 2:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global requires at least two spatial feature vectors to form two clusters."
        )

    centroid_a, centroid_b = _initialize_two_centroids(vectors=vectors, torch_module=torch_module)
    centroids = torch_module.stack([centroid_a, centroid_b], dim=0)
    labels = torch_module.full((vectors.shape[0],), fill_value=-1, dtype=torch_module.long, device=vectors.device)
    tolerance = float(settings["kmeans_convergence_tolerance"])
    max_iterations = int(settings["kmeans_max_iterations"])

    for _ in range(max_iterations):
        similarities = vectors @ centroids.transpose(0, 1)
        new_labels = similarities.argmax(dim=1)
        if int((new_labels == 0).sum().item()) == 0 or int((new_labels == 1).sum().item()) == 0:
            raise Sam3FeatureClusterGlobalRuntimeError(
                "feature_cluster_global collapsed to an empty cluster during global 2-way feature clustering."
            )

        updated_centroids = []
        for cluster_id in (0, 1):
            centroid = vectors[new_labels == cluster_id].mean(dim=0)
            centroid = centroid / centroid.norm().clamp_min(1e-6)
            updated_centroids.append(centroid)
        updated_centroids_tensor = torch_module.stack(updated_centroids, dim=0)

        centroid_shift = float((updated_centroids_tensor - centroids).norm().item())
        centroids = updated_centroids_tensor
        if torch_module.equal(new_labels, labels) or centroid_shift <= tolerance:
            labels = new_labels
            break
        labels = new_labels

    label_map = labels.reshape(height, width).detach().cpu().numpy().astype(np.int32)
    return label_map


def _initialize_two_centroids(vectors, torch_module: Any):
    mean_vector = vectors.mean(dim=0)
    mean_vector = mean_vector / mean_vector.norm().clamp_min(1e-6)
    first_index = int((vectors @ mean_vector).argmin().item())
    centroid_a = vectors[first_index]

    second_index = int((vectors @ centroid_a).argmin().item())
    centroid_b = vectors[second_index]
    similarity = float(torch_module.dot(centroid_a, centroid_b).item())
    if first_index == second_index or similarity >= (1.0 - 1e-6):
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global could not find two distinct feature centroids for global clustering."
        )
    return centroid_a, centroid_b


def _label_map_to_cluster_masks(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterGlobalRuntimeError(
            f"feature_cluster_global expected binary cluster labels (0, 1), got {unique_values}."
        )
    return labels == 0, labels == 1


def _coerce_refined_mask(mask_array: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask_array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global refinement returned an unexpected mask shape "
            f"{array.shape}; expected a 2D mask."
        )

    result = np.asarray(array, dtype=bool)
    if result.shape != expected_shape:
        raise Sam3FeatureClusterGlobalRuntimeError(
            f"feature_cluster_global refinement returned shape {result.shape}, expected {expected_shape}."
        )
    return result


def _collect_prompt_aligned_candidates(
    masks: Any,
    scores: Any,
    coarse_mask: np.ndarray,
    expected_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    masks_array = np.asarray(masks.detach().cpu().numpy())
    scores_array = np.asarray(scores.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if masks_array.ndim != 3:
        raise Sam3FeatureClusterGlobalRuntimeError(
            f"feature_cluster_global refinement expected a candidate-mask tensor, got {masks_array.shape}."
        )

    candidates: list[dict[str, Any]] = []
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
        candidates.append(candidate)

    if not candidates:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global refinement produced no valid candidate masks after prompt alignment."
        )
    candidates.sort(key=lambda candidate: (candidate["prompt_iou"], candidate["score"]), reverse=True)
    return candidates


def _select_prompt_aligned_mask_pair(
    candidates_a: list[dict[str, Any]],
    candidates_b: list[dict[str, Any]],
    coarse_mask_a: np.ndarray,
    coarse_mask_b: np.ndarray,
) -> dict[str, Any]:
    if not candidates_a or not candidates_b:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global refinement requires non-empty candidate sets for both prompts."
        )

    prompt_a = np.asarray(coarse_mask_a, dtype=bool)
    prompt_b = np.asarray(coarse_mask_b, dtype=bool)
    best_pair: dict[str, Any] | None = None
    for candidate_a in candidates_a:
        cross_iou_a_to_b = _binary_iou(candidate_a["mask"], prompt_b)
        for candidate_b in candidates_b:
            cross_iou_b_to_a = _binary_iou(candidate_b["mask"], prompt_a)
            overlap_iou = _binary_iou(candidate_a["mask"], candidate_b["mask"])
            selection_score = (
                (candidate_a["prompt_iou"] - cross_iou_a_to_b)
                + (candidate_b["prompt_iou"] - cross_iou_b_to_a)
                - overlap_iou
                + (0.05 * (candidate_a["score"] + candidate_b["score"]))
            )
            candidate_pair = {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "selection_score": float(selection_score),
                "overlap_iou": float(overlap_iou),
                "cross_iou_a_to_b": float(cross_iou_a_to_b),
                "cross_iou_b_to_a": float(cross_iou_b_to_a),
            }
            if best_pair is None or (
                candidate_pair["selection_score"],
                candidate_a["prompt_iou"] + candidate_b["prompt_iou"],
                -candidate_pair["overlap_iou"],
                candidate_a["score"] + candidate_b["score"],
            ) > (
                best_pair["selection_score"],
                best_pair["candidate_a"]["prompt_iou"] + best_pair["candidate_b"]["prompt_iou"],
                -best_pair["overlap_iou"],
                best_pair["candidate_a"]["score"] + best_pair["candidate_b"]["score"],
            ):
                best_pair = candidate_pair

    if best_pair is None:
        raise Sam3FeatureClusterGlobalRuntimeError(
            "feature_cluster_global refinement could not choose a valid non-degenerate A/B candidate pair."
        )
    return best_pair


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 1.0
