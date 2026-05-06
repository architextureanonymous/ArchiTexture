from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    Sam3FeatureClusterCoarseToFineGlobalRunner,
    _align_mask_pair_to_reference,
    _cluster_coarsest_level_2way,
    _cluster_pooled_coarsest_level_2way,
)
from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID


@dataclass(frozen=True)
class MaskPromptInvarianceBranch:
    """One prompt branch in the SAM mask-prompt invariance side experiment."""

    name: str
    prompt_mask_a: np.ndarray
    prompt_mask_b: np.ndarray
    final_mask_a: np.ndarray
    final_mask_b: np.ndarray
    refined_score_a: float
    refined_score_b: float
    selected_candidate_rank_a: int
    selected_candidate_rank_b: int
    selected_candidate_index_a: int
    selected_candidate_index_b: int


@dataclass(frozen=True)
class MaskPromptInvarianceRefinement:
    """Artifacts for the raw-vs-pooled-vs-random SAM mask-prompt side experiment."""

    raw_branch: MaskPromptInvarianceBranch
    pooled_branch: MaskPromptInvarianceBranch
    random_branch: MaskPromptInvarianceBranch
    raw_to_pooled_final_alignment_used: str
    raw_to_pooled_final_mean_iou: float
    random_to_pooled_final_alignment_used: str
    random_to_pooled_final_mean_iou: float
    raw_to_pooled_aligned_final_mask_a: np.ndarray
    raw_to_pooled_aligned_final_mask_b: np.ndarray
    random_to_pooled_aligned_final_mask_a: np.ndarray
    random_to_pooled_aligned_final_mask_b: np.ndarray
    coarsest_level_name: str
    coarsest_level_resolution: tuple[int, int]
    pooled_grid_resolution: tuple[int, int]


class Sam3MaskPromptInvarianceRunner(Sam3FeatureClusterCoarseToFineGlobalRunner):
    """Compare raw, pooled, and random mask prompts under the same SAM refinement path."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_prompt_invariance(self, image: Image.Image) -> MaskPromptInvarianceRefinement:
        """Run raw, pooled, and random prompt branches through identical SAM refinement."""

        official_backend, base_state, image_size, feature_levels = self._prepare_feature_levels(image)
        coarsest_level = feature_levels[0]

        raw_native_label_map = _cluster_coarsest_level_2way(
            feature_map=coarsest_level.feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        pooled_native_label_map, pooled_grid_resolution = _cluster_pooled_coarsest_level_2way(
            feature_map=coarsest_level.feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        random_native_label_map = _build_random_label_map_like(
            reference_label_map=pooled_native_label_map,
            image=image,
        )

        raw_run = self._run_from_initial_label_map(
            initial_label_map=raw_native_label_map,
            feature_levels=feature_levels[:1],
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )
        pooled_run = self._run_from_initial_label_map(
            initial_label_map=pooled_native_label_map,
            feature_levels=feature_levels[:1],
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )
        random_run = self._run_from_initial_label_map(
            initial_label_map=random_native_label_map,
            feature_levels=feature_levels[:1],
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )

        raw_alignment = _align_mask_pair_to_reference(
            candidate_mask_a=raw_run.refined_mask_a,
            candidate_mask_b=raw_run.refined_mask_b,
            reference_mask_a=pooled_run.refined_mask_a,
            reference_mask_b=pooled_run.refined_mask_b,
        )
        random_alignment = _align_mask_pair_to_reference(
            candidate_mask_a=random_run.refined_mask_a,
            candidate_mask_b=random_run.refined_mask_b,
            reference_mask_a=pooled_run.refined_mask_a,
            reference_mask_b=pooled_run.refined_mask_b,
        )

        return MaskPromptInvarianceRefinement(
            raw_branch=_convert_run_to_branch("raw", raw_run),
            pooled_branch=_convert_run_to_branch("pooled", pooled_run),
            random_branch=_convert_run_to_branch("random", random_run),
            raw_to_pooled_final_alignment_used=str(raw_alignment["alignment_used"]),
            raw_to_pooled_final_mean_iou=float(raw_alignment["chosen_mean_iou"]),
            random_to_pooled_final_alignment_used=str(random_alignment["alignment_used"]),
            random_to_pooled_final_mean_iou=float(random_alignment["chosen_mean_iou"]),
            raw_to_pooled_aligned_final_mask_a=np.asarray(raw_alignment["mask_a"], dtype=bool),
            raw_to_pooled_aligned_final_mask_b=np.asarray(raw_alignment["mask_b"], dtype=bool),
            random_to_pooled_aligned_final_mask_a=np.asarray(random_alignment["mask_a"], dtype=bool),
            random_to_pooled_aligned_final_mask_b=np.asarray(random_alignment["mask_b"], dtype=bool),
            coarsest_level_name=coarsest_level.name,
            coarsest_level_resolution=coarsest_level.resolution,
            pooled_grid_resolution=pooled_grid_resolution,
        )


def _convert_run_to_branch(name: str, run: Any) -> MaskPromptInvarianceBranch:
    return MaskPromptInvarianceBranch(
        name=name,
        prompt_mask_a=np.asarray(run.prompt_mask_a, dtype=bool),
        prompt_mask_b=np.asarray(run.prompt_mask_b, dtype=bool),
        final_mask_a=np.asarray(run.refined_mask_a, dtype=bool),
        final_mask_b=np.asarray(run.refined_mask_b, dtype=bool),
        refined_score_a=float(run.refined_score_a),
        refined_score_b=float(run.refined_score_b),
        selected_candidate_rank_a=int(run.selected_candidate_rank_a),
        selected_candidate_rank_b=int(run.selected_candidate_rank_b),
        selected_candidate_index_a=int(run.selected_candidate_index_a),
        selected_candidate_index_b=int(run.selected_candidate_index_b),
    )


def _build_random_label_map_like(reference_label_map: np.ndarray, image: Image.Image) -> np.ndarray:
    """Create a deterministic random binary prompt with the same label balance as the reference."""

    labels = np.asarray(reference_label_map, dtype=np.int32)
    height, width = labels.shape
    total = height * width
    ones = int((labels == 1).sum())
    if ones <= 0 or ones >= total:
        ones = total // 2

    flat = np.zeros(total, dtype=np.int32)
    flat[:ones] = 1
    image_bytes = np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()
    rng = np.random.default_rng(zlib.adler32(image_bytes) & 0xFFFFFFFF)
    rng.shuffle(flat)
    return flat.reshape(height, width)
