from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from rwtd_sam3.models.sam3_runner import DEFAULT_MODEL_ID, Sam3Runner


FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS: dict[str, float | int | str] = {
    "feature_source": "backbone_fpn",
    "kmeans_max_iterations": 25,
    "kmeans_convergence_tolerance": 1e-4,
    "boundary_band_radius": 2,
    "refinement_iterations_per_level": 2,
    "refine_confidence_threshold": 0.0,
}
FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS,
    "coarsest_init_pool_kernel_size": 3,
    "coarsest_init_pool_stride": 3,
}
FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_SETTINGS,
    "positionality_top_channel_count": 8,
    "null_noise_seed": 0,
    "null_noise_std": 4.0,
    "null_blur_radius": 12.0,
    "null_axis_threshold_flag": 0.9,
    "flip_consistency_content_threshold": 0.9,
    "flip_consistency_axis_threshold": 0.75,
}
FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS: dict[str, float | int | str] = {
    **FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS,
}


class Sam3FeatureClusterCoarseToFineGlobalRuntimeError(RuntimeError):
    """Raised when coarse-to-fine SAM-feature clustering cannot produce valid masks."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class _FeatureLevel:
    """One normalized SAM FPN feature level kept at native resolution."""

    name: str
    feature_map: Any
    resolution: tuple[int, int]


@dataclass(frozen=True)
class _MultiscaleClusterRun:
    """Intermediate coarse-to-fine run state before GT-based evaluation."""

    coarsest_label_map: np.ndarray
    level_label_maps: tuple[np.ndarray, ...]
    level_names: tuple[str, ...]
    level_resolutions: tuple[tuple[int, int], ...]
    rough_mask_a: np.ndarray
    rough_mask_b: np.ndarray
    refined_mask_a: np.ndarray
    refined_mask_b: np.ndarray
    refined_score_a: float
    refined_score_b: float
    refined_pair_selection_score: float
    refined_pair_overlap_iou: float
    cluster_pixel_count_a: int
    cluster_pixel_count_b: int
    prompt_mask_a: np.ndarray
    prompt_mask_b: np.ndarray
    prompt_candidates_a: tuple[dict[str, Any], ...]
    prompt_candidates_b: tuple[dict[str, Any], ...]
    selected_candidate_rank_a: int
    selected_candidate_rank_b: int
    selected_candidate_index_a: int
    selected_candidate_index_b: int


@dataclass(frozen=True)
class FeatureClusterCoarseToFineGlobalRefinement:
    """Artifacts emitted by one coarse-to-fine multiscale SAM-feature pass.

    Attributes:
        coarsest_label_map: Image-space binary label map from the coarsest feature level.
        level_label_maps: Image-space binary label maps after each feature level, ordered
            from coarsest to finest.
        level_names: Feature-level names aligned with ``level_label_maps``.
        level_resolutions: Native ``(height, width)`` per feature level.
        rough_mask_a: Final coarse binary mask for cluster A before SAM refinement.
        rough_mask_b: Final coarse binary mask for cluster B before SAM refinement.
        refined_mask_a: Final refined SAM mask for cluster A.
        refined_mask_b: Final refined SAM mask for cluster B.
        refined_score_a: Official SAM confidence score for cluster A.
        refined_score_b: Official SAM confidence score for cluster B.
        refined_pair_selection_score: Joint score used to choose the final refined A/B
            pair from the two prompt-specific candidate sets.
        refined_pair_overlap_iou: IoU overlap between the chosen final refined masks.
        cluster_pixel_count_a: Positive-pixel count for the final coarse cluster-A mask.
        cluster_pixel_count_b: Positive-pixel count for the final coarse cluster-B mask.
    """

    coarsest_label_map: np.ndarray
    level_label_maps: tuple[np.ndarray, ...]
    level_names: tuple[str, ...]
    level_resolutions: tuple[tuple[int, int], ...]
    rough_mask_a: np.ndarray
    rough_mask_b: np.ndarray
    refined_mask_a: np.ndarray
    refined_mask_b: np.ndarray
    refined_score_a: float
    refined_score_b: float
    refined_pair_selection_score: float
    refined_pair_overlap_iou: float
    cluster_pixel_count_a: int
    cluster_pixel_count_b: int
    coarsest_native_label_map: np.ndarray | None = None
    coarsest_init_mode: str = "raw_global"
    coarsest_pool_kernel_size: int | None = None
    coarsest_pool_stride: int | None = None
    pooled_grid_resolution: tuple[int, int] | None = None
    coarsest_component_stats: dict[str, Any] | None = None
    comparison_baseline_coarsest_label_map: np.ndarray | None = None
    comparison_baseline_coarsest_native_label_map: np.ndarray | None = None
    comparison_baseline_final_mask_a: np.ndarray | None = None
    comparison_baseline_final_mask_b: np.ndarray | None = None
    comparison_baseline_component_stats: dict[str, Any] | None = None
    comparison_baseline_alignment_used: str | None = None
    comparison_baseline_direct_mean_iou: float | None = None
    comparison_baseline_swapped_mean_iou: float | None = None
    comparison_baseline_chosen_mean_iou: float | None = None
    prompt_mask_a: np.ndarray | None = None
    prompt_mask_b: np.ndarray | None = None
    prompt_candidates_a: tuple[dict[str, Any], ...] | None = None
    prompt_candidates_b: tuple[dict[str, Any], ...] | None = None
    selected_candidate_rank_a: int | None = None
    selected_candidate_rank_b: int | None = None
    selected_candidate_index_a: int | None = None
    selected_candidate_index_b: int | None = None
    comparison_baseline_prompt_mask_a: np.ndarray | None = None
    comparison_baseline_prompt_mask_b: np.ndarray | None = None
    comparison_baseline_prompt_candidates_a: tuple[dict[str, Any], ...] | None = None
    comparison_baseline_prompt_candidates_b: tuple[dict[str, Any], ...] | None = None
    comparison_baseline_selected_candidate_rank_a: int | None = None
    comparison_baseline_selected_candidate_rank_b: int | None = None
    comparison_baseline_selected_candidate_index_a: int | None = None
    comparison_baseline_selected_candidate_index_b: int | None = None
    comparison_baseline_raw_final_mask_a: np.ndarray | None = None
    comparison_baseline_raw_final_mask_b: np.ndarray | None = None
    multiscale_refinement_applied: bool = True
    sam_refinement_applied: bool = True
    raw_diagnostic_native_label_map: np.ndarray | None = None
    raw_diagnostic_label_map: np.ndarray | None = None
    baseline_branch_native_label_map: np.ndarray | None = None
    baseline_branch_label_map: np.ndarray | None = None
    baseline_branch_mask_a: np.ndarray | None = None
    baseline_branch_mask_b: np.ndarray | None = None
    flip_avg_native_label_map: np.ndarray | None = None
    flip_avg_label_map: np.ndarray | None = None
    flip_avg_mask_a: np.ndarray | None = None
    flip_avg_mask_b: np.ndarray | None = None
    projection_removed_native_label_map: np.ndarray | None = None
    projection_removed_label_map: np.ndarray | None = None
    null_constant_label_map: np.ndarray | None = None
    null_noise_label_map: np.ndarray | None = None
    null_blur_label_map: np.ndarray | None = None
    label_positionality_diagnostics: dict[str, Any] | None = None
    feature_positionality_diagnostics: dict[str, Any] | None = None
    projection_removed_feature_positionality_diagnostics: dict[str, Any] | None = None
    baseline_branch_diagnostics: dict[str, Any] | None = None
    projection_removed_branch_diagnostics: dict[str, Any] | None = None
    flip_avg_branch_diagnostics: dict[str, Any] | None = None
    flip_test_diagnostics: dict[str, Any] | None = None
    null_test_diagnostics: dict[str, Any] | None = None
    recommended_mitigation: str | None = None
    recommendation_reason: str | None = None
    edge_debiased_native_label_map: np.ndarray | None = None
    edge_debiased_label_map: np.ndarray | None = None
    edge_debiased_mask_a: np.ndarray | None = None
    edge_debiased_mask_b: np.ndarray | None = None
    flip_avg_plus_edge_debiased_native_label_map: np.ndarray | None = None
    flip_avg_plus_edge_debiased_label_map: np.ndarray | None = None
    flip_avg_plus_edge_debiased_mask_a: np.ndarray | None = None
    flip_avg_plus_edge_debiased_mask_b: np.ndarray | None = None
    raw_edge_positionality_diagnostics: dict[str, Any] | None = None
    raw_feature_edge_positionality_diagnostics: dict[str, Any] | None = None
    baseline_branch_edge_diagnostics: dict[str, Any] | None = None
    flip_avg_branch_edge_diagnostics: dict[str, Any] | None = None
    edge_debiased_feature_edge_positionality_diagnostics: dict[str, Any] | None = None
    edge_debiased_branch_edge_diagnostics: dict[str, Any] | None = None
    flip_avg_feature_edge_positionality_diagnostics: dict[str, Any] | None = None
    flip_avg_plus_edge_debiased_feature_edge_positionality_diagnostics: dict[str, Any] | None = None
    flip_avg_plus_edge_debiased_branch_edge_diagnostics: dict[str, Any] | None = None


class Sam3FeatureClusterCoarseToFineGlobalRunner:
    """Cluster the coarsest SAM feature level and refine labels only at finer levels."""

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
        self.settings = dict(FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_SETTINGS)
        if settings is not None:
            self.settings.update(settings)

        self._sam3_runner = Sam3Runner(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
        )

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Run coarsest-only clustering followed by boundary-band refinement on finer levels."""

        official_backend, base_state, image_size, feature_levels = self._prepare_feature_levels(image)
        coarsest_native_label_map = _cluster_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        run = self._run_from_initial_label_map(
            initial_label_map=coarsest_native_label_map,
            feature_levels=feature_levels,
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=run.coarsest_label_map,
            level_label_maps=run.level_label_maps,
            level_names=run.level_names,
            level_resolutions=run.level_resolutions,
            rough_mask_a=run.rough_mask_a,
            rough_mask_b=run.rough_mask_b,
            refined_mask_a=run.refined_mask_a,
            refined_mask_b=run.refined_mask_b,
            refined_score_a=run.refined_score_a,
            refined_score_b=run.refined_score_b,
            refined_pair_selection_score=run.refined_pair_selection_score,
            refined_pair_overlap_iou=run.refined_pair_overlap_iou,
            cluster_pixel_count_a=run.cluster_pixel_count_a,
            cluster_pixel_count_b=run.cluster_pixel_count_b,
            coarsest_native_label_map=coarsest_native_label_map,
            coarsest_init_mode="raw_global",
            coarsest_component_stats=_compute_label_map_component_stats(coarsest_native_label_map),
            prompt_mask_a=run.prompt_mask_a,
            prompt_mask_b=run.prompt_mask_b,
            prompt_candidates_a=run.prompt_candidates_a,
            prompt_candidates_b=run.prompt_candidates_b,
            selected_candidate_rank_a=run.selected_candidate_rank_a,
            selected_candidate_rank_b=run.selected_candidate_rank_b,
            selected_candidate_index_a=run.selected_candidate_index_a,
            selected_candidate_index_b=run.selected_candidate_index_b,
        )

    def _prepare_feature_levels(
        self,
        image: Image.Image,
    ) -> tuple[Any, dict[str, Any], tuple[int, int], list[_FeatureLevel]]:
        official_backend = self._sam3_runner._ensure_official_backend()
        return self._prepare_feature_levels_with_backend(image, official_backend)

    def extract_pooled_feature_map_for_visualization(self, image: Image.Image):
        """Expose the coarsest feature map used by the pooled PCA overlay."""

        official_backend, _, _, feature_levels = self._prepare_feature_levels(image)
        return feature_levels[0].feature_map

    def _prepare_feature_levels_with_backend(
        self,
        image: Image.Image,
        official_backend: Any,
    ) -> tuple[Any, dict[str, Any], tuple[int, int], list[_FeatureLevel]]:
        official_backend.processor.set_confidence_threshold(float(self.settings["refine_confidence_threshold"]))

        base_state = official_backend.processor.set_image(image, state={})
        if "language_features" not in base_state["backbone_out"]:
            text_outputs = official_backend.model.backbone.forward_text(["visual"], device=official_backend.device)
            base_state["backbone_out"].update(text_outputs)

        image_size = (image.size[1], image.size[0])
        feature_levels = _extract_multiscale_sam_features(
            backbone_out=base_state["backbone_out"],
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        if not feature_levels:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global could not extract any usable FPN feature levels."
            )
        return official_backend, base_state, image_size, feature_levels

    def _run_from_initial_label_map(
        self,
        initial_label_map: np.ndarray,
        feature_levels: list[_FeatureLevel],
        image_size: tuple[int, int],
        base_state: dict[str, Any],
        official_backend: Any,
    ) -> _MultiscaleClusterRun:
        level_names: list[str] = []
        level_resolutions: list[tuple[int, int]] = []
        image_space_label_maps: list[np.ndarray] = []
        current_label_map = np.asarray(initial_label_map, dtype=np.int32)

        level_names.append(feature_levels[0].name)
        level_resolutions.append(feature_levels[0].resolution)
        image_space_label_maps.append(
            _upsample_label_map(
                current_label_map,
                target_size=image_size,
                torch_module=official_backend.torch,
            )
        )

        for level in feature_levels[1:]:
            upsampled_labels = _upsample_label_map(
                current_label_map,
                target_size=level.resolution,
                torch_module=official_backend.torch,
            )
            current_label_map = _refine_labels_with_finer_features(
                feature_map=level.feature_map,
                initial_label_map=upsampled_labels,
                torch_module=official_backend.torch,
                settings=self.settings,
            )
            level_names.append(level.name)
            level_resolutions.append(level.resolution)
            image_space_label_maps.append(
                _upsample_label_map(
                    current_label_map,
                    target_size=image_size,
                    torch_module=official_backend.torch,
                )
            )

        final_label_map = image_space_label_maps[-1]
        rough_mask_a, rough_mask_b = _label_map_to_cluster_masks(final_label_map)
        cluster_pixel_count_a = int(rough_mask_a.sum())
        cluster_pixel_count_b = int(rough_mask_b.sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global produced an empty final cluster after multiscale refinement.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

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
        return _MultiscaleClusterRun(
            coarsest_label_map=image_space_label_maps[0],
            level_label_maps=tuple(image_space_label_maps),
            level_names=tuple(level_names),
            level_resolutions=tuple(level_resolutions),
            rough_mask_a=rough_mask_a,
            rough_mask_b=rough_mask_b,
            refined_mask_a=selected_pair["candidate_a"]["mask"],
            refined_mask_b=selected_pair["candidate_b"]["mask"],
            refined_score_a=float(selected_pair["candidate_a"]["score"]),
            refined_score_b=float(selected_pair["candidate_b"]["score"]),
            refined_pair_selection_score=float(selected_pair["selection_score"]),
            refined_pair_overlap_iou=float(selected_pair["overlap_iou"]),
            cluster_pixel_count_a=cluster_pixel_count_a,
            cluster_pixel_count_b=cluster_pixel_count_b,
            prompt_mask_a=np.asarray(rough_mask_a, dtype=bool),
            prompt_mask_b=np.asarray(rough_mask_b, dtype=bool),
            prompt_candidates_a=_summarize_prompt_candidates(candidates_a),
            prompt_candidates_b=_summarize_prompt_candidates(candidates_b),
            selected_candidate_rank_a=int(selected_pair["candidate_a"]["rank"]),
            selected_candidate_rank_b=int(selected_pair["candidate_b"]["rank"]),
            selected_candidate_index_a=int(selected_pair["candidate_a"]["index"]),
            selected_candidate_index_b=int(selected_pair["candidate_b"]["index"]),
        )

    def _collect_mask_prompt_candidates(
        self,
        base_state: dict[str, Any],
        coarse_mask: np.ndarray,
        torch_module: Any,
        official_backend: Any,
    ) -> list[dict[str, Any]]:
        if int(np.asarray(coarse_mask, dtype=bool).sum()) == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global produced an empty coarse cluster mask before SAM refinement."
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
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global refinement returned no masks after applying the cluster prompt."
            )

        return _collect_prompt_aligned_candidates(
            masks=masks,
            scores=scores,
            coarse_mask=np.asarray(coarse_mask, dtype=bool),
            expected_shape=(int(base_state["original_height"]), int(base_state["original_width"])),
        )


class Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner(Sam3FeatureClusterCoarseToFineGlobalRunner):
    """Coarse-to-fine clustering with a pooled coarsest initialization for fewer tiny islands."""

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

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Run pooled coarsest init, then reuse the unchanged finer-level refinement path."""

        official_backend, base_state, image_size, feature_levels = self._prepare_feature_levels(image)
        baseline_native_label_map = _cluster_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        pooled_native_label_map, pooled_grid_resolution = _cluster_pooled_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )

        baseline_run = self._run_from_initial_label_map(
            initial_label_map=baseline_native_label_map,
            feature_levels=feature_levels,
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )
        pooled_run = self._run_from_initial_label_map(
            initial_label_map=pooled_native_label_map,
            feature_levels=feature_levels,
            image_size=image_size,
            base_state=base_state,
            official_backend=official_backend,
        )
        baseline_alignment = _align_mask_pair_to_reference(
            candidate_mask_a=baseline_run.refined_mask_a,
            candidate_mask_b=baseline_run.refined_mask_b,
            reference_mask_a=pooled_run.refined_mask_a,
            reference_mask_b=pooled_run.refined_mask_b,
        )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=pooled_run.coarsest_label_map,
            level_label_maps=pooled_run.level_label_maps,
            level_names=pooled_run.level_names,
            level_resolutions=pooled_run.level_resolutions,
            rough_mask_a=pooled_run.rough_mask_a,
            rough_mask_b=pooled_run.rough_mask_b,
            refined_mask_a=pooled_run.refined_mask_a,
            refined_mask_b=pooled_run.refined_mask_b,
            refined_score_a=pooled_run.refined_score_a,
            refined_score_b=pooled_run.refined_score_b,
            refined_pair_selection_score=pooled_run.refined_pair_selection_score,
            refined_pair_overlap_iou=pooled_run.refined_pair_overlap_iou,
            cluster_pixel_count_a=pooled_run.cluster_pixel_count_a,
            cluster_pixel_count_b=pooled_run.cluster_pixel_count_b,
            coarsest_native_label_map=pooled_native_label_map,
            coarsest_init_mode="pooled_avg_pool",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(pooled_native_label_map),
            comparison_baseline_coarsest_label_map=baseline_run.coarsest_label_map,
            comparison_baseline_coarsest_native_label_map=baseline_native_label_map,
            comparison_baseline_final_mask_a=baseline_alignment["mask_a"],
            comparison_baseline_final_mask_b=baseline_alignment["mask_b"],
            comparison_baseline_component_stats=_compute_label_map_component_stats(baseline_native_label_map),
            comparison_baseline_alignment_used=str(baseline_alignment["alignment_used"]),
            comparison_baseline_direct_mean_iou=float(baseline_alignment["direct_mean_iou"]),
            comparison_baseline_swapped_mean_iou=float(baseline_alignment["swapped_mean_iou"]),
            comparison_baseline_chosen_mean_iou=float(baseline_alignment["chosen_mean_iou"]),
            prompt_mask_a=pooled_run.prompt_mask_a,
            prompt_mask_b=pooled_run.prompt_mask_b,
            prompt_candidates_a=pooled_run.prompt_candidates_a,
            prompt_candidates_b=pooled_run.prompt_candidates_b,
            selected_candidate_rank_a=pooled_run.selected_candidate_rank_a,
            selected_candidate_rank_b=pooled_run.selected_candidate_rank_b,
            selected_candidate_index_a=pooled_run.selected_candidate_index_a,
            selected_candidate_index_b=pooled_run.selected_candidate_index_b,
            comparison_baseline_prompt_mask_a=baseline_run.prompt_mask_a,
            comparison_baseline_prompt_mask_b=baseline_run.prompt_mask_b,
            comparison_baseline_prompt_candidates_a=baseline_run.prompt_candidates_a,
            comparison_baseline_prompt_candidates_b=baseline_run.prompt_candidates_b,
            comparison_baseline_selected_candidate_rank_a=baseline_run.selected_candidate_rank_a,
            comparison_baseline_selected_candidate_rank_b=baseline_run.selected_candidate_rank_b,
            comparison_baseline_selected_candidate_index_a=baseline_run.selected_candidate_index_a,
            comparison_baseline_selected_candidate_index_b=baseline_run.selected_candidate_index_b,
            comparison_baseline_raw_final_mask_a=baseline_run.refined_mask_a,
            comparison_baseline_raw_final_mask_b=baseline_run.refined_mask_b,
        )


class Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner(
    Sam3FeatureClusterCoarseToFineGlobalRunner
):
    """Skip finer-level refinement but still refine the pooled coarsest masks with SAM."""

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

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Prompt-refine the pooled coarsest partition without any finer-level updates."""

        official_backend, base_state, image_size, feature_levels = self._prepare_feature_levels(image)
        baseline_native_label_map = _cluster_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        pooled_native_label_map, pooled_grid_resolution = _cluster_pooled_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )

        baseline_run = self._run_from_initial_label_map(
            initial_label_map=baseline_native_label_map,
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
        baseline_alignment = _align_mask_pair_to_reference(
            candidate_mask_a=baseline_run.refined_mask_a,
            candidate_mask_b=baseline_run.refined_mask_b,
            reference_mask_a=pooled_run.refined_mask_a,
            reference_mask_b=pooled_run.refined_mask_b,
        )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=pooled_run.coarsest_label_map,
            level_label_maps=pooled_run.level_label_maps,
            level_names=pooled_run.level_names,
            level_resolutions=pooled_run.level_resolutions,
            rough_mask_a=pooled_run.rough_mask_a,
            rough_mask_b=pooled_run.rough_mask_b,
            refined_mask_a=pooled_run.refined_mask_a,
            refined_mask_b=pooled_run.refined_mask_b,
            refined_score_a=pooled_run.refined_score_a,
            refined_score_b=pooled_run.refined_score_b,
            refined_pair_selection_score=pooled_run.refined_pair_selection_score,
            refined_pair_overlap_iou=pooled_run.refined_pair_overlap_iou,
            cluster_pixel_count_a=pooled_run.cluster_pixel_count_a,
            cluster_pixel_count_b=pooled_run.cluster_pixel_count_b,
            coarsest_native_label_map=pooled_native_label_map,
            coarsest_init_mode="pooled_avg_pool_direct_prompt",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(pooled_native_label_map),
            comparison_baseline_coarsest_label_map=baseline_run.coarsest_label_map,
            comparison_baseline_coarsest_native_label_map=baseline_native_label_map,
            comparison_baseline_final_mask_a=baseline_alignment["mask_a"],
            comparison_baseline_final_mask_b=baseline_alignment["mask_b"],
            comparison_baseline_component_stats=_compute_label_map_component_stats(baseline_native_label_map),
            comparison_baseline_alignment_used=str(baseline_alignment["alignment_used"]),
            comparison_baseline_direct_mean_iou=float(baseline_alignment["direct_mean_iou"]),
            comparison_baseline_swapped_mean_iou=float(baseline_alignment["swapped_mean_iou"]),
            comparison_baseline_chosen_mean_iou=float(baseline_alignment["chosen_mean_iou"]),
            prompt_mask_a=pooled_run.prompt_mask_a,
            prompt_mask_b=pooled_run.prompt_mask_b,
            prompt_candidates_a=pooled_run.prompt_candidates_a,
            prompt_candidates_b=pooled_run.prompt_candidates_b,
            selected_candidate_rank_a=pooled_run.selected_candidate_rank_a,
            selected_candidate_rank_b=pooled_run.selected_candidate_rank_b,
            selected_candidate_index_a=pooled_run.selected_candidate_index_a,
            selected_candidate_index_b=pooled_run.selected_candidate_index_b,
            comparison_baseline_prompt_mask_a=baseline_run.prompt_mask_a,
            comparison_baseline_prompt_mask_b=baseline_run.prompt_mask_b,
            comparison_baseline_prompt_candidates_a=baseline_run.prompt_candidates_a,
            comparison_baseline_prompt_candidates_b=baseline_run.prompt_candidates_b,
            comparison_baseline_selected_candidate_rank_a=baseline_run.selected_candidate_rank_a,
            comparison_baseline_selected_candidate_rank_b=baseline_run.selected_candidate_rank_b,
            comparison_baseline_selected_candidate_index_a=baseline_run.selected_candidate_index_a,
            comparison_baseline_selected_candidate_index_b=baseline_run.selected_candidate_index_b,
            comparison_baseline_raw_final_mask_a=baseline_run.refined_mask_a,
            comparison_baseline_raw_final_mask_b=baseline_run.refined_mask_b,
            multiscale_refinement_applied=False,
            sam_refinement_applied=True,
        )


class Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner(
    Sam3FeatureClusterCoarseToFineGlobalRunner
):
    """Use the pooled coarsest partition directly as the final two-mask prediction set."""

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

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Cluster the pooled coarsest level and evaluate that partition directly without SAM."""

        official_backend, _, image_size, feature_levels = self._prepare_feature_levels(image)
        pooled_native_label_map, pooled_grid_resolution = _cluster_pooled_coarsest_level_2way(
            feature_map=feature_levels[0].feature_map,
            torch_module=official_backend.torch,
            settings=self.settings,
        )
        image_space_label_map = _upsample_label_map(
            pooled_native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        rough_mask_a, rough_mask_b = _label_map_to_cluster_masks(image_space_label_map)
        cluster_pixel_count_a = int(rough_mask_a.sum())
        cluster_pixel_count_b = int(rough_mask_b.sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only produced an empty pooled coarse mask.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=image_space_label_map,
            level_label_maps=(image_space_label_map,),
            level_names=(feature_levels[0].name,),
            level_resolutions=(feature_levels[0].resolution,),
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
            coarsest_native_label_map=pooled_native_label_map,
            coarsest_init_mode="pooled_avg_pool_coarse_only",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(pooled_native_label_map),
            multiscale_refinement_applied=False,
            sam_refinement_applied=False,
        )


class Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner(
    Sam3FeatureClusterCoarseToFineGlobalRunner
):
    """Diagnose and de-bias coarsest-level coordinate leakage before pooled coarse-only clustering.

    This research variant keeps the current pooled coarse-only output contract, but adds
    three coarsest-only branches:

    - baseline pooled init on the original coarsest features
    - coordinate-projection removal before pooled init
    - flip-averaged coarsest features before pooled init

    The active prediction returned by this runner is the projection-removed branch.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_POOLED_INIT_DEBIASED_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Run the pooled coarse-only experiment with positionality diagnostics and de-biasing."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)

        raw_native_label_map = _cluster_label_map_from_numpy_feature_map(
            feature_map=raw_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        raw_diagnostic_label_map = _upsample_label_map(
            raw_native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        label_positionality_diagnostics = _compute_label_positionality_diagnostics(raw_native_label_map)
        feature_positionality_diagnostics = _compute_feature_positionality_diagnostics(
            raw_feature_map,
            top_channel_count=int(self.settings["positionality_top_channel_count"]),
        )
        projection_removed_feature_map, projection_removed_feature_positionality = (
            _remove_coordinate_projection_from_feature_map(
                raw_feature_map,
                top_channel_count=int(self.settings["positionality_top_channel_count"]),
            )
        )

        baseline_native_label_map, pooled_grid_resolution = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=raw_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        projection_native_label_map, _ = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=projection_removed_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        flip_average_feature_map, flip_test_diagnostics = self._build_flip_average_feature_map(
            image=image,
            official_backend=official_backend,
            reference_shape=raw_feature_map.shape,
            original_raw_label_map=raw_native_label_map,
            original_label_positionality=label_positionality_diagnostics,
        )
        flip_avg_native_label_map, _ = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=flip_average_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )

        baseline_branch = _label_map_branch_from_native_map(
            native_label_map=baseline_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        projection_branch = _label_map_branch_from_native_map(
            native_label_map=projection_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        flip_avg_branch = _label_map_branch_from_native_map(
            native_label_map=flip_avg_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )

        projection_component_stats = _compute_label_map_component_stats(projection_native_label_map)
        null_test_diagnostics = self._run_null_image_tests(
            image=image,
            official_backend=official_backend,
            torch_module=official_backend.torch,
            reference_image_size=image_size,
        )
        baseline_branch_diagnostics = _build_branch_label_diagnostics("baseline_pooled", baseline_native_label_map)
        projection_branch_diagnostics = _build_branch_label_diagnostics(
            "projection_removed",
            projection_native_label_map,
        )
        flip_avg_branch_diagnostics = _build_branch_label_diagnostics("flip_averaged", flip_avg_native_label_map)
        recommended_mitigation, recommendation_reason = _recommend_positionality_mitigation(
            baseline_branch_diagnostics,
            projection_branch_diagnostics,
            flip_avg_branch_diagnostics,
        )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=projection_branch["image_space_label_map"],
            level_label_maps=(projection_branch["image_space_label_map"],),
            level_names=(coarsest_level.name,),
            level_resolutions=(coarsest_level.resolution,),
            rough_mask_a=projection_branch["mask_a"],
            rough_mask_b=projection_branch["mask_b"],
            refined_mask_a=projection_branch["mask_a"],
            refined_mask_b=projection_branch["mask_b"],
            refined_score_a=0.0,
            refined_score_b=0.0,
            refined_pair_selection_score=0.0,
            refined_pair_overlap_iou=float(_binary_iou(projection_branch["mask_a"], projection_branch["mask_b"])),
            cluster_pixel_count_a=int(projection_branch["mask_a"].sum()),
            cluster_pixel_count_b=int(projection_branch["mask_b"].sum()),
            coarsest_native_label_map=projection_native_label_map,
            coarsest_init_mode="pooled_avg_pool_projection_removed_coarse_only",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=projection_component_stats,
            multiscale_refinement_applied=False,
            sam_refinement_applied=False,
            raw_diagnostic_native_label_map=raw_native_label_map,
            raw_diagnostic_label_map=raw_diagnostic_label_map,
            baseline_branch_native_label_map=baseline_native_label_map,
            baseline_branch_label_map=baseline_branch["image_space_label_map"],
            baseline_branch_mask_a=baseline_branch["mask_a"],
            baseline_branch_mask_b=baseline_branch["mask_b"],
            flip_avg_native_label_map=flip_avg_native_label_map,
            flip_avg_label_map=flip_avg_branch["image_space_label_map"],
            flip_avg_mask_a=flip_avg_branch["mask_a"],
            flip_avg_mask_b=flip_avg_branch["mask_b"],
            projection_removed_native_label_map=projection_native_label_map,
            projection_removed_label_map=projection_branch["image_space_label_map"],
            null_constant_label_map=null_test_diagnostics["constant_gray"]["image_space_label_map"],
            null_noise_label_map=null_test_diagnostics["weak_noise"]["image_space_label_map"],
            null_blur_label_map=null_test_diagnostics["strong_blur"]["image_space_label_map"],
            label_positionality_diagnostics=label_positionality_diagnostics,
            feature_positionality_diagnostics=feature_positionality_diagnostics,
            projection_removed_feature_positionality_diagnostics=projection_removed_feature_positionality,
            baseline_branch_diagnostics=baseline_branch_diagnostics,
            projection_removed_branch_diagnostics=projection_branch_diagnostics,
            flip_avg_branch_diagnostics=flip_avg_branch_diagnostics,
            flip_test_diagnostics=flip_test_diagnostics,
            null_test_diagnostics=null_test_diagnostics,
            recommended_mitigation=recommended_mitigation,
            recommendation_reason=recommendation_reason,
        )

    def _build_flip_average_feature_map(
        self,
        image: Image.Image,
        official_backend: Any,
        reference_shape: tuple[int, int, int],
        original_raw_label_map: np.ndarray,
        original_label_positionality: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        transform_names = ("identity", "hflip", "vflip", "hvflip")
        feature_maps: list[np.ndarray] = []
        flip_diagnostics: dict[str, Any] = {
            "coordinate_basis": "1,x,y,x^2,xy,y^2",
            "random_seed": None,
            "transforms": {},
        }

        for transform_name in transform_names:
            transformed_image = _apply_image_transform(image, transform_name)
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(transformed_image, official_backend)
            transformed_feature_map = _feature_map_to_numpy(feature_levels[0].feature_map)
            if transformed_feature_map.shape != reference_shape:
                raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                    "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only got inconsistent "
                    f"coarsest feature shapes across flip tests: expected {reference_shape}, "
                    f"got {transformed_feature_map.shape} for {transform_name}."
                )
            restored_feature_map = _undo_feature_transform(transformed_feature_map, transform_name)
            feature_maps.append(restored_feature_map)

            if transform_name == "identity":
                continue
            restored_label_map = _cluster_label_map_from_numpy_feature_map(
                feature_map=restored_feature_map,
                torch_module=official_backend.torch,
                device=official_backend.device,
                settings=self.settings,
            )
            restored_label_positionality = _compute_label_positionality_diagnostics(restored_label_map)
            partition_alignment = _align_mask_pair_to_reference(
                candidate_mask_a=restored_label_map == 0,
                candidate_mask_b=restored_label_map == 1,
                reference_mask_a=original_raw_label_map == 0,
                reference_mask_b=original_raw_label_map == 1,
            )
            flip_diagnostics["transforms"][transform_name] = {
                "cluster_mean_iou_to_original": float(partition_alignment["chosen_mean_iou"]),
                "label_positionality_index": restored_label_positionality["label_positionality_index"],
                "label_positionality_best_model": restored_label_positionality["best_model_name"],
                "label_positionality_best_iou": restored_label_positionality["best_iou"],
                "label_positionality_best_balanced_accuracy": restored_label_positionality[
                    "best_balanced_accuracy"
                ],
            }

        flip_average_feature_map = _normalize_numpy_feature_map(np.mean(np.stack(feature_maps, axis=0), axis=0))
        transform_diagnostics = flip_diagnostics["transforms"]
        if transform_diagnostics:
            mean_flip_consistency = float(
                np.mean([diagnostics["cluster_mean_iou_to_original"] for diagnostics in transform_diagnostics.values()])
            )
            max_flip_positionality = float(
                max(diagnostics["label_positionality_index"] for diagnostics in transform_diagnostics.values())
            )
            flip_diagnostics["mean_unflipped_cluster_iou_to_original"] = mean_flip_consistency
            flip_diagnostics["max_unflipped_label_positionality_index"] = max_flip_positionality
            flip_diagnostics["bias_interpretation"] = _infer_flip_bias_interpretation(
                mean_flip_consistency=mean_flip_consistency,
                max_flip_positionality=max_flip_positionality,
                original_positionality=float(original_label_positionality["label_positionality_index"]),
                settings=self.settings,
            )
        else:
            flip_diagnostics["mean_unflipped_cluster_iou_to_original"] = 1.0
            flip_diagnostics["max_unflipped_label_positionality_index"] = float(
                original_label_positionality["label_positionality_index"]
            )
            flip_diagnostics["bias_interpretation"] = "content_consistent"
        return flip_average_feature_map, flip_diagnostics

    def _run_null_image_tests(
        self,
        image: Image.Image,
        official_backend: Any,
        torch_module: Any,
        reference_image_size: tuple[int, int],
    ) -> dict[str, Any]:
        null_images = {
            "constant_gray": _build_constant_gray_image(image),
            "weak_noise": _build_weak_noise_image(
                image,
                seed=int(self.settings["null_noise_seed"]),
                std=float(self.settings["null_noise_std"]),
            ),
            "strong_blur": _build_strong_blur_image(
                image,
                radius=float(self.settings["null_blur_radius"]),
            ),
        }
        diagnostics: dict[str, Any] = {}
        for null_name, null_image in null_images.items():
            _, _, _, feature_levels = self._prepare_feature_levels_with_backend(null_image, official_backend)
            null_feature_map = _feature_map_to_numpy(feature_levels[0].feature_map)
            null_raw_label_map = _cluster_label_map_from_numpy_feature_map(
                feature_map=null_feature_map,
                torch_module=torch_module,
                device=official_backend.device,
                settings=self.settings,
            )
            label_diagnostics = _compute_label_positionality_diagnostics(null_raw_label_map)
            image_space_label_map = _upsample_label_map(
                null_raw_label_map,
                target_size=reference_image_size,
                torch_module=torch_module,
            )
            diagnostics[null_name] = {
                "image_space_label_map": image_space_label_map,
                "label_positionality_index": label_diagnostics["label_positionality_index"],
                "best_model_name": label_diagnostics["best_model_name"],
                "best_iou": label_diagnostics["best_iou"],
                "best_balanced_accuracy": label_diagnostics["best_balanced_accuracy"],
                "axis_threshold_index": label_diagnostics["axis_threshold_index"],
                "axis_split_flag": bool(
                    label_diagnostics["axis_threshold_index"] >= float(self.settings["null_axis_threshold_flag"])
                ),
            }
        return diagnostics


class Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner(
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner
):
    """Use flip-averaged coarsest features before pooled clustering, then evaluate directly."""

    def extract_pooled_feature_map_for_visualization(self, image: Image.Image):
        """Expose the pooled flip-averaged coarsest feature map used for clustering."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, _, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        flip_average_feature_map, _ = self._build_flip_average_feature_map(
            image=image,
            official_backend=official_backend,
            reference_shape=raw_feature_map.shape,
            original_raw_label_map=_cluster_label_map_from_numpy_feature_map(
                feature_map=raw_feature_map,
                torch_module=official_backend.torch,
                device=official_backend.device,
                settings=self.settings,
            ),
            original_label_positionality=_compute_label_positionality_diagnostics(
                _cluster_label_map_from_numpy_feature_map(
                    feature_map=raw_feature_map,
                    torch_module=official_backend.torch,
                    device=official_backend.device,
                    settings=self.settings,
                )
            ),
        )
        kernel_size = int(self.settings["coarsest_init_pool_kernel_size"])
        stride = int(self.settings["coarsest_init_pool_stride"])
        if kernel_size <= 0 or stride <= 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only requires positive pooling kernel/stride values.",
                diagnostics={
                    "coarsest_init_pool_kernel_size": kernel_size,
                    "coarsest_init_pool_stride": stride,
                },
            )
        pooled_flip_average_feature_map = official_backend.torch.nn.functional.avg_pool2d(
            official_backend.torch.as_tensor(flip_average_feature_map, dtype=official_backend.torch.float32)[None],
            kernel_size=kernel_size,
            stride=stride,
        )[0]
        if pooled_flip_average_feature_map.shape[-2] < 1 or pooled_flip_average_feature_map.shape[-1] < 1:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only collapsed the pooled coarsest feature map during visualization pooling.",
                diagnostics={
                    "coarsest_resolution": (int(raw_feature_map.shape[-2]), int(raw_feature_map.shape[-1])),
                    "coarsest_init_pool_kernel_size": kernel_size,
                    "coarsest_init_pool_stride": stride,
                },
            )
        return pooled_flip_average_feature_map

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Run the flip-averaged pooled coarse-only experiment with the standard output contract."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        raw_native_label_map = _cluster_label_map_from_numpy_feature_map(
            feature_map=raw_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        raw_label_positionality = _compute_label_positionality_diagnostics(raw_native_label_map)
        flip_average_feature_map, _ = self._build_flip_average_feature_map(
            image=image,
            official_backend=official_backend,
            reference_shape=raw_feature_map.shape,
            original_raw_label_map=raw_native_label_map,
            original_label_positionality=raw_label_positionality,
        )
        flip_avg_native_label_map, pooled_grid_resolution = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=flip_average_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        flip_avg_branch = _label_map_branch_from_native_map(
            native_label_map=flip_avg_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        cluster_pixel_count_a = int(flip_avg_branch["mask_a"].sum())
        cluster_pixel_count_b = int(flip_avg_branch["mask_b"].sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only produced an empty pooled "
                "coarse mask after flip-averaged feature extraction.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=flip_avg_branch["image_space_label_map"],
            level_label_maps=(flip_avg_branch["image_space_label_map"],),
            level_names=(coarsest_level.name,),
            level_resolutions=(coarsest_level.resolution,),
            rough_mask_a=flip_avg_branch["mask_a"],
            rough_mask_b=flip_avg_branch["mask_b"],
            refined_mask_a=flip_avg_branch["mask_a"],
            refined_mask_b=flip_avg_branch["mask_b"],
            refined_score_a=0.0,
            refined_score_b=0.0,
            refined_pair_selection_score=0.0,
            refined_pair_overlap_iou=float(_binary_iou(flip_avg_branch["mask_a"], flip_avg_branch["mask_b"])),
            cluster_pixel_count_a=cluster_pixel_count_a,
            cluster_pixel_count_b=cluster_pixel_count_b,
            coarsest_native_label_map=flip_avg_native_label_map,
            coarsest_init_mode="pooled_avg_pool_flip_avg_coarse_only",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(flip_avg_native_label_map),
            multiscale_refinement_applied=False,
            sam_refinement_applied=False,
        )


class Sam3FeatureClusterCoarseToFineGlobalFlipAvgPlusEdgeDebiasRunner(
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner
):
    """Use flip-averaged features, then project out symmetric edge bias before pooled clustering."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        hf_token: str | None = None,
        official_checkpoint_path: str | None = None,
        settings: dict[str, float | int | str] | None = None,
    ) -> None:
        resolved_settings = dict(FEATURE_CLUSTER_COARSE_TO_FINE_GLOBAL_FLIP_AVG_PLUS_EDGE_DEBIAS_SETTINGS)
        if settings is not None:
            resolved_settings.update(settings)
        super().__init__(
            model_id=model_id,
            device=device,
            hf_token=hf_token,
            official_checkpoint_path=official_checkpoint_path,
            settings=resolved_settings,
        )

    def generate_feature_clusters(self, image: Image.Image) -> FeatureClusterCoarseToFineGlobalRefinement:
        """Compare baseline, flip-only, edge-only, and flip-plus-edge coarse partitions."""

        official_backend = self._sam3_runner._ensure_official_backend()
        _, _, image_size, feature_levels = self._prepare_feature_levels_with_backend(image, official_backend)
        coarsest_level = feature_levels[0]
        raw_feature_map = _feature_map_to_numpy(coarsest_level.feature_map)
        top_channel_count = int(self.settings["positionality_top_channel_count"])

        raw_native_label_map = _cluster_label_map_from_numpy_feature_map(
            feature_map=raw_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        raw_label_map = _upsample_label_map(
            raw_native_label_map,
            target_size=image_size,
            torch_module=official_backend.torch,
        )
        raw_label_positionality = _compute_label_positionality_diagnostics(raw_native_label_map)
        raw_edge_positionality = _compute_edge_label_positionality_diagnostics(raw_native_label_map)
        raw_feature_edge_positionality = _compute_edge_feature_positionality_diagnostics(
            raw_feature_map,
            top_channel_count=top_channel_count,
        )

        baseline_native_label_map, pooled_grid_resolution = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=raw_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        baseline_branch = _label_map_branch_from_native_map(
            native_label_map=baseline_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        baseline_branch_edge_diagnostics = _build_branch_edge_diagnostics(
            "baseline_pooled",
            baseline_native_label_map,
        )

        edge_debiased_feature_map, edge_debiased_feature_positionality = (
            _remove_edge_coordinate_projection_from_feature_map(
                raw_feature_map,
                top_channel_count=top_channel_count,
            )
        )
        edge_debiased_native_label_map, _ = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=edge_debiased_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        edge_debiased_branch = _label_map_branch_from_native_map(
            native_label_map=edge_debiased_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        edge_debiased_branch_edge_diagnostics = _build_branch_edge_diagnostics(
            "edge_debiased",
            edge_debiased_native_label_map,
        )

        flip_average_feature_map, flip_test_diagnostics = self._build_flip_average_feature_map(
            image=image,
            official_backend=official_backend,
            reference_shape=raw_feature_map.shape,
            original_raw_label_map=raw_native_label_map,
            original_label_positionality=raw_label_positionality,
        )
        flip_avg_feature_edge_positionality = _compute_edge_feature_positionality_diagnostics(
            flip_average_feature_map,
            top_channel_count=top_channel_count,
        )
        flip_avg_native_label_map, _ = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=flip_average_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        flip_avg_branch = _label_map_branch_from_native_map(
            native_label_map=flip_avg_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        flip_avg_branch_edge_diagnostics = _build_branch_edge_diagnostics(
            "flip_averaged",
            flip_avg_native_label_map,
        )

        flip_avg_plus_edge_debiased_feature_map, flip_avg_plus_edge_debiased_feature_positionality = (
            _remove_edge_coordinate_projection_from_feature_map(
                flip_average_feature_map,
                top_channel_count=top_channel_count,
            )
        )
        flip_avg_plus_edge_debiased_native_label_map, _ = _cluster_pooled_label_map_from_numpy_feature_map(
            feature_map=flip_avg_plus_edge_debiased_feature_map,
            torch_module=official_backend.torch,
            device=official_backend.device,
            settings=self.settings,
        )
        flip_avg_plus_edge_debiased_branch = _label_map_branch_from_native_map(
            native_label_map=flip_avg_plus_edge_debiased_native_label_map,
            image_size=image_size,
            torch_module=official_backend.torch,
        )
        flip_avg_plus_edge_debiased_branch_edge_diagnostics = _build_branch_edge_diagnostics(
            "flip_avg_plus_edge_debiased",
            flip_avg_plus_edge_debiased_native_label_map,
        )

        cluster_pixel_count_a = int(flip_avg_plus_edge_debiased_branch["mask_a"].sum())
        cluster_pixel_count_b = int(flip_avg_plus_edge_debiased_branch["mask_b"].sum())
        if cluster_pixel_count_a == 0 or cluster_pixel_count_b == 0:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "flip_avg_plus_edge_debias produced an empty pooled coarse mask after the symmetric edge-debias step.",
                diagnostics={
                    "cluster_pixel_count_a": cluster_pixel_count_a,
                    "cluster_pixel_count_b": cluster_pixel_count_b,
                },
            )

        return FeatureClusterCoarseToFineGlobalRefinement(
            coarsest_label_map=flip_avg_plus_edge_debiased_branch["image_space_label_map"],
            level_label_maps=(flip_avg_plus_edge_debiased_branch["image_space_label_map"],),
            level_names=(coarsest_level.name,),
            level_resolutions=(coarsest_level.resolution,),
            rough_mask_a=flip_avg_plus_edge_debiased_branch["mask_a"],
            rough_mask_b=flip_avg_plus_edge_debiased_branch["mask_b"],
            refined_mask_a=flip_avg_plus_edge_debiased_branch["mask_a"],
            refined_mask_b=flip_avg_plus_edge_debiased_branch["mask_b"],
            refined_score_a=0.0,
            refined_score_b=0.0,
            refined_pair_selection_score=0.0,
            refined_pair_overlap_iou=float(
                _binary_iou(
                    flip_avg_plus_edge_debiased_branch["mask_a"],
                    flip_avg_plus_edge_debiased_branch["mask_b"],
                )
            ),
            cluster_pixel_count_a=cluster_pixel_count_a,
            cluster_pixel_count_b=cluster_pixel_count_b,
            coarsest_native_label_map=flip_avg_plus_edge_debiased_native_label_map,
            coarsest_init_mode="pooled_avg_pool_flip_avg_plus_edge_debias_coarse_only",
            coarsest_pool_kernel_size=int(self.settings["coarsest_init_pool_kernel_size"]),
            coarsest_pool_stride=int(self.settings["coarsest_init_pool_stride"]),
            pooled_grid_resolution=pooled_grid_resolution,
            coarsest_component_stats=_compute_label_map_component_stats(flip_avg_plus_edge_debiased_native_label_map),
            multiscale_refinement_applied=False,
            sam_refinement_applied=False,
            raw_diagnostic_native_label_map=raw_native_label_map,
            raw_diagnostic_label_map=raw_label_map,
            baseline_branch_native_label_map=baseline_native_label_map,
            baseline_branch_label_map=baseline_branch["image_space_label_map"],
            baseline_branch_mask_a=baseline_branch["mask_a"],
            baseline_branch_mask_b=baseline_branch["mask_b"],
            flip_avg_native_label_map=flip_avg_native_label_map,
            flip_avg_label_map=flip_avg_branch["image_space_label_map"],
            flip_avg_mask_a=flip_avg_branch["mask_a"],
            flip_avg_mask_b=flip_avg_branch["mask_b"],
            edge_debiased_native_label_map=edge_debiased_native_label_map,
            edge_debiased_label_map=edge_debiased_branch["image_space_label_map"],
            edge_debiased_mask_a=edge_debiased_branch["mask_a"],
            edge_debiased_mask_b=edge_debiased_branch["mask_b"],
            flip_avg_plus_edge_debiased_native_label_map=flip_avg_plus_edge_debiased_native_label_map,
            flip_avg_plus_edge_debiased_label_map=flip_avg_plus_edge_debiased_branch["image_space_label_map"],
            flip_avg_plus_edge_debiased_mask_a=flip_avg_plus_edge_debiased_branch["mask_a"],
            flip_avg_plus_edge_debiased_mask_b=flip_avg_plus_edge_debiased_branch["mask_b"],
            raw_edge_positionality_diagnostics=raw_edge_positionality,
            raw_feature_edge_positionality_diagnostics=raw_feature_edge_positionality,
            baseline_branch_edge_diagnostics=baseline_branch_edge_diagnostics,
            flip_avg_branch_edge_diagnostics=flip_avg_branch_edge_diagnostics,
            edge_debiased_feature_edge_positionality_diagnostics=edge_debiased_feature_positionality,
            edge_debiased_branch_edge_diagnostics=edge_debiased_branch_edge_diagnostics,
            flip_avg_feature_edge_positionality_diagnostics=flip_avg_feature_edge_positionality,
            flip_avg_plus_edge_debiased_feature_edge_positionality_diagnostics=(
                flip_avg_plus_edge_debiased_feature_positionality
            ),
            flip_avg_plus_edge_debiased_branch_edge_diagnostics=flip_avg_plus_edge_debiased_branch_edge_diagnostics,
            flip_test_diagnostics=flip_test_diagnostics,
            recommended_mitigation="flip_avg_plus_edge_debias",
            recommendation_reason=(
                "Flip averaging removes directional bias first; the symmetric edge-basis projection then targets the "
                "remaining border-related leakage before pooled clustering."
            ),
        )


def _feature_map_to_numpy(feature_map: Any) -> np.ndarray:
    return np.asarray(feature_map.detach().cpu().numpy(), dtype=np.float32)


def _normalize_numpy_feature_map(feature_map: np.ndarray) -> np.ndarray:
    normalized = np.asarray(feature_map, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=0, keepdims=True)
    safe_denominator = np.where(norms > 1e-6, norms, 1.0).astype(np.float32)
    normalized = normalized / safe_denominator
    normalized = np.where(norms > 1e-6, normalized, 0.0)
    return normalized.astype(np.float32)


def _cluster_label_map_from_numpy_feature_map(
    feature_map: np.ndarray,
    torch_module: Any,
    device: Any,
    settings: dict[str, float | int | str],
) -> np.ndarray:
    feature_tensor = torch_module.as_tensor(
        _normalize_numpy_feature_map(feature_map),
        dtype=torch_module.float32,
        device=device,
    )
    return _cluster_coarsest_level_2way(feature_tensor, torch_module=torch_module, settings=settings)


def _cluster_pooled_label_map_from_numpy_feature_map(
    feature_map: np.ndarray,
    torch_module: Any,
    device: Any,
    settings: dict[str, float | int | str],
) -> tuple[np.ndarray, tuple[int, int]]:
    feature_tensor = torch_module.as_tensor(
        _normalize_numpy_feature_map(feature_map),
        dtype=torch_module.float32,
        device=device,
    )
    return _cluster_pooled_coarsest_level_2way(feature_tensor, torch_module=torch_module, settings=settings)


def _label_map_branch_from_native_map(
    native_label_map: np.ndarray,
    image_size: tuple[int, int],
    torch_module: Any,
) -> dict[str, np.ndarray]:
    image_space_label_map = _upsample_label_map(native_label_map, target_size=image_size, torch_module=torch_module)
    mask_a, mask_b = _label_map_to_cluster_masks(image_space_label_map)
    return {
        "image_space_label_map": image_space_label_map,
        "mask_a": mask_a,
        "mask_b": mask_b,
    }


def _build_coordinate_basis(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_values = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    basis = np.stack(
        [
            np.ones_like(grid_x),
            grid_x,
            grid_y,
            grid_x * grid_x,
            grid_x * grid_y,
            grid_y * grid_y,
        ],
        axis=-1,
    ).reshape(-1, 6)
    return basis.astype(np.float32), grid_x.astype(np.float32), grid_y.astype(np.float32)


def _build_edge_symmetric_basis(height: int, width: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _, grid_x, grid_y = _build_coordinate_basis(height, width)
    x_squared = np.square(grid_x, dtype=np.float32)
    y_squared = np.square(grid_y, dtype=np.float32)
    radial_squared = x_squared + y_squared
    anisotropy = x_squared - y_squared
    d_edge = np.minimum.reduce(
        [
            grid_x + 1.0,
            1.0 - grid_x,
            grid_y + 1.0,
            1.0 - grid_y,
        ]
    ).astype(np.float32)
    d_edge_squared = np.square(d_edge, dtype=np.float32)
    scalar_features = {
        "x_squared": x_squared.reshape(-1).astype(np.float32),
        "y_squared": y_squared.reshape(-1).astype(np.float32),
        "radial_squared": radial_squared.reshape(-1).astype(np.float32),
        "anisotropy": anisotropy.reshape(-1).astype(np.float32),
        "d_edge": d_edge.reshape(-1).astype(np.float32),
        "d_edge_squared": d_edge_squared.reshape(-1).astype(np.float32),
    }
    basis = np.stack(
        [
            np.ones_like(scalar_features["x_squared"]),
            scalar_features["x_squared"],
            scalar_features["y_squared"],
            scalar_features["radial_squared"],
            scalar_features["anisotropy"],
            scalar_features["d_edge"],
            scalar_features["d_edge_squared"],
        ],
        axis=-1,
    ).astype(np.float32)
    return basis, scalar_features


def _compute_feature_positionality_diagnostics(
    feature_map: np.ndarray,
    *,
    top_channel_count: int,
) -> dict[str, Any]:
    channels, height, width = feature_map.shape
    flat_features = np.asarray(feature_map, dtype=np.float32).reshape(channels, height * width).T
    basis, _, _ = _build_coordinate_basis(height, width)
    basis_pinv = np.linalg.pinv(basis)
    coefficients = basis_pinv @ flat_features
    positional_features = basis @ coefficients
    residual = flat_features - positional_features
    total_var = np.sum((flat_features - flat_features.mean(axis=0, keepdims=True)) ** 2, axis=0)
    residual_var = np.sum(residual**2, axis=0)
    safe_total_var = np.clip(total_var, a_min=1e-12, a_max=None)
    r2 = 1.0 - (residual_var / safe_total_var)
    r2 = np.where(total_var > 1e-12, r2, 0.0)
    ranked_channels = np.argsort(r2)[::-1]
    top_channels = [
        {"channel_index": int(channel_index), "r2": float(r2[channel_index])}
        for channel_index in ranked_channels[: max(1, top_channel_count)]
    ]
    return {
        "feature_positionality_index": float(np.mean(r2)),
        "mean_r2": float(np.mean(r2)),
        "max_r2": float(np.max(r2)),
        "top_channels": top_channels,
        "spatial_coordinates_appended": False,
        "coordinate_basis": "1,x,y,x^2,xy,y^2",
        "channel_count": int(channels),
        "spatial_resolution": f"{height}x{width}",
    }


def _remove_coordinate_projection_from_feature_map(
    feature_map: np.ndarray,
    *,
    top_channel_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    channels, height, width = feature_map.shape
    flat_features = np.asarray(feature_map, dtype=np.float32).reshape(channels, height * width).T
    basis, _, _ = _build_coordinate_basis(height, width)
    basis_pinv = np.linalg.pinv(basis)
    positional_features = basis @ (basis_pinv @ flat_features)
    debiased_features = flat_features - positional_features
    debiased_feature_map = _normalize_numpy_feature_map(debiased_features.T.reshape(channels, height, width))
    diagnostics = _compute_feature_positionality_diagnostics(
        debiased_feature_map,
        top_channel_count=top_channel_count,
    )
    diagnostics["mitigation"] = "projection_removed"
    return debiased_feature_map, diagnostics


def _compute_edge_feature_positionality_diagnostics(
    feature_map: np.ndarray,
    *,
    top_channel_count: int,
) -> dict[str, Any]:
    channels, height, width = feature_map.shape
    flat_features = np.asarray(feature_map, dtype=np.float32).reshape(channels, height * width).T
    basis, _ = _build_edge_symmetric_basis(height, width)
    basis_pinv = np.linalg.pinv(basis)
    coefficients = basis_pinv @ flat_features
    positional_features = basis @ coefficients
    residual = flat_features - positional_features
    total_var = np.sum((flat_features - flat_features.mean(axis=0, keepdims=True)) ** 2, axis=0)
    residual_var = np.sum(residual**2, axis=0)
    safe_total_var = np.clip(total_var, a_min=1e-12, a_max=None)
    r2 = 1.0 - (residual_var / safe_total_var)
    r2 = np.where(total_var > 1e-12, r2, 0.0)
    ranked_channels = np.argsort(r2)[::-1]
    top_channels = [
        {"channel_index": int(channel_index), "r2": float(r2[channel_index])}
        for channel_index in ranked_channels[: max(1, top_channel_count)]
    ]
    return {
        "feature_positionality_index": float(np.mean(r2)),
        "mean_r2": float(np.mean(r2)),
        "max_r2": float(np.max(r2)),
        "top_channels": top_channels,
        "spatial_coordinates_appended": False,
        "coordinate_basis": "1,x^2,y^2,x^2+y^2,x^2-y^2,d_edge,d_edge^2",
        "channel_count": int(channels),
        "spatial_resolution": f"{height}x{width}",
    }


def _remove_edge_coordinate_projection_from_feature_map(
    feature_map: np.ndarray,
    *,
    top_channel_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    channels, height, width = feature_map.shape
    flat_features = np.asarray(feature_map, dtype=np.float32).reshape(channels, height * width).T
    basis, _ = _build_edge_symmetric_basis(height, width)
    basis_pinv = np.linalg.pinv(basis)
    positional_features = basis @ (basis_pinv @ flat_features)
    debiased_features = flat_features - positional_features
    debiased_feature_map = _normalize_numpy_feature_map(debiased_features.T.reshape(channels, height, width))
    diagnostics = _compute_edge_feature_positionality_diagnostics(
        debiased_feature_map,
        top_channel_count=top_channel_count,
    )
    diagnostics["mitigation"] = "edge_projection_removed"
    return debiased_feature_map, diagnostics


def _compute_label_positionality_diagnostics(label_map: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"Positionality diagnostics expected binary labels (0, 1), got {unique_values}."
        )

    height, width = labels.shape
    basis, grid_x, grid_y = _build_coordinate_basis(height, width)
    flat_labels = labels.reshape(-1).astype(bool)

    x_thresholds = _coordinate_thresholds(grid_x.reshape(-1))
    y_thresholds = _coordinate_thresholds(grid_y.reshape(-1))
    model_diagnostics = {
        "vertical_threshold": _fit_threshold_model(grid_x.reshape(-1), x_thresholds, flat_labels),
        "horizontal_threshold": _fit_threshold_model(grid_y.reshape(-1), y_thresholds, flat_labels),
        "affine_half_plane": _fit_basis_separator(basis[:, :3], flat_labels),
        "quadratic_basis": _fit_basis_separator(basis, flat_labels),
    }
    best_model_name, best_model_metrics = max(
        model_diagnostics.items(),
        key=lambda item: (item[1]["best_balanced_accuracy"], item[1]["best_iou"]),
    )
    axis_threshold_index = max(
        model_diagnostics["vertical_threshold"]["best_balanced_accuracy"],
        model_diagnostics["horizontal_threshold"]["best_balanced_accuracy"],
    )
    return {
        "label_positionality_index": float(best_model_metrics["best_balanced_accuracy"]),
        "best_model_name": str(best_model_name),
        "best_iou": float(best_model_metrics["best_iou"]),
        "best_balanced_accuracy": float(best_model_metrics["best_balanced_accuracy"]),
        "axis_threshold_index": float(axis_threshold_index),
        "models": model_diagnostics,
    }


def _compute_edge_label_positionality_diagnostics(label_map: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"Edge positionality diagnostics expected binary labels (0, 1), got {unique_values}."
        )

    height, width = labels.shape
    basis, scalar_features = _build_edge_symmetric_basis(height, width)
    flat_labels = labels.reshape(-1).astype(bool)
    model_diagnostics = {
        "constant": _fit_basis_separator(basis[:, :1], flat_labels),
        "x_squared_threshold": _fit_threshold_model(
            scalar_features["x_squared"],
            _coordinate_thresholds(scalar_features["x_squared"]),
            flat_labels,
        ),
        "y_squared_threshold": _fit_threshold_model(
            scalar_features["y_squared"],
            _coordinate_thresholds(scalar_features["y_squared"]),
            flat_labels,
        ),
        "radial_squared_threshold": _fit_threshold_model(
            scalar_features["radial_squared"],
            _coordinate_thresholds(scalar_features["radial_squared"]),
            flat_labels,
        ),
        "anisotropy_threshold": _fit_threshold_model(
            scalar_features["anisotropy"],
            _coordinate_thresholds(scalar_features["anisotropy"]),
            flat_labels,
        ),
        "d_edge_threshold": _fit_threshold_model(
            scalar_features["d_edge"],
            _coordinate_thresholds(scalar_features["d_edge"]),
            flat_labels,
        ),
        "d_edge_squared_threshold": _fit_threshold_model(
            scalar_features["d_edge_squared"],
            _coordinate_thresholds(scalar_features["d_edge_squared"]),
            flat_labels,
        ),
        "edge_symmetric_basis": _fit_basis_separator(basis, flat_labels),
    }
    best_model_name, best_model_metrics = max(
        model_diagnostics.items(),
        key=lambda item: (item[1]["best_balanced_accuracy"], item[1]["best_iou"]),
    )
    return {
        "edge_positionality_index": float(best_model_metrics["best_balanced_accuracy"]),
        "best_model_name": str(best_model_name),
        "best_iou": float(best_model_metrics["best_iou"]),
        "best_balanced_accuracy": float(best_model_metrics["best_balanced_accuracy"]),
        "coordinate_basis": "1,x^2,y^2,x^2+y^2,x^2-y^2,d_edge,d_edge^2",
        "models": model_diagnostics,
    }


def _coordinate_thresholds(values: np.ndarray) -> np.ndarray:
    unique_values = np.unique(np.asarray(values, dtype=np.float32))
    if unique_values.size <= 1:
        return np.asarray([float(unique_values[0])] if unique_values.size == 1 else [0.0], dtype=np.float32)
    return ((unique_values[:-1] + unique_values[1:]) * 0.5).astype(np.float32)


def _fit_threshold_model(values: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if thresholds.size == 0:
        thresholds = np.asarray([0.0], dtype=np.float32)
    prediction_matrix = values[None, :] > thresholds[:, None]
    metrics = _evaluate_prediction_matrix(prediction_matrix, labels)
    best_index = int(np.argmax(metrics["best_balanced_accuracy"] + (1e-6 * metrics["best_iou"])))
    return {
        "best_iou": float(metrics["best_iou"][best_index]),
        "best_balanced_accuracy": float(metrics["best_balanced_accuracy"][best_index]),
        "best_threshold": float(thresholds[best_index]),
        "assignment_used": str(metrics["assignment_used"][best_index]),
    }


def _fit_basis_separator(basis: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    design = np.asarray(basis, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    coefficients = np.linalg.pinv(design) @ targets
    scores = design @ coefficients
    unique_scores = np.unique(scores)
    if unique_scores.size <= 1:
        thresholds = np.asarray([float(unique_scores[0])] if unique_scores.size == 1 else [0.0], dtype=np.float32)
    else:
        thresholds = ((unique_scores[:-1] + unique_scores[1:]) * 0.5).astype(np.float32)
    prediction_matrix = scores[None, :] > thresholds[:, None]
    metrics = _evaluate_prediction_matrix(prediction_matrix, labels)
    best_index = int(np.argmax(metrics["best_balanced_accuracy"] + (1e-6 * metrics["best_iou"])))
    return {
        "best_iou": float(metrics["best_iou"][best_index]),
        "best_balanced_accuracy": float(metrics["best_balanced_accuracy"][best_index]),
        "best_threshold": float(thresholds[best_index]),
        "assignment_used": str(metrics["assignment_used"][best_index]),
        "coefficients": [float(value) for value in coefficients.tolist()],
    }


def _evaluate_prediction_matrix(prediction_matrix: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    predictions = np.asarray(prediction_matrix, dtype=bool)
    targets = np.asarray(labels, dtype=bool)[None, :]
    positives = float(targets.sum())
    negatives = float(targets.shape[1] - positives)
    tp = np.logical_and(predictions, targets).sum(axis=1).astype(np.float32)
    tn = np.logical_and(~predictions, ~targets).sum(axis=1).astype(np.float32)
    fp = np.logical_and(predictions, ~targets).sum(axis=1).astype(np.float32)
    fn = np.logical_and(~predictions, targets).sum(axis=1).astype(np.float32)

    direct_iou = 0.5 * (
        (tp / np.clip(tp + fp + fn, a_min=1.0, a_max=None))
        + (tn / np.clip(tn + fp + fn, a_min=1.0, a_max=None))
    )
    direct_balanced_accuracy = 0.5 * (
        (tp / np.clip(positives, a_min=1.0, a_max=None))
        + (tn / np.clip(negatives, a_min=1.0, a_max=None))
    )

    swapped_iou = 0.5 * (
        (fn / np.clip(fn + tn + tp, a_min=1.0, a_max=None))
        + (fp / np.clip(fp + tn + tp, a_min=1.0, a_max=None))
    )
    swapped_balanced_accuracy = 0.5 * (
        (fn / np.clip(positives, a_min=1.0, a_max=None))
        + (fp / np.clip(negatives, a_min=1.0, a_max=None))
    )

    use_swapped = swapped_balanced_accuracy > direct_balanced_accuracy
    best_iou = np.where(use_swapped, swapped_iou, direct_iou)
    best_balanced_accuracy = np.where(use_swapped, swapped_balanced_accuracy, direct_balanced_accuracy)
    assignment_used = np.where(use_swapped, "swapped", "direct")
    return {
        "best_iou": best_iou,
        "best_balanced_accuracy": best_balanced_accuracy,
        "assignment_used": assignment_used,
    }


def _apply_image_transform(image: Image.Image, transform_name: str) -> Image.Image:
    if transform_name == "identity":
        return image.copy()
    if transform_name == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform_name == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform_name == "hvflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    raise ValueError(f"Unsupported transform '{transform_name}'.")


def _undo_feature_transform(feature_map: np.ndarray, transform_name: str) -> np.ndarray:
    restored = np.asarray(feature_map, dtype=np.float32)
    if transform_name in {"hflip", "hvflip"}:
        restored = np.flip(restored, axis=2)
    if transform_name in {"vflip", "hvflip"}:
        restored = np.flip(restored, axis=1)
    return np.ascontiguousarray(restored)


def _infer_flip_bias_interpretation(
    *,
    mean_flip_consistency: float,
    max_flip_positionality: float,
    original_positionality: float,
    settings: dict[str, float | int | str],
) -> str:
    content_threshold = float(settings["flip_consistency_content_threshold"])
    axis_threshold = float(settings["flip_consistency_axis_threshold"])
    if mean_flip_consistency >= content_threshold:
        return "content_consistent"
    if mean_flip_consistency < axis_threshold and max_flip_positionality >= original_positionality:
        return "absolute_axis_sensitive"
    return "mixed"


def _build_constant_gray_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    constant = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
    return Image.fromarray(constant, mode="RGB")


def _build_weak_noise_image(image: Image.Image, *, seed: int, std: float) -> Image.Image:
    width, height = image.size
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=128.0, scale=std, size=(height, width, 3))
    clipped = np.clip(np.rint(noise), 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def _build_strong_blur_image(image: Image.Image, *, radius: float) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(0.0, radius)))


def _build_branch_label_diagnostics(branch_name: str, native_label_map: np.ndarray) -> dict[str, Any]:
    label_positionality = _compute_label_positionality_diagnostics(native_label_map)
    return {
        "branch_name": branch_name,
        "label_positionality_index": float(label_positionality["label_positionality_index"]),
        "best_model_name": str(label_positionality["best_model_name"]),
        "best_iou": float(label_positionality["best_iou"]),
        "best_balanced_accuracy": float(label_positionality["best_balanced_accuracy"]),
        "axis_threshold_index": float(label_positionality["axis_threshold_index"]),
        "models": label_positionality["models"],
    }


def _build_branch_edge_diagnostics(branch_name: str, native_label_map: np.ndarray) -> dict[str, Any]:
    label_positionality = _compute_edge_label_positionality_diagnostics(native_label_map)
    return {
        "branch_name": branch_name,
        "edge_positionality_index": float(label_positionality["edge_positionality_index"]),
        "best_model_name": str(label_positionality["best_model_name"]),
        "best_iou": float(label_positionality["best_iou"]),
        "best_balanced_accuracy": float(label_positionality["best_balanced_accuracy"]),
        "coordinate_basis": str(label_positionality["coordinate_basis"]),
        "models": label_positionality["models"],
    }


def _recommend_positionality_mitigation(
    baseline_branch_diagnostics: dict[str, Any],
    projection_branch_diagnostics: dict[str, Any],
    flip_avg_branch_diagnostics: dict[str, Any],
) -> tuple[str, str]:
    baseline_index = float(baseline_branch_diagnostics["label_positionality_index"])
    candidates = [
        ("projection_removed", projection_branch_diagnostics),
        ("flip_averaged", flip_avg_branch_diagnostics),
    ]
    best_name = "projection_removed"
    best_payload = projection_branch_diagnostics
    best_tuple = (
        baseline_index - float(best_payload["label_positionality_index"]),
        -float(best_payload["axis_threshold_index"]),
    )
    for name, payload in candidates[1:]:
        candidate_tuple = (
            baseline_index - float(payload["label_positionality_index"]),
            -float(payload["axis_threshold_index"]),
        )
        if candidate_tuple > best_tuple:
            best_name = name
            best_payload = payload
            best_tuple = candidate_tuple
    reason = (
        f"{best_name} lowers pooled label positionality from {baseline_index:.3f} to "
        f"{float(best_payload['label_positionality_index']):.3f} while reducing the axis-threshold index to "
        f"{float(best_payload['axis_threshold_index']):.3f}."
    )
    return best_name, reason


def _extract_multiscale_sam_features(
    backbone_out: dict[str, Any],
    torch_module: Any,
    settings: dict[str, float | int | str],
) -> list[_FeatureLevel]:
    if str(settings["feature_source"]) != "backbone_fpn":
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global currently supports only feature_source='backbone_fpn'.",
            diagnostics={"feature_source": settings["feature_source"]},
        )
    feature_levels = backbone_out.get("backbone_fpn")
    if not feature_levels:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global could not find backbone_fpn features in the official SAM state."
        )

    extracted_levels: list[_FeatureLevel] = []
    for level_index, feature_tensor in enumerate(feature_levels):
        if feature_tensor.ndim != 4 or feature_tensor.shape[0] < 1:
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global expected a 4D feature map with a batch dimension, "
                f"got {tuple(feature_tensor.shape)} at level {level_index}."
            )
        normalized_map = _normalize_feature_map(feature_tensor[:1].to(dtype=torch_module.float32)[0], torch_module)
        resolution = (int(normalized_map.shape[-2]), int(normalized_map.shape[-1]))
        extracted_levels.append(
            _FeatureLevel(
                name=f"fpn_{level_index}",
                feature_map=normalized_map,
                resolution=resolution,
            )
        )

    return sorted(
        extracted_levels,
        key=lambda level: (int(level.resolution[0] * level.resolution[1]), level.name),
    )


def _normalize_feature_map(feature_map: Any, torch_module: Any):
    norms = feature_map.norm(dim=0, keepdim=True)
    safe_denominator = torch_module.where(norms > 1e-6, norms, torch_module.ones_like(norms))
    normalized = feature_map / safe_denominator
    return torch_module.where(norms > 1e-6, normalized, torch_module.zeros_like(normalized))


def _cluster_pooled_coarsest_level_2way(
    feature_map: Any,
    torch_module: Any,
    settings: dict[str, float | int | str],
) -> tuple[np.ndarray, tuple[int, int]]:
    kernel_size = int(settings["coarsest_init_pool_kernel_size"])
    stride = int(settings["coarsest_init_pool_stride"])
    if kernel_size <= 0 or stride <= 0:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global_pooled_init requires positive pooling kernel/stride values.",
            diagnostics={
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    pooled_feature_map = torch_module.nn.functional.avg_pool2d(
        feature_map[None],
        kernel_size=kernel_size,
        stride=stride,
    )[0]
    if pooled_feature_map.shape[-2] < 1 or pooled_feature_map.shape[-1] < 1:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global_pooled_init collapsed the coarsest feature map during pooling.",
            diagnostics={
                "coarsest_resolution": (int(feature_map.shape[-2]), int(feature_map.shape[-1])),
                "coarsest_init_pool_kernel_size": kernel_size,
                "coarsest_init_pool_stride": stride,
            },
        )

    pooled_feature_map = _normalize_feature_map(pooled_feature_map.to(dtype=torch_module.float32), torch_module)
    pooled_label_map = _cluster_coarsest_level_2way(
        feature_map=pooled_feature_map,
        torch_module=torch_module,
        settings=settings,
    )
    upsampled_label_map = _upsample_label_map(
        pooled_label_map,
        target_size=(int(feature_map.shape[-2]), int(feature_map.shape[-1])),
        torch_module=torch_module,
    )
    return upsampled_label_map, (int(pooled_label_map.shape[0]), int(pooled_label_map.shape[1]))


def _cluster_coarsest_level_2way(feature_map: Any, torch_module: Any, settings: dict[str, float | int | str]) -> np.ndarray:
    channels, height, width = tuple(int(dimension) for dimension in feature_map.shape)
    del channels
    vectors = feature_map.reshape(feature_map.shape[0], height * width).transpose(0, 1).contiguous()
    if vectors.shape[0] < 2:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global requires at least two coarsest-level spatial vectors."
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
            raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
                "feature_cluster_coarse_to_fine_global collapsed to an empty cluster during coarsest-level clustering."
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

    return labels.reshape(height, width).detach().cpu().numpy().astype(np.int32)


def _refine_labels_with_finer_features(
    feature_map: Any,
    initial_label_map: np.ndarray,
    torch_module: Any,
    settings: dict[str, float | int | str],
) -> np.ndarray:
    expected_shape = (int(feature_map.shape[-2]), int(feature_map.shape[-1]))
    label_map = np.asarray(initial_label_map, dtype=np.int32)
    if label_map.shape != expected_shape:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"feature_cluster_coarse_to_fine_global expected label map {expected_shape}, got {label_map.shape}."
        )

    band_radius = int(settings["boundary_band_radius"])
    iterations = max(1, int(settings["refinement_iterations_per_level"]))
    current_labels = torch_module.as_tensor(label_map, dtype=torch_module.long, device=feature_map.device)

    for _ in range(iterations):
        band_mask = _make_boundary_band(current_labels, radius=band_radius, torch_module=torch_module)
        if int(band_mask.sum().item()) == 0:
            break

        prototype_a, prototype_b = _compute_cluster_prototypes(
            feature_map=feature_map,
            label_map=current_labels,
            band_mask=band_mask,
            torch_module=torch_module,
        )
        flattened_vectors = feature_map.reshape(feature_map.shape[0], -1).transpose(0, 1).contiguous()
        flat_labels = current_labels.reshape(-1).clone()
        uncertain_mask = band_mask.reshape(-1)
        uncertain_vectors = flattened_vectors[uncertain_mask]
        similarity_a = uncertain_vectors @ prototype_a
        similarity_b = uncertain_vectors @ prototype_b
        updated_labels = (similarity_b > similarity_a).to(dtype=torch_module.long)
        if torch_module.equal(flat_labels[uncertain_mask], updated_labels):
            break
        flat_labels[uncertain_mask] = updated_labels
        current_labels = flat_labels.reshape(expected_shape)

    refined_label_map = current_labels.detach().cpu().numpy().astype(np.int32)
    unique_values = tuple(int(value) for value in np.unique(refined_label_map))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"feature_cluster_coarse_to_fine_global expected refined labels (0, 1), got {unique_values}."
        )
    return refined_label_map


def _make_boundary_band(label_map: Any, radius: int, torch_module: Any):
    labels = label_map.to(dtype=torch_module.long)
    boundary = torch_module.zeros_like(labels, dtype=torch_module.bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if radius <= 0:
        return boundary
    dilated = torch_module.nn.functional.max_pool2d(
        boundary.to(dtype=torch_module.float32)[None, None],
        kernel_size=(radius * 2) + 1,
        stride=1,
        padding=radius,
    )
    return dilated[0, 0] > 0.5


def _compute_cluster_prototypes(feature_map: Any, label_map: Any, band_mask: Any, torch_module: Any):
    flat_vectors = feature_map.reshape(feature_map.shape[0], -1).transpose(0, 1).contiguous()
    flat_labels = label_map.reshape(-1)
    flat_band = band_mask.reshape(-1)

    support_masks = [
        torch_module.logical_and(flat_labels == 0, torch_module.logical_not(flat_band)),
        torch_module.logical_and(flat_labels == 1, torch_module.logical_not(flat_band)),
    ]
    if int(support_masks[0].sum().item()) == 0 or int(support_masks[1].sum().item()) == 0:
        support_masks = [flat_labels == 0, flat_labels == 1]
    if int(support_masks[0].sum().item()) == 0 or int(support_masks[1].sum().item()) == 0:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global could not build both prototypes at a finer feature level."
        )

    prototypes = []
    for support_mask in support_masks:
        prototype = flat_vectors[support_mask].mean(dim=0)
        prototype = prototype / prototype.norm().clamp_min(1e-6)
        prototypes.append(prototype)
    return prototypes[0], prototypes[1]


def _upsample_label_map(label_map: np.ndarray, target_size: tuple[int, int], torch_module: Any) -> np.ndarray:
    label_tensor = torch_module.as_tensor(label_map, dtype=torch_module.float32)[None, None]
    upsampled = torch_module.nn.functional.interpolate(
        label_tensor,
        size=target_size,
        mode="nearest",
    )[0, 0]
    return np.asarray(upsampled.detach().cpu().numpy(), dtype=np.int32)


def _initialize_two_centroids(vectors, torch_module: Any):
    mean_vector = vectors.mean(dim=0)
    mean_vector = mean_vector / mean_vector.norm().clamp_min(1e-6)
    first_index = int((vectors @ mean_vector).argmin().item())
    centroid_a = vectors[first_index]

    second_index = int((vectors @ centroid_a).argmin().item())
    centroid_b = vectors[second_index]
    similarity = float(torch_module.dot(centroid_a, centroid_b).item())
    if first_index == second_index or similarity >= (1.0 - 1e-6):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global could not find two distinct feature centroids."
        )
    return centroid_a, centroid_b


def _label_map_to_cluster_masks(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"feature_cluster_coarse_to_fine_global expected binary cluster labels (0, 1), got {unique_values}."
        )
    return labels == 0, labels == 1


def _align_mask_pair_to_reference(
    candidate_mask_a: np.ndarray,
    candidate_mask_b: np.ndarray,
    reference_mask_a: np.ndarray,
    reference_mask_b: np.ndarray,
) -> dict[str, Any]:
    direct_mean_iou = float(
        np.mean(
            [
                _compute_binary_iou(candidate_mask_a, reference_mask_a),
                _compute_binary_iou(candidate_mask_b, reference_mask_b),
            ]
        )
    )
    swapped_mean_iou = float(
        np.mean(
            [
                _compute_binary_iou(candidate_mask_a, reference_mask_b),
                _compute_binary_iou(candidate_mask_b, reference_mask_a),
            ]
        )
    )
    if swapped_mean_iou > direct_mean_iou:
        return {
            "mask_a": np.asarray(candidate_mask_b, dtype=bool),
            "mask_b": np.asarray(candidate_mask_a, dtype=bool),
            "alignment_used": "swapped",
            "direct_mean_iou": direct_mean_iou,
            "swapped_mean_iou": swapped_mean_iou,
            "chosen_mean_iou": swapped_mean_iou,
        }
    return {
        "mask_a": np.asarray(candidate_mask_a, dtype=bool),
        "mask_b": np.asarray(candidate_mask_b, dtype=bool),
        "alignment_used": "direct",
        "direct_mean_iou": direct_mean_iou,
        "swapped_mean_iou": swapped_mean_iou,
        "chosen_mean_iou": direct_mean_iou,
    }


def _compute_binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    boolean_a = np.asarray(mask_a, dtype=bool)
    boolean_b = np.asarray(mask_b, dtype=bool)
    intersection = int(np.logical_and(boolean_a, boolean_b).sum())
    union = int(np.logical_or(boolean_a, boolean_b).sum())
    return float(intersection / union) if union else 1.0


def _compute_label_map_component_stats(label_map: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(label_map, dtype=np.int32)
    unique_values = tuple(int(value) for value in np.unique(labels))
    if unique_values != (0, 1):
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"feature_cluster_coarse_to_fine_global expected binary cluster labels (0, 1), got {unique_values}."
        )

    return {
        "cluster_0": _compute_cluster_component_stats(labels, cluster_value=0),
        "cluster_1": _compute_cluster_component_stats(labels, cluster_value=1),
    }


def _compute_cluster_component_stats(label_map: np.ndarray, cluster_value: int) -> dict[str, Any]:
    height, width = label_map.shape
    visited = np.zeros((height, width), dtype=bool)
    component_sizes: list[int] = []

    for row_index in range(height):
        for column_index in range(width):
            if visited[row_index, column_index] or int(label_map[row_index, column_index]) != cluster_value:
                continue
            queue: deque[tuple[int, int]] = deque([(row_index, column_index)])
            visited[row_index, column_index] = True
            component_size = 0

            while queue:
                current_row, current_column = queue.popleft()
                component_size += 1
                for neighbor_row, neighbor_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                        continue
                    if visited[neighbor_row, neighbor_column]:
                        continue
                    if int(label_map[neighbor_row, neighbor_column]) != cluster_value:
                        continue
                    visited[neighbor_row, neighbor_column] = True
                    queue.append((neighbor_row, neighbor_column))

            component_sizes.append(component_size)

    size_histogram = Counter(component_sizes)
    tiny_component_pixels = int(sum(size for size in component_sizes if size <= 2))
    cluster_pixels = int((label_map == cluster_value).sum())
    return {
        "cluster_value": int(cluster_value),
        "num_components": len(component_sizes),
        "component_size_histogram": {str(size): int(count) for size, count in sorted(size_histogram.items())},
        "num_1_cell_components": int(size_histogram.get(1, 0)),
        "num_2_cell_components": int(size_histogram.get(2, 0)),
        "tiny_component_pixels": tiny_component_pixels,
        "tiny_component_pixel_fraction": float(tiny_component_pixels / cluster_pixels) if cluster_pixels else 0.0,
        "largest_component_size": int(max(component_sizes)) if component_sizes else 0,
        "smallest_component_size": int(min(component_sizes)) if component_sizes else 0,
    }


def _coerce_refined_mask(mask_array: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask_array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global refinement returned an unexpected mask shape "
            f"{array.shape}; expected a 2D mask."
        )

    result = np.asarray(array, dtype=bool)
    if result.shape != expected_shape:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            f"feature_cluster_coarse_to_fine_global refinement returned shape {result.shape}, expected {expected_shape}."
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
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global refinement expected a candidate-mask tensor, "
            f"got {masks_array.shape}."
        )

    candidates: list[dict[str, Any]] = []
    prompt_mask = np.asarray(coarse_mask, dtype=bool)
    for index, candidate_mask in enumerate(masks_array):
        normalized_mask = _coerce_refined_mask(candidate_mask, expected_shape=expected_shape)
        intersection = int(np.logical_and(normalized_mask, prompt_mask).sum())
        union = int(np.logical_or(normalized_mask, prompt_mask).sum())
        prompt_iou = float(intersection / union) if union else 1.0
        score = float(scores_array[index]) if index < len(scores_array) else 0.0
        candidates.append(
            {
                "mask": normalized_mask,
                "score": score,
                "prompt_iou": prompt_iou,
                "index": index,
            }
        )

    if not candidates:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global refinement produced no valid candidate masks after prompt alignment."
        )
    candidates.sort(key=lambda candidate: (candidate["prompt_iou"], candidate["score"]), reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def _select_prompt_aligned_mask_pair(
    candidates_a: list[dict[str, Any]],
    candidates_b: list[dict[str, Any]],
    coarse_mask_a: np.ndarray,
    coarse_mask_b: np.ndarray,
) -> dict[str, Any]:
    if not candidates_a or not candidates_b:
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global refinement requires non-empty candidate sets for both prompts."
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
        raise Sam3FeatureClusterCoarseToFineGlobalRuntimeError(
            "feature_cluster_coarse_to_fine_global refinement could not choose a valid A/B candidate pair."
        )
    return best_pair


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 1.0


def _summarize_prompt_candidates(
    candidates: list[dict[str, Any]],
    limit: int = 3,
) -> tuple[dict[str, Any], ...]:
    """Keep only the top prompt-aligned candidates for lightweight diagnostics."""

    summary: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        summary.append(
            {
                "mask": np.asarray(candidate["mask"], dtype=bool),
                "score": float(candidate["score"]),
                "prompt_iou": float(candidate["prompt_iou"]),
                "index": int(candidate["index"]),
                "rank": int(candidate.get("rank", len(summary) + 1)),
            }
        )
    return tuple(summary)
