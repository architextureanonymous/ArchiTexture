"""Cross-dataset experiment registry for current-method frozen-feature variants.

This module is the single source of truth for frozen-feature clustering experiments
that are intended to run across the repository's binary texture datasets:
RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS. It exists to prevent experiment
drift between CLI surfaces, dataset adapters, and README tables.

Primary entrypoints:
- ``get_cross_dataset_experiment_spec()``: resolve one registered experiment.
- ``resolve_cross_dataset_experiment_model_id()``: map the shared CLI model-id
  flag onto the correct backbone-specific default.
- ``build_cross_dataset_experiment_runner()``: instantiate the experiment's
  runner with the standard model/device arguments.

Inputs are experiment ids plus the shared frozen-backbone backend arguments.
Outputs are registered metadata objects and initialized runner instances. Only
experiments that produce a binary two-mask partition through the repository's
coarse partitioning runner family are registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rwtd_sam3.models.sam2_feature_cluster_runner import (
    Sam2FeatureClusterCoarseOnlyRunner,
    Sam2FeatureClusterFlipAvgCoarseOnlyRunner,
)
from rwtd_sam3.models.sam2_runner import DEFAULT_SAM2_MODEL_ID
from rwtd_sam3.models.sam3_boundary_refine_sweep_runner import Sam3BoundaryRefineSweepRunner
from rwtd_sam3.models.sam3_feature_cluster_coarse_to_fine_runner import (
    Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner,
    Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner,
    Sam3FeatureClusterCoarseToFineGlobalRunner,
)


CROSS_DATASET_EXPERIMENT_DATASETS = ("rwtd", "stld", "caid", "detexture_ade20k", "cstd", "glas")


@dataclass(frozen=True)
class CrossDatasetExperimentSpec:
    """Metadata for one experiment that is expected to run on every registered binary dataset.

    Attributes:
        name: Stable CLI / artifact identifier.
        summary: One-line repo-facing description used in docs.
        runner_cls: Callable runner class initialized with the shared frozen-backbone
            arguments.
        supported_datasets: Dataset ids this experiment is expected to support.
            Current experiments registered here must include every binary
            dataset promoted into the cross-dataset contract.
        visual_contract: Preview contract identifier used in docs and run
            metadata.
        promotion_note: Short note that explains why the experiment belongs in
            the cross-dataset registry.
        backbone_family: Frozen backbone family used by the experiment.
        default_model_id: Recommended default model id for the selected
            backbone family.
    """

    name: str
    summary: str
    runner_cls: type
    supported_datasets: tuple[str, ...]
    visual_contract: str
    promotion_note: str
    backbone_family: str
    default_model_id: str


def _build_registry() -> dict[str, CrossDatasetExperimentSpec]:
    return {
        "feature_cluster_coarse_to_fine_global": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global",
            summary="Coarsest-only global init, finer-level boundary-band updates, then SAM mask prompting.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_to_fine_global_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init",
            summary="Average-pool the coarsest feature level before the multiscale coarse-to-fine run.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalPooledInitRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_to_fine_global_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_coarse_only",
            summary="Pooled coarsest partition only; no finer refinement and no SAM prompt refinement.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalPooledInitCoarseOnlyRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_only_three_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2",
            summary="Vanilla pooled coarsest CFC head on a frozen SAM-2 coarsest image embedding.",
            runner_cls=Sam2FeatureClusterCoarseOnlyRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_only_three_panel",
            promotion_note="Registered current-method ablation that swaps only the frozen backbone from SAM-3 to SAM-2 while keeping the vanilla CFC head unchanged.",
            backbone_family="sam2",
            default_model_id=DEFAULT_SAM2_MODEL_ID,
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only",
            summary="Flip-average coarsest features across identity/hflip/vflip/hvflip, then run pooled coarse-only.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalPooledInitFlipAvgCoarseOnlyRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_only_three_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only_sam2",
            summary="Flip-average frozen SAM-2 coarsest image embeddings, then run the same pooled coarse-only CFC head.",
            runner_cls=Sam2FeatureClusterFlipAvgCoarseOnlyRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_only_three_panel",
            promotion_note="Registered current-method ablation that keeps the vanilla flip-averaged CFC pipeline fixed while swapping the frozen backbone from SAM-3 to SAM-2.",
            backbone_family="sam2",
            default_model_id=DEFAULT_SAM2_MODEL_ID,
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_debiased_coarse_only",
            summary="Position-leakage diagnosis branch with projection removal, flip tests, and null-image controls.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalPooledInitDebiasedCoarseOnlyRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="positionality_diagnosis_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "feature_cluster_coarse_to_fine_global_pooled_init_direct": CrossDatasetExperimentSpec(
            name="feature_cluster_coarse_to_fine_global_pooled_init_direct",
            summary="Use the pooled coarsest partition as the SAM mask prompt, with no finer feature refinement.",
            runner_cls=Sam3FeatureClusterCoarseToFineGlobalPooledInitDirectRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="coarse_to_fine_global_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
        "boundary_refine_sweep": CrossDatasetExperimentSpec(
            name="boundary_refine_sweep",
            summary="Compare conservative stage-B boundary-only local refinement variants on top of the strong flip-averaged coarse partition.",
            runner_cls=Sam3BoundaryRefineSweepRunner,
            supported_datasets=CROSS_DATASET_EXPERIMENT_DATASETS,
            visual_contract="boundary_refine_sweep_panel",
            promotion_note="Registered current-method experiment; expected to run on RWTD, STLD, CAID, DeTexture ADE20K, CSTD, and GlaS.",
            backbone_family="sam3",
            default_model_id="facebook/sam3",
        ),
    }


CROSS_DATASET_EXPERIMENT_REGISTRY = _build_registry()
CROSS_DATASET_EXPERIMENT_VARIANTS = tuple(CROSS_DATASET_EXPERIMENT_REGISTRY.keys())
DEFAULT_CROSS_DATASET_EXPERIMENT = "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only"


def get_cross_dataset_experiment_spec(variant: str) -> CrossDatasetExperimentSpec:
    """Return the registered cross-dataset experiment metadata for ``variant``."""

    try:
        return CROSS_DATASET_EXPERIMENT_REGISTRY[variant]
    except KeyError as exc:
        expected = ", ".join(CROSS_DATASET_EXPERIMENT_VARIANTS)
        raise ValueError(f"Unknown cross-dataset experiment '{variant}'. Expected one of: {expected}.") from exc


def resolve_cross_dataset_experiment_model_id(variant: str, requested_model_id: str | None) -> str:
    """Resolve the actual model id used by one registered experiment."""

    spec = get_cross_dataset_experiment_spec(variant)
    if requested_model_id in {None, "", "facebook/sam3"} and spec.backbone_family == "sam2":
        return spec.default_model_id
    return requested_model_id or spec.default_model_id


def build_cross_dataset_experiment_runner(
    variant: str,
    *,
    model_id: str,
    device: str,
    hf_token: str | None,
    official_checkpoint_path: str | None,
) -> Any:
    """Instantiate the registered runner for one cross-dataset experiment."""

    spec = get_cross_dataset_experiment_spec(variant)
    resolved_model_id = resolve_cross_dataset_experiment_model_id(variant, model_id)
    return spec.runner_cls(
        model_id=resolved_model_id,
        device=device,
        hf_token=hf_token,
        official_checkpoint_path=official_checkpoint_path,
    )
