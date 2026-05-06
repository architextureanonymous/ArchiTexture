"""Repo-wide evaluation contract metadata and validation helpers.

This module is the single source of truth for repository-level evaluation
policy. It intentionally does not centralize metric computation yet. Existing
routes still compute metrics inside their own evaluators, but they should obey
the split-role, protocol-family, artifact, and primary-metric rules defined
here.
"""

from __future__ import annotations

from dataclasses import dataclass


REPO_EVALUATION_CONTRACT_VERSION = "repo_eval_v1"

# Backward-compatibility constants for the older shared binary comparison view.
LEGACY_COMPARISON_VIEW_CONTRACT = "architexture_binary_v1"
LEGACY_COMPARISON_VIEW_PRIMARY_METRIC = "eval_miou"
LEGACY_COMPARISON_VIEW_SECONDARY_METRIC = "eval_ari"

SUPPORTED_PROTOCOL_FAMILIES = (
    "upstream_faithful",
    "repo_comparable",
    "route_primary",
)
SUPPORTED_METRIC_VIEWS = (
    "route_primary",
    "partition_invariant",
    "direct_foreground",
    "upstream_faithful",
)
SUPPORTED_SAFETY_TAGS = (
    "paper_safe",
    "ablation_only",
    "debug_only",
)
SUPPORTED_SPLIT_ROLES = ("train", "val", "test")
REQUIRED_SELECTION_SPLIT = "val"
REQUIRED_HEADLINE_SPLIT = "test"


@dataclass(frozen=True)
class MetricPair:
    """A route-facing primary metric pair."""

    metric_view: str
    primary_metric_name: str
    secondary_metric_name: str


@dataclass(frozen=True)
class RouteEvaluationContract:
    """Repo-wide evaluation requirements for one benchmark route."""

    route_name: str
    summary: str
    default_protocol_family: str
    supported_protocol_families: tuple[str, ...]
    default_primary_metric_pair: MetricPair
    allowed_primary_metric_pairs: tuple[MetricPair, ...]
    supported_metric_views: tuple[str, ...]
    fairness_notes: tuple[str, ...]


PARTITION_PRIMARY = MetricPair(
    metric_view="partition_invariant",
    primary_metric_name="eval_miou",
    secondary_metric_name="eval_ari",
)
DIRECT_FOREGROUND_PRIMARY = MetricPair(
    metric_view="direct_foreground",
    primary_metric_name="direct_foreground_iou",
    secondary_metric_name="direct_foreground_dice",
)
UPSTREAM_FAITHFUL_PRIMARY = MetricPair(
    metric_view="upstream_faithful",
    primary_metric_name="upstream_eval_iou",
    secondary_metric_name="upstream_eval_dice",
)


ROUTE_EVALUATION_CONTRACTS: dict[str, RouteEvaluationContract] = {
    "rwtd": RouteEvaluationContract(
        route_name="rwtd",
        summary="Two-region texture partition benchmark.",
        default_protocol_family="route_primary",
        supported_protocol_families=("route_primary", "repo_comparable"),
        default_primary_metric_pair=PARTITION_PRIMARY,
        allowed_primary_metric_pairs=(PARTITION_PRIMARY,),
        supported_metric_views=("route_primary", "partition_invariant"),
        fairness_notes=(
            "Partition-invariant metrics are primary.",
            "Direct foreground metrics are not route-defining.",
        ),
    ),
    "detexture_ade20k": RouteEvaluationContract(
        route_name="detexture_ade20k",
        summary="Binary DeTexture ADE20K route under a partition view.",
        default_protocol_family="route_primary",
        supported_protocol_families=("route_primary", "repo_comparable"),
        default_primary_metric_pair=PARTITION_PRIMARY,
        allowed_primary_metric_pairs=(PARTITION_PRIMARY,),
        supported_metric_views=("route_primary", "partition_invariant"),
        fairness_notes=(
            "Paper-facing reporting should remain partition-invariant.",
        ),
    ),
    "caid": RouteEvaluationContract(
        route_name="caid",
        summary="CAID binary route treated as a partition benchmark.",
        default_protocol_family="route_primary",
        supported_protocol_families=("route_primary", "repo_comparable"),
        default_primary_metric_pair=PARTITION_PRIMARY,
        allowed_primary_metric_pairs=(PARTITION_PRIMARY,),
        supported_metric_views=("route_primary", "partition_invariant"),
        fairness_notes=(
            "Fail the route if preprocessing erases a class.",
        ),
    ),
    "stld": RouteEvaluationContract(
        route_name="stld",
        summary="STLD foreground segmentation route.",
        default_protocol_family="route_primary",
        supported_protocol_families=("route_primary", "repo_comparable"),
        default_primary_metric_pair=DIRECT_FOREGROUND_PRIMARY,
        allowed_primary_metric_pairs=(DIRECT_FOREGROUND_PRIMARY,),
        supported_metric_views=(
            "route_primary",
            "partition_invariant",
            "direct_foreground",
        ),
        fairness_notes=(
            "Do not select checkpoints on test.",
            "Direct foreground metrics are the benchmark primary view.",
        ),
    ),
    "glas": RouteEvaluationContract(
        route_name="glas",
        summary="GlaS foreground segmentation route.",
        default_protocol_family="route_primary",
        supported_protocol_families=("route_primary", "repo_comparable"),
        default_primary_metric_pair=DIRECT_FOREGROUND_PRIMARY,
        allowed_primary_metric_pairs=(DIRECT_FOREGROUND_PRIMARY,),
        supported_metric_views=(
            "route_primary",
            "partition_invariant",
            "direct_foreground",
        ),
        fairness_notes=(
            "Direct foreground metrics are the benchmark primary view.",
        ),
    ),
    "monuseg_frozen_head": RouteEvaluationContract(
        route_name="monuseg_frozen_head",
        summary="Frozen-feature CNN-head MoNuSeg route.",
        default_protocol_family="repo_comparable",
        supported_protocol_families=(
            "repo_comparable",
            "route_primary",
            "upstream_faithful",
        ),
        default_primary_metric_pair=UPSTREAM_FAITHFUL_PRIMARY,
        allowed_primary_metric_pairs=(
            UPSTREAM_FAITHFUL_PRIMARY,
            DIRECT_FOREGROUND_PRIMARY,
        ),
        supported_metric_views=(
            "route_primary",
            "partition_invariant",
            "direct_foreground",
            "upstream_faithful",
        ),
        fairness_notes=(
            "The run must declare which MoNuSeg view is paper-facing.",
            "AutoSAM-style metrics and repo-native metrics must not be mixed silently.",
        ),
    ),
    "monuseg_autosam_reproduction": RouteEvaluationContract(
        route_name="monuseg_autosam_reproduction",
        summary="Protocol-faithful AutoSAM reproduction route on MoNuSeg.",
        default_protocol_family="upstream_faithful",
        supported_protocol_families=("upstream_faithful",),
        default_primary_metric_pair=UPSTREAM_FAITHFUL_PRIMARY,
        allowed_primary_metric_pairs=(UPSTREAM_FAITHFUL_PRIMARY,),
        supported_metric_views=(
            "route_primary",
            "partition_invariant",
            "direct_foreground",
            "upstream_faithful",
        ),
        fairness_notes=(
            "Upstream-faithful metrics define the headline result.",
            "All protocol deviations must be machine-recorded.",
        ),
    ),
}


def get_route_evaluation_contract(route_name: str) -> RouteEvaluationContract:
    """Return the frozen route contract or raise for unknown routes."""

    try:
        return ROUTE_EVALUATION_CONTRACTS[route_name]
    except KeyError as exc:
        available = ", ".join(sorted(ROUTE_EVALUATION_CONTRACTS))
        raise KeyError(
            f"Unknown route '{route_name}'. Available routes: {available}."
        ) from exc


def validate_repo_split_contract(selection_split: str, headline_split: str) -> None:
    """Enforce the repo-wide fairness rules for split roles."""

    if selection_split not in SUPPORTED_SPLIT_ROLES:
        raise ValueError(
            f"Unsupported selection split '{selection_split}'. "
            f"Expected one of {SUPPORTED_SPLIT_ROLES}."
        )
    if headline_split not in SUPPORTED_SPLIT_ROLES:
        raise ValueError(
            f"Unsupported headline split '{headline_split}'. "
            f"Expected one of {SUPPORTED_SPLIT_ROLES}."
        )
    if selection_split != REQUIRED_SELECTION_SPLIT:
        raise ValueError(
            "Repo evaluation contract forbids checkpoint selection outside the "
            f"validation split. Expected selection_split='{REQUIRED_SELECTION_SPLIT}', "
            f"got '{selection_split}'."
        )
    if headline_split != REQUIRED_HEADLINE_SPLIT:
        raise ValueError(
            "Repo evaluation contract requires test-only headline reporting. "
            f"Expected headline_split='{REQUIRED_HEADLINE_SPLIT}', "
            f"got '{headline_split}'."
        )


def build_protocol_record(
    *,
    route_name: str,
    protocol_family: str,
    primary_metric_pair: MetricPair,
    selection_split: str,
    headline_split: str,
    safety_tag: str,
    allow_train_selection: bool = False,
    deviations: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build a normalized protocol record for artifact writing."""

    route_contract = get_route_evaluation_contract(route_name)
    if allow_train_selection:
        if selection_split != "train":
            raise ValueError(
                "allow_train_selection=True only permits selection_split='train'. "
                f"Got selection_split='{selection_split}'."
            )
        if headline_split != REQUIRED_HEADLINE_SPLIT:
            raise ValueError(
                "allow_train_selection=True still requires headline_split='test'. "
                f"Got headline_split='{headline_split}'."
            )
    else:
        validate_repo_split_contract(selection_split, headline_split)

    if protocol_family not in SUPPORTED_PROTOCOL_FAMILIES:
        raise ValueError(
            f"Unsupported protocol family '{protocol_family}'. "
            f"Expected one of {SUPPORTED_PROTOCOL_FAMILIES}."
        )
    if protocol_family not in route_contract.supported_protocol_families:
        raise ValueError(
            f"Route '{route_name}' does not support protocol family "
            f"'{protocol_family}'. Allowed values: "
            f"{route_contract.supported_protocol_families}."
        )
    if primary_metric_pair not in route_contract.allowed_primary_metric_pairs:
        raise ValueError(
            f"Route '{route_name}' does not allow primary metric pair "
            f"{primary_metric_pair}. Allowed values: "
            f"{route_contract.allowed_primary_metric_pairs}."
        )
    if safety_tag not in SUPPORTED_SAFETY_TAGS:
        raise ValueError(
            f"Unsupported safety tag '{safety_tag}'. "
            f"Expected one of {SUPPORTED_SAFETY_TAGS}."
        )

    return {
        "repo_evaluation_contract_version": REPO_EVALUATION_CONTRACT_VERSION,
        "route_name": route_contract.route_name,
        "route_summary": route_contract.summary,
        "protocol_family": protocol_family,
        "supported_metric_views": list(route_contract.supported_metric_views),
        "primary_metric": {
            "metric_view": primary_metric_pair.metric_view,
            "primary_metric_name": primary_metric_pair.primary_metric_name,
            "secondary_metric_name": primary_metric_pair.secondary_metric_name,
        },
        "selection_split": selection_split,
        "headline_split": headline_split,
        "safety_tag": safety_tag,
        "deviations": list(deviations or ()),
        "fairness_notes": list(route_contract.fairness_notes),
    }
