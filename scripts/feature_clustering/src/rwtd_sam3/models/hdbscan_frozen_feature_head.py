"""Minimal HDBSCAN partition head for frozen pooled SAM features.

This helper wraps sklearn's built-in ``HDBSCAN`` so the repository can run a
simple nonparametric clustering ablation on the same pooled coarsest frozen SAM
feature vectors used by the deterministic multi-region baseline.

The frozen SAM encoder, coarsest-level selection, pooling, normalization, and
nearest-neighbor upsampling all remain outside this file. This module only
replaces the partition head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rwtd_sam3.utils.deterministic_clustering import l2_normalize_rows


HDBSCAN_FROZEN_FEATURE_SETTINGS: dict[str, float | int | str | bool | None] = {
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples": None,
    "hdbscan_cluster_selection_epsilon": 0.0,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_allow_single_cluster": True,
    "hdbscan_all_noise_policy": "collapse_to_single_cluster",
}


@dataclass(frozen=True)
class HdbscanFrozenFeatureResult:
    """Outputs from one HDBSCAN fit on pooled frozen features."""

    labels: np.ndarray
    num_clusters: int
    cluster_sizes: tuple[int, ...]
    diagnostics: dict[str, Any]


def fit_hdbscan_frozen_features(
    vectors: np.ndarray,
    *,
    settings: dict[str, float | int | str | bool | None],
) -> HdbscanFrozenFeatureResult:
    """Run sklearn HDBSCAN on pooled frozen features and emit hard labels."""

    from sklearn.cluster import HDBSCAN

    data = l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"HDBSCAN frozen-feature head expects a 2D matrix, got shape {data.shape}.")
    if data.shape[0] < 2:
        raise ValueError("HDBSCAN frozen-feature head requires at least two pooled feature vectors.")

    requested_min_cluster_size = int(settings.get("hdbscan_min_cluster_size", 5) or 5)
    min_cluster_size = max(2, min(requested_min_cluster_size, int(data.shape[0])))
    requested_min_samples = settings.get("hdbscan_min_samples", None)
    min_samples = None if requested_min_samples is None else max(1, min(int(requested_min_samples), int(data.shape[0])))

    estimator = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=float(settings.get("hdbscan_cluster_selection_epsilon", 0.0) or 0.0),
        metric=str(settings.get("hdbscan_metric", "euclidean")),
        cluster_selection_method=str(settings.get("hdbscan_cluster_selection_method", "eom")),
        allow_single_cluster=bool(settings.get("hdbscan_allow_single_cluster", True)),
        n_jobs=1,
        copy=True,
    )
    raw_labels = estimator.fit_predict(data).astype(np.int32)
    if raw_labels.shape[0] != data.shape[0]:
        raise RuntimeError("HDBSCAN frozen-feature head returned a label vector with an unexpected shape.")

    raw_noise_point_count = int(np.sum(raw_labels == -1))
    raw_noise_fraction = float(raw_noise_point_count / float(data.shape[0]))
    raw_num_non_noise_clusters = int(len(set(int(label) for label in raw_labels.tolist() if int(label) >= 0)))
    all_noise_policy = str(settings.get("hdbscan_all_noise_policy", "collapse_to_single_cluster"))
    all_noise_repair_used = False
    if np.all(raw_labels == -1):
        if all_noise_policy != "collapse_to_single_cluster":
            raise RuntimeError(
                "HDBSCAN frozen-feature head labeled every pooled feature vector as noise and the configured "
                f"all-noise policy '{all_noise_policy}' does not define a deterministic repair."
            )
        raw_labels = np.zeros_like(raw_labels, dtype=np.int32)
        all_noise_repair_used = True

    labels = _reindex_hdbscan_labels(raw_labels)
    label_values, counts = np.unique(labels, return_counts=True)
    cluster_sizes = tuple(int(counts[index]) for index in np.argsort(label_values, kind="stable"))
    noise_point_count = 0 if all_noise_repair_used else raw_noise_point_count
    diagnostics = {
        "algorithm_family": "sklearn_hdbscan",
        "requested_min_cluster_size": requested_min_cluster_size,
        "effective_min_cluster_size": min_cluster_size,
        "requested_min_samples": requested_min_samples,
        "effective_min_samples": min_samples,
        "cluster_selection_epsilon": float(settings.get("hdbscan_cluster_selection_epsilon", 0.0) or 0.0),
        "cluster_selection_method": str(settings.get("hdbscan_cluster_selection_method", "eom")),
        "metric": str(settings.get("hdbscan_metric", "euclidean")),
        "allow_single_cluster": bool(settings.get("hdbscan_allow_single_cluster", True)),
        "all_noise_policy": all_noise_policy,
        "all_noise_repair_used": all_noise_repair_used,
        "noise_point_count": noise_point_count,
        "noise_fraction": float(noise_point_count / float(data.shape[0])),
        "noise_point_count_before_repair": raw_noise_point_count,
        "noise_fraction_before_repair": raw_noise_fraction,
        "probabilities_mean": float(np.mean(getattr(estimator, 'probabilities_', np.ones((data.shape[0],), dtype=np.float32)))),
        "cluster_persistence": [float(value) for value in getattr(estimator, 'cluster_persistence_', [])],
        "raw_num_non_noise_clusters": raw_num_non_noise_clusters,
        "final_num_clusters": int(np.max(labels)) + 1,
    }
    return HdbscanFrozenFeatureResult(
        labels=labels,
        num_clusters=int(np.max(labels)) + 1,
        cluster_sizes=cluster_sizes,
        diagnostics=diagnostics,
    )


def _reindex_hdbscan_labels(raw_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(raw_labels, dtype=np.int32).reshape(-1)
    label_values, counts = np.unique(labels, return_counts=True)
    order = sorted(
        range(len(label_values)),
        key=lambda idx: (-int(counts[idx]), int(label_values[idx])),
    )
    remapped = np.empty_like(labels, dtype=np.int32)
    for new_label, source_index in enumerate(order):
        remapped[labels == int(label_values[source_index])] = int(new_label)
    return remapped
