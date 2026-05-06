"""Deterministic clustering helpers for prompt-free feature partitioning experiments.

This module implements small reusable utilities for deterministic K-way
clustering without any random initialization. It is used by the multi-region
DeTexture ablation both for frozen SAM feature partitioning and for recovering
stable integer GT labels from JPEG-compressed color masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DeterministicClusteringResult:
    """One deterministic clustering solution."""

    labels: np.ndarray
    centroids: np.ndarray
    cluster_sizes: tuple[int, ...]
    inertia: float


@dataclass(frozen=True)
class DeterministicModelSelectionResult:
    """Best deterministic clustering choice over a small candidate K range."""

    num_clusters: int
    clustering: DeterministicClusteringResult
    criterion_name: str
    criterion_value: float
    scores_by_k: dict[int, float]


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D feature matrix row-wise."""

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 1e-6, norms, 1.0).astype(np.float32)
    normalized = matrix / safe
    return np.where(norms > 1e-6, normalized, 0.0).astype(np.float32)


def deterministic_kmeans(
    vectors: np.ndarray,
    num_clusters: int,
    *,
    metric: str,
    max_iterations: int,
    tolerance: float,
    sample_weights: np.ndarray | None = None,
) -> DeterministicClusteringResult:
    """Run deterministic Lloyd updates with farthest-point initialization.

    Supported metrics:
    - ``cosine``: vectors are L2-normalized row-wise and assigned by maximum cosine similarity
    - ``euclidean``: vectors are assigned by minimum squared Euclidean distance
    """

    data = np.asarray(vectors, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"deterministic_kmeans expects a 2D matrix, got shape {data.shape}.")
    num_points = int(data.shape[0])
    if num_points < 1:
        raise ValueError("deterministic_kmeans requires at least one vector.")
    if num_clusters < 1 or num_clusters > num_points:
        raise ValueError(
            f"deterministic_kmeans requires 1 <= num_clusters <= num_points, got {num_clusters} for {num_points} points."
        )
    if metric not in {"cosine", "euclidean"}:
        raise ValueError(f"Unsupported metric '{metric}'.")

    if sample_weights is None:
        weights = np.ones((num_points,), dtype=np.float32)
    else:
        weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != num_points:
            raise ValueError("sample_weights must have one value per vector.")
        if np.any(weights <= 0):
            raise ValueError("sample_weights must be strictly positive.")

    working = l2_normalize_rows(data) if metric == "cosine" else data.astype(np.float32, copy=True)
    centroids = _initialize_farthest_point_centroids(working, num_clusters=num_clusters, metric=metric, weights=weights)
    labels = np.full((num_points,), fill_value=-1, dtype=np.int32)

    for _ in range(int(max_iterations)):
        new_labels = _assign_clusters(working, centroids, metric=metric)
        if len(np.unique(new_labels)) != num_clusters:
            new_labels = _repair_empty_clusters(working, new_labels, centroids, metric=metric, weights=weights)
        updated_centroids = _update_centroids(
            working,
            labels=new_labels,
            num_clusters=num_clusters,
            metric=metric,
            weights=weights,
        )
        centroid_shift = float(np.linalg.norm(updated_centroids - centroids))
        centroids = updated_centroids
        if np.array_equal(new_labels, labels) or centroid_shift <= float(tolerance):
            labels = new_labels
            break
        labels = new_labels

    if len(np.unique(labels)) != num_clusters:
        raise ValueError("deterministic_kmeans ended with a degenerate empty-cluster state.")

    labels, centroids = _reindex_clusters(labels, centroids)
    cluster_sizes = tuple(int((labels == cluster_id).sum()) for cluster_id in range(num_clusters))
    inertia = _compute_inertia(working, labels, centroids, metric=metric, weights=weights)
    return DeterministicClusteringResult(
        labels=labels.astype(np.int32),
        centroids=centroids.astype(np.float32),
        cluster_sizes=cluster_sizes,
        inertia=float(inertia),
    )


def select_num_clusters_by_silhouette(
    vectors: np.ndarray,
    cluster_values: Iterable[int],
    *,
    metric: str,
    max_iterations: int,
    tolerance: float,
) -> DeterministicModelSelectionResult:
    """Select K deterministically by mean silhouette score."""

    scores_by_k: dict[int, float] = {}
    best_k: int | None = None
    best_result: DeterministicClusteringResult | None = None
    best_score = float("-inf")

    data = np.asarray(vectors, dtype=np.float32)
    for num_clusters in cluster_values:
        if num_clusters < 1 or num_clusters > data.shape[0]:
            continue
        result = deterministic_kmeans(
            data,
            num_clusters=num_clusters,
            metric=metric,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        score = compute_silhouette_score(data, result.labels, metric=metric)
        scores_by_k[int(num_clusters)] = float(score)
        candidate = (float(score), -int(num_clusters))
        best_candidate = (best_score, -best_k) if best_k is not None else (float("-inf"), 0)
        if candidate > best_candidate:
            best_k = int(num_clusters)
            best_result = result
            best_score = float(score)

    if best_k is None or best_result is None:
        raise ValueError("select_num_clusters_by_silhouette could not evaluate any valid K candidates.")

    return DeterministicModelSelectionResult(
        num_clusters=best_k,
        clustering=best_result,
        criterion_name="silhouette",
        criterion_value=float(best_score),
        scores_by_k=scores_by_k,
    )


def select_num_clusters_by_bic(
    vectors: np.ndarray,
    cluster_values: Iterable[int],
    *,
    metric: str,
    max_iterations: int,
    tolerance: float,
    sample_weights: np.ndarray | None = None,
) -> DeterministicModelSelectionResult:
    """Select K deterministically using a small BIC-style penalized inertia score."""

    data = np.asarray(vectors, dtype=np.float32)
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    scores_by_k: dict[int, float] = {}
    best_k: int | None = None
    best_result: DeterministicClusteringResult | None = None
    best_score = float("inf")
    total_weight = float(np.sum(weights)) if weights is not None else float(data.shape[0])
    dimension = float(data.shape[1])

    for num_clusters in cluster_values:
        if num_clusters < 1 or num_clusters > data.shape[0]:
            continue
        result = deterministic_kmeans(
            data,
            num_clusters=num_clusters,
            metric=metric,
            max_iterations=max_iterations,
            tolerance=tolerance,
            sample_weights=weights,
        )
        variance = max(float(result.inertia) / max(total_weight, 1.0), 1e-8)
        num_parameters = float(num_clusters) * (dimension + 1.0)
        bic = total_weight * np.log(variance) + num_parameters * np.log(max(total_weight, 2.0))
        scores_by_k[int(num_clusters)] = float(bic)
        candidate = (float(-bic), -int(num_clusters))
        best_candidate = (float(-best_score), -best_k) if best_k is not None else (float("-inf"), 0)
        if candidate > best_candidate:
            best_k = int(num_clusters)
            best_result = result
            best_score = float(bic)

    if best_k is None or best_result is None:
        raise ValueError("select_num_clusters_by_bic could not evaluate any valid K candidates.")

    return DeterministicModelSelectionResult(
        num_clusters=best_k,
        clustering=best_result,
        criterion_name="bic",
        criterion_value=float(best_score),
        scores_by_k=scores_by_k,
    )


def compute_silhouette_score(vectors: np.ndarray, labels: np.ndarray, *, metric: str) -> float:
    """Compute the mean silhouette score for a deterministic clustering."""

    data = np.asarray(vectors, dtype=np.float32)
    cluster_labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if data.ndim != 2 or cluster_labels.ndim != 1 or cluster_labels.shape[0] != data.shape[0]:
        raise ValueError("compute_silhouette_score expects matching [N, C] vectors and [N] labels.")
    unique_labels = tuple(int(value) for value in np.unique(cluster_labels))
    if len(unique_labels) <= 1:
        return 0.0

    if metric == "cosine":
        working = l2_normalize_rows(data)
        distances = 1.0 - np.clip(working @ working.T, -1.0, 1.0)
    elif metric == "euclidean":
        deltas = data[:, None, :] - data[None, :, :]
        distances = np.sum(deltas * deltas, axis=-1)
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")

    silhouettes = np.zeros((data.shape[0],), dtype=np.float32)
    for index in range(data.shape[0]):
        label = cluster_labels[index]
        same_cluster = cluster_labels == label
        same_cluster[index] = False
        if int(same_cluster.sum()) == 0:
            silhouettes[index] = 0.0
            continue
        intra = float(np.mean(distances[index, same_cluster]))
        nearest_other = float("inf")
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = cluster_labels == other_label
            if int(other_cluster.sum()) == 0:
                continue
            nearest_other = min(nearest_other, float(np.mean(distances[index, other_cluster])))
        if not np.isfinite(nearest_other):
            silhouettes[index] = 0.0
            continue
        denominator = max(intra, nearest_other)
        silhouettes[index] = 0.0 if denominator <= 1e-8 else float((nearest_other - intra) / denominator)
    return float(np.mean(silhouettes))


def _initialize_farthest_point_centroids(
    vectors: np.ndarray,
    *,
    num_clusters: int,
    metric: str,
    weights: np.ndarray,
) -> np.ndarray:
    mean_vector = np.average(vectors, axis=0, weights=weights).astype(np.float32)
    if metric == "cosine":
        mean_vector = l2_normalize_rows(mean_vector[None])[0]
    first_index = int(np.argmax(_distance_to_reference(vectors, mean_vector, metric=metric)))
    chosen_indices = [first_index]

    while len(chosen_indices) < num_clusters:
        chosen = vectors[np.asarray(chosen_indices, dtype=np.int32)]
        distance_matrix = _pairwise_distances(vectors, chosen, metric=metric)
        nearest_distance = np.min(distance_matrix, axis=1)
        nearest_distance[np.asarray(chosen_indices, dtype=np.int32)] = -1.0
        next_index = int(np.argmax(nearest_distance))
        if next_index in chosen_indices or float(nearest_distance[next_index]) <= 1e-8:
            raise ValueError("Deterministic farthest-point init could not find enough distinct centroids.")
        chosen_indices.append(next_index)
    return vectors[np.asarray(chosen_indices, dtype=np.int32)].astype(np.float32)


def _assign_clusters(vectors: np.ndarray, centroids: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "cosine":
        similarities = vectors @ centroids.T
        return np.argmax(similarities, axis=1).astype(np.int32)
    distances = _pairwise_distances(vectors, centroids, metric=metric)
    return np.argmin(distances, axis=1).astype(np.int32)


def _repair_empty_clusters(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: str,
    weights: np.ndarray,
) -> np.ndarray:
    repaired = np.asarray(labels, dtype=np.int32).copy()
    num_clusters = int(centroids.shape[0])
    for cluster_id in range(num_clusters):
        if int((repaired == cluster_id).sum()) > 0:
            continue
        cluster_weights = np.array([float(weights[repaired == idx].sum()) for idx in range(num_clusters)], dtype=np.float32)
        donor_cluster = int(np.argmax(cluster_weights))
        donor_indices = np.where(repaired == donor_cluster)[0]
        if donor_indices.size <= 1:
            raise ValueError("Deterministic empty-cluster repair could not find a donor cluster with more than one point.")
        donor_distances = _distance_to_reference(vectors[donor_indices], centroids[donor_cluster], metric=metric)
        reassigned_index = int(donor_indices[int(np.argmax(donor_distances))])
        repaired[reassigned_index] = cluster_id
    return repaired


def _update_centroids(
    vectors: np.ndarray,
    *,
    labels: np.ndarray,
    num_clusters: int,
    metric: str,
    weights: np.ndarray,
) -> np.ndarray:
    centroids: list[np.ndarray] = []
    for cluster_id in range(num_clusters):
        support = labels == cluster_id
        if int(support.sum()) == 0:
            raise ValueError("deterministic_kmeans encountered an empty cluster during centroid update.")
        cluster_weights = weights[support]
        centroid = np.average(vectors[support], axis=0, weights=cluster_weights).astype(np.float32)
        if metric == "cosine":
            centroid = l2_normalize_rows(centroid[None])[0]
        centroids.append(centroid.astype(np.float32))
    return np.stack(centroids, axis=0).astype(np.float32)


def _pairwise_distances(vectors: np.ndarray, references: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "cosine":
        return 1.0 - np.clip(vectors @ references.T, -1.0, 1.0)
    deltas = vectors[:, None, :] - references[None, :, :]
    return np.sum(deltas * deltas, axis=-1)


def _distance_to_reference(vectors: np.ndarray, reference: np.ndarray, *, metric: str) -> np.ndarray:
    return _pairwise_distances(vectors, reference[None], metric=metric)[:, 0]


def _compute_inertia(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: str,
    weights: np.ndarray,
) -> float:
    distances = _pairwise_distances(vectors, centroids, metric=metric)
    assigned = distances[np.arange(vectors.shape[0]), labels]
    return float(np.sum(assigned * weights))


def _reindex_clusters(labels: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(np.flip(centroids, axis=1).T)
    remapped_labels = np.empty_like(labels, dtype=np.int32)
    remapped_centroids = centroids[order].astype(np.float32)
    for new_label, old_label in enumerate(order):
        remapped_labels[labels == int(old_label)] = int(new_label)
    return remapped_labels, remapped_centroids
