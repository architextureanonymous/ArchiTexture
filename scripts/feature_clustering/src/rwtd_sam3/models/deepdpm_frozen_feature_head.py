"""Compact DeepDPM-style clustering head for frozen pooled SAM features.

This module does not implement the full official DeepDPM repository. The
original method is a trainable nonparametric deep clustering framework with
dynamic cluster count, split/merge steps, and amortized assignment learning.
The public package is not vendored in this repository, so this file provides a
small in-repo adaptation that preserves the key scientific property needed for
our ablation:

- the SAM encoder stays frozen
- pooled coarsest SAM feature vectors are the only inputs
- the partition head is trainable rather than a fixed k-means rule
- the cluster count is inferred dynamically with deterministic split/merge
- hard labels are emitted for nearest-neighbour upsampling and evaluation

Primary entrypoint:
- ``fit_deepdpm_frozen_features()``: train one compact DeepDPM-style head on a
  single pooled feature matrix ``[N, C]`` and return hard labels plus training
  diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rwtd_sam3.utils.deterministic_clustering import deterministic_kmeans, l2_normalize_rows


DEEPDPM_FROZEN_FEATURE_SETTINGS: dict[str, float | int] = {
    "deepdpm_seed": 0,
    "deepdpm_init_clusters": 2,
    "deepdpm_min_clusters": 1,
    "deepdpm_max_clusters": 8,
    "deepdpm_hidden_dim": 128,
    "deepdpm_embedding_dim": 64,
    "deepdpm_outer_iterations": 6,
    "deepdpm_inner_epochs": 50,
    "deepdpm_learning_rate": 1e-2,
    "deepdpm_weight_decay": 1e-4,
    "deepdpm_temperature": 0.1,
    "deepdpm_target_sharpen_power": 2.0,
    "deepdpm_balance_weight": 0.05,
    "deepdpm_separation_weight": 0.02,
    "deepdpm_split_dispersion_threshold": 0.18,
    "deepdpm_merge_similarity_threshold": 0.985,
    "deepdpm_min_cluster_fraction": 0.03,
    "deepdpm_stop_label_change_fraction": 0.002,
    "deepdpm_patience_outer": 2,
}


@dataclass(frozen=True)
class DeepDpmFrozenFeatureResult:
    """Outputs from one compact DeepDPM-style fit on pooled frozen features."""

    labels: np.ndarray
    num_clusters: int
    cluster_sizes: tuple[int, ...]
    loss_history: tuple[float, ...]
    final_loss: float
    epochs_completed: int
    inferred_k_trace: tuple[int, ...]
    split_merge_events: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def fit_deepdpm_frozen_features(
    vectors: np.ndarray,
    *,
    device: str | Any,
    settings: dict[str, float | int],
) -> DeepDpmFrozenFeatureResult:
    """Train a compact DeepDPM-style clustering head on pooled frozen features."""

    import torch
    import torch.nn.functional as F

    data = l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"DeepDPM frozen-feature head expects a 2D matrix, got shape {data.shape}.")
    if data.shape[0] < 1:
        raise ValueError("DeepDPM frozen-feature head requires at least one feature vector.")

    seed = int(settings["deepdpm_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    tensor_device = torch.device(device)
    x = torch.as_tensor(data, dtype=torch.float32, device=tensor_device)
    num_points, input_dim = int(x.shape[0]), int(x.shape[1])

    init_clusters = min(max(1, int(settings["deepdpm_init_clusters"])), num_points)
    min_clusters = min(max(1, int(settings["deepdpm_min_clusters"])), num_points)
    max_clusters = min(max(init_clusters, int(settings["deepdpm_max_clusters"])), num_points)
    hidden_dim = max(4, int(settings["deepdpm_hidden_dim"]))
    embedding_dim = max(4, int(settings["deepdpm_embedding_dim"]))

    module = _DeepDpmFrozenFeatureModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        init_num_clusters=init_clusters,
        seed=seed,
        device=tensor_device,
    )
    module.initialize_from_vectors(data, num_clusters=init_clusters)
    optimizer = _build_optimizer(
        module,
        learning_rate=float(settings["deepdpm_learning_rate"]),
        weight_decay=float(settings["deepdpm_weight_decay"]),
    )

    patience_outer = max(1, int(settings["deepdpm_patience_outer"]))
    stop_fraction = float(settings["deepdpm_stop_label_change_fraction"])
    min_cluster_fraction = float(settings["deepdpm_min_cluster_fraction"])
    split_threshold = float(settings["deepdpm_split_dispersion_threshold"])
    merge_threshold = float(settings["deepdpm_merge_similarity_threshold"])
    max_outer = max(1, int(settings["deepdpm_outer_iterations"]))
    inner_epochs = max(1, int(settings["deepdpm_inner_epochs"]))
    temperature = float(settings["deepdpm_temperature"])
    sharpen_power = float(settings["deepdpm_target_sharpen_power"])
    balance_weight = float(settings["deepdpm_balance_weight"])
    separation_weight = float(settings["deepdpm_separation_weight"])

    loss_history: list[float] = []
    split_merge_events: list[dict[str, Any]] = []
    inferred_k_trace: list[int] = [module.num_clusters]
    previous_labels: np.ndarray | None = None
    stable_rounds = 0
    epochs_completed = 0

    for outer_index in range(max_outer):
        for _ in range(inner_epochs):
            module.train()
            z, logits, assignments = module(x, temperature=temperature)
            q = torch.softmax(logits, dim=1)
            p = _build_deepdpm_target_distribution(q, power=sharpen_power)
            clustering_loss = torch.sum(
                p * (torch.log(torch.clamp(p, min=1e-6)) - torch.log(torch.clamp(q, min=1e-6))),
                dim=1,
            ).mean()
            mean_q = q.mean(dim=0)
            uniform = torch.full_like(mean_q, 1.0 / float(mean_q.numel()))
            balance_loss = torch.sum(
                uniform * (torch.log(torch.clamp(uniform, min=1e-6)) - torch.log(torch.clamp(mean_q, min=1e-6)))
            )
            normalized_centers = F.normalize(module.cluster_centers, dim=1)
            if normalized_centers.shape[0] > 1:
                row_index, col_index = torch.triu_indices(normalized_centers.shape[0], normalized_centers.shape[0], offset=1)
                upper = (normalized_centers @ normalized_centers.transpose(0, 1))[row_index, col_index]
                separation_loss = upper.mean()
            else:
                separation_loss = torch.zeros((), dtype=torch.float32, device=tensor_device)
            loss = clustering_loss + (balance_weight * balance_loss) + (separation_weight * separation_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))
            epochs_completed += 1

        with torch.no_grad():
            module.eval()
            z, logits, assignments = module(x, temperature=temperature)
            labels = assignments.detach().cpu().numpy().astype(np.int32)
            labels = _reindex_labels_by_cluster_size(labels)
            cluster_sizes = _cluster_sizes_from_labels(labels)
            label_change_fraction = 1.0 if previous_labels is None else float(np.mean(labels != previous_labels))
            previous_labels = labels.copy()

            empty_removed = False
            if 0 in cluster_sizes:
                split_merge_events.append(
                    {
                        "stage": "remove_empty",
                        "outer_iteration": outer_index,
                        "num_clusters_before": module.num_clusters,
                        "num_clusters_after": int(sum(size > 0 for size in cluster_sizes)),
                        "cluster_sizes_before": list(cluster_sizes),
                    }
                )
                module.set_cluster_centers(_remove_empty_centers(module.cluster_centers, labels))
                optimizer = _build_optimizer(
                    module,
                    learning_rate=float(settings["deepdpm_learning_rate"]),
                    weight_decay=float(settings["deepdpm_weight_decay"]),
                )
                empty_removed = True

            changed = empty_removed
            if not changed and module.num_clusters < max_clusters:
                split_event = _maybe_split_cluster(
                    embeddings=z.detach().cpu().numpy(),
                    labels=labels,
                    centers=module.cluster_centers.detach().cpu().numpy(),
                    total_points=num_points,
                    min_cluster_fraction=min_cluster_fraction,
                    split_dispersion_threshold=split_threshold,
                )
                if split_event is not None:
                    module.set_cluster_centers(torch.as_tensor(split_event["new_centers"], dtype=torch.float32, device=tensor_device))
                    optimizer = _build_optimizer(
                        module,
                        learning_rate=float(settings["deepdpm_learning_rate"]),
                        weight_decay=float(settings["deepdpm_weight_decay"]),
                    )
                    split_merge_events.append(
                        {
                            "stage": "split",
                            "outer_iteration": outer_index,
                            "num_clusters_before": split_event["num_clusters_before"],
                            "num_clusters_after": split_event["num_clusters_after"],
                            "cluster_id": split_event["cluster_id"],
                            "dispersion": split_event["dispersion"],
                            "cluster_size": split_event["cluster_size"],
                        }
                    )
                    changed = True

            if not changed and module.num_clusters > min_clusters:
                merge_event = _maybe_merge_clusters(
                    centers=module.cluster_centers.detach().cpu().numpy(),
                    labels=labels,
                    min_clusters=min_clusters,
                    min_cluster_fraction=min_cluster_fraction,
                    merge_similarity_threshold=merge_threshold,
                )
                if merge_event is not None:
                    module.set_cluster_centers(torch.as_tensor(merge_event["new_centers"], dtype=torch.float32, device=tensor_device))
                    optimizer = _build_optimizer(
                        module,
                        learning_rate=float(settings["deepdpm_learning_rate"]),
                        weight_decay=float(settings["deepdpm_weight_decay"]),
                    )
                    split_merge_events.append(
                        {
                            "stage": "merge",
                            "outer_iteration": outer_index,
                            "num_clusters_before": merge_event["num_clusters_before"],
                            "num_clusters_after": merge_event["num_clusters_after"],
                            "cluster_a": merge_event["cluster_a"],
                            "cluster_b": merge_event["cluster_b"],
                            "center_similarity": merge_event["center_similarity"],
                        }
                    )
                    changed = True

            inferred_k_trace.append(module.num_clusters)
            if changed or label_change_fraction > stop_fraction:
                stable_rounds = 0
            else:
                stable_rounds += 1
            if stable_rounds >= patience_outer:
                break

    with torch.no_grad():
        module.eval()
        _, logits, assignments = module(x, temperature=temperature)
        final_labels = assignments.detach().cpu().numpy().astype(np.int32)
        final_labels = _reindex_labels_by_cluster_size(final_labels)
        final_cluster_sizes = _cluster_sizes_from_labels(final_labels)
        if not final_cluster_sizes or any(size <= 0 for size in final_cluster_sizes):
            raise RuntimeError("DeepDPM frozen-feature head ended in a degenerate clustering state.")

    diagnostics = {
        "algorithm_family": "compact_deepdpm_style_split_merge_head",
        "seed": seed,
        "device": str(tensor_device),
        "input_num_vectors": num_points,
        "input_dim": input_dim,
        "epochs_completed": epochs_completed,
        "final_loss": float(loss_history[-1]) if loss_history else 0.0,
        "inferred_k_trace": list(inferred_k_trace),
        "split_merge_event_count": len(split_merge_events),
        "split_merge_events": split_merge_events,
        "hyperparameters": {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in settings.items()
            if key.startswith("deepdpm_")
        },
    }
    return DeepDpmFrozenFeatureResult(
        labels=final_labels,
        num_clusters=int(np.max(final_labels)) + 1,
        cluster_sizes=final_cluster_sizes,
        loss_history=tuple(float(value) for value in loss_history),
        final_loss=float(loss_history[-1]) if loss_history else 0.0,
        epochs_completed=epochs_completed,
        inferred_k_trace=tuple(int(value) for value in inferred_k_trace),
        split_merge_events=tuple(split_merge_events),
        diagnostics=diagnostics,
    )


class _DeepDpmFrozenFeatureModule:
    """Small trainable clustering head with dynamic cluster centers."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        init_num_clusters: int,
        seed: int,
        device: Any,
    ) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)
        self.cluster_centers = nn.Parameter(torch.empty((init_num_clusters, embedding_dim), dtype=torch.float32, device=device))
        nn.init.xavier_uniform_(self.cluster_centers)

    @property
    def num_clusters(self) -> int:
        return int(self.cluster_centers.shape[0])

    def parameters(self):
        return list(self.encoder.parameters()) + [self.cluster_centers]

    def train(self) -> None:
        self.encoder.train()

    def eval(self) -> None:
        self.encoder.eval()

    def encode(self, x):
        import torch.nn.functional as F
        return F.normalize(self.encoder(x), dim=1)

    def __call__(self, x, *, temperature: float):
        import torch
        import torch.nn.functional as F

        z = self.encode(x)
        centers = F.normalize(self.cluster_centers, dim=1)
        logits = (z @ centers.transpose(0, 1)) / max(float(temperature), 1e-6)
        assignments = torch.argmax(logits, dim=1)
        return z, logits, assignments

    def initialize_from_vectors(self, vectors: np.ndarray, *, num_clusters: int) -> None:
        import torch

        clustering = deterministic_kmeans(
            vectors,
            num_clusters=num_clusters,
            metric="cosine",
            max_iterations=25,
            tolerance=1e-4,
        )
        tensor = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.encode(tensor)
            centers = []
            for cluster_id in range(num_clusters):
                support = torch.as_tensor(clustering.labels == cluster_id, dtype=torch.bool, device=self.device)
                if int(support.sum().item()) == 0:
                    centers.append(z[torch.argmax(z.norm(dim=1))])
                    continue
                centers.append(z[support].mean(dim=0))
            stacked = torch.stack(centers, dim=0)
            stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self.cluster_centers = torch.nn.Parameter(stacked.to(dtype=torch.float32, device=self.device))

    def set_cluster_centers(self, new_centers) -> None:
        import torch
        import torch.nn.functional as F

        centers = F.normalize(new_centers.to(device=self.device, dtype=torch.float32), dim=1)
        self.cluster_centers = torch.nn.Parameter(centers)


def _build_optimizer(module: _DeepDpmFrozenFeatureModule, *, learning_rate: float, weight_decay: float):
    import torch
    return torch.optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)


def _build_deepdpm_target_distribution(q, *, power: float):
    import torch

    sharpened = torch.pow(torch.clamp(q, min=1e-6), float(power))
    weights = sharpened / torch.clamp(sharpened.sum(dim=0, keepdim=True), min=1e-6)
    return weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)


def _cluster_sizes_from_labels(labels: np.ndarray) -> tuple[int, ...]:
    label_values, counts = np.unique(np.asarray(labels, dtype=np.int32), return_counts=True)
    if label_values.size == 0:
        return ()
    order = np.argsort(label_values, kind="stable")
    return tuple(int(counts[index]) for index in order)


def _reindex_labels_by_cluster_size(labels: np.ndarray) -> np.ndarray:
    raw = np.asarray(labels, dtype=np.int32)
    label_values, counts = np.unique(raw, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    remapped = np.empty_like(raw, dtype=np.int32)
    for new_label, old_index in enumerate(order):
        remapped[raw == int(label_values[old_index])] = int(new_label)
    return remapped


def _remove_empty_centers(centers, labels: np.ndarray):
    import torch

    label_values = tuple(int(value) for value in np.unique(np.asarray(labels, dtype=np.int32)))
    kept = [centers[int(label_id)] for label_id in label_values]
    return torch.stack(kept, dim=0)


def _maybe_split_cluster(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    total_points: int,
    min_cluster_fraction: float,
    split_dispersion_threshold: float,
) -> dict[str, Any] | None:
    num_clusters = int(centers.shape[0])
    min_points = max(2, int(np.ceil(float(min_cluster_fraction) * float(total_points))))
    candidates: list[tuple[float, int, int]] = []
    for cluster_id in range(num_clusters):
        support = labels == cluster_id
        cluster_size = int(support.sum())
        if cluster_size < max(4, min_points * 2):
            continue
        cluster_vectors = embeddings[support]
        center = centers[cluster_id : cluster_id + 1]
        dispersion = float(np.mean(1.0 - np.clip(cluster_vectors @ center.T, -1.0, 1.0)))
        if dispersion > float(split_dispersion_threshold):
            candidates.append((dispersion, cluster_size, cluster_id))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    _, cluster_size, cluster_id = candidates[0]
    support = labels == cluster_id
    cluster_vectors = embeddings[support]
    split = deterministic_kmeans(
        cluster_vectors,
        num_clusters=2,
        metric="cosine",
        max_iterations=25,
        tolerance=1e-4,
    )
    child_sizes = _cluster_sizes_from_labels(split.labels)
    if any(size < min_points for size in child_sizes):
        return None

    new_centers = []
    for existing_cluster in range(num_clusters):
        if existing_cluster == cluster_id:
            for child_cluster in range(2):
                child_support = split.labels == child_cluster
                child_center = cluster_vectors[child_support].mean(axis=0)
                child_center = child_center / max(np.linalg.norm(child_center), 1e-6)
                new_centers.append(child_center.astype(np.float32))
        else:
            new_centers.append(np.asarray(centers[existing_cluster], dtype=np.float32))
    return {
        "num_clusters_before": num_clusters,
        "num_clusters_after": num_clusters + 1,
        "cluster_id": int(cluster_id),
        "cluster_size": int(cluster_size),
        "dispersion": float(candidates[0][0]),
        "new_centers": np.stack(new_centers, axis=0).astype(np.float32),
    }


def _maybe_merge_clusters(
    *,
    centers: np.ndarray,
    labels: np.ndarray,
    min_clusters: int,
    min_cluster_fraction: float,
    merge_similarity_threshold: float,
) -> dict[str, Any] | None:
    num_clusters = int(centers.shape[0])
    if num_clusters <= int(min_clusters):
        return None
    normalized_centers = l2_normalize_rows(np.asarray(centers, dtype=np.float32))
    labels_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    cluster_sizes = np.zeros((num_clusters,), dtype=np.int32)
    label_values, counts = np.unique(labels_array, return_counts=True)
    for label_value, count in zip(label_values, counts):
        if 0 <= int(label_value) < num_clusters:
            cluster_sizes[int(label_value)] = int(count)
    total_points = max(int(labels_array.size), 1)
    tiny_limit = max(1, int(np.ceil(float(min_cluster_fraction) * float(total_points))))

    best_pair: tuple[float, int, int] | None = None
    for cluster_a in range(num_clusters):
        for cluster_b in range(cluster_a + 1, num_clusters):
            similarity = float(np.clip(np.dot(normalized_centers[cluster_a], normalized_centers[cluster_b]), -1.0, 1.0))
            tiny_cluster = min(int(cluster_sizes[cluster_a]), int(cluster_sizes[cluster_b])) <= tiny_limit
            if similarity < float(merge_similarity_threshold) and not tiny_cluster:
                continue
            candidate = (similarity, -cluster_a, -cluster_b)
            if best_pair is None or candidate > best_pair:
                best_pair = candidate
    if best_pair is None:
        return None

    similarity, neg_a, neg_b = best_pair
    cluster_a = int(-neg_a)
    cluster_b = int(-neg_b)
    merged_center = (
        (cluster_sizes[cluster_a] * normalized_centers[cluster_a])
        + (cluster_sizes[cluster_b] * normalized_centers[cluster_b])
    )
    merged_center = merged_center / max(np.linalg.norm(merged_center), 1e-6)

    new_centers = []
    for cluster_id in range(num_clusters):
        if cluster_id == cluster_a:
            new_centers.append(merged_center.astype(np.float32))
        elif cluster_id == cluster_b:
            continue
        else:
            new_centers.append(normalized_centers[cluster_id].astype(np.float32))
    return {
        "num_clusters_before": num_clusters,
        "num_clusters_after": num_clusters - 1,
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "center_similarity": float(similarity),
        "new_centers": np.stack(new_centers, axis=0).astype(np.float32),
    }
