from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import cosine_similarity, region_variance

EPS = 1e-8


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass(frozen=True)
class MergeConfig:
    adjacency_dilation: int = 3
    merge_threshold: float = 0.50
    w_texture: float = 0.65
    w_boundary: float = 0.30
    w_hetero: float = 0.20


@dataclass(frozen=True)
class EdgeDecision:
    i: int
    j: int
    score: float
    texture_sim: float
    boundary_weakness: float
    hetero_penalty: float
    merged: bool


def _boundary_map(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    g = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return g.astype(bool)


def _normalized_grad(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    gm = gm / (float(gm.max()) + EPS)
    return gm


def _interface_region(mask_a: np.ndarray, mask_b: np.ndarray, dilated_a: np.ndarray, dilated_b: np.ndarray) -> np.ndarray:
    # Pixels where masks are close but not strongly overlapping.
    near = np.logical_and(dilated_a > 0, dilated_b > 0)
    interior = np.logical_or(mask_a > 0, mask_b > 0)
    return np.logical_and(near, np.logical_not(interior))


def _components_from_uf(uf: UnionFind, n: int) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


def merge_masks(
    image_rgb: np.ndarray,
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
    feature_map: np.ndarray,
    cfg: MergeConfig,
) -> tuple[list[np.ndarray], list[EdgeDecision]]:
    n = len(proposals)
    if n == 0:
        return [], []
    if n == 1:
        return [proposals[0].astype(np.uint8)], []

    kernel_size = 2 * cfg.adjacency_dilation + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = [cv2.dilate((m > 0).astype(np.uint8), k) for m in proposals]
    boundaries = [_boundary_map(m) for m in proposals]
    grad = _normalized_grad(image_rgb)
    var_cache = [region_variance(feature_map, m) for m in proposals]

    uf = UnionFind(n)
    decisions: list[EdgeDecision] = []

    for i in range(n):
        for j in range(i + 1, n):
            near = np.logical_and(dilated[i] > 0, dilated[j] > 0)
            contact = np.logical_and(boundaries[i], boundaries[j])
            if not np.any(near) and not np.any(contact):
                continue

            tex = cosine_similarity(descriptors[i], descriptors[j])
            interface = _interface_region(proposals[i], proposals[j], dilated[i], dilated[j])
            if np.any(interface):
                boundary_strength = float(grad[interface].mean())
            else:
                boundary_strength = float(grad[np.logical_and(near, np.logical_not(contact))].mean()) if np.any(near) else 1.0
            boundary_weakness = 1.0 - np.clip(boundary_strength, 0.0, 1.0)

            union_mask = np.logical_or(proposals[i] > 0, proposals[j] > 0).astype(np.uint8)
            union_var = region_variance(feature_map, union_mask)
            baseline_var = 0.5 * (var_cache[i] + var_cache[j])
            hetero_penalty = max(0.0, union_var - baseline_var)

            score = (
                cfg.w_texture * tex
                + cfg.w_boundary * boundary_weakness
                - cfg.w_hetero * hetero_penalty
            )
            do_merge = score >= cfg.merge_threshold
            if do_merge:
                uf.union(i, j)

            decisions.append(
                EdgeDecision(
                    i=i,
                    j=j,
                    score=float(score),
                    texture_sim=float(tex),
                    boundary_weakness=float(boundary_weakness),
                    hetero_penalty=float(hetero_penalty),
                    merged=bool(do_merge),
                )
            )

    comps = _components_from_uf(uf, n)
    merged_masks: list[np.ndarray] = []
    for comp in comps:
        out = np.zeros_like(proposals[0], dtype=np.uint8)
        for idx in comp:
            out = np.logical_or(out, proposals[idx] > 0)
        merged_masks.append(out.astype(np.uint8))

    return merged_masks, decisions
