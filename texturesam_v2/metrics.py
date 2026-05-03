from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .io_utils import ensure_binary, ensure_binary_gt

EPS = 1e-7


@dataclass(frozen=True)
class PairMetrics:
    iou0: float
    iou1: float
    iou: float
    ari: float
    used_inversion: bool


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + EPS))


def ari_score_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.reshape(-1).astype(np.int32)
    g = gt.reshape(-1).astype(np.int32)
    if np.unique(g).size == 1 and np.unique(p).size == 1:
        return 1.0
    return float(adjusted_rand_score(g, p))


def rwtd_invariant_metrics(pred: np.ndarray, gt: np.ndarray) -> PairMetrics:
    p = ensure_binary(pred)
    g = ensure_binary_gt(gt, strict=False)
    g_inv = 1 - g

    i0 = iou_score(p, g)
    i1 = iou_score(p, g_inv)
    use_inv = i1 > i0

    g_chosen = g_inv if use_inv else g
    ari = ari_score_binary(p, g_chosen)

    return PairMetrics(iou0=i0, iou1=i1, iou=max(i0, i1), ari=ari, used_inversion=use_inv)
