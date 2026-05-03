from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from .features import compute_texture_feature_map, cosine_similarity, region_descriptor
from .ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from .ptd_learned import (
    PTDLearnedTrainConfig,
    _component_features,
    _fragment_mask,
    _texture_canvas,
    _voronoi_labels,
)
from .ptd_v3 import PTDV3MergeScorer

EPS = 1e-8


@dataclass(frozen=True)
class PTDV8PartitionTrainConfig:
    ptd_root: Path
    ptd_encoder_ckpt: Path
    ptd_v3_bundle: Path
    out_bundle: Path
    out_metrics_json: Path

    num_samples: int = 2400
    val_fraction: float = 0.20
    image_size: int = 256
    min_regions: int = 3
    max_regions: int = 6
    min_fg_frags: int = 4
    max_fg_frags: int = 10
    random_seed: int = 1337
    min_area: int = 24
    max_class_pool: int = 4
    multi_target_prob: float = 0.70
    max_proposals_per_sample: int = 18
    max_candidates_per_sample: int = 56


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    uni = float(np.logical_or(aa, bb).sum())
    return 1.0 if uni <= 0 else inter / uni


def _logit(p: float) -> float:
    q = float(np.clip(p, 1e-5, 1.0 - 1e-5))
    return float(math.log(q / (1.0 - q)))


def _grad_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    return g / (float(g.max()) + EPS)


def _dedupe_masks(masks: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for m in masks:
        b = (m > 0).astype(np.uint8)
        if int(b.sum()) < min_area:
            continue
        k = np.packbits(b, axis=None).tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out


def _proposal_union(proposals: list[np.ndarray]) -> np.ndarray:
    if not proposals:
        return np.zeros((1, 1), dtype=np.uint8)
    u = np.zeros_like(proposals[0], dtype=np.uint8)
    for p in proposals:
        u = np.logical_or(u > 0, p > 0)
    return u.astype(np.uint8)


def _proposal_consensus_masks(proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    if not proposals:
        return []
    stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
    freq = stack.mean(axis=0)
    out: list[np.ndarray] = []
    for t in (0.12, 0.20, 0.28, 0.36, 0.46, 0.58, 0.72):
        m = (freq >= t).astype(np.uint8)
        if int(m.sum()) >= min_area:
            out.append(m)
    return _dedupe_masks(out, min_area=min_area)


def _border_touch_ratio(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area <= 0:
        return 1.0
    border = np.zeros_like(m, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touch = int(np.logical_and(m > 0, border).sum())
    return float(touch / max(area, 1))


def _num_components(mask: np.ndarray) -> int:
    n, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return int(max(n - 1, 0))


def _hole_ratio(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area <= 0:
        return 1.0
    inv = 1 - m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return 0.0
    h, w = m.shape
    border = set(np.unique(lbl[0, :]).tolist())
    border.update(np.unique(lbl[h - 1, :]).tolist())
    border.update(np.unique(lbl[:, 0]).tolist())
    border.update(np.unique(lbl[:, w - 1]).tolist())

    holes = 0
    for c in range(1, n):
        if c in border:
            continue
        holes += int(stats[c, cv2.CC_STAT_AREA])
    return float(holes / max(area, 1))


def _proposal_support(mask: np.ndarray, proposals: list[np.ndarray]) -> tuple[float, float]:
    m = mask > 0
    if int(m.sum()) <= 0 or not proposals:
        return 0.0, 0.0
    vals: list[float] = []
    for pm in proposals:
        pp = pm > 0
        pa = int(pp.sum())
        if pa <= 0:
            continue
        inter = float(np.logical_and(pp, m).sum())
        vals.append(inter / max(pa, 1))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


def _consensus_best_iou(mask: np.ndarray, consensus_masks: list[np.ndarray]) -> float:
    if not consensus_masks:
        return 0.0
    return float(max(_iou(mask, c) for c in consensus_masks))


def _coverage_vs_union(mask: np.ndarray, union: np.ndarray) -> tuple[float, float, float]:
    m = mask > 0
    u = union > 0
    ma = int(m.sum())
    ua = int(u.sum())
    if ma <= 0 or ua <= 0:
        return 0.0, 0.0, 0.0
    inter = float(np.logical_and(m, u).sum())
    cov = float(inter / max(ua, 1))
    prec = float(inter / max(ma, 1))
    area_vs_union = float(ma / max(ua, 1))
    return cov, prec, area_vs_union


def _component_descriptors(
    *,
    components: list[np.ndarray],
    proposals: list[np.ndarray],
    proposal_desc: list[np.ndarray],
) -> list[np.ndarray]:
    if not components:
        return []
    out: list[np.ndarray] = []
    for cm in components:
        m = cm > 0
        if int(m.sum()) <= 0:
            out.append(np.zeros((proposal_desc[0].shape[0],), dtype=np.float32))
            continue

        idx: list[int] = []
        ov_best = -1.0
        best_i = 0
        for i, pm in enumerate(proposals):
            pp = pm > 0
            pa = int(pp.sum())
            if pa <= 0:
                continue
            ov = float(np.logical_and(pp, m).sum() / max(pa, 1))
            if ov >= 0.50:
                idx.append(i)
            if ov > ov_best:
                ov_best = ov
                best_i = i

        if not idx:
            idx = [best_i]
        d = np.stack([proposal_desc[i] for i in idx], axis=0).mean(axis=0)
        out.append(d.astype(np.float32))
    return out


def _component_centroids(components: list[np.ndarray]) -> list[tuple[float, float]]:
    c: list[tuple[float, float]] = []
    for m in components:
        yy, xx = np.where(m > 0)
        if len(yy) == 0:
            c.append((0.5, 0.5))
            continue
        h, w = m.shape
        c.append((float(yy.mean() / max(h - 1, 1)), float(xx.mean() / max(w - 1, 1))))
    return c


def _selected_component_stats(
    *,
    mask: np.ndarray,
    components: list[np.ndarray],
    comp_desc: list[np.ndarray],
) -> tuple[float, float, float, float, float]:
    if not components:
        return 0.0, 1.0, 0.0, 0.0, 0.0

    sel: list[int] = []
    m = mask > 0
    for i, cm in enumerate(components):
        c = cm > 0
        ca = int(c.sum())
        if ca <= 0:
            continue
        ov = float(np.logical_and(c, m).sum() / max(ca, 1))
        if ov >= 0.50:
            sel.append(i)

    if not sel:
        best_i = 0
        best_ov = -1
        for i, cm in enumerate(components):
            ov = int(np.logical_and(cm > 0, m).sum())
            if ov > best_ov:
                best_ov = ov
                best_i = i
        sel = [best_i]

    all_desc = np.stack(comp_desc, axis=0).astype(np.float32)
    sel_desc = np.stack([comp_desc[i] for i in sel], axis=0).astype(np.float32)

    comp_count_norm = float(min(len(sel), 12) / 12.0)
    desc_var = float(np.var(sel_desc, axis=0).mean()) if len(sel) > 1 else 0.0

    others = [i for i in range(len(components)) if i not in sel]
    if others:
        out_desc = np.stack([comp_desc[i] for i in others], axis=0).astype(np.float32)
        desc_sep = float(np.linalg.norm(sel_desc.mean(axis=0) - out_desc.mean(axis=0)))
    else:
        desc_sep = float(np.linalg.norm(sel_desc.mean(axis=0)))

    if len(sel) >= 2:
        sims: list[float] = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                sims.append(float(cosine_similarity(sel_desc[i], sel_desc[j])))
        desc_sim = float(np.mean(sims)) if sims else 0.0
    else:
        desc_sim = 1.0

    centers = _component_centroids(components)
    if len(sel) >= 2:
        d: list[float] = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                yi, xi = centers[sel[i]]
                yj, xj = centers[sel[j]]
                d.append(float(math.sqrt((yi - yj) ** 2 + (xi - xj) ** 2)))
        spread = float(np.mean(d)) if d else 0.0
    else:
        spread = 0.0

    return comp_count_norm, desc_var, desc_sep, desc_sim, spread


def _compose_synthetic_target_union(
    *,
    backend: PTDImageBackend,
    class_to_entries: dict[int, list],
    rng: np.random.Generator,
    cfg: PTDV8PartitionTrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    h = int(cfg.image_size)
    w = int(cfg.image_size)

    n_regions = int(rng.integers(cfg.min_regions, cfg.max_regions + 1))
    labels = _voronoi_labels(h, w, n_regions, rng)

    cls_ids = list(class_to_entries.keys())
    pool_k = int(min(max(2, n_regions // 2 + 1), cfg.max_class_pool, max(1, len(cls_ids))))
    pool = rng.choice(cls_ids, size=pool_k, replace=False if len(cls_ids) >= pool_k else True).tolist()

    region_classes: list[int] = []
    for _ in range(n_regions):
        region_classes.append(int(pool[int(rng.integers(0, len(pool)))]))

    uniq = sorted(list(set(region_classes)))
    if len(uniq) < 2:
        # force at least two classes
        alt = int(cls_ids[int(rng.integers(0, len(cls_ids)))])
        region_classes[0] = alt
        uniq = sorted(list(set(region_classes)))

    # Often force target class to appear in multiple disjoint regions.
    target_class = int(uniq[int(rng.integers(0, len(uniq)))])
    if rng.random() < float(cfg.multi_target_prob):
        counts = {c: region_classes.count(c) for c in uniq}
        multi = [c for c, cnt in counts.items() if cnt >= 2]
        if multi:
            target_class = int(multi[int(rng.integers(0, len(multi)))])

    image = np.zeros((h, w, 3), dtype=np.uint8)
    for ridx, cid in enumerate(region_classes, start=1):
        entries = class_to_entries[int(cid)]
        e = entries[int(rng.integers(0, len(entries)))]
        tex = backend.read_rgb(e.rel_path)
        can = _texture_canvas(tex, h=h, w=w, rng=rng)
        m = labels == ridx
        image[m] = can[m]

    target_regions = [ridx + 1 for ridx, c in enumerate(region_classes) if int(c) == int(target_class)]
    gt = np.isin(labels, np.array(target_regions, dtype=np.int32)).astype(np.uint8)
    return image, gt, labels.astype(np.int32), region_classes, int(target_class)


def _make_synthetic_proposals_union(
    *,
    gt_union: np.ndarray,
    labels: np.ndarray,
    region_classes: list[int],
    target_class: int,
    rng: np.random.Generator,
    cfg: PTDV8PartitionTrainConfig,
) -> list[np.ndarray]:
    proposals: list[np.ndarray] = []
    h, w = gt_union.shape

    target_masks: list[np.ndarray] = []
    for ridx, cid in enumerate(region_classes, start=1):
        if int(cid) != int(target_class):
            continue
        m = (labels == ridx).astype(np.uint8)
        if int(m.sum()) < cfg.min_area:
            continue
        target_masks.append(m)

    for tm in target_masks:
        proposals.extend(_fragment_mask(tm, rng, cfg.min_fg_frags, cfg.max_fg_frags))
        if rng.random() < 0.55:
            proposals.append(tm.astype(np.uint8))

    if target_masks:
        u = np.zeros_like(target_masks[0], dtype=np.uint8)
        for tm in target_masks:
            u = np.logical_or(u > 0, tm > 0)
        proposals.append(u.astype(np.uint8))

    # Background distractors.
    other = [ridx for ridx, cid in enumerate(region_classes, start=1) if int(cid) != int(target_class)]
    rng.shuffle(other)
    if other:
        n_take = int(rng.integers(1, min(5, len(other) + 1)))
        for rid in other[:n_take]:
            m = (labels == rid).astype(np.uint8)
            if rng.random() < 0.80:
                ksz = int(rng.integers(3, 13))
                if ksz % 2 == 0:
                    ksz += 1
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                if rng.random() < 0.50:
                    m = cv2.erode(m, k, iterations=1)
                else:
                    m = cv2.dilate(m, k, iterations=1)
            proposals.append(m.astype(np.uint8))

    # Mixed distractors combining target and non-target parts.
    if target_masks and other:
        for _ in range(int(rng.integers(2, 6))):
            rid_bg = int(other[int(rng.integers(0, len(other)))])
            m = np.logical_or(
                target_masks[int(rng.integers(0, len(target_masks)))] > 0,
                labels == rid_bg,
            ).astype(np.uint8)
            if rng.random() < 0.6:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            proposals.append(m)

    # Random geometric negatives.
    for _ in range(int(rng.integers(2, 6))):
        m = np.zeros((h, w), dtype=np.uint8)
        x0 = int(rng.integers(0, max(1, w - 24)))
        y0 = int(rng.integers(0, max(1, h - 24)))
        ww = int(rng.integers(18, max(19, min(100, w - x0))))
        hh = int(rng.integers(18, max(19, min(100, h - y0))))
        m[y0 : y0 + hh, x0 : x0 + ww] = 1
        proposals.append(m)

    if rng.random() < 0.35:
        proposals.append(gt_union.astype(np.uint8))

    dedup = _dedupe_masks(proposals, min_area=cfg.min_area)
    if len(dedup) <= cfg.max_proposals_per_sample:
        return dedup

    # Keep larger proposals first; this keeps recall while reducing encoder cost.
    dedup = sorted(dedup, key=lambda m: int((m > 0).sum()), reverse=True)
    return dedup[: int(cfg.max_proposals_per_sample)]


def _build_partition_candidates(
    *,
    components: list[np.ndarray],
    component_desc: list[np.ndarray],
    proposals: list[np.ndarray],
    min_area: int,
    component_scores: np.ndarray | None,
    rng: np.random.Generator | None,
) -> list[np.ndarray]:
    if not components:
        return []

    cands: list[np.ndarray] = []
    cands.extend([(c > 0).astype(np.uint8) for c in components])

    # Add consensus and proposal-union masks.
    cands.extend(_proposal_consensus_masks(proposals, min_area=min_area))
    if proposals:
        cands.append(_proposal_union(proposals).astype(np.uint8))

    n = len(components)
    if component_scores is None or len(component_scores) != n:
        order = np.argsort([int((c > 0).sum()) for c in components])[::-1].tolist()
    else:
        order = np.argsort(component_scores)[::-1].tolist()

    # Top-k unions by component confidence.
    for k in range(2, min(7, n + 1)):
        idx = order[:k]
        u = np.zeros_like(components[0], dtype=np.uint8)
        for i in idx:
            u = np.logical_or(u > 0, components[i] > 0)
        cands.append(u.astype(np.uint8))

    # Similarity-based unions from anchors.
    dmat = None
    if n >= 2 and component_desc:
        D = np.stack(component_desc, axis=0).astype(np.float32)
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
        dmat = (Dn @ Dn.T).astype(np.float32)

        for anchor in order[: min(5, n)]:
            for thr in (0.52, 0.62, 0.72):
                idx = [j for j in range(n) if float(dmat[anchor, j]) >= thr]
                if len(idx) < 2:
                    continue
                u = np.zeros_like(components[0], dtype=np.uint8)
                for j in idx:
                    u = np.logical_or(u > 0, components[j] > 0)
                cands.append(u.astype(np.uint8))

        # KMeans cluster unions.
        kmax = min(5, n)
        for k in range(2, kmax + 1):
            try:
                km = KMeans(n_clusters=k, random_state=0, n_init=4)
                lbl = km.fit_predict(D)
            except Exception:
                continue
            for cid in range(k):
                idx = np.where(lbl == cid)[0].tolist()
                if len(idx) < 1:
                    continue
                u = np.zeros_like(components[0], dtype=np.uint8)
                for j in idx:
                    u = np.logical_or(u > 0, components[j] > 0)
                cands.append(u.astype(np.uint8))

    # Randomized unions for training-time candidate diversity.
    if rng is not None and n >= 2:
        n_rand = min(28, n * 5)
        weights = np.ones((n,), dtype=np.float32)
        if component_scores is not None and len(component_scores) == n:
            s = np.asarray(component_scores, dtype=np.float32)
            s = s - float(s.min())
            weights = s + 1e-3
        weights = weights / float(weights.sum() + EPS)

        for _ in range(n_rand):
            k = int(rng.integers(2, min(7, n) + 1))
            idx = rng.choice(n, size=k, replace=False, p=weights).tolist()
            u = np.zeros_like(components[0], dtype=np.uint8)
            for j in idx:
                u = np.logical_or(u > 0, components[j] > 0)
            cands.append(u.astype(np.uint8))

        # Similarity-driven random expansions from random seeds.
        if dmat is not None:
            for _ in range(min(12, n * 2)):
                seed = int(rng.integers(0, n))
                thr = float(rng.uniform(0.48, 0.75))
                idx = [j for j in range(n) if float(dmat[seed, j]) >= thr]
                if len(idx) < 2:
                    continue
                u = np.zeros_like(components[0], dtype=np.uint8)
                for j in idx:
                    u = np.logical_or(u > 0, components[j] > 0)
                cands.append(u.astype(np.uint8))

    dedup = _dedupe_masks(cands, min_area=min_area)
    if len(dedup) <= 56:
        return dedup
    dedup = sorted(dedup, key=lambda m: int((m > 0).sum()), reverse=True)
    return dedup[:56]


def _candidate_feature(
    *,
    mask: np.ndarray,
    atom_components: list[np.ndarray],
    atom_component_desc: list[np.ndarray],
    feature_map: np.ndarray,
    grad: np.ndarray,
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
    consensus_masks: list[np.ndarray],
    proposal_union: np.ndarray,
    v3_score: float,
) -> np.ndarray:
    base = _component_features(
        comp_mask=(mask > 0).astype(np.uint8),
        feature_map=feature_map,
        grad=grad,
        proposals=proposals,
        descriptors=descriptors,
    ).astype(np.float32)

    m = (mask > 0).astype(np.uint8)
    area_ratio = float(m.mean())
    border_ratio = _border_touch_ratio(m)
    n_cc = float(min(_num_components(m), 12) / 12.0)
    holes = _hole_ratio(m)
    support_mean, support_max = _proposal_support(m, proposals)
    cons_iou = _consensus_best_iou(m, consensus_masks)
    union_cov, union_prec, area_vs_union = _coverage_vs_union(m, proposal_union)

    comp_count_norm, desc_var, desc_sep, desc_sim, spread = _selected_component_stats(
        mask=m,
        components=atom_components,
        comp_desc=atom_component_desc,
    )

    extra = np.array(
        [
            area_ratio,
            border_ratio,
            n_cc,
            holes,
            support_mean,
            support_max,
            cons_iou,
            union_cov,
            union_prec,
            area_vs_union,
            comp_count_norm,
            desc_var,
            desc_sep,
            desc_sim,
            spread,
            float(v3_score),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra], axis=0).astype(np.float32)


# Extra feature indices (relative to candidate feature tail).
IDX_AREA_RATIO = -16
IDX_BORDER_RATIO = -15
IDX_NCC_NORM = -14
IDX_HOLE_RATIO = -13
IDX_SUPPORT_MEAN = -12
IDX_SUPPORT_MAX = -11
IDX_CONS_IOU = -10
IDX_UNION_COV = -9
IDX_UNION_PREC = -8
IDX_AREA_VS_UNION = -7
IDX_COMP_COUNT = -6
IDX_DESC_VAR = -5
IDX_DESC_SEP = -4
IDX_DESC_SIM = -3
IDX_SPREAD = -2
IDX_V3_SCORE = -1


def _candidate_scalar_score(
    *,
    reg_pred: float,
    cls_prob: float,
    feat: np.ndarray,
    tau_cov: float,
    tau_area: float,
    tau_sep: float,
) -> float:
    area_ratio = float(feat[IDX_AREA_RATIO])
    border_ratio = float(feat[IDX_BORDER_RATIO])
    ncc_norm = float(feat[IDX_NCC_NORM])
    hole_ratio = float(feat[IDX_HOLE_RATIO])
    support_mean = float(feat[IDX_SUPPORT_MEAN])
    support_max = float(feat[IDX_SUPPORT_MAX])
    cons_iou = float(feat[IDX_CONS_IOU])
    union_cov = float(feat[IDX_UNION_COV])
    union_prec = float(feat[IDX_UNION_PREC])
    area_vs_union = float(feat[IDX_AREA_VS_UNION])
    comp_count = float(feat[IDX_COMP_COUNT])
    desc_var = float(feat[IDX_DESC_VAR])
    desc_sep = float(feat[IDX_DESC_SEP])
    desc_sim = float(feat[IDX_DESC_SIM])
    spread = float(feat[IDX_SPREAD])
    v3_score = float(feat[IDX_V3_SCORE])

    score = (
        1.40 * float(reg_pred)
        + 0.26 * _logit(float(cls_prob))
        + 0.18 * math.tanh(0.35 * v3_score)
        + 0.10 * support_mean
        + 0.08 * support_max
        + 0.10 * cons_iou
        + 0.20 * union_cov
        + 0.10 * area_vs_union
        + 0.10 * desc_sep
        + 0.06 * desc_sim
        + 0.03 * comp_count
        + 0.03 * spread
        - 0.06 * border_ratio
        - 0.04 * max(0.0, ncc_norm - 0.65)
        - 0.08 * hole_ratio
        - 0.05 * desc_var
    )

    # Robustness constraints to avoid tiny-mask collapse and very low-coverage picks.
    if union_cov < tau_cov and area_vs_union < tau_area:
        score -= 0.95
    if area_vs_union < 0.025:
        score -= 1.20
    if union_prec < 0.10 and area_vs_union > 0.90:
        score -= 0.35
    if area_vs_union > 1.85:
        score -= 0.24 * (area_vs_union - 1.85)

    if desc_sep < tau_sep and union_cov > 0.40:
        score -= 0.28 * (tau_sep - desc_sep)

    # Keep small-region capability, but suppress near-empty masks.
    if area_ratio < 0.006:
        score -= 0.45

    return float(score)


def _choose_candidate_idx(
    *,
    reg_pred: np.ndarray,
    cls_prob: np.ndarray,
    feats: np.ndarray,
    tau_cov: float,
    tau_area: float,
    tau_sep: float,
) -> int:
    best_i = 0
    best_s = -1e12
    for i in range(feats.shape[0]):
        s = _candidate_scalar_score(
            reg_pred=float(reg_pred[i]),
            cls_prob=float(cls_prob[i]),
            feat=feats[i],
            tau_cov=tau_cov,
            tau_area=tau_area,
            tau_sep=tau_sep,
        )
        if s > best_s:
            best_s = s
            best_i = i
    return int(best_i)


def train_ptd_v8_partition_models(cfg: PTDV8PartitionTrainConfig) -> dict[str, float | int]:
    rng = np.random.default_rng(cfg.random_seed)
    backend = PTDImageBackend(cfg.ptd_root)
    _, entries = load_ptd_entries(cfg.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=cfg.random_seed, root=cfg.ptd_root)
    class_to_entries = group_entries_by_class(split.train)

    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=cfg.ptd_encoder_ckpt, device="cuda"))
    v3 = PTDV3MergeScorer(cfg.ptd_v3_bundle)

    n_val = max(40, int(round(cfg.num_samples * cfg.val_fraction)))
    n_train = max(120, int(cfg.num_samples - n_val))
    target_total = n_train + n_val

    X_cls_tr: list[np.ndarray] = []
    y_cls_tr: list[int] = []
    X_cls_va: list[np.ndarray] = []
    y_cls_va: list[int] = []

    X_reg_tr: list[np.ndarray] = []
    y_reg_tr: list[float] = []
    X_reg_va: list[np.ndarray] = []
    y_reg_va: list[float] = []

    # Candidate-group validation for threshold tuning.
    val_groups: list[tuple[np.ndarray, np.ndarray]] = []

    produced = 0
    attempts = 0
    while produced < target_total and attempts < target_total * 6:
        attempts += 1

        image, gt, labels, region_classes, target_class = _compose_synthetic_target_union(
            backend=backend,
            class_to_entries=class_to_entries,
            rng=rng,
            cfg=cfg,
        )
        if int(gt.sum()) < cfg.min_area:
            continue

        proposals = _make_synthetic_proposals_union(
            gt_union=gt,
            labels=labels,
            region_classes=region_classes,
            target_class=target_class,
            rng=rng,
            cfg=cfg,
        )
        if len(proposals) < 2:
            continue

        feat_map = compute_texture_feature_map(image)
        emb = encoder.encode_regions(image, proposals)
        hand = [region_descriptor(feat_map, m) for m in proposals]
        desc = [np.concatenate([h, e], axis=0).astype(np.float32) for h, e in zip(hand, emb)]

        atom_components, _, _ = v3.merge_components(
            image_rgb=image,
            proposals=proposals,
            descriptors=desc,
            feature_map=feat_map,
        )
        if not atom_components:
            continue

        if len(atom_components) > 14:
            atom_components = sorted(atom_components, key=lambda m: int((m > 0).sum()), reverse=True)[:14]

        grad = _grad_map(image)
        consensus = _proposal_consensus_masks(proposals, min_area=cfg.min_area)
        union = _proposal_union(proposals)
        atom_desc = _component_descriptors(
            components=atom_components,
            proposals=proposals,
            proposal_desc=desc,
        )

        atom_scores: list[float] = []
        for am in atom_components:
            bf = _component_features(
                comp_mask=(am > 0).astype(np.uint8),
                feature_map=feat_map,
                grad=grad,
                proposals=proposals,
                descriptors=desc,
            ).astype(np.float32)
            atom_scores.append(float(v3.score_model.predict(bf[None, :])[0]))

        candidates = _build_partition_candidates(
            components=atom_components,
            component_desc=atom_desc,
            proposals=proposals,
            min_area=cfg.min_area,
            component_scores=np.array(atom_scores, dtype=np.float32),
            rng=rng,
        )

        # Add GT-shape perturbations as positives / near-positives.
        gt_u8 = (gt > 0).astype(np.uint8)
        candidates.append(gt_u8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        candidates.append(cv2.erode(gt_u8, k, iterations=1))
        candidates.append(cv2.dilate(gt_u8, k, iterations=1))
        candidates = _dedupe_masks(candidates, min_area=cfg.min_area)
        if len(candidates) > cfg.max_candidates_per_sample:
            c_sorted = sorted(candidates, key=lambda m: int((m > 0).sum()), reverse=True)
            candidates = c_sorted[: int(cfg.max_candidates_per_sample)]
        if not candidates:
            continue

        local_feats: list[np.ndarray] = []
        local_iou: list[float] = []

        for cm in candidates:
            bf = _component_features(
                comp_mask=(cm > 0).astype(np.uint8),
                feature_map=feat_map,
                grad=grad,
                proposals=proposals,
                descriptors=desc,
            ).astype(np.float32)
            v3_score = float(v3.score_model.predict(bf[None, :])[0])
            f = _candidate_feature(
                mask=cm,
                atom_components=[(c > 0).astype(np.uint8) for c in atom_components],
                atom_component_desc=atom_desc,
                feature_map=feat_map,
                grad=grad,
                proposals=proposals,
                descriptors=desc,
                consensus_masks=consensus,
                proposal_union=union,
                v3_score=v3_score,
            )
            iou = _iou(cm, gt)

            local_feats.append(f)
            local_iou.append(iou)

            if iou >= 0.70:
                y_cls = 1
            elif iou <= 0.28:
                y_cls = 0
            else:
                y_cls = None
            if y_cls is not None:
                if produced >= n_train:
                    X_cls_va.append(f)
                    y_cls_va.append(y_cls)
                else:
                    X_cls_tr.append(f)
                    y_cls_tr.append(y_cls)

            if produced >= n_train:
                X_reg_va.append(f)
                y_reg_va.append(float(iou))
            else:
                X_reg_tr.append(f)
                y_reg_tr.append(float(iou))

        if not local_feats:
            continue

        if produced >= n_train:
            val_groups.append((np.stack(local_feats, axis=0).astype(np.float32), np.array(local_iou, dtype=np.float32)))

        produced += 1
        if produced % 60 == 0:
            print(
                f"[PTD-v8-partition] generated={produced}/{target_total} "
                f"cls_tr={len(X_cls_tr)} cls_va={len(X_cls_va)} reg_tr={len(X_reg_tr)} reg_va={len(X_reg_va)}"
            )

    if len(X_reg_tr) < 250 or len(X_cls_tr) < 120:
        raise RuntimeError(
            f"Insufficient PTD-v8 training data: reg_tr={len(X_reg_tr)} cls_tr={len(X_cls_tr)} generated={produced}"
        )

    Xc_tr = np.stack(X_cls_tr, axis=0).astype(np.float32)
    yc_tr = np.array(y_cls_tr, dtype=np.int32)
    Xc_va = np.stack(X_cls_va, axis=0).astype(np.float32) if X_cls_va else np.zeros((0, Xc_tr.shape[1]), dtype=np.float32)
    yc_va = np.array(y_cls_va, dtype=np.int32)

    cls_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=9,
        max_iter=520,
        min_samples_leaf=20,
        l2_regularization=2e-3,
        random_state=cfg.random_seed,
    )
    cls_model.fit(Xc_tr, yc_tr)

    if len(Xc_va) > 0 and len(np.unique(yc_va)) > 1:
        p = cls_model.predict_proba(Xc_va)[:, 1]
        best_t = 0.56
        best_f1 = -1.0
        for t in np.linspace(0.30, 0.88, 59):
            yhat = (p >= t).astype(np.int32)
            f1 = float(f1_score(yc_va, yhat))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        auc = float(roc_auc_score(yc_va, p))
    else:
        best_t = 0.56
        best_f1 = 0.0
        auc = 0.0

    Xr_tr = np.stack(X_reg_tr, axis=0).astype(np.float32)
    yr_tr = np.array(y_reg_tr, dtype=np.float32)
    Xr_va = np.stack(X_reg_va, axis=0).astype(np.float32) if X_reg_va else np.zeros((0, Xr_tr.shape[1]), dtype=np.float32)
    yr_va = np.array(y_reg_va, dtype=np.float32)

    reg_model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=11,
        max_iter=640,
        min_samples_leaf=24,
        l2_regularization=2e-3,
        random_state=cfg.random_seed,
    )
    reg_model.fit(Xr_tr, yr_tr)

    if len(Xr_va) > 0:
        reg_va_pred = reg_model.predict(Xr_va)
        reg_mae = float(mean_absolute_error(yr_va, reg_va_pred))
    else:
        reg_mae = 0.0

    # Tune score-threshold hyperparameters on PTD synthetic validation groups only.
    tau_cov_grid = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
    tau_area_grid = [0.03, 0.06, 0.10, 0.14, 0.18, 0.22]
    tau_sep_grid = [0.30, 0.45, 0.60, 0.75, 0.90]

    best_tau_cov = 0.16
    best_tau_area = 0.10
    best_tau_sep = 0.60
    best_sel_iou = -1.0

    if val_groups:
        for tc in tau_cov_grid:
            for ta in tau_area_grid:
                for ts in tau_sep_grid:
                    sel_ious: list[float] = []
                    for Xg, yg in val_groups:
                        cls_p = cls_model.predict_proba(Xg)[:, 1].astype(np.float32)
                        reg_p = reg_model.predict(Xg).astype(np.float32)
                        idx = _choose_candidate_idx(
                            reg_pred=reg_p,
                            cls_prob=cls_p,
                            feats=Xg,
                            tau_cov=float(tc),
                            tau_area=float(ta),
                            tau_sep=float(ts),
                        )
                        sel_ious.append(float(yg[idx]))
                    mi = float(np.mean(sel_ious)) if sel_ious else 0.0
                    if mi > best_sel_iou:
                        best_sel_iou = mi
                        best_tau_cov = float(tc)
                        best_tau_area = float(ta)
                        best_tau_sep = float(ts)

    payload = {
        "feature_version": "ptd_learned_v8_partition",
        "selector_model": cls_model,
        "regression_model": reg_model,
        "selector_threshold": float(best_t),
        "coverage_tau_cov": float(best_tau_cov),
        "coverage_tau_area": float(best_tau_area),
        "separation_tau": float(best_tau_sep),
        "min_area": int(cfg.min_area),
        "v3_bundle_path": str(cfg.ptd_v3_bundle),
    }
    cfg.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_bundle.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "num_samples_generated": int(produced),
        "train_cls_examples": int(len(X_cls_tr)),
        "val_cls_examples": int(len(X_cls_va)),
        "train_reg_examples": int(len(X_reg_tr)),
        "val_reg_examples": int(len(X_reg_va)),
        "val_groups": int(len(val_groups)),
        "val_auc_cls": float(auc),
        "val_f1_cls_best": float(best_f1),
        "val_mae_reg": float(reg_mae),
        "selector_threshold": float(best_t),
        "coverage_tau_cov": float(best_tau_cov),
        "coverage_tau_area": float(best_tau_area),
        "separation_tau": float(best_tau_sep),
        "val_mean_selected_iou": float(best_sel_iou),
    }
    cfg.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class PTDV8PartitionScorer:
    def __init__(self, bundle_path: Path):
        with Path(bundle_path).open("rb") as f:
            payload = pickle.load(f)
        self.selector_model = payload["selector_model"]
        self.regression_model = payload["regression_model"]
        self.selector_threshold = float(payload.get("selector_threshold", 0.56))
        self.coverage_tau_cov = float(payload.get("coverage_tau_cov", 0.16))
        self.coverage_tau_area = float(payload.get("coverage_tau_area", 0.10))
        self.separation_tau = float(payload.get("separation_tau", 0.60))
        self.min_area = int(payload.get("min_area", 24))
        self.v3 = PTDV3MergeScorer(Path(payload["v3_bundle_path"]))

        self._last_atom_components: list[np.ndarray] = []
        self._last_atom_desc: list[np.ndarray] = []

    def _features_for_masks(
        self,
        *,
        image_rgb: np.ndarray,
        masks: list[np.ndarray],
        atom_components: list[np.ndarray],
        atom_component_desc: list[np.ndarray],
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
        consensus: list[np.ndarray],
        proposal_union: np.ndarray,
    ) -> np.ndarray:
        grad = _grad_map(image_rgb)
        feats: list[np.ndarray] = []
        for cm in masks:
            base_feat = _component_features(
                comp_mask=(cm > 0).astype(np.uint8),
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
            ).astype(np.float32)
            v3_score = float(self.v3.score_model.predict(base_feat[None, :])[0])
            f = _candidate_feature(
                mask=cm,
                atom_components=atom_components,
                atom_component_desc=atom_component_desc,
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
                consensus_masks=consensus,
                proposal_union=proposal_union,
                v3_score=v3_score,
            )
            feats.append(f)
        return np.stack(feats, axis=0).astype(np.float32)

    def merge_components(
        self,
        *,
        image_rgb: np.ndarray,
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> tuple[list[np.ndarray], list[list[int]], list]:
        atom_components, _, decisions = self.v3.merge_components(
            image_rgb=image_rgb,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )
        if not atom_components:
            self._last_atom_components = []
            self._last_atom_desc = []
            return [], [], decisions

        atom_desc = _component_descriptors(
            components=atom_components,
            proposals=proposals,
            proposal_desc=descriptors,
        )

        grad = _grad_map(image_rgb)
        atom_scores: list[float] = []
        for am in atom_components:
            bf = _component_features(
                comp_mask=(am > 0).astype(np.uint8),
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
            ).astype(np.float32)
            atom_scores.append(float(self.v3.score_model.predict(bf[None, :])[0]))

        partition_cands = _build_partition_candidates(
            components=atom_components,
            component_desc=atom_desc,
            proposals=proposals,
            min_area=self.min_area,
            component_scores=np.array(atom_scores, dtype=np.float32),
            rng=None,
        )

        all_cands: list[np.ndarray] = []
        all_cands.extend([(c > 0).astype(np.uint8) for c in atom_components])
        all_cands.extend(partition_cands)

        dedup = _dedupe_masks(all_cands, min_area=self.min_area)
        idxs = [[i] for i in range(len(dedup))]

        self._last_atom_components = [(c > 0).astype(np.uint8) for c in atom_components]
        self._last_atom_desc = [d.astype(np.float32) for d in atom_desc]

        return dedup, idxs, decisions

    def score_components(
        self,
        *,
        image_rgb: np.ndarray,
        components: list[np.ndarray],
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> list[float]:
        if not components:
            return []

        consensus = _proposal_consensus_masks(proposals, min_area=self.min_area)
        proposal_union = _proposal_union(proposals)

        atom_components = self._last_atom_components
        atom_desc = self._last_atom_desc
        if not atom_components or len(atom_components) != len(atom_desc):
            atom_components = [(c > 0).astype(np.uint8) for c in components]
            atom_desc = _component_descriptors(
                components=atom_components,
                proposals=proposals,
                proposal_desc=descriptors,
            )

        X = self._features_for_masks(
            image_rgb=image_rgb,
            masks=components,
            atom_components=atom_components,
            atom_component_desc=atom_desc,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
            consensus=consensus,
            proposal_union=proposal_union,
        )

        cls_prob = self.selector_model.predict_proba(X)[:, 1].astype(np.float32)
        reg_pred = self.regression_model.predict(X).astype(np.float32)

        out: list[float] = []
        for i in range(X.shape[0]):
            s = _candidate_scalar_score(
                reg_pred=float(reg_pred[i]),
                cls_prob=float(cls_prob[i]),
                feat=X[i],
                tau_cov=self.coverage_tau_cov,
                tau_area=self.coverage_tau_area,
                tau_sep=self.separation_tau,
            )
            out.append(float(s))
        return out
