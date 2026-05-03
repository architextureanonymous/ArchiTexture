from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from .features import compute_texture_feature_map, region_descriptor, region_variance
from .ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from .ptd_learned import PTDLearnedTrainConfig, _component_features, _compose_synthetic, _make_synthetic_proposals
from .ptd_v3 import PTDV3MergeScorer

EPS = 1e-8


@dataclass(frozen=True)
class PTDV6CoverageTrainConfig:
    ptd_root: Path
    ptd_encoder_ckpt: Path
    ptd_v3_bundle: Path
    out_bundle: Path
    out_metrics_json: Path

    num_samples: int = 1800
    val_fraction: float = 0.2
    image_size: int = 256
    min_regions: int = 2
    max_regions: int = 4
    min_fg_frags: int = 4
    max_fg_frags: int = 9
    random_seed: int = 1337
    min_area: int = 24


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    uni = float(np.logical_or(aa, bb).sum())
    return 1.0 if uni <= 0 else inter / uni


def _logit(p: float) -> float:
    q = float(np.clip(p, 1e-5, 1.0 - 1e-5))
    return float(math.log(q / (1.0 - q)))


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
    for t in (0.16, 0.24, 0.32, 0.40, 0.50, 0.62, 0.74):
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


def _candidate_feature(
    *,
    mask: np.ndarray,
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
    n_cc = float(min(_num_components(m), 8) / 8.0)
    support_mean, support_max = _proposal_support(m, proposals)
    cons_iou = _consensus_best_iou(m, consensus_masks)
    union_cov, union_prec, area_vs_union = _coverage_vs_union(m, proposal_union)

    extra = np.array(
        [
            area_ratio,
            border_ratio,
            n_cc,
            support_mean,
            support_max,
            cons_iou,
            union_cov,
            union_prec,
            area_vs_union,
            float(v3_score),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra], axis=0).astype(np.float32)


def _build_training_candidates(
    *,
    components: list[np.ndarray],
    proposals: list[np.ndarray],
    gt: np.ndarray,
    rng: np.random.Generator,
    min_area: int,
) -> list[np.ndarray]:
    cands: list[np.ndarray] = []
    cands.extend([(c > 0).astype(np.uint8) for c in components])
    cands.extend(_proposal_consensus_masks(proposals, min_area=min_area))

    if proposals:
        pu = np.logical_or.reduce([(p > 0) for p in proposals]).astype(np.uint8)
        cands.append(pu)
    if components:
        cu = np.logical_or.reduce([(c > 0) for c in components]).astype(np.uint8)
        cands.append(cu)

    n = len(components)
    if n >= 2:
        n_rand = min(24, n * (n - 1) // 2 + n)
        for _ in range(n_rand):
            k = int(rng.integers(2, min(6, n) + 1))
            idx = rng.choice(n, size=k, replace=False).tolist()
            u = np.zeros_like(components[0], dtype=np.uint8)
            for i in idx:
                u = np.logical_or(u, components[i] > 0)
            cands.append(u.astype(np.uint8))

    if int(gt.sum()) >= min_area:
        cands.append((gt > 0).astype(np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cands.append(cv2.erode((gt > 0).astype(np.uint8), k, iterations=1))
        cands.append(cv2.dilate((gt > 0).astype(np.uint8), k, iterations=1))

    return _dedupe_masks(cands, min_area=min_area)


def _build_inference_set_candidates(
    *,
    components: list[np.ndarray],
    comp_probs: np.ndarray,
    min_area: int,
    selector_threshold: float,
) -> list[np.ndarray]:
    if not components:
        return []
    n = len(components)
    order = np.argsort(comp_probs)[::-1].tolist()

    cands: list[np.ndarray] = []
    cands.extend([(c > 0).astype(np.uint8) for c in components])

    top_k = min(6, n)
    for k in range(2, top_k + 1):
        idx = [i for i in order[:k] if float(comp_probs[i]) >= selector_threshold]
        if len(idx) < 2:
            continue
        u = np.zeros_like(components[0], dtype=np.uint8)
        for i in idx:
            u = np.logical_or(u, components[i] > 0)
        cands.append(u.astype(np.uint8))

    selected: list[int] = []
    curr = np.zeros_like(components[0], dtype=np.uint8)
    min_new = max(24, int(round(0.006 * curr.size)))
    for i in order:
        p = float(comp_probs[i])
        if not selected:
            selected.append(i)
            curr = (components[i] > 0).astype(np.uint8)
            continue
        if p < selector_threshold:
            continue
        cand = np.logical_or(curr > 0, components[i] > 0).astype(np.uint8)
        new_px = int(np.logical_and(cand > 0, curr == 0).sum())
        if new_px < min_new:
            continue
        selected.append(i)
        curr = cand
    if int(curr.sum()) >= min_area and len(selected) >= 2:
        cands.append(curr.astype(np.uint8))

    return _dedupe_masks(cands, min_area=min_area)


# Extra feature indices (relative to candidate feature tail).
IDX_AREA_RATIO = -10
IDX_BORDER_RATIO = -9
IDX_NCC_NORM = -8
IDX_SUPPORT_MEAN = -7
IDX_SUPPORT_MAX = -6
IDX_CONS_IOU = -5
IDX_UNION_COV = -4
IDX_UNION_PREC = -3
IDX_AREA_VS_UNION = -2
IDX_V3_SCORE = -1


def _candidate_scalar_score(
    *,
    reg_pred: float,
    cls_prob: float,
    feat: np.ndarray,
    tau_cov: float,
    tau_area: float,
) -> float:
    area_ratio = float(feat[IDX_AREA_RATIO])
    border_ratio = float(feat[IDX_BORDER_RATIO])
    ncc_norm = float(feat[IDX_NCC_NORM])
    support_mean = float(feat[IDX_SUPPORT_MEAN])
    support_max = float(feat[IDX_SUPPORT_MAX])
    cons_iou = float(feat[IDX_CONS_IOU])
    union_cov = float(feat[IDX_UNION_COV])
    union_prec = float(feat[IDX_UNION_PREC])
    area_vs_union = float(feat[IDX_AREA_VS_UNION])
    v3_score = float(feat[IDX_V3_SCORE])

    score = (
        1.35 * float(reg_pred)
        + 0.24 * _logit(float(cls_prob))
        + 0.22 * math.tanh(0.35 * v3_score)
        + 0.10 * support_mean
        + 0.08 * support_max
        + 0.08 * cons_iou
        + 0.18 * union_cov
        + 0.08 * area_vs_union
        - 0.08 * border_ratio
        - 0.10 * max(0.0, ncc_norm - 0.25)
    )

    # Robustness constraints (soft penalties) to avoid tiny-mask collapse.
    if union_cov < tau_cov and area_vs_union < tau_area:
        score -= 0.85
    if area_vs_union < 0.03:
        score -= 1.10
    if union_prec < 0.12 and area_vs_union > 0.85:
        score -= 0.35
    if area_vs_union > 1.70:
        score -= 0.18 * (area_vs_union - 1.70)

    # Mild area prior for natural images: discourage vanishing masks.
    if area_ratio < 0.010:
        score -= 0.35
    return float(score)


def _choose_candidate_idx(
    *,
    reg_pred: np.ndarray,
    cls_prob: np.ndarray,
    feats: np.ndarray,
    tau_cov: float,
    tau_area: float,
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
        )
        if s > best_s:
            best_s = s
            best_i = i
    return int(best_i)


def train_ptd_v6_coverage_models(cfg: PTDV6CoverageTrainConfig) -> dict[str, float | int]:
    rng = np.random.default_rng(cfg.random_seed)
    backend = PTDImageBackend(cfg.ptd_root)
    _, entries = load_ptd_entries(cfg.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=cfg.random_seed, root=cfg.ptd_root)
    class_to_entries = group_entries_by_class(split.train)

    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=cfg.ptd_encoder_ckpt, device="cuda"))
    v3 = PTDV3MergeScorer(cfg.ptd_v3_bundle)

    n_val = max(100, int(round(cfg.num_samples * cfg.val_fraction)))
    n_train = max(250, int(cfg.num_samples - n_val))
    target_total = n_train + n_val

    X_cls_tr: list[np.ndarray] = []
    y_cls_tr: list[int] = []
    X_cls_va: list[np.ndarray] = []
    y_cls_va: list[int] = []

    X_reg_tr: list[np.ndarray] = []
    y_reg_tr: list[float] = []
    X_reg_va: list[np.ndarray] = []
    y_reg_va: list[float] = []

    # For threshold tuning.
    val_groups: list[tuple[np.ndarray, np.ndarray]] = []

    produced = 0
    attempts = 0
    while produced < target_total and attempts < target_total * 6:
        attempts += 1
        image, gt, labels, fg_region = _compose_synthetic(
            backend=backend,
            class_to_entries=class_to_entries,
            rng=rng,
            image_size=cfg.image_size,
            min_regions=cfg.min_regions,
            max_regions=cfg.max_regions,
        )
        proposals = _make_synthetic_proposals(
            gt=gt,
            labels=labels,
            fg_region=fg_region,
            rng=rng,
            cfg=PTDLearnedTrainConfig(
                ptd_root=cfg.ptd_root,
                ptd_encoder_ckpt=cfg.ptd_encoder_ckpt,
                out_bundle=cfg.out_bundle,
                out_metrics_json=cfg.out_metrics_json,
                num_samples=cfg.num_samples,
                val_fraction=cfg.val_fraction,
                image_size=cfg.image_size,
                min_regions=cfg.min_regions,
                max_regions=cfg.max_regions,
                min_fg_frags=cfg.min_fg_frags,
                max_fg_frags=cfg.max_fg_frags,
                random_seed=cfg.random_seed,
                min_area=cfg.min_area,
            ),
        )
        if len(proposals) < 2:
            continue

        feat_map = compute_texture_feature_map(image)
        emb = encoder.encode_regions(image, proposals)
        hand = [region_descriptor(feat_map, m) for m in proposals]
        desc = [np.concatenate([h, e], axis=0).astype(np.float32) for h, e in zip(hand, emb)]

        components, _, _ = v3.merge_components(
            image_rgb=image,
            proposals=proposals,
            descriptors=desc,
            feature_map=feat_map,
        )
        if not components:
            continue

        candidates = _build_training_candidates(
            components=components,
            proposals=proposals,
            gt=(gt > 0).astype(np.uint8),
            rng=rng,
            min_area=cfg.min_area,
        )
        if not candidates:
            continue

        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / (float(grad.max()) + EPS)

        consensus = _proposal_consensus_masks(proposals, min_area=cfg.min_area)
        union = _proposal_union(proposals)

        local_feats: list[np.ndarray] = []
        local_iou: list[float] = []
        for cm in candidates:
            base_feat = _component_features(
                comp_mask=(cm > 0).astype(np.uint8),
                feature_map=feat_map,
                grad=grad,
                proposals=proposals,
                descriptors=desc,
            ).astype(np.float32)
            v3_score = float(v3.score_model.predict(base_feat[None, :])[0])
            f = _candidate_feature(
                mask=cm,
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

            if iou >= 0.72:
                y_cls = 1
            elif iou <= 0.30:
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
                f"[PTD-v6-cov] generated={produced}/{target_total} "
                f"cls_tr={len(X_cls_tr)} cls_va={len(X_cls_va)} reg_tr={len(X_reg_tr)} reg_va={len(X_reg_va)}"
            )

    if len(X_reg_tr) < 600 or len(X_cls_tr) < 300:
        raise RuntimeError(
            f"Insufficient PTD-v6 training data: reg_tr={len(X_reg_tr)} cls_tr={len(X_cls_tr)} generated={produced}"
        )

    Xc_tr = np.stack(X_cls_tr, axis=0).astype(np.float32)
    yc_tr = np.array(y_cls_tr, dtype=np.int32)
    Xc_va = np.stack(X_cls_va, axis=0).astype(np.float32) if X_cls_va else np.zeros((0, Xc_tr.shape[1]), dtype=np.float32)
    yc_va = np.array(y_cls_va, dtype=np.int32)

    cls_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=460,
        min_samples_leaf=20,
        l2_regularization=2e-3,
        random_state=cfg.random_seed,
    )
    cls_model.fit(Xc_tr, yc_tr)

    if len(Xc_va) > 0 and len(np.unique(yc_va)) > 1:
        p = cls_model.predict_proba(Xc_va)[:, 1]
        best_t = 0.58
        best_f1 = -1.0
        for t in np.linspace(0.35, 0.86, 52):
            yhat = (p >= t).astype(np.int32)
            f1 = float(f1_score(yc_va, yhat))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        auc = float(roc_auc_score(yc_va, p))
    else:
        best_t = 0.58
        best_f1 = 0.0
        auc = 0.0

    Xr_tr = np.stack(X_reg_tr, axis=0).astype(np.float32)
    yr_tr = np.array(y_reg_tr, dtype=np.float32)
    Xr_va = np.stack(X_reg_va, axis=0).astype(np.float32) if X_reg_va else np.zeros((0, Xr_tr.shape[1]), dtype=np.float32)
    yr_va = np.array(y_reg_va, dtype=np.float32)

    reg_model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=10,
        max_iter=560,
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

    # Tune coverage thresholds on PTD synthetic validation groups only.
    tau_cov_grid = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
    tau_area_grid = [0.04, 0.08, 0.12, 0.16, 0.20]

    best_tau_cov = 0.16
    best_tau_area = 0.10
    best_sel_iou = -1.0

    if val_groups:
        for tc in tau_cov_grid:
            for ta in tau_area_grid:
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
                    )
                    sel_ious.append(float(yg[idx]))
                mi = float(np.mean(sel_ious)) if sel_ious else 0.0
                if mi > best_sel_iou:
                    best_sel_iou = mi
                    best_tau_cov = float(tc)
                    best_tau_area = float(ta)

    payload = {
        "feature_version": "ptd_learned_v6_coverage",
        "selector_model": cls_model,
        "regression_model": reg_model,
        "selector_threshold": float(best_t),
        "coverage_tau_cov": float(best_tau_cov),
        "coverage_tau_area": float(best_tau_area),
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
        "val_mean_selected_iou": float(best_sel_iou),
    }
    cfg.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class PTDV6CoverageScorer:
    def __init__(self, bundle_path: Path):
        with Path(bundle_path).open("rb") as f:
            payload = pickle.load(f)
        self.selector_model = payload["selector_model"]
        self.regression_model = payload["regression_model"]
        self.selector_threshold = float(payload.get("selector_threshold", 0.58))
        self.coverage_tau_cov = float(payload.get("coverage_tau_cov", 0.16))
        self.coverage_tau_area = float(payload.get("coverage_tau_area", 0.10))
        self.min_area = int(payload.get("min_area", 24))
        self.v3 = PTDV3MergeScorer(Path(payload["v3_bundle_path"]))

    def _grad_map(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        return g / (float(g.max()) + EPS)

    def _features_for_masks(
        self,
        *,
        image_rgb: np.ndarray,
        masks: list[np.ndarray],
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
        consensus: list[np.ndarray],
        proposal_union: np.ndarray,
    ) -> np.ndarray:
        grad = self._grad_map(image_rgb)
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
        base_components, _, decisions = self.v3.merge_components(
            image_rgb=image_rgb,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )
        if not base_components:
            return [], [], decisions

        consensus = _proposal_consensus_masks(proposals, min_area=self.min_area)
        proposal_union = _proposal_union(proposals)

        Xc = self._features_for_masks(
            image_rgb=image_rgb,
            masks=[(c > 0).astype(np.uint8) for c in base_components],
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
            consensus=consensus,
            proposal_union=proposal_union,
        )
        probs = self.selector_model.predict_proba(Xc)[:, 1].astype(np.float32)

        set_cands = _build_inference_set_candidates(
            components=base_components,
            comp_probs=probs,
            min_area=self.min_area,
            selector_threshold=self.selector_threshold,
        )

        all_cands: list[np.ndarray] = []
        all_cands.extend([(c > 0).astype(np.uint8) for c in base_components])
        all_cands.extend(set_cands)
        all_cands.extend(consensus)
        if proposals:
            all_cands.append(proposal_union.astype(np.uint8))

        dedup = _dedupe_masks(all_cands, min_area=self.min_area)
        idxs = [[i] for i in range(len(dedup))]
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
        X = self._features_for_masks(
            image_rgb=image_rgb,
            masks=components,
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
            )
            out.append(float(s))
        return out
