from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from .features import compute_texture_feature_map, region_descriptor
from .ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from .ptd_learned import PTDLearnedTrainConfig, _component_features, _compose_synthetic, _make_synthetic_proposals
from .ptd_v3 import PTDV3MergeScorer

EPS = 1e-8


@dataclass(frozen=True)
class PTDV4SetTrainConfig:
    ptd_root: Path
    ptd_encoder_ckpt: Path
    ptd_v3_bundle: Path
    out_bundle: Path
    out_metrics_json: Path

    num_samples: int = 1400
    val_fraction: float = 0.18
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


def _proposal_consensus_masks(proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
    if not proposals:
        return []
    stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
    freq = stack.mean(axis=0)
    out: list[np.ndarray] = []
    for t in (0.18, 0.26, 0.34, 0.42, 0.50, 0.60):
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


def _candidate_feature(
    *,
    mask: np.ndarray,
    feature_map: np.ndarray,
    grad: np.ndarray,
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
    consensus_masks: list[np.ndarray],
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

    extra = np.array(
        [area_ratio, border_ratio, n_cc, support_mean, support_max, cons_iou],
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
        n_rand = min(16, n * (n - 1) // 2 + n)
        for _ in range(n_rand):
            k = int(rng.integers(2, min(5, n) + 1))
            idx = rng.choice(n, size=k, replace=False).tolist()
            u = np.zeros_like(components[0], dtype=np.uint8)
            for i in idx:
                u = np.logical_or(u, components[i] > 0)
            cands.append(u.astype(np.uint8))

    # Include GT-shape perturbations as hard positives/near-positives.
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

    # Top-k unions by selector confidence.
    top_k = min(4, n)
    for k in range(2, top_k + 1):
        idx = [i for i in order[:k] if float(comp_probs[i]) >= selector_threshold]
        if len(idx) < 2:
            continue
        u = np.zeros_like(components[0], dtype=np.uint8)
        for i in idx:
            u = np.logical_or(u, components[i] > 0)
        cands.append(u.astype(np.uint8))

    # Greedy union with novelty constraint.
    selected: list[int] = []
    curr = np.zeros_like(components[0], dtype=np.uint8)
    min_new = max(24, int(round(0.008 * curr.size)))
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


def train_ptd_v4_set_models(cfg: PTDV4SetTrainConfig) -> dict[str, float | int]:
    rng = np.random.default_rng(cfg.random_seed)
    backend = PTDImageBackend(cfg.ptd_root)
    _, entries = load_ptd_entries(cfg.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=cfg.random_seed, root=cfg.ptd_root)
    class_to_entries = group_entries_by_class(split.train)

    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=cfg.ptd_encoder_ckpt, device="cuda"))
    v3 = PTDV3MergeScorer(cfg.ptd_v3_bundle)

    n_val = max(90, int(round(cfg.num_samples * cfg.val_fraction)))
    n_train = max(180, int(cfg.num_samples - n_val))
    target_total = n_train + n_val

    Xtr: list[np.ndarray] = []
    ytr: list[int] = []
    Xva: list[np.ndarray] = []
    yva: list[int] = []

    produced = 0
    attempts = 0
    while produced < target_total and attempts < target_total * 5:
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

        grad_map = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(grad_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grad_map, cv2.CV_32F, 0, 1, ksize=3)
        grad_map = np.sqrt(gx * gx + gy * gy)
        grad_map = grad_map / (float(grad_map.max()) + EPS)
        consensus = _proposal_consensus_masks(proposals, min_area=cfg.min_area)

        local_X: list[np.ndarray] = []
        local_y: list[int] = []
        for cm in candidates:
            iou = _iou(cm, gt)
            if iou >= 0.72:
                y = 1
            elif iou <= 0.32:
                y = 0
            else:
                continue
            x = _candidate_feature(
                mask=cm,
                feature_map=feat_map,
                grad=grad_map,
                proposals=proposals,
                descriptors=desc,
                consensus_masks=consensus,
            )
            local_X.append(x)
            local_y.append(y)

        if not local_X:
            continue

        is_val = produced >= n_train
        if is_val:
            Xva.extend(local_X)
            yva.extend(local_y)
        else:
            Xtr.extend(local_X)
            ytr.extend(local_y)
        produced += 1

        if produced % 50 == 0:
            print(
                f"[PTD-v4-set] generated={produced}/{target_total} "
                f"train_examples={len(Xtr)} val_examples={len(Xva)}"
            )

    if len(Xtr) < 300:
        raise RuntimeError(f"Insufficient PTD-v4 set examples: {len(Xtr)}")

    Xtr_arr = np.stack(Xtr, axis=0).astype(np.float32)
    ytr_arr = np.array(ytr, dtype=np.int32)
    Xva_arr = np.stack(Xva, axis=0).astype(np.float32) if Xva else np.zeros((0, Xtr_arr.shape[1]), dtype=np.float32)
    yva_arr = np.array(yva, dtype=np.int32)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=7,
        max_iter=420,
        min_samples_leaf=20,
        l2_regularization=2e-3,
        random_state=cfg.random_seed,
    )
    model.fit(Xtr_arr, ytr_arr)

    if len(Xva_arr) > 0 and len(np.unique(yva_arr)) > 1:
        p = model.predict_proba(Xva_arr)[:, 1]
        best_t = 0.62
        best_f1 = -1.0
        best_rec_at_prec = -1.0
        for t in np.linspace(0.30, 0.88, 59):
            yhat = (p >= t).astype(np.int32)
            tp = int(np.logical_and(yhat == 1, yva_arr == 1).sum())
            fp = int(np.logical_and(yhat == 1, yva_arr == 0).sum())
            fn = int(np.logical_and(yhat == 0, yva_arr == 1).sum())
            prec = float(tp / max(tp + fp, 1))
            rec = float(tp / max(tp + fn, 1))
            f1 = float(f1_score(yva_arr, yhat))
            if prec >= 0.90 and rec > best_rec_at_prec:
                best_rec_at_prec = rec
                best_t = float(t)
            if f1 > best_f1:
                best_f1 = f1
        if best_rec_at_prec < 0:
            best_t = float(max(best_t, 0.62))
        auc = float(roc_auc_score(yva_arr, p))
    else:
        best_t = 0.62
        best_f1 = 0.0
        auc = 0.0

    payload = {
        "feature_version": "ptd_learned_v4_set",
        "selector_model": model,
        "selector_threshold": float(best_t),
        "min_area": int(cfg.min_area),
        "v3_bundle_path": str(cfg.ptd_v3_bundle),
    }
    cfg.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_bundle.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "num_samples_generated": int(produced),
        "train_examples": int(len(Xtr)),
        "val_examples": int(len(Xva)),
        "val_auc": float(auc),
        "val_f1_best": float(best_f1),
        "selector_threshold": float(best_t),
    }
    cfg.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class PTDV4SetScorer:
    def __init__(self, bundle_path: Path):
        with Path(bundle_path).open("rb") as f:
            payload = pickle.load(f)
        self.selector_model = payload["selector_model"]
        self.selector_threshold = float(payload.get("selector_threshold", 0.50))
        self.min_area = int(payload.get("min_area", 24))
        self.v3 = PTDV3MergeScorer(Path(payload["v3_bundle_path"]))

    def _grad_map(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        return g / (float(g.max()) + EPS)

    def merge_components(
        self,
        *,
        image_rgb: np.ndarray,
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> tuple[list[np.ndarray], list[list[int]], list]:
        base_components, base_indices, decisions = self.v3.merge_components(
            image_rgb=image_rgb,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )
        if not base_components:
            return [], [], decisions

        grad = self._grad_map(image_rgb)
        consensus = _proposal_consensus_masks(proposals, min_area=self.min_area)
        Xc = []
        for cm in base_components:
            Xc.append(
                _candidate_feature(
                    mask=cm,
                    feature_map=feature_map,
                    grad=grad,
                    proposals=proposals,
                    descriptors=descriptors,
                    consensus_masks=consensus,
                )
            )
        probs = self.selector_model.predict_proba(np.stack(Xc, axis=0))[:, 1].astype(np.float32)

        set_cands = _build_inference_set_candidates(
            components=base_components,
            comp_probs=probs,
            min_area=self.min_area,
            selector_threshold=self.selector_threshold,
        )

        all_cands: list[np.ndarray] = []
        all_cands.extend([(c > 0).astype(np.uint8) for c in base_components])
        all_cands.extend(set_cands)

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
        grad = self._grad_map(image_rgb)
        consensus = _proposal_consensus_masks(proposals, min_area=self.min_area)

        feats: list[np.ndarray] = []
        base_feats: list[np.ndarray] = []
        for cm in components:
            f = _candidate_feature(
                mask=cm,
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
                consensus_masks=consensus,
            )
            feats.append(f)
            base_feats.append(
                _component_features(
                    comp_mask=(cm > 0).astype(np.uint8),
                    feature_map=feature_map,
                    grad=grad,
                    proposals=proposals,
                    descriptors=descriptors,
                ).astype(np.float32)
            )

        arr = np.stack(feats, axis=0).astype(np.float32)
        parr = self.selector_model.predict_proba(arr)[:, 1].astype(np.float32)
        brr = np.stack(base_feats, axis=0).astype(np.float32)
        v3_pred = self.v3.score_model.predict(brr).astype(np.float32)

        out: list[float] = []
        for i in range(arr.shape[0]):
            p = float(parr[i])
            base = float(v3_pred[i])
            area_ratio = float(arr[i][10])
            border_ratio = float(arr[i][11])
            ncc_norm = float(arr[i][12])
            support_mean = float(arr[i][13])
            support_max = float(arr[i][14])
            cons_iou = float(arr[i][15])

            border_pen = 0.0
            if area_ratio >= 0.36:
                border_pen = 0.22 * border_ratio
            if area_ratio >= 0.60:
                border_pen += 0.18 * border_ratio

            area_pen = 0.0
            if area_ratio >= 0.72:
                area_pen = 0.45 * (area_ratio - 0.72)
            frag_pen = 0.12 * max(0.0, ncc_norm - 0.25)
            low_prob_pen = 0.40 * max(0.0, self.selector_threshold - p)

            score = (
                0.84 * base
                + 0.16 * _logit(p)
                + 0.04 * support_mean
                + 0.03 * support_max
                + 0.04 * cons_iou
                - border_pen
                - area_pen
                - frag_pen
                - low_prob_pen
            )
            out.append(float(score))
        return out
