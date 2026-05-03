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

from .features import cosine_similarity, region_variance
from .merge import EdgeDecision
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from .ptd_learned import (
    PTDLearnedTrainConfig,
    _component_features,
    _generate_sample,
    _grad_map,
    _pair_features,
    _UnionFind,
)
from .ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries

EPS = 1e-8


@dataclass(frozen=True)
class PTDV3TrainConfig:
    ptd_root: Path
    ptd_encoder_ckpt: Path
    out_bundle: Path
    out_metrics_json: Path

    # We intentionally train with more synthetic samples than v2 to improve robustness.
    num_samples: int = 2000
    val_fraction: float = 0.18
    image_size: int = 256
    min_regions: int = 2
    max_regions: int = 4
    min_fg_frags: int = 4
    max_fg_frags: int = 9
    random_seed: int = 1337
    min_area: int = 24
    adjacency_dilation: int = 3


@dataclass
class _ClusterState:
    members: list[int]
    mask: np.ndarray
    area: int
    var: float
    desc_mean: np.ndarray


def _logit(p: float) -> float:
    q = float(np.clip(p, 1e-5, 1.0 - 1e-5))
    return float(math.log(q / (1.0 - q)))


def train_ptd_v3_models(cfg: PTDV3TrainConfig) -> dict[str, float | int]:
    rng = np.random.default_rng(cfg.random_seed)
    backend = PTDImageBackend(cfg.ptd_root)
    _, entries = load_ptd_entries(cfg.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=cfg.random_seed, root=cfg.ptd_root)
    class_to_entries = group_entries_by_class(split.train)
    encoder = PTDTextureEncoder(PTDEncoderConfig(checkpoint=cfg.ptd_encoder_ckpt, device="cuda"))

    # Reuse PTD synthetic generator from v2, but train stronger gradient-boosting learners.
    n_val = max(80, int(round(cfg.num_samples * cfg.val_fraction)))
    n_train = max(200, int(cfg.num_samples - n_val))

    pair_X_tr: list[np.ndarray] = []
    pair_y_tr: list[int] = []
    pair_X_va: list[np.ndarray] = []
    pair_y_va: list[int] = []
    comp_X_tr: list[np.ndarray] = []
    comp_y_tr: list[float] = []
    comp_X_va: list[np.ndarray] = []
    comp_y_va: list[float] = []

    target_total = n_train + n_val
    produced = 0
    attempts = 0
    while produced < target_total and attempts < target_total * 4:
        attempts += 1
        sample = _generate_sample(
            backend=backend,
            class_to_entries=class_to_entries,
            encoder=encoder,
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
                adjacency_dilation=cfg.adjacency_dilation,
            ),
            rng=rng,
        )
        if sample is None:
            continue

        pX, py, cX, cy = sample
        # Hard-negative upweighting: pairs with moderate texture similarity but wrong label.
        pX_aug: list[np.ndarray] = []
        py_aug: list[int] = []
        for x, y in zip(pX, py):
            pX_aug.append(x)
            py_aug.append(y)
            tex = float(x[0])
            near = float(x[4])
            if y == 0 and tex > 0.58 and near > 0.05:
                pX_aug.append(x)
                py_aug.append(y)

        is_val = produced >= n_train
        if is_val:
            pair_X_va.extend(pX_aug)
            pair_y_va.extend(py_aug)
            comp_X_va.extend(cX)
            comp_y_va.extend(cy)
        else:
            pair_X_tr.extend(pX_aug)
            pair_y_tr.extend(py_aug)
            comp_X_tr.extend(cX)
            comp_y_tr.extend(cy)
        produced += 1

        if produced % 60 == 0:
            print(
                f"[PTD-v3] generated={produced}/{target_total} "
                f"pair_tr={len(pair_X_tr)} pair_va={len(pair_X_va)} "
                f"comp_tr={len(comp_X_tr)} comp_va={len(comp_X_va)}"
            )

    if len(pair_X_tr) < 300 or len(comp_X_tr) < 300:
        raise RuntimeError(
            "Insufficient PTD synthetic training data for PTD-v3 "
            f"(pair_tr={len(pair_X_tr)} comp_tr={len(comp_X_tr)})."
        )

    Xp_tr = np.stack(pair_X_tr, axis=0).astype(np.float32)
    yp_tr = np.array(pair_y_tr, dtype=np.int32)
    Xp_va = np.stack(pair_X_va, axis=0).astype(np.float32) if pair_X_va else np.zeros((0, Xp_tr.shape[1]), dtype=np.float32)
    yp_va = np.array(pair_y_va, dtype=np.int32)

    pair_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=10,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=cfg.random_seed,
    )
    pair_model.fit(Xp_tr, yp_tr)

    if len(Xp_va) > 0 and len(np.unique(yp_va)) > 1:
        p = pair_model.predict_proba(Xp_va)[:, 1]
        best_t = 0.50
        best_f1 = -1.0
        # Slightly high threshold to reduce false positive merges.
        for t in np.linspace(0.42, 0.86, 45):
            yhat = (p >= t).astype(np.int32)
            f1 = float(f1_score(yp_va, yhat))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        pair_auc = float(roc_auc_score(yp_va, p))
    else:
        best_t = 0.56
        best_f1 = 0.0
        pair_auc = 0.0

    Xc_tr = np.stack(comp_X_tr, axis=0).astype(np.float32)
    yc_tr = np.array(comp_y_tr, dtype=np.float32)
    Xc_va = np.stack(comp_X_va, axis=0).astype(np.float32) if comp_X_va else np.zeros((0, Xc_tr.shape[1]), dtype=np.float32)
    yc_va = np.array(comp_y_va, dtype=np.float32)

    score_model = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=11,
        max_iter=520,
        min_samples_leaf=24,
        l2_regularization=2e-3,
        random_state=cfg.random_seed,
    )
    score_model.fit(Xc_tr, yc_tr)

    if len(Xc_va) > 0:
        pred_va = score_model.predict(Xc_va)
        score_mae = float(mean_absolute_error(yc_va, pred_va))
    else:
        score_mae = 0.0

    payload = {
        "pair_model": pair_model,
        "score_model": score_model,
        "pair_threshold": float(best_t),
        "adjacency_dilation": int(cfg.adjacency_dilation),
        "min_area": int(cfg.min_area),
        "feature_version": "ptd_learned_v3_graph",
        "graph_merge": {
            "strong_prob": max(float(best_t), 0.62),
            "gain_threshold": 0.06,
            "neg_prob": 0.28,
            "absorb_threshold": 0.02,
        },
    }
    cfg.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_bundle.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "num_samples_generated": int(produced),
        "pair_train_examples": int(len(pair_X_tr)),
        "pair_val_examples": int(len(pair_X_va)),
        "comp_train_examples": int(len(comp_X_tr)),
        "comp_val_examples": int(len(comp_X_va)),
        "pair_val_auc": float(pair_auc),
        "pair_val_f1_best": float(best_f1),
        "pair_threshold": float(best_t),
        "score_val_mae": float(score_mae),
    }
    cfg.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class PTDV3MergeScorer:
    def __init__(self, bundle_path: Path):
        with Path(bundle_path).open("rb") as f:
            payload = pickle.load(f)
        self.pair_model = payload["pair_model"]
        self.score_model = payload["score_model"]
        self.pair_threshold = float(payload.get("pair_threshold", 0.56))
        self.adjacency_dilation = int(payload.get("adjacency_dilation", 3))
        self.min_area = int(payload.get("min_area", 24))

        gm = payload.get("graph_merge", {})
        self.strong_prob = float(gm.get("strong_prob", max(self.pair_threshold, 0.62)))
        self.gain_threshold = float(gm.get("gain_threshold", 0.06))
        self.neg_prob = float(gm.get("neg_prob", 0.28))
        self.absorb_threshold = float(gm.get("absorb_threshold", 0.02))

    def _pair_graph(
        self,
        *,
        image_rgb: np.ndarray,
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> tuple[dict[tuple[int, int], tuple[float, float, float, float, float]], list[EdgeDecision]]:
        grad = _grad_map(image_rgb)
        meta: list[tuple[int, int, float, float, float, np.ndarray]] = []
        for i in range(len(proposals)):
            for j in range(i + 1, len(proposals)):
                feat, tex, bweak, hetero = _pair_features(
                    mask_i=proposals[i],
                    mask_j=proposals[j],
                    desc_i=descriptors[i],
                    desc_j=descriptors[j],
                    feature_map=feature_map,
                    grad=grad,
                    adjacency_dilation=self.adjacency_dilation,
                )
                if float(feat[4]) <= 1e-4 and float(feat[5]) <= 1e-4:
                    continue
                meta.append((i, j, tex, bweak, hetero, feat))

        if not meta:
            return {}, []

        feats = np.stack([m[-1] for m in meta], axis=0).astype(np.float32)
        probs = self.pair_model.predict_proba(feats)[:, 1]

        edge_map: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}
        decisions: list[EdgeDecision] = []
        for k, (i, j, tex, bweak, hetero, _) in enumerate(meta):
            p = float(probs[k])
            # Weight is a signed confidence used for global partition gain.
            w = (
                0.95 * _logit(p)
                + 0.22 * float(tex)
                + 0.18 * float(bweak)
                - 0.24 * float(hetero)
            )
            edge_map[(i, j)] = (p, w, float(tex), float(bweak), float(hetero))
            decisions.append(
                EdgeDecision(
                    i=i,
                    j=j,
                    score=float(p),
                    texture_sim=float(tex),
                    boundary_weakness=float(bweak),
                    hetero_penalty=float(hetero),
                    merged=False,
                )
            )
        return edge_map, decisions

    def _cluster_gain(
        self,
        *,
        a: _ClusterState,
        b: _ClusterState,
        edge_map: dict[tuple[int, int], tuple[float, float, float, float, float]],
        feature_map: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        probs: list[float] = []
        ws: list[float] = []
        bweak: list[float] = []
        hetero_edges: list[float] = []
        neg = 0

        for i in a.members:
            for j in b.members:
                x, y = (i, j) if i < j else (j, i)
                meta = edge_map.get((x, y))
                if meta is None:
                    continue
                p, w, _, bw, he = meta
                probs.append(float(p))
                ws.append(float(w))
                bweak.append(float(bw))
                hetero_edges.append(float(he))
                if p <= self.neg_prob:
                    neg += 1

        if not probs:
            return -1e6, 0.0, 1.0, 0.0, 0.0

        avg_p = float(np.mean(probs))
        max_p = float(np.max(probs))
        neg_frac = float(neg / max(len(probs), 1))
        avg_w = float(np.mean(ws))
        avg_bw = float(np.mean(bweak)) if bweak else 0.0

        u = np.logical_or(a.mask > 0, b.mask > 0).astype(np.uint8)
        ua = int(u.sum())
        if ua <= 0:
            hetero_union = 1.0
        else:
            uvar = float(region_variance(feature_map, u))
            base = float((a.var * a.area + b.var * b.area) / max(a.area + b.area, 1))
            hetero_union = max(0.0, uvar - base)

        dsim = float(cosine_similarity(a.desc_mean, b.desc_mean))

        gain = (
            0.95 * avg_w
            + 0.35 * max_p
            + 0.22 * avg_bw
            + 0.24 * dsim
            - 0.42 * hetero_union
            - 0.45 * neg_frac
        )
        return gain, avg_p, neg_frac, max_p, hetero_union

    def merge_components(
        self,
        *,
        image_rgb: np.ndarray,
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> tuple[list[np.ndarray], list[list[int]], list[EdgeDecision]]:
        n = len(proposals)
        if n == 0:
            return [], [], []
        if n == 1:
            return [proposals[0].astype(np.uint8)], [[0]], []

        edge_map, decisions = self._pair_graph(
            image_rgb=image_rgb,
            proposals=proposals,
            descriptors=descriptors,
            feature_map=feature_map,
        )

        if not edge_map:
            return [p.astype(np.uint8) for p in proposals], [[i] for i in range(n)], decisions

        uf = _UnionFind(n)
        clus: dict[int, _ClusterState] = {}
        for i in range(n):
            m = (proposals[i] > 0).astype(np.uint8)
            clus[i] = _ClusterState(
                members=[i],
                mask=m,
                area=int(m.sum()),
                var=float(region_variance(feature_map, m)),
                desc_mean=descriptors[i].astype(np.float32),
            )

        # Global agglomerative partition: high-confidence edges first.
        edge_order = sorted(edge_map.items(), key=lambda kv: kv[1][1], reverse=True)
        for (i, j), _ in edge_order:
            ri = uf.find(i)
            rj = uf.find(j)
            if ri == rj:
                continue
            ci = clus.get(ri)
            cj = clus.get(rj)
            if ci is None or cj is None:
                continue

            gain, avg_p, neg_frac, max_p, hetero_union = self._cluster_gain(
                a=ci,
                b=cj,
                edge_map=edge_map,
                feature_map=feature_map,
            )
            if not np.isfinite(gain):
                continue

            should_merge = (
                gain >= self.gain_threshold
                and neg_frac <= 0.45
                and hetero_union <= 0.48
                and (max_p >= self.strong_prob or avg_p >= self.pair_threshold)
            )
            if not should_merge:
                continue

            uf.union(ri, rj)
            nr = uf.find(ri)
            oa = clus.pop(ri, None)
            ob = clus.pop(rj, None)
            if oa is None or ob is None:
                continue

            um = np.logical_or(oa.mask > 0, ob.mask > 0).astype(np.uint8)
            ua = int(um.sum())
            nmembers = oa.members + ob.members
            ndesc = np.mean(np.stack([descriptors[k] for k in nmembers], axis=0), axis=0).astype(np.float32)
            nvar = float(region_variance(feature_map, um))
            clus[nr] = _ClusterState(members=nmembers, mask=um, area=ua, var=nvar, desc_mean=ndesc)

        # Absorb singleton fragments if they strongly support a neighboring cluster.
        roots = sorted(list(clus.keys()))
        for r in roots:
            if r not in clus:
                continue
            c = clus[r]
            if len(c.members) != 1:
                continue
            i = c.members[0]
            best_r = None
            best_gain = -1e9
            for q in list(clus.keys()):
                if q == r:
                    continue
                cq = clus[q]
                gain, avg_p, neg_frac, max_p, _ = self._cluster_gain(
                    a=c,
                    b=cq,
                    edge_map=edge_map,
                    feature_map=feature_map,
                )
                if neg_frac > 0.35:
                    continue
                if max_p < self.pair_threshold:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_r = q
            if best_r is None or best_gain < self.absorb_threshold:
                continue

            uf.union(r, best_r)
            nr = uf.find(r)
            oa = clus.pop(r, None)
            ob = clus.pop(best_r, None)
            if oa is None or ob is None:
                continue
            um = np.logical_or(oa.mask > 0, ob.mask > 0).astype(np.uint8)
            ua = int(um.sum())
            nmembers = oa.members + ob.members
            ndesc = np.mean(np.stack([descriptors[k] for k in nmembers], axis=0), axis=0).astype(np.float32)
            nvar = float(region_variance(feature_map, um))
            clus[nr] = _ClusterState(members=nmembers, mask=um, area=ua, var=nvar, desc_mean=ndesc)

        comp_indices = [sorted(v.members) for v in clus.values()]
        comp_indices.sort(key=lambda idxs: (-len(idxs), idxs[0]))
        comp_masks: list[np.ndarray] = []
        for idxs in comp_indices:
            m = np.zeros_like(proposals[0], dtype=np.uint8)
            for k in idxs:
                m = np.logical_or(m, proposals[k] > 0)
            comp_masks.append(m.astype(np.uint8))

        # Update decision flags using final partition.
        root_cache = {i: uf.find(i) for i in range(n)}
        out_decisions: list[EdgeDecision] = []
        for d in decisions:
            out_decisions.append(
                EdgeDecision(
                    i=d.i,
                    j=d.j,
                    score=d.score,
                    texture_sim=d.texture_sim,
                    boundary_weakness=d.boundary_weakness,
                    hetero_penalty=d.hetero_penalty,
                    merged=bool(root_cache[d.i] == root_cache[d.j]),
                )
            )
        return comp_masks, comp_indices, out_decisions

    def _proposal_consensus_candidates(self, proposals: list[np.ndarray], min_area: int) -> list[np.ndarray]:
        if not proposals:
            return []
        stack = np.stack([(p > 0).astype(np.float32) for p in proposals], axis=0)
        freq = stack.mean(axis=0)

        out: list[np.ndarray] = []
        for t in (0.20, 0.28, 0.36, 0.44, 0.52, 0.62):
            m = (freq >= t).astype(np.uint8)
            if int(m.sum()) >= min_area:
                out.append(m)

        if out:
            # Keep largest consensus too (anti-fragmentation prior).
            for m in list(out):
                n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if n > 2:
                    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
                    lm = (lbl == largest).astype(np.uint8)
                    if int(lm.sum()) >= min_area:
                        out.append(lm)
        return out

    def score_components(
        self,
        *,
        image_rgb: np.ndarray,
        components: list[np.ndarray],
        proposals: list[np.ndarray],
        descriptors: list[np.ndarray],
        feature_map: np.ndarray,
    ) -> list[float]:
        grad = _grad_map(image_rgb)

        candidates: list[np.ndarray] = []
        candidates.extend([(c > 0).astype(np.uint8) for c in components])
        candidates.extend(self._proposal_consensus_candidates(proposals, min_area=self.min_area))

        # Add high-recall fallback candidates.
        if proposals:
            union = np.logical_or.reduce([(p > 0) for p in proposals]).astype(np.uint8)
            if int(union.sum()) >= self.min_area:
                candidates.append(union)

        # Deduplicate and area filter.
        dedup: list[np.ndarray] = []
        seen: set[bytes] = set()
        for m in candidates:
            b = (m > 0).astype(np.uint8)
            if int(b.sum()) < self.min_area:
                continue
            k = np.packbits(b, axis=None).tobytes()
            if k in seen:
                continue
            seen.add(k)
            dedup.append(b)

        if not dedup:
            return []

        comp_feats: list[np.ndarray] = []
        scores_prior: list[float] = []
        for cm in dedup:
            ft = _component_features(
                comp_mask=cm,
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
            )
            comp_feats.append(ft)

            # Prior terms not requiring labels.
            pa = max(int(cm.sum()), 1)
            support = 0.0
            if proposals:
                ovs = []
                for pm in proposals:
                    inter = float(np.logical_and(pm > 0, cm > 0).sum())
                    ovs.append(inter / max(int((pm > 0).sum()), 1))
                support = float(np.mean(ovs))

            # Encourage coherent single-region outputs.
            n_cc, _, _, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
            frag = max(0, n_cc - 2)

            scores_prior.append(float(0.08 * support - 0.09 * frag))

        arr = np.stack(comp_feats, axis=0)
        pred = self.score_model.predict(arr).astype(np.float32)
        final = pred + np.array(scores_prior, dtype=np.float32)
        return [float(x) for x in final]
