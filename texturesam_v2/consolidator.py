from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np

from .dtd_encoder import DTDEncoderConfig, DTDTextureEncoder
from .features import compute_texture_feature_map, mean_feature, region_descriptor, region_variance
from .merge import EdgeDecision, MergeConfig, merge_masks
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder
from .ptd_learned import PTDLearnedMergeScorer
from .ptd_v3 import PTDV3MergeScorer
from .ptd_v4_set import PTDV4SetScorer
from .ptd_v6_coverage import PTDV6CoverageScorer
from .ptd_v8_partition import PTDV8PartitionScorer


@dataclass(frozen=True)
class ConsolidationConfig:
    min_area: int = 32
    close_kernel: int = 5
    keep_largest_component: bool = True
    hole_area_threshold: int = 64

    descriptor_mode: str = "handcrafted"  # handcrafted | dtd_cnn | hybrid | ptd_convnext | hybrid_ptd
    dtd_checkpoint: Path | None = None
    dtd_device: str = "cuda"
    ptd_checkpoint: Path | None = None
    ptd_device: str = "cuda"
    ptd_use_ring_context: bool = False
    ptd_ring_dilation: int = 9
    ptd_ring_min_pixels: int = 24
    learned_bundle: Path | None = None

    merge: MergeConfig = MergeConfig()

    objective_lambda: float = 0.45
    objective_mu: float = 0.30
    learned_margin_min: float = 0.0


@dataclass(frozen=True)
class CandidateScore:
    score: float
    delta: float
    variance: float
    fragmentation: float


@dataclass(frozen=True)
class ConsolidationDebug:
    num_input_proposals: int
    num_merged_components: int
    selected_index: int
    selected_score: CandidateScore
    edge_decisions: list[EdgeDecision]


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    m = (mask > 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(m, connectivity=8)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    n, lbl, stats, _ = _connected_components(mask)
    if n <= 1:
        return (mask > 0).astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas) + 1)
    return (lbl == largest).astype(np.uint8)


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    inv = 1 - m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return m

    h, w = m.shape
    border_labels = set(np.unique(lbl[0, :]).tolist())
    border_labels.update(np.unique(lbl[h - 1, :]).tolist())
    border_labels.update(np.unique(lbl[:, 0]).tolist())
    border_labels.update(np.unique(lbl[:, w - 1]).tolist())

    out = m.copy()
    for c in range(1, n):
        if c in border_labels:
            continue
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area <= max_area:
            out[lbl == c] = 1
    return out


def _fragmentation_penalty(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    n_cc, _, _, _ = _connected_components(m)
    cc_pen = max(0, n_cc - 2)

    inv = 1 - m
    n_h, lbl_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n_h <= 1:
        holes = 0
    else:
        h, w = m.shape
        border = set(np.unique(lbl_h[0, :]).tolist())
        border.update(np.unique(lbl_h[h - 1, :]).tolist())
        border.update(np.unique(lbl_h[:, 0]).tolist())
        border.update(np.unique(lbl_h[:, w - 1]).tolist())
        holes = 0
        for c in range(1, n_h):
            if c in border:
                continue
            if int(stats_h[c, cv2.CC_STAT_AREA]) > 0:
                holes += 1

    return float(cc_pen + 0.5 * holes)


def _postprocess(mask: np.ndarray, cfg: ConsolidationConfig) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)

    if cfg.close_kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if cfg.hole_area_threshold > 0:
        m = _fill_small_holes(m, cfg.hole_area_threshold)

    if cfg.keep_largest_component:
        m = _keep_largest(m)

    return m.astype(np.uint8)


def _score_candidate(feature_map: np.ndarray, mask: np.ndarray, cfg: ConsolidationConfig) -> CandidateScore:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0 or int(m.sum()) == m.size:
        return CandidateScore(score=-1e6, delta=0.0, variance=1.0, fragmentation=10.0)

    inside_mean = mean_feature(feature_map, m)
    outside_mean = mean_feature(feature_map, 1 - m)
    delta = float(np.linalg.norm(inside_mean - outside_mean))
    var = float(region_variance(feature_map, m))
    frag = _fragmentation_penalty(m)

    score = delta - cfg.objective_lambda * var - cfg.objective_mu * frag
    return CandidateScore(score=float(score), delta=delta, variance=var, fragmentation=frag)


def _dedupe_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for m in masks:
        b = (m > 0).astype(np.uint8)
        k = np.packbits(b, axis=None).tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out


def _fallback_candidates_from_image(image_rgb: np.ndarray, cfg: ConsolidationConfig) -> list[np.ndarray]:
    """
    Deterministic image-only fallback used when SAM proposals are missing.
    Produces a small candidate pool without any dataset supervision.
    """
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), sigmaX=1.1)
    lap = lap / (float(lap.max()) + 1e-8)

    lab = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ab = lab[..., 1:3]
    ab_mean = ab.reshape(-1, 2).mean(axis=0)
    color_delta = np.linalg.norm(ab - ab_mean[None, None, :], axis=2)
    color_delta = cv2.GaussianBlur(color_delta, (0, 0), sigmaX=1.1)
    color_delta = color_delta / (float(color_delta.max()) + 1e-8)

    maps = [
        lap,
        color_delta,
        0.6 * lap + 0.4 * color_delta,
    ]
    quantiles = (55, 65, 75, 85)
    max_area = int(round(0.98 * gray.size))

    cands: list[np.ndarray] = []
    for mp in maps:
        flat = mp.reshape(-1)
        for q in quantiles:
            thr = float(np.percentile(flat, q))
            for invert in (False, True):
                raw = (mp <= thr) if invert else (mp >= thr)
                m = _postprocess(raw.astype(np.uint8), cfg)
                area = int(m.sum())
                if area < cfg.min_area or area > max_area:
                    continue
                cands.append(m.astype(np.uint8))

    cands = _dedupe_masks(cands)
    if cands:
        return cands

    # Last-resort deterministic non-empty seed.
    h, w = gray.shape
    seed = np.zeros((h, w), dtype=np.uint8)
    rh = max(6, h // 8)
    rw = max(6, w // 8)
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - rh), min(h, cy + rh)
    x0, x1 = max(0, cx - rw), min(w, cx + rw)
    seed[y0:y1, x0:x1] = 1
    return [_postprocess(seed, cfg).astype(np.uint8)]


def _select_best(scored_masks: list[np.ndarray], feature_map: np.ndarray, cfg: ConsolidationConfig) -> tuple[np.ndarray, int, CandidateScore]:
    scored = [_score_candidate(feature_map, m, cfg) for m in scored_masks]
    idx = int(np.argmax([s.score for s in scored]))
    return scored_masks[idx].astype(np.uint8), idx, scored[idx]


class TextureSAMV2Consolidator:
    def __init__(self, cfg: ConsolidationConfig | None = None):
        self.cfg = cfg or ConsolidationConfig()
        self.dtd_encoder: DTDTextureEncoder | None = None
        self.ptd_encoder: PTDTextureEncoder | None = None
        self.learned_models: PTDLearnedMergeScorer | None = None
        if self.cfg.descriptor_mode in {"dtd_cnn", "hybrid"}:
            if self.cfg.dtd_checkpoint is None:
                raise ValueError("descriptor_mode requires dtd_checkpoint")
            self.dtd_encoder = DTDTextureEncoder(
                DTDEncoderConfig(checkpoint=self.cfg.dtd_checkpoint, device=self.cfg.dtd_device)
            )
        if self.cfg.descriptor_mode in {"ptd_convnext", "hybrid_ptd"}:
            if self.cfg.ptd_checkpoint is None:
                raise ValueError("descriptor_mode requires ptd_checkpoint")
            self.ptd_encoder = PTDTextureEncoder(
                PTDEncoderConfig(
                    checkpoint=self.cfg.ptd_checkpoint,
                    device=self.cfg.ptd_device,
                    use_ring_context=self.cfg.ptd_use_ring_context,
                    ring_dilation=self.cfg.ptd_ring_dilation,
                    ring_min_pixels=self.cfg.ptd_ring_min_pixels,
                )
            )
        if self.cfg.learned_bundle is not None:
            try:
                with Path(self.cfg.learned_bundle).open("rb") as f:
                    payload = pickle.load(f)
                version = str(payload.get("feature_version", ""))
            except Exception:
                version = ""
            if version.startswith("ptd_learned_v3"):
                self.learned_models = PTDV3MergeScorer(self.cfg.learned_bundle)
            elif version.startswith("ptd_learned_v4_set"):
                self.learned_models = PTDV4SetScorer(self.cfg.learned_bundle)
            elif version.startswith("ptd_learned_v6_coverage"):
                self.learned_models = PTDV6CoverageScorer(self.cfg.learned_bundle)
            elif version.startswith("ptd_learned_v8_partition"):
                self.learned_models = PTDV8PartitionScorer(self.cfg.learned_bundle)
            else:
                self.learned_models = PTDLearnedMergeScorer(self.cfg.learned_bundle)

    def _build_descriptors(self, image_rgb: np.ndarray, feat_map: np.ndarray, proposals: list[np.ndarray]) -> list[np.ndarray]:
        mode = self.cfg.descriptor_mode
        if mode == "handcrafted":
            return [region_descriptor(feat_map, m) for m in proposals]

        if mode == "dtd_cnn":
            assert self.dtd_encoder is not None
            dtd_desc = self.dtd_encoder.encode_regions(image_rgb, proposals)
            return [d.astype(np.float32) for d in dtd_desc]

        if mode == "hybrid":
            assert self.dtd_encoder is not None
            dtd_desc = self.dtd_encoder.encode_regions(image_rgb, proposals)
            hand = [region_descriptor(feat_map, m) for m in proposals]
            return [np.concatenate([h, d], axis=0).astype(np.float32) for h, d in zip(hand, dtd_desc)]

        if mode == "ptd_convnext":
            assert self.ptd_encoder is not None
            return [d.astype(np.float32) for d in self.ptd_encoder.encode_regions(image_rgb, proposals)]

        if mode == "hybrid_ptd":
            assert self.ptd_encoder is not None
            ptd_desc = self.ptd_encoder.encode_regions(image_rgb, proposals)
            hand = [region_descriptor(feat_map, m) for m in proposals]
            return [np.concatenate([h, d], axis=0).astype(np.float32) for h, d in zip(hand, ptd_desc)]

        raise ValueError(f"Unknown descriptor_mode: {mode}")

    def __call__(self, image_rgb: np.ndarray, proposals: list[np.ndarray]) -> tuple[np.ndarray, ConsolidationDebug]:
        if len(proposals) == 0:
            empty = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            dbg = ConsolidationDebug(
                num_input_proposals=0,
                num_merged_components=0,
                selected_index=-1,
                selected_score=CandidateScore(score=-1e6, delta=0.0, variance=1.0, fragmentation=0.0),
                edge_decisions=[],
            )
            return empty, dbg

        clean_props = []
        for p in proposals:
            m = (p > 0).astype(np.uint8)
            if int(m.sum()) >= self.cfg.min_area:
                clean_props.append(m)
        if not clean_props:
            empty = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            dbg = ConsolidationDebug(
                num_input_proposals=len(proposals),
                num_merged_components=0,
                selected_index=-1,
                selected_score=CandidateScore(score=-1e6, delta=0.0, variance=1.0, fragmentation=0.0),
                edge_decisions=[],
            )
            return empty, dbg

        feat_map = compute_texture_feature_map(image_rgb)
        desc = self._build_descriptors(image_rgb=image_rgb, feat_map=feat_map, proposals=clean_props)

        if self.learned_models is None:
            merged_components, edge_decisions = merge_masks(
                image_rgb=image_rgb,
                proposals=clean_props,
                descriptors=desc,
                feature_map=feat_map,
                cfg=self.cfg.merge,
            )
            learned_scores: list[float] | None = None
        else:
            merged_components, _, edge_decisions = self.learned_models.merge_components(
                image_rgb=image_rgb,
                proposals=clean_props,
                descriptors=desc,
                feature_map=feat_map,
            )
            learned_scores = self.learned_models.score_components(
                image_rgb=image_rgb,
                components=merged_components,
                proposals=clean_props,
                descriptors=desc,
                feature_map=feat_map,
            )

        if not merged_components:
            merged_components = clean_props

        scored: list[CandidateScore] = []
        processed_components: list[np.ndarray] = []
        for c in merged_components:
            pp = _postprocess(c, self.cfg)
            processed_components.append(pp)
            scored.append(_score_candidate(feat_map, pp, self.cfg))

        if learned_scores is None or len(learned_scores) != len(processed_components):
            idx = int(np.argmax([s.score for s in scored]))
        else:
            arr = np.asarray(learned_scores, dtype=np.float32)
            idx_learned = int(np.argmax(arr))
            idx = idx_learned
            # Margin-aware guard: if learned ranking is ambiguous, trust the
            # more conservative handcrafted objective among current components.
            if float(self.cfg.learned_margin_min) > 0.0 and arr.size >= 2:
                top2 = np.sort(np.partition(arr, -2)[-2:])
                margin = float(top2[1] - top2[0]) if top2.size == 2 else 0.0
                if margin < float(self.cfg.learned_margin_min):
                    idx = int(np.argmax([s.score for s in scored]))
            scored[idx] = CandidateScore(
                score=float(arr[idx_learned]),
                delta=scored[idx].delta,
                variance=scored[idx].variance,
                fragmentation=scored[idx].fragmentation,
            )
        best = processed_components[idx].astype(np.uint8)

        dbg = ConsolidationDebug(
            num_input_proposals=len(clean_props),
            num_merged_components=len(merged_components),
            selected_index=idx,
            selected_score=scored[idx],
            edge_decisions=edge_decisions,
        )
        return best, dbg
