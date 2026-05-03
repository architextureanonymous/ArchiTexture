from __future__ import annotations

import functools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from .features import cosine_similarity, mean_feature, region_descriptor, region_variance
from .merge import EdgeDecision
from .ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries, split_ptd_entries
from .ptd_encoder import PTDEncoderConfig, PTDTextureEncoder

EPS = 1e-8
HYBRID_HANDCRAFT_SCALE = 0.65


@dataclass(frozen=True)
class PTDLearnedTrainConfig:
    ptd_root: Path
    ptd_encoder_ckpt: Path
    out_bundle: Path
    out_metrics_json: Path
    descriptor_mode: str = "hybrid_ptd"
    num_samples: int = 1200
    val_fraction: float = 0.15
    image_size: int = 256
    min_regions: int = 2
    max_regions: int = 4
    min_fg_frags: int = 4
    max_fg_frags: int = 9
    random_seed: int = 1337
    pair_n_estimators: int = 450
    score_n_estimators: int = 500
    min_area: int = 24
    adjacency_dilation: int = 3
    synthetic_layout: str = "voronoi"
    stmd_additions_per_region: int = 10
    stmd_smooth_kernel: int = 10
    shape_mask_root: Path | None = None
    shape_min_area_ratio: float = 0.08
    shape_max_area_ratio: float = 0.55
    shape_rotation_max_deg: float = 40.0
    shape_margin_fraction: float = 0.05


class _UnionFind:
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


def _texture_canvas(tex: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ih, iw = tex.shape[:2]
    if ih < 8 or iw < 8:
        return cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)

    crop_h = int(np.clip(rng.uniform(0.45, 0.95) * ih, 8, ih))
    crop_w = int(np.clip(rng.uniform(0.45, 0.95) * iw, 8, iw))
    y0 = int(rng.integers(0, max(1, ih - crop_h + 1)))
    x0 = int(rng.integers(0, max(1, iw - crop_w + 1)))
    crop = tex[y0 : y0 + crop_h, x0 : x0 + crop_w]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    # Mild photometric jitter.
    alpha = float(rng.uniform(0.85, 1.20))
    beta = float(rng.uniform(-18.0, 18.0))
    out = np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return out


def _voronoi_labels(h: int, w: int, n_regions: int, rng: np.random.Generator) -> np.ndarray:
    ys = rng.uniform(0, h, size=(n_regions,))
    xs = rng.uniform(0, w, size=(n_regions,))
    yy, xx = np.mgrid[0:h, 0:w]
    d = []
    for i in range(n_regions):
        scale = float(rng.uniform(0.85, 1.35))
        di = (xx - xs[i]) ** 2 + scale * (yy - ys[i]) ** 2
        d.append(di[..., None])
    stack = np.concatenate(d, axis=2)
    lbl = np.argmin(stack, axis=2).astype(np.int32) + 1

    # Smooth region boundaries so fragments are less axis-aligned.
    for _ in range(2):
        lbl = cv2.medianBlur(lbl.astype(np.uint8), 5).astype(np.int32)
    return lbl


def _sample_gaussian_blob(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    center_y: float | None = None,
    center_x: float | None = None,
) -> np.ndarray:
    cy = float(rng.uniform(0.15 * h, 0.85 * h) if center_y is None else center_y)
    cx = float(rng.uniform(0.15 * w, 0.85 * w) if center_x is None else center_x)
    sy = float(rng.uniform(0.06, 0.22) * h)
    sx = float(rng.uniform(0.06, 0.22) * w)
    ang = float(rng.uniform(0.0, math.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y0 = yy - cy
    x0 = xx - cx
    ca = math.cos(ang)
    sa = math.sin(ang)
    xr = ca * x0 + sa * y0
    yr = -sa * x0 + ca * y0
    score = np.exp(-0.5 * ((xr / max(sx, 1.0)) ** 2 + (yr / max(sy, 1.0)) ** 2))
    thr = float(rng.uniform(0.22, 0.70))
    return (score >= thr).astype(np.uint8)


def _sample_uniform_blob(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    center_y: float | None = None,
    center_x: float | None = None,
) -> np.ndarray:
    cy = float(rng.uniform(0.15 * h, 0.85 * h) if center_y is None else center_y)
    cx = float(rng.uniform(0.15 * w, 0.85 * w) if center_x is None else center_x)
    hh = int(max(8, round(rng.uniform(0.10, 0.34) * h)))
    ww = int(max(8, round(rng.uniform(0.10, 0.34) * w)))
    ang = float(rng.uniform(0.0, 180.0))
    rr = ((cx, cy), (ww, hh), ang)
    box = cv2.boxPoints(rr)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(out, np.round(box).astype(np.int32), 1)
    return out


def _sample_stmd_primitive(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    center_y: float | None = None,
    center_x: float | None = None,
) -> np.ndarray:
    if rng.random() < 0.5:
        return _sample_gaussian_blob(h, w, rng, center_y=center_y, center_x=center_x)
    return _sample_uniform_blob(h, w, rng, center_y=center_y, center_x=center_x)


def _random_point_in_mask(mask: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return float(rng.uniform(0.15 * h, 0.85 * h)), float(rng.uniform(0.15 * w, 0.85 * w))
    idx = int(rng.integers(0, len(ys)))
    return float(ys[idx]), float(xs[idx])


def _smooth_label_mode(labels: np.ndarray, n_labels: int, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k <= 1:
        return labels.astype(np.int32)
    counts: list[np.ndarray] = []
    for lab in range(n_labels + 1):
        m = (labels == lab).astype(np.float32)
        counts.append(
            cv2.boxFilter(m, ddepth=-1, ksize=(k, k), normalize=False, borderType=cv2.BORDER_REFLECT)[..., None]
        )
    stack = np.concatenate(counts, axis=2)
    return np.argmax(stack, axis=2).astype(np.int32)


def _stmd_labels(
    h: int,
    w: int,
    n_regions: int,
    rng: np.random.Generator,
    *,
    additions_per_region: int,
    smooth_kernel: int,
) -> np.ndarray:
    region_masks: list[np.ndarray] = []
    for _ in range(int(n_regions)):
        region = _sample_stmd_primitive(h, w, rng)
        for _ in range(int(additions_per_region)):
            py, px = _random_point_in_mask(region, rng)
            region = np.logical_or(region > 0, _sample_stmd_primitive(h, w, rng, center_y=py, center_x=px) > 0)
        region_masks.append(region.astype(np.uint8))

    stack = np.stack([(m > 0).astype(np.uint8) for m in region_masks], axis=0)
    labels = np.zeros((h, w), dtype=np.int32)
    count = stack.sum(axis=0)

    single = count == 1
    if np.any(single):
        labels[single] = np.argmax(stack[:, single], axis=0).astype(np.int32) + 1

    overlap_y, overlap_x = np.where(count > 1)
    for y, x in zip(overlap_y.tolist(), overlap_x.tolist(), strict=False):
        owners = np.flatnonzero(stack[:, y, x] > 0)
        labels[y, x] = int(owners[int(rng.integers(0, len(owners)))]) + 1

    labels = _smooth_label_mode(labels, int(n_regions), int(smooth_kernel))
    return labels.astype(np.int32)


@functools.lru_cache(maxsize=4096)
def _load_binary_shape(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert("L"))
        except Exception as exc:
            raise FileNotFoundError(f"Failed to read shape mask: {path}") from exc
    mask = arr > 127
    if float(mask.mean()) > 0.50:
        mask = ~mask
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise RuntimeError(f"Shape mask is empty after thresholding: {path}")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return mask[y0:y1, x0:x1].astype(np.uint8)


def _rotate_binary_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return mask.astype(np.uint8)
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])
    out_w = int(math.ceil(w * cos + h * sin))
    out_h = int(math.ceil(w * sin + h * cos))
    mat[0, 2] += out_w / 2.0 - w / 2.0
    mat[1, 2] += out_h / 2.0 - h / 2.0
    rot = cv2.warpAffine(
        (mask > 0).astype(np.uint8) * 255,
        mat,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    ys, xs = np.where(rot > 0)
    if len(ys) == 0:
        return (mask > 0).astype(np.uint8)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return (rot[y0:y1, x0:x1] > 0).astype(np.uint8)


def _mpeg7_shape_labels(
    h: int,
    w: int,
    rng: np.random.Generator,
    *,
    shape_paths: list[Path],
    cfg: PTDLearnedTrainConfig,
) -> np.ndarray:
    if not shape_paths:
        raise RuntimeError("synthetic_layout='mpeg7_shapes' requires non-empty shape paths.")

    min_area_ratio = float(np.clip(cfg.shape_min_area_ratio, 0.01, 0.90))
    max_area_ratio = float(np.clip(cfg.shape_max_area_ratio, min_area_ratio + 1e-3, 0.95))
    margin = int(round(float(cfg.shape_margin_fraction) * min(h, w)))
    margin = max(0, min(margin, max((min(h, w) // 2) - 8, 0)))
    max_inner_h = max(16, h - 2 * margin)
    max_inner_w = max(16, w - 2 * margin)

    last_mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(32):
        src = _load_binary_shape(shape_paths[int(rng.integers(0, len(shape_paths)))])
        mask = src.copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if float(cfg.shape_rotation_max_deg) > 0.0:
            ang = float(rng.uniform(-cfg.shape_rotation_max_deg, cfg.shape_rotation_max_deg))
            mask = _rotate_binary_mask(mask, ang)

        sh, sw = mask.shape
        fit = min(max_inner_h / max(sh, 1), max_inner_w / max(sw, 1))
        fit = max(fit, 0.05)
        scale = float(rng.uniform(0.35, 0.95)) * fit
        out_h = max(12, int(round(sh * scale)))
        out_w = max(12, int(round(sw * scale)))
        if out_h >= h or out_w >= w:
            continue

        resized = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        resized = cv2.morphologyEx((resized > 0).astype(np.uint8), cv2.MORPH_CLOSE, k)
        area_ratio = float(resized.sum() / max(h * w, 1))
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            last_mask = resized
            continue

        y_lo = margin
        y_hi = max(y_lo, h - margin - out_h)
        x_lo = margin
        x_hi = max(x_lo, w - margin - out_w)
        y0 = int(rng.integers(y_lo, y_hi + 1))
        x0 = int(rng.integers(x_lo, x_hi + 1))
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
        labels = np.ones((h, w), dtype=np.int32)
        labels[canvas > 0] = 2
        return labels

    if int(last_mask.sum()) <= 0:
        fallback = np.zeros((h, w), dtype=np.uint8)
        cy = h // 2
        cx = w // 2
        ry = max(12, int(round(0.18 * h)))
        rx = max(12, int(round(0.20 * w)))
        cv2.ellipse(fallback, (cx, cy), (rx, ry), 0.0, 0, 360, 1, -1)
        labels = np.ones((h, w), dtype=np.int32)
        labels[fallback > 0] = 2
        return labels

    fh, fw = last_mask.shape
    y0 = max(0, (h - fh) // 2)
    x0 = max(0, (w - fw) // 2)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y0 : y0 + min(fh, h), x0 : x0 + min(fw, w)] = last_mask[: min(fh, h), : min(fw, w)]
    labels = np.ones((h, w), dtype=np.int32)
    labels[canvas > 0] = 2
    return labels


def _compose_synthetic(
    *,
    backend: PTDImageBackend,
    class_to_entries: dict[int, list],
    rng: np.random.Generator,
    cfg: PTDLearnedTrainConfig,
    shape_paths: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    h = int(cfg.image_size)
    w = int(cfg.image_size)
    cls_ids = list(class_to_entries.keys())
    layout = str(cfg.synthetic_layout)
    if layout == "mpeg7_shapes":
        n_regions = 2
        sel_cls = rng.choice(cls_ids, size=n_regions, replace=False if len(cls_ids) >= n_regions else True).tolist()
        labels = _mpeg7_shape_labels(h, w, rng, shape_paths=shape_paths or [], cfg=cfg)
        fg_region = 2
    else:
        n_regions = int(rng.integers(cfg.min_regions, cfg.max_regions + 1))
        sel_cls = rng.choice(cls_ids, size=n_regions, replace=False if len(cls_ids) >= n_regions else True).tolist()
        fg_region = int(rng.integers(1, n_regions + 1))

    if layout == "stmd_blobs":
        labels = _stmd_labels(
            h,
            w,
            n_regions,
            rng,
            additions_per_region=int(cfg.stmd_additions_per_region),
            smooth_kernel=int(cfg.stmd_smooth_kernel),
        )
    elif layout != "mpeg7_shapes":
        labels = _voronoi_labels(h, w, n_regions, rng)
    image = np.zeros((h, w, 3), dtype=np.uint8)

    for ridx, cid in enumerate(sel_cls, start=1):
        entries = class_to_entries[int(cid)]
        e = entries[int(rng.integers(0, len(entries)))]
        tex = backend.read_rgb(e.rel_path)
        can = _texture_canvas(tex, h=h, w=w, rng=rng)
        m = labels == ridx
        image[m] = can[m]

    gt = (labels == fg_region).astype(np.uint8)
    return image, gt, labels, fg_region


def _fragment_mask(mask: np.ndarray, rng: np.random.Generator, min_fg_frags: int, max_fg_frags: int) -> list[np.ndarray]:
    h, w = mask.shape
    out: list[np.ndarray] = []
    n = int(rng.integers(min_fg_frags, max_fg_frags + 1))
    yy, xx = np.mgrid[0:h, 0:w]

    for _ in range(n):
        m = mask.copy().astype(np.uint8)

        if rng.random() < 0.90:
            ksz = int(rng.integers(3, 15))
            if ksz % 2 == 0:
                ksz += 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            if rng.random() < 0.55:
                it = int(rng.integers(1, 3))
                m = cv2.erode(m, k, iterations=it)
            else:
                it = int(rng.integers(1, 2))
                m = cv2.dilate(m, k, iterations=it)
                m = np.logical_and(m > 0, mask > 0).astype(np.uint8)

        if rng.random() < 0.90:
            a, b = rng.normal(size=(2,))
            c = float(rng.uniform(-0.65, 0.65) * max(h, w))
            plane = a * (xx - w / 2.0) + b * (yy - h / 2.0) + c
            if rng.random() < 0.5:
                m[plane > 0] = 0
            else:
                m[plane < 0] = 0

        out.append(m.astype(np.uint8))
    return out


def _make_synthetic_proposals(
    *,
    gt: np.ndarray,
    labels: np.ndarray,
    fg_region: int,
    rng: np.random.Generator,
    cfg: PTDLearnedTrainConfig,
) -> list[np.ndarray]:
    proposals = _fragment_mask(gt, rng, cfg.min_fg_frags, cfg.max_fg_frags)
    h, w = gt.shape

    # Add background distractors from non-foreground regions.
    unique_regs = [int(x) for x in np.unique(labels) if int(x) > 0 and int(x) != int(fg_region)]
    rng.shuffle(unique_regs)
    for rid in unique_regs[: int(rng.integers(1, min(4, len(unique_regs) + 1)) )]:
        m = (labels == rid).astype(np.uint8)
        if rng.random() < 0.8:
            ksz = int(rng.integers(3, 11))
            if ksz % 2 == 0:
                ksz += 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            if rng.random() < 0.5:
                m = cv2.erode(m, k, iterations=1)
            else:
                m = cv2.dilate(m, k, iterations=1)
        proposals.append(m.astype(np.uint8))

    # Add random geometric negatives.
    for _ in range(int(rng.integers(1, 4))):
        m = np.zeros((h, w), dtype=np.uint8)
        x0 = int(rng.integers(0, max(1, w - 24)))
        y0 = int(rng.integers(0, max(1, h - 24)))
        ww = int(rng.integers(18, max(19, min(90, w - x0))))
        hh = int(rng.integers(18, max(19, min(90, h - y0))))
        m[y0 : y0 + hh, x0 : x0 + ww] = 1
        proposals.append(m)

    # Optional GT-like full region proposal.
    if rng.random() < 0.7:
        proposals.append(gt.astype(np.uint8))

    return _dedupe_masks(proposals, min_area=cfg.min_area)


def _majority_label(mask: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    m = mask > 0
    if not np.any(m):
        return 0, 0.0
    vals, cnt = np.unique(labels[m], return_counts=True)
    if len(vals) == 0:
        return 0, 0.0
    idx = int(np.argmax(cnt))
    maj = int(vals[idx])
    pur = float(cnt[idx] / max(int(m.sum()), 1))
    return maj, pur


def _grad_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    return g / (float(g.max()) + EPS)


def _fragmentation_penalty(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cc_pen = max(0, n_cc - 2)
    holes = 0
    inv = 1 - m
    n_h, lbl_h, stats_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n_h > 1:
        h, w = m.shape
        border = set(np.unique(lbl_h[0, :]).tolist())
        border.update(np.unique(lbl_h[h - 1, :]).tolist())
        border.update(np.unique(lbl_h[:, 0]).tolist())
        border.update(np.unique(lbl_h[:, w - 1]).tolist())
        for c in range(1, n_h):
            if c in border:
                continue
            if int(stats_h[c, cv2.CC_STAT_AREA]) > 0:
                holes += 1
    return float(cc_pen + 0.5 * holes)


def _pair_features(
    *,
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    desc_i: np.ndarray,
    desc_j: np.ndarray,
    feature_map: np.ndarray,
    grad: np.ndarray,
    adjacency_dilation: int,
) -> tuple[np.ndarray, float, float, float]:
    m1 = (mask_i > 0).astype(np.uint8)
    m2 = (mask_j > 0).astype(np.uint8)
    area1 = int(m1.sum())
    area2 = int(m2.sum())

    ksz = 2 * int(adjacency_dilation) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    d1 = cv2.dilate(m1, k, iterations=1)
    d2 = cv2.dilate(m2, k, iterations=1)
    near = np.logical_and(d1 > 0, d2 > 0)

    b1 = cv2.morphologyEx(m1, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    b2 = cv2.morphologyEx(m2, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    contact = np.logical_and(b1, b2)

    inter = int(np.logical_and(m1 > 0, m2 > 0).sum())
    uni = int(np.logical_or(m1 > 0, m2 > 0).sum())
    iou = 1.0 if uni == 0 else float(inter / max(uni, 1))

    near_ratio = float(near.sum() / max(min(area1, area2), 1))
    contact_ratio = float(contact.sum() / max(min(area1, area2), 1))

    interface = np.logical_and(near, np.logical_not(np.logical_or(m1 > 0, m2 > 0)))
    if np.any(interface):
        b_strength = float(grad[interface].mean())
    elif np.any(near):
        b_strength = float(grad[near].mean())
    else:
        b_strength = 1.0
    b_weak = 1.0 - float(np.clip(b_strength, 0.0, 1.0))

    union = np.logical_or(m1 > 0, m2 > 0).astype(np.uint8)
    u_var = region_variance(feature_map, union)
    base_var = 0.5 * (region_variance(feature_map, m1) + region_variance(feature_map, m2))
    hetero = max(0.0, float(u_var - base_var))

    h, w = m1.shape
    y1, x1 = np.where(m1 > 0)
    y2, x2 = np.where(m2 > 0)
    if len(y1) and len(y2):
        cy1, cx1 = float(y1.mean() / max(h - 1, 1)), float(x1.mean() / max(w - 1, 1))
        cy2, cx2 = float(y2.mean() / max(h - 1, 1)), float(x2.mean() / max(w - 1, 1))
        cdist = math.sqrt((cy1 - cy2) ** 2 + (cx1 - cx2) ** 2)
    else:
        cdist = 1.0

    tex = float(cosine_similarity(desc_i, desc_j))
    union_ratio = float(uni / max(h * w, 1))
    area_min = float(min(area1, area2) / max(h * w, 1))
    area_max = float(max(area1, area2) / max(h * w, 1))

    feat = np.array(
        [
            tex,
            area_min,
            area_max,
            iou,
            near_ratio,
            contact_ratio,
            b_weak,
            hetero,
            cdist,
            union_ratio,
        ],
        dtype=np.float32,
    )
    return feat, tex, b_weak, hetero


def _component_features(
    *,
    comp_mask: np.ndarray,
    feature_map: np.ndarray,
    grad: np.ndarray,
    proposals: list[np.ndarray],
    descriptors: list[np.ndarray],
) -> np.ndarray:
    m = (comp_mask > 0).astype(np.uint8)
    h, w = m.shape
    area = int(m.sum())
    if area <= 0:
        return np.zeros((10,), dtype=np.float32)

    area_ratio = float(area / max(h * w, 1))
    inside = mean_feature(feature_map, m)
    outside = mean_feature(feature_map, 1 - m)
    delta = float(np.linalg.norm(inside - outside))
    var = float(region_variance(feature_map, m))
    frag = float(_fragmentation_penalty(m))

    bd = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    b_strength = float(grad[bd].mean()) if np.any(bd) else 0.0

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    compact = float((perim * perim) / (4.0 * math.pi * max(area, 1)))

    in_idx: list[int] = []
    for i, pm in enumerate(proposals):
        pa = int((pm > 0).sum())
        if pa <= 0:
            continue
        ov = float(np.logical_and(pm > 0, m > 0).sum() / max(pa, 1))
        if ov >= 0.50:
            in_idx.append(i)

    if not in_idx:
        best_i = 0
        best_ov = -1.0
        for i, pm in enumerate(proposals):
            ov = float(np.logical_and(pm > 0, m > 0).sum())
            if ov > best_ov:
                best_ov = ov
                best_i = i
        in_idx = [best_i]

    out_idx = [i for i in range(len(proposals)) if i not in in_idx]
    in_desc = np.stack([descriptors[i] for i in in_idx], axis=0)
    in_mean = in_desc.mean(axis=0)
    in_var = float(np.var(in_desc, axis=0).mean())
    if out_idx:
        out_desc = np.stack([descriptors[i] for i in out_idx], axis=0)
        out_mean = out_desc.mean(axis=0)
        desc_delta = float(np.linalg.norm(in_mean - out_mean))
    else:
        desc_delta = float(np.linalg.norm(in_mean))

    n_props = float(len(in_idx) / max(len(proposals), 1))
    return np.array(
        [area_ratio, delta, var, frag, b_strength, compact, in_var, desc_delta, n_props, float(len(in_idx))],
        dtype=np.float32,
    )


def _build_training_descriptors(
    *,
    image_rgb: np.ndarray,
    feature_map: np.ndarray,
    proposals: list[np.ndarray],
    encoder: PTDTextureEncoder | None,
    descriptor_mode: str,
) -> list[np.ndarray]:
    mode = str(descriptor_mode)
    if mode == "handcrafted":
        return [region_descriptor(feature_map, m).astype(np.float32) for m in proposals]

    if mode == "ptd_convnext":
        if encoder is None:
            raise RuntimeError("descriptor_mode='ptd_convnext' requires a PTD encoder.")
        return [d.astype(np.float32) for d in encoder.encode_regions(image_rgb, proposals)]

    if mode == "hybrid_ptd":
        if encoder is None:
            raise RuntimeError("descriptor_mode='hybrid_ptd' requires a PTD encoder.")
        hand = [region_descriptor(feature_map, m).astype(np.float32) for m in proposals]
        emb = encoder.encode_regions(image_rgb, proposals)
        return [
            np.concatenate([HYBRID_HANDCRAFT_SCALE * h, e.astype(np.float32)], axis=0).astype(np.float32)
            for h, e in zip(hand, emb)
        ]

    raise ValueError(f"Unsupported descriptor_mode for PTD learned training: {mode}")


def _generate_sample(
    *,
    backend: PTDImageBackend,
    class_to_entries: dict[int, list],
    encoder: PTDTextureEncoder | None,
    cfg: PTDLearnedTrainConfig,
    rng: np.random.Generator,
    shape_paths: list[Path] | None = None,
):
    from .features import compute_texture_feature_map

    image, gt, labels, fg_region = _compose_synthetic(
        backend=backend,
        class_to_entries=class_to_entries,
        rng=rng,
        cfg=cfg,
        shape_paths=shape_paths,
    )
    proposals = _make_synthetic_proposals(gt=gt, labels=labels, fg_region=fg_region, rng=rng, cfg=cfg)
    if len(proposals) < 2:
        return None

    feat_map = compute_texture_feature_map(image)
    grad = _grad_map(image)
    desc = _build_training_descriptors(
        image_rgb=image,
        feature_map=feat_map,
        proposals=proposals,
        encoder=encoder,
        descriptor_mode=cfg.descriptor_mode,
    )

    majors = [_majority_label(m, labels) for m in proposals]
    pair_X: list[np.ndarray] = []
    pair_y: list[int] = []

    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            fi, tex, bweak, hetero = _pair_features(
                mask_i=proposals[i],
                mask_j=proposals[j],
                desc_i=desc[i],
                desc_j=desc[j],
                feature_map=feat_map,
                grad=grad,
                adjacency_dilation=cfg.adjacency_dilation,
            )
            # only train on potentially adjacent pairs (same inference regime)
            if float(fi[4]) <= 1e-4 and float(fi[5]) <= 1e-4:
                continue
            li, pi = majors[i]
            lj, pj = majors[j]
            if min(pi, pj) < 0.45:
                continue
            y = 1 if (li == lj and pi >= 0.60 and pj >= 0.60) else 0
            pair_X.append(fi)
            pair_y.append(int(y))

    if not pair_X:
        return None

    # Component candidates for global scorer.
    cand_masks: list[np.ndarray] = []
    cand_masks.extend(proposals)

    fg_like = [k for k, (lab, pur) in enumerate(majors) if lab == fg_region and pur >= 0.50]
    if fg_like:
        u = np.zeros_like(gt, dtype=np.uint8)
        for k in fg_like:
            u = np.logical_or(u, proposals[k] > 0)
        cand_masks.append(u.astype(np.uint8))

    n_union = int(rng.integers(6, 13))
    for _ in range(n_union):
        take = int(rng.integers(2, min(6, len(proposals)) + 1))
        idx = rng.choice(len(proposals), size=take, replace=False).tolist()
        u = np.zeros_like(gt, dtype=np.uint8)
        for k in idx:
            u = np.logical_or(u, proposals[k] > 0)
        cand_masks.append(u.astype(np.uint8))

    cand_masks.append(np.logical_or.reduce([(p > 0) for p in proposals]).astype(np.uint8))
    cand_masks = _dedupe_masks(cand_masks, min_area=cfg.min_area)

    comp_X: list[np.ndarray] = []
    comp_y: list[float] = []
    gt_b = (gt > 0).astype(np.uint8)
    for cm in cand_masks:
        ft = _component_features(comp_mask=cm, feature_map=feat_map, grad=grad, proposals=proposals, descriptors=desc)
        inter = float(np.logical_and(cm > 0, gt_b > 0).sum())
        uni = float(np.logical_or(cm > 0, gt_b > 0).sum())
        iou = 1.0 if uni <= 0 else inter / uni
        comp_X.append(ft)
        comp_y.append(float(iou))

    return pair_X, pair_y, comp_X, comp_y


def train_ptd_learned_models(cfg: PTDLearnedTrainConfig) -> dict[str, float | int]:
    rng = np.random.default_rng(cfg.random_seed)
    backend = PTDImageBackend(cfg.ptd_root)
    _, entries = load_ptd_entries(cfg.ptd_root)
    split = split_ptd_entries(entries, val_fraction=0.10, split_seed=cfg.random_seed, root=cfg.ptd_root)
    class_to_entries = group_entries_by_class(split.train)
    use_ptd_descriptor = str(cfg.descriptor_mode) in {"ptd_convnext", "hybrid_ptd"}
    encoder = (
        PTDTextureEncoder(PTDEncoderConfig(checkpoint=cfg.ptd_encoder_ckpt, device="cuda"))
        if use_ptd_descriptor
        else None
    )
    shape_paths: list[Path] | None = None
    if str(cfg.synthetic_layout) == "mpeg7_shapes":
        if cfg.shape_mask_root is None:
            raise RuntimeError("synthetic_layout='mpeg7_shapes' requires shape_mask_root.")
        shape_root = Path(cfg.shape_mask_root)
        if not shape_root.exists():
            raise FileNotFoundError(f"Shape mask root does not exist: {shape_root}")
        shape_paths = sorted(
            p
            for pat in ("*.gif", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
            for p in shape_root.rglob(pat)
            if p.is_file()
        )
        if not shape_paths:
            raise RuntimeError(f"No usable shape masks found under {shape_root}")

    n_val = max(50, int(round(cfg.num_samples * cfg.val_fraction)))
    n_train = max(100, int(cfg.num_samples - n_val))

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
    while produced < target_total and attempts < target_total * 3:
        attempts += 1
        sample = _generate_sample(
            backend=backend,
            class_to_entries=class_to_entries,
            encoder=encoder,
            cfg=cfg,
            rng=rng,
            shape_paths=shape_paths,
        )
        if sample is None:
            continue
        pX, py, cX, cy = sample
        is_val = produced >= n_train
        if is_val:
            pair_X_va.extend(pX)
            pair_y_va.extend(py)
            comp_X_va.extend(cX)
            comp_y_va.extend(cy)
        else:
            pair_X_tr.extend(pX)
            pair_y_tr.extend(py)
            comp_X_tr.extend(cX)
            comp_y_tr.extend(cy)
        produced += 1

        if produced % 50 == 0:
            print(
                f"[PTD-learned] generated={produced}/{target_total} "
                f"pair_tr={len(pair_X_tr)} pair_va={len(pair_X_va)} "
                f"comp_tr={len(comp_X_tr)} comp_va={len(comp_X_va)}"
            )

    if len(pair_X_tr) < 200 or len(comp_X_tr) < 200:
        raise RuntimeError(
            "Insufficient PTD synthetic training data for learned consolidation "
            f"(pair_tr={len(pair_X_tr)} comp_tr={len(comp_X_tr)})."
        )

    Xp_tr = np.stack(pair_X_tr, axis=0)
    yp_tr = np.array(pair_y_tr, dtype=np.int32)
    Xp_va = np.stack(pair_X_va, axis=0) if pair_X_va else np.zeros((0, Xp_tr.shape[1]), dtype=np.float32)
    yp_va = np.array(pair_y_va, dtype=np.int32)

    pair_model = RandomForestClassifier(
        n_estimators=int(cfg.pair_n_estimators),
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(cfg.random_seed),
    )
    pair_model.fit(Xp_tr, yp_tr)

    if len(Xp_va) > 0 and len(np.unique(yp_va)) > 1:
        p = pair_model.predict_proba(Xp_va)[:, 1]
        best_t = 0.50
        best_f1 = -1.0
        for t in np.linspace(0.30, 0.80, 26):
            yhat = (p >= t).astype(np.int32)
            f1 = float(f1_score(yp_va, yhat))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        pair_auc = float(roc_auc_score(yp_va, p))
    else:
        best_t = 0.50
        best_f1 = 0.0
        pair_auc = 0.0

    Xc_tr = np.stack(comp_X_tr, axis=0)
    yc_tr = np.array(comp_y_tr, dtype=np.float32)
    Xc_va = np.stack(comp_X_va, axis=0) if comp_X_va else np.zeros((0, Xc_tr.shape[1]), dtype=np.float32)
    yc_va = np.array(comp_y_va, dtype=np.float32)

    score_model = RandomForestRegressor(
        n_estimators=int(cfg.score_n_estimators),
        max_depth=24,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=int(cfg.random_seed),
    )
    score_model.fit(Xc_tr, yc_tr)

    if len(Xc_va) > 0:
        pred_va = score_model.predict(Xc_va)
        score_mae = float(mean_absolute_error(yc_va, pred_va))
    else:
        score_mae = 0.0

    cfg.out_bundle.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair_model": pair_model,
        "score_model": score_model,
        "pair_threshold": float(best_t),
        "adjacency_dilation": int(cfg.adjacency_dilation),
        "min_area": int(cfg.min_area),
        "feature_version": "ptd_learned_v2",
        "descriptor_mode": str(cfg.descriptor_mode),
    }
    with cfg.out_bundle.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "descriptor_mode": str(cfg.descriptor_mode),
        "num_samples_generated": int(produced),
        "synthetic_layout": str(cfg.synthetic_layout),
        "pair_train_examples": int(len(pair_X_tr)),
        "pair_val_examples": int(len(pair_X_va)),
        "comp_train_examples": int(len(comp_X_tr)),
        "comp_val_examples": int(len(comp_X_va)),
        "pair_val_auc": float(pair_auc),
        "pair_val_f1_best": float(best_f1),
        "pair_threshold": float(best_t),
        "score_val_mae": float(score_mae),
        "shape_mask_root": "" if cfg.shape_mask_root is None else str(cfg.shape_mask_root),
        "shape_mask_count": 0 if shape_paths is None else int(len(shape_paths)),
    }
    cfg.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    import json

    cfg.out_metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class PTDLearnedMergeScorer:
    def __init__(self, bundle_path: Path):
        with Path(bundle_path).open("rb") as f:
            payload = pickle.load(f)
        self.pair_model = payload["pair_model"]
        self.score_model = payload["score_model"]
        self.pair_threshold = float(payload.get("pair_threshold", 0.50))
        self.adjacency_dilation = int(payload.get("adjacency_dilation", 3))
        self.min_area = int(payload.get("min_area", 24))

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

        grad = _grad_map(image_rgb)
        uf = _UnionFind(n)
        decisions: list[EdgeDecision] = []
        pair_meta: list[tuple[int, int, float, float, float]] = []
        pair_feat: list[np.ndarray] = []

        for i in range(n):
            for j in range(i + 1, n):
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
                pair_meta.append((i, j, float(tex), float(bweak), float(hetero)))
                pair_feat.append(feat)

        if pair_feat:
            pf = np.stack(pair_feat, axis=0)
            probs = self.pair_model.predict_proba(pf)[:, 1]
        else:
            probs = np.zeros((0,), dtype=np.float32)

        for k, (i, j, tex, bweak, hetero) in enumerate(pair_meta):
            prob = float(probs[k])
            do_merge = prob >= self.pair_threshold
            if do_merge:
                uf.union(i, j)
            decisions.append(
                EdgeDecision(
                    i=i,
                    j=j,
                    score=prob,
                    texture_sim=tex,
                    boundary_weakness=bweak,
                    hetero_penalty=hetero,
                    merged=bool(do_merge),
                )
            )

        groups: dict[int, list[int]] = {}
        for i in range(n):
            r = uf.find(i)
            groups.setdefault(r, []).append(i)

        comp_indices = list(groups.values())
        comp_masks: list[np.ndarray] = []
        for idxs in comp_indices:
            m = np.zeros_like(proposals[0], dtype=np.uint8)
            for k in idxs:
                m = np.logical_or(m, proposals[k] > 0)
            comp_masks.append(m.astype(np.uint8))
        return comp_masks, comp_indices, decisions

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
        feats: list[np.ndarray] = []
        for cm in components:
            feats.append(
                _component_features(
                comp_mask=cm,
                feature_map=feature_map,
                grad=grad,
                proposals=proposals,
                descriptors=descriptors,
            )
            )
        if not feats:
            return []
        arr = np.stack(feats, axis=0)
        pred = self.score_model.predict(arr).astype(np.float32)
        return [float(x) for x in pred]
