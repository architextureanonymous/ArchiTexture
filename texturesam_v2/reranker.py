from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .features import compute_texture_feature_map, mean_feature, region_variance
from .io_utils import (
    ensure_binary,
    ensure_binary_gt,
    ensure_dir,
    infer_rwtd_dirs,
    list_rwtd_images,
    read_image_rgb,
    read_mask_raw,
    write_binary_mask,
)
from .metrics import rwtd_invariant_metrics

PROMPT_MASK_RE = re.compile(r"^rwtd_(?P<image_id>\d+)_p\d+_m\d+\.png$")


@dataclass(frozen=True)
class RerankerConfig:
    proposal_roots: list[Path]
    out_dir: Path
    rwtd_root: Path
    max_images: int | None = None
    cv_folds: int = 5
    mode: str = "cv"  # cv | in_sample
    n_estimators: int = 700
    max_depth: int = 20
    min_samples_leaf: int = 2
    random_seed: int = 1337
    apply_refine: bool = False
    refine_iters: int = 2
    refine_min_area: int = 50


@dataclass(frozen=True)
class CandidateRef:
    image_id: str
    image_num: int
    run_idx: int
    source_name: str
    mask: np.ndarray


def _collect_proposals_for_image(image_num: int, proposal_roots: list[Path], shape: tuple[int, int]) -> list[CandidateRef]:
    out: list[CandidateRef] = []
    image_id = f"rwtd_{image_num}"

    for run_idx, root in enumerate(proposal_roots):
        files = sorted(root.glob(f"{image_id}_p*_m*.png"))
        masks_run: list[np.ndarray] = []

        for p in files:
            m = PROMPT_MASK_RE.match(p.name)
            if m is None:
                continue
            arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            b = ensure_binary(arr)
            if b.shape != shape:
                continue
            masks_run.append(b)
            out.append(CandidateRef(image_id=image_id, image_num=image_num, run_idx=run_idx, source_name=p.name, mask=b))

        if masks_run:
            u = np.zeros(shape, dtype=np.uint8)
            for m in masks_run:
                u = np.logical_or(u, m > 0)
            out.append(
                CandidateRef(
                    image_id=image_id,
                    image_num=image_num,
                    run_idx=run_idx,
                    source_name=f"run{run_idx}_union",
                    mask=u.astype(np.uint8),
                )
            )

    dedup: dict[bytes, CandidateRef] = {}
    for c in out:
        key = np.packbits(c.mask.astype(np.uint8), axis=None).tobytes()
        if key not in dedup:
            dedup[key] = c
    return list(dedup.values())


def _compactness(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 1e6
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = 0.0
    for c in cnts:
        perim += float(cv2.arcLength(c, True))
    return float((perim * perim) / (4.0 * math.pi * max(area, 1)))


def _holes(mask: np.ndarray) -> int:
    m = (mask > 0).astype(np.uint8)
    inv = 1 - m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return 0

    border = set(np.unique(lbl[0, :]).tolist())
    border.update(np.unique(lbl[-1, :]).tolist())
    border.update(np.unique(lbl[:, 0]).tolist())
    border.update(np.unique(lbl[:, -1]).tolist())

    holes = 0
    for c in range(1, n):
        if c in border:
            continue
        if int(stats[c, cv2.CC_STAT_AREA]) > 0:
            holes += 1
    return holes


def _boundary_edge_strength(mask: np.ndarray, edge_map: np.ndarray) -> float:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bd = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, k).astype(bool)
    if not np.any(bd):
        return 0.0
    return float(edge_map[bd].mean())


def _build_feature_vector(
    mask: np.ndarray,
    run_idx: int,
    n_runs: int,
    gray: np.ndarray,
    edge_map: np.ndarray,
    texture_map: np.ndarray,
    consensus_mean_iou: float,
    consensus_max_iou: float,
) -> np.ndarray:
    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    area_ratio = float(area / max(h * w, 1))

    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cc_count = max(0, n_cc - 1)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if cc_count > 0 else 0
    largest_ratio = float(largest / max(area, 1))

    compact = _compactness(m)
    holes = _holes(m)

    b_edge = _boundary_edge_strength(m, edge_map)
    in_gray = float(gray[m > 0].mean()) if area > 0 else 0.0
    out_gray = float(gray[m == 0].mean()) if area < h * w else 0.0
    gray_delta = abs(in_gray - out_gray)

    tex_delta = float(np.linalg.norm(mean_feature(texture_map, m) - mean_feature(texture_map, 1 - m)))
    tex_var = float(region_variance(texture_map, m))

    run_one_hot = np.zeros(n_runs + 1, dtype=np.float32)
    idx = run_idx if 0 <= run_idx < n_runs else n_runs
    run_one_hot[idx] = 1.0

    base = np.array(
        [
            area_ratio,
            float(cc_count),
            largest_ratio,
            compact,
            float(holes),
            b_edge,
            gray_delta,
            tex_delta,
            tex_var,
            consensus_mean_iou,
            consensus_max_iou,
        ],
        dtype=np.float32,
    )

    return np.concatenate([base, run_one_hot], axis=0).astype(np.float32)


def _pairwise_iou_matrix(masks: list[np.ndarray]) -> np.ndarray:
    n = len(masks)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
        a = masks[i].astype(bool)
        for j in range(i + 1, n):
            b = masks[j].astype(bool)
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            v = 1.0 if uni == 0 else float(inter / uni)
            out[i, j] = v
            out[j, i] = v
    return out


def _grabcut_refine(mask: np.ndarray, image_rgb: np.ndarray, iters: int, min_area: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < min_area or int(m.sum()) > m.size - min_area:
        return m

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_fg = cv2.erode(m, k, iterations=1)
    sure_bg = cv2.erode(1 - m, k, iterations=1)

    gc = np.full(m.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc[sure_bg > 0] = cv2.GC_BGD
    gc[sure_fg > 0] = cv2.GC_FGD

    unknown = np.logical_and(sure_fg == 0, sure_bg == 0)
    gc[unknown] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(img_bgr, gc, None, bgd_model, fgd_model, int(iters), cv2.GC_INIT_WITH_MASK)
    except Exception:
        return m

    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 1:
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        out = (lbl == largest).astype(np.uint8)
    return out


def run_multibank_reranker(cfg: RerankerConfig) -> dict[str, float | int | str]:
    image_dir, label_dir = infer_rwtd_dirs(cfg.rwtd_root)
    image_paths = list_rwtd_images(image_dir)
    if cfg.max_images is not None:
        image_paths = image_paths[: cfg.max_images]
    image_path_by_num = {int(p.stem.split("_")[-1]): p for p in image_paths}

    proposal_roots = [Path(p) for p in cfg.proposal_roots]
    n_runs = len(proposal_roots)

    X: list[np.ndarray] = []
    y_iou: list[float] = []
    y_ari: list[float] = []
    groups: list[int] = []
    refs: list[CandidateRef] = []

    for image_path in image_paths:
        image_id = image_path.stem
        image_num = int(image_id.split("_")[-1])

        image_rgb = read_image_rgb(image_path)
        gt = read_mask_raw(label_dir / f"{image_id}.png")
        gt_b = ensure_binary_gt(gt, source_name=str(label_dir / f"{image_id}.png"))

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        edge_map = (cv2.Canny((gray * 255.0).astype(np.uint8), 80, 160).astype(np.float32) / 255.0)
        texture_map = compute_texture_feature_map(image_rgb)

        cands = _collect_proposals_for_image(image_num, proposal_roots, image_rgb.shape[:2])
        if not cands:
            cands = [CandidateRef(image_id=image_id, image_num=image_num, run_idx=n_runs, source_name="empty", mask=np.zeros(image_rgb.shape[:2], dtype=np.uint8))]

        cand_masks = [c.mask for c in cands]
        iou_mat = _pairwise_iou_matrix(cand_masks)

        for idx, cand in enumerate(cands):
            # Exclude self for consensus statistics.
            if len(cands) > 1:
                vals = np.concatenate([iou_mat[idx, :idx], iou_mat[idx, idx + 1 :]], axis=0)
                cons_mean = float(vals.mean())
                cons_max = float(vals.max())
            else:
                cons_mean = 0.0
                cons_max = 0.0

            feat = _build_feature_vector(
                mask=cand.mask,
                run_idx=cand.run_idx,
                n_runs=n_runs,
                gray=gray,
                edge_map=edge_map,
                texture_map=texture_map,
                consensus_mean_iou=cons_mean,
                consensus_max_iou=cons_max,
            )

            met = rwtd_invariant_metrics(cand.mask, gt_b)
            X.append(feat)
            y_iou.append(float(met.iou))
            y_ari.append(float(met.ari))
            groups.append(image_num)
            refs.append(cand)

    X_np = np.stack(X, axis=0)
    y_np = np.array(y_iou, dtype=np.float32)
    groups_np = np.array(groups, dtype=np.int32)

    pred = np.zeros_like(y_np)

    if cfg.mode == "cv":
        gkf = GroupKFold(n_splits=cfg.cv_folds)
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_np, y_np, groups_np), 1):
            model = RandomForestRegressor(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.random_seed + fold,
                n_jobs=-1,
            )
            model.fit(X_np[tr_idx], y_np[tr_idx])
            pred[te_idx] = model.predict(X_np[te_idx]).astype(np.float32)
    elif cfg.mode == "in_sample":
        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_seed,
            n_jobs=-1,
        )
        model.fit(X_np, y_np)
        pred = model.predict(X_np).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    by_image: dict[int, int] = {}
    for idx, g in enumerate(groups_np.tolist()):
        if g not in by_image or pred[idx] > pred[by_image[g]]:
            by_image[g] = idx

    out_dir = ensure_dir(cfg.out_dir)
    masks_dir = ensure_dir(out_dir / "masks")

    rows = []
    sel_iou = []
    sel_ari = []
    for image_num, idx in sorted(by_image.items()):
        cand = refs[idx]
        image_id = f"rwtd_{image_num}"
        gt_b = ensure_binary_gt(
            read_mask_raw(label_dir / f"{image_id}.png"),
            source_name=str(label_dir / f"{image_id}.png"),
        )
        image_rgb = read_image_rgb(image_path_by_num[image_num])
        out_mask = cand.mask
        if cfg.apply_refine:
            out_mask = _grabcut_refine(
                mask=cand.mask,
                image_rgb=image_rgb,
                iters=cfg.refine_iters,
                min_area=cfg.refine_min_area,
            )

        met = rwtd_invariant_metrics(out_mask, gt_b)
        sel_iou.append(float(met.iou))
        sel_ari.append(float(met.ari))

        write_binary_mask(masks_dir / f"{image_id}.png", out_mask)

        rows.append(
            {
                "image_id": image_id,
                "selected_source": cand.source_name,
                "selected_run_idx": cand.run_idx,
                "predicted_iou": float(pred[idx]),
                "actual_iou": float(met.iou),
                "actual_ari": float(met.ari),
            }
        )

    import csv
    import json

    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_images": int(len(rows)),
        "rwtd_root": str(cfg.rwtd_root),
        "proposal_roots": [str(p) for p in proposal_roots],
        "mode": cfg.mode,
        "cv_folds": int(cfg.cv_folds),
        "n_estimators": int(cfg.n_estimators),
        "max_depth": int(cfg.max_depth),
        "min_samples_leaf": int(cfg.min_samples_leaf),
        "apply_refine": bool(cfg.apply_refine),
        "refine_iters": int(cfg.refine_iters),
        "refine_min_area": int(cfg.refine_min_area),
        "selected_miou": float(np.mean(sel_iou)) if sel_iou else 0.0,
        "selected_ari": float(np.mean(sel_ari)) if sel_ari else 0.0,
        "num_candidates_total": int(len(refs)),
        "avg_candidates_per_image": float(len(refs) / max(len(rows), 1)),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary
