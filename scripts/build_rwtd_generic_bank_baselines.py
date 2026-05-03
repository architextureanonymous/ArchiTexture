#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.features import compute_texture_feature_map, region_descriptor
from texturesam_v2.io_utils import ensure_binary_gt, infer_rwtd_dirs, list_rwtd_images, read_image_rgb, read_mask_raw, write_binary_mask
from texturesam_v2.metrics import rwtd_invariant_metrics
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build stronger generic frozen-bank baselines for RWTD. "
            "These baselines use only the prompt proposal bank plus generic "
            "descriptor/graph heuristics, without the learned ArchiTexture commitment stack."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--baseline-official-root", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--cc-sim-thresh", type=float, default=0.68)
    p.add_argument("--cc-iou-thresh", type=float, default=0.18)
    p.add_argument("--greedy-sim-thresh", type=float, default=0.60)
    return p.parse_args()


def _parse_baseline_ids(mask_root: Path) -> set[int]:
    out: set[int] = set()
    for p in mask_root.glob("mask_*_*.png"):
        try:
            out.add(int(p.stem.split("_")[-1]))
        except Exception:
            continue
    return out


def _run_eval(pred: Path, gt_dir: Path, upstream_root: Path, out_json: Path) -> dict:
    cmd = [
        "python",
        str(ROOT / "scripts" / "eval_upstream_texture_metrics.py"),
        "--pred-folder",
        str(pred),
        "--gt-folder",
        str(gt_dir),
        "--upstream-root",
        str(upstream_root),
        "--out-json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _union_masks(proposals: list[np.ndarray], idxs: list[int]) -> np.ndarray:
    if not proposals or not idxs:
        raise ValueError("Cannot union an empty proposal selection.")
    out = np.zeros_like(proposals[0], dtype=np.uint8)
    for idx in idxs:
        out = np.logical_or(out > 0, proposals[idx] > 0)
    return out.astype(np.uint8)


def _pairwise_iou(proposals: list[np.ndarray]) -> np.ndarray:
    n = len(proposals)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        a = proposals[i] > 0
        for j in range(i + 1, n):
            b = proposals[j] > 0
            inter = float(np.logical_and(a, b).sum())
            uni = float(np.logical_or(a, b).sum())
            sim = 0.0 if uni <= 0 else inter / uni
            out[i, j] = sim
            out[j, i] = sim
    return out


def _pairwise_cosine(descs: list[np.ndarray]) -> np.ndarray:
    n = len(descs)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    D = np.stack(descs, axis=0).astype(np.float32)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + EPS)
    out = (D @ D.T).astype(np.float32)
    np.fill_diagonal(out, 1.0)
    return out


def _centrality(sim: np.ndarray) -> np.ndarray:
    if sim.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return sim.mean(axis=1).astype(np.float32)


def _selection_score(mask: np.ndarray, idxs: list[int], sim: np.ndarray, centrality: np.ndarray) -> float:
    area_frac = float(mask.mean())
    balance = 1.0 - min(abs(area_frac - 0.5) / 0.5, 1.0)
    if len(idxs) >= 2:
        block = sim[np.ix_(idxs, idxs)]
        internal = float((block.sum() - len(idxs)) / max(len(idxs) * (len(idxs) - 1), 1))
    else:
        internal = 0.0
    central = float(np.mean(centrality[idxs])) if idxs else 0.0
    size_penalty = 0.0 if len(idxs) >= 1 else 1.0
    return 0.55 * balance + 0.30 * internal + 0.15 * central - 0.05 * size_penalty


def _pick_best_from_groups(groups: list[list[int]], proposals: list[np.ndarray], sim: np.ndarray, centrality: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if not groups:
        raise ValueError("No proposal groups were supplied.")
    best_mask = _union_masks(proposals, groups[0])
    best_group = groups[0]
    best_score = _selection_score(best_mask, best_group, sim, centrality)
    for group in groups[1:]:
        if not group:
            continue
        mask = _union_masks(proposals, group)
        score = _selection_score(mask, group, sim, centrality)
        if score > best_score:
            best_mask = mask
            best_group = group
            best_score = score
    return best_mask, best_group


def _connected_components(adj: np.ndarray) -> list[list[int]]:
    n = adj.shape[0]
    seen = np.zeros((n,), dtype=bool)
    out: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            nbrs = np.where(adj[cur])[0].tolist()
            for nbr in nbrs:
                if seen[nbr]:
                    continue
                seen[nbr] = True
                stack.append(nbr)
        out.append(sorted(comp))
    return out


def _medoid_single(proposals: list[np.ndarray], sim: np.ndarray, centrality: np.ndarray) -> tuple[np.ndarray, list[int]]:
    groups = [[i] for i in range(len(proposals))]
    return _pick_best_from_groups(groups, proposals, sim, centrality)


def _affinity_cc(
    proposals: list[np.ndarray],
    sim: np.ndarray,
    iou: np.ndarray,
    centrality: np.ndarray,
    sim_thresh: float,
    iou_thresh: float,
) -> tuple[np.ndarray, list[int]]:
    adj = (sim >= sim_thresh) | (iou >= iou_thresh)
    np.fill_diagonal(adj, True)
    comps = _connected_components(adj)
    return _pick_best_from_groups(comps, proposals, sim, centrality)


def _spectral_partition(proposals: list[np.ndarray], sim: np.ndarray, centrality: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if len(proposals) < 2:
        return _medoid_single(proposals, sim, centrality)
    aff = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    aff = aff + np.eye(len(proposals), dtype=np.float32) * 1e-3
    try:
        labels = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        ).fit_predict(aff)
    except Exception:
        return _medoid_single(proposals, sim, centrality)
    groups = [np.where(labels == cid)[0].tolist() for cid in sorted(set(labels.tolist()))]
    groups = [g for g in groups if g]
    return _pick_best_from_groups(groups, proposals, sim, centrality)


def _agglomerative_partition(proposals: list[np.ndarray], sim: np.ndarray, centrality: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if len(proposals) < 2:
        return _medoid_single(proposals, sim, centrality)
    dist = np.clip(1.0 - np.clip(sim, -1.0, 1.0), 0.0, 2.0)
    try:
        model = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="average")
    try:
        labels = model.fit_predict(dist)
    except Exception:
        return _medoid_single(proposals, sim, centrality)
    groups = [np.where(labels == cid)[0].tolist() for cid in sorted(set(labels.tolist()))]
    groups = [g for g in groups if g]
    return _pick_best_from_groups(groups, proposals, sim, centrality)


def _greedy_merge(
    proposals: list[np.ndarray],
    sim: np.ndarray,
    centrality: np.ndarray,
    sim_thresh: float,
) -> tuple[np.ndarray, list[int]]:
    n = len(proposals)
    if n == 0:
        raise ValueError("Cannot run greedy merge without proposals.")
    start = int(np.argmax(centrality))
    current = [start]
    best = current[:]
    best_mask = _union_masks(proposals, best)
    best_score = _selection_score(best_mask, best, sim, centrality)
    order = np.argsort(sim[start])[::-1].tolist()
    for idx in order:
        if idx in current:
            continue
        mean_sim = float(np.mean(sim[idx, current]))
        if mean_sim < sim_thresh:
            continue
        candidate = current + [idx]
        mask = _union_masks(proposals, candidate)
        score = _selection_score(mask, candidate, sim, centrality)
        if score > best_score + 1e-4:
            current = candidate
            best = candidate[:]
            best_mask = mask
            best_score = score
    return best_mask, best


def main() -> None:
    args = parse_args()
    image_dir, label_dir = infer_rwtd_dirs(args.rwtd_root)
    images = list_rwtd_images(image_dir)
    gt_dir = args.upstream_root / "Kaust256" / "labeles"
    store = PromptMaskProposalStore(args.prompt_masks_root, ProposalLoadConfig(min_area=int(args.min_area)))
    baseline_ids = _parse_baseline_ids(args.baseline_official_root)

    methods = [
        "proposal_medoid_single",
        "affinity_threshold_cc",
        "spectral_bipartition",
        "agglomerative_bipartition",
        "greedy_similarity_merge",
    ]
    method_dirs: dict[str, Path] = {}
    for method in methods:
        full_dir = args.out_root / method / "official_export"
        common_dir = args.out_root / method / "official_export_common253"
        if full_dir.exists():
            shutil.rmtree(full_dir)
        if common_dir.exists():
            shutil.rmtree(common_dir)
        full_dir.mkdir(parents=True, exist_ok=True)
        common_dir.mkdir(parents=True, exist_ok=True)
        method_dirs[method] = full_dir

    per_image_rows: list[dict[str, object]] = []
    for i, image_path in enumerate(images, start=1):
        image_id = int(image_path.stem)
        image_rgb = read_image_rgb(image_path)
        gt = ensure_binary_gt(read_mask_raw(label_dir / f"{image_id}.png"), strict=False)
        shape = gt.shape
        proposals = store.load(image_id, expected_shape=shape)

        if not proposals:
            zero = np.zeros(shape, dtype=np.uint8)
            preds = {method: zero for method in methods}
            row: dict[str, object] = {"image_id": image_id, "proposal_count": 0}
            for method in methods:
                met = rwtd_invariant_metrics(zero, gt)
                row[f"{method}_iou"] = float(met.iou)
                row[f"{method}_ari"] = float(met.ari)
            per_image_rows.append(row)
            for method, mask in preds.items():
                write_binary_mask(method_dirs[method] / f"mask_0_{image_id}.png", mask)
                if image_id in baseline_ids:
                    write_binary_mask(args.out_root / method / "official_export_common253" / f"mask_0_{image_id}.png", mask)
            continue

        feat_map = compute_texture_feature_map(image_rgb)
        descs = [region_descriptor(feat_map, m) for m in proposals]
        sim = _pairwise_cosine(descs)
        iou = _pairwise_iou(proposals)
        centrality = _centrality(sim)

        preds_with_groups = {
            "proposal_medoid_single": _medoid_single(proposals, sim, centrality),
            "affinity_threshold_cc": _affinity_cc(
                proposals,
                sim,
                iou,
                centrality,
                sim_thresh=float(args.cc_sim_thresh),
                iou_thresh=float(args.cc_iou_thresh),
            ),
            "spectral_bipartition": _spectral_partition(proposals, sim, centrality),
            "agglomerative_bipartition": _agglomerative_partition(proposals, sim, centrality),
            "greedy_similarity_merge": _greedy_merge(
                proposals,
                sim,
                centrality,
                sim_thresh=float(args.greedy_sim_thresh),
            ),
        }

        row = {
            "image_id": image_id,
            "proposal_count": len(proposals),
            "centrality_mean": float(centrality.mean()),
            "centrality_max": float(centrality.max()),
        }
        for method, (mask, picked) in preds_with_groups.items():
            met = rwtd_invariant_metrics(mask, gt)
            row[f"{method}_num_selected"] = len(picked)
            row[f"{method}_area_frac"] = float(mask.mean())
            row[f"{method}_iou"] = float(met.iou)
            row[f"{method}_ari"] = float(met.ari)
            write_binary_mask(method_dirs[method] / f"mask_0_{image_id}.png", mask)
            if image_id in baseline_ids:
                write_binary_mask(args.out_root / method / "official_export_common253" / f"mask_0_{image_id}.png", mask)

        per_image_rows.append(row)

        if i % 32 == 0:
            print(f"[rwtd-generic-baselines] processed {i}/{len(images)} images", flush=True)

    eval_rows: list[dict[str, object]] = []
    for method in methods:
        full_json = args.out_root / method / "official_eval_full256.json"
        common_json = args.out_root / method / "official_eval_common253.json"
        full = _run_eval(args.out_root / method / "official_export", gt_dir, args.upstream_root, full_json)
        common = _run_eval(args.out_root / method / "official_export_common253", gt_dir, args.upstream_root, common_json)
        eval_rows.append(
            {
                "method": method,
                "full256_miou": float(full["noagg_official"]["overall_average_iou"]),
                "full256_ari": float(full["noagg_official"]["overall_average_rand_index"]),
                "full256_num_pred_image_ids": int(full["noagg_official"]["num_pred_image_ids"]),
                "common253_miou": float(common["noagg_official"]["overall_average_iou"]),
                "common253_ari": float(common["noagg_official"]["overall_average_rand_index"]),
                "common253_num_pred_image_ids": int(common["noagg_official"]["num_pred_image_ids"]),
            }
        )

    per_image_csv = args.out_root / "rwtd_generic_baselines_per_image.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_csv = args.out_root / "rwtd_generic_baselines_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "full256_miou",
                "full256_ari",
                "full256_num_pred_image_ids",
                "common253_miou",
                "common253_ari",
                "common253_num_pred_image_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    payload = {
        "num_images": len(images),
        "rwtd_root": str(args.rwtd_root),
        "prompt_masks_root": str(args.prompt_masks_root),
        "summary_csv": str(summary_csv.resolve()),
        "per_image_csv": str(per_image_csv.resolve()),
        "methods": eval_rows,
    }
    (args.out_root / "rwtd_generic_baselines_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
