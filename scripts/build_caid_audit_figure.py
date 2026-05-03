#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_stld_audit_figure import (
    PANEL_COLORS,
    _build_cfg_from_summary,
    _build_compatibility_graph_panel,
    _build_montage_panel,
    _diff_panel,
    _draw_text,
    _load_mask,
    _load_rgb,
    _overlay_boundary,
    _overlay_mask,
    _proposal_support_panel,
    _resize_mask,
    _resize_rgb,
)
from texturesam_v2.consolidator import TextureSAMV2Consolidator, _postprocess
from texturesam_v2.features import compute_texture_feature_map
from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig


DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments" / "caid_eval_20260313" / "full_0p3"
DEFAULT_OUT = ROOT / "TextureSum2_paper" / "figures" / "fig_caid_audit.png"
DEFAULT_IMAGE_ID = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Figure-3-style CAID audit figure.")
    p.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p.add_argument("--image-id", type=int, default=DEFAULT_IMAGE_ID)
    p.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    p.add_argument("--panel-width", type=int, default=420)
    p.add_argument("--panel-height", type=int, default=310)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    summary_json = experiment_root / "stageA_0p3" / "strict_ptd_learned" / "summary.json"
    stagea_summary = json.loads((experiment_root / "stageA_0p3" / "stageA_learned_only_summary.json").read_text())
    benchmark_root = Path(stagea_summary["dataset_root"])
    prompt_masks_root = Path(stagea_summary["prompt_masks_root"])
    image_id = int(args.image_id)

    image = _load_rgb(benchmark_root / "images" / f"{image_id}.png")
    gt_mask = _load_mask(benchmark_root / "labels" / f"{image_id}.png")

    cfg = _build_cfg_from_summary(summary_json)
    store = PromptMaskProposalStore(prompt_masks_root, ProposalLoadConfig(min_area=cfg.min_area))
    proposals = store.load(image_id, expected_shape=image.shape[:2])
    if len(proposals) < 2:
        raise RuntimeError(f"Selected CAID case {image_id} does not have enough proposals for an audit figure.")

    consolidator = TextureSAMV2Consolidator(cfg)
    feat_map = compute_texture_feature_map(image)
    clean_props = [(p > 0).astype("uint8") for p in proposals if int((p > 0).sum()) >= cfg.min_area]
    if len(clean_props) < 2:
        raise RuntimeError(f"Selected CAID case {image_id} does not have enough clean proposals.")

    desc = consolidator._build_descriptors(image_rgb=image, feat_map=feat_map, proposals=clean_props)
    if consolidator.learned_models is None:
        raise RuntimeError("Expected a learned CAID bundle for the audit figure, but no learned model was loaded.")

    merged_components, comp_indices, edge_decisions = consolidator.learned_models.merge_components(
        image_rgb=image,
        proposals=clean_props,
        descriptors=desc,
        feature_map=feat_map,
    )
    processed_components = [_postprocess(comp, cfg) for comp in merged_components]
    component_scores = consolidator.learned_models.score_components(
        image_rgb=image,
        components=merged_components,
        proposals=clean_props,
        descriptors=desc,
        feature_map=feat_map,
    )
    if not processed_components:
        raise RuntimeError(f"Selected CAID case {image_id} produced no merged components.")
    if len(component_scores) != len(processed_components):
        component_scores = [0.0 for _ in processed_components]

    ranked = sorted(
        zip(processed_components, comp_indices, component_scores, strict=False),
        key=lambda item: float(item[2]),
        reverse=True,
    )
    ranked_components = [item[0].astype("uint8") for item in ranked]
    ranked_indices = [list(item[1]) for item in ranked]
    ranked_scores = [float(item[2]) for item in ranked]
    selected_mask = ranked_components[0]
    runner_up_mask = ranked_components[1] if len(ranked_components) > 1 else selected_mask * 0

    import numpy as np

    union_mask = np.logical_or.reduce([(p > 0) for p in clean_props]).astype("uint8")
    merged_edges = [(int(d.i), int(d.j)) for d in edge_decisions if d.merged]
    proposal_labels = [f"P{idx + 1}" for idx in range(len(clean_props))]
    component_labels = []
    for idx, (members, score) in enumerate(zip(ranked_indices, ranked_scores, strict=False)):
        member_text = ",".join(str(m + 1) for m in members)
        component_labels.append(f"C{idx + 1} [{member_text}] {score:.2f}")

    out_w = int(args.panel_width)
    out_h = int(args.panel_height)
    base = _resize_rgb(image, out_w, out_h)
    gt = _resize_mask(gt_mask, out_w, out_h)
    union = _resize_mask(union_mask, out_w, out_h)
    resized_props = [_resize_mask(p, out_w, out_h) for p in clean_props]
    resized_components = [_resize_mask(c, out_w, out_h) for c in ranked_components]
    resized_selected = _resize_mask(selected_mask, out_w, out_h)
    resized_runner = _resize_mask(runner_up_mask, out_w, out_h)

    panel_input = _overlay_boundary(base, gt)
    panel_gt = _overlay_boundary(_overlay_mask(base, gt, (56, 190, 88), alpha=0.28, thickness=2), gt)
    panel_union = _overlay_boundary(_overlay_mask(base, union, (149, 117, 205), alpha=0.30, thickness=2), gt)
    panel_proposals = _overlay_boundary(
        _build_compatibility_graph_panel(base, resized_props, PANEL_COLORS, proposal_labels, []),
        gt,
    )
    panel_graph = _overlay_boundary(
        _build_compatibility_graph_panel(base, resized_props, PANEL_COLORS, proposal_labels, merged_edges),
        gt,
    )
    panel_candidates = _build_montage_panel(base, resized_components, PANEL_COLORS, component_labels)
    panel_selected = _overlay_boundary(_overlay_mask(base, resized_selected, (244, 140, 35), alpha=0.33, thickness=2), gt)
    panel_runner = _overlay_boundary(_overlay_mask(base, resized_runner, (66, 133, 244), alpha=0.30, thickness=2), gt)
    panel_support = _proposal_support_panel(base, resized_props, gt)
    panel_diff = _diff_panel(base, resized_selected, gt)

    fig, axes = plt.subplots(2, 5, figsize=(18.2, 7.8), dpi=230)
    panels = [
        panel_input,
        panel_gt,
        panel_union,
        panel_proposals,
        panel_graph,
        panel_candidates,
        panel_selected,
        panel_runner,
        panel_support,
        panel_diff,
    ]
    titles = [
        f"1) CAID-{image_id} Input",
        "2) GT Partition",
        "3) Proposal Union",
        "4) Proposal Fragments",
        "5) Compatibility Graph",
        "6) Ranked Components",
        "7) Selected Component",
        "8) Runner-Up Component",
        "9) Proposal Support (Pixel Frequency)",
        "10) Diff Final vs GT",
    ]
    notes = [
        "",
        "",
        "",
        f"{len(clean_props)} frozen proposals",
        f"{len(merged_edges)} kept edges",
        f"{len(ranked_components)} scored components",
        f"C1 score {ranked_scores[0]:.2f}",
        f"C2 score {ranked_scores[1]:.2f}" if len(ranked_scores) > 1 else "no runner-up",
        "",
        "orange=TP red=FP blue=FN",
    ]
    for ax, panel, title, note in zip(axes.flatten(), panels, titles, notes, strict=False):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=6)
        if note:
            _draw_text(ax, note)

    fig.subplots_adjust(left=0.006, right=0.998, top=0.94, bottom=0.03, wspace=0.022, hspace=0.14)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, bbox_inches="tight")
    plt.close(fig)
    print(str(args.out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
