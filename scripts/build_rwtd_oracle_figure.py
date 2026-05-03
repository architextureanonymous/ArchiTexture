#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORACLE_JSON = ROOT / "reports" / "reviewer_oracles_rwtd_full256" / "rwtd_proposal_oracles_summary.json"
BASELINE_JSON = ROOT / "reports" / "reviewer_generic_bank_baselines_rwtd" / "rwtd_generic_baselines_summary.json"
OUT_PATH = ROOT.parent / "paper_neurips_unified" / "figures" / "fig_rwtd_oracle_decomposition.png"


def _by_method(rows: list[dict]) -> dict[str, dict]:
    return {row["method"]: row for row in rows}


def main() -> None:
    oracle = json.loads(ORACLE_JSON.read_text())
    baselines = json.loads(BASELINE_JSON.read_text())
    om = _by_method(oracle["methods"])
    bm = _by_method(baselines["methods"])

    methods = [
        ("TextureSAM 0.3", 0.4684171525279216, 0.616292078259886, "#8f8f8f"),
        ("Medoid single", bm["proposal_medoid_single"]["common253_miou"], bm["proposal_medoid_single"]["common253_ari"], "#b6b6b6"),
        ("Spectral split", bm["spectral_bipartition"]["common253_miou"], bm["spectral_bipartition"]["common253_ari"], "#9cc2e5"),
        ("Core only", om["core_v9"]["common253_miou"], om["core_v9"]["common253_ari"], "#7aa6d8"),
        ("ArchiTexture", om["final_release"]["common253_miou"], om["final_release"]["common253_ari"], "#f39c34"),
        ("Single oracle", om["single_proposal_oracle"]["common253_miou"], om["single_proposal_oracle"]["common253_ari"], "#4daf7c"),
        ("Top-k union oracle", om["topk_union_oracle"]["common253_miou"], om["topk_union_oracle"]["common253_ari"], "#57b88b"),
        ("Rescue oracle", om["rescue_candidate_oracle"]["common253_miou"], om["rescue_candidate_oracle"]["common253_ari"], "#2c9c69"),
        ("Bank upper bound", om["bank_upper_bound_oracle"]["common253_miou"], om["bank_upper_bound_oracle"]["common253_ari"], "#176c4a"),
    ]

    labels = [m[0] for m in methods]
    miou = np.asarray([m[1] for m in methods], dtype=float)
    ari = np.asarray([m[2] for m in methods], dtype=float)
    colors = [m[3] for m in methods]
    y = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    metrics = [("mIoU", miou), ("ARI", ari)]
    for ax, (title, vals) in zip(axes, metrics):
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.25, 0.90)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel(title)
        for yy, val in zip(y, vals):
            ax.text(val + 0.008, yy, f"{val:.3f}", va="center", fontsize=8)

    axes[0].set_title("RWTD common-253: overlap quality")
    axes[1].set_title("RWTD common-253: partition coherence")
    fig.suptitle("Frozen-bank recoverability on RWTD", fontsize=14, y=1.02)

    out_delta_miou = om["bank_upper_bound_oracle"]["common253_miou"] - om["final_release"]["common253_miou"]
    out_delta_ari = om["bank_upper_bound_oracle"]["common253_ari"] - om["final_release"]["common253_ari"]
    fig.text(
        0.5,
        -0.02,
        (
            f"Bank upper bound over final system: +{out_delta_miou:.3f} mIoU, +{out_delta_ari:.3f} ARI. "
            "A single best frozen proposal already exceeds the learned system on both metrics."
        ),
        ha="center",
        fontsize=9,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
