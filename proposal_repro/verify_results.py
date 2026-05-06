#!/usr/bin/env python3
"""
Verify ArchiTexture proposal-space route results against paper numbers.

Runs the official evaluators on committed prediction masks and prints a
comparison table. Writes verified_results.json on success.

- RWTD: eval_upstream_texture_metrics.py (wraps eval_no_agg_masks.py from
  TextureSAM_upstream_20260303); per-GT-instance IoU/ARI on full-256.
- STLD: eval_stld_direct.py; direct-foreground IoU/ARI on all-200.
- ControlNet bridge: eval_binary_partition_maskbank.py; partition-invariant
  IoU/ARI on the 1742-image synthetic benchmark.
- DeTexture ADE20K: eval_two_mask_partition_maskbank.py; two-mask
  partition-invariant IoU/ARI on the 56-image curated validation set.

Usage:
    python proposal_repro/verify_results.py

Expected (paper Table 1):
    RWTD full-256             mIoU=0.4611  ARI=0.6966
    STLD covered-182          mIoU=0.7195  ARI=0.7791
    ControlNet (invariant)    mIoU=0.6803  ARI=0.6039
    DeTexture ADE20K (invar.) mIoU=0.5008  ARI=0.3675
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "ArchiTexture_NeurIPS_ED_submission_20260502"
BUNDLE     = SUBMISSION / "proposal-space-route"
UPSTREAM   = SUBMISSION / "TextureSAM_upstream_20260303"

# ── RWTD paths ────────────────────────────────────────────────────────────────
RWTD_PRED         = BUNDLE / "reports/release_swinb_full256_audit/official_export"
RWTD_GT           = UPSTREAM / "Kaust256/labeles"
RWTD_EVAL_SCRIPT  = BUNDLE / "scripts/eval_upstream_texture_metrics.py"

# ── STLD paths ────────────────────────────────────────────────────────────────
STLD_GALLERY      = BUNDLE / "experiments/khan_synthetic_gallery_20260312"
STLD_BENCH        = STLD_GALLERY / "benchmark"
STLD_PRED         = STLD_GALLERY / "eval/strict_ptd_learned/masks"
STLD_UNION        = STLD_GALLERY / "union_masks"
STLD_HANDCRAFTED  = STLD_GALLERY / "eval/strict_handcrafted/masks"
STLD_HEURISTIC    = STLD_GALLERY / "eval/strict_ptd_heuristic/masks"
STLD_EVAL_SCRIPT  = BUNDLE / "scripts/eval_stld_direct.py"

# ── ControlNet bridge paths ───────────────────────────────────────────────────
CONTROLNET_DATA        = BUNDLE / "data/synthetic_texture_perlin_stitched_recovered/synthetic_texture_perlin_stitched"
CONTROLNET_PRED        = BUNDLE / "experiments/perlin_controlnet_eval_20260312/full_0p3/stageA_0p3/strict_ptd_learned/masks"
CONTROLNET_EVAL_SCRIPT = BUNDLE / "scripts/eval_binary_partition_maskbank.py"

# ── DeTexture ADE20K paths ────────────────────────────────────────────────────
DETEXTURE_EVAL_ROOT    = BUNDLE / "experiments/detexture_ade20k_eval_20260317"
DETEXTURE_BENCH        = DETEXTURE_EVAL_ROOT / "benchmarks/detexture_validation_refined"
DETEXTURE_PRED         = DETEXTURE_EVAL_ROOT / "full_validation/stageA_0p3/strict_ptd_learned/masks"
DETEXTURE_EVAL_SCRIPT  = BUNDLE / "scripts/eval_two_mask_partition_maskbank.py"

# ── paper targets ─────────────────────────────────────────────────────────────
PAPER = {
    "rwtd_full256":         {"miou": 0.46110414433944885, "ari": 0.6965752621397988},
    "stld_covered182":      {"miou": 0.7194617553453895,  "ari": 0.7791098448761597},
    "controlnet_invariant": {"miou": 0.6803454871408711,  "ari": 0.6038941422804043},
    "detexture_invariant":  {"miou": 0.500760085576693,   "ari": 0.36753242099773936},
}

TOLERANCE = 1e-4


def _check_paths() -> None:
    required = [
        RWTD_PRED, RWTD_GT, RWTD_EVAL_SCRIPT,
        STLD_BENCH, STLD_PRED, STLD_UNION,
        STLD_HANDCRAFTED, STLD_HEURISTIC, STLD_EVAL_SCRIPT,
        CONTROLNET_DATA, CONTROLNET_PRED, CONTROLNET_EVAL_SCRIPT,
        DETEXTURE_BENCH, DETEXTURE_PRED, DETEXTURE_EVAL_SCRIPT,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        if not SUBMISSION.exists():
            print(
                "ERROR: The OpenReview supplementary bundle is not present.\n\n"
                "  Expected at: ArchiTexture_NeurIPS_ED_submission_20260502/\n\n"
                "  This verifier re-runs the proposal-space evaluators against committed\n"
                "  prediction masks and requires the supplementary ZIP from OpenReview.\n\n"
                "  To run without the bundle, use the feature-clustering route instead:\n"
                "    python scripts/feature_clustering/repro_table_1.py \\\n"
                "      --rwtd-root datasets/RWTD --stld-root datasets/STLD \\\n"
                "      --output-root outputs/repro_table_1\n",
                file=sys.stderr,
            )
        else:
            for p in missing:
                print(f"MISSING: {p}", file=sys.stderr)
        sys.exit(1)


# ── RWTD evaluator ────────────────────────────────────────────────────────────
def _rwtd_eval() -> dict:
    """Call eval_upstream_texture_metrics.py (wraps eval_no_agg_masks.py) on
    committed RWTD prediction masks."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BUNDLE) + os.pathsep + env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "rwtd_eval.json"
        result = subprocess.run(
            [
                sys.executable, str(RWTD_EVAL_SCRIPT),
                "--pred-folder",   str(RWTD_PRED),
                "--gt-folder",     str(RWTD_GT),
                "--upstream-root", str(UPSTREAM),
                "--out-json",      str(out_json),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("eval_upstream_texture_metrics.py failed")
        summary = json.loads(out_json.read_text())

    noagg = summary["noagg_official"]
    return {
        "num_gt_instances_evaluated": noagg["num_gt_instances_evaluated"],
        "miou": noagg["overall_average_iou"],
        "ari":  noagg["overall_average_rand_index"],
    }


# ── STLD evaluator ────────────────────────────────────────────────────────────
def _stld_eval() -> dict:
    """Call eval_stld_direct.py (direct-foreground IoU/ARI on all-200 STLD).

    Scores four methods; we report the 'architexture' all-200 row.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BUNDLE) + os.pathsep + env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                sys.executable, str(STLD_EVAL_SCRIPT),
                "--benchmark-root",      str(STLD_BENCH),
                "--proposal-union-root", str(STLD_UNION),
                "--handcrafted-root",    str(STLD_HANDCRAFTED),
                "--heuristic-root",      str(STLD_HEURISTIC),
                "--learned-root",        str(STLD_PRED),
                "--out-root",            tmp,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("eval_stld_direct.py failed")
        summary = json.loads(result.stdout)

    arch = summary["methods"]["architexture"]
    return {
        "num_images": summary["num_images"],
        "n_covered":  summary["proposal_union_covered"],
        "covered": {"miou": arch["covered"]["miou"], "ari": arch["covered"]["ari"]},
        "all":     {"miou": arch["all"]["miou"],     "ari": arch["all"]["ari"]},
    }


# ── ControlNet bridge evaluator ───────────────────────────────────────────────
def _controlnet_eval() -> dict:
    """Call eval_binary_partition_maskbank.py on committed ControlNet masks.

    The benchmark data uses zero-padded image names (000000.png …) under
    images/ and regions/.  The evaluator expects labels/{int}.png, so we
    build a temporary benchmark root with integer-named symlinks.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BUNDLE) + os.pathsep + env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # symlink images/ as-is (eval reads image_ids from it via int(stem))
        (tmp_path / "images").symlink_to(CONTROLNET_DATA / "images")

        # create labels/ with integer-named symlinks → regions/{zero-padded}.png
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        for src in sorted((CONTROLNET_DATA / "regions").glob("*.png")):
            int_id = int(src.stem)
            (labels_dir / f"{int_id}.png").symlink_to(src)

        out_json = tmp_path / "controlnet_eval.json"
        out_csv  = tmp_path / "controlnet_eval.csv"

        result = subprocess.run(
            [
                sys.executable, str(CONTROLNET_EVAL_SCRIPT),
                "--benchmark-root", str(tmp_path),
                "--method", "ArchiTexture", str(CONTROLNET_PRED),
                "--out-json", str(out_json),
                "--out-csv",  str(out_csv),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("eval_binary_partition_maskbank.py failed")
        summary = json.loads(out_json.read_text())

    arch = summary["methods"]["ArchiTexture"]
    return {
        "num_images": summary["num_images"],
        "coverage":   arch["coverage"],
        "direct":     arch["direct"],
        "invariant":  arch["invariant"],
    }


# ── DeTexture ADE20K evaluator ────────────────────────────────────────────────
def _detexture_eval() -> dict:
    """Call eval_two_mask_partition_maskbank.py on committed DeTexture ADE20K masks.

    Uses the two-mask partition protocol: pixels where both GT masks agree are
    ignored; IoU/ARI is computed only on the exclusive valid region. Reports the
    orientation-invariant (best-of-two-orientations) all-images metric.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BUNDLE) + os.pathsep + env.get("PYTHONPATH", "")

    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "detexture_eval.json"
        out_csv  = Path(tmp) / "detexture_eval.csv"
        result = subprocess.run(
            [
                sys.executable, str(DETEXTURE_EVAL_SCRIPT),
                "--benchmark-root", str(DETEXTURE_BENCH),
                "--method", "ArchiTexture", str(DETEXTURE_PRED),
                "--out-json", str(out_json),
                "--out-csv",  str(out_csv),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("eval_two_mask_partition_maskbank.py failed")
        summary = json.loads(out_json.read_text())

    arch = summary["methods"]["ArchiTexture"]
    return {
        "num_images": summary["num_images"],
        "coverage":   arch["coverage"],
        "direct":     arch["direct"],
        "invariant":  arch["invariant"],
    }


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> int:
    _check_paths()

    print("Running RWTD eval …")
    rwtd = _rwtd_eval()
    print("Running STLD eval …")
    stld = _stld_eval()
    print("Running ControlNet bridge eval …")
    controlnet = _controlnet_eval()
    print("Running DeTexture ADE20K eval …")
    detexture = _detexture_eval()

    print()
    print("=" * 62)
    print(f"{'Route':<22} {'Metric':<8} {'Got':>8}  {'Paper':>8}  {'Match':>5}")
    print("-" * 62)

    results: dict = {}
    all_match = True

    def row(label: str, got: float, target: float) -> None:
        nonlocal all_match
        ok = abs(got - target) <= TOLERANCE
        if not ok:
            all_match = False
        sym = "OK" if ok else "FAIL"
        print(f"  {label:<20} {_fmt(got):>8}  {_fmt(target):>8}  {sym:>5}")

    print("  RWTD full-256")
    row("mIoU", rwtd["miou"], PAPER["rwtd_full256"]["miou"])
    row("ARI",  rwtd["ari"],  PAPER["rwtd_full256"]["ari"])
    results["rwtd_full256"] = rwtd

    print("  STLD covered-182")
    row("mIoU", stld["covered"]["miou"], PAPER["stld_covered182"]["miou"])
    row("ARI",  stld["covered"]["ari"],  PAPER["stld_covered182"]["ari"])
    results["stld_covered182"] = stld

    print("  ControlNet bridge (invariant)")
    row("mIoU", controlnet["invariant"]["all"]["miou"], PAPER["controlnet_invariant"]["miou"])
    row("ARI",  controlnet["invariant"]["all"]["ari"],  PAPER["controlnet_invariant"]["ari"])
    results["controlnet_invariant"] = controlnet

    print("  DeTexture ADE20K (invariant)")
    row("mIoU", detexture["invariant"]["all"]["miou"], PAPER["detexture_invariant"]["miou"])
    row("ARI",  detexture["invariant"]["all"]["ari"],  PAPER["detexture_invariant"]["ari"])
    results["detexture_invariant"] = detexture

    print("=" * 62)
    verdict = "ALL MATCH" if all_match else "MISMATCH — see rows above"
    print(f"Verdict: {verdict}")
    print()

    out = ROOT / "proposal_repro" / "verified_results.json"
    out.write_text(json.dumps({"paper": PAPER, "reproduced": results, "match": all_match}, indent=2))
    print(f"Saved → {out.relative_to(ROOT)}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
