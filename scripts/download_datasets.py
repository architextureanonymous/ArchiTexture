#!/usr/bin/env python3
"""Download ArchiTexture benchmark datasets from Kaggle and prepare them for eval.

Downloads:
  - ControlNet PTD 1742  → datasets/ControlNet_PTD_1742/
  - DeTexture ADE20K 56  → datasets/ADE20k_Detexture_56/

Then runs a CPU-only smoke check (file count + layout validation) on each.

Usage:
  python scripts/download_datasets.py [--repo-root REPO_ROOT] [--skip-download] [--smoke-only]

Requires:
  pip install kaggle
  A kaggle.json at ~/.kaggle/kaggle.json  OR  KAGGLE_USERNAME + KAGGLE_KEY env vars.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


CONTROLNET_KAGGLE_SLUG = "architexanonymous/architexture-controlnet-ptd-1742"
DETEXTURE_KAGGLE_SLUG  = "architexanonymous/architexture-detexture-ade20k-56"

CONTROLNET_DEST   = "datasets/ControlNet_PTD_1742"
DETEXTURE_DEST    = "datasets/ADE20k_Detexture_56"


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def _kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        ["kaggle", "datasets", "download", slug, "--unzip", "-p", str(dest)],
        cwd=dest.parent,
    )


# ── ControlNet ────────────────────────────────────────────────────────────────

def download_controlnet(repo_root: Path) -> Path:
    dest = repo_root / CONTROLNET_DEST
    print(f"\n[1/2] Downloading ControlNet PTD 1742 → {dest} …")
    _kaggle_download(CONTROLNET_KAGGLE_SLUG, dest)
    return dest


def smoke_controlnet(dest: Path) -> None:
    print(f"  Smoke-checking ControlNet layout at {dest} …")
    images  = list((dest / "images").glob("*"))  if (dest / "images").is_dir()  else []
    regions = list((dest / "regions").glob("*")) if (dest / "regions").is_dir() else []
    edges   = list((dest / "edges").glob("*"))   if (dest / "edges").is_dir()   else []
    ok = True
    for folder, expected, files in [
        ("images",  1742, images),
        ("regions", 1742, regions),
        ("edges",   1742, edges),
    ]:
        n = len(files)
        status = "OK" if n == expected else f"WARN (got {n}, expected {expected})"
        print(f"    {folder}/: {n} files — {status}")
        if n != expected:
            ok = False
    if not ok:
        print("  WARNING: unexpected file counts — check the download.")
    else:
        print("  ControlNet smoke check PASSED.")


# ── DeTexture ADE20K 56 ───────────────────────────────────────────────────────

def download_detexture(repo_root: Path) -> Path:
    dest = repo_root / DETEXTURE_DEST
    tmp  = repo_root / "datasets" / "_detexture_kaggle_raw"
    print(f"\n[2/2] Downloading DeTexture ADE20K 56 → {dest} …")
    _kaggle_download(DETEXTURE_KAGGLE_SLUG, tmp)
    _transform_detexture(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def _transform_detexture(src: Path, dest: Path) -> None:
    """Convert Kaggle layout → eval layout.

    Kaggle:   images/{id}.png   labels_a/{id}.png   labels_b/{id}.png
    Eval:     assets/crops/{id}.png
              assets/masks/{id}_mask_a.png
              assets/masks/{id}_mask_b.png
    """
    crops_dir = dest / "assets" / "crops"
    masks_dir = dest / "assets" / "masks"
    crops_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    images_src   = src / "images"
    labels_a_src = src / "labels_a"
    labels_b_src = src / "labels_b"

    for required in (images_src, labels_a_src, labels_b_src):
        if not required.is_dir():
            sys.exit(
                f"Expected '{required}' in Kaggle download but it was not found. "
                "The dataset structure may have changed — check the Kaggle page."
            )

    images = sorted(images_src.glob("*"), key=lambda p: p.name)
    n_copied = 0
    for img_path in images:
        if not img_path.is_file():
            continue
        stem = img_path.stem
        shutil.copy2(img_path, crops_dir / img_path.name)
        mask_a = labels_a_src / img_path.name
        mask_b = labels_b_src / img_path.name
        if not mask_a.is_file():
            print(f"  WARNING: missing labels_a/{img_path.name}")
            continue
        if not mask_b.is_file():
            print(f"  WARNING: missing labels_b/{img_path.name}")
            continue
        shutil.copy2(mask_a, masks_dir / f"{stem}_mask_a.png")
        shutil.copy2(mask_b, masks_dir / f"{stem}_mask_b.png")
        n_copied += 1

    print(f"  Layout transform done: {n_copied} samples written to {dest}/assets/")


def smoke_detexture(dest: Path) -> None:
    print(f"  Smoke-checking DeTexture layout at {dest} …")
    crops = list((dest / "assets" / "crops").glob("*")) if (dest / "assets" / "crops").is_dir() else []
    masks = list((dest / "assets" / "masks").glob("*")) if (dest / "assets" / "masks").is_dir() else []
    n_crops  = len(crops)
    n_masks  = len(masks)
    n_pairs  = n_masks // 2
    ok = True
    print(f"    assets/crops/: {n_crops} files — {'OK' if n_crops == 56 else f'WARN (expected 56)'}")
    print(f"    assets/masks/: {n_masks} files ({n_pairs} pairs) — {'OK' if n_pairs == 56 else f'WARN (expected 56 pairs)'}")
    if n_crops != 56 or n_pairs != 56:
        ok = False

    # verify every crop has both masks
    missing = []
    for crop in (dest / "assets" / "crops").glob("*"):
        stem = crop.stem
        if not (dest / "assets" / "masks" / f"{stem}_mask_a.png").is_file():
            missing.append(f"{stem}_mask_a.png")
        if not (dest / "assets" / "masks" / f"{stem}_mask_b.png").is_file():
            missing.append(f"{stem}_mask_b.png")
    if missing:
        print(f"  WARNING: {len(missing)} mask files missing: {missing[:4]} …")
        ok = False

    if not ok:
        print("  WARNING: DeTexture smoke check found issues.")
    else:
        print("  DeTexture smoke check PASSED.")


# ── eval commands ─────────────────────────────────────────────────────────────

def print_eval_commands(repo_root: Path) -> None:
    cstd_root    = repo_root / CONTROLNET_DEST
    detex_root   = repo_root / DETEXTURE_DEST
    variant      = "feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2"

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Datasets ready. Run evaluation:                                 ║
╚══════════════════════════════════════════════════════════════════╝

── Proposal-space (all 4 datasets, no GPU needed) ──────────────────
  python proposal_repro/verify_results.py

── Feature-clustering — ControlNet PTD 1742 (GPU required) ─────────""")
    print(f"  python -m scripts.feature_clustering.main eval-cstd-binary \\")
    print(f"    --dataset-root {cstd_root} \\")
    print(f"    --variant {variant} \\")
    print( "    --device cuda --failure-policy skip \\")
    print( "    --output-dir outputs/controlnet_fc_eval")
    print("""
── Feature-clustering — DeTexture ADE20K 56 (GPU required) ─────────""")
    print(f"  python -m scripts.feature_clustering.main eval-detexture-binary \\")
    print(f"    --dataset-root {detex_root} \\")
    print(f"    --variant {variant} \\")
    print( "    --device cuda --failure-policy skip \\")
    print( "    --output-dir outputs/detexture_fc_eval")
    print("""
── Feature-clustering — RWTD and STLD ──────────────────────────────
  RWTD and STLD image data are not publicly redistributed.
  Mount local dataset drops to datasets/RWTD and datasets/STLD,
  then run:

  python scripts/feature_clustering/repro_table_1.py \\
    --output-root outputs/repro_table_1 \\
    --rwtd-root datasets/RWTD \\
    --stld-root datasets/STLD \\
    --cstd-root datasets/CSTD
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."),
                        help="Path to the ArchiTexture repo root (default: cwd)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download; only run smoke checks on existing dirs")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Alias for --skip-download")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    skip = args.skip_download or args.smoke_only

    if not skip:
        cstd_dest  = download_controlnet(repo_root)
        detex_dest = download_detexture(repo_root)
    else:
        cstd_dest  = repo_root / CONTROLNET_DEST
        detex_dest = repo_root / DETEXTURE_DEST
        print(f"Skipping download. Checking existing dirs …")

    print("\n── Smoke checks ───────────────────────────────────────────────────")
    smoke_controlnet(cstd_dest)
    smoke_detexture(detex_dest)

    print_eval_commands(repo_root)


if __name__ == "__main__":
    main()
