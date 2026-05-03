from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.io_utils import ensure_binary, ensure_binary_gt, read_mask_raw
from texturesam_v2.metrics import rwtd_invariant_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strict PTD-v9 conservative gate between v7 and v8 masks. "
            "No RWTD-label training required in gate logic."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--v7-masks-root", type=Path, required=True)
    p.add_argument("--v8-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)

    p.add_argument("--low-cov7", type=float, default=0.22)
    p.add_argument("--min-cov-gain8", type=float, default=0.12)
    p.add_argument("--min-prec8", type=float, default=0.45)
    p.add_argument("--max-area-u8", type=float, default=1.60)
    p.add_argument("--min-area-u8", type=float, default=0.05)

    # Tiny-mask rescue.
    p.add_argument("--tiny-img7", type=float, default=0.012)
    p.add_argument("--min-img8", type=float, default=0.025)
    p.add_argument("--max-img8", type=float, default=0.85)
    p.add_argument("--min-prec8-tiny", type=float, default=0.30)
    return p.parse_args()


def _read_bin(path: Path, *, is_gt: bool = False) -> np.ndarray:
    a = read_mask_raw(path)
    if is_gt:
        return ensure_binary_gt(a, source_name=str(path))
    return ensure_binary(a)


def _proposal_union(prompt_root: Path, image_id: str, shape: tuple[int, int]) -> np.ndarray:
    masks: list[np.ndarray] = []
    for p in prompt_root.glob(f"rwtd_{image_id}_p*_m*.png"):
        m = ensure_binary(read_mask_raw(p))
        if m.shape == shape:
            masks.append(m)
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    u = np.logical_or.reduce([m > 0 for m in masks]).astype(np.uint8)
    return u


def _cov_prec_area(mask: np.ndarray, union: np.ndarray) -> tuple[float, float, float]:
    m = mask > 0
    u = union > 0
    ma = int(m.sum())
    ua = int(u.sum())
    if ua <= 0 or ma <= 0:
        return 0.0, 0.0, 0.0
    inter = float(np.logical_and(m, u).sum())
    cov = float(inter / max(ua, 1))
    prec = float(inter / max(ma, 1))
    area_u = float(ma / max(ua, 1))
    return cov, prec, area_u


def main() -> None:
    args = parse_args()

    image_dir = args.rwtd_root / "images"
    gt_dir = args.rwtd_root / "labels"
    if not image_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(
            f"Expected RWTD layout under {args.rwtd_root}: images/ and labels/"
        )

    out_dir = args.out_root / "strict_ptd_v9_v7_v8_gate"
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    image_ids = sorted([p.stem for p in image_dir.glob("*.jpg")], key=lambda s: int(s))
    rows: list[dict[str, float | int | str]] = []

    switched = 0
    for iid in image_ids:
        gt = _read_bin(gt_dir / f"{iid}.png", is_gt=True)
        m7 = _read_bin(args.v7_masks_root / f"{iid}.png")
        m8 = _read_bin(args.v8_masks_root / f"{iid}.png")
        union = _proposal_union(args.prompt_masks_root, iid, gt.shape)

        c7, p7, a7 = _cov_prec_area(m7, union)
        c8, p8, a8 = _cov_prec_area(m8, union)

        area7 = float((m7 > 0).sum() / max(m7.size, 1))
        area8 = float((m8 > 0).sum() / max(m8.size, 1))

        use8 = False

        # Coverage-first rescue: v7 is clearly under-covering proposal union.
        if (
            c7 < args.low_cov7
            and (c8 - c7) > args.min_cov_gain8
            and p8 > args.min_prec8
            and a8 < args.max_area_u8
            and a8 > args.min_area_u8
        ):
            use8 = True

        # Tiny-mask rescue.
        if (
            area7 < args.tiny_img7
            and area8 > args.min_img8
            and area8 < args.max_img8
            and p8 > args.min_prec8_tiny
        ):
            use8 = True

        pred = m8 if use8 else m7
        if use8:
            switched += 1

        Image.fromarray((pred * 255).astype(np.uint8)).save(out_masks / f"{iid}.png")

        met = rwtd_invariant_metrics(pred, gt)
        rows.append(
            {
                "image_id": iid,
                "v9_iou": float(met.iou),
                "v9_ari": float(met.ari),
                "used_v8": int(use8),
                "cov7": float(c7),
                "prec7": float(p7),
                "areaU7": float(a7),
                "cov8": float(c8),
                "prec8": float(p8),
                "areaU8": float(a8),
            }
        )

    miou = float(np.mean([float(r["v9_iou"]) for r in rows])) if rows else 0.0
    ari = float(np.mean([float(r["v9_ari"]) for r in rows])) if rows else 0.0

    summary = {
        "num_images": int(len(rows)),
        "switched_to_v8": int(switched),
        "v9_miou": float(miou),
        "v9_ari": float(ari),
        "gate": {
            "low_cov7": float(args.low_cov7),
            "min_cov_gain8": float(args.min_cov_gain8),
            "min_prec8": float(args.min_prec8),
            "max_area_u8": float(args.max_area_u8),
            "min_area_u8": float(args.min_area_u8),
            "tiny_img7": float(args.tiny_img7),
            "min_img8": float(args.min_img8),
            "max_img8": float(args.max_img8),
            "min_prec8_tiny": float(args.min_prec8_tiny),
        },
        "artifacts": {
            "per_image": str((out_dir / "per_image.csv").resolve()),
            "masks": str(out_masks.resolve()),
        },
    }

    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
