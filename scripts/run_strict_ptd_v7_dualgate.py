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
            "Strict PTD-v7 dual-AI gate: use v4 as primary and v6 as rescue only when "
            "v4 appears low-coverage against proposal union. No RWTD-label tuning in gate logic."
        )
    )
    p.add_argument("--rwtd-root", type=Path, required=True)
    p.add_argument("--prompt-masks-root", type=Path, required=True)
    p.add_argument("--v4-masks-root", type=Path, required=True)
    p.add_argument("--v6-masks-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)

    p.add_argument("--low-cov", type=float, default=0.18)
    p.add_argument("--low-area-u", type=float, default=0.22)
    p.add_argument("--gain-cov", type=float, default=0.15)
    p.add_argument("--min-prec6", type=float, default=0.55)
    p.add_argument("--max-area-u6", type=float, default=1.40)

    p.add_argument("--tiny-img4", type=float, default=0.010)
    p.add_argument("--min-img6", type=float, default=0.020)
    p.add_argument("--max-img6", type=float, default=0.80)
    p.add_argument("--min-prec6-tiny", type=float, default=0.40)
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

    out_dir = args.out_root / "strict_ptd_v7_dualgate"
    out_masks = out_dir / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    image_ids = sorted([p.stem for p in image_dir.glob("*.jpg")], key=lambda s: int(s))
    rows: list[dict[str, float | int | str]] = []

    switched = 0
    for iid in image_ids:
        gt = _read_bin(gt_dir / f"{iid}.png", is_gt=True)
        m4 = _read_bin(args.v4_masks_root / f"{iid}.png")
        m6 = _read_bin(args.v6_masks_root / f"{iid}.png")
        union = _proposal_union(args.prompt_masks_root, iid, gt.shape)

        c4, p4, a4 = _cov_prec_area(m4, union)
        c6, p6, a6 = _cov_prec_area(m6, union)

        area4 = float((m4 > 0).sum() / max(m4.size, 1))
        area6 = float((m6 > 0).sum() / max(m6.size, 1))

        use6 = False
        if (
            c4 < args.low_cov
            and a4 < args.low_area_u
            and (c6 - c4) > args.gain_cov
            and p6 > args.min_prec6
            and a6 < args.max_area_u6
        ):
            use6 = True
        if (
            area4 < args.tiny_img4
            and area6 > args.min_img6
            and area6 < args.max_img6
            and p6 > args.min_prec6_tiny
        ):
            use6 = True

        pred = m6 if use6 else m4
        if use6:
            switched += 1

        Image.fromarray((pred * 255).astype(np.uint8)).save(out_masks / f"{iid}.png")

        met = rwtd_invariant_metrics(pred, gt)
        rows.append(
            {
                "image_id": iid,
                "v7_iou": float(met.iou),
                "v7_ari": float(met.ari),
                "used_v6": int(use6),
                "cov4": float(c4),
                "prec4": float(p4),
                "areaU4": float(a4),
                "cov6": float(c6),
                "prec6": float(p6),
                "areaU6": float(a6),
            }
        )

    miou = float(np.mean([float(r["v7_iou"]) for r in rows])) if rows else 0.0
    ari = float(np.mean([float(r["v7_ari"]) for r in rows])) if rows else 0.0

    summary = {
        "num_images": int(len(rows)),
        "switched_to_v6": int(switched),
        "v7_miou": float(miou),
        "v7_ari": float(ari),
        "gate": {
            "low_cov": float(args.low_cov),
            "low_area_u": float(args.low_area_u),
            "gain_cov": float(args.gain_cov),
            "min_prec6": float(args.min_prec6),
            "max_area_u6": float(args.max_area_u6),
            "tiny_img4": float(args.tiny_img4),
            "min_img6": float(args.min_img6),
            "max_img6": float(args.max_img6),
            "min_prec6_tiny": float(args.min_prec6_tiny),
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
