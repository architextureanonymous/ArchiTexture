#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.ptd_learned import _load_binary_shape, _rotate_binary_mask, _texture_canvas

BRODATZ_RE = re.compile(r"^(1\.[123])\.(\d{2})\.tiff$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a held-out Khan/STLD-style synthetic benchmark: two Brodatz textures "
            "composited with one MPEG-7 foreground shape."
        )
    )
    p.add_argument("--textures-root", type=Path, required=True)
    p.add_argument("--shape-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--shape-min-area-ratio", type=float, default=0.08)
    p.add_argument("--shape-max-area-ratio", type=float, default=0.55)
    p.add_argument("--shape-rotation-max-deg", type=float, default=40.0)
    p.add_argument("--shape-margin-fraction", type=float, default=0.05)
    return p.parse_args()


def _discover_groups(textures_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(textures_root.glob("*.tiff")):
        match = BRODATZ_RE.match(path.name)
        if match is None:
            continue
        groups.setdefault(match.group(2), []).append(path)
    if not groups:
        raise RuntimeError(f"No Brodatz 1.1/1.2/1.3 textures found under {textures_root}")
    return dict(sorted(groups.items()))


def _discover_shapes(shape_root: Path) -> list[Path]:
    paths = sorted(
        p
        for pat in ("*.gif", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        for p in shape_root.rglob(pat)
        if p.is_file()
    )
    if not paths:
        raise RuntimeError(f"No shape files found under {shape_root}")
    return paths


def _read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"Failed to read texture: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _place_shape(
    *,
    h: int,
    w: int,
    shape_path: Path,
    rng: np.random.Generator,
    min_area_ratio: float,
    max_area_ratio: float,
    rotation_max_deg: float,
    margin_fraction: float,
) -> tuple[np.ndarray, float]:
    margin = int(round(float(margin_fraction) * min(h, w)))
    margin = max(0, min(margin, max((min(h, w) // 2) - 8, 0)))
    max_inner_h = max(16, h - 2 * margin)
    max_inner_w = max(16, w - 2 * margin)

    last_mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(32):
        mask = _load_binary_shape(shape_path).copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rotation_max_deg > 0.0:
            angle = float(rng.uniform(-rotation_max_deg, rotation_max_deg))
            mask = _rotate_binary_mask(mask, angle)

        sh, sw = mask.shape
        fit = min(max_inner_h / max(sh, 1), max_inner_w / max(sw, 1))
        fit = max(fit, 0.05)
        scale = float(rng.uniform(0.35, 0.95)) * fit
        out_h = max(12, int(round(sh * scale)))
        out_w = max(12, int(round(sw * scale)))
        if out_h >= h or out_w >= w:
            continue

        resized = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        resized = cv2.morphologyEx((resized > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
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
        return canvas, float(canvas.mean())

    if int(last_mask.sum()) <= 0:
        fallback = np.zeros((h, w), dtype=np.uint8)
        cy = h // 2
        cx = w // 2
        ry = max(12, int(round(0.18 * h)))
        rx = max(12, int(round(0.20 * w)))
        cv2.ellipse(fallback, (cx, cy), (rx, ry), 0.0, 0, 360, 1, -1)
        return fallback, float(fallback.mean())

    fh, fw = last_mask.shape
    y0 = max(0, (h - fh) // 2)
    x0 = max(0, (w - fw) // 2)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y0 : y0 + min(fh, h), x0 : x0 + min(fw, w)] = last_mask[: min(fh, h), : min(fw, w)]
    return canvas, float(canvas.mean())


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    groups = _discover_groups(args.textures_root)
    group_ids = list(groups.keys())
    shapes = _discover_shapes(args.shape_root)

    out_root = args.out_root
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    texture_cache = {path: _read_rgb(path) for paths in groups.values() for path in paths}
    rows: list[dict[str, object]] = []

    h = int(args.image_size)
    w = int(args.image_size)

    for image_id in range(1, int(args.num_images) + 1):
        bg_id, fg_id = rng.choice(group_ids, size=2, replace=False if len(group_ids) >= 2 else True).tolist()
        bg_src = groups[str(bg_id)][int(rng.integers(0, len(groups[str(bg_id)])))]
        fg_src = groups[str(fg_id)][int(rng.integers(0, len(groups[str(fg_id)])))]
        shape_path = shapes[int(rng.integers(0, len(shapes)))]

        bg = _texture_canvas(texture_cache[bg_src], h=h, w=w, rng=rng)
        fg = _texture_canvas(texture_cache[fg_src], h=h, w=w, rng=rng)
        mask, area_ratio = _place_shape(
            h=h,
            w=w,
            shape_path=shape_path,
            rng=rng,
            min_area_ratio=float(args.shape_min_area_ratio),
            max_area_ratio=float(args.shape_max_area_ratio),
            rotation_max_deg=float(args.shape_rotation_max_deg),
            margin_fraction=float(args.shape_margin_fraction),
        )

        image = bg.copy()
        image[mask > 0] = fg[mask > 0]
        label = (mask > 0).astype(np.uint8) * 255

        image_path = images_dir / f"{image_id}.png"
        label_path = labels_dir / f"{image_id}.png"
        ok_img = cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        ok_lbl = cv2.imwrite(str(label_path), label)
        if not ok_img or not ok_lbl:
            raise RuntimeError(f"Failed to write synthetic sample {image_id}")

        rows.append(
            {
                "image_id": int(image_id),
                "background_class": f"brodatz_{bg_id}",
                "foreground_class": f"brodatz_{fg_id}",
                "background_source": bg_src.name,
                "foreground_source": fg_src.name,
                "shape_file": shape_path.name,
                "foreground_area_ratio": float(area_ratio),
            }
        )

    summary = {
        "out_root": str(out_root),
        "textures_root": str(args.textures_root),
        "shape_root": str(args.shape_root),
        "num_images": int(args.num_images),
        "image_size": int(args.image_size),
        "seed": int(args.seed),
        "class_count": int(len(group_ids)),
        "shape_count": int(len(shapes)),
        "shape_min_area_ratio": float(args.shape_min_area_ratio),
        "shape_max_area_ratio": float(args.shape_max_area_ratio),
        "foreground_area_ratio_mean": float(np.mean([float(r["foreground_area_ratio"]) for r in rows])) if rows else 0.0,
        "foreground_area_ratio_min": float(np.min([float(r["foreground_area_ratio"]) for r in rows])) if rows else 0.0,
        "foreground_area_ratio_max": float(np.max([float(r["foreground_area_ratio"]) for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    (out_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
