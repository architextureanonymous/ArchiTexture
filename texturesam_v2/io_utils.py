from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_rwtd_dirs(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "images", root / "labels"),
        (root / "images", root / "masks"),
        (root / "originals", root / "masks"),
        (root / "review" / "originals", root / "review" / "masks"),
    ]
    for img_d, mask_d in candidates:
        if img_d.exists() and mask_d.exists():
            return img_d, mask_d
    raise FileNotFoundError(f"Could not infer RWTD image/mask dirs under {root}")


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_raw(path: Path) -> np.ndarray:
    # Use PIL to preserve indexed PNG labels (mode "P") exactly.
    try:
        with Image.open(path) as im:
            m = np.array(im)
    except FileNotFoundError:
        raise
    except Exception:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
    if m.ndim == 3 and m.shape[-1] == 4:
        m = m[..., :3]
    return m


def _background_color_from_corners(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    corners = np.stack(
        [
            mask_rgb[0, 0, :],
            mask_rgb[0, w - 1, :],
            mask_rgb[h - 1, 0, :],
            mask_rgb[h - 1, w - 1, :],
        ],
        axis=0,
    )
    colors, counts = np.unique(corners, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))]


def ensure_binary(
    mask: np.ndarray,
    threshold: int = 127,
    *,
    nonzero_is_foreground: bool = False,
    strict: bool = False,
    source_name: str | None = None,
) -> np.ndarray:
    m = mask
    if m.ndim == 3:
        if m.shape[-1] == 4:
            m = m[..., :3]
        if nonzero_is_foreground:
            if m.shape[-1] == 1:
                m = m[..., 0]
            else:
                uniq = int(np.unique(m.reshape(-1, m.shape[-1]), axis=0).shape[0])
                if strict and uniq > 256:
                    src = source_name or "<unknown>"
                    raise ValueError(
                        f"Mask looks like an RGB image (too many colors for a label): {src}, unique_colors={uniq}"
                    )
                bg = _background_color_from_corners(m)
                return np.any(m != bg[None, None, :], axis=-1).astype(np.uint8)
        else:
            # Preserve exact values for indexed/gray-like encodings.
            if m.shape[-1] == 1:
                m = m[..., 0]
            elif np.array_equal(m[..., 0], m[..., 1]) and np.array_equal(m[..., 0], m[..., 2]):
                m = m[..., 0]
            else:
                m = m.astype(np.float32).mean(axis=-1)
    m = m.astype(np.float32, copy=False)
    if m.size == 0:
        return np.zeros_like(m, dtype=np.uint8)
    if nonzero_is_foreground:
        return (m > 0).astype(np.uint8)
    if float(np.nanmax(m)) <= 1.0 and float(np.nanmin(m)) >= 0.0:
        return (m > 0).astype(np.uint8)
    return (m > float(threshold)).astype(np.uint8)


def ensure_binary_gt(mask: np.ndarray, *, strict: bool = True, source_name: str | None = None) -> np.ndarray:
    # GT should treat any non-zero class/label as foreground for RWTD binary evaluation.
    return ensure_binary(mask, nonzero_is_foreground=True, strict=strict, source_name=source_name)


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask > 0).astype(np.uint8) * 255
    ok = cv2.imwrite(str(path), m)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {path}")


def list_rwtd_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(items)
