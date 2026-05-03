#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "synthetic_texture_perlin_stitched_eval_numeric"
TEX_ROOT = (
    ROOT
    / "experiments"
    / "perlin_controlnet_eval_20260312"
    / "full_0p3"
    / "union_masks_0p3"
)
ARCH_ROOT = (
    ROOT
    / "experiments"
    / "perlin_controlnet_eval_20260312"
    / "full_0p3"
    / "stageA_0p3"
    / "strict_ptd_learned"
    / "masks"
)
OUT_PATH = ROOT / "TextureSum2_paper" / "figures" / "fig_controlnet_bridge_examples.png"

CASE_IDS = [
    ("1013", "1013: TextureSAM leaks across the stitched transition"),
    ("1255", "1255: TextureSAM over-broadens the generator boundary"),
]

COLUMN_LABELS = [
    "Input",
    "GT partition",
    "TextureSAM union view",
    "ArchiTexture Stage-A",
]

GT_COLOR = np.array([44, 170, 44], dtype=np.float32)
TEX_COLOR = np.array([220, 60, 60], dtype=np.float32)
ARCH_COLOR = np.array([242, 148, 34], dtype=np.float32)


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = _load_font(18)
SMALL_FONT = _load_font(15)


def overlay_mask(image_path: Path, mask_path: Path, color: np.ndarray) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if not mask_path.exists():
        return image

    mask = Image.open(mask_path).convert("L")
    rgb = np.asarray(image).astype(np.float32)
    mask_arr = np.asarray(mask) > 127
    rgb[mask_arr] = 0.58 * rgb[mask_arr] + 0.42 * color

    edge = np.zeros_like(mask_arr, dtype=bool)
    edge[:-1, :] |= mask_arr[:-1, :] != mask_arr[1:, :]
    edge[1:, :] |= mask_arr[1:, :] != mask_arr[:-1, :]
    edge[:, :-1] |= mask_arr[:, :-1] != mask_arr[:, 1:]
    edge[:, 1:] |= mask_arr[:, 1:] != mask_arr[:, :-1]
    rgb[edge] = color
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def make_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (255, 255, 255))
    framed = ImageOps.contain(image, (size[0] - 18, size[1] - 18))
    x = (size[0] - framed.width) // 2
    y = (size[1] - framed.height) // 2
    canvas.paste(framed, (x, y))
    return canvas


def main() -> None:
    panel_w, panel_h = 300, 228
    cols = len(COLUMN_LABELS)
    rows = len(CASE_IDS)
    header_h = 34
    gutter = 12
    row_label_h = 24
    width = cols * panel_w + (cols - 1) * gutter
    height = header_h + rows * (panel_h + row_label_h) + (rows - 1) * 10

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for c, label in enumerate(COLUMN_LABELS):
        x = c * (panel_w + gutter) + 10
        draw.text((x, 6), label, fill=(10, 10, 10), font=FONT)

    for r, (image_id, row_label) in enumerate(CASE_IDS):
        y0 = header_h + r * (panel_h + row_label_h + 10)
        img_path = DATA_ROOT / "images" / f"{image_id}.png"
        gt_path = DATA_ROOT / "labels" / f"{image_id}.png"
        tex_path = TEX_ROOT / f"{image_id}.png"
        arch_path = ARCH_ROOT / f"{image_id}.png"

        panels = [
            Image.open(img_path).convert("RGB"),
            overlay_mask(img_path, gt_path, GT_COLOR),
            overlay_mask(img_path, tex_path, TEX_COLOR),
            overlay_mask(img_path, arch_path, ARCH_COLOR),
        ]

        for c, panel in enumerate(panels):
            x0 = c * (panel_w + gutter)
            canvas.paste(make_panel(panel, (panel_w, panel_h)), (x0, y0))

        draw.text((8, y0 + panel_h + 2), row_label, fill=(20, 20, 20), font=SMALL_FONT)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
