#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image, ImageOps

from build_texturesam2_ai_docs import copy_file, ensure_clean_dir, load_mask, load_rows, load_json, overlay_panel


TEMPLATE_STYLES = Path("external/rwtd_miner_public_site/docs/texturesam2_ai_gallery/styles.css")


@dataclass
class Entry:
    image_id: int
    status: str
    status_label: str
    texturesam_present: bool
    proposal_count: int
    merged_components: int
    texturesam_direct_miou: float | None
    texturesam_direct_ari: float | None
    texturesam_invariant_miou: float | None
    texturesam_invariant_ari: float | None
    architexture_direct_miou: float | None
    architexture_direct_ari: float | None
    architexture_invariant_miou: float | None
    architexture_invariant_ari: float | None
    delta_direct_miou: float | None
    delta_direct_ari: float | None
    delta_invariant_miou: float | None
    delta_invariant_ari: float | None
    beats_direct_miou: bool
    beats_direct_ari: bool
    beats_invariant_miou: bool
    beats_invariant_ari: bool
    summary_line: str
    note_line: str
    image_rgb: str
    gt_overlay: str
    edge_overlay: str | None
    texturesam_union_overlay: str | None
    architexture_overlay: str
    raw_image: str
    raw_gt: str
    raw_edge: str | None
    raw_texturesam_union: str | None
    raw_architexture: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Publish the ControlNet-stitched PTD benchmark explorer using the same gallery layout "
            "as the existing RWTD/STLD public pages."
        )
    )
    p.add_argument("--site-root", type=Path, required=True)
    p.add_argument("--out-subdir", type=str, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--edges-dir", type=Path, required=True)
    p.add_argument("--texturesam-maskbank", type=Path, required=True)
    p.add_argument("--texturesam-union-dir", type=Path, required=True)
    p.add_argument("--architexture-masks", type=Path, required=True)
    p.add_argument("--per-image-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--page-title", type=str, required=True)
    p.add_argument("--hero-kicker", type=str, required=True)
    p.add_argument("--subtitle", type=str, required=True)
    p.add_argument("--home-href", type=str, default="../index.html")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--sam2-original-summary", type=Path)
    p.add_argument("--sam2-original-label", type=str, default="SAM2 original")
    return p.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def parse_int(value: str | None) -> int:
    if value is None:
        return 0
    text = value.strip()
    if not text:
        return 0
    return int(float(text))


def load_rows_by_id(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["image_id"]): row for row in load_rows(path)}


def save_web_image(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img = ImageOps.fit(img, (size, size), method=Image.Resampling.LANCZOS)
    img.save(dst, quality=88, optimize=True)


def save_web_mask(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("L")
    img = img.resize((size, size), resample=Image.Resampling.NEAREST)
    img.save(dst, optimize=True)


def relative_asset(path: Path, out_dir: Path) -> str:
    return "./" + path.relative_to(out_dir).as_posix()


def blank_panel(size: tuple[int, int], label: str) -> Image.Image:
    blank = Image.new("RGB", size, (241, 244, 246))
    return overlay_panel(blank, None, label, "#6a7b87")


def save_panel(panel: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def panel_from_mask(base: Image.Image, mask_path: Path | None, label: str, color: str) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return blank_panel(base.size, label)
    return overlay_panel(base, load_mask(mask_path), label, color)


def summarize_case(
    *,
    texturesam_present: bool,
    delta_direct_miou: float | None,
    delta_direct_ari: float | None,
    delta_invariant_miou: float | None,
    delta_invariant_ari: float | None,
) -> tuple[str, str, str, str]:
    if not texturesam_present:
        return (
            "texturesam_miss",
            "TextureSAM miss",
            "TextureSAM emitted no masks on this image, while ArchiTexture still returned a single region.",
            "TextureSAM union is blank because the public mask bank emitted zero proposals here. Direct and invariant scores for TextureSAM are therefore zero on this image.",
        )

    inv_win = (delta_invariant_miou or 0.0) > 0.0 and (delta_invariant_ari or 0.0) > 0.0
    direct_win = (delta_direct_miou or 0.0) > 0.0 and (delta_direct_ari or 0.0) > 0.0

    if inv_win and direct_win:
        return (
            "covered",
            "Strong ArchiTexture win",
            "ArchiTexture beats TextureSAM on both direct foreground metrics and partition-invariant metrics.",
            "Direct metrics use the provided canonical foreground mask. Invariant metrics allow complement matching and are often the fairer view for a two-region binary partition.",
        )
    if inv_win and (delta_direct_miou or 0.0) > 0.0 and (delta_direct_ari or 0.0) <= 0.0:
        return (
            "covered",
            "Partition win, direct ARI split",
            "ArchiTexture is stronger as a binary partition and also improves direct IoU, but TextureSAM keeps the higher direct ARI against the canonical foreground side.",
            "This usually means the two methods disagree mainly on which side should be foreground, not on the existence of the transition itself.",
        )
    if inv_win:
        return (
            "covered",
            "Partition win",
            "ArchiTexture improves the partition-invariant binary split even though the canonical-foreground view remains mixed.",
            "For this benchmark, invariant metrics are important because the generated image contains one transition between two textures and the provided region mask chooses one side as foreground.",
        )
    if (delta_direct_miou or 0.0) > 0.0:
        return (
            "covered",
            "Direct IoU win",
            "ArchiTexture improves direct IoU against the provided foreground mask, but the broader partition picture remains mixed.",
            "TextureSAM is shown as a union view for visualization, while the reported TextureSAM scores are computed from its full mask bank.",
        )
    return (
        "covered",
        "Mixed / TextureSAM-favored",
        "This is a covered comparison where TextureSAM remains stronger on the displayed metrics.",
        "The visual panel still shows the same image, GT, TextureSAM union view, and ArchiTexture result so these cases can be audited directly.",
    )


def build_page_summary(entries: list[Entry], summary_json: dict) -> dict[str, object]:
    arch = summary_json["methods"]["ArchiTexture_small_pre_ring_hi_0p3"]
    ts = summary_json["methods"]["TextureSAM_0p3"]
    return {
        "num_images": int(summary_json["num_images"]),
        "texturesam_coverage": int(ts["coverage"]),
        "architexture_coverage": int(arch["coverage"]),
        "texturesam_direct_all_miou": float(ts["direct"]["all"]["miou"]),
        "texturesam_direct_all_ari": float(ts["direct"]["all"]["ari"]),
        "architexture_direct_all_miou": float(arch["direct"]["all"]["miou"]),
        "architexture_direct_all_ari": float(arch["direct"]["all"]["ari"]),
        "texturesam_invariant_all_miou": float(ts["invariant"]["all"]["miou"]),
        "texturesam_invariant_all_ari": float(ts["invariant"]["all"]["ari"]),
        "architexture_invariant_all_miou": float(arch["invariant"]["all"]["miou"]),
        "architexture_invariant_all_ari": float(arch["invariant"]["all"]["ari"]),
        "delta_direct_all_miou": float(arch["direct"]["all"]["miou"] - ts["direct"]["all"]["miou"]),
        "delta_direct_all_ari": float(arch["direct"]["all"]["ari"] - ts["direct"]["all"]["ari"]),
        "delta_invariant_all_miou": float(arch["invariant"]["all"]["miou"] - ts["invariant"]["all"]["miou"]),
        "delta_invariant_all_ari": float(arch["invariant"]["all"]["ari"] - ts["invariant"]["all"]["ari"]),
        "texturesam_missing": int(summary_json["num_images"] - ts["coverage"]),
    }


def maybe_add_sam2_original(page_summary: dict[str, object], sam2_summary: dict | None, sam2_label: str) -> dict[str, object]:
    if sam2_summary is None:
        return page_summary
    method = sam2_summary.get("methods", {}).get("SAM2_original")
    if not method:
        return page_summary
    page_summary.update(
        {
            "sam2_original_label": sam2_label,
            "sam2_original_coverage": int(method.get("coverage", 0)),
            "sam2_original_invariant_miou_all": float(method.get("invariant", {}).get("all", {}).get("miou", 0.0)),
            "sam2_original_invariant_ari_all": float(method.get("invariant", {}).get("all", {}).get("ari", 0.0)),
        }
    )
    return page_summary


def build_manifest(
    *,
    out_dir: Path,
    images_dir: Path,
    gt_dir: Path,
    edges_dir: Path,
    texturesam_union_dir: Path,
    architexture_masks: Path,
    per_image_csv: Path,
    metadata_csv: Path,
    image_size: int,
) -> list[Entry]:
    eval_rows = load_rows_by_id(per_image_csv)
    meta_rows = load_rows_by_id(metadata_csv)

    raw_dirs = {
        "images": out_dir / "assets" / "raw" / "images",
        "gt": out_dir / "assets" / "raw" / "gt",
        "edges": out_dir / "assets" / "raw" / "edges",
        "texturesam_union": out_dir / "assets" / "raw" / "texturesam_union",
        "architexture": out_dir / "assets" / "raw" / "architexture",
    }
    detail_dirs = {
        "gt": out_dir / "assets" / "detail" / "gt",
        "edge": out_dir / "assets" / "detail" / "edge",
        "texturesam_union": out_dir / "assets" / "detail" / "texturesam_union",
        "architexture": out_dir / "assets" / "detail" / "architexture",
    }
    for path in list(raw_dirs.values()) + list(detail_dirs.values()):
        path.mkdir(parents=True, exist_ok=True)

    entries: list[Entry] = []
    image_paths = sorted(
        (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: int(p.stem),
    )

    for image_path in image_paths:
        image_id = int(image_path.stem)
        gt_path = gt_dir / f"{image_id}.png"
        edge_path = edges_dir / f"{image_id}.png"
        texturesam_union_path = texturesam_union_dir / f"{image_id}.png"
        architexture_path = architexture_masks / f"{image_id}.png"
        if not gt_path.exists() or not architexture_path.exists():
            raise FileNotFoundError(f"Missing required assets for image {image_id}")

        eval_row = eval_rows[image_id]
        meta_row = meta_rows[image_id]

        texturesam_present = parse_bool(eval_row["TextureSAM_0p3_present"])
        texturesam_direct_miou = parse_float(eval_row["TextureSAM_0p3_direct_miou"])
        texturesam_direct_ari = parse_float(eval_row["TextureSAM_0p3_direct_ari"])
        texturesam_invariant_miou = parse_float(eval_row["TextureSAM_0p3_invariant_miou"])
        texturesam_invariant_ari = parse_float(eval_row["TextureSAM_0p3_invariant_ari"])
        architexture_direct_miou = parse_float(eval_row["ArchiTexture_small_pre_ring_hi_0p3_direct_miou"])
        architexture_direct_ari = parse_float(eval_row["ArchiTexture_small_pre_ring_hi_0p3_direct_ari"])
        architexture_invariant_miou = parse_float(eval_row["ArchiTexture_small_pre_ring_hi_0p3_invariant_miou"])
        architexture_invariant_ari = parse_float(eval_row["ArchiTexture_small_pre_ring_hi_0p3_invariant_ari"])

        delta_direct_miou = None if texturesam_direct_miou is None or architexture_direct_miou is None else architexture_direct_miou - texturesam_direct_miou
        delta_direct_ari = None if texturesam_direct_ari is None or architexture_direct_ari is None else architexture_direct_ari - texturesam_direct_ari
        delta_invariant_miou = None if texturesam_invariant_miou is None or architexture_invariant_miou is None else architexture_invariant_miou - texturesam_invariant_miou
        delta_invariant_ari = None if texturesam_invariant_ari is None or architexture_invariant_ari is None else architexture_invariant_ari - texturesam_invariant_ari

        status, status_label, summary_line, note_line = summarize_case(
            texturesam_present=texturesam_present,
            delta_direct_miou=delta_direct_miou,
            delta_direct_ari=delta_direct_ari,
            delta_invariant_miou=delta_invariant_miou,
            delta_invariant_ari=delta_invariant_ari,
        )

        raw_image_out = raw_dirs["images"] / f"{image_id}.jpg"
        raw_gt_out = raw_dirs["gt"] / f"{image_id}.png"
        raw_edge_out = raw_dirs["edges"] / f"{image_id}.png"
        raw_texturesam_union_out = raw_dirs["texturesam_union"] / f"{image_id}.png"
        raw_architexture_out = raw_dirs["architexture"] / f"{image_id}.png"

        save_web_image(image_path, raw_image_out, image_size)
        save_web_mask(gt_path, raw_gt_out, image_size)
        if edge_path.exists():
            save_web_mask(edge_path, raw_edge_out, image_size)
        if texturesam_union_path.exists():
            save_web_mask(texturesam_union_path, raw_texturesam_union_out, image_size)
        save_web_mask(architexture_path, raw_architexture_out, image_size)

        base = Image.open(raw_image_out).convert("RGB")
        gt_overlay_path = detail_dirs["gt"] / f"{image_id}.png"
        edge_overlay_path = detail_dirs["edge"] / f"{image_id}.png"
        texturesam_union_overlay_path = detail_dirs["texturesam_union"] / f"{image_id}.png"
        architexture_overlay_path = detail_dirs["architexture"] / f"{image_id}.png"

        save_panel(overlay_panel(base, load_mask(raw_gt_out), "GT foreground", "#15936f"), gt_overlay_path)
        if edge_path.exists():
            save_panel(overlay_panel(base, load_mask(raw_edge_out), "Boundary mask", "#6a7b87"), edge_overlay_path)
        else:
            save_panel(blank_panel(base.size, "Boundary mask"), edge_overlay_path)
        save_panel(
            panel_from_mask(
                base,
                raw_texturesam_union_out if texturesam_union_path.exists() else None,
                "TextureSAM union view",
                "#2c7be5",
            ),
            texturesam_union_overlay_path,
        )
        save_panel(overlay_panel(base, load_mask(raw_architexture_out), "ArchiTexture", "#d46e2b"), architexture_overlay_path)

        entries.append(
            Entry(
                image_id=image_id,
                status=status,
                status_label=status_label,
                texturesam_present=texturesam_present,
                proposal_count=parse_int(eval_row["TextureSAM_0p3_mask_count"]),
                merged_components=parse_int(meta_row.get("merged_components")),
                texturesam_direct_miou=texturesam_direct_miou,
                texturesam_direct_ari=texturesam_direct_ari,
                texturesam_invariant_miou=texturesam_invariant_miou,
                texturesam_invariant_ari=texturesam_invariant_ari,
                architexture_direct_miou=architexture_direct_miou,
                architexture_direct_ari=architexture_direct_ari,
                architexture_invariant_miou=architexture_invariant_miou,
                architexture_invariant_ari=architexture_invariant_ari,
                delta_direct_miou=delta_direct_miou,
                delta_direct_ari=delta_direct_ari,
                delta_invariant_miou=delta_invariant_miou,
                delta_invariant_ari=delta_invariant_ari,
                beats_direct_miou=(delta_direct_miou or 0.0) > 0.0,
                beats_direct_ari=(delta_direct_ari or 0.0) > 0.0,
                beats_invariant_miou=(delta_invariant_miou or 0.0) > 0.0,
                beats_invariant_ari=(delta_invariant_ari or 0.0) > 0.0,
                summary_line=summary_line,
                note_line=note_line,
                image_rgb=relative_asset(raw_image_out, out_dir),
                gt_overlay=relative_asset(gt_overlay_path, out_dir),
                edge_overlay=relative_asset(edge_overlay_path, out_dir),
                texturesam_union_overlay=relative_asset(texturesam_union_overlay_path, out_dir),
                architexture_overlay=relative_asset(architexture_overlay_path, out_dir),
                raw_image=relative_asset(raw_image_out, out_dir),
                raw_gt=relative_asset(raw_gt_out, out_dir),
                raw_edge=relative_asset(raw_edge_out, out_dir) if edge_path.exists() else None,
                raw_texturesam_union=relative_asset(raw_texturesam_union_out, out_dir) if texturesam_union_path.exists() else None,
                raw_architexture=relative_asset(raw_architexture_out, out_dir),
            )
        )

    return entries


def write_gallery(
    *,
    out_dir: Path,
    entries: list[Entry],
    summary_json_path: Path,
    summary_json: dict,
    page_title: str,
    hero_kicker: str,
    subtitle: str,
    home_href: str,
    sam2_summary_path: Path | None,
    sam2_summary: dict | None,
    sam2_label: str,
) -> None:
    manifest = [
        {
            "image_id": entry.image_id,
            "status": entry.status,
            "status_label": entry.status_label,
            "texturesam_present": entry.texturesam_present,
            "proposal_count": entry.proposal_count,
            "merged_components": entry.merged_components,
            "texturesam_direct_miou": entry.texturesam_direct_miou,
            "texturesam_direct_ari": entry.texturesam_direct_ari,
            "texturesam_invariant_miou": entry.texturesam_invariant_miou,
            "texturesam_invariant_ari": entry.texturesam_invariant_ari,
            "architexture_direct_miou": entry.architexture_direct_miou,
            "architexture_direct_ari": entry.architexture_direct_ari,
            "architexture_invariant_miou": entry.architexture_invariant_miou,
            "architexture_invariant_ari": entry.architexture_invariant_ari,
            "delta_direct_miou": entry.delta_direct_miou,
            "delta_direct_ari": entry.delta_direct_ari,
            "delta_invariant_miou": entry.delta_invariant_miou,
            "delta_invariant_ari": entry.delta_invariant_ari,
            "beats_direct_miou": entry.beats_direct_miou,
            "beats_direct_ari": entry.beats_direct_ari,
            "beats_invariant_miou": entry.beats_invariant_miou,
            "beats_invariant_ari": entry.beats_invariant_ari,
            "summary_line": entry.summary_line,
            "note_line": entry.note_line,
            "image_rgb": entry.image_rgb,
            "gt_overlay": entry.gt_overlay,
            "edge_overlay": entry.edge_overlay,
            "texturesam_union_overlay": entry.texturesam_union_overlay,
            "architexture_overlay": entry.architexture_overlay,
            "raw_image": entry.raw_image,
            "raw_gt": entry.raw_gt,
            "raw_edge": entry.raw_edge,
            "raw_texturesam_union": entry.raw_texturesam_union,
            "raw_architexture": entry.raw_architexture,
        }
        for entry in entries
    ]
    manifest_path = out_dir / "assets" / "controlnet_manifest.json"
    summary_out_path = out_dir / "assets" / summary_json_path.name
    page_summary_path = out_dir / "assets" / "controlnet_manifest_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    copy_file(summary_json_path, summary_out_path)
    if sam2_summary_path is not None:
        copy_file(sam2_summary_path, out_dir / "assets" / sam2_summary_path.name)
    page_summary = maybe_add_sam2_original(build_page_summary(entries, summary_json), sam2_summary, sam2_label)
    page_summary_path.write_text(json.dumps(page_summary, indent=2), encoding="utf-8")

    if not TEMPLATE_STYLES.exists():
        raise FileNotFoundError(f"Missing gallery style template: {TEMPLATE_STYLES}")
    copy_file(TEMPLATE_STYLES, out_dir / "styles.css")

    html_text = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$page_title</title>
  <link rel="stylesheet" href="./styles.css" />
</head>
<body>
  <header class="hero wrap">
    <p class="kicker">$hero_kicker</p>
    <h1>Browse All ControlNet-Stitched Images, Inspect One in Depth</h1>
    <p>$subtitle</p>
    <div class="links">
      <a href="./assets/controlnet_manifest.json">Download Manifest</a>
      <a href="./assets/$summary_name">Download Metric Summary</a>
      <a href="$home_href">Results Home</a>
    </div>
  </header>

  <main class="wrap">
    <section class="panel stats-panel">
      <div class="stats">
        <div class="stat"><div class="value" id="statCount">-</div><div class="label">Images</div></div>
        <div class="stat"><div class="value" id="statCoverage">-</div><div class="label">TextureSAM coverage</div></div>
        <div class="stat"><div class="value" id="statTsDirect">-</div><div class="label">TextureSAM direct mIoU / ARI</div></div>
        <div class="stat"><div class="value" id="statArchDirect">-</div><div class="label">ArchiTexture direct mIoU / ARI</div></div>
        <div class="stat"><div class="value" id="statTsInv">-</div><div class="label">TextureSAM invariant mIoU / ARI</div></div>
        <div class="stat"><div class="value" id="statArchInv">-</div><div class="label">ArchiTexture invariant mIoU / ARI</div></div>
        <div class="stat"><div class="value" id="statSam2">-</div><div class="label" id="statSam2Label">SAM2 original invariant mIoU / ARI</div></div>
      </div>
      <p class="note">This page uses the ControlNet-stitched PTD benchmark: one transition between two PTD texture samples, regenerated into more realistic images with ControlNet. Direct metrics score the provided canonical foreground mask. Partition-invariant metrics allow complement matching and are often the more faithful view of a two-texture binary partition.</p>
    </section>

    <section class="explorer">
      <aside class="panel browser">
        <div class="section-head">
          <h2>ControlNet Gallery</h2>
          <p id="galleryMeta">Loading...</p>
        </div>

        <div class="browser-controls">
          <label class="control">
            <span>Search image ID</span>
            <input id="searchInput" type="search" inputmode="numeric" placeholder="e.g., 224" />
          </label>

          <label class="control">
            <span>Subset</span>
            <select id="subsetSelect">
              <option value="all">All images</option>
              <option value="covered">Covered only</option>
              <option value="texturesam_miss">TextureSAM misses</option>
            </select>
          </label>

          <label class="control">
            <span>Focus</span>
            <select id="focusSelect">
              <option value="all">All cases</option>
              <option value="inv_win_both">ArchiTexture wins both invariant metrics</option>
              <option value="direct_win_iou">ArchiTexture wins direct IoU</option>
              <option value="texturesam_direct_ari">TextureSAM wins direct ARI</option>
              <option value="hard_case">ArchiTexture invariant IoU below 0.50</option>
            </select>
          </label>

          <label class="control">
            <span>Sort by</span>
            <select id="sortSelect">
              <option value="image_id">Image ID</option>
              <option value="delta_invariant_miou">Delta invariant mIoU</option>
              <option value="delta_direct_miou">Delta direct mIoU</option>
              <option value="proposal_count">TextureSAM mask count</option>
              <option value="arch_invariant_miou">ArchiTexture invariant mIoU</option>
            </select>
          </label>

          <label class="control control-range">
            <span>Min invariant mIoU gain: <strong id="minDeltaValue">-1.00</strong></span>
            <input id="minDelta" type="range" min="-1" max="1" value="-1" step="0.01" />
          </label>

          <div class="control flags">
            <span>Quick flags</span>
            <label class="toggle-row"><input type="checkbox" id="flagDirectIoU" /> direct IoU win</label>
            <label class="toggle-row"><input type="checkbox" id="flagInvariantAri" /> invariant ARI win</label>
            <label class="toggle-row"><input type="checkbox" id="flagMiss" /> TextureSAM miss</label>
          </div>

          <button id="resetBtn" type="button">Reset filters</button>
        </div>

        <div id="thumbGrid" class="thumb-grid"></div>
      </aside>

      <section class="panel detail" id="detailPanel">
        <div class="section-head">
          <h2 id="detailTitle">Case -</h2>
          <p id="detailSubtitle">Select an image from the gallery.</p>
        </div>

        <div class="detail-hero">
          <img id="detailHeroImage" alt="Selected image" />
        </div>

        <div id="detailBadges" class="badges"></div>
        <div id="detailMetrics" class="metric-row"></div>
        <p id="detailWhy" class="why"></p>
        <p id="detailNote" class="note"></p>

        <h3>Diagnostic Views</h3>
        <div id="diagGrid" class="diag-grid"></div>

        <details class="dictionary" open>
          <summary>Dictionary: How to Read This Page</summary>
          <dl>
            <dt>Direct metrics</dt><dd>IoU and ARI against the provided canonical foreground mask.</dd>
            <dt>Invariant metrics</dt><dd>IoU and ARI after allowing complement matching, which is often fairer for a two-region binary partition.</dd>
            <dt>TextureSAM union view</dt><dd>A visualization-only union of TextureSAM's mask bank. The reported TextureSAM numbers on this page are still computed from the full mask bank, not from this union image.</dd>
            <dt>TextureSAM miss</dt><dd>The public TextureSAM checkpoint emitted no masks for this image. ArchiTexture still returns one region because it always outputs a single final mask.</dd>
            <dt>Boundary mask</dt><dd>The released stitched-edge annotation for the same image.</dd>
          </dl>
        </details>

        <div id="downloads" class="downloads"></div>
      </section>
    </section>
  </main>

  <script>
    (function () {
      const state = { data: [], filtered: [], selectedId: null, minDelta: -1 };
      const els = {
        searchInput: document.getElementById('searchInput'),
        subsetSelect: document.getElementById('subsetSelect'),
        focusSelect: document.getElementById('focusSelect'),
        sortSelect: document.getElementById('sortSelect'),
        minDelta: document.getElementById('minDelta'),
        minDeltaValue: document.getElementById('minDeltaValue'),
        flagDirectIoU: document.getElementById('flagDirectIoU'),
        flagInvariantAri: document.getElementById('flagInvariantAri'),
        flagMiss: document.getElementById('flagMiss'),
        resetBtn: document.getElementById('resetBtn'),
        thumbGrid: document.getElementById('thumbGrid'),
        galleryMeta: document.getElementById('galleryMeta'),
        detailTitle: document.getElementById('detailTitle'),
        detailSubtitle: document.getElementById('detailSubtitle'),
        detailHeroImage: document.getElementById('detailHeroImage'),
        detailBadges: document.getElementById('detailBadges'),
        detailMetrics: document.getElementById('detailMetrics'),
        detailWhy: document.getElementById('detailWhy'),
        detailNote: document.getElementById('detailNote'),
        diagGrid: document.getElementById('diagGrid'),
        downloads: document.getElementById('downloads'),
        statCount: document.getElementById('statCount'),
        statCoverage: document.getElementById('statCoverage'),
        statTsDirect: document.getElementById('statTsDirect'),
        statArchDirect: document.getElementById('statArchDirect'),
        statTsInv: document.getElementById('statTsInv'),
        statArchInv: document.getElementById('statArchInv'),
        statSam2: document.getElementById('statSam2'),
        statSam2Label: document.getElementById('statSam2Label')
      };

      function num(v) { const x = Number(v); return Number.isFinite(x) ? x : null; }
      function fmt(v, d) { const x = num(v); return x === null ? 'n/a' : x.toFixed(d); }
      function fmtSigned(v, d) { const x = num(v); if (x === null) return 'n/a'; return (x >= 0 ? '+' : '') + x.toFixed(d); }
      function boolFlag(row, key) { return row[key] === true || row[key] === 1 || row[key] === '1'; }

      function rowMatches(row) {
        const q = els.searchInput.value.trim();
        if (q && !String(row.image_id).includes(q)) return false;
        if (els.subsetSelect.value === 'covered' && row.status !== 'covered') return false;
        if (els.subsetSelect.value === 'texturesam_miss' && row.status !== 'texturesam_miss') return false;

        if (els.focusSelect.value === 'inv_win_both' && !(boolFlag(row, 'beats_invariant_miou') && boolFlag(row, 'beats_invariant_ari'))) return false;
        if (els.focusSelect.value === 'direct_win_iou' && !boolFlag(row, 'beats_direct_miou')) return false;
        if (els.focusSelect.value === 'texturesam_direct_ari' && boolFlag(row, 'beats_direct_ari')) return false;
        if (els.focusSelect.value === 'hard_case') {
          const arch = num(row.architexture_invariant_miou);
          if (arch === null || arch >= 0.5) return false;
        }

        if (els.flagDirectIoU.checked && !boolFlag(row, 'beats_direct_miou')) return false;
        if (els.flagInvariantAri.checked && !boolFlag(row, 'beats_invariant_ari')) return false;
        if (els.flagMiss.checked && row.status !== 'texturesam_miss') return false;

        if (state.minDelta > -1) {
          const delta = num(row.delta_invariant_miou);
          if (delta === null || delta < state.minDelta) return false;
        }
        return true;
      }

      function sortRows(a, b) {
        const key = els.sortSelect.value;
        if (key === 'delta_invariant_miou') return (num(b.delta_invariant_miou) ?? -999) - (num(a.delta_invariant_miou) ?? -999);
        if (key === 'delta_direct_miou') return (num(b.delta_direct_miou) ?? -999) - (num(a.delta_direct_miou) ?? -999);
        if (key === 'proposal_count') return (num(b.proposal_count) ?? -999) - (num(a.proposal_count) ?? -999);
        if (key === 'arch_invariant_miou') return (num(b.architexture_invariant_miou) ?? -999) - (num(a.architexture_invariant_miou) ?? -999);
        return Number(a.image_id) - Number(b.image_id);
      }

      function addBadge(parent, text, cls) {
        const span = document.createElement('span');
        span.className = 'badge ' + (cls || '');
        span.textContent = text;
        parent.appendChild(span);
      }

      function addMetric(parent, label, value) {
        const span = document.createElement('span');
        span.innerHTML = label + ' <strong>' + value + '</strong>';
        parent.appendChild(span);
      }

      function makeTile(label, path) {
        if (!path) return null;
        const tile = document.createElement('div');
        tile.className = 'diag-tile';
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = path;
        img.alt = label;
        const cap = document.createElement('p');
        cap.textContent = label;
        tile.appendChild(img);
        tile.appendChild(cap);
        img.addEventListener('error', function () { tile.remove(); });
        return tile;
      }

      function updateFiltered() {
        state.filtered = state.data.filter(rowMatches).sort(sortRows);
        if (!state.filtered.length) { state.selectedId = null; return; }
        const visible = state.filtered.some(function (r) { return Number(r.image_id) === Number(state.selectedId); });
        if (!visible) state.selectedId = Number(state.filtered[0].image_id);
      }

      function renderThumbGrid() {
        els.thumbGrid.innerHTML = '';
        state.filtered.forEach(function (row) {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'thumb-card' + (Number(row.image_id) === Number(state.selectedId) ? ' active' : '');
          const img = document.createElement('img');
          img.loading = 'lazy';
          img.src = row.image_rgb || '';
          img.alt = 'Case ' + row.image_id;
          const cap = document.createElement('div');
          cap.className = 'thumb-cap';
          const delta = num(row.delta_invariant_miou);
          cap.textContent = row.status === 'texturesam_miss'
            ? 'ID ' + row.image_id + ' | miss'
            : 'ID ' + row.image_id + ' | ' + (delta === null ? 'n/a' : ((delta >= 0 ? '+' : '') + delta.toFixed(2)));
          btn.appendChild(img);
          btn.appendChild(cap);
          btn.addEventListener('click', function () { state.selectedId = Number(row.image_id); render(); });
          els.thumbGrid.appendChild(btn);
        });
        els.galleryMeta.textContent = 'Showing ' + state.filtered.length + ' / ' + state.data.length + ' images';
      }

      function renderDetail() {
        const row = state.filtered.find(function (r) { return Number(r.image_id) === Number(state.selectedId); });
        if (!row) {
          els.detailTitle.textContent = 'Case -';
          els.detailSubtitle.textContent = 'No image matches current filters.';
          els.detailHeroImage.removeAttribute('src');
          els.detailBadges.innerHTML = '';
          els.detailMetrics.innerHTML = '';
          els.detailWhy.textContent = '';
          els.detailNote.textContent = '';
          els.diagGrid.innerHTML = '';
          els.downloads.innerHTML = '';
          return;
        }

        els.detailTitle.textContent = 'Case ' + row.image_id;
        els.detailSubtitle.textContent = row.status_label || 'case';
        els.detailHeroImage.src = row.image_rgb || '';
        els.detailHeroImage.alt = 'Case ' + row.image_id;

        els.detailBadges.innerHTML = '';
        addBadge(els.detailBadges, row.status_label || 'case', row.status === 'texturesam_miss' ? 'gray' : 'blue');
        addBadge(els.detailBadges, 'TextureSAM masks ' + (row.proposal_count ?? 'n/a'));
        addBadge(els.detailBadges, 'merged components ' + (row.merged_components ?? 'n/a'));
        if (boolFlag(row, 'beats_direct_miou')) addBadge(els.detailBadges, 'direct IoU win', 'orange');
        if (boolFlag(row, 'beats_invariant_miou')) addBadge(els.detailBadges, 'invariant IoU win', 'blue');
        if (boolFlag(row, 'beats_invariant_ari')) addBadge(els.detailBadges, 'invariant ARI win', 'blue');

        els.detailMetrics.innerHTML = '';
        addMetric(els.detailMetrics, 'TextureSAM direct mIoU', fmt(row.texturesam_direct_miou, 4));
        addMetric(els.detailMetrics, 'ArchiTexture direct mIoU', fmt(row.architexture_direct_miou, 4));
        addMetric(els.detailMetrics, 'Delta direct mIoU', fmtSigned(row.delta_direct_miou, 4));
        addMetric(els.detailMetrics, 'TextureSAM direct ARI', fmt(row.texturesam_direct_ari, 4));
        addMetric(els.detailMetrics, 'ArchiTexture direct ARI', fmt(row.architexture_direct_ari, 4));
        addMetric(els.detailMetrics, 'Delta direct ARI', fmtSigned(row.delta_direct_ari, 4));
        addMetric(els.detailMetrics, 'TextureSAM invariant mIoU', fmt(row.texturesam_invariant_miou, 4));
        addMetric(els.detailMetrics, 'ArchiTexture invariant mIoU', fmt(row.architexture_invariant_miou, 4));
        addMetric(els.detailMetrics, 'Delta invariant mIoU', fmtSigned(row.delta_invariant_miou, 4));
        addMetric(els.detailMetrics, 'TextureSAM invariant ARI', fmt(row.texturesam_invariant_ari, 4));
        addMetric(els.detailMetrics, 'ArchiTexture invariant ARI', fmt(row.architexture_invariant_ari, 4));
        addMetric(els.detailMetrics, 'Delta invariant ARI', fmtSigned(row.delta_invariant_ari, 4));

        els.detailWhy.textContent = row.summary_line || '';
        els.detailNote.textContent = row.note_line || '';

        els.diagGrid.innerHTML = '';
        [
          ['GT foreground', row.gt_overlay],
          ['Boundary mask', row.edge_overlay],
          ['TextureSAM union view', row.texturesam_union_overlay],
          ['ArchiTexture', row.architexture_overlay]
        ].forEach(function (item) {
          const tile = makeTile(item[0], item[1]);
          if (tile) els.diagGrid.appendChild(tile);
        });

        els.downloads.innerHTML = '';
        [
          ['Image', row.raw_image],
          ['GT mask', row.raw_gt],
          ['Boundary mask', row.raw_edge],
          ['TextureSAM union mask', row.raw_texturesam_union],
          ['ArchiTexture mask', row.raw_architexture]
        ].forEach(function (pair) {
          if (!pair[1]) return;
          const a = document.createElement('a');
          a.href = pair[1];
          a.textContent = pair[0];
          els.downloads.appendChild(a);
        });
      }

      function setStats(summary) {
        els.statCount.textContent = String(summary.num_images ?? state.data.length);
        els.statCoverage.textContent = String(summary.texturesam_coverage) + ' / ' + String(summary.num_images);
        els.statTsDirect.textContent = fmt(summary.texturesam_direct_all_miou, 4) + ' / ' + fmt(summary.texturesam_direct_all_ari, 4);
        els.statArchDirect.textContent = fmt(summary.architexture_direct_all_miou, 4) + ' / ' + fmt(summary.architexture_direct_all_ari, 4);
        els.statTsInv.textContent = fmt(summary.texturesam_invariant_all_miou, 4) + ' / ' + fmt(summary.texturesam_invariant_all_ari, 4);
        els.statArchInv.textContent = fmt(summary.architexture_invariant_all_miou, 4) + ' / ' + fmt(summary.architexture_invariant_all_ari, 4);
        if (summary.sam2_original_label) {
          els.statSam2Label.textContent = summary.sam2_original_label + ' invariant mIoU / ARI';
          els.statSam2.textContent = fmt(summary.sam2_original_invariant_miou_all, 4) + ' / ' + fmt(summary.sam2_original_invariant_ari_all, 4);
        } else {
          els.statSam2.textContent = 'n/a';
        }
      }

      function bindControls() {
        els.searchInput.addEventListener('input', render);
        els.subsetSelect.addEventListener('change', render);
        els.focusSelect.addEventListener('change', render);
        els.sortSelect.addEventListener('change', render);
        els.flagDirectIoU.addEventListener('change', render);
        els.flagInvariantAri.addEventListener('change', render);
        els.flagMiss.addEventListener('change', render);
        els.minDelta.addEventListener('input', function () {
          state.minDelta = parseFloat(els.minDelta.value);
          els.minDeltaValue.textContent = Number(state.minDelta).toFixed(2);
          render();
        });
        els.resetBtn.addEventListener('click', function () {
          els.searchInput.value = '';
          els.subsetSelect.value = 'all';
          els.focusSelect.value = 'all';
          els.sortSelect.value = 'image_id';
          els.flagDirectIoU.checked = false;
          els.flagInvariantAri.checked = false;
          els.flagMiss.checked = false;
          state.minDelta = -1;
          els.minDelta.value = '-1';
          els.minDeltaValue.textContent = '-1.00';
          render();
        });
      }

      function render() {
        updateFiltered();
        renderThumbGrid();
        renderDetail();
      }

      async function init() {
        try {
          const [dataResp, sumResp] = await Promise.all([
            fetch('./assets/controlnet_manifest.json', { cache: 'no-store' }),
            fetch('./assets/controlnet_manifest_summary.json', { cache: 'no-store' })
          ]);
          if (!dataResp.ok) throw new Error('Could not load manifest');
          if (!sumResp.ok) throw new Error('Could not load summary');
          const data = await dataResp.json();
          const summary = await sumResp.json();
          if (!Array.isArray(data) || !data.length) throw new Error('Empty manifest');
          state.data = data.slice().sort(sortRows);
          state.selectedId = Number(state.data[0].image_id);
          setStats(summary);
          bindControls();
          render();
        } catch (err) {
          els.galleryMeta.textContent = 'Failed to load';
          els.detailSubtitle.textContent = String(err);
        }
      }

      init();
    })();
  </script>
</body>
</html>
"""
    ).substitute(
        page_title=page_title,
        hero_kicker=hero_kicker,
        subtitle=subtitle,
        summary_name=summary_json_path.name,
        home_href=home_href,
    )
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.site_root / args.out_subdir
    ensure_clean_dir(out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    summary_json = load_json(args.summary_json)
    sam2_summary = load_json(args.sam2_original_summary) if args.sam2_original_summary else None
    entries = build_manifest(
        out_dir=out_dir,
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        edges_dir=args.edges_dir,
        texturesam_union_dir=args.texturesam_union_dir,
        architexture_masks=args.architexture_masks,
        per_image_csv=args.per_image_csv,
        metadata_csv=args.metadata_csv,
        image_size=args.image_size,
    )

    write_gallery(
        out_dir=out_dir,
        entries=entries,
        summary_json_path=args.summary_json,
        summary_json=summary_json,
        page_title=args.page_title,
        hero_kicker=args.hero_kicker,
        subtitle=args.subtitle,
        home_href=args.home_href,
        sam2_summary_path=args.sam2_original_summary,
        sam2_summary=sam2_summary,
        sam2_label=args.sam2_original_label,
    )


if __name__ == "__main__":
    main()
