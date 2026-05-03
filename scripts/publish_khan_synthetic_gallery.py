#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

from PIL import Image

from build_texturesam2_ai_docs import copy_file, ensure_clean_dir, fit_rgb, load_mask, overlay_panel


TEMPLATE_STYLES = Path("external/rwtd_miner_public_site/docs/texturesam2_ai_gallery/styles.css")


@dataclass
class Entry:
    image_id: int
    proposal_union_present: bool
    proposal_count: int
    merged_components: int
    proposal_union_iou: float | None
    proposal_union_ari: float | None
    handcrafted_iou: float | None
    handcrafted_ari: float | None
    heuristic_iou: float | None
    heuristic_ari: float | None
    architexture_iou: float | None
    architexture_ari: float | None
    delta_iou_vs_union: float | None
    delta_ari_vs_union: float | None
    beats_union: bool
    beats_handcrafted: bool
    beats_heuristic: bool
    status: str
    status_label: str
    summary_line: str
    note_line: str
    image_rgb: str
    gt_overlay: str
    proposal_union_overlay: str
    handcrafted_overlay: str
    heuristic_overlay: str
    architexture_overlay: str
    raw_image: str
    raw_gt: str
    raw_proposal_union: str | None
    raw_handcrafted: str | None
    raw_heuristic: str | None
    raw_architexture: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Publish an STLD synthetic explorer using the same browser/detail template as the RWTD gallery."
        )
    )
    p.add_argument("--site-root", type=Path, required=True)
    p.add_argument("--out-subdir", type=str, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--proposal-union-masks", type=Path, required=True)
    p.add_argument("--handcrafted-masks", type=Path, required=True)
    p.add_argument("--heuristic-masks", type=Path, required=True)
    p.add_argument("--architexture-masks", type=Path, required=True)
    p.add_argument("--per-image-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--page-title", type=str, required=True)
    p.add_argument("--hero-kicker", type=str, required=True)
    p.add_argument("--subtitle", type=str, required=True)
    p.add_argument("--home-href", type=str, default="../index.html")
    p.add_argument("--sam2-original-summary", type=Path)
    p.add_argument("--sam2-original-label", type=str, default="SAM2 original")
    return p.parse_args()


def load_rows_by_id(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(row["image_id"]): row for row in csv.DictReader(f)}


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def parse_int(value: str | None) -> int:
    if value is None:
        return 0
    text = value.strip()
    if not text:
        return 0
    return int(float(text))


def blank_panel(size: tuple[int, int], label: str, color: str) -> Image.Image:
    blank = Image.new("RGB", size, (241, 244, 246))
    return overlay_panel(blank, None, label, color)


def save_panel(panel: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def panel_from_mask(base: Image.Image, mask_path: Path | None, label: str, color: str) -> Image.Image:
    if mask_path is None or not mask_path.exists():
        return blank_panel(base.size, label, color)
    return overlay_panel(base, load_mask(mask_path), label, color)


def relative_asset(path: Path, out_dir: Path) -> str:
    return "./" + path.relative_to(out_dir).as_posix()


def summarize_case(
    *,
    proposal_union_present: bool,
    beats_union: bool,
    beats_handcrafted: bool,
    beats_heuristic: bool,
    delta_iou_vs_union: float | None,
) -> tuple[str, str, str]:
    if not proposal_union_present:
        return (
            "proposal_bank_miss",
            "Proposal-bank miss",
            "Frozen SAM emitted no proposal masks. ArchiTexture is shown through its no-proposal fallback path.",
        )
    if beats_union and beats_handcrafted and beats_heuristic:
        return (
            "covered",
            "Covered comparison",
            "ArchiTexture beats proposal union, handcrafted consolidation, and the PTD heuristic on the canonical STLD foreground target.",
        )
    if beats_union:
        return (
            "covered",
            "Covered comparison",
            "ArchiTexture improves on proposal union on the canonical STLD foreground target, but not on every alternative route.",
        )
    if delta_iou_vs_union is not None and delta_iou_vs_union <= -0.1:
        return (
            "covered",
            "Covered comparison",
            "This is a hard covered case where the learned consolidation falls noticeably below raw proposal union.",
        )
    return (
        "covered",
        "Covered comparison",
        "This is a covered STLD case scored directly against the canonical foreground mask without partition flipping.",
    )


def build_manifest(
    *,
    out_dir: Path,
    images_dir: Path,
    gt_dir: Path,
    proposal_union_masks: Path,
    handcrafted_masks: Path,
    heuristic_masks: Path,
    architexture_masks: Path,
    per_image_csv: Path,
    metadata_csv: Path,
) -> list[Entry]:
    direct_rows = load_rows_by_id(per_image_csv)
    metadata_rows = load_rows_by_id(metadata_csv)

    raw_dirs = {
        "images": out_dir / "assets" / "raw" / "images",
        "gt": out_dir / "assets" / "raw" / "gt",
        "proposal_union": out_dir / "assets" / "raw" / "proposal_union",
        "handcrafted": out_dir / "assets" / "raw" / "handcrafted",
        "heuristic": out_dir / "assets" / "raw" / "heuristic",
        "architexture": out_dir / "assets" / "raw" / "architexture",
    }
    detail_dirs = {
        "gt": out_dir / "assets" / "detail" / "gt",
        "proposal_union": out_dir / "assets" / "detail" / "proposal_union",
        "handcrafted": out_dir / "assets" / "detail" / "handcrafted",
        "heuristic": out_dir / "assets" / "detail" / "heuristic",
        "architexture": out_dir / "assets" / "detail" / "architexture",
    }
    for path in list(raw_dirs.values()) + list(detail_dirs.values()):
        path.mkdir(parents=True, exist_ok=True)

    entries: list[Entry] = []
    image_paths = sorted(
        (
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda p: int(p.stem),
    )

    for image_path in image_paths:
        image_id = int(image_path.stem)
        gt_path = gt_dir / f"{image_id}.png"
        proposal_union_path = proposal_union_masks / f"{image_id}.png"
        handcrafted_path = handcrafted_masks / f"{image_id}.png"
        heuristic_path = heuristic_masks / f"{image_id}.png"
        architexture_path = architexture_masks / f"{image_id}.png"
        if not gt_path.exists() or not architexture_path.exists():
            raise FileNotFoundError(f"Missing required STLD assets for image {image_id}")

        direct_row = direct_rows[image_id]
        metadata_row = metadata_rows.get(image_id, {})
        proposal_union_present = parse_bool(direct_row["proposal_union_present"])

        proposal_union_iou = parse_float(direct_row["proposal_union_iou"])
        proposal_union_ari = parse_float(direct_row["proposal_union_ari"])
        handcrafted_iou = parse_float(direct_row["handcrafted_iou"])
        handcrafted_ari = parse_float(direct_row["handcrafted_ari"])
        heuristic_iou = parse_float(direct_row["ptd_heuristic_iou"])
        heuristic_ari = parse_float(direct_row["ptd_heuristic_ari"])
        architexture_iou = parse_float(direct_row["architexture_iou"])
        architexture_ari = parse_float(direct_row["architexture_ari"])

        delta_iou_vs_union = None
        delta_ari_vs_union = None
        if proposal_union_iou is not None and architexture_iou is not None:
            delta_iou_vs_union = architexture_iou - proposal_union_iou
        if proposal_union_ari is not None and architexture_ari is not None:
            delta_ari_vs_union = architexture_ari - proposal_union_ari

        beats_union = (
            proposal_union_iou is not None
            and architexture_iou is not None
            and architexture_iou > proposal_union_iou
        )
        beats_handcrafted = (
            handcrafted_iou is not None
            and architexture_iou is not None
            and architexture_iou > handcrafted_iou
        )
        beats_heuristic = (
            heuristic_iou is not None
            and architexture_iou is not None
            and architexture_iou > heuristic_iou
        )
        status, status_label, summary_line = summarize_case(
            proposal_union_present=proposal_union_present,
            beats_union=beats_union,
            beats_handcrafted=beats_handcrafted,
            beats_heuristic=beats_heuristic,
            delta_iou_vs_union=delta_iou_vs_union,
        )

        note_line = (
            "Direct STLD metrics use the canonical foreground mask. The page does not invert or relabel partitions to favor any method."
        )
        if not proposal_union_present:
            note_line = (
                "Proposal union is blank because the frozen proposal bank emitted zero masks. This card is separated from normal same-bank comparisons."
            )

        raw_image_name = f"{image_id}{image_path.suffix.lower()}"
        raw_image_out = raw_dirs["images"] / raw_image_name
        raw_gt_out = raw_dirs["gt"] / f"{image_id}.png"
        raw_proposal_union_out = raw_dirs["proposal_union"] / f"{image_id}.png"
        raw_handcrafted_out = raw_dirs["handcrafted"] / f"{image_id}.png"
        raw_heuristic_out = raw_dirs["heuristic"] / f"{image_id}.png"
        raw_architexture_out = raw_dirs["architexture"] / f"{image_id}.png"

        copy_file(image_path, raw_image_out)
        copy_file(gt_path, raw_gt_out)
        if proposal_union_path.exists():
            copy_file(proposal_union_path, raw_proposal_union_out)
        if handcrafted_path.exists():
            copy_file(handcrafted_path, raw_handcrafted_out)
        if heuristic_path.exists():
            copy_file(heuristic_path, raw_heuristic_out)
        if architexture_path.exists():
            copy_file(architexture_path, raw_architexture_out)

        base = fit_rgb(image_path)
        gt_overlay_path = detail_dirs["gt"] / f"{image_id}.png"
        proposal_union_overlay_path = detail_dirs["proposal_union"] / f"{image_id}.png"
        handcrafted_overlay_path = detail_dirs["handcrafted"] / f"{image_id}.png"
        heuristic_overlay_path = detail_dirs["heuristic"] / f"{image_id}.png"
        architexture_overlay_path = detail_dirs["architexture"] / f"{image_id}.png"

        save_panel(overlay_panel(base, load_mask(gt_path), "GT foreground", "#15936f"), gt_overlay_path)
        save_panel(panel_from_mask(base, proposal_union_path if proposal_union_path.exists() else None, "Proposal union", "#2c7be5"), proposal_union_overlay_path)
        save_panel(panel_from_mask(base, handcrafted_path if handcrafted_path.exists() else None, "Handcrafted", "#6a7b87"), handcrafted_overlay_path)
        save_panel(panel_from_mask(base, heuristic_path if heuristic_path.exists() else None, "PTD heuristic", "#b97a1b"), heuristic_overlay_path)
        save_panel(panel_from_mask(base, architexture_path, "ArchiTexture", "#d46e2b"), architexture_overlay_path)

        entries.append(
            Entry(
                image_id=image_id,
                proposal_union_present=proposal_union_present,
                proposal_count=parse_int(metadata_row.get("proposal_count")),
                merged_components=parse_int(metadata_row.get("merged_components")),
                proposal_union_iou=proposal_union_iou,
                proposal_union_ari=proposal_union_ari,
                handcrafted_iou=handcrafted_iou,
                handcrafted_ari=handcrafted_ari,
                heuristic_iou=heuristic_iou,
                heuristic_ari=heuristic_ari,
                architexture_iou=architexture_iou,
                architexture_ari=architexture_ari,
                delta_iou_vs_union=delta_iou_vs_union,
                delta_ari_vs_union=delta_ari_vs_union,
                beats_union=beats_union,
                beats_handcrafted=beats_handcrafted,
                beats_heuristic=beats_heuristic,
                status=status,
                status_label=status_label,
                summary_line=summary_line,
                note_line=note_line,
                image_rgb=relative_asset(raw_image_out, out_dir),
                gt_overlay=relative_asset(gt_overlay_path, out_dir),
                proposal_union_overlay=relative_asset(proposal_union_overlay_path, out_dir),
                handcrafted_overlay=relative_asset(handcrafted_overlay_path, out_dir),
                heuristic_overlay=relative_asset(heuristic_overlay_path, out_dir),
                architexture_overlay=relative_asset(architexture_overlay_path, out_dir),
                raw_image=relative_asset(raw_image_out, out_dir),
                raw_gt=relative_asset(raw_gt_out, out_dir),
                raw_proposal_union=relative_asset(raw_proposal_union_out, out_dir) if proposal_union_path.exists() else None,
                raw_handcrafted=relative_asset(raw_handcrafted_out, out_dir) if handcrafted_path.exists() else None,
                raw_heuristic=relative_asset(raw_heuristic_out, out_dir) if heuristic_path.exists() else None,
                raw_architexture=relative_asset(raw_architexture_out, out_dir) if architexture_path.exists() else None,
            )
        )

    return entries


def build_page_summary(entries: list[Entry], summary_json: dict) -> dict[str, object]:
    covered = [entry for entry in entries if entry.proposal_union_present]
    return {
        "num_images": int(summary_json["num_images"]),
        "covered_count": int(summary_json["proposal_union_covered"]),
        "miss_count": int(summary_json["proposal_union_missing"]),
        "beats_union_count": int(sum(1 for entry in covered if entry.beats_union)),
        "beats_handcrafted_count": int(sum(1 for entry in covered if entry.beats_handcrafted)),
        "beats_heuristic_count": int(sum(1 for entry in covered if entry.beats_heuristic)),
        "proposal_union_miou_covered": float(summary_json["methods"]["proposal_union"]["covered"]["miou"]),
        "proposal_union_ari_covered": float(summary_json["methods"]["proposal_union"]["covered"]["ari"]),
        "architexture_miou_covered": float(summary_json["methods"]["architexture"]["covered"]["miou"]),
        "architexture_ari_covered": float(summary_json["methods"]["architexture"]["covered"]["ari"]),
        "delta_iou_covered": float(summary_json["delta_vs_union"]["covered_miou"]),
        "delta_ari_covered": float(summary_json["delta_vs_union"]["covered_ari"]),
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
            "sam2_original_miou_all": float(method.get("all", {}).get("miou", 0.0)),
            "sam2_original_ari_all": float(method.get("all", {}).get("ari", 0.0)),
        }
    )
    return page_summary


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
            "proposal_union_present": entry.proposal_union_present,
            "proposal_count": entry.proposal_count,
            "merged_components": entry.merged_components,
            "proposal_union_iou": entry.proposal_union_iou,
            "proposal_union_ari": entry.proposal_union_ari,
            "handcrafted_iou": entry.handcrafted_iou,
            "handcrafted_ari": entry.handcrafted_ari,
            "heuristic_iou": entry.heuristic_iou,
            "heuristic_ari": entry.heuristic_ari,
            "architexture_iou": entry.architexture_iou,
            "architexture_ari": entry.architexture_ari,
            "delta_iou_vs_union": entry.delta_iou_vs_union,
            "delta_ari_vs_union": entry.delta_ari_vs_union,
            "beats_union": entry.beats_union,
            "beats_handcrafted": entry.beats_handcrafted,
            "beats_heuristic": entry.beats_heuristic,
            "summary_line": entry.summary_line,
            "note_line": entry.note_line,
            "image_rgb": entry.image_rgb,
            "gt_overlay": entry.gt_overlay,
            "proposal_union_overlay": entry.proposal_union_overlay,
            "handcrafted_overlay": entry.handcrafted_overlay,
            "heuristic_overlay": entry.heuristic_overlay,
            "architexture_overlay": entry.architexture_overlay,
            "raw_image": entry.raw_image,
            "raw_gt": entry.raw_gt,
            "raw_proposal_union": entry.raw_proposal_union,
            "raw_handcrafted": entry.raw_handcrafted,
            "raw_heuristic": entry.raw_heuristic,
            "raw_architexture": entry.raw_architexture,
        }
        for entry in entries
    ]

    manifest_path = out_dir / "assets" / "stld_manifest.json"
    summary_out_path = out_dir / "assets" / summary_json_path.name
    page_summary_path = out_dir / "assets" / "stld_manifest_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    copy_file(summary_json_path, summary_out_path)
    if sam2_summary_path is not None:
        copy_file(sam2_summary_path, out_dir / "assets" / sam2_summary_path.name)
    page_summary = maybe_add_sam2_original(build_page_summary(entries, summary_json), sam2_summary, sam2_label)
    page_summary_path.write_text(json.dumps(page_summary, indent=2), encoding="utf-8")

    if TEMPLATE_STYLES.exists():
        copy_file(TEMPLATE_STYLES, out_dir / "styles.css")
    else:
        raise FileNotFoundError(f"Missing RWTD gallery styles template: {TEMPLATE_STYLES}")

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
    <h1>Browse All STLD Images, Inspect One in Depth</h1>
    <p>$subtitle</p>
    <div class="links">
      <a href="./assets/stld_manifest.json">Download STLD Manifest</a>
      <a href="./assets/$summary_name">Download Direct STLD Summary</a>
      <a href="$home_href">Results Home</a>
    </div>
  </header>

  <main class="wrap">
    <section class="panel stats-panel">
      <div class="stats">
        <div class="stat"><div class="value" id="statCount">-</div><div class="label">Synthetic images</div></div>
        <div class="stat"><div class="value" id="statCovered">-</div><div class="label">Covered comparisons</div></div>
        <div class="stat"><div class="value" id="statMiss">-</div><div class="label">Proposal-bank misses</div></div>
        <div class="stat"><div class="value" id="statUnion">-</div><div class="label">Proposal union mIoU</div></div>
        <div class="stat"><div class="value" id="statArch">-</div><div class="label">ArchiTexture mIoU</div></div>
        <div class="stat"><div class="value" id="statAri">-</div><div class="label">ArchiTexture ARI</div></div>
        <div class="stat"><div class="value" id="statSam2">-</div><div class="label" id="statSam2Label">SAM2 original mIoU / ARI</div></div>
      </div>
      <p class="note">This page uses STLD (Shape-Tailored Local Descriptors), a held-out synthetic benchmark built from Brodatz textures and MPEG-7 shapes. Metrics are direct foreground IoU and ARI against the canonical foreground mask. Proposal-bank misses are shown, but separated from normal same-bank comparisons because their ArchiTexture output comes from the no-proposal fallback path.</p>
    </section>

    <section class="explorer">
      <aside class="panel browser">
        <div class="section-head">
          <h2>STLD Gallery</h2>
          <p id="galleryMeta">Loading...</p>
        </div>

        <div class="browser-controls">
          <label class="control">
            <span>Search image ID</span>
            <input id="searchInput" type="search" inputmode="numeric" placeholder="e.g., 23" />
          </label>

          <label class="control">
            <span>Subset</span>
            <select id="subsetSelect">
              <option value="all">All images</option>
              <option value="covered">Covered only</option>
              <option value="proposal_bank_miss">Proposal-bank misses</option>
            </select>
          </label>

          <label class="control">
            <span>Focus</span>
            <select id="focusSelect">
              <option value="all">All cases</option>
              <option value="beats_union">Beats proposal union</option>
              <option value="beats_all">Beats all routes</option>
              <option value="hard_case">ArchiTexture IoU below 0.50</option>
            </select>
          </label>

          <label class="control">
            <span>Sort by</span>
            <select id="sortSelect">
              <option value="image_id">Image ID</option>
              <option value="delta_iou">Delta IoU vs union</option>
              <option value="architexture_iou">ArchiTexture IoU</option>
              <option value="proposal_count">Proposal count</option>
            </select>
          </label>

          <label class="control control-range">
            <span>Min IoU gain vs union: <strong id="minDeltaValue">-1.00</strong></span>
            <input id="minDelta" type="range" min="-1" max="1" value="-1" step="0.01" />
          </label>

          <div class="control flags">
            <span>Quick flags</span>
            <label class="toggle-row"><input type="checkbox" id="flagBeatsHandcrafted" /> beats handcrafted</label>
            <label class="toggle-row"><input type="checkbox" id="flagBeatsHeuristic" /> beats PTD heuristic</label>
            <label class="toggle-row"><input type="checkbox" id="flagMiss" /> proposal-bank miss</label>
          </div>

          <button id="resetBtn" type="button">Reset filters</button>
        </div>

        <div id="thumbGrid" class="thumb-grid"></div>
      </aside>

      <section class="panel detail" id="detailPanel">
        <div class="section-head">
          <h2 id="detailTitle">STLD -</h2>
          <p id="detailSubtitle">Select an image from the gallery.</p>
        </div>

        <div class="detail-hero">
          <img id="detailHeroImage" alt="Selected STLD image" />
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
            <dt>Covered comparison</dt><dd>A case where frozen SAM produced at least one proposal, so proposal union and ArchiTexture are directly comparable on the same bank.</dd>
            <dt>Proposal-bank miss</dt><dd>A case where the frozen proposal bank emitted zero masks. Proposal union is blank and ArchiTexture is showing its no-proposal fallback.</dd>
            <dt>Direct IoU / ARI</dt><dd>Scores against the canonical STLD foreground mask. No complement matching or partition flipping is used for display or evaluation.</dd>
            <dt>Proposal union</dt><dd>The raw union of the frozen proposal-bank masks.</dd>
            <dt>Handcrafted</dt><dd>The non-PTD compatibility route from the same strict Stage-A evaluation.</dd>
            <dt>PTD heuristic</dt><dd>The PTD-driven heuristic route from the same evaluation run.</dd>
            <dt>ArchiTexture</dt><dd>The learned consolidation output on top of the same frozen proposal bank.</dd>
          </dl>
        </details>

        <div id="downloads" class="downloads"></div>
      </section>
    </section>
  </main>

  <script>
    (function () {
      const state = {
        data: [],
        filtered: [],
        selectedId: null,
        minDelta: -1
      };

      const els = {
        searchInput: document.getElementById('searchInput'),
        subsetSelect: document.getElementById('subsetSelect'),
        focusSelect: document.getElementById('focusSelect'),
        sortSelect: document.getElementById('sortSelect'),
        minDelta: document.getElementById('minDelta'),
        minDeltaValue: document.getElementById('minDeltaValue'),
        flagBeatsHandcrafted: document.getElementById('flagBeatsHandcrafted'),
        flagBeatsHeuristic: document.getElementById('flagBeatsHeuristic'),
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
        statCovered: document.getElementById('statCovered'),
        statMiss: document.getElementById('statMiss'),
        statUnion: document.getElementById('statUnion'),
        statArch: document.getElementById('statArch'),
        statAri: document.getElementById('statAri'),
        statSam2: document.getElementById('statSam2'),
        statSam2Label: document.getElementById('statSam2Label')
      };

      function num(v) {
        const x = Number(v);
        return Number.isFinite(x) ? x : null;
      }

      function fmt(v, d) {
        const x = num(v);
        return x === null ? 'n/a' : x.toFixed(d);
      }

      function fmtSigned(v, d) {
        const x = num(v);
        if (x === null) return 'n/a';
        return (x >= 0 ? '+' : '') + x.toFixed(d);
      }

      function boolFlag(row, key) {
        return row[key] === true || row[key] === 1 || row[key] === '1';
      }

      function rowMatches(row) {
        const q = els.searchInput.value.trim();
        if (q && !String(row.image_id).includes(q)) return false;

        if (els.subsetSelect.value !== 'all' && row.status !== els.subsetSelect.value) return false;

        if (els.focusSelect.value === 'beats_union' && !boolFlag(row, 'beats_union')) return false;
        if (els.focusSelect.value === 'beats_all' && !(boolFlag(row, 'beats_union') && boolFlag(row, 'beats_handcrafted') && boolFlag(row, 'beats_heuristic'))) return false;
        if (els.focusSelect.value === 'hard_case') {
          const arch = num(row.architexture_iou);
          if (arch === null || arch >= 0.5) return false;
        }

        if (els.flagBeatsHandcrafted.checked && !boolFlag(row, 'beats_handcrafted')) return false;
        if (els.flagBeatsHeuristic.checked && !boolFlag(row, 'beats_heuristic')) return false;
        if (els.flagMiss.checked && row.status !== 'proposal_bank_miss') return false;

        if (state.minDelta > -1) {
          const delta = num(row.delta_iou_vs_union);
          if (delta === null || delta < state.minDelta) return false;
        }

        return true;
      }

      function sortRows(a, b) {
        const key = els.sortSelect.value;
        if (key === 'delta_iou') {
          return (num(b.delta_iou_vs_union) ?? -999) - (num(a.delta_iou_vs_union) ?? -999);
        }
        if (key === 'architexture_iou') {
          return (num(b.architexture_iou) ?? -999) - (num(a.architexture_iou) ?? -999);
        }
        if (key === 'proposal_count') {
          return (num(b.proposal_count) ?? -999) - (num(a.proposal_count) ?? -999);
        }
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
        if (!state.filtered.length) {
          state.selectedId = null;
          return;
        }
        const selectedStillVisible = state.filtered.some(function (r) { return Number(r.image_id) === Number(state.selectedId); });
        if (!selectedStillVisible) state.selectedId = Number(state.filtered[0].image_id);
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
          img.alt = 'STLD ' + row.image_id;

          const cap = document.createElement('div');
          cap.className = 'thumb-cap';
          const delta = num(row.delta_iou_vs_union);
          cap.textContent = row.status === 'proposal_bank_miss'
            ? 'STLD ' + row.image_id + ' | miss'
            : 'STLD ' + row.image_id + ' | ' + (delta === null ? 'n/a' : ((delta >= 0 ? '+' : '') + delta.toFixed(2)));

          btn.appendChild(img);
          btn.appendChild(cap);
          btn.addEventListener('click', function () {
            state.selectedId = Number(row.image_id);
            render();
          });
          els.thumbGrid.appendChild(btn);
        });

        els.galleryMeta.textContent = 'Showing ' + state.filtered.length + ' / ' + state.data.length + ' images';
      }

      function renderDetail() {
        const row = state.filtered.find(function (r) { return Number(r.image_id) === Number(state.selectedId); });
        if (!row) {
          els.detailTitle.textContent = 'STLD -';
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

        els.detailTitle.textContent = 'STLD ' + row.image_id;
        els.detailSubtitle.textContent = row.status_label || 'STLD case';
        els.detailHeroImage.src = row.image_rgb || '';
        els.detailHeroImage.alt = 'STLD ' + row.image_id;

        els.detailBadges.innerHTML = '';
        addBadge(els.detailBadges, row.status_label || 'case', row.status === 'proposal_bank_miss' ? 'gray' : 'blue');
        addBadge(els.detailBadges, 'proposals ' + (row.proposal_count ?? 'n/a'));
        addBadge(els.detailBadges, 'merged components ' + (row.merged_components ?? 'n/a'));
        if (boolFlag(row, 'beats_union')) addBadge(els.detailBadges, 'beats union', 'orange');
        if (boolFlag(row, 'beats_handcrafted')) addBadge(els.detailBadges, 'beats handcrafted', 'blue');
        if (boolFlag(row, 'beats_heuristic')) addBadge(els.detailBadges, 'beats PTD heuristic', 'blue');

        els.detailMetrics.innerHTML = '';
        addMetric(els.detailMetrics, 'Proposal union IoU', fmt(row.proposal_union_iou, 4));
        addMetric(els.detailMetrics, 'Handcrafted IoU', fmt(row.handcrafted_iou, 4));
        addMetric(els.detailMetrics, 'PTD heuristic IoU', fmt(row.heuristic_iou, 4));
        addMetric(els.detailMetrics, 'ArchiTexture IoU', fmt(row.architexture_iou, 4));
        addMetric(els.detailMetrics, 'Proposal union ARI', fmt(row.proposal_union_ari, 4));
        addMetric(els.detailMetrics, 'ArchiTexture ARI', fmt(row.architexture_ari, 4));
        addMetric(els.detailMetrics, 'Delta IoU vs union', fmtSigned(row.delta_iou_vs_union, 4));
        addMetric(els.detailMetrics, 'Delta ARI vs union', fmtSigned(row.delta_ari_vs_union, 4));

        els.detailWhy.textContent = row.summary_line || '';
        els.detailNote.textContent = row.note_line || '';

        els.diagGrid.innerHTML = '';
        [
          ['GT foreground', row.gt_overlay],
          ['Proposal union', row.proposal_union_overlay],
          ['Handcrafted', row.handcrafted_overlay],
          ['PTD heuristic', row.heuristic_overlay],
          ['ArchiTexture', row.architexture_overlay]
        ].forEach(function (item) {
          const tile = makeTile(item[0], item[1]);
          if (tile) els.diagGrid.appendChild(tile);
        });

        els.downloads.innerHTML = '';
        [
          ['Synthetic image', row.raw_image],
          ['GT mask', row.raw_gt],
          ['Proposal union mask', row.raw_proposal_union],
          ['Handcrafted mask', row.raw_handcrafted],
          ['PTD heuristic mask', row.raw_heuristic],
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
        els.statCovered.textContent = String(summary.covered_count ?? 0);
        els.statMiss.textContent = String(summary.miss_count ?? 0);
        els.statUnion.textContent = fmt(summary.proposal_union_miou_covered, 4);
        els.statArch.textContent = fmt(summary.architexture_miou_covered, 4);
        els.statAri.textContent = fmt(summary.architexture_ari_covered, 4);
        if (summary.sam2_original_label) {
          els.statSam2Label.textContent = summary.sam2_original_label + ' mIoU / ARI';
          els.statSam2.textContent = fmt(summary.sam2_original_miou_all, 4) + ' / ' + fmt(summary.sam2_original_ari_all, 4);
        } else {
          els.statSam2.textContent = 'n/a';
        }
      }

      function bindControls() {
        els.searchInput.addEventListener('input', render);
        els.subsetSelect.addEventListener('change', render);
        els.focusSelect.addEventListener('change', render);
        els.sortSelect.addEventListener('change', render);
        els.flagBeatsHandcrafted.addEventListener('change', render);
        els.flagBeatsHeuristic.addEventListener('change', render);
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
          els.flagBeatsHandcrafted.checked = false;
          els.flagBeatsHeuristic.checked = false;
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
            fetch('./assets/stld_manifest.json', { cache: 'no-store' }),
            fetch('./assets/stld_manifest_summary.json', { cache: 'no-store' })
          ]);

          if (!dataResp.ok) throw new Error('Could not load STLD manifest');
          if (!sumResp.ok) throw new Error('Could not load STLD summary');

          const data = await dataResp.json();
          const summary = await sumResp.json();

          if (!Array.isArray(data) || !data.length) throw new Error('Empty STLD manifest');

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

    summary_json = json.loads(args.summary_json.read_text(encoding="utf-8"))
    sam2_summary = json.loads(args.sam2_original_summary.read_text(encoding="utf-8")) if args.sam2_original_summary else None
    entries = build_manifest(
        out_dir=out_dir,
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        proposal_union_masks=args.proposal_union_masks,
        handcrafted_masks=args.handcrafted_masks,
        heuristic_masks=args.heuristic_masks,
        architexture_masks=args.architexture_masks,
        per_image_csv=args.per_image_csv,
        metadata_csv=args.metadata_csv,
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
