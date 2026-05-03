#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PAPER_ROOT.parent
TEXTURESAM_ROOT = WORKSPACE_ROOT / "TextureSAM-v2"
if str(TEXTURESAM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXTURESAM_ROOT))

from texturesam_v2.proposals import PromptMaskProposalStore, ProposalLoadConfig


OUT_DIR = PAPER_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PREVIEW_ROOT = (
    WORKSPACE_ROOT
    / "research-site"
    / "site"
    / "experiments"
    / "sam2-vs-sam3-frozen-feature-clustering"
    / "assets"
    / "all_previews"
)
FEATURE_SCALE_TABLE = (
    WORKSPACE_ROOT
    / "research-site"
    / "site"
    / "experiments"
    / "sam3-stage2-coarse-vs-fine-probe"
    / "assets"
    / "tables"
    / "summary_table.csv"
)

RWTD_ORACLES = (
    TEXTURESAM_ROOT
    / "reports"
    / "reviewer_oracles_rwtd_full256"
    / "rwtd_proposal_oracles_per_image.csv"
)
RWTD_SELECTOR = (
    TEXTURESAM_ROOT
    / "reports"
    / "final_round_learned_single_selector_rwtd"
    / "rwtd_learned_single_selector_per_image.csv"
)
RWTD_SELECTOR_MASKS = (
    TEXTURESAM_ROOT
    / "reports"
    / "final_round_learned_single_selector_rwtd"
    / "official_export"
)
RWTD_PROMPTS = (
    TEXTURESAM_ROOT
    / "reports"
    / "strict_ptd_v11_multibank"
    / "overnight_multibank_promptstyle_fixed"
)
RWTD_AUDIT_CASES = (
    TEXTURESAM_ROOT
    / "reports"
    / "release_swinb_full256_audit"
    / "texturesam2_acute_learned_rescue"
    / "diagnostics"
    / "audit_cases"
)

STLD_ORACLES = (
    TEXTURESAM_ROOT / "reports" / "reviewer_oracles_stld" / "stld_proposal_oracles_per_image.csv"
)
STLD_SELECTOR = (
    TEXTURESAM_ROOT
    / "reports"
    / "final_round_learned_single_selector_stld"
    / "stld_learned_single_selector_per_image.csv"
)
STLD_SELECTOR_MASKS = TEXTURESAM_ROOT / "reports" / "final_round_learned_single_selector_stld" / "masks"
STLD_FINAL_MASKS = (
    TEXTURESAM_ROOT
    / "experiments"
    / "khan_synthetic_gallery_20260312"
    / "eval"
    / "strict_ptd_learned"
    / "masks"
)
STLD_BENCHMARK = TEXTURESAM_ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "benchmark"
STLD_PROMPTS = TEXTURESAM_ROOT / "experiments" / "khan_synthetic_gallery_20260312" / "promptstyle"

CONTROLNET_FIG = WORKSPACE_ROOT / "paper_main" / "figures" / "fig_controlnet_bridge_examples.png"
CAID_FIG = WORKSPACE_ROOT / "paper_main" / "figures" / "fig_caid_audit.png"


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = _load_font(28)
SMALL_FONT = _load_font(22)
TINY_FONT = _load_font(18)

GT_COLOR = np.array([44, 170, 44], dtype=np.float32)
SELECTOR_COLOR = np.array([124, 92, 214], dtype=np.float32)
CORE_COLOR = np.array([66, 133, 244], dtype=np.float32)
FINAL_COLOR = np.array([242, 148, 34], dtype=np.float32)
ORACLE_COLOR = np.array([220, 60, 60], dtype=np.float32)


def _load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: np.ndarray, alpha: float = 0.34) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    mask_bool = mask > 0
    if mask_bool.any():
        rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * color

        edge = np.zeros_like(mask_bool, dtype=bool)
        edge[:-1, :] |= mask_bool[:-1, :] != mask_bool[1:, :]
        edge[1:, :] |= mask_bool[1:, :] != mask_bool[:-1, :]
        edge[:, :-1] |= mask_bool[:, :-1] != mask_bool[:, 1:]
        edge[:, 1:] |= mask_bool[:, 1:] != mask_bool[:, :-1]
        rgb[edge] = color
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def _load_rwtd_oracle_mask(image_id: int, index: int, expected_shape: tuple[int, int]) -> np.ndarray:
    store = PromptMaskProposalStore(RWTD_PROMPTS, ProposalLoadConfig(min_area=16))
    proposals = store.load(image_id, expected_shape=expected_shape)
    if index < 0 or index >= len(proposals):
        return np.zeros(expected_shape, dtype=np.uint8)
    return proposals[index]


def _load_stld_oracle_mask(image_id: int, index: int, expected_shape: tuple[int, int]) -> np.ndarray:
    store = PromptMaskProposalStore(STLD_PROMPTS, ProposalLoadConfig(min_area=16))
    proposals = store.load(image_id, expected_shape=expected_shape)
    if index < 0 or index >= len(proposals):
        return np.zeros(expected_shape, dtype=np.uint8)
    return proposals[index]


def _best_oracle_mask(store: PromptMaskProposalStore, image_id: int, gt_mask: np.ndarray) -> np.ndarray:
    proposals = store.load(image_id, expected_shape=gt_mask.shape)
    if not proposals:
        return np.zeros_like(gt_mask)
    gt_bool = gt_mask > 0
    best_iou = -1.0
    best = proposals[0]
    for proposal in proposals:
        prop_bool = proposal > 0
        inter = np.logical_and(prop_bool, gt_bool).sum()
        union = np.logical_or(prop_bool, gt_bool).sum()
        iou = float(inter) / float(union) if union else 0.0
        if iou > best_iou:
            best_iou = iou
            best = proposal
    return best


def _panel(image: Image.Image, title: str, note: str = "", size: tuple[int, int] = (300, 214)) -> Image.Image:
    title_h = 34
    note_h = 28 if note else 0
    canvas = Image.new("RGB", (size[0], size[1] + title_h + note_h), (255, 255, 255))
    framed = ImageOps.contain(image.convert("RGB"), (size[0] - 10, size[1] - 10))
    x = (size[0] - framed.width) // 2
    y = title_h + (size[1] - framed.height) // 2
    canvas.paste(framed, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(15, 15, 15), font=SMALL_FONT)
    if note:
        draw.text((8, size[1] + title_h + 2), note, fill=(70, 70, 70), font=TINY_FONT)
    return canvas


def build_feature_gallery() -> Path:
    rows = [
        (
            "Repeated patterns",
            [
                ("RWTD-101", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "101.webp"),
                ("RWTD-128", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "128.webp"),
            ],
        ),
        (
            "Boundary transitions",
            [
                ("RWTD-183", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "183.webp"),
                ("RWTD-74", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "74.webp"),
            ],
        ),
        (
            "Shape versus texture cues",
            [
                ("RWTD-186", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "186.webp"),
                ("RWTD-308", FEATURE_PREVIEW_ROOT / "rwtd_sam2_coarse" / "308.webp"),
            ],
        ),
        (
            "Synthetic silhouettes",
            [
                ("STLD-12", FEATURE_PREVIEW_ROOT / "stld_sam2_coarse" / "12.webp"),
                ("STLD-90", FEATURE_PREVIEW_ROOT / "stld_sam2_coarse" / "90.webp"),
            ],
        ),
    ]
    panel_w, panel_h = 950, 420
    row_header_h, gutter_x, gutter_y = 42, 24, 26
    width = panel_w * 2 + gutter_x
    height = len(rows) * (row_header_h + panel_h) + (len(rows) - 1) * gutter_y
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for ridx, (row_title, panels) in enumerate(rows):
        y0 = ridx * (row_header_h + panel_h + gutter_y)
        draw.text((4, y0), row_title, fill=(20, 20, 20), font=FONT)
        for cidx, (label, path) in enumerate(panels):
            panel_y = y0 + row_header_h
            x0 = cidx * (panel_w + gutter_x)
            image = Image.open(path).convert("RGB")
            card = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
            framed = ImageOps.contain(image, (panel_w - 8, panel_h - 36))
            card.paste(framed, ((panel_w - framed.width) // 2, 6))
            d = ImageDraw.Draw(card)
            d.text((10, panel_h - 28), label, fill=(50, 50, 50), font=SMALL_FONT)
            canvas.paste(card, (x0, panel_y))

    out = OUT_DIR / "fig_feature_gallery.png"
    canvas.save(out)
    return out


def build_feature_scale_diagnostics() -> Path:
    df = pd.read_csv(FEATURE_SCALE_TABLE)
    df = df.sort_values("mean_ari", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), dpi=240)

    run_labels = [
        "c+1 learned",
        "all uniform",
        "all learned",
        "coarse",
        "mid",
        "fine aligned",
        "fine native",
    ]
    axes[0].bar(run_labels, df["mean_ari"], color=["#d97706", "#64748b", "#2563eb", "#059669", "#0f766e", "#7c3aed", "#a855f7"])
    axes[0].set_ylabel("Mean ARI")
    axes[0].set_ylim(0.0, 0.9)
    axes[0].set_title("A. Mean ARI by archived run")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    weight_rows = df[df["run_id"].isin(["learned_global_gates_all_scales", "learned_global_gates_coarse_plus_next_finer"])].copy()
    weight_rows = weight_rows.set_index("run_id").loc[
        ["learned_global_gates_all_scales", "learned_global_gates_coarse_plus_next_finer"]
    ]
    gate_labels = ["all learned", "coarse+mid learned"]
    w_fpn2 = [0.42962387204170227, 0.5764782428741455]
    w_fpn1 = [0.3147805631160736, 0.4235217273235321]
    w_fpn0 = [0.25559553503990173, 0.0]
    axes[1].bar(gate_labels, w_fpn2, label="fpn_2", color="#d97706")
    axes[1].bar(gate_labels, w_fpn1, bottom=w_fpn2, label="fpn_1", color="#0f766e")
    axes[1].bar(gate_labels, w_fpn0, bottom=np.array(w_fpn2) + np.array(w_fpn1), label="fpn_0", color="#2563eb")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("B. Learned gate weights")
    axes[1].set_ylabel("Weight mass")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    single_df = df[df["run_id"].isin(["fpn_2_only", "fpn_1_only", "fpn_0_only_aligned", "fpn_0_only_native"])].copy()
    single_df["label"] = ["coarse", "mid", "fine aligned", "fine native"]
    x = np.arange(len(single_df))
    width = 0.36
    axes[2].bar(x - width / 2, single_df["mean_ari"], width=width, label="ARI", color="#d97706")
    axes[2].bar(x + width / 2, single_df["mean_miou"], width=width, label="mIoU", color="#94a3b8")
    axes[2].set_xticks(x, single_df["label"], rotation=20)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("C. Single-scale readout")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = OUT_DIR / "fig_feature_scale_diagnostics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _rwtd_row_specs() -> list[tuple[int, str]]:
    return [
        (186, "RWTD-186: weak singleton score, but conservative commitment recovers the correct partition."),
        (74, "RWTD-74: rescue repairs a collapsed core and closes most of the gap to the bank oracle."),
        (308, "RWTD-308: a strong singleton exists in the bank, but the deployed stack abstains from it."),
        (183, "RWTD-183: low-margin wrong-partition case; rescue flips a strong core into the opposite side."),
    ]


def build_rwtd_case_gallery() -> Path:
    oracles = pd.read_csv(RWTD_ORACLES).set_index("image_id")
    selector = pd.read_csv(RWTD_SELECTOR).set_index("image_id")
    cols = ["Input", "GT", "Fragments", "Selector", "Core", "Final", "Oracle single"]
    panel_size = (250, 188)
    row_label_h = 36
    gutter_x = 14
    gutter_y = 22
    width = len(cols) * panel_size[0] + (len(cols) - 1) * gutter_x
    height = len(_rwtd_row_specs()) * (row_label_h + panel_size[1] + 64) + (len(_rwtd_row_specs()) - 1) * gutter_y + 34
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for cidx, col in enumerate(cols):
        x0 = cidx * (panel_size[0] + gutter_x)
        draw.text((x0 + 4, 6), col, fill=(20, 20, 20), font=SMALL_FONT)

    for ridx, (image_id, desc) in enumerate(_rwtd_row_specs()):
        y0 = 34 + ridx * (row_label_h + panel_size[1] + 64 + gutter_y)
        draw.text((0, y0), desc, fill=(35, 35, 35), font=TINY_FONT)
        case_root = RWTD_AUDIT_CASES / str(image_id)
        input_img = Image.open(case_root / "image_rgb.png").convert("RGB")
        gt_mask = _load_mask(case_root / "gt_partition_mask.png")
        fragments = Image.open(case_root / "candidate_stack_overlay.png").convert("RGB")
        selector_mask = _load_mask(RWTD_SELECTOR_MASKS / f"mask_0_{image_id}.png")
        core_mask = _load_mask(case_root / "core_mask.png")
        final_mask = _load_mask(case_root / "final_selected_mask.png")
        oracle_mask = _load_rwtd_oracle_mask(image_id, int(oracles.loc[image_id, "single_index"]), gt_mask.shape)

        panels = [
            _panel(input_img, "", "", panel_size),
            _panel(_overlay_mask(input_img, gt_mask, GT_COLOR), "", "", panel_size),
            _panel(fragments, "", f"{int(oracles.loc[image_id, 'proposal_count'])} frozen proposals", panel_size),
            _panel(
                _overlay_mask(input_img, selector_mask, SELECTOR_COLOR),
                "",
                f"IoU {selector.loc[image_id, 'rwtd_iou']:.3f} / ARI {selector.loc[image_id, 'rwtd_ari']:.3f}",
                panel_size,
            ),
            _panel(
                _overlay_mask(input_img, core_mask, CORE_COLOR),
                "",
                f"IoU {oracles.loc[image_id, 'core_iou']:.3f} / ARI {oracles.loc[image_id, 'core_ari']:.3f}",
                panel_size,
            ),
            _panel(
                _overlay_mask(input_img, final_mask, FINAL_COLOR),
                "",
                f"IoU {oracles.loc[image_id, 'final_iou']:.3f} / ARI {oracles.loc[image_id, 'final_ari']:.3f}",
                panel_size,
            ),
            _panel(
                _overlay_mask(input_img, oracle_mask, ORACLE_COLOR),
                "",
                f"IoU {oracles.loc[image_id, 'single_iou']:.3f} / ARI {oracles.loc[image_id, 'single_ari']:.3f}",
                panel_size,
            ),
        ]
        for cidx, panel in enumerate(panels):
            x0 = cidx * (panel_size[0] + gutter_x)
            canvas.paste(panel, (x0, y0 + row_label_h))

    out = OUT_DIR / "fig_rwtd_case_gallery.png"
    canvas.save(out)
    return out


def _stld_row_specs() -> list[tuple[int, str]]:
    return [
        (26, "STLD-26: a two-proposal bank where the selector, final mask, and oracle all coincide."),
        (90, "STLD-90: the mug silhouette is already captured by a coherent singleton proposal."),
        (173, "STLD-173: low-contrast synthetic case where top-1 nearly saturates the bank upper bound."),
        (127, "STLD-127: small counterexample where the final decision trails the top-1 selector."),
    ]


def build_stld_case_gallery() -> Path:
    oracles = pd.read_csv(STLD_ORACLES).set_index("image_id")
    selector = pd.read_csv(STLD_SELECTOR).set_index("image_id")
    store = PromptMaskProposalStore(STLD_PROMPTS, ProposalLoadConfig(min_area=16))
    cols = ["Input", "GT", "Selector", "Final", "Oracle single"]
    panel_size = (272, 190)
    row_label_h = 34
    gutter_x = 16
    gutter_y = 24
    width = len(cols) * panel_size[0] + (len(cols) - 1) * gutter_x
    height = len(_stld_row_specs()) * (row_label_h + panel_size[1] + 64) + (len(_stld_row_specs()) - 1) * gutter_y + 34
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for cidx, col in enumerate(cols):
        x0 = cidx * (panel_size[0] + gutter_x)
        draw.text((x0 + 4, 6), col, fill=(20, 20, 20), font=SMALL_FONT)

    for ridx, (image_id, desc) in enumerate(_stld_row_specs()):
        y0 = 34 + ridx * (row_label_h + panel_size[1] + 64 + gutter_y)
        draw.text((0, y0), desc, fill=(35, 35, 35), font=TINY_FONT)
        input_img = Image.open(STLD_BENCHMARK / "images" / f"{image_id}.png").convert("RGB")
        gt_mask = _load_mask(STLD_BENCHMARK / "labels" / f"{image_id}.png")
        selector_mask = _load_mask(STLD_SELECTOR_MASKS / f"{image_id}.png")
        final_mask = _load_mask(STLD_FINAL_MASKS / f"{image_id}.png")
        oracle_mask = _best_oracle_mask(store, image_id, gt_mask)
        if oracle_mask.sum() == 0:
            oracle_mask = selector_mask

        panels = [
            _panel(input_img, "", "", panel_size),
            _panel(_overlay_mask(input_img, gt_mask, GT_COLOR), "", f"{int(oracles.loc[image_id, 'proposal_count'])} proposals", panel_size),
            _panel(
                _overlay_mask(input_img, selector_mask, SELECTOR_COLOR),
                "",
                f"IoU {selector.loc[image_id, 'direct_iou']:.3f} / ARI {selector.loc[image_id, 'direct_ari']:.3f}",
                panel_size,
            ),
            _panel(
                _overlay_mask(input_img, final_mask, FINAL_COLOR),
                "",
                f"IoU {oracles.loc[image_id, 'final_iou']:.3f} / ARI {oracles.loc[image_id, 'final_ari']:.3f}",
                panel_size,
            ),
            _panel(
                _overlay_mask(input_img, oracle_mask, ORACLE_COLOR),
                "",
                f"IoU {oracles.loc[image_id, 'single_iou']:.3f} / ARI {oracles.loc[image_id, 'single_ari']:.3f}",
                panel_size,
            ),
        ]
        for cidx, panel in enumerate(panels):
            x0 = cidx * (panel_size[0] + gutter_x)
            canvas.paste(panel, (x0, y0 + row_label_h))

    out = OUT_DIR / "fig_stld_case_gallery.png"
    canvas.save(out)
    return out


def copy_supporting_route_figures() -> tuple[Path, Path]:
    out_control = OUT_DIR / "fig_controlnet_bridge_examples.png"
    out_caid = OUT_DIR / "fig_caid_audit.png"
    if CONTROLNET_FIG.exists():
        shutil.copyfile(CONTROLNET_FIG, out_control)
    if CAID_FIG.exists():
        shutil.copyfile(CAID_FIG, out_caid)
    return out_control, out_caid


def main() -> None:
    outputs = [
        build_feature_gallery(),
        build_feature_scale_diagnostics(),
        build_rwtd_case_gallery(),
        build_stld_case_gallery(),
    ]
    outputs.extend(copy_supporting_route_figures())
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
