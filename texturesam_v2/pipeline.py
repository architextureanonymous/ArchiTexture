from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .consolidator import ConsolidationConfig, TextureSAMV2Consolidator
from .io_utils import ensure_dir, infer_rwtd_dirs, list_rwtd_images, read_image_rgb, read_mask_raw, write_binary_mask
from .metrics import rwtd_invariant_metrics
from .proposals import PromptMaskProposalStore, ProposalLoadConfig, union_proposals


@dataclass(frozen=True)
class EvalConfig:
    rwtd_root: Path
    prompt_masks_root: Path
    out_dir: Path
    baseline_masks_root: Path | None = None
    max_images: int | None = None


class TextureSAMV2Pipeline:
    def __init__(self, consolidation_cfg: ConsolidationConfig):
        self.consolidator = TextureSAMV2Consolidator(consolidation_cfg)

    def run(self, cfg: EvalConfig) -> dict[str, float | int | str]:
        image_dir, label_dir = infer_rwtd_dirs(cfg.rwtd_root)
        images = list_rwtd_images(image_dir)
        if cfg.max_images is not None:
            images = images[: cfg.max_images]

        store = PromptMaskProposalStore(cfg.prompt_masks_root, ProposalLoadConfig(min_area=self.consolidator.cfg.min_area))

        out_dir = ensure_dir(cfg.out_dir)
        out_masks = ensure_dir(out_dir / "masks")

        rows: list[dict[str, float | int | str]] = []

        for image_path in images:
            image_id = image_path.stem
            id_num = int(image_id.split("_")[-1])
            gt_path = label_dir / f"{image_id}.png"

            image = read_image_rgb(image_path)
            gt = read_mask_raw(gt_path)

            proposals = store.load(id_num, expected_shape=image.shape[:2])
            pred, debug = self.consolidator(image, proposals)
            write_binary_mask(out_masks / f"{image_id}.png", pred)

            pred_met = rwtd_invariant_metrics(pred, gt)

            union_mask = union_proposals(proposals, shape=image.shape[:2])
            union_met = rwtd_invariant_metrics(union_mask, gt)

            row: dict[str, float | int | str] = {
                "image_id": image_id,
                "proposal_count": int(len(proposals)),
                "merged_components": int(debug.num_merged_components),
                "v2_iou": float(pred_met.iou),
                "v2_ari": float(pred_met.ari),
                "union_iou": float(union_met.iou),
                "union_ari": float(union_met.ari),
                "v2_score": float(debug.selected_score.score),
                "v2_delta": float(debug.selected_score.delta),
                "v2_var": float(debug.selected_score.variance),
                "v2_frag": float(debug.selected_score.fragmentation),
            }

            if cfg.baseline_masks_root is not None:
                base_candidates = [
                    cfg.baseline_masks_root / f"{image_id}.png",
                    cfg.baseline_masks_root / f"mask_0_{id_num}.png",
                    cfg.baseline_masks_root / f"mask_0_{image_id}.png",
                ]
                base_path = next((p for p in base_candidates if p.exists()), None)
                if base_path is not None:
                    base = read_mask_raw(base_path)
                    base_met = rwtd_invariant_metrics(base, gt)
                    row["baseline_iou"] = float(base_met.iou)
                    row["baseline_ari"] = float(base_met.ari)

            rows.append(row)

        summary = self._summarize(rows, cfg)

        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        self._write_rows(out_dir / "per_image.csv", rows)

        return summary

    def _write_rows(self, path: Path, rows: list[dict[str, float | int | str]]) -> None:
        if not rows:
            return
        keys = sorted(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def _summarize(self, rows: list[dict[str, float | int | str]], cfg: EvalConfig) -> dict[str, float | int | str]:
        def mean_of(key: str) -> float:
            vals = [float(r[key]) for r in rows if key in r]
            return float(np.mean(vals)) if vals else 0.0

        def _jsonify_paths(x):
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, dict):
                return {k: _jsonify_paths(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_jsonify_paths(v) for v in x]
            if isinstance(x, tuple):
                return tuple(_jsonify_paths(v) for v in x)
            return x

        cfg_dict = _jsonify_paths(asdict(self.consolidator.cfg))

        out = {
            "num_images": int(len(rows)),
            "rwtd_root": str(cfg.rwtd_root),
            "prompt_masks_root": str(cfg.prompt_masks_root),
            "baseline_masks_root": str(cfg.baseline_masks_root) if cfg.baseline_masks_root is not None else "",
            "descriptor_mode": self.consolidator.cfg.descriptor_mode,
            "config": cfg_dict,
            "v2_miou": mean_of("v2_iou"),
            "v2_ari": mean_of("v2_ari"),
            "union_miou": mean_of("union_iou"),
            "union_ari": mean_of("union_ari"),
            "mean_proposal_count": mean_of("proposal_count"),
            "mean_merged_components": mean_of("merged_components"),
        }

        if any("baseline_iou" in r for r in rows):
            out["baseline_miou"] = mean_of("baseline_iou")
            out["baseline_ari"] = mean_of("baseline_ari")
            out["delta_miou_vs_baseline"] = float(out["v2_miou"] - out["baseline_miou"])
            out["delta_ari_vs_baseline"] = float(out["v2_ari"] - out["baseline_ari"])

        out["delta_miou_vs_union"] = float(out["v2_miou"] - out["union_miou"])
        out["delta_ari_vs_union"] = float(out["v2_ari"] - out["union_ari"])
        return out
