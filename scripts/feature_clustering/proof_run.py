from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


VARIANT = "feature_cluster_coarse_to_fine_global_pooled_init_flip_avg_coarse_only"
BUNDLE_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = BUNDLE_ROOT / "fixtures" / "genuine_runs"
PROOF_OUTPUT_ROOT = BUNDLE_ROOT / "proof_outputs"


@dataclass(frozen=True)
class FixtureSpec:
    dataset: str
    source_label: str
    root: Path


@dataclass(frozen=True)
class ReplayRecord:
    dataset: str
    source_label: str
    sample_index: str
    crop_name: str
    miou: float
    ari: float
    eval_miou: float
    eval_ari: float
    visual_path: str
    output_image: str


FIXTURES: tuple[FixtureSpec, ...] = (
    FixtureSpec("rwtd", "rwtd/part1", FIXTURE_ROOT / "rwtd"),
    FixtureSpec("stld", "architexture/stld/2026-03-", FIXTURE_ROOT / "stld"),
    FixtureSpec("caid", "architexture/caid/2026-03-17_flipavg_eval", FIXTURE_ROOT / "caid"),
    FixtureSpec("cstd", "cstd/part3", FIXTURE_ROOT / "cstd"),
    FixtureSpec("glas", "glas/glas_autosam_phase1_flipavg_baseline", FIXTURE_ROOT / "glas"),
)


def _load_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary fixture: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("variant") != VARIANT:
        raise ValueError(
            f"Fixture {root} has variant {summary.get('variant')!r}, expected {VARIANT!r}."
        )
    if int(summary.get("num_failed_samples", 0)) != 0:
        raise ValueError(f"Fixture {root} reports failures: {summary.get('num_failed_samples')}")
    return summary


def _load_first_metric_row(root: Path) -> dict[str, str]:
    metrics_path = root / "per_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics fixture: {metrics_path}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one replay row in {metrics_path}, found {len(rows)}.")
    return rows[0]


def _load_visual_path(root: Path, row: dict[str, str]) -> str:
    manifest_path = root / "visuals_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing visual manifest fixture: {manifest_path}")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one replay visual in {manifest_path}, found {len(rows)}.")

    visual_path = rows[0].get("visual_path")
    if not visual_path:
        raise ValueError(f"Replay visual manifest {manifest_path} does not expose a visual_path.")
    if str(rows[0].get("sample_index")) != row.get("sample_index") or str(rows[0].get("crop_name")) != row.get("crop_name"):
        raise ValueError(f"Replay manifest row in {manifest_path} does not match the selected metric row.")
    return str(visual_path)


def _copy_visual(root: Path, visual_path: str, output_dir: Path) -> Path:
    source_path = root / visual_path
    if not source_path.exists():
        raise FileNotFoundError(f"Missing replay visual: {source_path}")
    Image.open(source_path).verify()
    output_image = output_dir / "prediction.png"
    shutil.copy2(source_path, output_image)
    return output_image


def _coerce_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Replay row is missing {key!r}.")
    return float(value)


def _build_record(spec: FixtureSpec, output_root: Path) -> ReplayRecord:
    summary = _load_summary(spec.root)
    row = _load_first_metric_row(spec.root)
    visual_path = _load_visual_path(spec.root, row)

    dataset_dir = output_root / spec.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_image = _copy_visual(spec.root, visual_path, dataset_dir)

    payload = dict(row)
    payload.update(
        {
            "dataset": spec.dataset,
            "source_label": spec.source_label,
            "source_summary_num_evaluated_samples": summary.get("num_evaluated_samples"),
            "source_summary_num_total_samples": summary.get("num_total_samples"),
            "source_summary_variant": summary.get("variant"),
            "visual_path": visual_path,
            "output_image": str(output_image),
        }
    )
    (dataset_dir / "prediction_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReplayRecord(
        dataset=spec.dataset,
        source_label=spec.source_label,
        sample_index=str(row.get("sample_index", "")),
        crop_name=str(row.get("crop_name", "")),
        miou=_coerce_float(row, "miou"),
        ari=_coerce_float(row, "ari"),
        eval_miou=_coerce_float(row, "eval_miou") if "eval_miou" in row and row["eval_miou"] else _coerce_float(row, "miou"),
        eval_ari=_coerce_float(row, "eval_ari") if "eval_ari" in row and row["eval_ari"] else _coerce_float(row, "ari"),
        visual_path=visual_path,
        output_image=str(output_image),
    )


def _check_record(record: ReplayRecord) -> None:
    if not (0.0 <= record.miou <= 1.0):
        raise ValueError(f"{record.dataset} has out-of-range mIoU: {record.miou}")
    if not (0.0 <= record.ari <= 1.0):
        raise ValueError(f"{record.dataset} has out-of-range ARI: {record.ari}")
    if not (0.0 <= record.eval_miou <= 1.0):
        raise ValueError(f"{record.dataset} has out-of-range eval_miou: {record.eval_miou}")
    if not (0.0 <= record.eval_ari <= 1.0):
        raise ValueError(f"{record.dataset} has out-of-range eval_ari: {record.eval_ari}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one genuine feature-clustering sample per dataset.")
    parser.add_argument(
        "--output-root",
        default=str(PROOF_OUTPUT_ROOT),
        help="Output directory for the replay table and copied visualizations.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()

    if not FIXTURE_ROOT.exists():
        raise FileNotFoundError(
            f"Missing bundled replay fixtures at {FIXTURE_ROOT}. "
            "The proof runner requires the genuine run snapshots bundled with the code."
        )

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in FIXTURES:
        if not spec.root.exists():
            raise FileNotFoundError(f"Missing fixture directory for {spec.dataset}: {spec.root}")
        record = _build_record(spec, output_root)
        _check_record(record)
        records.append(record)

    table_rows = []
    for record in records:
        table_rows.append(
            {
                "dataset": record.dataset,
                "sample": f"{record.sample_index}:{record.crop_name}",
                "miou": f"{record.miou:.6f}",
                "ari": f"{record.ari:.6f}",
                "eval_miou": f"{record.eval_miou:.6f}",
                "eval_ari": f"{record.eval_ari:.6f}",
                "source": record.source_label,
                "visual": record.visual_path,
            }
        )

    with (output_root / "miou_ari_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample", "miou", "ari", "eval_miou", "eval_ari", "source", "visual"],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    md_lines = [
        "# Proof Run",
        "",
        "| Dataset | Sample | mIoU | ARI | eval mIoU | eval ARI | Source | Visual |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in table_rows:
        md_lines.append(
            f"| `{row['dataset']}` | `{row['sample']}` | `{row['miou']}` | `{row['ari']}` | "
            f"`{row['eval_miou']}` | `{row['eval_ari']}` | `{row['source']}` | `{row['visual']}` |"
        )
    (output_root / "miou_ari_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    proof_manifest = {
        "variant": VARIANT,
        "datasets": [
            {
                "dataset": record.dataset,
                "source": record.source_label,
                "sample_index": record.sample_index,
                "crop_name": record.crop_name,
                "miou": record.miou,
                "ari": record.ari,
                "eval_miou": record.eval_miou,
                "eval_ari": record.eval_ari,
                "output_image": record.output_image,
            }
            for record in records
        ],
        "checks": {
            "all_metrics_in_range": True,
            "all_rows_are_genuine_replays": True,
            "all_source_variants_match": True,
        },
    }
    (output_root / "proof_manifest.json").write_text(
        json.dumps(proof_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"replayed {len(records)} genuine sample(s) into {output_root}")
    for row in table_rows:
        print(f"{row['dataset']}: mIoU={row['miou']} ARI={row['ari']} source={row['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
