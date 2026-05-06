"""Shared artifact writer for repo-contract evaluation runs.

This is the first migration step toward a single evaluation entry point.
Existing routes still compute their own per-sample metrics and route-specific
summary fields, but artifact assembly and contract enforcement flow through this
module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from rwtd_sam3.eval.evaluation_contract import (
    MetricPair,
    build_protocol_record,
    get_route_evaluation_contract,
)
from rwtd_sam3.eval.runner import write_csv, write_json, write_jsonl, write_text


def _resolve_route_primary_summary(summary: dict[str, Any]) -> dict[str, Any]:
    route_primary = summary.get("route_primary")
    if isinstance(route_primary, dict):
        return route_primary
    primary_metric_name = summary.get("primary_metric_name")
    secondary_metric_name = summary.get("secondary_metric_name")
    mean_metrics = dict(summary.get("mean_metrics", {}))
    return {
        "supported": True,
        "reason": None,
        "metric_view": summary.get("primary_metric_view", summary.get("foreground_evaluation_view")),
        "primary_metric_name": primary_metric_name,
        "secondary_metric_name": secondary_metric_name,
        "primary_metric_value": (
            float(mean_metrics[primary_metric_name])
            if primary_metric_name in mean_metrics and mean_metrics[primary_metric_name] is not None
            else summary.get("primary_metric_value")
        ),
        "secondary_metric_value": (
            float(mean_metrics[secondary_metric_name])
            if secondary_metric_name in mean_metrics and mean_metrics[secondary_metric_name] is not None
            else summary.get("secondary_metric_value")
        ),
    }


def _build_metric_view_payload(
    *,
    supported: bool,
    reason: str | None,
    metric_names: dict[str, str],
    mean_metrics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "supported": bool(supported),
        "reason": reason,
    }
    for public_name, summary_key in metric_names.items():
        payload[public_name] = (
            float(mean_metrics[summary_key])
            if supported and summary_key in mean_metrics and mean_metrics[summary_key] is not None
            else None
        )
    return payload


def _build_metric_view_section(
    *,
    mean_metrics: dict[str, Any],
    supported_views: Sequence[str],
    primary_metric_pair: MetricPair,
) -> dict[str, Any]:
    def supported(view_name: str) -> bool:
        return str(view_name) in supported_views

    return {
        "route_primary": {
            "supported": True,
            "reason": None,
            "metric_view": primary_metric_pair.metric_view,
            "primary_metric_name": primary_metric_pair.primary_metric_name,
            "secondary_metric_name": primary_metric_pair.secondary_metric_name,
            "primary_metric_value": float(mean_metrics.get(primary_metric_pair.primary_metric_name))
            if mean_metrics.get(primary_metric_pair.primary_metric_name) is not None
            else None,
            "secondary_metric_value": float(mean_metrics.get(primary_metric_pair.secondary_metric_name))
            if mean_metrics.get(primary_metric_pair.secondary_metric_name) is not None
            else None,
        },
        "partition_invariant": _build_metric_view_payload(
            supported=supported("partition_invariant"),
            reason=None if supported("partition_invariant") else "unsupported for this route",
            metric_names={"miou": "eval_miou", "ari": "eval_ari"},
            mean_metrics=mean_metrics,
        ),
        "direct_foreground": _build_metric_view_payload(
            supported=supported("direct_foreground"),
            reason=None if supported("direct_foreground") else "unsupported for this route",
            metric_names={
                "iou": "direct_foreground_iou",
                "dice": "direct_foreground_dice",
                "precision": "direct_foreground_precision",
                "recall": "direct_foreground_recall",
            },
            mean_metrics=mean_metrics,
        ),
        "upstream_faithful": _build_metric_view_payload(
            supported=supported("upstream_faithful"),
            reason=None if supported("upstream_faithful") else "unsupported for this route",
            metric_names={"iou": "upstream_eval_iou", "dice": "upstream_eval_dice"},
            mean_metrics=mean_metrics,
        ),
    }


def _build_benchmark_manifest_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        manifest_rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "split": row.get("split"),
                "sample_index": row.get("sample_index"),
                "crop_name": row.get("crop_name"),
                "grade_label": row.get("grade_label"),
            }
        )
    return manifest_rows


def _build_paper_row(summary: dict[str, Any]) -> dict[str, Any]:
    route_primary = _resolve_route_primary_summary(summary)
    return {
        "dataset_id": summary.get("dataset_id"),
        "route_name": summary.get("route_name"),
        "split": summary.get("split"),
        "protocol_family": summary.get("protocol_family"),
        "checkpoint_role": summary.get("checkpoint_role"),
        "primary_metric_name": route_primary["primary_metric_name"],
        "primary_metric_value": route_primary["primary_metric_value"],
        "secondary_metric_name": route_primary["secondary_metric_name"],
        "secondary_metric_value": route_primary["secondary_metric_value"],
        "n_total": summary.get("n_total"),
        "n_evaluated": summary.get("n_evaluated"),
        "n_failed": summary.get("n_failed"),
        "coverage_rate": summary.get("coverage_rate"),
        "safety_tag": summary.get("safety_tag"),
    }


def build_quality_summary_record(summary: dict[str, Any]) -> dict[str, Any]:
    route_primary = _resolve_route_primary_summary(summary)
    required_artifacts = [
        "summary.json",
        "summary.md",
        "fairness.md",
        "paper_row.json",
        "benchmark_manifest.csv",
        "protocol.json",
        "per_sample_metrics.csv",
        "per_sample_metrics.jsonl",
    ]
    if summary.get("n_failed", 0):
        required_artifacts.append("eval/failures.csv")
    status = "complete" if summary.get("coverage_rate") == 1.0 and summary.get("n_failed", 0) == 0 else "degraded"
    return {
        "status": status,
        "route_name": summary.get("route_name"),
        "protocol_family": summary.get("protocol_family"),
        "selection_split": summary.get("selection_split"),
        "headline_split": summary.get("headline_split"),
        "selection_metric_name": summary.get("selection_metric_name"),
        "selection_metric_value": summary.get("selection_metric_value"),
        "best_epoch": summary.get("best_epoch"),
        "final_epoch": summary.get("final_epoch"),
        "coverage_rate": summary.get("coverage_rate"),
        "required_artifacts": required_artifacts,
        "route_primary": route_primary,
    }


def render_quality_summary_markdown(summary: dict[str, Any]) -> str:
    """Render the stable quality-summary block for route-local or shared runs."""

    quality_summary = build_quality_summary_record(summary)
    route_primary = quality_summary["route_primary"]
    lines = [
        "## Summary Quality",
        "",
        f"- Status: `{quality_summary['status']}`",
        f"- Route: `{quality_summary['route_name']}`",
        f"- Protocol family: `{quality_summary['protocol_family']}`",
        f"- Selection split: `{quality_summary['selection_split']}`",
        f"- Headline split: `{quality_summary['headline_split']}`",
        f"- Selection metric: `{quality_summary['selection_metric_name']}` = `{quality_summary['selection_metric_value']}`",
        f"- Best epoch: `{quality_summary['best_epoch']}`",
        f"- Final epoch: `{quality_summary['final_epoch']}`",
        f"- Coverage: `{quality_summary['coverage_rate']}`",
        f"- Route-primary metric pair: `{route_primary['primary_metric_name']}` / `{route_primary['secondary_metric_name']}`",
        f"- Required artifacts: {', '.join(f'`{artifact}`' for artifact in quality_summary['required_artifacts'])}",
        "",
    ]
    return "\n".join(lines)


def _build_fairness_markdown(summary: dict[str, Any], protocol_record: dict[str, Any]) -> str:
    route_primary = _resolve_route_primary_summary(summary)
    lines = [
        "# Fairness Notes",
        "",
        f"- Route: `{summary.get('route_name')}`",
        f"- Protocol family: `{summary.get('protocol_family')}`",
        f"- Selection split: `{summary.get('selection_split')}`",
        f"- Headline split: `{summary.get('headline_split')}`",
        f"- Safety tag: `{summary.get('safety_tag')}`",
        f"- Primary metric pair: `{route_primary['primary_metric_name']}` / `{route_primary['secondary_metric_name']}`",
        f"- Coverage: `{summary.get('n_evaluated')}` / `{summary.get('n_total')}` (`{summary.get('coverage_rate'):.6f}`)",
        f"- Validation policy: `{summary.get('validation_split_policy')}`",
        f"- Validation manifest: `{summary.get('validation_subset_manifest_output_path') or summary.get('validation_subset_manifest_path')}`",
        "",
        "## Deviations",
        "",
    ]
    deviations = list(protocol_record.get("deviations", ()))
    if deviations:
        lines.extend(f"- `{deviation}`" for deviation in deviations)
    else:
        lines.append("- `none`")
    lines.extend(
        [
            "",
            "## Contract Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in protocol_record.get("fairness_notes", ()))
    lines.append("")
    return "\n".join(lines)


def _append_contract_markdown(base_markdown: str, summary: dict[str, Any]) -> str:
    route_primary = _resolve_route_primary_summary(summary)
    quality_summary = build_quality_summary_record(summary)
    lines = [
        base_markdown.rstrip(),
        "",
        "## Evaluation Contract",
        "",
        f"- Repo evaluation contract: `{summary.get('repo_evaluation_contract_version')}`",
        f"- Protocol family: `{summary.get('protocol_family')}`",
        f"- Safety tag: `{summary.get('safety_tag')}`",
        f"- Coverage: `{summary.get('n_evaluated')}` / `{summary.get('n_total')}` (`{summary.get('coverage_rate'):.6f}`)",
        f"- Route-primary metric: `{route_primary['primary_metric_name']}` = `{route_primary['primary_metric_value']}`",
        f"- Route-primary secondary metric: `{route_primary['secondary_metric_name']}` = `{route_primary['secondary_metric_value']}`",
    ]
    lines.append("")
    lines.append(render_quality_summary_markdown(summary).rstrip())
    lines.append("")
    return "\n".join(lines)


def evaluate_run(
    *,
    output_dir: Path | None,
    route_name: str,
    split_role: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    protocol_family: str,
    primary_metric_pair: MetricPair,
    selection_split: str,
    headline_split: str,
    safety_tag: str,
    allow_train_selection: bool = False,
    summary_markdown_builder: Callable[[dict[str, Any]], str],
    summary_csv_flattener: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    failures: Sequence[dict[str, Any]] | None = None,
    deviations: Sequence[str] | None = None,
    visual_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one route summary and write repo-contract evaluation artifacts."""

    if not rows:
        raise ValueError("evaluate_run requires at least one evaluated sample row.")

    route_contract = get_route_evaluation_contract(route_name)
    resolved_deviations = list(deviations or ())
    if allow_train_selection:
        resolved_deviations.append(
            "selection_split=train was intentionally allowed because this run uses the full official train split without an explicit held-out validation subset."
        )
    protocol_record = build_protocol_record(
        route_name=route_name,
        protocol_family=protocol_family,
        primary_metric_pair=primary_metric_pair,
        selection_split=selection_split,
        headline_split=headline_split,
        safety_tag=safety_tag,
        allow_train_selection=allow_train_selection,
        deviations=resolved_deviations,
    )

    failures_rows = [dict(row) for row in (failures or ())]
    n_total = len(rows) + len(failures_rows)
    n_evaluated = len(rows)
    n_failed = len(failures_rows)
    coverage_rate = float(n_evaluated / n_total) if n_total else 0.0
    mean_metrics = dict(summary.get("mean_metrics", {}))
    selection_metric_name = str(summary.get("selection_metric"))
    selection_metric_value = (
        float(mean_metrics[selection_metric_name])
        if selection_metric_name in mean_metrics and mean_metrics[selection_metric_name] is not None
        else None
    )

    summary = dict(summary)
    summary.update(
        {
            "repo_evaluation_contract_version": protocol_record["repo_evaluation_contract_version"],
            "route_name": route_name,
            "protocol_family": protocol_family,
            "selection_split": selection_split,
            "headline_split": headline_split,
            "split_role": split_role,
            "safety_tag": safety_tag,
            "n_total": n_total,
            "n_evaluated": n_evaluated,
            "n_failed": n_failed,
            "coverage_rate": coverage_rate,
            "selection_metric_name": selection_metric_name,
            "selection_metric_value": selection_metric_value,
            "best_epoch": summary.get("best_epoch_by_selection_metric"),
            "final_epoch": summary.get("num_epochs"),
        }
    )
    summary["quality_summary"] = build_quality_summary_record(summary)
    summary.update(
        _build_metric_view_section(
            mean_metrics=mean_metrics,
            supported_views=route_contract.supported_metric_views,
            primary_metric_pair=primary_metric_pair,
        )
    )

    benchmark_manifest_rows = _build_benchmark_manifest_rows(rows)
    paper_row = _build_paper_row(summary)
    fairness_markdown = _build_fairness_markdown(summary, protocol_record)

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        eval_dir = resolved_output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_csv(resolved_output_dir / "benchmark_manifest.csv", benchmark_manifest_rows)
        write_json(resolved_output_dir / "protocol.json", protocol_record)
        write_csv(resolved_output_dir / "per_sample_metrics.csv", rows)
        write_jsonl(resolved_output_dir / "per_sample_metrics.jsonl", rows)
        write_csv(eval_dir / "per_sample_metrics.csv", rows)
        write_jsonl(eval_dir / "per_sample_metrics.jsonl", rows)
        if visual_records is not None:
            write_jsonl(resolved_output_dir / "visuals_manifest.jsonl", list(visual_records))
        if failures_rows:
            write_csv(eval_dir / "failures.csv", failures_rows)
        else:
            write_text(eval_dir / "failures.csv", "")
        write_json(eval_dir / f"{split_role}_metrics.json", summary)
        write_json(resolved_output_dir / "summary.json", summary)
        if summary_csv_flattener is not None:
            write_csv(resolved_output_dir / "summary.csv", [summary_csv_flattener(summary)])
        write_text(
            resolved_output_dir / "summary.md",
            _append_contract_markdown(summary_markdown_builder(summary), summary),
        )
        write_text(resolved_output_dir / "fairness.md", fairness_markdown)
        write_json(resolved_output_dir / "paper_row.json", paper_row)
    return summary


def rewrite_contract_summary(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    summary_markdown_builder: Callable[[dict[str, Any]], str],
    summary_csv_flattener: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Rewrite summary-side artifacts after a route adds extra summary metadata."""

    resolved_output_dir = Path(output_dir)
    eval_dir = resolved_output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(resolved_output_dir / "summary.json", summary)
    write_json(eval_dir / f"{summary['split_role']}_metrics.json", summary)
    if summary_csv_flattener is not None:
        write_csv(resolved_output_dir / "summary.csv", [summary_csv_flattener(summary)])
    write_text(
        resolved_output_dir / "summary.md",
        _append_contract_markdown(summary_markdown_builder(summary), summary),
    )
    write_text(
        resolved_output_dir / "fairness.md",
        _build_fairness_markdown(summary, {"deviations": summary.get("profile_deviations", {}), "fairness_notes": ()}),
    )
    write_json(resolved_output_dir / "paper_row.json", _build_paper_row(summary))
