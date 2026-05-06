"""Shared evaluation-contract entrypoints."""

from .evaluate_run import (
    build_quality_summary_record,
    evaluate_run,
    render_quality_summary_markdown,
    rewrite_contract_summary,
)

__all__ = [
    "build_quality_summary_record",
    "evaluate_run",
    "render_quality_summary_markdown",
    "rewrite_contract_summary",
]
