"""Standard helpers for rich per-run ``experiment_terms.md`` documents.

This module defines the repository standard for run-local experiment terms.
Each results directory should contain an experiment-specific markdown file that
answers four reviewer questions without opening the code:

1. What hypothesis is this run testing?
2. What exact method/pipeline was executed?
3. What data split and training/evaluation separation were used?
4. What outputs, metrics, and failure conditions should be expected?

Builders for individual experiments should supply concrete section content from
their own implementation details rather than reusing broad repository-wide
glossaries. The goal is that future experiments can reuse this renderer while
keeping the content specific to the run that produced the directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ExperimentTermsSection:
    """One rendered section inside a run-local ``experiment_terms.md`` file.

    Args:
        title: Human-readable section title rendered as a second-level heading.
        paragraphs: Dense prose paragraphs rendered before bullets or code.
        bullets: Flat bullet list rendered verbatim in order.
        code_block: Optional fenced code block body rendered with the provided
            ``code_language``.
        code_language: Optional Markdown info-string for ``code_block``.
    """

    title: str
    paragraphs: Sequence[str] = field(default_factory=tuple)
    bullets: Sequence[str] = field(default_factory=tuple)
    code_block: str | None = None
    code_language: str = ""


def render_experiment_terms_markdown(
    *,
    title: str,
    summary_lines: Sequence[str],
    sections: Sequence[ExperimentTermsSection],
    related_paths: Sequence[str] = (),
) -> str:
    """Render a detailed run-local experiment terms markdown document."""

    lines: list[str] = [f"# {title}", ""]
    lines.extend(str(line) for line in summary_lines)
    if summary_lines:
        lines.append("")
    if related_paths:
        lines.extend(["## Source References", ""])
        for relative_path in related_paths:
            lines.append(f"- `{relative_path}`")
        lines.append("")
    for section in sections:
        lines.extend([f"## {section.title}", ""])
        for paragraph in section.paragraphs:
            lines.extend([str(paragraph), ""])
        for bullet in section.bullets:
            lines.append(f"- {bullet}")
        if section.bullets:
            lines.append("")
        if section.code_block is not None:
            lines.append(f"```{section.code_language}".rstrip())
            lines.append(section.code_block.rstrip())
            lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def format_shape_mapping(name_to_shape: dict[str, tuple[int, ...]]) -> list[str]:
    """Return a stable bullet list for named tensor shapes."""

    return [f"`{name}`: `{tuple(int(value) for value in shape)}`" for name, shape in name_to_shape.items()]


def render_relative_path(path: str | Path) -> str:
    """Render a repository-relative path for markdown display."""

    return str(path).replace("\\", "/")
