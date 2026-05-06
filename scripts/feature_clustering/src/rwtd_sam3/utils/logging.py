"""Repository-wide logging configuration helper.

This module centralizes the console logging format used by every CLI entrypoint
so dataset inspection, prediction, and evaluation commands all emit logs in the
same shape.

Primary entrypoint:
- ``configure_logging()``: configure the root logging level and quiet known
  noisy third-party loggers.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure repository-wide logging with a single, predictable format."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
