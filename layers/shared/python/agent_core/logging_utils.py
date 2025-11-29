"""Centralized logging helpers for the agent core."""

from __future__ import annotations

import logging
import os
from typing import Iterable


def _coerce_level(env_level: str | None, default: int) -> int:
    """Best-effort conversion of LOG_LEVEL strings to logging constants."""

    if not env_level:
        return default

    level = logging.getLevelName(env_level.upper())
    if isinstance(level, int):
        return level
    return default


def _tune_noise_sources(level: int, noisy_loggers: Iterable[str]) -> None:
    """Dial down very chatty dependencies unless explicit DEBUG was requested."""

    target = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(target)


def setup_logging(verbosity: int = 0) -> int:
    """Configure root logging based on CLI flags or the LOG_LEVEL env var."""

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    level = _coerce_level(os.getenv("LOG_LEVEL"), level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _tune_noise_sources(level, noisy_loggers=("boto3", "botocore", "urllib3"))
    return level
