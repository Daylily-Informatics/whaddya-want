"""Lightweight logging helpers for agent components."""

from __future__ import annotations

import logging
import os
from typing import Final


DEFAULT_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


def _level_from_verbosity(verbosity: int) -> int:
    if verbosity >= 2:
        return logging.DEBUG
    if verbosity == 1:
        return logging.INFO
    return logging.WARNING


def configure_logging(verbosity: int = 0) -> None:
    """Configure root logging based on ``verbosity``.

    ``verbosity`` typically comes from ``-v`` / ``-vv`` CLI flags. We allow
    the ``LOG_LEVEL`` environment variable to override when running in Lambda
    or other non-CLI contexts.
    """

    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), logging.WARNING)
    else:
        level = _level_from_verbosity(max(verbosity, 0))

    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        force=True,
    )

    # Reduce noise from noisy dependencies but keep our logs chatty at -vv.
    for noisy in ("boto3", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, level))
