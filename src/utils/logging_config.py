"""Logging configuration helpers for the NEM trading system."""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> Logger:
    """Configure and return a logger instance.

    Parameters
    ----------
    name:
        Name of the logger to configure.
    level:
        Logging level, INFO by default.
    log_file:
        Optional path to a log file. When provided the handler will write
        logs to the given file in addition to stdout.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger
