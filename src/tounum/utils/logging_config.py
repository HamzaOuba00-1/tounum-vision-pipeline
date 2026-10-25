"""Logging setup shared by the command-line interface and the library."""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "tounum"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure readable console logs and quieten TensorFlow's C++ logs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger of the package logger (``tounum.<name>``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
