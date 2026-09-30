"""Structured logging setup shared by every script/module.

Kept deliberately dependency-free (stdlib `logging` only) so it works
identically whether or not torch/diffusers are installed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False

_SENSITIVE_KEY_FRAGMENTS = ("token", "api_key", "apikey", "secret", "password")


class _RedactSecretsFilter(logging.Filter):
    """Best-effort filter that prevents obviously secret-looking values
    (anything that looks like `HF_TOKEN=...` or `api_key: ...`) from being
    written to log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        lowered = msg.lower()
        if any(frag in lowered for frag in _SENSITIVE_KEY_FRAGMENTS) and ("=" in msg or ":" in msg):
            record.msg = "[log message redacted -- appeared to contain a secret/token]"
            record.args = ()
        return True


def setup_logging(
    name: str = "ravi_varma",
    log_dir: Optional[str] = "outputs/logs",
    level: int = logging.INFO,
    filename: str = "ravi_varma.log",
) -> logging.Logger:
    """Configure and return a logger. Safe to call multiple times."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already configured (e.g. called again from a sub-module).
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(_RedactSecretsFilter())
    logger.addHandler(stream_handler)

    if log_dir:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(Path(log_dir) / filename)
            file_handler.setFormatter(fmt)
            file_handler.addFilter(_RedactSecretsFilter())
            logger.addHandler(file_handler)
        except OSError:
            # Non-fatal: fall back to console-only logging (e.g. read-only fs).
            logger.warning("Could not create log directory %s; logging to console only.", log_dir)

    logger.propagate = False
    return logger


def get_logger(name: str = "ravi_varma") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
