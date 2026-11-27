"""Logging setup.

Two rules the CLI contract depends on:

* **Logs go to stderr, data goes to stdout.** ``admetus predict -o -`` must be
  pipeable without log lines corrupting the CSV.
* **``--log-format json``** emits one JSON object per line, because automated
  callers capture logs and should not have to parse prose.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "admetus"

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything passed via `extra=` that is not a standard LogRecord field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}


def configure(level: str = "info", fmt: str = "text", quiet: bool = False) -> logging.Logger:
    """Install a single stderr handler on the package logger and return it.

    Args:
        level: One of ``debug``, ``info``, ``warning``, ``error``.
        fmt: ``text`` or ``json``.
        quiet: Raise the threshold to ``warning``, whatever ``level`` says.

    Returns:
        The configured ``admetus`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)

    resolved = logging.WARNING if quiet else _LEVELS.get(level, logging.INFO)
    logger.setLevel(resolved)
    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return the package logger, or a child of it."""
    return logging.getLogger(LOGGER_NAME if suffix is None else f"{LOGGER_NAME}.{suffix}")


def progress_disabled() -> bool:
    """Say whether progress bars should be suppressed.

    A progress bar written to a captured (non-tty) stderr fills an agent's log
    with carriage returns, so it is off unless a human is watching.
    """
    return not sys.stderr.isatty()


__all__ = ["JsonFormatter", "configure", "get_logger", "progress_disabled"]
