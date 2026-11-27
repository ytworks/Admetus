"""Shared CLI plumbing.

Conventions the command line keeps, because the primary caller is a program:

* logs go to standard error, data to standard output, so output can be piped;
* ``--json`` prints one machine-readable object and nothing else;
* exit codes are stable and documented in :mod:`admetus.errors`;
* every path or resource flag has an environment variable, because a sandboxed
  caller sets environment rather than rewriting a command line;
* nothing ever prompts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from admetus.errors import EXIT_INTERNAL, EXIT_INTERRUPTED, AdmetusError
from admetus.logging import configure
from admetus.version import SCHEMA_VERSION, get_version

ENV_MODELS_DIR = "ADMETUS_MODELS_DIR"
ENV_DATA_DIR = "ADMETUS_DATA_DIR"
ENV_CACHE_DIR = "ADMETUS_CACHE_DIR"
ENV_JOBS = "ADMETUS_JOBS"


def default_models_dir() -> Path:
    """Where bundles live unless told otherwise."""
    override = os.environ.get(ENV_MODELS_DIR)
    if override:
        return Path(override)
    return Path(cache_root()) / "models"


def default_data_dir() -> Path:
    """Where fetched datasets live unless told otherwise."""
    return Path(os.environ.get(ENV_DATA_DIR, "data/raw"))


def cache_root() -> Path:
    """The cache directory, honouring the usual environment variables."""
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "admetus"


def default_jobs() -> int | None:
    """Worker count from the environment, if set."""
    raw = os.environ.get(ENV_JOBS)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def emit(payload: dict[str, Any]) -> None:
    """Print one JSON object to standard output."""
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("admetus_version", get_version())
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_report(path: Path | None, payload: dict[str, Any]) -> None:
    """Write a report file, even when the run failed."""
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("admetus_version", get_version())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def setup_logging(ctx: click.Context) -> None:
    """Configure logging from the root options."""
    root = ctx.find_root().params
    configure(
        level=root.get("log_level", "info"),
        fmt=root.get("log_format", "text"),
        quiet=root.get("quiet", False),
    )


def fail(exc: AdmetusError, as_json: bool = False) -> None:
    """Report an error the way the caller asked for, then exit with its code."""
    payload = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    remediation = getattr(exc, "remediation", None)
    if remediation:
        payload["remediation"] = remediation
    if as_json:
        emit(payload)
    else:
        click.echo(f"error: {exc}", err=True)
    raise SystemExit(exc.exit_code)


def run_guarded(fn, as_json: bool = False):
    """Run a command body, turning our exceptions into documented exit codes."""
    try:
        return fn()
    except AdmetusError as exc:
        fail(exc, as_json=as_json)
    except KeyboardInterrupt:
        click.echo("interrupted", err=True)
        raise SystemExit(EXIT_INTERRUPTED) from None
    except Exception as exc:  # noqa: BLE001 - the documented internal-error path
        if as_json:
            emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        else:
            click.echo(f"internal error: {type(exc).__name__}: {exc}", err=True)
        if os.environ.get("ADMETUS_TRACEBACK"):
            raise
        raise SystemExit(EXIT_INTERNAL) from None


endpoint_option = click.option(
    "--endpoint", "endpoints", multiple=True, metavar="KEY",
    help="Endpoint to act on. Repeatable. Run 'admetus endpoints' to list them.",
)
all_option = click.option("--all", "select_all", is_flag=True, help="Act on every endpoint.")
json_option = click.option("--json", "as_json", is_flag=True, help="Print one JSON object and nothing else.")
models_dir_option = click.option(
    "--models-dir", type=click.Path(path_type=Path), default=None,
    help=f"Where model bundles live. Defaults to ${ENV_MODELS_DIR} or the cache directory.",
)
data_dir_option = click.option(
    "--data-dir", type=click.Path(path_type=Path), default=None,
    help=f"Where fetched datasets live. Defaults to ${ENV_DATA_DIR} or data/raw.",
)


def stderr_note(message: str) -> None:
    """Write a human note to standard error, keeping stdout clean for data."""
    click.echo(message, err=True, file=sys.stderr)


__all__ = [
    "ENV_CACHE_DIR",
    "ENV_DATA_DIR",
    "ENV_JOBS",
    "ENV_MODELS_DIR",
    "all_option",
    "cache_root",
    "data_dir_option",
    "default_data_dir",
    "default_jobs",
    "default_models_dir",
    "emit",
    "endpoint_option",
    "fail",
    "json_option",
    "models_dir_option",
    "run_guarded",
    "setup_logging",
    "stderr_note",
    "write_report",
]
