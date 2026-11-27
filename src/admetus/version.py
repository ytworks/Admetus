"""Single source of truth for the package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

#: Schema version of the machine-readable JSON documents the CLI emits
#: (``endpoints --json``, ``--report``, ``doctor --json``). Bump it whenever a
#: field is removed or changes meaning, so automated callers can refuse politely.
SCHEMA_VERSION = 1


def get_version() -> str:
    """Return the installed package version, or a marker when running from a source tree."""
    try:
        return _version("admetus")
    except PackageNotFoundError:  # pragma: no cover - only hit without an install
        return "0.0.0+unknown"


__all__ = ["SCHEMA_VERSION", "get_version"]
