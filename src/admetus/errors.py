"""Exception types and the process exit codes they map to.

The exit codes are part of the CLI contract documented in the README: automated
callers branch on them, so they must stay stable.

================  ==========================================================
Exit code         Meaning
================  ==========================================================
``0``             Success. Rows whose prediction is NaN do not change this.
``1``             Unexpected internal error. Never raised deliberately.
``2``             Usage or input error (``click.UsageError`` also exits 2).
``3``             Model bundles missing or failing verification.
``4``             Only under ``--strict``, or when every row failed.
``5``             Network or download failure, so a caller can tell whether
                  retrying is worthwhile.
``130``           Interrupted (SIGINT).
================  ==========================================================
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_NO_MODEL = 3
EXIT_PARTIAL_FAILURE = 4
EXIT_NETWORK = 5
EXIT_INTERRUPTED = 130


class AdmetusError(Exception):
    """Base class for every error this package raises deliberately."""

    exit_code: int = EXIT_INTERNAL


class UsageError(AdmetusError):
    """The request itself is malformed: bad flag, bad column, unreadable input."""

    exit_code = EXIT_USAGE


class InputDataError(UsageError):
    """The input file is not what it claims to be (ragged rows, missing column)."""


class ModelError(AdmetusError):
    """A model bundle is missing, incomplete, or fails verification."""

    exit_code = EXIT_NO_MODEL


class SpecMismatchError(ModelError):
    """A bundle was built with a different featurizer than the one available here.

    Refusing is the only safe answer: the column space would silently disagree.
    """


class NetworkError(AdmetusError):
    """A download failed. Distinct from :class:`ModelError` so retries are informed."""

    exit_code = EXIT_NETWORK


class DependencyError(AdmetusError):
    """A native dependency is installed but unusable, with remediation attached."""

    exit_code = EXIT_INTERNAL

    def __init__(self, message: str, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation

    def __str__(self) -> str:
        base = super().__str__()
        if self.remediation:
            return f"{base}\n\nTo fix this, run:\n    {self.remediation}"
        return base


__all__ = [
    "EXIT_INTERNAL",
    "EXIT_INTERRUPTED",
    "EXIT_NETWORK",
    "EXIT_NO_MODEL",
    "EXIT_OK",
    "EXIT_PARTIAL_FAILURE",
    "EXIT_USAGE",
    "AdmetusError",
    "DependencyError",
    "InputDataError",
    "ModelError",
    "NetworkError",
    "SpecMismatchError",
    "UsageError",
]
