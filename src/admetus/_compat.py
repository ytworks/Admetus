"""The single place that imports LightGBM.

LightGBM's wheels link against an OpenMP runtime that they do not ship. On macOS
the wheel records ``@rpath/libomp.dylib`` and expects Homebrew's ``libomp``; on
Linux it expects ``libgomp.so.1``. ``pip``/``uv`` install the wheel happily and
the failure only surfaces on first import, as a ``dlopen`` traceback that says
nothing about how to fix it.

Every import of LightGBM goes through :func:`load_lightgbm` so that failure
becomes one actionable sentence instead.
"""

from __future__ import annotations

import platform
import sys
from functools import lru_cache
from types import ModuleType

from admetus.errors import DependencyError

#: Remediation per platform, keyed by ``platform.system()``.
_OPENMP_REMEDIATION = {
    "Darwin": "brew install libomp",
    "Linux": "apt-get install -y libgomp1   # or: dnf install libgomp",
}


def openmp_remediation() -> str:
    """Return the command that installs the OpenMP runtime on this platform."""
    return _OPENMP_REMEDIATION.get(platform.system(), "install an OpenMP runtime for your platform")


def _is_openmp_failure(exc: BaseException) -> bool:
    """Say whether a failed LightGBM import looks like a missing OpenMP runtime."""
    text = str(exc).lower()
    return any(token in text for token in ("libomp", "libgomp", "omp.dll", "openmp"))


@lru_cache(maxsize=1)
def load_lightgbm() -> ModuleType:
    """Import and return the ``lightgbm`` module, or raise with remediation.

    Raises:
        DependencyError: LightGBM is installed but its OpenMP runtime is missing,
            or it is not installed at all. The message names the fix.
    """
    try:
        import lightgbm
    except ImportError as exc:
        raise DependencyError(
            "LightGBM is not installed. It is required for both training and prediction.",
            remediation="uv pip install 'admetus[train]'   # or: pip install lightgbm",
        ) from exc
    except OSError as exc:
        if _is_openmp_failure(exc):
            raise DependencyError(
                "LightGBM is installed but its OpenMP runtime could not be loaded, so the "
                f"native library will not open on this {platform.system()} machine.\n"
                f"Loader said: {exc}",
                remediation=openmp_remediation(),
            ) from exc
        raise DependencyError(
            f"LightGBM is installed but failed to load its native library: {exc}",
            remediation=openmp_remediation(),
        ) from exc
    return lightgbm


def lightgbm_status() -> dict[str, object]:
    """Describe whether LightGBM is usable here, for ``admetus doctor``.

    Never raises: ``doctor`` has to report a broken install rather than die on it.
    """
    try:
        lightgbm = load_lightgbm()
    except DependencyError as exc:
        return {
            "ok": False,
            "version": None,
            "error": str(exc.args[0]),
            "remediation": exc.remediation,
        }
    return {"ok": True, "version": lightgbm.__version__, "error": None, "remediation": None}


def python_status() -> dict[str, object]:
    """Describe the running interpreter, for ``admetus doctor``."""
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


__all__ = ["lightgbm_status", "load_lightgbm", "openmp_remediation", "python_status"]
