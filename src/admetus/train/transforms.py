"""Label transforms for regression endpoints.

Four endpoints are published on a linear scale that spans several decades with a
heavy right tail: volume of distribution (0.01 to 700 L/kg), half-life (0.065 to
1200 hours) and the two intrinsic clearance assays. Fitting squared error on that
scale lets a handful of extreme compounds dominate the loss, and the bulk of the
data ends up poorly ranked.

Training in log space changes the weighting, not the metric. All four are scored
by Spearman correlation, which is rank-based and therefore identical in either
space, so the transform is free.

The clearance assays reach the limit of quantification, so ``log10(0)`` is a real
risk rather than a hypothetical. :class:`Log10Safe` floors at half the smallest
positive value in the training set and records that floor in the bundle, so
inference reproduces training exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from admetus.errors import InputDataError


@dataclass(frozen=True)
class Transform:
    """Identity transform. Base class for the real ones."""

    name: str = "identity"

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Map published labels to the space the model is fitted in."""
        return np.asarray(y, dtype=np.float64)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Map model output back to the published scale."""
        return np.asarray(y, dtype=np.float64)

    @property
    def is_log(self) -> bool:
        """Whether the fitted space is logarithmic.

        Callers use this to decide how to report spread: a standard deviation in
        log units is not a standard deviation in L/kg or hours, so it must not be
        presented as one.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class Log10Safe(Transform):
    """Base-10 logarithm with a floor under the smallest positive value.

    Attributes:
        floor: Values at or below this are raised to it before taking the log.
    """

    name: str = "log10_safe"
    floor: float = 1e-6

    def forward(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64)
        out = np.log10(np.maximum(values, self.floor))
        if not np.all(np.isfinite(out)):
            raise InputDataError(
                f"log10 transform produced non-finite values with floor {self.floor:g}; "
                "the label column is not what the endpoint declares"
            )
        return out

    def inverse(self, y: np.ndarray) -> np.ndarray:
        # Clipped so the exponent cannot overflow on a wild prediction.
        return np.power(10.0, np.clip(np.asarray(y, dtype=np.float64), -30.0, 30.0))

    @property
    def is_log(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "floor": self.floor}


IDENTITY = Transform()


def fit(name: str | None, y_train: np.ndarray) -> Transform:
    """Build a transform from its name and the training labels.

    The floor is derived from the training set only, so it is part of the fitted
    model rather than something recomputed at inference.

    Args:
        name: ``None``, ``log10`` or ``log10_safe``.
        y_train: Training labels on the published scale.

    Returns:
        A fitted :class:`Transform`.

    Raises:
        InputDataError: The name is unknown, or the labels cannot be log-transformed.
    """
    if name in (None, "identity"):
        return IDENTITY
    if name not in ("log10", "log10_safe"):
        raise InputDataError(f"unknown label transform {name!r}")

    values = np.asarray(y_train, dtype=np.float64)
    positive = values[values > 0]
    if positive.size == 0:
        raise InputDataError(
            f"transform {name!r} needs positive labels but none of the {values.size} "
            "training labels are above zero"
        )
    if name == "log10" and np.any(values <= 0):
        n_bad = int(np.count_nonzero(values <= 0))
        raise InputDataError(
            f"transform 'log10' cannot handle {n_bad} label(s) at or below zero. "
            "Use 'log10_safe', which floors them at half the smallest positive value."
        )
    return Log10Safe(floor=float(positive.min()) / 2.0)


def from_dict(payload: dict[str, Any] | None) -> Transform:
    """Rebuild a transform from a bundle record."""
    if not payload or payload.get("name") in (None, "identity"):
        return IDENTITY
    name = payload["name"]
    if name in ("log10", "log10_safe"):
        return Log10Safe(floor=float(payload["floor"]))
    raise InputDataError(f"unknown label transform {name!r} in bundle")


__all__ = ["IDENTITY", "Log10Safe", "Transform", "fit", "from_dict"]
