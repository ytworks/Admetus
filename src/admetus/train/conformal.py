"""Prediction intervals with a coverage guarantee.

The spread across fold models is a weak uncertainty estimate on its own: the
models share hyperparameters and a feature basis, 75% of each one's training rows
appear in every other's, and with five folds the standard error of the spread
itself is about 0.35 of the spread. It carries no observation-noise term and no
distribution-shift term.

Normalized split conformal prediction rescues it. The out-of-fold residuals that
already exist are divided by the ensemble spread, and a single quantile of that
ratio turns the spread from a pretend standard deviation into a calibrated width.
It costs no extra training and adds two numbers to the bundle.

Coverage holds for exchangeable points. Scaffold-shifted compounds are not
exchangeable with the training set, but the folds are scaffold-grouped, so the
residuals used for calibration already reflect that shift. The honest move is to
measure achieved coverage on the held-out set and record it, which is what the
trainer does -- a nominal 90% interval that covers 70% is reported as covering 70%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalInterval:
    """A fitted interval width.

    Attributes:
        quantile: The conformal multiplier.
        beta: Floor added to the spread, so a zero-spread prediction still gets width.
        level: Nominal coverage.
        is_log: Whether the interval lives in log space, making it multiplicative
            once mapped back to the published scale.
    """

    quantile: float
    beta: float
    level: float = 0.9
    is_log: bool = False

    def bounds(self, prediction: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` in the space the model was fitted in."""
        half = self.quantile * (np.asarray(spread, dtype=np.float64) + self.beta)
        prediction = np.asarray(prediction, dtype=np.float64)
        return prediction - half, prediction + half

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "beta": self.beta,
            "level": self.level,
            "is_log": self.is_log,
        }


def from_dict(payload: dict[str, Any] | None) -> ConformalInterval | None:
    """Rebuild from a bundle record."""
    if not payload:
        return None
    return ConformalInterval(
        quantile=float(payload["quantile"]),
        beta=float(payload["beta"]),
        level=float(payload.get("level", 0.9)),
        is_log=bool(payload.get("is_log", False)),
    )


def fit(
    y_true: np.ndarray,
    prediction: np.ndarray,
    spread: np.ndarray,
    level: float = 0.9,
    is_log: bool = False,
) -> ConformalInterval | None:
    """Fit the interval width on out-of-fold predictions.

    Args:
        y_true: Labels in the fitted space.
        prediction: Out-of-fold predictions in the same space.
        spread: Ensemble spread per row.
        level: Nominal coverage.
        is_log: Whether the fitted space is logarithmic.

    Returns:
        The fitted interval, or ``None`` when there are too few rows for the
        finite-sample quantile to mean anything.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(prediction) & np.isfinite(spread)
    y_true, prediction, spread = y_true[mask], prediction[mask], spread[mask]
    n = y_true.size
    if n < 20:
        return None

    positive = spread[spread > 0]
    beta = float(np.median(positive)) if positive.size else 1.0
    if beta <= 0:
        beta = 1.0

    residual = np.abs(y_true - prediction) / (spread + beta)
    # The finite-sample conformal rank, which is slightly above the plain quantile.
    rank = min(n, int(np.ceil((n + 1) * level)))
    quantile = float(np.sort(residual)[rank - 1])
    return ConformalInterval(quantile=quantile, beta=beta, level=level, is_log=is_log)


def coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Share of labels that fall inside their interval."""
    y_true = np.asarray(y_true, dtype=np.float64)
    inside = (y_true >= np.asarray(low)) & (y_true <= np.asarray(high))
    return float(np.mean(inside)) if y_true.size else float("nan")


__all__ = ["ConformalInterval", "coverage", "fit", "from_dict"]
