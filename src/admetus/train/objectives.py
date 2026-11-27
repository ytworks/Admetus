"""Turning fold predictions into one number to search against.

Averaging a metric over folds is both biased and noisy when folds are small. One
endpoint here has 190 positives in total, so a fold's validation set holds around
38 and its average precision is close to noise -- and that noise would *be* the
hyperparameter search's objective.

Pooling every out-of-fold prediction into a single vector and scoring it once is
far steadier, and it matches what the published benchmark does on its single test
set. Half-life illustrates the gap: a fold Spearman over 107 compounds has a
standard error near 0.10, wider than the spread of the leaderboard's top ten,
while the pooled value over 533 is near 0.043.

For regression there is a catch. Scaffold-grouped folds differ in their mean
level, and pooling raw predictions lets those offsets inflate rank correlation on
their own. Predictions are therefore centred per fold before pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from admetus.train import metrics as metrics_mod
from admetus.train.cv import FoldFit


@dataclass(frozen=True)
class PooledOutOfFold:
    """Out-of-fold predictions assembled across folds.

    Attributes:
        prediction: One prediction per training row.
        covered: Rows that some fold held out, so the prediction is honest.
        per_fold: Each fold's own metric, kept as a diagnostic only.
    """

    prediction: np.ndarray
    covered: np.ndarray
    per_fold: list[float]

    @property
    def n_covered(self) -> int:
        return int(np.count_nonzero(self.covered))


def pool(fits: list[FoldFit], n_rows: int, task: str, metric: str, y: np.ndarray) -> PooledOutOfFold:
    """Assemble out-of-fold predictions from fitted folds.

    Args:
        fits: One entry per fold.
        n_rows: Size of the training set the folds index into.
        task: ``classification`` or ``regression``.
        metric: The endpoint's metric, for the per-fold diagnostics.
        y: Labels in the fitted space.

    Returns:
        The pooled predictions.
    """
    prediction = np.full(n_rows, np.nan, dtype=np.float64)
    covered = np.zeros(n_rows, dtype=bool)
    per_fold: list[float] = []

    for fit in fits:
        values = fit.valid_pred
        if task == "regression":
            # Remove the fold's own level so between-fold offsets cannot flatter
            # a rank correlation computed over the pooled vector.
            values = values - np.mean(values) + np.mean(y[fit.valid_rows])
        prediction[fit.valid_rows] = values
        covered[fit.valid_rows] = True
        try:
            per_fold.append(metrics_mod.compute(metric, y[fit.valid_rows], fit.valid_pred))
        except (ValueError, IndexError):  # pragma: no cover - degenerate fold
            per_fold.append(float("nan"))

    return PooledOutOfFold(prediction=prediction, covered=covered, per_fold=per_fold)


def score(pooled: PooledOutOfFold, y: np.ndarray, metric: str) -> float:
    """Score the pooled predictions, over the covered rows only."""
    mask = pooled.covered & np.isfinite(pooled.prediction)
    if not np.any(mask):
        return float("-inf")
    try:
        value = metrics_mod.compute(metric, y[mask], pooled.prediction[mask])
    except ValueError:  # pragma: no cover - single-class pool
        return float("-inf")
    if not np.isfinite(value):
        return float("-inf")
    return float(value)


def as_objective(value: float, metric: str) -> float:
    """Turn a metric into something Optuna maximizes.

    Error metrics are negated rather than minimized, so every study in the
    project has the same direction and the logs read consistently.
    """
    if value == float("-inf"):
        return float("-inf")
    return value if metrics_mod.HIGHER_IS_BETTER[metric] else -value


def fold_diagnostics(pooled: PooledOutOfFold) -> dict[str, float]:
    """Summarize the per-fold metrics, as a diagnostic and never as the headline."""
    values = np.asarray([v for v in pooled.per_fold if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"fold_mean": float("nan"), "fold_sd": float("nan"), "n_folds": 0}
    return {
        "fold_mean": float(values.mean()),
        "fold_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "n_folds": int(values.size),
    }


__all__ = ["PooledOutOfFold", "as_objective", "fold_diagnostics", "pool", "score"]
