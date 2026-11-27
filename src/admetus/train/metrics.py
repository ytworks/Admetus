"""Metrics, computed the way the published benchmark computes them.

Two details decide whether a number is comparable with anything published:

**AUPRC is average precision**, ``sklearn.metrics.average_precision_score``, not
the trapezoidal area under an interpolated precision-recall curve. On an
imbalanced set the two differ materially, and the benchmark uses the former.

**Metrics are computed on the published scale.** Several endpoints are trained in
log space; computing mean absolute error there would produce a number that looks
like a world record and means nothing. Predictions are inverse-transformed before
scoring, always.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Metrics where a larger value is better.
HIGHER_IS_BETTER = {"auroc": True, "auprc": True, "spearman": True, "mae": False, "rmse": False}


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the receiver operating characteristic curve."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision.

    Deliberately not ``auc(recall, precision)``: the trapezoidal version
    interpolates between operating points that do not exist and gives a different
    answer on imbalanced data.
    """
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(diff**2)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation. Returns ``nan`` when either side is constant."""
    from scipy.stats import spearmanr

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of a probability. Lower is better."""
    return float(np.mean((np.asarray(y_prob, float) - np.asarray(y_true, float)) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins.

    Reported next to the ranking metrics so the claim that probabilities are
    calibrated is checkable rather than asserted.
    """
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    if y_prob.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = which == b
        if not np.any(mask):
            continue
        total += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def matthews(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    """Matthews correlation coefficient for hard labels."""
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred_label))


#: Metric name to implementation.
METRICS = {
    "auroc": auroc,
    "auprc": auprc,
    "mae": mae,
    "rmse": rmse,
    "spearman": spearman,
    "brier": brier,
}


def compute(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute one metric by name.

    Raises:
        ValueError: The metric name is unknown.
    """
    try:
        fn = METRICS[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}") from None
    return fn(np.asarray(y_true), np.asarray(y_pred))


def evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute every metric that applies to a task.

    Reporting the whole set, not just the primary metric, is what makes a
    comparison with another model's chosen metric possible.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if task == "classification":
        out = {
            "auroc": auroc(y_true, y_pred),
            "auprc": auprc(y_true, y_pred),
            "brier": brier(y_true, y_pred),
            "ece": expected_calibration_error(y_true, y_pred),
        }
        threshold, mcc = best_threshold(y_true, y_pred)
        out["mcc_at_best_threshold"] = mcc
        out["best_threshold"] = threshold
        return out
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes the Matthews correlation.

    Stored in the bundle and reported by ``admetus endpoints --json`` so a caller
    has an operating point to use instead of reaching for 0.5, which is rarely
    the right cut on an imbalanced endpoint.

    Returns:
        ``(threshold, mcc)``.
    """
    from sklearn.metrics import matthews_corrcoef

    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    candidates = np.unique(np.round(y_prob, 4))
    if candidates.size > 200:
        candidates = np.quantile(y_prob, np.linspace(0.01, 0.99, 200))
    best = (0.5, -1.0)
    for threshold in candidates:
        labels = (y_prob >= threshold).astype(int)
        if labels.min() == labels.max():
            continue
        score = float(matthews_corrcoef(y_true, labels))
        if score > best[1]:
            best = (float(threshold), score)
    return best


@dataclass(frozen=True)
class Interval:
    """A metric with a bootstrap confidence interval.

    A single number from a 56-compound test set invites over-reading, so every
    reported metric carries its uncertainty.
    """

    value: float
    low: float
    high: float
    n: int
    level: float = 0.95

    def to_dict(self) -> dict[str, float | int]:
        return {
            "value": self.value,
            "ci_low": self.low,
            "ci_high": self.high,
            "ci_level": self.level,
            "n": self.n,
        }

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.low:.4f}, {self.high:.4f}]"


def bootstrap(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap a confidence interval for one metric.

    Resamples that end up with a single class are skipped, since a ranking metric
    is undefined on them.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = y_true.size
    point = compute(metric, y_true, y_pred)
    if n < 2 or n_resamples <= 0:
        return Interval(point, float("nan"), float("nan"), n, level)

    rng = np.random.default_rng(seed)
    values: list[float] = []
    classification = metric in ("auroc", "auprc", "brier")
    for _ in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        if classification and np.unique(y_true[pick]).size < 2:
            continue
        try:
            values.append(compute(metric, y_true[pick], y_pred[pick]))
        except ValueError:  # pragma: no cover - degenerate resample
            continue
    if not values:  # pragma: no cover - only for pathological inputs
        return Interval(point, float("nan"), float("nan"), n, level)

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return Interval(point, float(low), float(high), n, level)


__all__ = [
    "HIGHER_IS_BETTER",
    "METRICS",
    "Interval",
    "auprc",
    "auroc",
    "best_threshold",
    "bootstrap",
    "brier",
    "compute",
    "evaluate",
    "expected_calibration_error",
    "mae",
    "matthews",
    "rmse",
    "spearman",
]
