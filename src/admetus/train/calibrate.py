"""Making a classification score an actual probability.

Two things can make the number written to the output CSV not a probability:

**Class weighting shifts the prior.** Weighting positives by ``w`` multiplies the
odds by ``w``, so the model's output is a score under a shifted prior rather than
a probability. That is undone exactly, in one line, with no fitting.

**The model was never asked to be calibrated.** Both reported metrics are
rank-based, so nothing in training pushed the probabilities towards honesty.

Undoing the prior shift is strictly monotone, so it leaves ranking untouched.
Calibration itself is only *weakly* monotone: isotonic regression is a step
function, and the flat regions it creates turn distinct scores into ties.
Measured on 1,500 rows at a 20% positive rate, that moved AUROC from 0.8423 to
0.8492 and average precision from 0.6922 to 0.6348 while halving the Brier score.
So the convenient claim that calibration is free in ranking terms is false for
isotonic, and candidates are chosen by measurement instead: both are fitted on the
out-of-fold predictions, and the one with the best Brier score is kept unless it
costs more than a small tolerance of the endpoint's own metric.

Fitting uses out-of-fold predictions that already exist, so it costs no extra
training. Reported metrics are computed on the calibrated output that actually
ships, so the published number describes the model a caller gets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from admetus.logging import get_logger

log = get_logger("train.calibrate")

#: Below this many rows, isotonic regression overfits the calibration curve, so
#: the two-parameter logistic fit is used instead.
ISOTONIC_MIN_ROWS = 1_000


def undo_class_weight(p: np.ndarray, scale_pos_weight: float) -> np.ndarray:
    """Remove the prior shift introduced by ``scale_pos_weight``.

    Weighting positives by ``w`` multiplies the odds by ``w``, so dividing them
    back recovers the unweighted probability exactly.
    """
    if scale_pos_weight == 1.0:
        return np.asarray(p, dtype=np.float64)
    weight = float(scale_pos_weight)
    values = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return values / (values + weight * (1.0 - values))


@dataclass(frozen=True)
class Calibrator:
    """A monotone map from score to probability.

    Attributes:
        kind: ``identity``, ``platt`` or ``isotonic``.
        scale_pos_weight: The weight whose prior shift is removed first.
        platt: ``(a, b)`` for the logistic fit.
        isotonic_x: Score breakpoints for the step function.
        isotonic_y: Calibrated value at each breakpoint.
    """

    kind: str = "identity"
    scale_pos_weight: float = 1.0
    platt: tuple[float, float] | None = None
    isotonic_x: tuple[float, ...] | None = None
    isotonic_y: tuple[float, ...] | None = None

    def apply(self, p: np.ndarray) -> np.ndarray:
        """Map scores to calibrated probabilities."""
        values = undo_class_weight(p, self.scale_pos_weight)
        if self.kind == "identity":
            return values
        if self.kind == "platt":
            a, b = self.platt
            return 1.0 / (1.0 + np.exp(-(a * values + b)))
        if self.kind == "isotonic":
            return np.interp(values, np.asarray(self.isotonic_x), np.asarray(self.isotonic_y))
        raise ValueError(f"unknown calibrator {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "scale_pos_weight": self.scale_pos_weight}
        if self.platt is not None:
            payload["platt"] = list(self.platt)
        if self.isotonic_x is not None:
            payload["isotonic_x"] = list(self.isotonic_x)
            payload["isotonic_y"] = list(self.isotonic_y)
        return payload


IDENTITY = Calibrator()


def from_dict(payload: dict[str, Any] | None) -> Calibrator:
    """Rebuild a calibrator from a bundle record."""
    if not payload:
        return IDENTITY
    kind = payload.get("kind", "identity")
    return Calibrator(
        kind=kind,
        scale_pos_weight=float(payload.get("scale_pos_weight", 1.0)),
        platt=tuple(payload["platt"]) if payload.get("platt") else None,
        isotonic_x=tuple(payload["isotonic_x"]) if payload.get("isotonic_x") else None,
        isotonic_y=tuple(payload["isotonic_y"]) if payload.get("isotonic_y") else None,
    )


def fit(
    y_true: np.ndarray,
    p_oof: np.ndarray,
    scale_pos_weight: float = 1.0,
    metric: str = "auroc",
    tolerance: float = 0.005,
    min_isotonic_rows: int = ISOTONIC_MIN_ROWS,
) -> Calibrator:
    """Fit a calibrator on out-of-fold predictions, choosing the kind by measurement.

    Candidates are the identity, a two-parameter logistic fit, and isotonic
    regression where there are enough rows for it. The one with the best Brier
    score wins, unless it costs more than ``tolerance`` of the endpoint's own
    ranking metric -- isotonic can, because its flat regions create ties.

    Args:
        y_true: Labels for the rows a fold held out.
        p_oof: Out-of-fold scores for those rows.
        scale_pos_weight: The weight used during training, undone before fitting.
        metric: The endpoint's ranking metric, protected by ``tolerance``.
        tolerance: How much of that metric may be given up for better calibration.
        min_isotonic_rows: Minimum rows before isotonic is considered.

    Returns:
        The chosen :class:`Calibrator`. Identity when nothing beats it.
    """
    from admetus.train import metrics as metrics_mod

    y_true = np.asarray(y_true, dtype=np.float64)
    scores = undo_class_weight(p_oof, scale_pos_weight)
    mask = np.isfinite(scores) & np.isfinite(y_true)
    y_true, scores = y_true[mask], scores[mask]

    identity = Calibrator(scale_pos_weight=scale_pos_weight)
    if y_true.size < 20 or np.unique(y_true).size < 2:
        return identity

    candidates = [identity, _fit_platt(scores, y_true, scale_pos_weight)]
    if y_true.size >= min_isotonic_rows:
        candidates.append(_fit_isotonic(scores, y_true, scale_pos_weight))

    baseline_rank = metrics_mod.compute(metric, y_true, scores)
    best = identity
    best_brier = metrics_mod.brier(y_true, scores)

    for candidate in candidates[1:]:
        if candidate is None:
            continue
        calibrated = candidate.apply(p_oof[mask])
        rank = metrics_mod.compute(metric, y_true, calibrated)
        if rank < baseline_rank - tolerance:
            log.debug(
                "rejected %s calibration: %s fell from %.4f to %.4f",
                candidate.kind, metric, baseline_rank, rank,
            )
            continue
        score = metrics_mod.brier(y_true, calibrated)
        if score < best_brier:
            best, best_brier = candidate, score

    if best is not identity:
        log.debug("chose %s calibration (brier %.4f)", best.kind, best_brier)
    return best


def _fit_platt(scores: np.ndarray, y_true: np.ndarray, scale_pos_weight: float) -> Calibrator | None:
    """Fit a two-parameter logistic map. Strictly monotone, so ranking is preserved."""
    from sklearn.linear_model import LogisticRegression

    try:
        model = LogisticRegression(solver="lbfgs", C=1e6)
        model.fit(scores.reshape(-1, 1), y_true.astype(int))
    except Exception:  # pragma: no cover - degenerate input
        return None
    return Calibrator(
        kind="platt",
        scale_pos_weight=scale_pos_weight,
        platt=(float(model.coef_[0][0]), float(model.intercept_[0])),
    )


def _fit_isotonic(scores: np.ndarray, y_true: np.ndarray, scale_pos_weight: float) -> Calibrator | None:
    """Fit isotonic regression, stored as breakpoints so the bundle stays plain JSON."""
    from sklearn.isotonic import IsotonicRegression

    try:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores, y_true)
    except Exception:  # pragma: no cover - degenerate input
        return None
    grid = np.unique(np.clip(np.quantile(scores, np.linspace(0, 1, 256)), 0.0, 1.0))
    return Calibrator(
        kind="isotonic",
        scale_pos_weight=scale_pos_weight,
        isotonic_x=tuple(float(v) for v in grid),
        isotonic_y=tuple(float(v) for v in model.predict(grid)),
    )


__all__ = ["IDENTITY", "ISOTONIC_MIN_ROWS", "Calibrator", "fit", "from_dict", "undo_class_weight"]
