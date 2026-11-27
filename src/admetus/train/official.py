"""The official benchmark protocol: the published split, five seeds, mean and spread.

The leaderboard's number is not one run. It is five runs that differ only in seed,
reported as a mean and a standard deviation over the published test set, and a single
run is therefore not comparable with it -- a run that happens to land well reads as an
improvement that five runs would not support.

What the seed moves here is the fold assignment inside the training portion, and
through it the hyperparameters and the ensemble. What it cannot move is the test set:
that is fixed by the archive, which is the point of the protocol.

Tuning inside each seed's run is the honest arrangement and the default. The
alternative -- reusing hyperparameters this project tuned on its own split of the full
dataset -- would select them with the official test compounds inside the search, which
is exactly the comparison the protocol exists to prevent. Running untuned is the
cheaper honest option and is reported as such: a floor, not a best effort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from admetus.data.benchmark import OFFICIAL_SEEDS, OfficialSplit
from admetus.data.registry import Endpoint
from admetus.errors import AdmetusError
from admetus.logging import get_logger
from admetus.train import tuning
from admetus.train.trainer import train_endpoint

log = get_logger("train.official")


@dataclass(frozen=True)
class SeedRun:
    """One seed's run against the published test set."""

    seed: int
    score: float
    n_test: int
    n_train: int
    seconds: float
    n_trials: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "score": self.score,
            "n_test": self.n_test,
            "n_train": self.n_train,
            "seconds": self.seconds,
            "n_trials": self.n_trials,
        }


@dataclass(frozen=True)
class OfficialResult:
    """The protocol's result for one endpoint.

    Attributes:
        key: Endpoint key.
        metric: The endpoint's official metric.
        runs: One entry per seed that completed.
        failures: Seeds that did not complete, with the reason.
        tuned: Whether each run tuned its own hyperparameters.
    """

    key: str
    metric: str
    runs: list[SeedRun]
    failures: list[dict[str, Any]] = field(default_factory=list)
    tuned: bool = True

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([r.score for r in self.runs], dtype=np.float64)

    @property
    def mean(self) -> float:
        return float(self.scores.mean()) if self.runs else float("nan")

    @property
    def sd(self) -> float:
        """Standard deviation over seeds, the spread the leaderboard reports.

        Uses the sample form, matching how a five-run spread is normally quoted,
        and is ``0.0`` rather than undefined for a single run.
        """
        values = self.scores
        if values.size < 2:
            return 0.0
        return float(values.std(ddof=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.key,
            "metric": self.metric,
            "mean": self.mean,
            "sd": self.sd,
            "n_seeds": len(self.runs),
            "seeds": [r.to_dict() for r in self.runs],
            "failures": self.failures,
            "tuned_per_seed": self.tuned,
            "protocol": "published train_val/test split, one run per seed, mean and sd over seeds",
            "comparable_with_leaderboard": True,
        }

    def describe(self) -> str:
        """One line, in the form the leaderboard uses."""
        if not self.runs:
            return f"{self.key}: no seed completed"
        return f"{self.key}: {self.metric} {self.mean:.4f} +/- {self.sd:.4f} over {len(self.runs)} seed(s)"


def run_endpoint(
    endpoint: Endpoint,
    split: OfficialSplit,
    seeds: tuple[int, ...] = OFFICIAL_SEEDS,
    folds: int = 5,
    tune: bool = True,
    time_budget: int | None = tuning.DEFAULT_TIME_BUDGET_SECONDS,
    num_threads: int = 0,
    work_dir: Path | None = None,
) -> OfficialResult:
    """Run the protocol for one endpoint.

    Each seed trains on the published training portion and is scored once on the
    published test set. Nothing is carried between seeds, so a run that fails costs
    only its own seed; the result records which seeds completed, because a mean over
    three seeds is not a mean over five and should not be read as one.

    Args:
        endpoint: Which endpoint.
        split: Its published split.
        seeds: Seeds to run. The protocol's own are 1 to 5.
        folds: Cross-validation folds inside the training portion.
        tune: Tune hyperparameters within each seed's run.
        time_budget: Wall-clock cap for tuning one seed, in seconds.
        num_threads: LightGBM threads.
        work_dir: Where each seed's tuning study is kept, so a long run resumes.
            The models themselves are not saved: the protocol reports a score, and
            the bundle that ships is the one ``admetus train`` writes.

    Returns:
        The :class:`OfficialResult`.
    """
    runs: list[SeedRun] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        started = time.time()
        seed_dir = Path(work_dir) / f"seed{seed}" if work_dir else None
        try:
            model = train_endpoint(
                endpoint,
                split.dataset,
                folds=folds,
                seed=seed,
                tune=tune,
                time_budget=time_budget,
                num_threads=num_threads,
                bootstrap_resamples=0,
                models_dir=seed_dir,
                holdout=split.holdout,
            )
        except AdmetusError as exc:
            log.error("%s seed %d: %s", endpoint.key, seed, exc)
            failures.append({"seed": seed, "error": str(exc)})
            continue
        score = float(model.test_metrics[endpoint.metric])
        trials = (model.cv_record or {}).get("n_trials")
        runs.append(
            SeedRun(
                seed=seed,
                score=score,
                n_test=int(model.provenance["n_test"]),
                n_train=int(model.provenance["n_train"]),
                seconds=round(time.time() - started, 1),
                n_trials=int(trials) if trials else None,
            )
        )
        log.info(
            "%s seed %d: %s = %.4f on %d published test rows",
            endpoint.key, seed, endpoint.metric, score, model.provenance["n_test"],
        )

    result = OfficialResult(
        key=endpoint.key, metric=endpoint.metric, runs=runs, failures=failures, tuned=tune
    )
    log.info("%s", result.describe())
    return result


__all__ = ["OfficialResult", "SeedRun", "run_endpoint"]
