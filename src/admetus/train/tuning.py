"""Hyperparameter search.

The search space is narrowed deliberately. With 50 trials and TPE's 10 startup
draws, only 40 are informed, and eight-plus dimensions is too thin to explore. The
learning rate is fixed, bagging is fixed, and the budget goes to the parameters
that actually regularize a very wide, very sparse binary feature space.

Several of the usual priors are wrong in that space:

* ``feature_fraction`` samples columns uniformly per tree. With tens of thousands
  of near-constant columns and only a few hundred informative ones, 0.3 is not
  regularization, it is a lottery -- so the range runs much lower.
* ``min_sum_hessian_in_leaf`` defaults to 1e-3, which for binary log-loss is
  inert: the hessian is at most 0.25 per row, so a single-row leaf is allowed.
* ``num_leaves`` defaults to 31, which cannot even be grown on a few hundred rows.

Class weighting is off by default and only offered where the positive rate
actually warrants it. Both reported classification metrics are rank-based, so
weighting buys little, and it shifts the probabilities that get written to the
output CSV. Where it is used, the shift is undone analytically afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from admetus.logging import get_logger
from admetus.train import cv

log = get_logger("train.tuning")

#: Positive rate above which class weighting is not offered at all.
BALANCED_POS_RATE = 0.20

#: Positive rate below which the full negative/positive ratio is offered.
SEVERE_POS_RATE = 0.05


@dataclass
class SearchResult:
    """Outcome of a search.

    Attributes:
        params: Best parameters found, ready to merge into the base set.
        value: Best objective value, in the direction Optuna maximized.
        score: The same value as the endpoint's own metric.
        n_trials: Trials completed.
        seed: Sampler seed, recorded so the search is reproducible.
        trials: Per-trial record for the bundle.
    """

    params: dict[str, Any]
    value: float
    score: float
    n_trials: int
    seed: int
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.params,
            "best_objective": self.value,
            "best_cv_score": self.score,
            "n_trials": self.n_trials,
            "sampler_seed": self.seed,
            # The search picks hyperparameters on the same fold predictions it
            # scores, so this number is optimistic. It ranks candidates; it is not
            # a performance estimate. Reported numbers come from the held-out set.
            "cv_score_biased": True,
        }


def class_weight_range(pos_rate: float, n_pos: int, n_neg: int) -> tuple[float, float] | None:
    """Decide whether to search ``scale_pos_weight``, and over what range.

    Returns ``None`` when the endpoint is balanced enough not to need it, which is
    most of them.
    """
    if n_pos == 0 or pos_rate >= BALANCED_POS_RATE:
        return None
    ratio = n_neg / n_pos
    if pos_rate >= SEVERE_POS_RATE:
        # The full ratio is an over-aggressive folk default; its square root is
        # enough to shift the gradient without flattening the probabilities.
        return (1.0, float(np.sqrt(ratio)))
    return (1.0, float(ratio))


def suggest(trial, n_train: int, tune_rounds: bool, weight_range: tuple[float, float] | None,
            tune_objective: bool) -> dict[str, Any]:
    """Draw one parameter set.

    Args:
        trial: The Optuna trial.
        n_train: Training rows, which bounds the leaf-size range.
        tune_rounds: Search the boosting round count instead of early-stopping it.
        weight_range: Range for ``scale_pos_weight``, or ``None`` to leave it at 1.
        tune_objective: Offer alternative regression objectives. Worth it where the
            metric is rank-based, since squared error is then not the aligned loss.
    """
    max_leaves = 16 if n_train < 500 else 64
    min_leaf_floor = max(5, round(0.03 * n_train))

    params: dict[str, Any] = {
        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 0.5, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, max_leaves),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", min_leaf_floor, max(min_leaf_floor + 1, min_leaf_floor * 4)
        ),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1.0, 20.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
    }
    if tune_rounds:
        params["num_boost_round"] = trial.suggest_int("num_boost_round", 100, 1500, log=True)
    if weight_range is not None:
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", *weight_range, log=True)
    if tune_objective:
        params["objective"] = trial.suggest_categorical(
            "objective", ["regression", "regression_l1", "huber"]
        )
    return params


def study_path(models_dir: Path, key: str) -> Path:
    """Where an endpoint's study database lives, so a search can be resumed."""
    return Path(models_dir) / key / "optuna.sqlite"


def search(
    evaluate: Callable[[dict[str, Any]], float],
    n_trials: int,
    n_train: int,
    tune_rounds: bool,
    weight_range: tuple[float, float] | None = None,
    tune_objective: bool = False,
    seed: int = 0,
    storage: Path | None = None,
    study_name: str = "admetus",
) -> SearchResult:
    """Run the search.

    Args:
        evaluate: Takes a parameter set, returns the objective to maximize.
        n_trials: Trials to run, counted from where a resumed study left off.
        n_train: Training rows.
        tune_rounds: Whether to search the round count.
        weight_range: Range for ``scale_pos_weight``, or ``None``.
        tune_objective: Whether to offer alternative regression objectives.
        seed: Sampler seed. Recorded, because an unseeded sampler would make the
            CLI's determinism promise false and change results on resume.
        storage: SQLite path for a resumable study.
        study_name: Study name inside the storage.

    Returns:
        The :class:`SearchResult`.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    storage_url = None
    if storage is not None:
        Path(storage).parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{Path(storage)}"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )

    already = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - already)
    if already:
        log.info("resuming study with %d completed trial(s); running %d more", already, remaining)

    def objective(trial) -> float:
        params = suggest(trial, n_train, tune_rounds, weight_range, tune_objective)
        return evaluate(params)

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("no trial completed; the objective failed on every parameter set")

    best = study.best_trial
    params = dict(best.params)
    return SearchResult(
        params=params,
        value=float(best.value),
        score=float(best.value if _maximizes(params) else best.value),
        n_trials=len(completed),
        seed=seed,
        trials=[{"number": t.number, "value": t.value, "params": t.params} for t in completed],
    )


def _maximizes(_params: dict[str, Any]) -> bool:
    """Objective values are always in maximize direction (see :func:`objectives.as_objective`)."""
    return True


def default_params(task: str, metric: str, n_train: int, num_threads: int = 0, seed: int = 0) -> dict[str, Any]:
    """A reasonable parameter set for when no search is run."""
    params = cv.base_params(task, metric, num_threads=num_threads, seed=seed)
    params.update(
        {
            "feature_fraction": 0.2,
            "num_leaves": 16 if n_train < 500 else 31,
            "min_data_in_leaf": max(5, round(0.03 * n_train)),
            "min_sum_hessian_in_leaf": 5.0,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
        }
    )
    return params


def merge(base: dict[str, Any], tuned: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    """Merge searched parameters into the base set.

    Returns:
        ``(params, num_boost_round)``, with the round count split out because
        LightGBM takes it as an argument rather than a parameter.
    """
    params = dict(base)
    rounds = None
    for key, value in tuned.items():
        if key == "num_boost_round":
            rounds = int(value)
        else:
            params[key] = value
    return params, rounds


__all__ = [
    "BALANCED_POS_RATE",
    "SEVERE_POS_RATE",
    "SearchResult",
    "class_weight_range",
    "default_params",
    "merge",
    "search",
    "study_path",
    "suggest",
]
