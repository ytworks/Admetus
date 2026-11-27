"""Automatic hyperparameter tuning.

Tuning is delegated to Optuna's stepwise LightGBM tuner, in its
cross-validation form. The CV variant matters: it accepts our own folds, so the
scaffold-grouped splits go straight in and the tuner optimizes a
cross-validation score rather than tuning each fold independently against its
own validation rows.

Imported as ``optuna_integration.lightgbm``. The older
``optuna.integration.lightgbm`` alias still resolves but emits a
``FutureWarning`` (deprecated in 4.9, slated for removal in 6.0), and this
project turns matching deprecation warnings into errors.

**The search space belongs to the tuner, not to us.** Measured, it runs about
65-68 trials across five stages: ``num_leaves`` (20), ``lambda_l1`` and
``lambda_l2`` (20), ``feature_fraction`` (13), ``bagging_fraction`` with
``bagging_freq`` (10), and ``min_child_samples`` (5).

Two consequences are real and measured, not inferred:

1. ``feature_fraction`` is drawn from a fixed ``linspace(0.4, 1.0, 7)``. On an
   89,852-column binary matrix at a density around 2e-3 that floor is far too
   high -- a probe picked 1.0 -- so the prevalence-based column pruning in
   :mod:`admetus.train.prune` is load-bearing. Whatever width reaches the tuner
   has to be a width where 0.4 is a sensible floor, and the post-pruning width is
   recorded per endpoint.
2. The tuner sets ``feature_pre_filter`` to ``False`` itself, because it varies
   ``min_child_samples``. LightGBM's own rare-column guard is therefore not
   available here, which leaves our explicit pruning as the only thing doing that
   job.

What we still set by rule, outside the tuner: the learning rate, the metric
aligned to the endpoint's official one, the regression objective, the class
weight, and the determinism flags. The tuner cannot search any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from admetus._compat import load_lightgbm
from admetus.logging import get_logger, progress_disabled
from admetus.train import cv

log = get_logger("train.tuning")

#: Metrics where the tuner should maximize.
MAXIMIZE_METRICS = frozenset({"auc", "average_precision", "ndcg", "map"})

#: Upper bound on boosting rounds inside the tuner. Early stopping decides the
#: actual count; the production ensemble is refitted at a fixed count afterwards.
TUNER_MAX_ROUNDS = 1_000

#: Rounds without improvement before the tuner stops a trial.
TUNER_EARLY_STOPPING = 50

#: Default wall-clock cap per endpoint. The tuner's trial count is effectively
#: fixed at 65-68, so a budget is the only way to bound 29 endpoints: a probe at
#: 600 rows and full width spent over a minute on one run, and real endpoints
#: reach 13,445 rows.
DEFAULT_TIME_BUDGET_SECONDS = 900


@dataclass
class TuneResult:
    """Outcome of an automatic tuning run.

    Attributes:
        params: The tuner's best parameters, merged over the fixed ones.
        score: The tuner's best cross-validation score.
        n_trials: Trials completed in the study.
        seed: Sampler seed, recorded so the run is reproducible.
        time_budget: Wall-clock cap applied, if any.
        boosters: Per-fold boosters from the best trial, used only to pick the
            production round count. Not shipped: the ensemble is refitted. Empty when
            the tuner could not recover them, which the caller must notice rather
            than quietly accept a default.
        tuned_keys: Which parameters the tuner chose, as opposed to our rules.
    """

    params: dict[str, Any]
    score: float
    n_trials: int
    seed: int
    time_budget: int | None
    boosters: list[Any] = field(default_factory=list)
    tuned_keys: list[str] = field(default_factory=list)
    #: The tuner raised partway through and this fold fell back to the best trial it
    #: had completed. Recorded in the bundle, because a fold tuned less than the
    #: others is something a reader should be told rather than left to infer.
    tuner_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """The record stored in the bundle."""
        return {
            "tuner": "optuna_integration.lightgbm.LightGBMTuner, one run per fold",
            "best_params": {k: v for k, v in self.params.items() if k != "num_threads"},
            "tuned_by_tuner": self.tuned_keys,
            "best_cv_score": self.score,
            "n_trials": self.n_trials,
            "sampler_seed": self.seed,
            "time_budget_seconds": self.time_budget,
            # The tuner selects on the very rows it early-stops against, so this
            # score is optimistic as well as being a selection score. Reported
            # numbers come from the held-out 20%.
            "cv_score_biased": True,
            "tuner_failed": self.tuner_failed,
            "search_space_owner": "tuner",
            "feature_fraction_grid_floor": 0.4,
        }


#: Where resumable tuning studies live, beside the model directory rather than
#: inside it.
STUDIES_DIR = "studies"


def booster_dir(models_dir: Path, study: str) -> Path:
    """Where one study keeps each trial's per-fold boosters.

    The tuner holds the best trial's booster in memory unless it is given somewhere
    to put it, and a resumed study has no memory of the run that found it. Without
    this, ``get_best_booster`` on a resumed study raises and the fold has no model.

    Keyed by the **study**, not by the endpoint, so a run under different settings
    cannot overwrite the file belonging to another study's winning trial.
    """
    return Path(models_dir) / STUDIES_DIR / f"{study}-boosters"


def study_path(models_dir: Path, key: str) -> Path:
    """Where an endpoint's study database lives, so tuning can be resumed.

    Beside the bundles, not inside one. It used to be ``<models>/<key>/optuna.sqlite``
    and :func:`admetus.bundle.save.save` replaces that whole directory atomically, so
    a successful run deleted the study it had just built: a `skin_reaction` run
    recorded 61 completed trials and left nothing behind, and raising the budget with
    `--force` started the search over.

    It is also not a model file, so publishing a bundle should not carry it.
    """
    return Path(models_dir) / STUDIES_DIR / f"{key}.sqlite"


def study_name(
    key: str,
    feature_set: str,
    n_columns: int,
    n_folds: int,
    params: dict[str, Any],
    seed: int = 0,
    fold_signature: str = "",
) -> str:
    """Name a study after the settings that decide whether its trials still apply.

    Resuming is only sound while the search is over the same thing. A different
    feature set, column count, fold count or objective all change what a trial's
    score means, so each gets its own study rather than adding trials to one that
    measured something else.

    The seed and the fold assignment are part of that identity, and leaving them out
    was wrong in a way that falsified the record: ``--seed 73`` resumed the seed-0
    study, reported ``sampler_seed: 73`` in the bundle, and shipped parameters that
    still carried seed 0. A study is now per seed, and ``fold_signature`` pins the
    actual partition so a study is never extended against different folds.
    """
    import hashlib

    signature = "|".join(
        str(x) for x in (
            feature_set,
            n_columns,
            n_folds,
            params.get("objective"),
            params.get("metric"),
            round(float(params.get("scale_pos_weight", 1.0)), 4),
            params.get("learning_rate"),
            seed,
            fold_signature,
        )
    )
    digest = hashlib.blake2b(signature.encode(), digest_size=5).hexdigest()
    return f"admetus-{key}-{digest}"


def fixed_params(task: str, metric: str, num_threads: int = 0, seed: int = 0) -> dict[str, Any]:
    """The parameters we set by rule, which the tuner then builds on.

    There is nothing left to set beyond the reference implementation's own, which
    :func:`admetus.train.cv.base_params` returns. What this used to add:

    * ``scale_pos_weight``, derived from the measured positive rate. The reference
      applies no class weighting, and its absence is the arrangement that was asked
      for. Both reported classification metrics rank rather than threshold, so the
      weighting bought little even where the imbalance is real.
    * ``regression_l1`` for the endpoints judged on MAE.
    * ``feature_pre_filter: False``, which the tuner sets for itself anyway.
    """
    return cv.base_params(task, metric, num_threads=num_threads, seed=seed)


def _direction(metric: str) -> str:
    """Which way the study optimizes, given the LightGBM metric name."""
    return "maximize" if metric in MAXIMIZE_METRICS else "minimize"


def _fallback_fit(
    lgb,
    x_train,
    y_train,
    x_valid,
    y_valid,
    params: dict[str, Any],
    study: Any,
    seed: int,
    time_budget: int | None,
) -> TuneResult:
    """Fit one fold when the tuner could not finish, using the best trial it had."""
    import optuna

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    merged = dict(params)
    score = float("nan")
    if completed:
        best_trial = min(
            completed,
            key=lambda t: t.value if study.direction == optuna.study.StudyDirection.MINIMIZE
            else -t.value,
        )
        merged.update(best_trial.params)
        score = float(best_trial.value)

    fit_set = lgb.Dataset(x_train, label=y_train, params=merged, free_raw_data=False)
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, reference=fit_set, params=merged, free_raw_data=False
    )
    booster = lgb.train(
        merged,
        fit_set,
        num_boost_round=TUNER_MAX_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(TUNER_EARLY_STOPPING, verbose=False)],
    )
    return TuneResult(
        params=merged,
        score=score,
        n_trials=len(completed),
        seed=seed,
        time_budget=time_budget,
        boosters=[booster],
        tuned_keys=sorted(k for k in merged if k not in params),
        tuner_failed=True,
    )


def tune_fold(
    x_train,
    y_train,
    x_valid,
    y_valid,
    params: dict[str, Any],
    seed: int = 0,
    time_budget: int | None = DEFAULT_TIME_BUDGET_SECONDS,
    storage: Path | None = None,
    study_name: str = "admetus",
) -> TuneResult:
    """Tune and fit **one** fold, returning the model the tuner settled on.

    This is the reference implementation's arrangement: ``lgb.train`` from
    ``optuna.integration.lightgbm`` is called once per fold, with that fold's
    validation rows as the watchlist, and the booster it returns is the ensemble
    member. Each fold therefore gets its own parameters and its own early-stopping
    point, and the 68 trials the tuner runs are per fold -- 204 for three folds,
    which is the trial count the reference's own notebooks show.

    It replaces a single ``LightGBMTunerCV`` run across all folds, which tuned one
    parameter set against the pooled fold score and then needed a separate fit. That
    is a different search over a different objective, and it was not what was asked
    for.

    Args:
        x_train: The fold's training features.
        y_train: The fold's training labels.
        x_valid: The fold's validation features, used for early stopping.
        y_valid: The fold's validation labels.
        params: The fixed parameters, including objective and metric.
        seed: Sampler seed, recorded so the run is reproducible.
        time_budget: Wall-clock cap for this fold, in seconds.
        storage: SQLite path for a resumable study.
        study_name: Study name inside the storage.

    Returns:
        A :class:`TuneResult` whose single booster is this fold's model.
    """
    import optuna
    from optuna_integration.lightgbm import LightGBMTuner

    lgb = load_lightgbm()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metric = str(params.get("metric", "binary_error"))
    storage_url = None
    if storage is not None:
        Path(storage).parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{Path(storage)}"

    study = optuna.create_study(
        direction=_direction(metric),
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )
    already = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if already:
        log.info("resuming tuning study with %d completed trial(s)", already)

    fit_set = lgb.Dataset(x_train, label=y_train, params=params, free_raw_data=False)
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, reference=fit_set, params=params, free_raw_data=False
    )

    tuner = LightGBMTuner(
        dict(params),
        fit_set,
        valid_sets=[valid_set],
        num_boost_round=TUNER_MAX_ROUNDS,
        callbacks=[lgb.early_stopping(TUNER_EARLY_STOPPING, verbose=False)],
        time_budget=time_budget,
        study=study,
        optuna_seed=seed,
        show_progress_bar=not progress_disabled(),
    )
    try:
        tuner.run()
    except Exception as exc:  # noqa: BLE001 - the tuner's own failure, not ours
        # LightGBMTuner raises `AssertionError: Should not reach.` from inside its
        # stepwise search on some runs, and a tuner that gives up must not take the
        # endpoint with it. Whatever trials completed are still in the study, so the
        # caller gets the best of those; with none, it gets the fixed parameters and
        # a straight early-stopped fit.
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        log.warning(
            "the tuner failed after %d completed trial(s) (%s: %s); falling back to the best "
            "of those", len(completed), type(exc).__name__, exc,
        )
        return _fallback_fit(
            lgb, x_train, y_train, x_valid, y_valid, params, study, seed, time_budget
        )

    best = dict(params)
    tuned_keys = sorted(
        k for k, v in tuner.best_params.items() if k not in params or params[k] != v
    )
    best.update(tuner.best_params)

    boosters: list[Any] = []
    try:
        boosters = [tuner.get_best_booster()]
    except Exception as exc:  # pragma: no cover - only if the tuner kept nothing
        log.warning("the tuner returned no booster for this fold (%s)", exc)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    return TuneResult(
        params=best,
        score=float(tuner.best_score),
        n_trials=completed,
        seed=seed,
        time_budget=time_budget,
        boosters=boosters,
        tuned_keys=tuned_keys,
    )


def default_params(task: str, metric: str, n_train: int, num_threads: int = 0, seed: int = 0) -> dict[str, Any]:
    """A reasonable parameter set for when tuning is skipped.

    Used by the fast path in tests and by ``--no-tune``; the tuner is the default.
    """
    params = fixed_params(task, metric, num_threads=num_threads, seed=seed)
    params.update(
        {
            "feature_fraction": 0.4,
            "num_leaves": 16 if n_train < 500 else 31,
            "min_child_samples": max(5, round(0.03 * n_train)),
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
        }
    )
    return params


__all__ = [
    "STUDIES_DIR",
    "booster_dir",
    "DEFAULT_TIME_BUDGET_SECONDS",
    "MAXIMIZE_METRICS",
    "TUNER_EARLY_STOPPING",
    "TUNER_MAX_ROUNDS",
    "TuneResult",
    "tune_fold",
    "default_params",
    "fixed_params",
    "study_name",
    "study_path",
]
