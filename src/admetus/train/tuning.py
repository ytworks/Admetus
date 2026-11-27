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

import numpy as np
from scipy import sparse

from admetus._compat import load_lightgbm
from admetus.data.splits import Split
from admetus.logging import get_logger, progress_disabled
from admetus.train import cv

log = get_logger("train.tuning")

#: Positive rate at or above which class weighting is not applied at all.
BALANCED_POS_RATE = 0.20

#: Positive rate below which the full negative/positive ratio is used.
SEVERE_POS_RATE = 0.05

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


def class_weight(pos_rate: float, n_pos: int, n_neg: int) -> float:
    """Pick ``scale_pos_weight`` from the measured positive rate.

    A fixed value rather than a range, because the tuner cannot search it.

    Both reported classification metrics rank rather than threshold, so weighting
    buys little; it is applied only where the imbalance is real. Of the twenty
    classification endpoints here, one assay is around 4% positive and two
    substrate datasets sit between 20 and 30%; the CYP inhibition panels are
    30-46% and need nothing.

    Returns:
        The weight, ``1.0`` when none is warranted.
    """
    if n_pos == 0 or n_neg == 0 or pos_rate >= BALANCED_POS_RATE:
        return 1.0
    ratio = n_neg / n_pos
    if pos_rate >= SEVERE_POS_RATE:
        # The full ratio is an over-aggressive folk default; its square root
        # shifts the gradient without flattening the probabilities.
        return float(np.sqrt(ratio))
    return float(ratio)


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

    def to_dict(self) -> dict[str, Any]:
        """The record stored in the bundle."""
        return {
            "tuner": "optuna_integration.lightgbm.LightGBMTunerCV",
            "best_params": {k: v for k, v in self.params.items() if k != "num_threads"},
            "tuned_by_tuner": self.tuned_keys,
            "best_cv_score": self.score,
            "n_trials": self.n_trials,
            "sampler_seed": self.seed,
            "time_budget_seconds": self.time_budget,
            # The tuner selects on the same fold predictions it scores, and its
            # objective is the mean over folds. This ranks candidates; it is not a
            # performance estimate. Reported numbers come from the held-out set.
            "cv_score_biased": True,
            "search_space_owner": "tuner",
            "feature_fraction_grid_floor": 0.4,
            "feature_pre_filter": False,
        }


#: Where resumable tuning studies live, beside the model directory rather than
#: inside it.
STUDIES_DIR = "studies"


def booster_dir(models_dir: Path, study: str) -> Path:
    """Where one study keeps each trial's per-fold boosters.

    ``LightGBMTunerCV`` holds the best trial's boosters in memory unless it is given
    somewhere to put them, and a resumed study has no memory of the run that found
    them. Without this, ``get_best_booster`` on a resumed study raises and the
    production round count silently falls back to its default.

    Keyed by the **study**, not by the endpoint. One directory per endpoint let a run
    under a different feature set overwrite the file belonging to another study's
    winning trial: the first study kept its score and its 65 trials, and its resumed
    ensemble came back at 350 rounds instead of 143, having loaded somebody else's
    booster. Each study now gets its own.
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


def fixed_params(
    task: str,
    metric: str,
    scale_pos_weight: float = 1.0,
    objective: str | None = None,
    num_threads: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """The parameters we set by rule, which the tuner then builds on."""
    params = cv.base_params(task, metric, num_threads=num_threads, seed=seed)
    if objective:
        params["objective"] = objective
    if scale_pos_weight != 1.0:
        params["scale_pos_weight"] = scale_pos_weight
    # The tuner varies min_child_samples, so it disables this itself. Stated here
    # so the bundle's record matches what actually ran.
    params["feature_pre_filter"] = False
    return params


def _direction(metric: str) -> str:
    """Which way the study optimizes, given the LightGBM metric name."""
    return "maximize" if metric in MAXIMIZE_METRICS else "minimize"


def auto_tune(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    folds: list[Split],
    params: dict[str, Any],
    seed: int = 0,
    time_budget: int | None = DEFAULT_TIME_BUDGET_SECONDS,
    storage: Path | None = None,
    study_name: str = "admetus",
    model_dir: Path | None = None,
) -> TuneResult:
    """Tune hyperparameters with the stepwise LightGBM tuner.

    Args:
        matrix: Training features, already narrowed to the endpoint's basis.
        y: Training labels in the fitted space.
        folds: Scaffold-grouped cross-validation folds, passed straight through.
        params: The parameters we fix by rule, including objective and metric.
        seed: Sampler seed. Recorded, since an unseeded sampler would make the
            command line's determinism promise false and change results on resume.
        time_budget: Wall-clock cap in seconds, or ``None`` for no cap.
        storage: SQLite path for a resumable study.
        study_name: Study name inside the storage.
        model_dir: Where the tuner persists per-trial boosters, so a resumed study
            can still produce the best trial's folds.

    Returns:
        The :class:`TuneResult`. Its per-fold boosters are the tuner's own; the
        ensemble that ships is fitted separately from the chosen parameters.
    """
    import optuna
    from optuna_integration.lightgbm import LightGBMTunerCV

    lgb = load_lightgbm()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metric = str(params.get("metric", "auc"))
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

    # The tuner takes folds as (train, valid) index pairs.
    fold_pairs = [(f.train, f.test) for f in folds]
    dataset = lgb.Dataset(matrix, label=y, params=params, free_raw_data=False)

    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    tuner = LightGBMTunerCV(
        dict(params),
        dataset,
        folds=fold_pairs,
        model_dir=str(model_dir) if model_dir else None,
        num_boost_round=TUNER_MAX_ROUNDS,
        callbacks=[lgb.early_stopping(TUNER_EARLY_STOPPING, verbose=False)],
        time_budget=time_budget,
        study=study,
        seed=seed,
        optuna_seed=seed,
        return_cvbooster=True,
        show_progress_bar=not progress_disabled(),
    )
    tuner.run()

    best = dict(params)
    # best_params is the full merged set, so diff it against what we fixed to
    # record only what the tuner actually chose.
    tuned_keys = sorted(
        k for k, v in tuner.best_params.items() if k not in params or params[k] != v
    )
    best.update(tuner.best_params)

    boosters: list[Any] = []
    try:
        cvbooster = tuner.get_best_booster()
        boosters = list(cvbooster.boosters)
    except Exception as exc:  # pragma: no cover - only if the tuner kept nothing
        log.warning("the tuner returned no per-fold boosters (%s); refitting from scratch", exc)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log.info(
        "tuned %d trial(s), best %s = %.4f, tuner chose %s",
        completed, metric, tuner.best_score, ", ".join(tuned_keys),
    )
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
    "BALANCED_POS_RATE",
    "STUDIES_DIR",
    "booster_dir",
    "DEFAULT_TIME_BUDGET_SECONDS",
    "MAXIMIZE_METRICS",
    "SEVERE_POS_RATE",
    "TUNER_EARLY_STOPPING",
    "TUNER_MAX_ROUNDS",
    "TuneResult",
    "auto_tune",
    "class_weight",
    "default_params",
    "fixed_params",
    "study_name",
    "study_path",
]
