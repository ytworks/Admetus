"""Fitting one cross-validation fold.

The subtlety this module exists for: if a fold's validation rows both stop the
boosting and score the trial, the resulting score is a maximum over hundreds of
correlated checkpoints rather than an estimate, and the hyperparameter search
starts preferring configurations whose validation curve happens to spike. That
favours exactly the settings that overfit -- high learning rate, many leaves, few
rows per leaf.

So the two jobs are separated:

* With at least ~1,000 training rows in the fold, 12% of the **fold's own training
  rows** become an early-stopping watchlist, and the fold's validation rows are
  used only for scoring. No extra model is fitted.
* Below that, a watchlist would be a few dozen rows and worse than useless, so
  early stopping is switched off entirely and the number of rounds becomes an
  ordinary search dimension instead.

Deployed models are refitted at a fixed round count, so no model that ships was
stopped using data it was later scored on, and all folds end up the same size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from admetus._compat import load_lightgbm
from admetus.data.splits import Split

#: Rounds without improvement before stopping.
EARLY_STOPPING_ROUNDS = 100

#: Upper bound on boosting rounds when early stopping is active.
MAX_BOOST_ROUNDS = 3_000


@dataclass(frozen=True)
class FoldFit:
    """One fitted fold.

    Attributes:
        booster: The fitted model.
        valid_pred: Predictions for the fold's validation rows, in fold order.
        valid_rows: Indices of those rows in the training-set frame.
        best_iteration: Rounds actually used.
        used_early_stopping: Whether a watchlist drove the round count.
    """

    booster: object
    valid_pred: np.ndarray
    valid_rows: np.ndarray
    best_iteration: int
    used_early_stopping: bool


def base_params(task: str, metric: str, num_threads: int = 0, seed: int = 0) -> dict[str, object]:
    """Parameters shared by every fit: the reference implementation's, plus knobs.

    The reference sets four things and nothing else::

        {'boosting_type': 'gbdt', 'objective': 'binary',
         'metric': 'binary_error', 'verbosity': -1}

    and for regression ``objective: regression`` with ``metric: rmse``. Those are
    what this returns. ``metric`` is not the metric the endpoint is judged on: a 0/1
    error rate at a fixed 0.5 threshold is what the reference tunes against, and
    aligning it with AUROC or average precision was an unrequested change.

    ``metric`` is also what the tuner optimizes, so this choice decides the search.
    The endpoint's own metric still decides what is *reported*, from the held-out
    20%, and the pooled out-of-fold score recorded in each bundle is computed with
    it too.

    Everything else here is a knob that does not change the model:

    * ``num_threads`` and ``seed`` make a run reproducible, which the command line
      promises; the reference leaves both to LightGBM.
    * ``max_bin`` of 2 is exact for this feature space rather than an
      approximation: every column is a 0/1 bit, so two bins hold it without loss,
      and the default 255 would only cost memory.
    * ``deterministic`` and ``force_row_wise`` buy bitwise reproducibility across
      thread counts.

    What is *not* here any more, each an unrequested addition: a learning rate, a
    bagging default, ``regression_l1`` for the endpoints judged on MAE, and
    ``feature_pre_filter``, which the tuner sets for itself.
    """
    params: dict[str, object] = {
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": num_threads,
        "seed": seed,
        "max_bin": 2,
        "deterministic": True,
        "force_row_wise": True,
    }
    _ = metric
    if task == "classification":
        params["objective"] = "binary"
        params["metric"] = "binary_error"
    else:
        params["objective"] = "regression"
        params["metric"] = "rmse"
    return params


def fit_fold(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    fold: Split,
    params: dict[str, object],
    num_boost_round: int | None = None,
    seed: int = 0,
    stop_on_fold: bool = False,
) -> FoldFit:
    """Fit one fold and predict its validation rows.

    Args:
        matrix: Feature matrix for the whole training set.
        y: Labels for the whole training set, in the fitted space.
        fold: Which rows train and which are held out.
        params: LightGBM parameters, already including the objective.
        num_boost_round: Fixed round count. When ``None``, early stopping decides.
        seed: Seed for the watchlist split, when one is carved.
        stop_on_fold: Early-stop on the fold's **own** validation rows, which is
            what the reference implementation does: each ensemble member stops on
            the rows it is then asked to predict. That makes this fold's prediction
            optimistic *as an estimate of that member's own accuracy*, so the number
            that gets reported comes from the 20% holdout and never from here. It
            does not follow that the pooled out-of-fold score exceeds the held-out
            one: that compares a single booster per row with the mean of all of
            them, and on the shipped bundles out-of-fold is the lower of the two at
            23 of 29 endpoints. The alternative -- carving an inner watchlist from
            the fold's training rows -- keeps the fold prediction honest but is not
            the arrangement that was asked for.

    Returns:
        The fitted fold.
    """
    lgb = load_lightgbm()

    train_rows, valid_rows = fold.train, fold.test
    x_train, y_train = matrix[train_rows], y[train_rows]

    if num_boost_round is not None:
        dataset = lgb.Dataset(x_train, label=y_train, params=params, free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
        used_early_stopping = False
        best_iteration = num_boost_round
    elif stop_on_fold:
        fit_set = lgb.Dataset(x_train, label=y_train, params=params, free_raw_data=False)
        valid_set = lgb.Dataset(
            matrix[valid_rows], label=y[valid_rows], reference=fit_set, params=params,
            free_raw_data=False,
        )
        booster = lgb.train(
            params,
            fit_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        used_early_stopping = True
        best_iteration = int(booster.best_iteration or booster.num_trees())
    else:
        raise ValueError(
            "fit_fold needs either a fixed num_boost_round or stop_on_fold; carving an inner "
            "watchlist from the fold's training rows is not what the reference implementation does"
        )

    prediction = booster.predict(matrix[valid_rows], num_iteration=best_iteration)
    return FoldFit(
        booster=booster,
        valid_pred=np.asarray(prediction, dtype=np.float64),
        valid_rows=valid_rows,
        best_iteration=best_iteration,
        used_early_stopping=used_early_stopping,
    )



__all__ = [
    "EARLY_STOPPING_ROUNDS",
    "MAX_BOOST_ROUNDS",
    "FoldFit",
    "base_params",
    "fit_fold",
]
