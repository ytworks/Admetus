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

#: Below this many training rows in a fold, early stopping does more harm than good.
MIN_ROWS_FOR_EARLY_STOPPING = 1_000

#: Share of a fold's training rows held back as an early-stopping watchlist.
WATCHLIST_FRACTION = 0.12

#: Rounds without improvement before stopping.
EARLY_STOPPING_ROUNDS = 100

#: Upper bound on boosting rounds when early stopping is active.
MAX_BOOST_ROUNDS = 3_000

#: Fixed learning rate. Searching it alongside the round count wastes budget on a
#: known-redundant axis, so the budget goes to regularization instead.
LEARNING_RATE = 0.03


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
    """Parameters shared by every fit.

    The tuning metric is aligned with the metric the endpoint is judged on. A
    0/1 error rate at a fixed 0.5 threshold, which is the usual default, measures
    something nobody reports.
    """
    params: dict[str, object] = {
        "boosting_type": "gbdt",
        "learning_rate": LEARNING_RATE,
        "verbosity": -1,
        "num_threads": num_threads,
        "seed": seed,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        # Every column is binary, so two bins is all the histogram needs.
        "max_bin": 2,
        # Bitwise reproducibility across thread counts, which the CLI promises.
        "deterministic": True,
        "force_row_wise": True,
        # Leave LightGBM's own rare-feature guard on: it drops columns that cannot
        # satisfy min_data_in_leaf. Turning it off to reuse a Dataset across trials
        # would discard exactly the protection this feature space needs.
        "feature_pre_filter": True,
    }
    if task == "classification":
        params["objective"] = "binary"
        params["metric"] = "average_precision" if metric == "auprc" else "auc"
    else:
        params["objective"] = "regression_l1" if metric == "mae" else "regression"
        params["metric"] = "l1" if metric == "mae" else "l2"
    return params


def _watchlist_split(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's training rows into a fitting set and a watchlist."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_watch = max(1, int(round(n_rows * WATCHLIST_FRACTION)))
    return np.sort(order[n_watch:]), np.sort(order[:n_watch])


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
            optimistic, so the number that gets reported comes from the 20% holdout
            and never from here. The alternative -- carving an inner watchlist from
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
    elif x_train.shape[0] < MIN_ROWS_FOR_EARLY_STOPPING:
        # Too small for a watchlist; the caller should have been tuning the round
        # count instead, but fall back to a fixed, modest number of rounds.
        dataset = lgb.Dataset(x_train, label=y_train, params=params, free_raw_data=False)
        booster = lgb.train(params, dataset, num_boost_round=300)
        used_early_stopping = False
        best_iteration = 300
    else:
        inner_fit, inner_watch = _watchlist_split(x_train.shape[0], seed)
        fit_set = lgb.Dataset(x_train[inner_fit], label=y_train[inner_fit], params=params, free_raw_data=False)
        watch_set = lgb.Dataset(
            x_train[inner_watch], label=y_train[inner_watch], reference=fit_set, params=params, free_raw_data=False
        )
        booster = lgb.train(
            params,
            fit_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[watch_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        used_early_stopping = True
        best_iteration = int(booster.best_iteration or booster.num_trees())

    prediction = booster.predict(matrix[valid_rows], num_iteration=best_iteration)
    return FoldFit(
        booster=booster,
        valid_pred=np.asarray(prediction, dtype=np.float64),
        valid_rows=valid_rows,
        best_iteration=best_iteration,
        used_early_stopping=used_early_stopping,
    )


def deployment_rounds(fits: list[FoldFit]) -> int:
    """Pick the fixed round count for the models that ship.

    The median of what the folds chose, with 10% headroom because the deployed
    models see slightly more data than the watchlist-reduced fits did.
    """
    if not fits:
        raise ValueError("no folds to summarize")
    median = float(np.median([f.best_iteration for f in fits]))
    return max(10, int(round(median * 1.1)))


__all__ = [
    "EARLY_STOPPING_ROUNDS",
    "LEARNING_RATE",
    "MAX_BOOST_ROUNDS",
    "MIN_ROWS_FOR_EARLY_STOPPING",
    "WATCHLIST_FRACTION",
    "FoldFit",
    "base_params",
    "deployment_rounds",
    "fit_fold",
]
