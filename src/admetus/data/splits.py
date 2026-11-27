"""Splitting the data, the way the reference implementation does.

Two stages, and the order matters:

1. A random 80/20 holdout over the whole dataset, seeded. The 20% is set aside and
   not looked at again until the model is finished.
2. ``KFold`` over the 80% only, giving one (train, valid) pair per fold. Each fold
   model early-stops on its own validation rows, and the ensemble is their mean.

This replaced a scaffold-grouped scheme that was not asked for. The scaffold split
sent the largest scaffold groups to training on purpose, which made the held-out set
the unusual chemistry and the estimate pessimistic -- defensible as a choice, but it
was a choice nobody requested, it answered a different question than the reference
implementation and the published benchmark both answer, and it cost real correctness:
grouping by input string rather than by standardized structure put 9 of 129 held-out
hERG molecules in a scaffold the training set also held.

``KFold`` is used without shuffling, as the reference does, because the rows reaching
it are already in random order: ``train_test_split`` shuffles before it cuts. Adding a
second shuffle would only make the fold assignment depend on two seeds instead of one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from admetus.errors import InputDataError

#: Share of the dataset held out, matching the reference implementation's default.
DEFAULT_TEST_FRACTION = 0.2

#: Cross-validation folds over the training portion, and so the number of models in
#: the ensemble. The reference implementation's default, kept as the default here;
#: ``--folds`` raises it when a steadier out-of-fold estimate is worth the compute.
DEFAULT_FOLDS = 3

#: Below this many rows in a validation fold, a metric computed on it is noise.
MIN_FOLD_ROWS = 20


@dataclass(frozen=True)
class Split:
    """A train/test partition by row index."""

    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.train, self.test).size:
            raise ValueError("train and test overlap")


def random_holdout(
    n_rows: int, test_fraction: float = DEFAULT_TEST_FRACTION, seed: int = 0
) -> Split:
    """Hold out a random share of the rows.

    Uses scikit-learn's ``train_test_split`` over row indices, so the partition is the
    one the reference implementation produces for the same seed and fraction.

    Raises:
        InputDataError: The fraction is unusable, or it leaves a side empty.
    """
    if not 0 < test_fraction < 1:
        raise InputDataError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    if n_rows < 2:
        raise InputDataError(f"cannot split {n_rows} row(s)")

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        np.arange(n_rows), test_size=test_fraction, random_state=seed
    )
    if train.size == 0 or test.size == 0:
        raise InputDataError(
            f"a {test_fraction:.0%} holdout of {n_rows} row(s) leaves "
            f"{train.size} training and {test.size} evaluation rows"
        )
    return Split(train=np.sort(train), test=np.sort(test))


def cv_folds(n_rows: int, n_splits: int = DEFAULT_FOLDS) -> list[Split]:
    """Build cross-validation folds over the training portion.

    Folds are reduced rather than abandoned when the data cannot support the
    requested count, and refused outright when even three folds would leave a
    validation fold too thin to score. The indices are positions within whatever
    array is passed, so the caller maps them back itself.

    Raises:
        InputDataError: Not enough rows for three scorable folds.
    """
    from sklearn.model_selection import KFold

    if n_rows < 3 * MIN_FOLD_ROWS:
        raise InputDataError(
            f"{n_rows} training row(s) cannot support three folds of at least "
            f"{MIN_FOLD_ROWS} validation rows each. Scoring folds that thin measures "
            "noise, so this is refused rather than tuned against."
        )

    requested = n_splits
    for candidate in range(n_splits, 2, -1):
        if n_rows // candidate >= MIN_FOLD_ROWS:
            n_splits = candidate
            break
    else:  # pragma: no cover - the guard above already covers this
        raise InputDataError(f"{n_rows} training row(s) cannot support three folds")

    if n_splits != requested:
        from admetus.logging import get_logger

        get_logger("data.splits").warning(
            "reduced to %d fold(s) from %d: %d training rows cannot fill more",
            n_splits, requested, n_rows,
        )

    indices = np.arange(n_rows)
    splitter = KFold(n_splits=n_splits)
    return [
        Split(train=train, test=test) for train, test in splitter.split(indices)
    ]


def fold_signature(folds: list[Split]) -> str:
    """A short digest of an actual fold partition.

    Part of a tuning study's identity: trials are only comparable while they were
    scored against the same validation rows, and a fold count alone does not say that.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=6)
    for fold in folds:
        digest.update(np.asarray(fold.test, dtype=np.int64).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def fold_report(folds: list[Split], y: np.ndarray, task: str) -> list[dict[str, object]]:
    """Summarize each fold, for the training log and the bundle record."""
    out: list[dict[str, object]] = []
    for i, fold in enumerate(folds):
        entry: dict[str, object] = {
            "fold": i,
            "n_train": int(fold.train.size),
            "n_valid": int(fold.test.size),
        }
        if task == "classification":
            entry["n_valid_positive"] = int(np.count_nonzero(y[fold.test] == 1))
        out.append(entry)
    return out


__all__ = [
    "DEFAULT_FOLDS",
    "DEFAULT_TEST_FRACTION",
    "MIN_FOLD_ROWS",
    "Split",
    "cv_folds",
    "fold_report",
    "fold_signature",
    "random_holdout",
]
