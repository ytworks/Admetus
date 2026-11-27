"""Splitting: the reference implementation's two stages, and what is refused.

The scaffold-grouped scheme this replaced had its own test module, which went with
it. What is left is small on purpose: a seeded random holdout and ``KFold`` over the
training portion. The tests that matter now are that the holdout is reproducible,
that the folds never see held-out rows, and that too little data is refused rather
than split into folds too thin to score.
"""

from __future__ import annotations

import numpy as np
import pytest

from admetus.data.splits import (
    DEFAULT_FOLDS,
    DEFAULT_TEST_FRACTION,
    MIN_FOLD_ROWS,
    cv_folds,
    fold_report,
    fold_signature,
    random_holdout,
)
from admetus.errors import InputDataError


def test_the_defaults_are_the_reference_implementation_s():
    """20% held out, and folds over the rest. The reference defaults to 3 folds."""
    assert DEFAULT_TEST_FRACTION == 0.2
    assert DEFAULT_FOLDS == 3


def test_the_holdout_is_the_requested_share_and_is_reproducible():
    first = random_holdout(500, test_fraction=0.2, seed=0)
    again = random_holdout(500, test_fraction=0.2, seed=0)

    assert first.test.size == 100
    assert first.train.size == 400
    assert np.array_equal(first.test, again.test)
    # Every row is on exactly one side.
    assert np.array_equal(np.union1d(first.train, first.test), np.arange(500))


def test_a_different_seed_gives_a_different_holdout():
    a = random_holdout(500, seed=0)
    b = random_holdout(500, seed=1)
    assert not np.array_equal(a.test, b.test)


def test_it_matches_sklearn_for_the_same_seed_order_included():
    """The partition and its order both have to match the reference implementation.

    The order is not cosmetic. `cv_folds` cuts contiguous blocks, so whatever order
    reaches it decides fold membership, and these dataset files are sorted by label.
    Sorting the indices back into file order gave `bioavailability` a validation fold
    of 171 rows that were 100% positive: `binary_error` was minimal at iteration 1,
    the model stopped there, and held-out AUROC fell to 0.5121 from the reference's
    0.6867.
    """
    from sklearn.model_selection import train_test_split

    expected_train, expected_test = train_test_split(
        np.arange(280), test_size=0.2, random_state=0
    )
    split = random_holdout(280, test_fraction=0.2, seed=0)

    assert np.array_equal(split.test, expected_test)
    assert np.array_equal(split.train, expected_train)
    # Which is to say: shuffled, not sorted.
    assert not np.array_equal(split.train, np.sort(split.train))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_an_unusable_holdout_fraction_is_refused(fraction):
    with pytest.raises(InputDataError, match="test_fraction"):
        random_holdout(500, test_fraction=fraction)


def test_folds_cover_every_training_row_exactly_once_as_validation():
    folds = cv_folds(300, n_splits=3)
    assert len(folds) == 3

    seen = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(seen), np.arange(300))
    for fold in folds:
        assert np.intersect1d(fold.train, fold.test).size == 0


def test_folds_are_reduced_rather_than_abandoned_when_data_is_thin():
    # 100 rows cannot fill five folds of 20, but can fill four... and three.
    folds = cv_folds(100, n_splits=5)
    assert 3 <= len(folds) <= 5
    assert min(f.test.size for f in folds) >= MIN_FOLD_ROWS - 1


def test_too_little_data_is_refused_not_fudged():
    """Scoring folds that thin measures noise, so it is refused loudly."""
    with pytest.raises(InputDataError, match="cannot support three folds"):
        cv_folds(3 * MIN_FOLD_ROWS - 1)


def test_the_fold_signature_pins_the_actual_partition():
    three = cv_folds(300, n_splits=3)
    five = cv_folds(300, n_splits=5)
    assert fold_signature(three) == fold_signature(cv_folds(300, n_splits=3))
    assert fold_signature(three) != fold_signature(five)


def test_the_fold_report_counts_positives_for_classification():
    y = np.array([0, 1] * 150, dtype=float)
    rows = fold_report(cv_folds(300, n_splits=3), y, "classification")
    assert len(rows) == 3
    assert all("n_valid_positive" in row for row in rows)
    assert sum(row["n_valid_positive"] for row in rows) == 150


def test_the_fold_report_omits_positives_for_regression():
    y = np.linspace(0.0, 1.0, 300)
    rows = fold_report(cv_folds(300, n_splits=3), y, "regression")
    assert all("n_valid_positive" not in row for row in rows)
