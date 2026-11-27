"""Feature-level leakage auditing."""

from __future__ import annotations

import numpy as np

from admetus.chem.featurizer import build_csr
from admetus.data.audit import audit_split, resolve_within_train, row_hashes


def matrix_from(rows: list[list[int]], n_cols: int = 16):
    return build_csr([np.array(r, dtype=np.int32) for r in rows], n_cols)


def test_identical_rows_hash_alike_and_different_rows_do_not():
    hashes = row_hashes(matrix_from([[1, 2], [1, 2], [3, 4]]))
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_column_order_does_not_change_the_hash():
    a = row_hashes(matrix_from([[5, 1, 3]]))
    b = row_hashes(matrix_from([[1, 3, 5]]))
    assert a == b


def test_a_clean_split_is_reported_clean():
    matrix = matrix_from([[1], [2], [3], [4]])
    result = audit_split(matrix, np.array([0.0, 1, 0, 1]), np.array([0, 1]), np.array([2, 3]))
    assert result.is_clean
    assert result.drop_from_train.size == 0


def test_a_duplicate_across_the_split_is_dropped_from_train_only():
    """Row 0 and row 2 are indistinguishable; the test row must survive."""
    matrix = matrix_from([[1, 2], [3], [1, 2], [4]])
    result = audit_split(matrix, np.array([1.0, 0, 1, 0]), np.array([0, 1]), np.array([2, 3]))
    assert not result.is_clean
    assert result.n_cross_split == 1
    assert result.drop_from_train.tolist() == [0]
    assert 2 not in result.drop_from_train.tolist()


def test_label_disagreement_across_the_split_is_counted():
    matrix = matrix_from([[1, 2], [1, 2]])
    result = audit_split(matrix, np.array([1.0, 0.0]), np.array([0]), np.array([1]))
    assert result.label_disagreements == 1


def test_duplicates_entirely_inside_train_are_reported_not_dropped():
    matrix = matrix_from([[1, 2], [1, 2], [9]])
    result = audit_split(matrix, np.array([1.0, 1.0, 0.0]), np.array([0, 1]), np.array([2]))
    assert result.is_clean
    assert result.within_train_groups == 1
    assert result.drop_from_train.size == 0


def test_audit_record_is_serializable():
    matrix = matrix_from([[1], [2]])
    payload = audit_split(matrix, np.array([0.0, 1.0]), np.array([0]), np.array([1])).to_dict()
    assert payload["n_rows"] == 2
    assert set(payload) >= {"n_duplicate_groups", "n_cross_split_groups", "n_dropped_from_train"}


def test_within_train_regression_duplicates_become_the_median():
    matrix = matrix_from([[1, 2], [1, 2], [1, 2], [7]])
    train = np.array([0, 1, 2, 3])
    keep, y, counts = resolve_within_train(matrix, np.array([1.0, 3.0, 11.0, 5.0]), train, "regression")
    assert counts["n_collapsed"] == 2
    assert y[keep[0]] == 3.0
    assert keep.size == 2


def test_within_train_contradictory_classification_duplicates_are_dropped():
    matrix = matrix_from([[1, 2], [1, 2], [7]])
    train = np.array([0, 1, 2])
    keep, _, counts = resolve_within_train(matrix, np.array([1.0, 0.0, 1.0]), train, "classification")
    assert counts["n_dropped_contradictory"] == 2
    assert keep.tolist() == [2]


def test_within_train_unanimous_duplicates_collapse_to_one():
    matrix = matrix_from([[1, 2], [1, 2], [7]])
    train = np.array([0, 1, 2])
    keep, _, counts = resolve_within_train(matrix, np.array([1.0, 1.0, 0.0]), train, "classification")
    assert counts["n_collapsed"] == 1
    assert keep.size == 2
