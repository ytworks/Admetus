"""Auditing a split for molecules the model cannot tell apart.

Deduplicating on SMILES is not enough. Both fingerprint blocks are blind to
stereochemistry -- Morgan runs without chirality and the pharmacophore signature
is purely topological -- so a pair of enantiomers, or two tautomers that
standardize apart, are *identical* as far as the model is concerned. If one is in
train and the other in test, the test score is inflated by a duplicate that no
SMILES comparison would catch.

The right key is therefore the feature vector itself. Hashing each row's retained
column indices catches stereochemical collapse, tautomer collapse, salt stripping
and plain duplication in a single pass.

When a collision crosses the split, rows are dropped from **train only**. Removing
test rows would change what the reported metric is measured on and break
comparability with the published benchmark.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from admetus.logging import get_logger

log = get_logger("data.audit")


@dataclass
class AuditResult:
    """What the audit found and what it recommends dropping.

    Attributes:
        n_rows: Rows examined.
        n_duplicate_groups: Groups of rows sharing a feature vector.
        n_cross_split: Groups spanning both sides of the split.
        drop_from_train: Train row indices to remove, ascending.
        label_disagreements: Cross-split groups whose labels also disagree.
        within_train_groups: Duplicate groups entirely inside train.
    """

    n_rows: int
    n_duplicate_groups: int = 0
    n_cross_split: int = 0
    drop_from_train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    label_disagreements: int = 0
    within_train_groups: int = 0

    @property
    def is_clean(self) -> bool:
        """Whether no duplicate feature vector spans the split."""
        return self.n_cross_split == 0

    def to_dict(self) -> dict[str, object]:
        """Return the record stored in the bundle's provenance."""
        return {
            "n_rows": self.n_rows,
            "n_duplicate_groups": self.n_duplicate_groups,
            "n_cross_split_groups": self.n_cross_split,
            "n_dropped_from_train": int(self.drop_from_train.size),
            "n_label_disagreements": self.label_disagreements,
            "n_within_train_groups": self.within_train_groups,
        }

    def summary(self) -> str:
        """One line for a log."""
        if self.is_clean:
            return f"no duplicate feature vectors span the split ({self.n_rows} rows)"
        return (
            f"{self.n_cross_split} feature vector(s) appear on both sides of the split; "
            f"dropping {self.drop_from_train.size} row(s) from train "
            f"({self.label_disagreements} with disagreeing labels)"
        )


def row_hashes(matrix: sparse.csr_matrix) -> list[bytes]:
    """Hash each row's set columns.

    The matrix is binary, so the sorted column indices determine the row
    completely and nothing is lost by hashing them alone.
    """
    matrix = matrix.tocsr()
    indices, indptr = matrix.indices, matrix.indptr
    out: list[bytes] = []
    for row in range(matrix.shape[0]):
        block = indices[indptr[row] : indptr[row + 1]]
        out.append(hashlib.blake2b(np.sort(block).tobytes(), digest_size=16).digest())
    return out


def audit_split(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> AuditResult:
    """Find feature vectors that appear on both sides of a split.

    Args:
        matrix: Feature matrix for all rows, in the order ``train`` and ``test``
            index into.
        y: Labels for all rows.
        train: Training row indices.
        test: Held-out row indices.

    Returns:
        An :class:`AuditResult`. ``drop_from_train`` holds the train rows to
        remove; test rows are never dropped.
    """
    hashes = row_hashes(matrix)
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for position, digest in enumerate(hashes):
        buckets[digest].append(position)

    test_set = set(test.tolist())
    train_set = set(train.tolist())

    drop: list[int] = []
    n_duplicate_groups = 0
    n_cross_split = 0
    disagreements = 0
    within_train = 0

    for positions in buckets.values():
        if len(positions) == 1:
            continue
        n_duplicate_groups += 1
        in_train = [p for p in positions if p in train_set]
        in_test = [p for p in positions if p in test_set]
        if in_test and in_train:
            n_cross_split += 1
            drop.extend(in_train)
            if len({float(y[p]) for p in positions}) > 1:
                disagreements += 1
        elif len(in_train) > 1:
            within_train += 1

    result = AuditResult(
        n_rows=matrix.shape[0],
        n_duplicate_groups=n_duplicate_groups,
        n_cross_split=n_cross_split,
        drop_from_train=np.array(sorted(drop), dtype=np.int64),
        label_disagreements=disagreements,
        within_train_groups=within_train,
    )
    if not result.is_clean:
        log.warning("%s", result.summary())
    else:
        log.debug("%s", result.summary())
    return result


def resolve_within_train(
    matrix: sparse.csr_matrix, y: np.ndarray, train: np.ndarray, task: str
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Collapse duplicate feature vectors inside the training set.

    Rows the model cannot distinguish but whose labels differ are contradictions:
    a regression target becomes the group median, and a classification group with
    disagreeing labels is dropped because it teaches nothing.

    Returns:
        ``(keep, y_resolved, counts)`` where ``keep`` indexes into ``train``'s own
        order, ``y_resolved`` is the full label array with medians applied, and
        ``counts`` records what happened.
    """
    hashes = row_hashes(matrix)
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for position in train.tolist():
        buckets[hashes[position]].append(position)

    y_resolved = np.asarray(y, dtype=np.float64).copy()
    keep: list[int] = []
    n_collapsed = 0
    n_dropped = 0

    for positions in buckets.values():
        if len(positions) == 1:
            keep.append(positions[0])
            continue
        values = {float(y_resolved[p]) for p in positions}
        if len(values) == 1:
            keep.append(positions[0])
            n_collapsed += len(positions) - 1
            continue
        if task == "classification":
            n_dropped += len(positions)
            continue
        survivor = positions[0]
        y_resolved[survivor] = float(np.median([y_resolved[p] for p in positions]))
        keep.append(survivor)
        n_collapsed += len(positions) - 1

    counts = {"n_collapsed": n_collapsed, "n_dropped_contradictory": n_dropped}
    if n_collapsed or n_dropped:
        log.debug(
            "train set: collapsed %d indistinguishable row(s), dropped %d contradictory row(s)",
            n_collapsed, n_dropped,
        )
    return np.array(sorted(keep), dtype=np.int64), y_resolved, counts


__all__ = ["AuditResult", "audit_split", "resolve_within_train", "row_hashes"]
