"""Choosing which feature columns to keep.

The full space is 89,852 columns and the datasets hold between 276 and 13,445
rows, so most columns carry nothing. Two reasons to drop them deliberately rather
than leave it to the model:

**Rare columns invite spurious splits.** A binary column set in exactly five
molecules separates them perfectly, on a balanced label, with probability
2 x 0.5^5 = 6.25%. With thousands of such columns, hundreds will look like
perfect five-sample rules.

**The threshold has to scale with the dataset.** A fixed count of five means
"present in 1.8% of molecules" at n=276 and "present in 0.04%" at n=13,445. The
band is therefore expressed as a prevalence, and it is two-sided: a column set in
99% of molecules is as uninformative as one set in 1%.

Selection uses the feature matrix only, never the labels, so it cannot leak label
information into the estimate. A label-aware filter such as mutual information
would, severely, at this ratio of columns to rows.

Columns are addressed through a **basis**: the retained raw column indices, plus
an ``int32`` map from every raw column to its position in the pruned space with
``-1`` for dropped columns. A basis is fitted per endpoint, on its training rows
only, and stored with the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from admetus.data.registry import MAX_FEATURES, PrevalenceRange
from admetus.errors import InputDataError
from admetus.logging import get_logger

log = get_logger("train.prune")


@dataclass(frozen=True)
class ColumnBasis:
    """A pruned column space.

    Attributes:
        kept: Raw column indices retained, ascending.
        n_raw: Width of the raw space the map covers.
    """

    kept: np.ndarray
    n_raw: int

    def __post_init__(self) -> None:
        kept = np.asarray(self.kept)
        if kept.ndim != 1:
            raise ValueError("kept must be one-dimensional")
        if kept.size and (np.any(np.diff(kept) <= 0) or kept[0] < 0 or kept[-1] >= self.n_raw):
            raise ValueError("kept must be strictly ascending and within the raw space")

    @property
    def n_cols(self) -> int:
        """Width of the pruned space."""
        return int(self.kept.size)

    @property
    def col_map(self) -> np.ndarray:
        """Map from raw column index to pruned position, ``-1`` where dropped.

        Monotonic by construction, because ``kept`` is ascending, so projecting a
        sorted row keeps it sorted.
        """
        out = np.full(self.n_raw, -1, dtype=np.int32)
        out[self.kept] = np.arange(self.kept.size, dtype=np.int32)
        return out

    def to_dict(self) -> dict[str, object]:
        """Record for the bundle. The indices themselves live in the npz."""
        return {"n_raw": self.n_raw, "n_cols": self.n_cols}

    def describe(self) -> str:
        share = self.n_cols / self.n_raw if self.n_raw else 0.0
        return f"{self.n_cols} of {self.n_raw} columns retained ({share:.2%})"


def column_counts(matrix: sparse.csr_matrix) -> np.ndarray:
    """Count the rows in which each column is non-zero."""
    binary = matrix.tocsc(copy=False)
    return np.diff(binary.indptr).astype(np.int64)


def fit_basis(
    matrix: sparse.csr_matrix,
    band: PrevalenceRange,
    max_features: int = MAX_FEATURES,
) -> ColumnBasis:
    """Select columns from a training matrix.

    Args:
        matrix: Training feature matrix. Only this matrix's rows may be used:
            including held-out rows would make the selection depend on data the
            estimate is supposed to be independent of.
        band: Prevalence band to retain.
        max_features: Cap on retained columns, applied by descending prevalence.

    Returns:
        The fitted :class:`ColumnBasis`.

    Raises:
        InputDataError: The band retains nothing.
    """
    n_rows, n_raw = matrix.shape
    if n_rows == 0:
        raise InputDataError("cannot fit a column basis on an empty matrix")

    counts = column_counts(matrix)
    prevalence = counts / n_rows
    keep = (
        (prevalence >= band.low)
        & (prevalence <= band.high)
        & (counts >= band.min_count)
        & (counts < n_rows)  # a column set in every row separates nothing
    )
    kept = np.flatnonzero(keep).astype(np.int32)

    if kept.size == 0:
        raise InputDataError(
            f"no column survived pruning at {band.describe()} over {n_rows} rows. "
            "The feature matrix is empty or the band is too narrow."
        )

    if kept.size > max_features:
        # Keep the most prevalent, which are the ones with enough support to split on.
        order = np.argsort(-counts[kept], kind="stable")[:max_features]
        kept = np.sort(kept[order]).astype(np.int32)
        log.debug("capped the basis at %d columns", max_features)

    basis = ColumnBasis(kept=kept, n_raw=int(n_raw))
    log.info("%s at %s over %d rows", basis.describe(), band.describe(), n_rows)
    return basis


def apply_basis(matrix: sparse.csr_matrix, basis: ColumnBasis) -> sparse.csr_matrix:
    """Narrow a raw-space matrix to a basis.

    Uses scipy's column indexing, which was measured at 11.5 ms for a
    10,000-row by 89,852-column matrix with 4.26 million non-zeros -- about 3% of
    the time that chunk takes to featurize, and faster than reindexing
    ``indices``/``indptr`` by hand (20.7 ms). Serving all 29 endpoints from one
    raw matrix therefore costs roughly 330 ms per chunk, so there is no reason to
    complicate the featurizer with a shared intermediate basis.

    Raises:
        InputDataError: The matrix is not in the raw space the basis describes.
    """
    if matrix.shape[1] != basis.n_raw:
        raise InputDataError(
            f"matrix has {matrix.shape[1]} columns but the basis describes a "
            f"{basis.n_raw}-column raw space; it was featurized against a different spec"
        )
    return matrix[:, basis.kept].tocsr()


def project(matrix: sparse.csr_matrix, col_map: np.ndarray, n_cols: int) -> sparse.csr_matrix:
    """Reindex a CSR matrix through a column map.

    Needed where a basis is applied to indices as they are produced rather than to
    an assembled matrix. For narrowing a matrix that already exists, prefer
    :func:`apply_basis`.

    Args:
        matrix: Source matrix.
        col_map: Monotonic map from source column to target position, ``-1`` to drop.
        n_cols: Width of the target space.
    """
    source = matrix.tocsr(copy=False)
    mapped = col_map[source.indices]
    keep = mapped >= 0

    # Row boundaries follow from the running count of retained entries, which
    # also handles empty rows without special cases.
    running = np.zeros(keep.size + 1, dtype=np.int64)
    np.cumsum(keep, out=running[1:])
    indptr = running[source.indptr]

    indices = mapped[keep].astype(np.int32, copy=False)
    data = np.ones(indices.size, dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(source.shape[0], n_cols))


__all__ = ["ColumnBasis", "apply_basis", "column_counts", "fit_basis", "project"]
