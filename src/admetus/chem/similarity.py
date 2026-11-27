"""Tanimoto similarity against a reference set, for applicability domain.

The question asked at inference is narrow: *is this molecule's nearest training
neighbour at least as similar as some threshold?* That is a boolean, and answering
it is much cheaper than computing the actual maximum:

* fingerprints are stored as packed bits, so a 2048-bit comparison is 32 64-bit
  words rather than 2048 floats. Packed, 116,000 reference fingerprints occupy
  about 30 MB; as ``float32`` they would be 950 MB;
* references are visited in order of popcount, and the bound
  ``|A and B| / |A or B| <= min(|A|,|B|) / max(|A|,|B|)`` lets whole blocks be
  skipped before any intersection is computed;
* the scan stops at the first reference that clears the threshold, which is the
  common case for an in-domain molecule.

The reference fingerprints are the model's own Morgan block, reused rather than
recomputed, so the applicability domain costs no second pass over the molecules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

#: Query rows per block. Keeps the intersection temporary near 65 MB at 4,000 references.
QUERY_BLOCK = 64


def pack_rows(matrix: sparse.csr_matrix, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack a binary sparse matrix into bytes, with each row's popcount.

    Args:
        matrix: Binary matrix whose columns are bit positions.
        n_bits: Width to pack to.

    Returns:
        ``(packed, popcount)`` where ``packed`` is ``uint8`` of shape
        ``(n_rows, ceil(n_bits / 8))``.
    """
    matrix = matrix.tocsr()
    n_rows = matrix.shape[0]
    dense = np.zeros((n_rows, n_bits), dtype=np.uint8)
    for row in range(n_rows):
        block = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
        if block.size:
            dense[row, block] = 1
    packed = np.packbits(dense, axis=1)
    popcount = dense.sum(axis=1).astype(np.int32)
    return packed, popcount


def pack_bits(indices_list: list[np.ndarray], n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack per-row set-bit indices into bytes, with each row's popcount."""
    n_rows = len(indices_list)
    dense = np.zeros((n_rows, n_bits), dtype=np.uint8)
    for row, block in enumerate(indices_list):
        if len(block):
            dense[row, np.asarray(block)] = 1
    return np.packbits(dense, axis=1), dense.sum(axis=1).astype(np.int32)


@dataclass
class Reference:
    """A packed reference set and the threshold that defines its domain.

    Attributes:
        packed: Packed reference fingerprints, sorted by descending popcount.
        popcount: Bits set per reference, descending.
        n_bits: Fingerprint width.
        threshold: Minimum nearest-neighbour similarity to count as in-domain.
        subsampled: Whether the reference set was reduced from the training set.
    """

    packed: np.ndarray
    popcount: np.ndarray
    n_bits: int
    threshold: float
    subsampled: bool = False

    @property
    def n_reference(self) -> int:
        return int(self.packed.shape[0])

    @classmethod
    def build(
        cls,
        matrix: sparse.csr_matrix,
        n_bits: int,
        threshold: float,
        max_reference: int | None = None,
        seed: int = 0,
    ) -> Reference:
        """Pack a training matrix into a reference set, sorted for early exit."""
        matrix = matrix.tocsr()
        subsampled = False
        if max_reference is not None and matrix.shape[0] > max_reference:
            rng = np.random.default_rng(seed)
            pick = np.sort(rng.choice(matrix.shape[0], size=max_reference, replace=False))
            matrix = matrix[pick]
            subsampled = True
        packed, popcount = pack_rows(matrix, n_bits)
        order = np.argsort(-popcount, kind="stable")
        return cls(
            packed=np.ascontiguousarray(packed[order]),
            popcount=popcount[order],
            n_bits=n_bits,
            threshold=threshold,
            subsampled=subsampled,
        )


def _intersections(query_packed: np.ndarray, reference_packed: np.ndarray) -> np.ndarray:
    """Popcount of the bitwise AND between every query and every reference."""
    both = np.bitwise_and(query_packed[:, None, :], reference_packed[None, :, :])
    return np.bitwise_count(both).sum(axis=2, dtype=np.int32)


def max_similarity(
    query_packed: np.ndarray,
    query_popcount: np.ndarray,
    reference: Reference,
    block: int = QUERY_BLOCK,
) -> np.ndarray:
    """Return each query's greatest Tanimoto similarity to the reference set."""
    n_query = query_packed.shape[0]
    out = np.zeros(n_query, dtype=np.float64)
    if reference.n_reference == 0 or n_query == 0:
        return out

    for start in range(0, n_query, block):
        stop = min(start + block, n_query)
        shared = _intersections(query_packed[start:stop], reference.packed)
        union = query_popcount[start:stop, None] + reference.popcount[None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(union > 0, shared / union, 0.0)
        out[start:stop] = similarity.max(axis=1)
    return out


def in_domain(
    query_packed: np.ndarray,
    query_popcount: np.ndarray,
    reference: Reference,
    block: int = QUERY_BLOCK,
) -> np.ndarray:
    """Say whether each query has a reference neighbour at or above the threshold.

    Uses the popcount bound to skip references that cannot clear the threshold
    before computing any intersection. For a query with ``a`` bits set and a
    reference with ``b``, Tanimoto cannot exceed ``min(a,b) / max(a,b)``, and the
    references are popcount-sorted so the survivors form a contiguous band.
    """
    n_query = query_packed.shape[0]
    out = np.zeros(n_query, dtype=bool)
    if reference.n_reference == 0 or n_query == 0:
        return out

    threshold = reference.threshold
    ref_pop = reference.popcount

    for start in range(0, n_query, block):
        stop = min(start + block, n_query)
        q_pop = query_popcount[start:stop]
        # Only references whose popcount ratio can reach the threshold matter.
        lo = np.min(q_pop) * threshold if threshold > 0 else 0
        hi = np.max(q_pop) / threshold if threshold > 0 else np.inf
        candidates = np.flatnonzero((ref_pop >= lo) & (ref_pop <= hi))
        if candidates.size == 0:
            continue
        shared = _intersections(query_packed[start:stop], reference.packed[candidates])
        union = q_pop[:, None] + ref_pop[candidates][None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(union > 0, shared / union, 0.0)
        out[start:stop] = similarity.max(axis=1) >= threshold
    return out


def nearest_neighbour_similarity(matrix: sparse.csr_matrix, n_bits: int) -> np.ndarray:
    """Each row's greatest similarity to any *other* row.

    Self-similarity is excluded, or the distribution would collapse onto 1.0 and
    any threshold derived from it would be meaningless.
    """
    packed, popcount = pack_rows(matrix, n_bits)
    n_rows = packed.shape[0]
    out = np.zeros(n_rows, dtype=np.float64)
    if n_rows < 2:
        return out

    for start in range(0, n_rows, QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, n_rows)
        shared = _intersections(packed[start:stop], packed)
        union = popcount[start:stop, None] + popcount[None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(union > 0, shared / union, 0.0)
        for offset in range(stop - start):
            similarity[offset, start + offset] = -1.0
        out[start:stop] = similarity.max(axis=1)
    return out


__all__ = [
    "QUERY_BLOCK",
    "Reference",
    "in_domain",
    "max_similarity",
    "nearest_neighbour_similarity",
    "pack_bits",
    "pack_rows",
]
