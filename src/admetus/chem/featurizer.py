"""Assembling feature vectors.

A molecule's feature vector is the Morgan block followed by the pharmacophore
block, so pharmacophore bit ``j`` lives at column ``morgan_bits + j``. Vectors
are produced as sorted ``int32`` column indices rather than dense arrays:

* the vector is binary, so positions carry all the information;
* dense ``float32`` would be 359,408 bytes per molecule at the full width, which
  is 4.67 GB for 13,000 molecules;
* indices are also exactly what the sparse matrix and the worker pipe want.

An optional column map projects the raw space onto a pruned basis as the bits are
produced, so matrices arrive already narrowed and no slicing is needed later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from scipy import sparse

from admetus.chem.morgan import morgan_on_bits
from admetus.chem.pharm2d import pharm2d_on_bits, warm_up
from admetus.chem.spec import FeaturizerSpec
from admetus.chem.standardize import standardize


@dataclass(frozen=True)
class Failure:
    """One molecule that could not be featurized.

    Attributes:
        position: Index into the input sequence.
        smiles: The input string, kept verbatim for the report.
        reason: Why it failed.
    """

    position: int
    smiles: str
    reason: str


@dataclass(frozen=True)
class FeaturizeResult:
    """Feature matrix for the molecules that succeeded.

    Attributes:
        matrix: CSR ``float32`` matrix with one row per successful molecule.
        positions: Input positions of those rows, ascending.
        canonical: Standardized canonical SMILES, parallel to ``positions``.
        changed: Whether standardization altered each molecule.
        failures: Molecules that produced no row.
    """

    matrix: sparse.csr_matrix
    positions: np.ndarray
    canonical: list[str]
    changed: np.ndarray
    failures: list[Failure]

    @property
    def n_ok(self) -> int:
        """Number of molecules that produced a row."""
        return int(self.positions.size)


def on_bits(mol: Chem.Mol, spec: FeaturizerSpec) -> np.ndarray:
    """Return the sorted set-bit column indices for one parsed molecule."""
    blocks = [morgan_on_bits(mol, spec.morgan_radius, spec.morgan_bits)]
    if spec.has_pharmacophore:
        blocks.append(pharm2d_on_bits(mol, spec) + spec.morgan_bits)
    idx = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    idx.sort()
    return idx


def project(idx: np.ndarray, col_map: np.ndarray | None) -> np.ndarray:
    """Project raw column indices onto a pruned basis.

    Args:
        idx: Sorted raw column indices.
        col_map: ``int32`` array of length ``spec.n_features`` mapping each raw
            column to its position in the pruned basis, with ``-1`` for columns
            that were dropped. ``None`` leaves the indices untouched.

    Returns:
        Sorted indices in the pruned basis.
    """
    if col_map is None:
        return idx
    mapped = col_map[idx]
    return mapped[mapped >= 0]


def featurize_one(
    smiles: str, spec: FeaturizerSpec, col_map: np.ndarray | None = None
) -> tuple[np.ndarray | None, str | None, str | None, bool]:
    """Standardize and featurize one SMILES string.

    Returns:
        A tuple of ``(indices, canonical, reason, changed)``. On failure
        ``indices`` and ``canonical`` are ``None`` and ``reason`` explains why;
        no substitute molecule is ever used.
    """
    result = standardize(smiles, spec.standardize)
    if not result.ok:
        return None, None, result.reason, False
    try:
        idx = on_bits(result.mol, spec)
    except Exception as exc:
        return None, None, f"featurization failed: {exc}", result.changed
    return project(idx, col_map), result.canonical, None, result.changed


def build_csr(index_lists: list[np.ndarray], n_cols: int) -> sparse.csr_matrix:
    """Assemble a binary CSR matrix from per-row column indices.

    The data array is all ones by construction, so it is created here rather than
    carried around or sent between processes.
    """
    if not index_lists:
        return sparse.csr_matrix((0, n_cols), dtype=np.float32)
    counts = np.fromiter((len(i) for i in index_lists), dtype=np.int64, count=len(index_lists))
    indptr = np.zeros(len(index_lists) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = (
        np.concatenate(index_lists).astype(np.int32, copy=False)
        if indptr[-1]
        else np.empty(0, dtype=np.int32)
    )
    data = np.ones(indptr[-1], dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(index_lists), n_cols))


def featurize_many(
    smiles: list[str],
    spec: FeaturizerSpec,
    col_map: np.ndarray | None = None,
    n_cols: int | None = None,
) -> FeaturizeResult:
    """Featurize a sequence of SMILES strings in order.

    Args:
        smiles: Input strings.
        spec: The feature space to build.
        col_map: Optional projection onto a pruned basis.
        n_cols: Width of the output matrix. Required when ``col_map`` is given,
            since the pruned width cannot be inferred from the map alone.

    Returns:
        A :class:`FeaturizeResult` holding one row per molecule that succeeded.

    Raises:
        ValueError: ``col_map`` was given without ``n_cols``.
    """
    if col_map is not None and n_cols is None:
        raise ValueError("n_cols is required when col_map is given")
    width = spec.n_features if col_map is None else int(n_cols)

    warm_up(spec)

    rows: list[np.ndarray] = []
    positions: list[int] = []
    canonical: list[str] = []
    changed: list[bool] = []
    failures: list[Failure] = []

    for position, text in enumerate(smiles):
        idx, canon, reason, was_changed = featurize_one(text, spec, col_map)
        if idx is None:
            failures.append(Failure(position, text if text is not None else "", reason or "unknown"))
            continue
        rows.append(idx)
        positions.append(position)
        canonical.append(canon)
        changed.append(was_changed)

    return FeaturizeResult(
        matrix=build_csr(rows, width),
        positions=np.asarray(positions, dtype=np.int64),
        canonical=canonical,
        changed=np.asarray(changed, dtype=bool),
        failures=failures,
    )


__all__ = [
    "FeaturizeResult",
    "Failure",
    "build_csr",
    "featurize_many",
    "featurize_one",
    "on_bits",
    "project",
]
