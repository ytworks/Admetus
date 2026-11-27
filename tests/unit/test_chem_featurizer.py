"""Feature assembly: widths, determinism, projection and failure handling."""

from __future__ import annotations

import hashlib
import multiprocessing

import numpy as np
import pytest

from admetus.chem.featurizer import build_csr, featurize_many, featurize_one, on_bits, project
from admetus.chem.spec import PRESETS, get_spec

#: Three fixed structures whose fingerprints pin the featurizer's output.
GOLDEN_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
]

#: Digest of the canonical SMILES and on-bit indices of :data:`GOLDEN_SMILES`.
#: A change here means the feature space moved and every trained model is stale.
GOLDEN_DIGESTS = {
    "morgan": "cac9aecac7f95504aa8a1400956aa6e4",
    "morgan_p2": "293670163f269ceb8a7fc9bf38ce370e",
    "morgan_p23": "286761bcfa26aadb11fd2b4b81970e42",
}


def golden_digest(feature_set: str) -> str:
    """Hash the canonical SMILES and on-bits of the golden molecules."""
    spec = PRESETS[feature_set]
    digest = hashlib.blake2b(digest_size=16)
    for smiles in GOLDEN_SMILES:
        idx, canonical, reason, _ = featurize_one(smiles, spec)
        assert idx is not None, reason
        digest.update(canonical.encode())
        digest.update(idx.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("feature_set", sorted(GOLDEN_DIGESTS))
def test_fingerprints_match_the_golden_digest(feature_set):
    assert golden_digest(feature_set) == GOLDEN_DIGESTS[feature_set]


@pytest.mark.parametrize(
    ("feature_set", "width"),
    [("morgan", 2048), ("morgan_p2", 2372), ("morgan_p23", 89_852)],
)
def test_matrix_width_is_the_declared_one(feature_set, width):
    result = featurize_many(["CCO"], get_spec(feature_set))
    assert result.matrix.shape == (1, width)


def test_blocks_are_laid_out_morgan_then_pharmacophore():
    """Pharmacophore bit j must live at column morgan_bits + j."""
    from rdkit import Chem

    from admetus.chem.morgan import morgan_on_bits
    from admetus.chem.pharm2d import pharm2d_on_bits

    spec = get_spec("morgan_p23")
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    combined = on_bits(mol, spec)
    morgan = morgan_on_bits(mol, spec.morgan_radius, spec.morgan_bits)
    pharm = pharm2d_on_bits(mol, spec)

    assert np.array_equal(combined[combined < spec.morgan_bits], np.sort(morgan))
    assert np.array_equal(combined[combined >= spec.morgan_bits] - spec.morgan_bits, np.sort(pharm))


def test_indices_are_sorted_and_int32():
    result = featurize_many(GOLDEN_SMILES, get_spec("morgan_p23"))
    assert result.matrix.indices.dtype == np.int32
    for row in range(result.n_ok):
        start, end = result.matrix.indptr[row], result.matrix.indptr[row + 1]
        assert np.all(np.diff(result.matrix.indices[start:end]) > 0)


def test_matrix_is_binary_float32():
    result = featurize_many(GOLDEN_SMILES, get_spec("morgan_p23"))
    assert result.matrix.dtype == np.float32
    assert np.all(result.matrix.data == 1.0)


def test_repeated_calls_agree():
    spec = get_spec("morgan_p23")
    first = featurize_many(GOLDEN_SMILES, spec)
    second = featurize_many(GOLDEN_SMILES, spec)
    assert np.array_equal(first.matrix.indices, second.matrix.indices)
    assert np.array_equal(first.matrix.indptr, second.matrix.indptr)


def _digest_in_worker(feature_set: str) -> str:
    """Run :func:`golden_digest` in a worker process."""
    return golden_digest(feature_set)


def test_fingerprints_agree_across_spawned_processes():
    """Inference runs in a spawn pool, so bits must not depend on the process."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        got = pool.map(_digest_in_worker, ["morgan_p23", "morgan_p2"])
    assert got == [GOLDEN_DIGESTS["morgan_p23"], GOLDEN_DIGESTS["morgan_p2"]]


def test_failures_are_reported_in_order_and_never_substituted():
    spec = get_spec("morgan_p23")
    smiles = ["CCO", "not_a_molecule", "CC(=O)O", "", "c1ccccc"]
    result = featurize_many(smiles, spec)

    assert result.positions.tolist() == [0, 2]
    assert result.canonical == ["CCO", "CC(=O)O"]
    assert [f.position for f in result.failures] == [1, 3, 4]
    assert all(f.reason for f in result.failures)
    # The input text is preserved verbatim for the report.
    assert result.failures[0].smiles == "not_a_molecule"
    # A failed molecule contributes no row at all.
    assert result.matrix.shape[0] == 2


def test_all_failures_give_an_empty_matrix_of_the_right_width():
    result = featurize_many(["bad", "worse"], get_spec("morgan_p23"))
    assert result.matrix.shape == (0, 89_852)
    assert result.n_ok == 0
    assert len(result.failures) == 2


def test_empty_input():
    result = featurize_many([], get_spec("morgan_p23"))
    assert result.matrix.shape == (0, 89_852)
    assert result.n_ok == 0


def test_projection_drops_unmapped_columns_and_renumbers():
    spec = get_spec("morgan_p23")
    raw = featurize_many(GOLDEN_SMILES, spec)
    keep = np.unique(raw.matrix.indices)[:40]
    col_map = np.full(spec.n_features, -1, dtype=np.int32)
    col_map[keep] = np.arange(keep.size, dtype=np.int32)

    projected = featurize_many(GOLDEN_SMILES, spec, col_map=col_map, n_cols=keep.size)
    assert projected.matrix.shape == (3, keep.size)
    assert projected.matrix.indices.max() < keep.size
    # Projecting then counting must equal selecting the kept columns directly.
    assert projected.matrix.nnz == raw.matrix[:, keep].nnz


def test_projection_of_an_unmapped_index_yields_nothing():
    col_map = np.full(10, -1, dtype=np.int32)
    assert project(np.array([3, 7], dtype=np.int32), col_map).size == 0


def test_projection_without_width_is_refused():
    spec = get_spec("morgan")
    col_map = np.zeros(spec.n_features, dtype=np.int32)
    with pytest.raises(ValueError, match="n_cols is required"):
        featurize_many(["CCO"], spec, col_map=col_map)


def test_build_csr_on_empty_rows():
    matrix = build_csr([np.empty(0, dtype=np.int32)], 100)
    assert matrix.shape == (1, 100)
    assert matrix.nnz == 0
