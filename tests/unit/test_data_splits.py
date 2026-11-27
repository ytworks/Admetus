"""Scaffold grouping and cross-validation folds."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from admetus.data.splits import (
    cv_folds,
    fold_report,
    murcko_framework,
    murcko_scaffold,
    scaffold_groups,
    scaffold_split,
)
from admetus.errors import InputDataError


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"),            # aspirin -> benzene
        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1ccccc1"),       # ibuprofen -> benzene
        ("CCO", ""),                                      # acyclic
        ("CC(=O)O", ""),
        ("C1CCCCC1", "C1CCCCC1"),
        ("O=C1CCCCC1", "O=C1CCCCC1"),                     # exocyclic double bond kept
        ("c1ccc(Cc2ccccc2)cc1", "c1ccc(Cc2ccccc2)cc1"),   # linker kept
        ("CCc1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccccc2)cc1"), # side chain stripped
    ],
)
def test_framework_of_known_molecules(smiles, expected):
    assert murcko_scaffold(smiles) == expected


def test_aromatic_ring_with_substituted_nitrogens_survives():
    """Caffeine's N-methyls are stripped; the purine-dione core must still come out."""
    scaffold = murcko_scaffold("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert scaffold
    assert "=" in scaffold  # the exocyclic carbonyls are retained


def test_analogues_share_a_scaffold_and_different_cores_do_not():
    assert murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") == murcko_scaffold("CCOc1ccccc1C(=O)O")
    assert murcko_scaffold("c1ccccc1C(=O)O") != murcko_scaffold("c1ccncc1C(=O)O")


def test_unparseable_smiles_has_no_scaffold():
    assert murcko_scaffold("not_a_molecule") == ""


def test_framework_of_a_framework_is_itself():
    for smiles in ("CC(=O)Oc1ccccc1C(=O)O", "c1ccc(Cc2ccccc2)cc1", "O=C1CCCCC1"):
        once = murcko_scaffold(smiles)
        assert murcko_framework(Chem.MolFromSmiles(once)) == once


def test_acyclic_molecules_get_their_own_group():
    """The empty scaffold must not pool every acyclic compound into one group."""
    groups = scaffold_groups(["CCO", "CCC", "CCN", "c1ccccc1C", "c1ccccc1CC"])
    assert len(set(groups[:3])) == 3           # three distinct acyclic groups
    assert groups[3] == groups[4]              # two benzenes share one group


def test_scaffold_split_keeps_groups_whole():
    smiles = (
        ["c1ccccc1C" + "C" * i for i in range(30)]
        + ["c1ccncc1C" + "C" * i for i in range(30)]
        + ["CCO" + "C" * i for i in range(20)]
    )
    split = scaffold_split(smiles, test_fraction=0.2)
    groups = scaffold_groups(smiles)
    assert not set(groups[split.train]) & set(groups[split.test])
    assert split.train.size + split.test.size == len(smiles)


def test_scaffold_split_refuses_an_impossible_fraction():
    with pytest.raises(InputDataError, match="test_fraction"):
        scaffold_split(["CCO"], test_fraction=0.0)


def test_scaffold_split_refuses_when_there_is_nothing_to_hold_out():
    with pytest.raises(InputDataError, match="too few distinct scaffolds"):
        scaffold_split(["c1ccccc1C"] * 20, test_fraction=0.2)


def _mixed_library(n_per_core: int = 40) -> tuple[list[str], np.ndarray]:
    cores = ["c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CCNCC1"]
    smiles: list[str] = []
    for core in cores:
        for i in range(n_per_core):
            smiles.append(core + "C" * (i % 5 + 1))
    y = np.array([1 if i % 4 == 0 else 0 for i in range(len(smiles))], dtype=float)
    return smiles, y


def test_folds_never_split_a_scaffold_group():
    smiles, y = _mixed_library()
    groups = scaffold_groups(smiles)
    for fold in cv_folds(smiles, y, "classification", n_splits=3, seed=0):
        assert not set(groups[fold.train]) & set(groups[fold.test])


def test_folds_cover_every_row_exactly_once_as_validation():
    smiles, y = _mixed_library()
    folds = cv_folds(smiles, y, "classification", n_splits=3, seed=0)
    seen = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(seen), np.arange(len(smiles)))


def test_folds_are_reproducible_for_a_seed():
    smiles, y = _mixed_library()
    a = cv_folds(smiles, y, "classification", n_splits=3, seed=7)
    b = cv_folds(smiles, y, "classification", n_splits=3, seed=7)
    assert [f.test.tolist() for f in a] == [f.test.tolist() for f in b]


def test_a_dataset_too_thin_to_score_is_refused_not_fudged():
    """Average precision on three positives is noise; refuse rather than tune on it."""
    smiles = ["c1ccccc1" + "C" * i for i in range(40)]
    y = np.zeros(40)
    y[:3] = 1
    with pytest.raises(InputDataError, match="positive"):
        cv_folds(smiles, y, "classification", n_splits=5, seed=0)


def test_fold_count_is_reduced_rather_than_abandoned():
    smiles, y = _mixed_library(n_per_core=12)
    folds = cv_folds(smiles, y, "classification", n_splits=5, seed=0, min_fold_positives=4)
    assert 3 <= len(folds) <= 5


def test_regression_folds_are_stratified_by_value():
    smiles, _ = _mixed_library()
    y = np.linspace(0, 100, len(smiles))
    folds = cv_folds(smiles, y, "regression", n_splits=3, seed=0)
    medians = [float(np.median(y[f.test])) for f in folds]
    assert max(medians) - min(medians) < 25


def test_fold_report_describes_each_fold():
    smiles, y = _mixed_library()
    folds = cv_folds(smiles, y, "classification", n_splits=3, seed=0)
    report = fold_report(folds, y, "classification")
    assert len(report) == len(folds)
    assert all(r["n_valid_positive"] > 0 for r in report)


def test_scaffold_grouping_must_see_the_standardized_structure():
    """A salt and its free base must not land on opposite sides of the split.

    The trainer used to group the input strings. A counterion that carries its own
    rings then shows up in the scaffold key, so the salt groups apart from the free
    base even though standardization collapses them to one molecule and the model
    cannot tell them apart. On the shipped hERG split that put 9 of 129 held-out
    molecules in a scaffold the training set also held.
    """
    from admetus.chem.featurizer import featurize_many
    from admetus.chem.spec import get_spec

    free_base = "CCCCN1CCCC1"
    tosylate_salt = "CCCCN1CCCC1.c1ccc(S(=O)(=O)O)cc1"
    raw = [free_base, tosylate_salt]

    canonical = list(featurize_many(raw, get_spec("morgan")).canonical)
    # Standardization reduces the salt to its largest fragment, so both rows are
    # the same molecule by the time the model sees them.
    assert canonical[0] == canonical[1]

    from_raw = list(scaffold_groups(raw))
    from_canonical = list(scaffold_groups(canonical))

    # The counterion's ring leaks into the key when the input string is grouped.
    assert from_raw[0] != from_raw[1]
    assert "c1ccccc1" in from_raw[1]
    # Grouping what the model sees keeps the pair together.
    assert from_canonical[0] == from_canonical[1]
