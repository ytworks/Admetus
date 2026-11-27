"""Standardization behaviour, including the rule that nothing is ever substituted."""

from __future__ import annotations

import pytest

from admetus.chem.standardize import standardize, validate_steps
from admetus.errors import UsageError


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CCO", "CCO"),
        ("OCC", "CCO"),                        # same molecule, written differently
        ("CC(=O)[O-].[Na+]", "CC(=O)O"),       # salt stripped and neutralized
        ("Cc1ccccc1C(=O)[O-].[K+]", "Cc1ccccc1C(=O)O"),
        ("C[NH3+].[Cl-]", "CN"),
        ("CCO.O", "CCO"),                      # hydrate
    ],
)
def test_standardization_targets(smiles, expected):
    result = standardize(smiles)
    assert result.ok
    assert result.canonical == expected


def test_different_spellings_converge():
    assert standardize("OCC").canonical == standardize("CCO").canonical


def test_salt_and_free_acid_converge():
    assert standardize("CC(=O)[O-].[Na+]").canonical == standardize("CC(=O)O").canonical


@pytest.mark.parametrize(
    "smiles",
    ["not_a_molecule", "", "   ", "c1ccccc", "C(C", None],
)
def test_failures_are_reported_never_substituted(smiles):
    """A failure must yield no molecule and a reason, never a stand-in."""
    result = standardize(smiles)
    assert not result.ok
    assert result.mol is None
    assert result.canonical is None
    assert result.reason


def test_standardized_molecule_is_usable_for_feature_matching():
    """FragmentParent can leave ring info uninitialized; the molecule must still work."""
    from admetus.chem.pharm2d import pharm2d_on_bits
    from admetus.chem.spec import get_spec

    result = standardize("Cc1ccccc1C(=O)[O-].[K+]")
    assert result.ok
    bits = pharm2d_on_bits(result.mol, get_spec("morgan_p23"))
    assert bits.size > 0


def test_changed_flag_reports_whether_anything_happened():
    assert standardize("CCO").changed is False
    assert standardize("CC(=O)[O-].[Na+]").changed is True


def test_pipeline_can_be_narrowed():
    """Datasets whose label describes the salt skip largest_fragment."""
    keep_salt = standardize("CC(=O)[O-].[Na+]", steps=("cleanup",))
    assert keep_salt.ok
    assert "Na" in keep_salt.canonical


def test_unknown_step_is_refused():
    with pytest.raises(UsageError, match="unknown standardization step"):
        validate_steps(("cleanup", "no_such_step"))
    with pytest.raises(UsageError):
        standardize("CCO", steps=("no_such_step",))
