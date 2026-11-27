"""The feature space contract.

These numbers are the contract between a trained model and the featurizer. If
RDKit changes its feature definitions or bin handling, every existing model
becomes invalid, so the widths are pinned here deliberately rather than derived.
"""

from __future__ import annotations

import numpy as np
import pytest

from admetus.chem.spec import (
    EXPECTED_FAMILY_COUNT,
    PHARM_SIG_SIZE,
    PRESETS,
    FeaturizerSpec,
    get_spec,
)


def test_preset_widths_are_the_documented_ones():
    # One feature space, the reference implementation's: 2,048 Morgan bits plus the
    # 2- and 3-point pharmacophore signature.
    assert get_spec().n_features == 89_852
    assert get_spec().morgan_bits == 2048
    assert get_spec().morgan_radius == 6
    assert get_spec().pharm_max_points == 3
    # The block widths stay pinned, because an RDKit change that moved them would
    # otherwise shift every bit index silently.
    assert PHARM_SIG_SIZE == {2: 324, 3: 87_804}


def test_pharmacophore_block_widths():
    # 2-point: C(8+1, 2) = 36 family pairs x 9 bins.
    assert PHARM_SIG_SIZE[2] == 36 * 9 == 324
    # 3-point: 2-point plus C(8+2, 3) = 120 family triples x 9^3 bin triples.
    assert PHARM_SIG_SIZE[3] == 324 + 120 * 9**3 == 87_804


def test_sig_size_matches_rdkit():
    """The declared width must equal what RDKit actually builds."""
    from admetus.chem.pharm2d import get_sig_factory

    for name in ("morgan_p23",):
        spec = get_spec(name)
        assert get_sig_factory(spec).GetSigSize() == spec.pharm_size


def test_feature_family_count_is_pinned():
    """Bit indices depend on the family set, so a change must be caught."""
    from admetus.chem.pharm2d import get_feature_factory

    assert len(get_feature_factory().GetFeatureFamilies()) == EXPECTED_FAMILY_COUNT


def test_unknown_preset_is_refused():
    with pytest.raises(ValueError, match="one feature space"):
        get_spec("no_such_set")


def test_round_trip_through_dict():
    for spec in PRESETS.values():
        assert FeaturizerSpec.from_dict(spec.to_dict()) == spec


def test_unknown_field_is_refused():
    payload = get_spec().to_dict()
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown featurizer spec fields"):
        FeaturizerSpec.from_dict(payload)


def test_spec_digest_is_stable_and_distinct():
    spec = get_spec()
    assert spec.spec_sha256 == FeaturizerSpec.from_dict(spec.to_dict()).spec_sha256
    digests = {s.spec_sha256 for s in PRESETS.values()}
    assert len(digests) == len(PRESETS)


def test_column_digest_tracks_the_retained_columns():
    """A bundle can share a spec yet disagree on columns; that must be detectable."""
    spec = get_spec()
    full = spec.column_sha256(None)
    a = spec.column_sha256(np.array([1, 2, 3], dtype=np.int32))
    b = spec.column_sha256(np.array([1, 2, 4], dtype=np.int32))
    assert len({full, a, b}) == 3
    # dtype must not change the answer
    assert a == spec.column_sha256(np.array([1, 2, 3], dtype=np.int64))


def test_invalid_specs_are_refused():
    with pytest.raises(ValueError, match="morgan_bits"):
        FeaturizerSpec(feature_set="x", morgan_radius=2, morgan_bits=0, pharm_max_points=None)
    with pytest.raises(ValueError, match="morgan_radius"):
        FeaturizerSpec(feature_set="x", morgan_radius=-1, morgan_bits=2048, pharm_max_points=None)
    with pytest.raises(ValueError, match="pharm_max_points"):
        FeaturizerSpec(feature_set="x", morgan_radius=2, morgan_bits=2048, pharm_max_points=4)


def test_triangle_pruning_is_refused_rather_than_guessed():
    spec = FeaturizerSpec(
        feature_set="x", morgan_radius=6, morgan_bits=2048, pharm_max_points=3, triangle_prune_bins=True
    )
    with pytest.raises(NotImplementedError):
        _ = spec.pharm_size
