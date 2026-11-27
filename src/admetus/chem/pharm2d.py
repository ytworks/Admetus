"""2D pharmacophore signature fingerprints.

The signature enumerates sets of 2 or 3 pharmacophore features and the
topological distance bin of each pair, so a bit means "a donor and an acceptor
4-5 bonds apart, with an aromatic 6-7 bonds from both". With RDKit's
``BaseFeatures.fdef`` (8 families) and 9 distance bins the widths are:

* 2-point: C(9, 2) = 36 family pairs x 9 bins = **324**
* 3-point: C(10, 3) = 120 family triples x 9^3 bin triples = **87,480**
* together: **87,804**

Two things matter for performance, both measured:

1. **The factory is built once per process, never per molecule.** Constructing it
   and calling ``Init()`` builds the scaffolding for all 87,804 bits.
2. **Cost grows steeply with molecule size**, because the 3-point term is cubic
   in the feature count: about 0.4 ms at 11 heavy atoms, 2 ms at 21, 8 ms at 31,
   21 ms at 41 and 133 ms at 71. The 2-point-only signature stays between 0.2 and
   4.8 ms across the same range, which is why ``morgan_p2`` is both narrower and
   far cheaper.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

from admetus.chem.spec import BASE_FEATURES, EXPECTED_FAMILY_COUNT, FeaturizerSpec

RDLogger.DisableLog("rdApp.*")


def fdef_path(name: str = BASE_FEATURES) -> str:
    """Resolve a feature definition set name to the file RDKit ships.

    Raises:
        ValueError: The name is not a feature definition set we know about.
    """
    if name != BASE_FEATURES:
        raise ValueError(f"unknown feature definition set {name!r}; only {BASE_FEATURES!r} is supported")
    return os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")


@lru_cache(maxsize=2)
def get_feature_factory(name: str = BASE_FEATURES):
    """Return the cached chemical feature factory.

    Raises:
        RuntimeError: RDKit's feature definitions no longer declare the expected
            number of families, which would shift every bit index.
    """
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path(name))
    families = factory.GetFeatureFamilies()
    if len(families) != EXPECTED_FAMILY_COUNT:
        raise RuntimeError(
            f"{name}.fdef declares {len(families)} feature families, expected {EXPECTED_FAMILY_COUNT}. "
            "Bit indices depend on the family set, so existing models would be invalid. "
            f"Families found: {families}"
        )
    return factory


@lru_cache(maxsize=4)
def _build_sig_factory(
    fdef: str, min_points: int, max_points: int, bins: tuple[tuple[int, int], ...], triangle_prune: bool
) -> SigFactory:
    """Build and initialize a signature factory. Cached: ``Init()`` is expensive."""
    factory = SigFactory(
        get_feature_factory(fdef),
        minPointCount=min_points,
        maxPointCount=max_points,
        trianglePruneBins=triangle_prune,
    )
    factory.SetBins(list(bins))
    factory.Init()
    return factory


def get_sig_factory(spec: FeaturizerSpec) -> SigFactory:
    """Return the cached signature factory for a spec.

    Raises:
        ValueError: The spec has no pharmacophore block.
        RuntimeError: The factory's width disagrees with the spec, which means a
            model trained against this spec would read the wrong columns.
    """
    if not spec.has_pharmacophore:
        raise ValueError(f"feature set {spec.feature_set!r} has no pharmacophore block")
    factory = _build_sig_factory(
        spec.pharm_fdef,
        spec.pharm_min_points,
        spec.pharm_max_points,
        spec.pharm_bins,
        spec.triangle_prune_bins,
    )
    actual = factory.GetSigSize()
    if actual != spec.pharm_size:
        raise RuntimeError(
            f"pharmacophore signature is {actual} bits but the spec declares {spec.pharm_size}. "
            "RDKit's feature definitions or bin handling changed; existing models are invalid."
        )
    return factory


def pharm2d_on_bits(mol: Chem.Mol, spec: FeaturizerSpec) -> np.ndarray:
    """Return the indices of set pharmacophore bits, as ``int32``."""
    factory = get_sig_factory(spec)
    fp = Generate.Gen2DFingerprint(mol, factory)
    return np.fromiter(fp.GetOnBits(), dtype=np.int32, count=fp.GetNumOnBits())


def warm_up(spec: FeaturizerSpec) -> None:
    """Build every cached factory this spec needs.

    Called from the worker initializer so the cost is paid once per process
    rather than on the first molecule of each batch.
    """
    get_feature_factory(spec.pharm_fdef)
    if spec.has_pharmacophore:
        get_sig_factory(spec)


__all__ = [
    "fdef_path",
    "get_feature_factory",
    "get_sig_factory",
    "pharm2d_on_bits",
    "warm_up",
]
