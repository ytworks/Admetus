"""Morgan (ECFP-style) circular fingerprints.

Cheap enough to be irrelevant to runtime: measured at 7-50 microseconds per
molecule against 0.4-130 milliseconds for the pharmacophore block.

Uses ``rdFingerprintGenerator``; the older ``AllChem.GetMorganFingerprintAsBitVect``
is deprecated from RDKit 2024.09.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@lru_cache(maxsize=8)
def get_generator(radius: int, n_bits: int):
    """Return a cached Morgan generator.

    Generators are stateless and reusable; building one per molecule would be
    pure waste. Cached per process, including inside each worker.
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_on_bits(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    """Return the indices of set bits, as ``int32``.

    Indices, not a dense array: the vector is binary and sparse, so only the
    positions carry information and they are what the sparse matrix and the
    inter-process pipe both want.
    """
    fp = get_generator(radius, n_bits).GetFingerprint(mol)
    return np.fromiter(fp.GetOnBits(), dtype=np.int32, count=fp.GetNumOnBits())


def morgan_bitvect(mol: Chem.Mol, radius: int, n_bits: int):
    """Return the raw RDKit bit vector, for similarity calculations."""
    return get_generator(radius, n_bits).GetFingerprint(mol)


__all__ = ["get_generator", "morgan_bitvect", "morgan_on_bits"]
