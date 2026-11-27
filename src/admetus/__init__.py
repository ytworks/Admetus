"""Admetus: ADMET property prediction from SMILES.

Models are LightGBM fold ensembles over a Morgan fingerprint concatenated with a
2D pharmacophore signature, trained one model per public dataset.

The public surface is the ``admetus`` command-line interface; this package is
importable but its Python API is not yet stable.
"""

from __future__ import annotations

from admetus.version import SCHEMA_VERSION, get_version

__version__ = get_version()

__all__ = ["SCHEMA_VERSION", "__version__", "get_version"]
