"""The featurizer specification: what a feature vector *is*, and its fingerprint.

A trained model is only meaningful against the exact column space it was trained
on. That space is defined by this spec, so every bundle stores the spec plus a
digest of it, and loading refuses on mismatch rather than silently predicting
from misaligned columns.

Two digests, because they answer different questions:

``spec_sha256``
    Covers the featurizer configuration alone. Two runs with the same spec
    produce the same raw feature vector.
``column_sha256``
    Covers the spec *and* the retained column indices. Two bundles can share a
    spec yet disagree on which columns survived pruning, and that disagreement
    is just as fatal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Topological distance bins for the pharmacophore signature, in bond counts.
#: The last bin is open-ended so distant feature pairs still land somewhere.
DEFAULT_PHARM_BINS: tuple[tuple[int, int], ...] = (
    (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 100),
)

#: Name of the feature definition file shipped with RDKit. Stored as a name, not
#: a path, so a bundle stays portable across installs.
BASE_FEATURES = "BaseFeatures"

#: Feature families that ``BaseFeatures.fdef`` defines. Pinned so that an RDKit
#: release adding or renaming a family is caught instead of silently shifting
#: every bit index.
EXPECTED_FAMILY_COUNT = 8

#: Signature widths for the pharmacophore block, verified against
#: ``SigFactory.GetSigSize()`` in the test suite.
#: 2-point: C(8+1, 2) = 36 family pairs x 9 bins.
#: 3-point: C(8+2, 3) = 120 family triples x 9^3 bin triples.
PHARM_SIG_SIZE = {2: 324, 3: 87_804}

#: Ordered SMILES standardization pipeline. Each name maps to a step in
#: :mod:`admetus.chem.standardize`.
DEFAULT_STANDARDIZE: tuple[str, ...] = ("cleanup", "largest_fragment", "uncharge")


@dataclass(frozen=True)
class FeaturizerSpec:
    """An immutable description of one feature space.

    Attributes:
        feature_set: Preset name, for human and registry use.
        morgan_radius: Morgan fingerprint radius.
        morgan_bits: Morgan fingerprint width.
        pharm_max_points: Maximum pharmacophore point count, or ``None`` to omit
            the pharmacophore block entirely.
        pharm_min_points: Minimum pharmacophore point count.
        pharm_bins: Topological distance bins.
        pharm_fdef: Name of the feature definition set.
        triangle_prune_bins: Whether to drop bin triples that violate the
            triangle inequality. ``False`` keeps the full 3-point space.
        standardize: Ordered standardization step names.
    """

    feature_set: str
    morgan_radius: int
    morgan_bits: int
    pharm_max_points: int | None
    pharm_min_points: int = 2
    pharm_bins: tuple[tuple[int, int], ...] = DEFAULT_PHARM_BINS
    pharm_fdef: str = BASE_FEATURES
    triangle_prune_bins: bool = False
    standardize: tuple[str, ...] = field(default=DEFAULT_STANDARDIZE)

    def __post_init__(self) -> None:
        if self.morgan_bits <= 0:
            raise ValueError(f"morgan_bits must be positive, got {self.morgan_bits}")
        if self.morgan_radius < 0:
            raise ValueError(f"morgan_radius must be non-negative, got {self.morgan_radius}")
        if self.pharm_max_points is not None:
            if self.pharm_max_points not in PHARM_SIG_SIZE:
                raise ValueError(
                    f"pharm_max_points must be one of {sorted(PHARM_SIG_SIZE)}, got {self.pharm_max_points}"
                )
            if self.pharm_min_points != 2:
                raise ValueError(f"pharm_min_points is pinned to 2, got {self.pharm_min_points}")

    @property
    def pharm_size(self) -> int:
        """Width of the pharmacophore block, ``0`` when it is omitted."""
        if self.pharm_max_points is None:
            return 0
        if self.triangle_prune_bins:
            # Pruning depends on the bin layout, so the width is not a constant we
            # can assert. Supporting it would mean querying RDKit here; refuse instead.
            raise NotImplementedError("triangle_prune_bins=True is not a supported configuration")
        return PHARM_SIG_SIZE[self.pharm_max_points]

    @property
    def n_features(self) -> int:
        """Total width of the raw feature vector, before any pruning."""
        return self.morgan_bits + self.pharm_size

    @property
    def has_pharmacophore(self) -> bool:
        """Whether this spec includes a pharmacophore block."""
        return self.pharm_max_points is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping, with tuples normalized to lists."""
        raw = asdict(self)
        raw["pharm_bins"] = [list(b) for b in self.pharm_bins]
        raw["standardize"] = list(self.standardize)
        return raw

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeaturizerSpec:
        """Rebuild a spec from :meth:`to_dict` output."""
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown featurizer spec fields: {sorted(unknown)}")
        data = dict(payload)
        if "pharm_bins" in data and data["pharm_bins"] is not None:
            data["pharm_bins"] = tuple(tuple(b) for b in data["pharm_bins"])
        if "standardize" in data and data["standardize"] is not None:
            data["standardize"] = tuple(data["standardize"])
        return cls(**data)

    def _canonical_json(self) -> str:
        """Serialize deterministically: sorted keys, no incidental whitespace."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def spec_sha256(self) -> str:
        """Digest of the featurizer configuration alone."""
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def column_sha256(self, kept_indices: np.ndarray | None) -> str:
        """Digest of the configuration together with the retained columns.

        Args:
            kept_indices: Sorted retained column indices, or ``None`` when the
                full raw space is used.

        Returns:
            Hex digest covering both the spec and the column selection.
        """
        digest = hashlib.sha256(self._canonical_json().encode("utf-8"))
        if kept_indices is None:
            digest.update(b"|full")
        else:
            idx = np.asarray(kept_indices, dtype=np.int32)
            if idx.ndim != 1:
                raise ValueError("kept_indices must be one-dimensional")
            digest.update(b"|kept:")
            digest.update(idx.tobytes())
        return digest.hexdigest()

    def describe(self) -> str:
        """Return a one-line human summary."""
        if self.has_pharmacophore:
            pharm = f"pharm2d({self.pharm_min_points}-{self.pharm_max_points} point, {self.pharm_size} bits)"
        else:
            pharm = "no pharmacophore"
        return (
            f"{self.feature_set}: morgan(r={self.morgan_radius}, {self.morgan_bits} bits) + "
            f"{pharm} = {self.n_features} features"
        )


#: The three presets the project trains and compares. ``morgan_p23`` is the default.
#: The one feature space, and the reference implementation's: a radius-6 Morgan
#: fingerprint over 2,048 bits concatenated with the 2- and 3-point pharmacophore
#: signature, 89,852 columns in all.
#:
#: There used to be three presets and a rule that picked between them by dataset
#: size, with three measured per-endpoint overrides on top. None of that was asked
#: for: the reference featurizes every dataset the same way, including the
#: 280-compound one its own notebook runs. The rule and the overrides are gone, and
#: with them the study that chose them -- whose scores were measured on a split that
#: no longer exists anyway.
DEFAULT_FEATURE_SET = "morgan_p23"

PRESETS: dict[str, FeaturizerSpec] = {
    DEFAULT_FEATURE_SET: FeaturizerSpec(
        feature_set=DEFAULT_FEATURE_SET, morgan_radius=6, morgan_bits=2048, pharm_max_points=3
    ),
}


def get_spec(feature_set: str = DEFAULT_FEATURE_SET) -> FeaturizerSpec:
    """Return the feature space.

    Takes a name so a bundle written by an older version can still say which space
    it used and be refused by name rather than by a confusing column mismatch.

    Raises:
        ValueError: The name is not the feature space this install builds.
    """
    try:
        return PRESETS[feature_set]
    except KeyError:
        raise ValueError(
            f"this install builds one feature space, {DEFAULT_FEATURE_SET!r}, and was asked for "
            f"{feature_set!r}. A bundle trained on another space cannot be used here."
        ) from None


__all__ = [
    "BASE_FEATURES",
    "DEFAULT_FEATURE_SET",
    "DEFAULT_PHARM_BINS",
    "DEFAULT_STANDARDIZE",
    "EXPECTED_FAMILY_COUNT",
    "PHARM_SIG_SIZE",
    "PRESETS",
    "FeaturizerSpec",
    "get_spec",
]
