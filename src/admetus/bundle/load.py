"""Reading a trained model back, and refusing one that does not match.

A model is only meaningful against the column space it was trained on, so loading
verifies three things before anything is predicted:

* the featurizer spec in the bundle matches a spec this install can build;
* the retained columns hash to what the bundle recorded;
* the three golden molecules fingerprint to the recorded digest, which catches an
  RDKit change that moved bit positions.

The first two are hard failures. The RDKit version alone is only a warning, since
a version bump that leaves the fingerprints identical is harmless and the golden
digest is the real test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from admetus._compat import load_lightgbm
from admetus.bundle.save import (
    BOOSTER_PATTERN,
    BUNDLE_FILE,
    COLUMNS_FILE,
    REFERENCE_FILE,
    golden_fingerprint,
)
from admetus.chem.similarity import Reference
from admetus.chem.spec import FeaturizerSpec
from admetus.errors import ModelError, SpecMismatchError
from admetus.logging import get_logger
from admetus.train import calibrate, conformal, transforms
from admetus.train.prune import ColumnBasis

log = get_logger("bundle.load")


@dataclass
class LoadedModel:
    """A model ready to predict.

    Attributes:
        key: Endpoint key, which is also the base output column name.
        task: ``classification`` or ``regression``.
        spec: The feature space to build.
        basis: The retained columns.
        boosters: The fold ensemble.
        transform: Label transform, for mapping output back to the published scale.
        calibrator: Probability calibration, classification only.
        interval: Conformal width, regression only.
        reference: Applicability-domain reference set.
        unit: Unit of the published label.
        output_range: Range predictions are clamped to.
        decision_threshold: Recommended operating point, classification only.
        metadata: The whole bundle record.
    """

    key: str
    task: str
    spec: FeaturizerSpec
    basis: ColumnBasis
    boosters: list[Any]
    transform: transforms.Transform
    calibrator: calibrate.Calibrator | None
    interval: conformal.ConformalInterval | None
    reference: Reference | None
    unit: str
    output_range: tuple[float | None, float | None] | None
    decision_threshold: float | None
    metadata: dict[str, Any]

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def n_boosters(self) -> int:
        """Taken from what was loaded, never from a stored count."""
        return len(self.boosters)


def read_metadata(directory: Path) -> dict[str, Any]:
    """Read a bundle's record without loading the models.

    Raises:
        ModelError: The bundle is missing or its record is unreadable.
    """
    path = Path(directory) / BUNDLE_FILE
    if not path.is_file():
        raise ModelError(
            f"no model bundle at {directory}. Run 'admetus models pull' or "
            f"'admetus train --endpoint {Path(directory).name}'."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} is not readable JSON: {exc}") from exc


def verify(directory: Path, check_fingerprint: bool = True) -> tuple[bool, list[str]]:
    """Check a bundle without loading it for prediction.

    Returns:
        ``(ok, problems)``.
    """
    problems: list[str] = []
    directory = Path(directory)
    try:
        meta = read_metadata(directory)
    except ModelError as exc:
        return False, [str(exc)]

    featurizer = meta.get("featurizer") or {}
    try:
        spec = FeaturizerSpec.from_dict(featurizer["spec"])
    except (KeyError, ValueError) as exc:
        return False, [f"featurizer spec is unusable: {exc}"]

    if spec.spec_sha256 != featurizer.get("spec_sha256"):
        problems.append("featurizer spec digest does not match the spec recorded beside it")

    columns = directory / COLUMNS_FILE
    if not columns.is_file():
        problems.append(f"{COLUMNS_FILE} is missing")
    else:
        kept = np.load(columns)["kept_indices"]
        if spec.column_sha256(kept) != featurizer.get("column_sha256"):
            problems.append("retained columns do not match the digest recorded for them")
        if int(kept.size) != int(featurizer.get("n_features", -1)):
            problems.append("retained column count disagrees with the bundle record")

    expected = int((meta.get("model") or {}).get("n_boosters", 0))
    found = sum(1 for i in range(expected) if (directory / BOOSTER_PATTERN.format(index=i)).is_file())
    if found != expected:
        problems.append(f"expected {expected} booster file(s), found {found}")

    if check_fingerprint:
        recorded = featurizer.get("golden_fingerprint")
        if recorded:
            try:
                actual = golden_fingerprint(spec)
            except Exception as exc:  # pragma: no cover - RDKit failure
                problems.append(f"could not fingerprint the golden molecules: {exc}")
            else:
                if actual != recorded:
                    problems.append(
                        "the golden molecules now fingerprint differently "
                        f"({actual[:16]} vs {recorded[:16]}); this install would read different "
                        "features than the model was trained on"
                    )
    return not problems, problems


@lru_cache(maxsize=64)
def load(directory: str, check_fingerprint: bool = True) -> LoadedModel:
    """Load one bundle for prediction.

    Cached per process: ``predict`` reads each bundle once and reuses it across
    every chunk rather than per chunk.

    Raises:
        ModelError: The bundle is missing or incomplete.
        SpecMismatchError: It does not match the feature space this install builds.
    """
    path = Path(directory)
    meta = read_metadata(path)
    featurizer = meta["featurizer"]
    spec = FeaturizerSpec.from_dict(featurizer["spec"])

    ok, problems = verify(path, check_fingerprint=check_fingerprint)
    if not ok:
        raise SpecMismatchError(
            f"model bundle at {path} cannot be used here:\n  - " + "\n  - ".join(problems)
        )

    recorded_rdkit = (meta.get("libraries") or {}).get("rdkit")
    if recorded_rdkit:
        import rdkit

        if rdkit.__version__ != recorded_rdkit:
            log.warning(
                "%s was trained with RDKit %s, this install has %s. The golden fingerprint "
                "matches, so the feature space is unchanged.",
                meta["endpoint"]["key"], recorded_rdkit, rdkit.__version__,
            )

    lgb = load_lightgbm()
    n_boosters = int(meta["model"]["n_boosters"])
    boosters = [
        lgb.Booster(model_file=str(path / BOOSTER_PATTERN.format(index=i))) for i in range(n_boosters)
    ]

    kept = np.load(path / COLUMNS_FILE)["kept_indices"].astype(np.int32)
    basis = ColumnBasis(kept=kept, n_raw=int(featurizer["n_raw_features"]))

    reference = None
    reference_path = path / REFERENCE_FILE
    domain = meta.get("applicability_domain") or {}
    if reference_path.is_file() and domain:
        payload = np.load(reference_path)
        reference = Reference(
            packed=payload["packed"],
            popcount=payload["popcount"].astype(np.int32),
            n_bits=int(domain["n_bits"]),
            threshold=float(domain["threshold"]),
            subsampled=bool(domain.get("subsampled", False)),
        )

    endpoint = meta["endpoint"]
    output_range = endpoint.get("output_range")
    return LoadedModel(
        key=endpoint["key"],
        task=endpoint["task"],
        spec=spec,
        basis=basis,
        boosters=boosters,
        transform=transforms.from_dict(meta.get("transform")),
        calibrator=calibrate.from_dict(meta.get("calibrator")),
        interval=conformal.from_dict(meta.get("interval")),
        reference=reference,
        unit=endpoint.get("unit", ""),
        output_range=tuple(output_range) if output_range else None,
        decision_threshold=meta.get("decision_threshold"),
        metadata=meta,
    )


def available(models_dir: Path) -> list[str]:
    """List endpoint keys with a bundle present, in alphabetical order."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir() if (p / BUNDLE_FILE).is_file())


__all__ = ["LoadedModel", "available", "load", "read_metadata", "verify"]
