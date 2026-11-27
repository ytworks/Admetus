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
import sys
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
    golden_fingerprint,
)
from admetus.chem.spec import FeaturizerSpec
from admetus.errors import ModelError, SpecMismatchError
from admetus.logging import get_logger
from admetus.train import calibrate, transforms
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
            f"no model bundle at {directory}. "
            f"Train one with 'admetus train --endpoint {Path(directory).name}'."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path} is not readable JSON: {exc}") from exc


#: The last section a complete LightGBM text model carries. A file cut short keeps
#: valid-looking tree lines and loses this, which is the shape a partial copy or a
#: half-finished download has. Only the trailer is required: the marker that closes
#: the tree list sits before the feature-importance and parameter sections, which on
#: a wide model are themselves larger than any tail worth reading.
BOOSTER_TRAILER = "end of parameters"

#: How much of the end of a booster file to read when checking it is whole. The
#: trailer sits in the last few hundred bytes; reading the whole 37 MB of a large
#: model to find it would make verification cost more than loading it.
BOOSTER_TAIL_BYTES = 8192


def _booster_problem(path: Path) -> str | None:
    """Say why a booster file is not a whole model, or ``None`` when it looks whole."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(64)
            handle.seek(max(0, size - BOOSTER_TAIL_BYTES))
            tail = handle.read()
    except OSError as exc:  # pragma: no cover - unreadable file
        return f"could not be read: {exc}"
    if not head.startswith(b"tree"):
        return "does not start like a LightGBM text model"
    text = tail.decode("utf-8", errors="replace")
    if BOOSTER_TRAILER not in text:
        return (
            "is cut short: a complete model ends with its parameter section, and this "
            "one does not. It would fail at the first prediction."
        )
    return None


#: How long the child process gets to load a bundle's boosters. The largest here is
#: 37 MB across five folds and loads in about a second.
LOAD_CHECK_TIMEOUT = 120.0


def _boosters_load(directory: Path) -> list[str]:
    """Load the boosters in a child process and return what went wrong, if anything.

    A non-zero exit with no readable answer is itself the answer: the loader died,
    which is what a native abort looks like from here.
    """
    import subprocess

    try:
        done = subprocess.run(
            [sys.executable, "-m", "admetus.bundle._loadcheck", str(directory)],
            capture_output=True, text=True, timeout=LOAD_CHECK_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return [f"the boosters could not be checked: {exc}"]

    try:
        answer = json.loads(done.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {done.returncode}"
        return [
            "LightGBM could not read the boosters: the loader exited with "
            f"{done.returncode} without reporting ({tail})"
        ]
    return [] if answer.get("ok") else list(answer.get("problems") or ["the boosters do not load"])


def verify(
    directory: Path, check_fingerprint: bool = True, check_boosters: bool = True
) -> tuple[bool, list[str]]:
    """Check a bundle without loading it for prediction.

    Args:
        directory: The bundle.
        check_fingerprint: Fingerprint the golden molecules, which catches an RDKit
            change that moved bit positions.
        check_boosters: Check that every booster loads and serves the bundle's own
            column space. Counting the files is not enough, and neither is reading
            them: a structural check on the end of the file caught a model cut in
            half and accepted a real one missing its last two bytes, because the
            damage was in the trailing metadata LightGBM reads after the marker. So
            the boosters are loaded, in a child process -- LightGBM's parser aborts
            the process on some truncations, raising from C++ through a path Python
            cannot catch, and a verifier must not be the thing that kills the run.
            Their feature counts are compared with the retained basis too, because
            an intact booster from another feature preset loads perfectly and then
            predicts on the wrong width.

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
    elif check_boosters:
        # The cheap structural look first, so an obviously broken file does not cost
        # a subprocess, then the only check that settles it.
        for index in range(expected):
            path = directory / BOOSTER_PATTERN.format(index=index)
            problem = _booster_problem(path)
            if problem:
                problems.append(f"{path.name} {problem}")
        if not problems:
            problems.extend(_boosters_load(directory))

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

    # The boosters are loaded immediately below, so verifying them here would read
    # every file twice.
    ok, problems = verify(path, check_fingerprint=check_fingerprint, check_boosters=False)
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
