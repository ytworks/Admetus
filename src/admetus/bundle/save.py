"""Writing a trained model to disk.

Boosters are stored as LightGBM's own text format, not pickled. A pickled
``Booster`` carries the object graph of the library version that made it, and
loading it under a different LightGBM is unreliable; the text format is the
documented interchange format and stays readable.

Everything else a prediction needs sits beside them in ``bundle.json``: the
featurizer spec and its digest, the retained columns, the label transform and its
floor, the output range, the calibrator, the decision threshold, the interval
width and its measured coverage, the metrics with their confidence intervals, and
where the data came from. Loading checks the digest and refuses a mismatch,
because a model read against the wrong column space would predict confidently
from the wrong features.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from admetus.train.trainer import TrainedModel
from admetus.version import SCHEMA_VERSION, get_version

BUNDLE_FILE = "bundle.json"
COLUMNS_FILE = "columns.npz"
BOOSTER_PATTERN = "fold_{index}.txt"


def _library_versions() -> dict[str, str]:
    """Record the versions that produced the model."""
    import lightgbm
    import rdkit
    import scipy
    import sklearn

    return {
        "admetus": get_version(),
        "rdkit": rdkit.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def golden_fingerprint(spec) -> str:
    """Hash the fingerprint of three fixed molecules.

    Stored so that an RDKit change which silently moves bit positions is caught at
    load time, rather than producing confident nonsense.
    """
    import hashlib

    from admetus.chem.featurizer import featurize_one

    molecules = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    ]
    digest = hashlib.blake2b(digest_size=16)
    for smiles in molecules:
        indices, canonical, reason, _ = featurize_one(smiles, spec)
        if indices is None:  # pragma: no cover - the fixtures always parse
            raise RuntimeError(f"golden molecule failed to featurize: {reason}")
        digest.update(canonical.encode())
        digest.update(indices.tobytes())
    return digest.hexdigest()


def save(model: TrainedModel, directory: Path) -> Path:
    """Write a trained model to ``directory/<endpoint key>``.

    The directory is replaced atomically: a new one is built alongside and swapped
    in, so an interrupted write never leaves a half-bundle that looks complete.

    Returns:
        The bundle directory.
    """
    directory = Path(directory)
    target = directory / model.endpoint.key
    staging = directory / f".{model.endpoint.key}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for index, booster in enumerate(model.boosters):
        booster.save_model(str(staging / BOOSTER_PATTERN.format(index=index)), num_iteration=model.rounds)

    np.savez_compressed(staging / COLUMNS_FILE, kept_indices=model.basis.kept.astype(np.int32))

    endpoint = model.endpoint
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "endpoint": {
            "key": endpoint.key,
            "dataset": endpoint.tdc_name,
            "label_name": endpoint.label_name,
            "task": endpoint.task,
            "metric": endpoint.metric,
            "unit": endpoint.unit,
            "output_range": list(endpoint.output_range) if endpoint.output_range else None,
            "in_official_group": endpoint.in_official_group,
            "correlated_with": list(endpoint.correlated_with),
            "description": endpoint.description,
        },
        "featurizer": {
            "spec": model.spec.to_dict(),
            "spec_sha256": model.spec.spec_sha256,
            "column_sha256": model.spec.column_sha256(model.basis.kept),
            "n_raw_features": model.spec.n_features,
            "n_features": model.basis.n_cols,
            "golden_fingerprint": golden_fingerprint(model.spec),
        },
        "model": {
            "n_boosters": model.n_boosters,
            "num_boost_round": model.rounds,
            "params": {k: v for k, v in model.params.items() if k != "num_threads"},
        },
        "transform": model.transform.to_dict(),
        "metrics": model.test_metrics,
        "cross_validation": model.cv_record,
        "provenance": model.provenance,
        "libraries": _library_versions(),
    }
    (staging / BUNDLE_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)
    return target


__all__ = [
    "BOOSTER_PATTERN",
    "BUNDLE_FILE",
    "COLUMNS_FILE",
    "golden_fingerprint",
    "save",
]
