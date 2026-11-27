"""The official benchmark group's own train/test splits.

``admetus evaluate`` reports a scaffold split this project chose itself. That number
is honest but it is not comparable with the published leaderboard, because the
leaderboard's compounds are different compounds. The benchmark group publishes its
splits as a single 1.4 MB archive, one directory per dataset holding ``train_val.csv``
and ``test.csv``, and this module reads them.

The archive is pinned by digest for the same reason the datasets are: a bundle that
claims a leaderboard-comparable number has to name the bytes that number came from.
The pin is trust on first use, as it is everywhere else here -- the source publishes
no checksum of its own.

Two things are deliberately *not* done:

* The ``train_val`` portion is not split again into train and valid. The protocol
  leaves its use open, and this pipeline has no use for a single validation set: the
  round count comes from scaffold-grouped cross-validation over the whole training
  portion, which uses the rows better than holding an eighth of them aside.
* The test labels are read once per run, to score. Nothing selects against them --
  not the feature set, not a hyperparameter, not a calibrator.

Only the 22 endpoints of the official group are covered. The other seven in the
registry have no published split, and asking for one raises rather than inventing it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from admetus.data.download import fetch, sha256_file
from admetus.data.loader import Dataset
from admetus.data.registry import Endpoint
from admetus.errors import InputDataError, UsageError
from admetus.logging import get_logger

log = get_logger("data.benchmark")

#: Harvard Dataverse file id of ``admet_group.zip``.
ARCHIVE_DATAVERSE_ID = 4426004

#: Digest of the archive as published. A mismatch is refused, as for the datasets.
ARCHIVE_SHA256 = "bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1"

ARCHIVE_FILE = "admet_group.zip"

#: Directory the archive unpacks into.
ARCHIVE_ROOT = "admet_group"

#: The seeds the leaderboard's protocol uses. Five runs, mean and standard
#: deviation reported, which is why a single run is not a comparable number.
OFFICIAL_SEEDS = (1, 2, 3, 4, 5)

#: Columns in the split files.
ID_COLUMN = "Drug_ID"
SMILES_COLUMN = "Drug"
LABEL_COLUMN = "Y"


@dataclass(frozen=True)
class OfficialSplit:
    """One endpoint's published split.

    Attributes:
        key: Endpoint key.
        dataset: Training and evaluation rows in one table, training rows first.
        holdout: Boolean mask over ``dataset`` rows marking the published test set.
        n_train_val: Size of the published training portion.
        n_test: Size of the published test set.
    """

    key: str
    dataset: Dataset
    holdout: np.ndarray
    n_train_val: int
    n_test: int


def archive_url() -> str:
    """Where the split archive is published."""
    return f"https://dataverse.harvard.edu/api/access/datafile/{ARCHIVE_DATAVERSE_ID}"


def archive_path(directory: Path) -> Path:
    """Where the archive is kept inside a data directory."""
    return Path(directory) / ARCHIVE_FILE


def split_directory(directory: Path, endpoint: Endpoint) -> Path:
    """Where one endpoint's split files live once the archive is unpacked.

    The archive names each directory after the dataset as the source publishes it,
    lowercased, which is exactly the registry's ``tdc_name`` folded down. Asserted
    for all 22 official endpoints in the tests, so this is a mapping rather than a
    guess.
    """
    return Path(directory) / ARCHIVE_ROOT / endpoint.tdc_name.lower()


def fetch_archive(directory: Path, overwrite: bool = False) -> Path:
    """Download and unpack the split archive, verifying its digest first.

    Returns:
        The unpacked ``admet_group`` directory.

    Raises:
        NetworkError: The download failed, or the bytes do not match the pin.
    """
    from admetus.data.download import extract_benchmark_archive

    directory = Path(directory)
    target = archive_path(directory)
    root = directory / ARCHIVE_ROOT
    if root.is_dir() and not overwrite and target.is_file() and sha256_file(target) == ARCHIVE_SHA256:
        log.debug("the benchmark splits are already unpacked in %s", root)
        return root

    result = fetch(archive_url(), target, overwrite=overwrite, expected_sha256=ARCHIVE_SHA256)
    names = extract_benchmark_archive(result.path, directory)
    log.info("unpacked %d entries from %s into %s", len(names), ARCHIVE_FILE, directory)
    return root


def _read_split_file(path: Path) -> tuple[list[str], list[str], list[float]]:
    """Read one ``train_val.csv`` or ``test.csv``."""
    if not path.is_file():
        raise InputDataError(
            f"the published split file is missing: {path}. "
            "Run 'admetus benchmark fetch' to download and unpack the archive."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        for column in (ID_COLUMN, SMILES_COLUMN, LABEL_COLUMN):
            if column not in header:
                raise InputDataError(f"{path.name} has no {column!r} column; header is {header}")
        ids: list[str] = []
        smiles: list[str] = []
        labels: list[float] = []
        for number, row in enumerate(reader, start=2):
            text = (row[LABEL_COLUMN] or "").strip()
            try:
                label = float(text)
            except ValueError:
                raise InputDataError(f"{path.name} line {number}: label {text!r} is not a number") from None
            ids.append(row[ID_COLUMN] or f"row{number}")
            smiles.append(row[SMILES_COLUMN])
            labels.append(label)
    if not smiles:
        raise InputDataError(f"{path} holds no rows")
    return ids, smiles, labels


def load_official_split(endpoint: Endpoint, directory: Path) -> OfficialSplit:
    """Read one endpoint's published split.

    Args:
        endpoint: Which endpoint. Must be in the official group.
        directory: The data directory the archive was unpacked into.

    Returns:
        The :class:`OfficialSplit`.

    Raises:
        UsageError: The endpoint has no published split.
        InputDataError: A split file is missing or malformed.
    """
    if not endpoint.in_official_group:
        raise UsageError(
            f"{endpoint.key} is not in the official benchmark group, so no published split exists "
            "for it. Its held-out number comes from this project's own scaffold split; see "
            "'admetus evaluate'."
        )
    base = split_directory(directory, endpoint)
    train_ids, train_smiles, train_y = _read_split_file(base / "train_val.csv")
    test_ids, test_smiles, test_y = _read_split_file(base / "test.csv")

    ids = train_ids + test_ids
    smiles = train_smiles + test_smiles
    y = np.asarray(train_y + test_y, dtype=np.float64)
    holdout = np.zeros(len(smiles), dtype=bool)
    holdout[len(train_smiles):] = True

    dataset = Dataset(
        key=endpoint.key,
        ids=ids,
        smiles=smiles,
        y=y,
        task=endpoint.task,
        provenance={
            "source": "official benchmark group split archive",
            "archive": ARCHIVE_FILE,
            "archive_sha256": ARCHIVE_SHA256,
            "dataset_directory": endpoint.tdc_name.lower(),
            "n_train_val": len(train_smiles),
            "n_test": len(test_smiles),
        },
    )
    log.info(
        "%s: published split, %d train_val + %d test rows",
        endpoint.key, len(train_smiles), len(test_smiles),
    )
    return OfficialSplit(
        key=endpoint.key,
        dataset=dataset,
        holdout=holdout,
        n_train_val=len(train_smiles),
        n_test=len(test_smiles),
    )


__all__ = [
    "ARCHIVE_DATAVERSE_ID",
    "ARCHIVE_FILE",
    "ARCHIVE_ROOT",
    "ARCHIVE_SHA256",
    "OFFICIAL_SEEDS",
    "OfficialSplit",
    "archive_path",
    "archive_url",
    "fetch_archive",
    "load_official_split",
    "split_directory",
]
