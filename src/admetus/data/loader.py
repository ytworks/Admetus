"""Reading published dataset files into a normalized table.

The published files do not share a shape, so normalization is explicit and
asserted rather than assumed:

* most carry ``Drug_ID / Drug / Y``;
* some older ones carry ``ID / X / Y``;
* the plasma-protein-binding file adds a ``Species`` column and must be filtered
  to one species;
* the Tox21 file is wide, with one column per assay and a NaN for every compound
  that assay was not run on.

Row counts therefore differ from the raw files, and each endpoint records the
count it expects so an upstream revision is noticed instead of silently changing
what gets trained.

No pandas: this reads a few thousand rows of TSV, which ``csv`` handles, and
keeping pandas out of the import path keeps the inference install small.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from admetus.data.registry import Endpoint
from admetus.errors import InputDataError
from admetus.logging import get_logger

log = get_logger("data.loader")

#: Accepted spellings of the identifier, structure and label columns.
ID_COLUMNS = ("Drug_ID", "ID")
SMILES_COLUMNS = ("Drug", "X", "SMILES")
LABEL_COLUMNS = ("Y",)

#: Raised past this share of unparseable labels: the file is not what we think.
MAX_BAD_LABEL_FRACTION = 0.01


@dataclass
class Dataset:
    """A normalized dataset ready to be featurized.

    Attributes:
        key: Endpoint key.
        ids: Compound identifiers, parallel to ``smiles``.
        smiles: Structures as published, before standardization.
        y: Labels, on the published scale.
        task: ``classification`` or ``regression``.
        provenance: What normalization did, for the bundle's record.
    """

    key: str
    ids: list[str]
    smiles: list[str]
    y: np.ndarray
    task: str
    provenance: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def n_positive(self) -> int:
        """Number of positive labels. Classification only."""
        return int(np.count_nonzero(self.y == 1))

    @property
    def pos_rate(self) -> float:
        """Share of positive labels, ``nan`` for regression or an empty set."""
        if self.task != "classification" or len(self) == 0:
            return float("nan")
        return self.n_positive / len(self)

    def summary(self) -> str:
        """One line for a log."""
        if self.task == "classification":
            return f"{self.key}: {len(self)} rows, {self.n_positive} positive ({self.pos_rate:.1%})"
        finite = self.y[np.isfinite(self.y)]
        if finite.size == 0:
            return f"{self.key}: {len(self)} rows, no finite labels"
        return (
            f"{self.key}: {len(self)} rows, y in [{finite.min():.4g}, {finite.max():.4g}], "
            f"median {float(np.median(finite)):.4g}"
        )


def _pick(header: list[str], candidates: tuple[str, ...], what: str, path: Path) -> int:
    """Return the index of the first candidate column present in the header."""
    for name in candidates:
        if name in header:
            return header.index(name)
    raise InputDataError(
        f"{path.name} has no {what} column. Looked for {list(candidates)}, header is {header}."
    )


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a TSV file into a header and rows."""
    if not path.is_file():
        raise InputDataError(f"dataset file is missing: {path}. Run 'admetus data fetch' first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise InputDataError(f"{path} is empty") from None
        rows = [row for row in reader if row]
    return header, rows


def _parse_label(text: str, task: str) -> float:
    """Parse one label cell, returning ``nan`` when it is absent or unreadable."""
    stripped = text.strip().strip('"')
    if not stripped or stripped.lower() in ("na", "nan", "none", "null"):
        return float("nan")
    try:
        value = float(stripped)
    except ValueError:
        return float("nan")
    if task == "classification" and value not in (0.0, 1.0):
        return float("nan")
    return value


def load(endpoint: Endpoint, directory: Path, verify_digest: bool = True) -> Dataset:
    """Read and normalize one endpoint's dataset.

    The file's digest is checked against the one pinned in the registry before any
    of it is read. The fetcher checks too, but a dataset can be altered after it
    was fetched, and a model's recorded provenance is only worth something if the
    bytes are verified at the point they are actually used.

    Args:
        endpoint: Which dataset to read.
        directory: Where ``admetus data fetch`` put the files.
        verify_digest: Check the file against the registry's pinned digest.
            Endpoints with no pin are read either way.

    Returns:
        A normalized :class:`Dataset`.

    Raises:
        InputDataError: The file is missing, does not match its pinned digest,
            has no recognizable columns, or is not the shape the endpoint
            describes.
    """
    path = Path(directory) / f"{endpoint.key}.tab"
    if verify_digest and endpoint.sha256:
        from admetus.data.download import sha256_file

        if not path.is_file():
            raise InputDataError(
                f"dataset file is missing: {path}. Run 'admetus data fetch' first."
            )
        actual = sha256_file(path)
        if actual != endpoint.sha256:
            raise InputDataError(
                f"{endpoint.key}: the dataset on disk does not match the digest pinned in the "
                f"registry, so it is not the data any model was trained on.\n"
                f"  pinned: {endpoint.sha256}\n"
                f"  found:  {actual}\n"
                f"  file:   {path}\n\n"
                "Re-fetch it with 'admetus data fetch --endpoint "
                f"{endpoint.key} --overwrite'."
            )
    header, rows = _read_rows(path)
    n_raw = len(rows)

    smiles_at = _pick(header, SMILES_COLUMNS, "structure", path)
    try:
        id_at = _pick(header, ID_COLUMNS, "identifier", path)
    except InputDataError:
        id_at = None

    if endpoint.label_name:
        if endpoint.label_name not in header:
            raise InputDataError(
                f"{path.name} has no label column {endpoint.label_name!r}. "
                f"Available: {[h for h in header if h not in SMILES_COLUMNS + ID_COLUMNS]}"
            )
        label_at = header.index(endpoint.label_name)
    else:
        label_at = _pick(header, LABEL_COLUMNS, "label", path)

    species_at = header.index("Species") if endpoint.species_filter and "Species" in header else None
    if endpoint.species_filter and species_at is None:
        raise InputDataError(
            f"{path.name} declares a species filter but has no Species column; header is {header}"
        )

    ids: list[str] = []
    smiles: list[str] = []
    labels: list[float] = []
    n_wrong_species = 0
    n_missing_label = 0
    n_short_row = 0
    widest = max(smiles_at, label_at, species_at or 0, id_at or 0)

    for position, row in enumerate(rows):
        if len(row) <= widest:
            n_short_row += 1
            continue
        if species_at is not None and row[species_at].strip().strip('"') != endpoint.species_filter:
            n_wrong_species += 1
            continue
        label = _parse_label(row[label_at], endpoint.task)
        if math.isnan(label):
            n_missing_label += 1
            continue
        structure = row[smiles_at].strip().strip('"')
        if not structure:
            n_missing_label += 1
            continue
        ids.append(row[id_at].strip().strip('"') if id_at is not None else str(position))
        smiles.append(structure)
        labels.append(label)

    if n_short_row:
        log.warning("%s: skipped %d row(s) with too few fields", endpoint.key, n_short_row)

    # Deduplicate on the published structure, as the upstream loaders do. Rows that
    # disagree on the label are resolved rather than silently kept as contradictions.
    deduped = _dedupe(ids, smiles, labels, endpoint)

    dataset = Dataset(
        key=endpoint.key,
        ids=deduped["ids"],
        smiles=deduped["smiles"],
        y=np.asarray(deduped["y"], dtype=np.float64),
        task=endpoint.task,
        provenance={
            "path": str(path),
            "header": header,
            "n_raw_rows": n_raw,
            "n_wrong_species": n_wrong_species,
            "n_missing_label": n_missing_label,
            "n_short_row": n_short_row,
            "n_duplicate_removed": deduped["n_removed"],
            "n_label_conflict": deduped["n_conflict"],
            "n_rows": len(deduped["smiles"]),
            "label_column": endpoint.label_name or header[label_at],
            "species_filter": endpoint.species_filter,
        },
    )
    _check_labels(dataset, endpoint, n_raw)
    return dataset


def _dedupe(ids: list[str], smiles: list[str], labels: list[float], endpoint: Endpoint) -> dict:
    """Collapse rows that share a structure.

    Regression takes the median of the group, which is robust to a single bad
    replicate. Classification keeps a unanimous label and drops the group when
    replicates disagree, because a contradictory label teaches nothing.
    """
    order: dict[str, list[int]] = {}
    for position, structure in enumerate(smiles):
        order.setdefault(structure, []).append(position)

    out_ids: list[str] = []
    out_smiles: list[str] = []
    out_y: list[float] = []
    n_removed = 0
    n_conflict = 0

    for structure, positions in order.items():
        if len(positions) == 1:
            out_ids.append(ids[positions[0]])
            out_smiles.append(structure)
            out_y.append(labels[positions[0]])
            continue
        n_removed += len(positions) - 1
        values = [labels[p] for p in positions]
        if endpoint.task == "classification":
            if len(set(values)) > 1:
                n_conflict += 1
                continue
            resolved = values[0]
        else:
            resolved = float(np.median(values))
        out_ids.append(ids[positions[0]])
        out_smiles.append(structure)
        out_y.append(resolved)

    if n_removed:
        log.debug("%s: collapsed %d duplicate row(s), dropped %d conflicting group(s)",
                  endpoint.key, n_removed, n_conflict)
    return {
        "ids": out_ids, "smiles": out_smiles, "y": out_y,
        "n_removed": n_removed, "n_conflict": n_conflict,
    }


def _check_labels(dataset: Dataset, endpoint: Endpoint, n_raw: int) -> None:
    """Assert the label column is what the endpoint claims, and warn on a row-count change."""
    if len(dataset) == 0:
        raise InputDataError(f"{endpoint.key}: no usable rows after normalization")

    if not np.all(np.isfinite(dataset.y)):
        raise InputDataError(f"{endpoint.key}: labels contain non-finite values after normalization")

    if endpoint.task == "classification":
        unexpected = set(np.unique(dataset.y)) - {0.0, 1.0}
        if unexpected:
            raise InputDataError(
                f"{endpoint.key} is classification but labels include {sorted(unexpected)}"
            )
        if dataset.n_positive == 0 or dataset.n_positive == len(dataset):
            raise InputDataError(f"{endpoint.key}: labels are all one class, nothing to learn")

    if endpoint.transform and np.any(dataset.y < 0):
        raise InputDataError(
            f"{endpoint.key} declares transform {endpoint.transform!r} but has negative labels"
        )

    bad = dataset.provenance["n_missing_label"]
    if n_raw and bad / n_raw > MAX_BAD_LABEL_FRACTION and not endpoint.label_name:
        log.warning(
            "%s: %d of %d rows had no usable label (%.1f%%). Check the label column.",
            endpoint.key, bad, n_raw, 100 * bad / n_raw,
        )

    expected = endpoint.expected_rows
    if expected is not None and len(dataset) != expected:
        log.warning(
            "%s: normalized to %d rows, registry expects %d. The published dataset may have "
            "been revised; the actual count is recorded in the bundle.",
            endpoint.key, len(dataset), expected,
        )


__all__ = ["Dataset", "load"]
