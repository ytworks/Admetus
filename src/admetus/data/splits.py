"""Splitting compounds so that an estimate means something.

A random split of a chemical dataset flatters a model: close analogues land on
both sides, and the model is scored on structures it has effectively already
seen. Grouping by Bemis-Murcko scaffold instead asks the question that matters
for prospective use, which is how the model does on chemistry it has not met.

Two traps this module exists to avoid:

**Acyclic molecules all share the empty scaffold.** ``MurckoScaffoldSmiles``
returns ``""`` for anything without a ring, and several datasets contain hundreds
of such compounds. Treated as one group they land wholesale in a single fold and
can be a fifth of the data. They get singleton groups instead.

**Grouped folds do not stratify.** One Tox21 assay here is 4.3% positive, and an
unstratified grouped 5-fold can hand a fold two or three positives. Average
precision on three positives is noise, and that noise would become the
hyperparameter search's objective. Classification therefore uses
``StratifiedGroupKFold``, and a fold too thin to score is refused loudly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger

from admetus.errors import InputDataError
from admetus.logging import get_logger

log = get_logger("data.splits")

# Per-molecule parse warnings are noise here; unparseable structures are handled
# through return values. Disabled again inside workers, since this does not
# survive process spawn.
RDLogger.DisableLog("rdApp.*")

#: No scaffold group may exceed this share of the dataset, or folds go lopsided.
MAX_GROUP_FRACTION = 0.05

#: A validation fold with fewer positives than this cannot be scored meaningfully.
MIN_FOLD_POSITIVES = 10

#: Quantile count used to stratify a regression target.
REGRESSION_STRATA = 10


@dataclass(frozen=True)
class Split:
    """A train/test partition by row index."""

    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.train, self.test).size:
            raise ValueError("train and test overlap")


def murcko_framework(mol: Chem.Mol) -> str:
    """Return the Bemis-Murcko framework of a parsed molecule as a SMILES string.

    The framework is the ring systems plus the linkers joining them. It is built
    by repeatedly peeling terminal atoms that are not in a ring, which is
    well-defined because removing such an atom never changes which atoms are in
    rings -- so ring membership is computed once and stays valid.

    Atoms double-bonded to a ring atom are kept. Stripping them would destroy the
    aromatic form of rings such as the purine-dione of caffeine.

    The result is produced with :func:`Chem.MolFragmentToSmiles` over the retained
    atoms rather than by building a new molecule. Removing a methyl from an
    aromatic nitrogen leaves a valence that cannot be sanitized on its own, and
    this string is only ever used as a grouping key, never re-parsed.

    Returns:
        A canonical SMILES for the framework, or ``""`` for an acyclic molecule.

    Note:
        This is implemented here rather than with ``Chem.MurckoDecompose`` because
        that function, and ``rdScaffoldNetwork``, both raise ``RuntimeError: bad
        any cast`` for every molecule in RDKit 2026.09.2.
    """
    n = mol.GetNumAtoms()
    ring_info = mol.GetRingInfo()
    in_ring = [ring_info.NumAtomRings(i) > 0 for i in range(n)]
    if not any(in_ring):
        return ""

    exocyclic_multiple = [False] * n
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() <= 1.0:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if in_ring[a] and not in_ring[b]:
            exocyclic_multiple[b] = True
        elif in_ring[b] and not in_ring[a]:
            exocyclic_multiple[a] = True

    neighbours = [[x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()] for i in range(n)]
    keep = set(range(n))
    while True:
        peel = {
            i
            for i in keep
            if not in_ring[i]
            and not exocyclic_multiple[i]
            and sum(1 for j in neighbours[i] if j in keep) <= 1
        }
        if not peel:
            break
        keep -= peel

    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(keep), canonical=True)


def murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko framework of a SMILES string.

    Returns ``""`` when the molecule has no ring, and also when it cannot be
    parsed: an unparseable structure gets a group of its own rather than being
    silently pooled with every other acyclic compound (see :func:`scaffold_groups`).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return murcko_framework(mol)


def scaffold_groups(smiles: list[str]) -> np.ndarray:
    """Assign a group label per molecule.

    Molecules with a ring share their scaffold's label. Molecules without one get
    a label of their own, keyed on their canonical SMILES, so the empty scaffold
    never becomes a single enormous group.
    """
    groups: list[str] = []
    for text in smiles:
        scaffold = murcko_scaffold(text)
        groups.append(scaffold if scaffold else f"\0acyclic:{text}")
    return np.asarray(groups, dtype=object)


def check_group_sizes(groups: np.ndarray, max_fraction: float = MAX_GROUP_FRACTION) -> None:
    """Warn when one group is large enough to distort any fold it lands in."""
    counts = Counter(groups.tolist())
    n = len(groups)
    biggest, size = counts.most_common(1)[0]
    if n and size / n > max_fraction:
        label = "acyclic singleton" if str(biggest).startswith("\0acyclic:") else repr(biggest)
        log.warning(
            "largest scaffold group holds %d of %d rows (%.1f%%), above the %.0f%% guideline: %s",
            size, n, 100 * size / n, 100 * max_fraction, label,
        )


def _strata(y: np.ndarray, task: str) -> np.ndarray:
    """Return the stratification label per row."""
    if task == "classification":
        return y.astype(int)
    # Quantile bins keep the target distribution comparable across folds.
    finite = y[np.isfinite(y)]
    if finite.size == 0:  # pragma: no cover - guarded upstream
        return np.zeros(y.size, dtype=int)
    edges = np.quantile(finite, np.linspace(0, 1, REGRESSION_STRATA + 1)[1:-1])
    return np.searchsorted(edges, y)


def scaffold_split(smiles: list[str], test_fraction: float = 0.2) -> Split:
    """Partition by scaffold, largest groups to train first.

    This mirrors the deterministic scheme the published benchmark uses: scaffold
    groups are ordered by size and filled into train until it is full, which puts
    the rare and singleton scaffolds in test. The test set is therefore the
    unusual chemistry on purpose, and the estimate is a pessimistic one.

    Raises:
        InputDataError: ``test_fraction`` is not a usable fraction.
    """
    if not 0 < test_fraction < 1:
        raise InputDataError(f"test_fraction must be between 0 and 1, got {test_fraction}")

    groups = scaffold_groups(smiles)
    check_group_sizes(groups)

    members: dict[str, list[int]] = {}
    for position, group in enumerate(groups):
        members.setdefault(group, []).append(position)

    # Largest first, ties broken by the group label so the split is reproducible.
    ordered = sorted(members.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))

    n_total = len(smiles)
    n_train_target = n_total - int(round(n_total * test_fraction))
    train: list[int] = []
    test: list[int] = []
    for _, positions in ordered:
        if len(train) + len(positions) <= n_train_target:
            train.extend(positions)
        else:
            test.extend(positions)

    if not test or not train:
        raise InputDataError(
            f"scaffold split produced {len(train)} train and {len(test)} test rows from "
            f"{n_total}; the dataset has too few distinct scaffolds to split this way"
        )
    return Split(np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(test, dtype=np.int64)))


def cv_folds(
    smiles: list[str],
    y: np.ndarray,
    task: str,
    n_splits: int = 5,
    seed: int = 0,
    min_fold_positives: int = MIN_FOLD_POSITIVES,
) -> list[Split]:
    """Build stratified, scaffold-grouped cross-validation folds.

    Folds are reduced rather than abandoned when the data cannot support the
    requested count, and refused outright when even three folds would leave a
    validation fold too thin to score.

    Args:
        smiles: Structures, used only to derive scaffold groups.
        y: Labels, used to stratify.
        task: ``classification`` or ``regression``.
        n_splits: Requested fold count.
        seed: Shuffling seed, recorded by the caller for reproducibility.
        min_fold_positives: Refuse if any validation fold has fewer positives.

    Returns:
        One :class:`Split` per fold, ``train`` and ``test`` being the fold's
        training rows and held-out rows.

    Raises:
        InputDataError: No fold count down to 3 gives scorable folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = scaffold_groups(smiles)
    check_group_sizes(groups)
    strata = _strata(np.asarray(y), task)
    indices = np.arange(len(smiles))

    requested = n_splits
    for candidate in range(n_splits, 2, -1):
        splitter = StratifiedGroupKFold(n_splits=candidate, shuffle=True, random_state=seed)
        try:
            folds = [
                Split(np.sort(train), np.sort(test))
                for train, test in splitter.split(indices, strata, groups)
            ]
        except ValueError as exc:
            log.debug("%d folds rejected by the splitter: %s", candidate, exc)
            continue

        if task == "classification":
            thin = [int(np.count_nonzero(y[f.test] == 1)) for f in folds]
            if min(thin) < min_fold_positives:
                log.debug("%d folds give a validation fold with only %d positives", candidate, min(thin))
                continue
        if min(f.test.size for f in folds) == 0:  # pragma: no cover - defensive
            continue

        if candidate != requested:
            log.warning(
                "reduced cross-validation from %d to %d folds: %d rows could not support more",
                requested, candidate, len(smiles),
            )
        return folds

    detail = ""
    if task == "classification":
        detail = f" It has {int(np.count_nonzero(np.asarray(y) == 1))} positive rows in total."
    raise InputDataError(
        f"cannot build cross-validation folds for {len(smiles)} rows with at least "
        f"{min_fold_positives} positives per validation fold.{detail} "
        "Lower min_fold_positives deliberately, or do not train this endpoint."
    )


def fold_report(folds: list[Split], y: np.ndarray, task: str) -> list[dict[str, object]]:
    """Summarize each fold, for the training log and the bundle record."""
    out: list[dict[str, object]] = []
    for i, fold in enumerate(folds):
        entry: dict[str, object] = {
            "fold": i,
            "n_train": int(fold.train.size),
            "n_valid": int(fold.test.size),
        }
        if task == "classification":
            entry["n_valid_positive"] = int(np.count_nonzero(y[fold.test] == 1))
            entry["valid_pos_rate"] = float(np.mean(y[fold.test] == 1))
        else:
            entry["valid_y_median"] = float(np.median(y[fold.test]))
        out.append(entry)
    return out


__all__ = [
    "MAX_GROUP_FRACTION",
    "MIN_FOLD_POSITIVES",
    "Split",
    "check_group_sizes",
    "cv_folds",
    "fold_report",
    "murcko_framework",
    "murcko_scaffold",
    "scaffold_groups",
    "scaffold_split",
]
