"""SMILES standardization.

Both fingerprint blocks are blind to some differences that SMILES can express:
Morgan runs with ``useChirality=False`` and the pharmacophore signature is purely
topological. Standardization makes that blindness explicit and consistent instead
of accidental, so the same compound written two ways lands on the same bits.

The pipeline is configurable per dataset because it is not always harmless. For
aqueous solubility, for instance, the measured value belongs to the salt, so
replacing a salt with its largest organic fragment trains on a label that
describes a different chemical species. Datasets like that omit that step.

Nothing here ever substitutes a different molecule for one that fails to parse.
A failure returns ``None`` with a reason, and the caller reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from admetus.errors import UsageError

# RDKit's warnings are per-molecule noise here; failures are reported through the
# return value instead. Disabled at import and again inside workers, since the
# setting does not survive process spawn.
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StandardizeResult:
    """Outcome of standardizing one SMILES string.

    Attributes:
        mol: The standardized molecule, or ``None`` on failure.
        canonical: Canonical SMILES of ``mol``, or ``None`` on failure.
        reason: Why it failed, or ``None`` on success.
        changed: Whether standardization altered the structure.
    """

    mol: Chem.Mol | None
    canonical: str | None
    reason: str | None = None
    changed: bool = False

    @property
    def ok(self) -> bool:
        """Whether a usable molecule came out."""
        return self.mol is not None


def _step_cleanup(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize, normalize functional groups, reionize, assign stereochemistry."""
    return rdMolStandardize.Cleanup(mol)


def _step_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest covalent fragment, dropping counter-ions and solvents."""
    return rdMolStandardize.FragmentParent(mol)


def _step_uncharge(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize charges where a neutral form exists."""
    return rdMolStandardize.Uncharger().uncharge(mol)


#: Step name to implementation. Spec and registry refer to steps by these names.
STEPS = {
    "cleanup": _step_cleanup,
    "largest_fragment": _step_largest_fragment,
    "uncharge": _step_uncharge,
}


def validate_steps(steps: tuple[str, ...]) -> None:
    """Raise if any step name is unknown.

    Raises:
        UsageError: A name is not a known standardization step.
    """
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise UsageError(
            f"unknown standardization steps {unknown}; expected a subset of {sorted(STEPS)}"
        )


def standardize(smiles: str, steps: tuple[str, ...] = ("cleanup", "largest_fragment", "uncharge")) -> StandardizeResult:
    """Parse and standardize one SMILES string.

    Args:
        smiles: Input SMILES. Surrounding whitespace is ignored.
        steps: Ordered step names, a subset of :data:`STEPS`.

    Returns:
        A :class:`StandardizeResult`. On failure ``mol`` is ``None`` and
        ``reason`` says what went wrong; no substitute molecule is ever returned.
    """
    if smiles is None:
        return StandardizeResult(None, None, "smiles is null")
    text = smiles.strip()
    if not text:
        return StandardizeResult(None, None, "smiles is empty")

    try:
        mol = Chem.MolFromSmiles(text)
    except UnicodeEncodeError:
        # The CSV reader uses surrogateescape, so a cell the declared encoding
        # cannot decode arrives as lone surrogates and RDKit's C++ boundary raises
        # when it encodes them. Left to propagate, one such cell aborted the whole
        # batch with exit 1 and the valid structures around it got no prediction.
        # It is a property of one row, so it is one row's failure.
        return StandardizeResult(
            None, None, "the smiles cell could not be decoded with the declared encoding"
        )
    if mol is None:
        return StandardizeResult(None, None, "rdkit could not parse the smiles")
    if mol.GetNumAtoms() == 0:
        return StandardizeResult(None, None, "molecule has no atoms")

    try:
        original = Chem.MolToSmiles(mol)
    except Exception as exc:  # pragma: no cover - canonicalization of a parsed mol
        return StandardizeResult(None, None, f"could not canonicalize: {exc}")

    for name in steps:
        step = STEPS.get(name)
        if step is None:
            raise UsageError(f"unknown standardization step {name!r}")
        try:
            mol = step(mol)
        except Exception as exc:
            return StandardizeResult(None, None, f"standardization step {name!r} failed: {exc}")
        if mol is None:
            return StandardizeResult(None, None, f"standardization step {name!r} removed the molecule")

    if mol.GetNumAtoms() == 0:
        return StandardizeResult(None, None, "standardization left no atoms")

    # The standardization steps can return a molecule whose ring information was
    # never initialized -- FragmentParent does, for instance. Canonical SMILES
    # does not need it, but pharmacophore feature matching does, and it fails
    # with "RingInfo not initialized" deep inside the matcher. Re-sanitize once
    # here so every consumer gets a fully initialized molecule.
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return StandardizeResult(None, None, f"could not sanitize after standardization: {exc}")

    try:
        canonical = Chem.MolToSmiles(mol)
    except Exception as exc:  # pragma: no cover
        return StandardizeResult(None, None, f"could not canonicalize after standardization: {exc}")

    return StandardizeResult(mol, canonical, None, changed=canonical != original)


@lru_cache(maxsize=65_536)
def canonicalize(smiles: str, steps: tuple[str, ...]) -> str | None:
    """Return the standardized canonical SMILES, or ``None`` on failure.

    Memoized, because deduplication asks the same question repeatedly within a
    chunk and standardization costs a millisecond or two per molecule.
    """
    return standardize(smiles, steps).canonical


__all__ = ["STEPS", "StandardizeResult", "canonicalize", "standardize", "validate_steps"]
