"""Featurizing molecules across processes.

RDKit's pharmacophore signature code is pure Python, so it holds the interpreter
lock throughout and threads give no speedup at all. Processes do.

Three things make the difference between a pool that helps and one that hurts:

* **One pool per run, not one per chunk.** Starting a worker costs 1.5 to 3.5
  seconds, almost all of it importing RDKit and building the signature
  scaffolding. Measured over 1,200 molecules, nine workers reached 848
  molecules per second including that startup.
* **Batches, not molecules.** A task of 64 molecules is several seconds of work
  against about a millisecond of inter-process overhead.
* **Scatter by index, not ordered iteration.** Six of the ten cores on the
  development machine are efficiency cores running this work at roughly 40% of a
  performance core, so completion order is uneven. Results carry their own
  indices and are placed into a preallocated list, which makes output order
  independent of completion order by construction.

Workers return set-bit indices rather than assembled matrices, which halves what
crosses the pipe: the data array is all ones and is recreated on the other side.
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass

import numpy as np

from admetus.chem.featurizer import featurize_one
from admetus.chem.spec import FeaturizerSpec
from admetus.logging import get_logger

log = get_logger("chem.parallel")

#: Molecules per worker task.
DEFAULT_BATCH_SIZE = 64

#: Below this many molecules, a pool costs more than it saves.
IN_PROCESS_THRESHOLD = 2_000

#: Leave one core for the parent, and do not go wild on a big machine.
MAX_DEFAULT_JOBS = 8

_WORKER_SPEC: FeaturizerSpec | None = None


def resolve_jobs(requested: int | None, n_molecules: int) -> int:
    """Decide how many worker processes to use.

    Honours a cgroup CPU limit where the interpreter exposes one, so a container
    does not oversubscribe.
    """
    if requested is not None and requested > 0:
        return 1 if n_molecules < IN_PROCESS_THRESHOLD and requested > 1 else requested
    if n_molecules < IN_PROCESS_THRESHOLD:
        return 1
    count = getattr(os, "process_cpu_count", None)
    available = count() if count else os.cpu_count()
    return max(1, min((available or 2) - 1, MAX_DEFAULT_JOBS))


def _init_worker(spec: FeaturizerSpec) -> None:
    """Prepare a worker process once.

    Builds the feature factory and signature scaffolding here rather than on the
    first molecule, pins each worker to one thread so the pool does not fight
    LightGBM's own threads, and silences RDKit's per-molecule warnings, which do
    not survive process spawn.
    """
    global _WORKER_SPEC
    os.environ["OMP_NUM_THREADS"] = "1"
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from admetus.chem.pharm2d import warm_up

    warm_up(spec)
    _WORKER_SPEC = spec


@dataclass
class Batch:
    """A unit of work.

    Attributes:
        positions: Where each result belongs in the caller's ordering.
        smiles: The strings to featurize.
    """

    positions: np.ndarray
    smiles: list[str]


@dataclass
class BatchResult:
    """What a worker sends back.

    Attributes:
        positions: Echoed from the batch, so the parent can scatter without
            depending on completion order.
        indices: Set-bit indices per molecule, ``None`` where it failed.
        canonical: Standardized SMILES, ``None`` where it failed.
        reasons: Failure reason, ``None`` on success.
        changed: Whether standardization altered each molecule.
    """

    positions: np.ndarray
    indices: list[np.ndarray | None]
    canonical: list[str | None]
    reasons: list[str | None]
    changed: list[bool]


def _featurize_batch(batch: Batch) -> BatchResult:
    """Featurize one batch inside a worker."""
    spec = _WORKER_SPEC
    indices: list[np.ndarray | None] = []
    canonical: list[str | None] = []
    reasons: list[str | None] = []
    changed: list[bool] = []
    for text in batch.smiles:
        idx, canon, reason, was_changed = featurize_one(text, spec)
        indices.append(idx)
        canonical.append(canon)
        reasons.append(reason)
        changed.append(was_changed)
    return BatchResult(batch.positions, indices, canonical, reasons, changed)


def make_batches(smiles: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[Batch]:
    """Cut a list of molecules into worker tasks."""
    batches: list[Batch] = []
    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        batches.append(Batch(np.arange(start, stop, dtype=np.int64), smiles[start:stop]))
    return batches


class Featurizer:
    """A reusable featurizer, parallel or in-process depending on the workload.

    Used as a context manager so the pool is created once and torn down once.
    """

    def __init__(self, spec: FeaturizerSpec, jobs: int = 1, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.spec = spec
        self.jobs = max(1, jobs)
        self.batch_size = batch_size
        self._pool = None

    def __enter__(self) -> Featurizer:
        if self.jobs > 1:
            # Pinned to spawn on every platform: fork deadlocks against the OpenMP
            # runtime, and Python 3.14 changed the Linux default anyway. One
            # behaviour everywhere is worth more than the faster start.
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                processes=self.jobs, initializer=_init_worker, initargs=(self.spec,)
            )
            log.debug("started %d worker process(es)", self.jobs)
        else:
            _init_worker(self.spec)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def run(self, smiles: list[str]) -> BatchResult:
        """Featurize molecules, returning results in the input's order."""
        n = len(smiles)
        indices: list[np.ndarray | None] = [None] * n
        canonical: list[str | None] = [None] * n
        reasons: list[str | None] = [None] * n
        changed: list[bool] = [False] * n

        batches = make_batches(smiles, self.batch_size)
        if self._pool is None:
            results = (_featurize_batch(b) for b in batches)
        else:
            # Unordered, because an efficiency-core task can take 2.5x as long and
            # ordered iteration would hold finished results waiting for a straggler.
            results = self._pool.imap_unordered(_featurize_batch, batches, chunksize=1)

        for result in results:
            for offset, position in enumerate(result.positions):
                indices[position] = result.indices[offset]
                canonical[position] = result.canonical[offset]
                reasons[position] = result.reasons[offset]
                changed[position] = result.changed[offset]

        return BatchResult(
            positions=np.arange(n, dtype=np.int64),
            indices=indices,
            canonical=canonical,
            reasons=reasons,
            changed=changed,
        )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "IN_PROCESS_THRESHOLD",
    "MAX_DEFAULT_JOBS",
    "Batch",
    "BatchResult",
    "Featurizer",
    "make_batches",
    "resolve_jobs",
]
