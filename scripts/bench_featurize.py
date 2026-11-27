"""End-to-end featurization throughput on a realistic drug-like library."""
import itertools
import multiprocessing as mp
import random
import time

from rdkit import Chem

from admetus.chem.featurizer import featurize_many
from admetus.chem.pharm2d import warm_up
from admetus.chem.spec import get_spec

# Combinatorial library with a drug-like size distribution.
CORES = ["c1ccc(cc1)", "c1ccc2ccccc2c1", "c1ccncc1", "c1cc2ccccc2s1", "c1ccc(cc1)C(=O)"]
LINKS = ["NC(=O)", "C(=O)N", "OCC", "NCC", "S(=O)(=O)N", "CC(O)", "N(C)C(=O)"]
TAILS = ["N1CCN(C)CC1", "c1ccc(O)cc1", "C(=O)O", "c1ccc(F)cc1", "N1CCOCC1", "C(C)(C)C", "c1cn[nH]c1"]

def library(n, seed=0):
    rng = random.Random(seed)
    out, seen = [], set()
    combos = list(itertools.product(CORES, LINKS, TAILS, LINKS, TAILS))
    rng.shuffle(combos)
    for a, b, c, d, e in combos:
        smi = a + b + c + d + e
        m = Chem.MolFromSmiles(smi)
        if m is None or not (18 <= m.GetNumHeavyAtoms() <= 40):
            continue
        cs = Chem.MolToSmiles(m)
        if cs in seen:
            continue
        seen.add(cs)
        out.append(cs)
        if len(out) >= n:
            break
    return out

SPEC_NAME = "morgan_p23"

def _init():
    warm_up(get_spec(SPEC_NAME))

def _work(batch):
    return featurize_many(batch, get_spec(SPEC_NAME)).matrix.nnz

if __name__ == "__main__":
    lib = library(1200)
    atoms = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in lib]
    print(f"library: {len(lib)} molecules, heavy atoms min/median/max = "
          f"{min(atoms)}/{sorted(atoms)[len(atoms)//2]}/{max(atoms)}")

    spec = get_spec(SPEC_NAME)
    warm_up(spec)
    t0 = time.perf_counter()
    r = featurize_many(lib[:300], spec)
    serial = time.perf_counter() - t0
    print(f"serial   : {serial:.2f} s for 300 -> {serial/300*1000:.2f} ms/mol, "
          f"{300/serial:.0f} mol/s, mean nnz {r.matrix.nnz/r.n_ok:.0f}")

    for jobs in (4, 9):
        batches = [lib[i:i+64] for i in range(0, len(lib), 64)]
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        with ctx.Pool(jobs, initializer=_init) as pool:
            total = sum(pool.imap_unordered(_work, batches, chunksize=1))
        dt = time.perf_counter() - t0
        print(f"jobs={jobs:<2}  : {dt:.2f} s for {len(lib)} -> {len(lib)/dt:.0f} mol/s "
              f"(incl. pool startup), speedup {(serial/300*len(lib))/dt:.1f}x")
        print(f"            projected 100k unique: {100_000/(len(lib)/dt)/60:.1f} min")
