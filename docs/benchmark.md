# Benchmarks

Everything here is measured, not estimated. Reproduce the featurization numbers with
`scripts/bench_featurize.py`, the accuracy tables with `scripts/benchmark_tables.py`,
and the feature-set study with `scripts/feature_set_study.py`.

Measurement host: Apple Silicon (arm64), 10 cores (4 performance + 6 efficiency),
16 GB RAM, macOS, Python 3.12.13, RDKit 2026.09.2, NumPy 2.5.5.

## Accuracy

Deleted, not relabelled. Every accuracy figure in this document was measured on a
scaffold-grouped holdout, which has been replaced by the reference implementation's
random 20% holdout with folds over the rest. Those numbers describe a pipeline this
code no longer is, and keeping them with a note would have left the project's most
quoted tables wrong in a way a reader cannot see.

All 29 endpoints are retraining. `python scripts/benchmark_tables.py` regenerates
the tables from the bundles, and `admetus benchmark run` the official five-seed
numbers.

What went with them: the prediction-interval coverage table, because conformal
intervals were removed, and the before/after measurement of two leaks in the scaffold
machinery, because that machinery is gone. The leak measurements are preserved in the
commit that recorded them, which is the right place for the history of code that no
longer exists.

The engineering measurements below -- feature widths, fingerprint cost, throughput,
pruning and slicing -- are unaffected by the split, because none of them depends on it.

## Feature space widths

Verified against `SigFactory.GetSigSize()` and pinned in `tests/unit/test_chem_spec.py`.

| feature set  | Morgan            | pharmacophore          | total  |
|--------------|-------------------|------------------------|--------|
| `morgan`     | r=2, 2048 bits    | none                   | 2,048  |
| `morgan_p2`  | r=6, 2048 bits    | 2-point, 324 bits      | 2,372  |
| `morgan_p23` | r=6, 2048 bits    | 2+3-point, 87,804 bits | 89,852 |

The pharmacophore widths follow from RDKit's `BaseFeatures.fdef` (8 feature families)
and 9 topological distance bins: 2-point is C(9,2)=36 family pairs x 9 bins = 324, and
3-point adds C(10,3)=120 family triples x 9^3=729 bin triples = 87,480.

Enabling `trianglePruneBins` would give 75,204 bits. It is not used, so the widths stay
constants the tests can assert.

## Fingerprint cost per molecule

Morgan is negligible: **7-50 microseconds**, three orders of magnitude below the
pharmacophore block. The pharmacophore signature dominates, and its 3-point term grows
steeply with the number of detected features, hence with molecule size.

| heavy atoms | `morgan_p23` | `morgan_p2` | on-bits (p23) |
|-------------|--------------|-------------|---------------|
| 11          | 0.4 ms       | 0.22 ms     | 33            |
| 21          | 2.3 ms       | 0.57 ms     | 253           |
| 31          | 8.4 ms       | 1.04 ms     | 616           |
| 41          | 21.5 ms      | 1.72 ms     | 884           |
| 51          | 44.5 ms      | 2.56 ms     | 985           |
| 71          | 132.8 ms     | 4.77 ms     | 1,019         |

`morgan_p2` is both 37x narrower and 10-30x cheaper, because skipping the 3-point
enumeration removes the cubic term. That matters for datasets small enough that the
full space is overkill anyway.

## End-to-end featurization throughput

A combinatorial drug-like library (1,200 molecules, 18-32 heavy atoms, median 25),
`morgan_p23`, spawn pool, batches of 64:

| workers | throughput | note                                  |
|---------|------------|---------------------------------------|
| 1       | 268 mol/s  | 3.73 ms/molecule                      |
| 4       | 751 mol/s  | includes pool startup                 |
| 9       | 848 mol/s  | includes pool startup                 |

**100,000 unique molecules project to roughly 2 minutes** at 9 workers. Mean non-zeros
per molecule is **426**, so a 10,000-row chunk holds about 4.3 million non-zeros rather
than the 25 million a 2,500-non-zero assumption would predict.

Speedup saturates near 3.2x rather than 9x for two reasons: six of the ten cores are
efficiency cores running pure-Python work at roughly 40% of a performance core, and
pool startup (RDKit import plus `SigFactory.Init()`, 1.5-3.5 s per worker) is a visible
share of a 1,200-molecule run. Both argue for one pool per process rather than one per
chunk, and for falling back to in-process work on small inputs.

## Why there is no vendored fast pharmacophore path

`rdkit.Chem.Pharm2D` is pure Python, so it holds the GIL throughout, and a profile of a
21-heavy-atom molecule (3.8 ms per call) shows where the time goes:

| cost | site |
|------|------|
| 63% cumulative | `Generate._ShortestPathsMatch` (12,090 calls) |
| 36% cumulative | `SigFactory.GetBitIdx` -> `_findBinIdx`, including a linear `list.index` scan (10,650 calls) |
| 10% | one-off feature detection per molecule |

Memoizing `GetBitIdx` and converting the distance matrix to nested Python lists would
plausibly give 1.5-2x. Against a measured 2 minutes per 100,000 molecules that does not
justify vendoring and maintaining a copy of RDKit's generator, which would silently
diverge from upstream. The profile is recorded here so the decision can be revisited if
a real workload shows otherwise.

## There is one feature space, and it is not chosen

Every endpoint is featurized the same way: a radius-6 Morgan fingerprint over 2,048
bits concatenated with the 2- and 3-point pharmacophore signature, 89,852 columns.
This is the reference implementation's featurization, which takes no account of
dataset size -- its own notebook runs the full space on 280 compounds.

What used to be here: three presets, a rule that picked between them below and above
1,000 rows, three measured per-endpoint overrides, and a 174-run study that chose
them. None of it was asked for. It is also the layer that produced the most
embarrassing finding of the review: the study compared each candidate against
whatever the registry currently said, so adopting an override made it its own
baseline and a rerun could not re-derive the earlier decision. The scores behind the
overrides were measured on a scaffold split that no longer exists either.

The measurements that remain useful are the ones about the space itself rather than
about choosing between spaces: the block widths below, the per-molecule cost, and the
throughput. Those are unchanged.

## Column pruning and slicing

Measured on a 10,000-row by 89,852-column matrix with 4.26 million non-zeros
(the measured mean of 426 non-zeros per molecule), narrowing to a 20,000-column basis:

| operation | time |
|-----------|------|
| `matrix[:, kept]` (scipy column indexing) | 11.5 ms |
| reindexing `indices`/`indptr` by hand     | 20.7 ms |

Serving all 29 endpoints from one raw matrix therefore costs about **330 ms per
10,000-row chunk**, roughly 3% of the time that chunk takes to featurize. scipy's
own indexing is the faster of the two, so `apply_basis` uses it and the featurizer
stays simple: there is no shared intermediate basis, and each endpoint's matrix is
sliced from the raw one on demand.

