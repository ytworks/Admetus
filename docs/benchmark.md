# Benchmarks

Everything here is measured, not estimated. Reproduce the featurization numbers with
`scripts/bench_featurize.py`, the accuracy tables with `scripts/benchmark_tables.py`,
and the feature-set study with `scripts/feature_set_study.py`.

Measurement host: Apple Silicon (arm64), 10 cores (4 performance + 6 efficiency),
16 GB RAM, macOS, Python 3.12.13, RDKit 2026.09.2, NumPy 2.5.5.

## Accuracy

Being regenerated. Every figure measured before this point described a pipeline with
machinery that has since been removed, so they were deleted rather than relabelled.
`python scripts/benchmark_tables.py` rebuilds the tables from the bundles.

## What the additions cost, measured

The pipeline was built as the reference implementation plus a series of additions,
each with a plausible argument behind it and none of them requested. Two of them were
measured on identical predictions, which is the only comparison that isolates an
effect; the rest are recorded as separate runs, because their evaluation compounds
differ and subtracting across them would attribute a split change to a model change.

**On identical predictions.** Held-out ROC-AUC for `carcinogens`, random seed-0
holdout, 56 test rows, from one trained model:

| | ROC-AUC |
|---|---|
| Raw ensemble mean | **0.7506** |
| After the Platt calibrator | **0.2494** |

That is the same numbers through two output paths, so the 0.50 is the calibrator and
nothing else. 0.2494 is 1 - 0.7506, the signature of a reversed ranking.

**Separate runs, separate evaluation sets.** These are not differences; each row is
its own run and the compounds scored are not the same:

| Run | Evaluation set | ROC-AUC |
|---|---|---|
| Calibration + pruning + scaffold-grouped split | scaffold holdout, 55 rows | 0.249 |
| Calibration removed, pruning and scaffold split kept | scaffold holdout, 55 rows | 0.751 |
| Reference implementation followed: random holdout, no pruning, per-fold tuning | random seed-0 holdout, 56 rows | 0.885 |
| The shipped `models-v0.1.0` bundle, same pipeline at a 240s-per-fold budget | random seed-0 holdout, 56 rows | 0.901 |
| *The reference implementation's own figure* | *its own holdout, 56 rows* | *0.865* |

The first two share an evaluation set and so can be compared with each other; the
third cannot be subtracted from either. The fourth is the bundle that ships, on the
same evaluation set as the third and differing from it only in tuning budget -- which
is also the reason the two are not the same number. The README quotes this endpoint's
AUPRC, 0.8164, because AUPRC is its registered metric; ROC-AUC is used here only
because it is what the earlier rows were measured in.

What the table supports is that the finished pipeline reaches the reference's figure,
and that the calibrator alone cost half of ROC-AUC on the predictions it was applied
to.

### How a calibrator inverts a ranking

Platt scaling fits `sigmoid(a*x + b)` with the slope unconstrained. On `carcinogens`
it fitted `a = -37.6`: a monotonically *decreasing* function, which reverses the
order of every prediction. Held-out ROC-AUC went from 0.7506 on the raw ensemble mean
to 0.2494 after calibration -- which is 1 - 0.7506, the signature of a flip.

It happens because the calibrator is fitted on out-of-fold predictions. A fold model
that early-stopped at one round can be anti-correlated with its own validation rows,
and on that data a flip genuinely is the better fit. The project already had a rule
that a calibrator costing more than a small tolerance of the endpoint's metric is
rejected -- but that rule is checked on the same out-of-fold data, where the flip
scores as an improvement.

### How a sorted file becomes a single-class fold

`random_holdout` sorted the indices `train_test_split` returns. `cv_folds` cuts
contiguous blocks, so file order decided fold membership, and these dataset files are
sorted by label. On `bioavailability`:

| | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| Positive rate, indices sorted | 35% | **100%** | 94% |
| Positive rate, shuffled order kept | 74% | 80% | 74% |

A validation fold that is entirely positive has `binary_error` minimal at iteration
one, so the model stops there. Held-out ROC-AUC was 0.5121 against the reference's
0.6867.

### The others

* **The applicability-domain flag** was uninformative at 16 of 29 endpoints: a flat
  similarity-to-error curve collapsed `dili`'s threshold to 0.0139, so it reported
  water as in-domain.
* **Conformal prediction intervals** over-covered at 8 of 9 regression endpoints,
  median 97.0% against a nominal 90%. Aspirin's Caco-2 interval was [-17.44, 8.18]
  on labels spanning -7.76 to -3.51.
* **Scaffold-grouped splitting** leaked: grouping by input string rather than
  standardized structure put 9 of 129 held-out hERG molecules in a training
  scaffold. Two full retrains went into fixing something that should not have
  existed.
* **Per-endpoint feature-set selection** chose three overrides from a study whose
  baseline was the registry it was deciding, so adopting an override made it its own
  baseline and the study could not re-derive its own earlier answer.

The pattern across three rounds of review is the measurement worth keeping: **of 25
findings, none were in the featurize-train-predict path.** All of them were in these
additions or in the machinery built to support them.

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

