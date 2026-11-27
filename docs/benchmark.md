# Benchmarks

Everything here is measured, not estimated. Reproduce the featurization numbers with
`scripts/bench_featurize.py`. Model accuracy tables are added once training has run;
this file deliberately contains no placeholder numbers.

Measurement host: Apple Silicon (arm64), 10 cores (4 performance + 6 efficiency),
16 GB RAM, macOS, Python 3.12.13, RDKit 2026.09.2, NumPy 2.5.5.

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

## Which feature set each endpoint gets

The registry picked the feature space from dataset size alone: `morgan_p2` below 1,000
rows, `morgan_p23` above. That rule was a guess about where the 87,480-bit 3-point
pharmacophore block starts paying for itself, and this is the measurement of it.
Reproduce it with `scripts/feature_set_study.py`.

Every endpoint was trained once per feature set, **untuned** so that the comparison is
of feature spaces rather than of how much tuner budget each one happened to get, and
ranked on the **pooled out-of-fold** score. Selection never touches the held-out set:
a feature space chosen against it would turn that number into a selection score. Each
candidate ran at two seeds, which move the fold assignment, and the spread between them
is the noise floor a challenger has to clear -- that is what stops the study from
chasing fold luck. 174 runs, 48 minutes of compute.

The size rule survives at 26 of 29 endpoints. Three measurements disagreed with it by
more than their own noise, and `endpoints.yaml` now carries those three overrides:

| endpoint | the rule said | measured | the margin |
|---|---|---|---|
| `caco2` | `morgan_p2` (906 rows) | `morgan_p23` | 0.3901 against 0.3964 MAE, a 0.0063 gain over a 0.0050 floor |
| `half_life` | `morgan_p2` (665 rows) | `morgan_p23` | 0.5680 against 0.5466 Spearman, twice its 0.0104 floor |
| `tox21_nr_ar` | `morgan_p23` (7,265 rows) | `morgan` | 0.5074 against 0.4834 average precision: the pharmacophore blocks **cost** 0.0240 here |

`tox21_nr_ar` is the one endpoint where the wider space measurably hurts, and the reason
is in the data rather than in the fingerprint: at 4.3% positive there are about 310
actives, so 89,852 columns cost more in variance than they add in signal. It is also the
only endpoint where plain `morgan` wins at all.

Two patterns in the table below are worth keeping in mind before anyone widens the
feature space again:

- **Most of the value is in the 324-bit 2-point block, not the 87,480-bit 3-point one.**
  On `solubility` the 2-point block takes MAE from 1.3181 to 1.0316 and the 3-point block
  adds a further 0.0099. The same shape holds at `lipophilicity`, `ppbr` and every CYP
  panel.
- **The 3-point block earns its keep on the large panels and the regression endpoints,
  and is noise on the small ones.** Above 7,000 rows it wins or ties everywhere except
  `tox21_nr_ar`; below 1,000 rows it is inside the noise floor at every endpoint except
  `caco2` and `half_life`.

Every score is a pooled out-of-fold mean over two seeds. It is biased by construction --
the folds chose the model -- and is **not** a performance estimate; the held-out and
official numbers are the ones to read for that. Bold marks the best measured candidate,
and the `chosen` column is what the registry uses, which differs wherever the winner did
not clear the noise floor.

| endpoint | metric | default | morgan | morgan_p2 | morgan_p23 | chosen | why |
|---|---|---|---|---|---|---|---|
| `caco2` | mae | `morgan_p2` | 0.4376 | 0.3964 | **0.3901** | `morgan_p23` | beats the default by 0.0063, above the 0.0050 noise floor |
| `pampa` | auroc | `morgan_p23` | 0.6940 | **0.7367** | 0.7218 | `morgan_p23` | ahead by only 0.0149, within the 0.0272 noise floor |
| `hia` | auroc | `morgan_p2` | 0.8755 | **0.9406** | 0.9281 | `morgan_p2` | the size rule's default is also the best measured |
| `pgp` | auroc | `morgan_p23` | 0.9068 | 0.9259 | **0.9310** | `morgan_p23` | the size rule's default is also the best measured |
| `bioavailability` | auroc | `morgan_p2` | 0.6688 | 0.6968 | **0.6974** | `morgan_p2` | ahead by only 0.0005, within the 0.0143 noise floor |
| `lipophilicity` | mae | `morgan_p23` | 0.7826 | 0.6823 | **0.6721** | `morgan_p23` | the size rule's default is also the best measured |
| `solubility` | mae | `morgan_p23` | 1.3181 | 1.0316 | **1.0217** | `morgan_p23` | the size rule's default is also the best measured |
| `bbb` | auroc | `morgan_p23` | 0.8903 | **0.9222** | 0.9205 | `morgan_p23` | ahead by only 0.0017, within the 0.0050 noise floor |
| `ppbr` | mae | `morgan_p23` | 10.4416 | 9.5541 | **9.5510** | `morgan_p23` | the size rule's default is also the best measured |
| `vdss` | spearman | `morgan_p23` | 0.6500 | **0.6894** | 0.6823 | `morgan_p23` | ahead by only 0.0071, within the 0.0117 noise floor |
| `cyp2c9_veith` | auprc | `morgan_p23` | 0.7119 | 0.7596 | **0.7629** | `morgan_p23` | the size rule's default is also the best measured |
| `cyp2d6_veith` | auprc | `morgan_p23` | 0.5831 | 0.6458 | **0.6462** | `morgan_p23` | the size rule's default is also the best measured |
| `cyp3a4_veith` | auprc | `morgan_p23` | 0.8054 | 0.8344 | **0.8367** | `morgan_p23` | the size rule's default is also the best measured |
| `cyp1a2_veith` | auprc | `morgan_p23` | 0.8702 | 0.8984 | **0.9009** | `morgan_p23` | the size rule's default is also the best measured |
| `cyp2c19_veith` | auprc | `morgan_p23` | 0.7917 | **0.8223** | 0.8196 | `morgan_p23` | ahead by only 0.0027, within the 0.0050 noise floor |
| `cyp2c9_substrate` | auprc | `morgan_p2` | 0.3275 | 0.3518 | **0.3770** | `morgan_p2` | ahead by only 0.0252, within the 0.0379 noise floor |
| `cyp2d6_substrate` | auprc | `morgan_p2` | 0.5361 | **0.5923** | 0.5805 | `morgan_p2` | the size rule's default is also the best measured |
| `cyp3a4_substrate` | auroc | `morgan_p2` | 0.6703 | **0.6878** | 0.6854 | `morgan_p2` | the size rule's default is also the best measured |
| `half_life` | spearman | `morgan_p2` | 0.4990 | 0.5466 | **0.5680** | `morgan_p23` | beats the default by 0.0214, above the 0.0104 noise floor |
| `clearance_hepatocyte` | spearman | `morgan_p23` | 0.4989 | 0.5135 | **0.5260** | `morgan_p23` | the size rule's default is also the best measured |
| `clearance_microsome` | spearman | `morgan_p23` | 0.5985 | **0.6079** | 0.5938 | `morgan_p23` | ahead by only 0.0142, within the 0.0250 noise floor |
| `ld50` | mae | `morgan_p23` | 0.6000 | 0.5816 | **0.5771** | `morgan_p23` | the size rule's default is also the best measured |
| `herg` | auroc | `morgan_p2` | 0.8291 | **0.8506** | 0.8496 | `morgan_p2` | the size rule's default is also the best measured |
| `herg_karim` | auroc | `morgan_p23` | 0.7725 | 0.8273 | **0.8332** | `morgan_p23` | the size rule's default is also the best measured |
| `ames` | auroc | `morgan_p23` | 0.7973 | **0.8063** | 0.8052 | `morgan_p23` | ahead by only 0.0011, within the 0.0063 noise floor |
| `dili` | auroc | `morgan_p2` | 0.8171 | 0.8313 | **0.8374** | `morgan_p2` | ahead by only 0.0060, within the 0.0126 noise floor |
| `carcinogens` | auprc | `morgan_p2` | **0.7297** | 0.7117 | 0.7150 | `morgan_p2` | ahead by only 0.0180, within the 0.0627 noise floor |
| `skin_reaction` | auroc | `morgan_p2` | 0.6077 | 0.6195 | **0.6238** | `morgan_p2` | ahead by only 0.0043, within the 0.0156 noise floor |
| `tox21_nr_ar` | auprc | `morgan_p23` | **0.5074** | 0.4999 | 0.4834 | `morgan` | beats the default by 0.0240, above the 0.0137 noise floor |

## Column pruning and per-endpoint slicing

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

## Scaffold grouping

Bemis-Murcko framework grouping over the fetched datasets, showing why acyclic
molecules need singleton groups rather than one shared empty-scaffold group:

| endpoint       | rows   | groups | acyclic | largest group |
|----------------|--------|--------|---------|---------------|
| `dili`         | 475    | 345    | 35      | 11.4%         |
| `carcinogens`  | 276    | 219    | 19      | 6.9%          |
| `vdss`         | 1,111  | 823    | 38      | 7.6%          |
| `cyp2d6_veith` | 13,130 | 8,322  | 273     | 4.0%          |
| `solubility`   | 9,982  | 5,051  | **2,942** | 17.6%       |
| `ld50`         | 7,342  | 3,606  | **1,930** | 20.9%       |
| `tox21_nr_ar`  | 7,265  | 3,972  | **1,704** | 18.9%       |

Pooled into one group, those 2,942 acyclic solubility compounds would be 29% of the
dataset landing wholesale in a single fold.

`Chem.MurckoDecompose` and `rdScaffoldNetwork` both raise `RuntimeError: bad any cast`
for every molecule in RDKit 2026.09.2, so the framework reduction is implemented
directly in `admetus/data/splits.py` and asserted against known scaffolds.

## Feature-level leakage found in the published splits

Hashing each row's retained columns finds molecule pairs the model cannot
distinguish, which deduplicating on SMILES does not:

| endpoint           | rows | indistinguishable groups | spanning the split |
|--------------------|------|--------------------------|--------------------|
| `herg`             | 645  | 78                       | 5                  |
| `carcinogens`      | 276  | 2                        | 1                  |
| `dili`             | 475  | 1                        | 0                  |
| `vdss`             | 1,111| 2                        | 0                  |
| `cyp2d6_substrate` | 663  | 1                        | 0                  |

Those are stereoisomers and tautomers with distinct SMILES and identical
fingerprints. Where a group spans the split, the training rows are dropped and the
held-out rows kept, so the reported metric stays measured on the same compounds as
the published benchmark.
