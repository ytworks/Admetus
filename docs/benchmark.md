# Benchmarks

Everything here is measured, not estimated. Reproduce the featurization numbers with
`scripts/bench_featurize.py`, the accuracy tables with `scripts/benchmark_tables.py`,
and the feature-set study with `scripts/feature_set_study.py`.

Measurement host: Apple Silicon (arm64), 10 cores (4 performance + 6 efficiency),
16 GB RAM, macOS, Python 3.12.13, RDKit 2026.09.2, NumPy 2.5.5.

## Accuracy

All 29 endpoints are trained. Two numbers are reported per endpoint and they measure
different things:

* **Held-out** is this project's own scaffold split, looked at exactly once, with a
  bootstrap interval. The split sends the largest scaffold groups to training, so the
  held-out set is the unusual chemistry on purpose and the estimate is pessimistic.
* **Official protocol** is the benchmark group's published split, trained on its
  `train_val` portion and scored once per seed on its test set, reported as the mean
  and standard deviation over the protocol's five seeds. This is the number that can
  sit beside the published leaderboard. It covers the 22 endpoints of the official
  group; the other seven have no published split.

The official numbers here were taken **with tuning off**, because five seeds per
endpoint is five tuning runs. They are a floor, not a best effort. 110 runs, 36
minutes. The feature space comes from the size rule, which depends on a row count
and on nothing else -- not from the measured overrides in the registry, because for
an endpoint of the official group those were selected over a dataset that holds the
published test compounds.

21 of the 22 are leaderboard-comparable. `solubility` is not: 1,995 of its 1,997
published test rows could be featurized, so its mean is over a subset and the table
says so.

Training the 29 endpoints took 282 minutes of wall clock across three parallel
workers at a 900-second tuner budget, the longest single endpoint being
`cyp1a2_veith` at 1,243 seconds.

### Held-out, on this project's scaffold split

| Endpoint | Task | Metric | Held-out | 95% CI | n | Feature set |
|---|---|---|---|---|---|---|
| `caco2` | regression | mae | 0.3786 | [0.3344, 0.4237] | 181 | `morgan_p23` |
| `pampa` | classification | auroc | 0.6927 | [0.6013, 0.7823] | 407 | `morgan_p23` |
| `hia` | classification | auroc | 0.9034 | [0.7483, 0.9972] | 116 | `morgan_p2` |
| `pgp` | classification | auroc | 0.8919 | [0.8462, 0.9306] | 242 | `morgan_p23` |
| `bioavailability` | classification | auroc | 0.6750 | [0.5507, 0.7955] | 128 | `morgan_p2` |
| `lipophilicity` | regression | mae | 0.6633 | [0.6261, 0.6990] | 840 | `morgan_p23` |
| `solubility` | regression | mae | 1.3204 | [1.2619, 1.3805] | 1996 | `morgan_p23` |
| `bbb` | classification | auroc | 0.8731 | [0.8256, 0.9135] | 393 | `morgan_p23` |
| `ppbr` | regression | mae | 9.1137 | [7.8984, 10.3877] | 323 | `morgan_p23` |
| `vdss` | regression | spearman | 0.6304 | [0.5308, 0.7162] | 222 | `morgan_p23` |
| `cyp2c9_veith` | classification | auprc | 0.7190 | [0.6846, 0.7542] | 2418 | `morgan_p23` |
| `cyp2d6_veith` | classification | auprc | 0.7299 | [0.6963, 0.7620] | 2626 | `morgan_p23` |
| `cyp3a4_veith` | classification | auprc | 0.7876 | [0.7625, 0.8122] | 2466 | `morgan_p23` |
| `cyp1a2_veith` | classification | auprc | 0.8661 | [0.8500, 0.8820] | 2516 | `morgan_p23` |
| `cyp2c19_veith` | classification | auprc | 0.7950 | [0.7711, 0.8192] | 2533 | `morgan_p23` |
| `cyp2c9_substrate` | classification | auprc | 0.2491 | [0.1477, 0.3892] | 133 | `morgan_p2` |
| `cyp2d6_substrate` | classification | auprc | 0.5038 | [0.3780, 0.6707] | 133 | `morgan_p2` |
| `cyp3a4_substrate` | classification | auroc | 0.6507 | [0.5529, 0.7409] | 133 | `morgan_p2` |
| `half_life` | regression | spearman | 0.2273 | [0.0576, 0.3876] | 133 | `morgan_p23` |
| `clearance_hepatocyte` | regression | spearman | 0.3804 | [0.2533, 0.5000] | 204 | `morgan_p23` |
| `clearance_microsome` | regression | spearman | 0.3869 | [0.2671, 0.4913] | 220 | `morgan_p23` |
| `ld50` | regression | mae | 0.5243 | [0.4990, 0.5489] | 1468 | `morgan_p23` |
| `herg` | classification | auroc | 0.8872 | [0.8238, 0.9386] | 129 | `morgan_p2` |
| `herg_karim` | classification | auroc | 0.8534 | [0.8389, 0.8672] | 2689 | `morgan_p23` |
| `ames` | classification | auroc | 0.8025 | [0.7796, 0.8244] | 1451 | `morgan_p23` |
| `dili` | classification | auroc | 0.8342 | [0.7518, 0.9119] | 95 | `morgan_p2` |
| `carcinogens` | classification | auprc | 0.8101 | [0.6002, 0.9453] | 55 | `morgan_p2` |
| `skin_reaction` | classification | auroc | 0.6245 | [0.4951, 0.7485] | 81 | `morgan_p2` |
| `tox21_nr_ar` | classification | auprc | 0.5079 | [0.3900, 0.6164] | 1451 | `morgan` |

### The official protocol, on the published split

Mean and standard deviation over the protocol's five seeds, tuning off.

| Endpoint | Metric | Official protocol | Seeds | Test rows scored | Comparable |
|---|---|---|---|---|---|
| `caco2` | mae | 0.3214 +/- 0.0038 | 5 | 182 | yes |
| `hia` | auroc | 0.9808 +/- 0.0027 | 5 | 117 | yes |
| `pgp` | auroc | 0.9284 +/- 0.0006 | 5 | 245 | yes |
| `bioavailability` | auroc | 0.7259 +/- 0.0130 | 5 | 128 | yes |
| `lipophilicity` | mae | 0.6808 +/- 0.0008 | 5 | 840 | yes |
| `solubility` | mae | 1.1871 +/- 0.0054 | 5 | 1995 of 1997 | no: 1995 of 1997 published test row(s) were scored; the rest could not be featurized |
| `bbb` | auroc | 0.8732 +/- 0.0019 | 5 | 406 | yes |
| `ppbr` | mae | 7.7789 +/- 0.0362 | 5 | 559 | yes |
| `vdss` | spearman | 0.6263 +/- 0.0042 | 5 | 226 | yes |
| `cyp2c9_veith` | auprc | 0.7228 +/- 0.0006 | 5 | 2419 | yes |
| `cyp2d6_veith` | auprc | 0.6615 +/- 0.0009 | 5 | 2626 | yes |
| `cyp3a4_veith` | auprc | 0.8288 +/- 0.0041 | 5 | 2467 | yes |
| `cyp2c9_substrate` | auprc | 0.3627 +/- 0.0062 | 5 | 135 | yes |
| `cyp2d6_substrate` | auprc | 0.6895 +/- 0.0114 | 5 | 135 | yes |
| `cyp3a4_substrate` | auroc | 0.6537 +/- 0.0061 | 5 | 135 | yes |
| `half_life` | spearman | 0.5551 +/- 0.0062 | 5 | 135 | yes |
| `clearance_hepatocyte` | spearman | 0.4009 +/- 0.0041 | 5 | 243 | yes |
| `clearance_microsome` | spearman | 0.5928 +/- 0.0093 | 5 | 221 | yes |
| `ld50` | mae | 0.7075 +/- 0.0011 | 5 | 1478 | yes |
| `herg` | auroc | 0.8454 +/- 0.0015 | 5 | 132 | yes |
| `ames` | auroc | 0.7745 +/- 0.0009 | 5 | 1457 | yes |
| `dili` | auroc | 0.9045 +/- 0.0035 | 5 | 96 | yes |

### Four things these tables say that are worth reading before quoting one

**1. The cross-validation score flattered the model at 26 of the 29 endpoints**,
by a median of 0.053 in each metric's own units. That is what
`cv_score_biased: true` has always asserted, now measured. The worst cases are
`solubility` (0.5001 of MAE), `half_life` (0.3355 of Spearman) and
`clearance_microsome` (0.2078). Held-out came out *better* at only three:
`carcinogens`, `cyp2d6_veith` and `herg`. Anyone tempted to report an out-of-fold
number as performance should start here.

**2. Eight endpoints are budget-limited, not search-limited.** At a 900-second
budget the tuner completed 14 to 19 trials on `solubility`, `herg_karim`, `ld50`,
`lipophilicity` and four of the five CYP inhibition panels, against 65 to 68 on the
small ones. Those eight got roughly a quarter of the search the rest did, so their
numbers are a floor for this architecture rather than its ceiling.

**3. Three endpoints have an interval that reaches no-signal, and saying so is the
result.** `half_life` held-out Spearman is 0.1982 with an interval of
[0.0259, 0.3578], whose lower end is indistinguishable from no correlation at all.
`skin_reaction` is AUROC 0.6245 [0.4951, 0.7485], straddling 0.5. `cyp2c9_substrate`
is average precision 0.2491 [0.1477, 0.3892] at a 21% positive rate, so the interval
contains the base rate. These are reported as they are rather than omitted.

**4. Which split is harder depends on the endpoint, in both directions.** The
official split is easier at `half_life` (0.1982 against 0.5353), `clearance_microsome`
(0.3869 against 0.5580) and `cyp2d6_substrate` (0.5038 against 0.6895), and harder at
`ld50` (MAE 0.5227 against 0.7075) and `vdss` (0.6435 against 0.6116). "The scaffold
split is the harder one" is not a safe generalization, and the gap between the two
columns is not a measure of either split's difficulty on its own.

### What the two leaks were worth, measured

Two leaks were found by review on this branch and both are fixed. Each was measured by
rerunning everything, because a leak's size is not something to reason about.

**Scaffold grouping saw the input string, not the standardized molecule.** On the hERG
split, 9 of 129 held-out molecules shared a standardized scaffold with a surviving
training row; after the fix, 0 do. Rerunning all 29 endpoints:

| | endpoints |
|---|---|
| held-out score got worse | 8 |
| held-out score got better | 7 |
| bit-identical | 13 |

Median change: **0.0000**. The 13 identical ones are datasets with no molecule whose
scaffold key changes under standardization, so their splits were already right. The
largest moves were `half_life` +0.0291 and `pampa` -0.0278, both at endpoints whose
intervals are too wide to read a difference that size.

**The official protocol inherited a feature set chosen with its own test labels.**
Three registry entries carry a measured override, and for an official-group endpoint
the data that chose it holds the published test compounds: 97 of Caco-2's 182 test
rows, 102 of half-life's 135. The protocol now takes the size rule instead.

| endpoint | with the inherited override | with the size rule | change |
|---|---|---|---|
| `caco2` | 0.3084 (`morgan_p23`) | 0.3214 (`morgan_p2`) | **0.0130 worse** |
| `half_life` | 0.5353 (`morgan_p23`) | 0.5551 (`morgan_p2`) | **0.0198 better** |

Opposite directions, which is the point worth keeping. A leak is not a thumb on the
scale that always flatters; it is a number you can no longer say what it measures.
Caco-2's leaked figure was optimistic and half-life's was pessimistic, and neither was
comparable with the leaderboard. Reporting the direction as "leaks inflate scores"
would have been a tidier story and a false one.

Both fixes also moved the official numbers at every other endpoint, because the fold
grouping inside the official run is the same code. Those changes are small -- 0.0006 to
0.035, largest at `clearance_microsome` (0.5580 to 0.5928) and `clearance_hepatocyte`
(0.3843 to 0.4009) -- and the three substrate endpoints did not move at all.

### Why the standard deviation over seeds is small, and not comparable

The spread over seeds runs from 0.0005 to 0.0362, which is far tighter than the
leaderboard's own error bars. That is a property of this pipeline, not a result:
the seed moves the fold assignment inside the training portion and nothing else.
`train_val` is not resampled into train and valid, because the round count comes
from scaffold-grouped cross-validation over the whole training portion. A protocol
that resamples train/valid per seed varies more.

So the **mean is comparable with the leaderboard and the standard deviation is not.**
Quoting our spread against theirs would understate the real variability of the
approach.

### Prediction interval coverage

Conformal intervals on the regression endpoints, measured on the held-out set
against the nominal level:

| Endpoint | Nominal | Measured coverage | Held-out rows |
|---|---|---|---|
| `caco2` | 90% | 100.0% | 181 |
| `lipophilicity` | 90% | 99.3% | 840 |
| `solubility` | 90% | 89.4% | 1996 |
| `ppbr` | 90% | 92.6% | 323 |
| `vdss` | 90% | 97.3% | 222 |
| `half_life` | 90% | 97.0% | 133 |
| `clearance_hepatocyte` | 90% | 93.1% | 204 |
| `clearance_microsome` | 90% | 96.4% | 220 |
| `ld50` | 90% | 97.0% | 1468 |

**Eight of the nine over-cover, with a median of 97.0% against a nominal 90%**, and
only `solubility` lands on the level. The intervals are therefore wider than
asked for almost everywhere.

The mechanism is visible in how the width is fitted: it comes from the out-of-fold
residuals of the fold models, each of which trains on four fifths of the data, while
the width is then applied to the production ensemble, which averages all five. The
ensemble's errors are smaller than the calibration set's, so an interval sized for the
calibration set is too wide for the thing it is attached to. Narrowing it is a change
to a measured quantity and has not been attempted here; the measured coverage is
recorded in every bundle so a caller can see what it actually gets.

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
