# Admetus

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)

**English** | [日本語](README.ja.md)

Admetus predicts ADMET properties from SMILES. It trains one LightGBM fold ensemble per public
dataset, over a Morgan fingerprint concatenated with a 2D pharmacophore signature, and serves them
through a command line built for programs rather than people: self-describing, non-interactive,
with stable exit codes.

Give it a CSV with a SMILES column and it appends prediction columns, leaving every other column
byte for byte as it found them.

> **Status: version 0.1, in development.** No trained models are published yet, so the accuracy
> tables below are empty on purpose rather than filled with placeholders. Train locally with
> `admetus train`. Interfaces may still change.

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Output columns](#output-columns)
- [Endpoints](#endpoints)
- [Benchmark](#benchmark)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Development](#development)
- [References](#references)
- [License](#license)

## Features

- **29 endpoints**, one model per public dataset. The 22 datasets of the Therapeutics Data Commons
  ADMET benchmark group, plus 7 further public ADMET datasets.
- **Your other columns are not touched.** Quoting, line endings, byte-order marks, text encoding,
  duplicate column names and column order all survive. Only the SMILES column is interpreted.
- **A molecule that fails is reported, never substituted.** Unparseable input yields empty
  prediction cells and a reason, and does not fail the run.
- **Automatic hyperparameter tuning**, over the same scaffold-grouped folds the model is
  cross-validated on. Studies are kept in SQLite and resume.
- **Honest numbers.** Every reported metric carries a bootstrap confidence interval, the
  cross-validation score is recorded as biased and never reported as performance, and small
  datasets are flagged so a caller knows not to over-read them.
- **Built for programs.** `admetus endpoints --json` describes every endpoint and output column, so
  a caller can plan an invocation without reading this file. `admetus doctor` says in one call
  whether the install can predict and what to fix if not.

## How it works

Each endpoint is trained on its own, because the datasets do not share compounds, units or
assays.

1. **Standardize, then split.** SMILES are cleaned, salts removed and charges neutralized before
   splitting, so two rows that collapse to the same molecule cannot land on opposite sides.
2. **Featurize.** A Morgan fingerprint and a 2D pharmacophore signature, concatenated. The
   pharmacophore block enumerates sets of 2 or 3 pharmacophore features and the topological
   distance between each pair, so a bit means something like "a donor and an acceptor 4 to 5 bonds
   apart, with an aromatic 6 to 7 bonds from both".
3. **Split by scaffold.** Scaffold groups are filled into training largest-first, which puts the
   rare chemistry in the held-out set deliberately. A random split would flatter the model by
   putting close analogues on both sides.
4. **Audit the split at the level the model can see.** Both fingerprint blocks are blind to
   stereochemistry, so a pair of enantiomers is one molecule as far as the model is concerned.
   Rows are hashed by feature vector and offending *training* rows dropped; held-out rows are kept,
   so the metric stays measured on the same compounds as the published benchmark.
5. **Prune columns.** Columns set in too few or too many training molecules are dropped. The band
   is a prevalence rather than a count, because a fixed count means "1.8% of molecules" on the
   smallest dataset and "0.04%" on the largest.
6. **Tune, then refit.** Hyperparameters come from the automatic tuner; the models that ship are
   refitted afterwards at a fixed number of boosting rounds, so nothing that ships was stopped
   using data it was then scored on.
7. **Calibrate and measure.** Out-of-fold predictions give the probability calibration, the
   recommended decision threshold and the conformal interval width. The held-out set is looked at
   exactly once.

## Requirements

- Python 3.12 or 3.13
- An OpenMP runtime. LightGBM's wheels link against one but do not ship it, so without it the
  first prediction fails with a `dlopen` error that explains nothing. `admetus doctor` names the
  fix.
- macOS (arm64 or x86_64) or Linux (amd64 or arm64)
- About 3 GB of memory for a prediction run over all endpoints

## Installation

On macOS, install the OpenMP runtime first.

```bash
brew install libomp
```

On Debian or Ubuntu.

```bash
sudo apt-get install -y libgomp1
```

Then install the package.

```bash
uv pip install -e '.[train]'
```

Check that the install can actually predict.

```bash
admetus doctor
```

### Docker

The image carries the OpenMP runtime, so nothing else is needed.

```bash
docker run --rm -v "$PWD:/data" ghcr.io/ytworks/admetus:latest endpoints
```

Prediction needs the bundles too. The image reads them from `/models`, so mount the
directory that holds them; mounting only `/data` leaves `/models` an empty volume and
the run cannot find a model.

```bash
docker run --rm -v "$PWD:/data" -v "$PWD/models:/models" ghcr.io/ytworks/admetus:latest predict -i /data/in.csv -o /data/out.csv --endpoint dili
```

## Quickstart

List what can be predicted, and which models are present.

```bash
admetus endpoints
```

Fetch a dataset and train a model for it. Each dataset's digest is pinned in the registry, so a
file that does not match is refused rather than adopted.

```bash
admetus data fetch --endpoint dili
```

```bash
admetus train --endpoint dili --time-budget 600
```

Predict for a single structure, without a file.

```bash
admetus predict --smiles 'CC(=O)Oc1ccccc1C(=O)O' --endpoint dili --json
```

Predict for a CSV, appending columns and writing a report.

```bash
admetus predict -i compounds.csv -o predicted.csv --endpoint dili --report report.json
```

Train every endpoint. This takes hours, skips endpoints that already have a model, and resumes.

```bash
admetus train --all --time-budget 900
```

Run the official protocol, which trains on the benchmark group's own published split and scores
its test set once per seed. The number this reports is the one comparable with the leaderboard;
`admetus evaluate` reports this project's own scaffold split instead.

```bash
admetus benchmark fetch
```

```bash
admetus benchmark run --all --time-budget 300 --report official.json
```

Package local bundles for a release, and install a published one.

```bash
admetus models package --out dist/models
```

```bash
admetus models pull --all
```

## CLI reference

Logs go to standard error and data to standard output, so output can be piped. `--json` prints one
machine-readable object and nothing else. Nothing prompts.

| Command | What it does |
|---|---|
| `admetus doctor` | Say whether this install can predict, and what to fix if not |
| `admetus endpoints` | List every endpoint with its task, metric, units and output columns |
| `admetus data fetch` | Download the public datasets and check them against pinned digests |
| `admetus data info` | Describe the fetched datasets after normalization |
| `admetus train` | Tune and train one or more endpoints |
| `admetus evaluate` | Report the held-out metrics recorded at training time |
| `admetus benchmark fetch` | Download the benchmark group's published train/test splits |
| `admetus benchmark run` | Run the official five-seed protocol on those splits |
| `admetus predict` | Add prediction columns to a CSV |
| `admetus models list` | List the model bundles present, optionally verifying them |
| `admetus models show` | Print a bundle's full record |
| `admetus models package` | Package bundles for release, with a manifest of digests |
| `admetus models pull` | Install published bundles, verifying every archive |

Every command that selects endpoints takes a repeatable `--endpoint KEY` plus `--all`.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. Rows whose prediction is empty do not change this |
| 1 | Unexpected internal error |
| 2 | Usage or input error |
| 3 | Model bundles missing or failing verification |
| 4 | Only under `--strict`, or when every row failed |
| 5 | Network or download failure, so a caller knows retrying may help |
| 130 | Interrupted |

### Environment variables

Every path and resource flag has one, because a sandboxed caller sets environment rather than
rewriting a command line: `ADMETUS_MODELS_DIR`, `ADMETUS_DATA_DIR`, `ADMETUS_CACHE_DIR`,
`ADMETUS_JOBS`.

## Output columns

Three columns per endpoint by default, appended at the end of the row.

| Column | Meaning |
|---|---|
| `<key>` | The prediction. A calibrated probability of the positive class for classification, or a value on the published scale for regression |
| `<key>_ens_sd` | Disagreement across the fold ensemble. On a log-scale endpoint this is named `<key>_sd_log10`, because a standard deviation of log values is not a standard deviation in the published unit |
| `<key>_in_domain` | Whether the molecule has a training neighbour at or above a similarity threshold. A heuristic flag, and the bundle records the similarity-to-error curve it was derived from |

One further column, `admetus_error`, carries the reason when a molecule could not be used.

`--interval` adds `<key>_lo` and `<key>_hi` for regression endpoints, and `--ad-score` adds the
nearest-neighbour similarity itself. `--column-prefix` namespaces everything, which makes a
collision with your own columns impossible.

`<key>_ens_sd` is deliberately not called a confidence. It measures how much the fold models
disagree, which is only one component of uncertainty and, for classification, is largely a function
of the prediction itself.

## Endpoints

29 endpoints. The 22 marked official are the datasets of the Therapeutics Data Commons ADMET
benchmark group.

| Area | Endpoints |
|---|---|
| Absorption | `caco2`, `pampa`, `hia`, `pgp`, `bioavailability` |
| Distribution | `lipophilicity`, `solubility`, `bbb`, `ppbr`, `vdss` |
| Metabolism | `cyp1a2_veith`, `cyp2c9_veith`, `cyp2c19_veith`, `cyp2d6_veith`, `cyp3a4_veith`, `cyp2c9_substrate`, `cyp2d6_substrate`, `cyp3a4_substrate` |
| Excretion | `half_life`, `clearance_hepatocyte`, `clearance_microsome` |
| Toxicity | `ld50`, `herg`, `herg_karim`, `ames`, `dili`, `carcinogens`, `skin_reaction`, `tox21_nr_ar` |

`admetus endpoints --json` gives the full description of each, including units, metric, the output
columns it adds, and whether a model is available locally.

Three things worth knowing before reading several columns together:

- `herg` and `herg_karim` are different curations of the same assay with substantial compound
  overlap. Their two columns are not independent evidence. For prediction prefer `herg_karim`,
  which is 20 times larger; for comparison with the published leaderboard use `herg`.
- The five CYP inhibition panels come from the same screening library, so their errors and
  domain flags are correlated.
- `tox21_nr_ar` is one assay, the androgen receptor, not general toxicity.

## Benchmark

All 29 endpoints are trained. Two columns, measuring different things:

- **Held-out** is this project's own scaffold split, looked at once, with a bootstrap
  interval. The largest scaffold groups go to training, so the held-out set is the
  unusual chemistry on purpose and the estimate is pessimistic.
- **Official protocol** is the benchmark group's published split: trained on its
  `train_val` portion, scored once per seed on its test set, mean and standard
  deviation over the protocol's five seeds. This is the column comparable with the
  published leaderboard, and it covers the 22 endpoints of the official group.
  **Taken with tuning off**, because five seeds is five tuning runs, so it is a floor
  rather than a best effort.

| Endpoint | Metric | Held-out (95% CI) | Official protocol | Notes |
|---|---|---|---|---|
| `caco2` | mae | 0.3786 [0.3344, 0.4237] | 0.3214 +/- 0.0038 |  |
| `pampa` | auroc | 0.6927 [0.6013, 0.7823] | no published split |  |
| `hia` | auroc | 0.9034 [0.7483, 0.9972] | 0.9808 +/- 0.0027 | small test set |
| `pgp` | auroc | 0.8919 [0.8462, 0.9306] | 0.9284 +/- 0.0006 |  |
| `bioavailability` | auroc | 0.6750 [0.5507, 0.7955] | 0.7259 +/- 0.0130 |  |
| `lipophilicity` | mae | 0.6633 [0.6261, 0.6990] | 0.6808 +/- 0.0008 | tuning budget-limited |
| `solubility` | mae | 1.3204 [1.2619, 1.3805] | 1.1871 +/- 0.0054 | tuning budget-limited, official run partial |
| `bbb` | auroc | 0.8731 [0.8256, 0.9135] | 0.8732 +/- 0.0019 |  |
| `ppbr` | mae | 9.1137 [7.8984, 10.3877] | 7.7789 +/- 0.0362 |  |
| `vdss` | spearman | 0.6304 [0.5308, 0.7162] | 0.6263 +/- 0.0042 |  |
| `cyp2c9_veith` | auprc | 0.7190 [0.6846, 0.7542] | 0.7228 +/- 0.0006 | tuning budget-limited |
| `cyp2d6_veith` | auprc | 0.7299 [0.6963, 0.7620] | 0.6615 +/- 0.0009 |  |
| `cyp3a4_veith` | auprc | 0.7876 [0.7625, 0.8122] | 0.8288 +/- 0.0041 | tuning budget-limited |
| `cyp1a2_veith` | auprc | 0.8661 [0.8500, 0.8820] | no published split | tuning budget-limited |
| `cyp2c19_veith` | auprc | 0.7950 [0.7711, 0.8192] | no published split | tuning budget-limited |
| `cyp2c9_substrate` | auprc | 0.2491 [0.1477, 0.3892] | 0.3627 +/- 0.0062 | interval reaches the base rate |
| `cyp2d6_substrate` | auprc | 0.5038 [0.3780, 0.6707] | 0.6895 +/- 0.0114 |  |
| `cyp3a4_substrate` | auroc | 0.6507 [0.5529, 0.7409] | 0.6537 +/- 0.0061 |  |
| `half_life` | spearman | 0.2273 [0.0576, 0.3876] | 0.5551 +/- 0.0062 |  |
| `clearance_hepatocyte` | spearman | 0.3804 [0.2533, 0.5000] | 0.4009 +/- 0.0041 |  |
| `clearance_microsome` | spearman | 0.3869 [0.2671, 0.4913] | 0.5928 +/- 0.0093 |  |
| `ld50` | mae | 0.5243 [0.4990, 0.5489] | 0.7075 +/- 0.0011 | tuning budget-limited |
| `herg` | auroc | 0.8872 [0.8238, 0.9386] | 0.8454 +/- 0.0015 |  |
| `herg_karim` | auroc | 0.8534 [0.8389, 0.8672] | no published split | tuning budget-limited |
| `ames` | auroc | 0.8025 [0.7796, 0.8244] | 0.7745 +/- 0.0009 |  |
| `dili` | auroc | 0.8342 [0.7518, 0.9119] | 0.9045 +/- 0.0035 | small test set |
| `carcinogens` | auprc | 0.8101 [0.6002, 0.9453] | no published split | small test set |
| `skin_reaction` | auroc | 0.6245 [0.4951, 0.7485] | no published split | small test set, interval reaches no signal |
| `tox21_nr_ar` | auprc | 0.5079 [0.3900, 0.6164] | no published split |  |

Read before quoting a number:

- **The standard deviation over seeds is not comparable with the leaderboard's.** Here
  the seed moves only the fold assignment inside the training portion; `train_val` is
  not resampled into train and valid. The mean is comparable, the spread is not.
- **Which split is harder depends on the endpoint, in both directions.** The official
  split is easier at `half_life`, `clearance_microsome` and `cyp2d6_substrate`, and
  harder at `ld50` and `vdss`.
- **Eight endpoints are budget-limited.** At a 900-second tuner budget they completed
  14 to 19 trials against 65 to 68 on the small ones, so those are floors for this
  architecture rather than its ceiling.
- **Three intervals reach no signal**, and that is the result rather than a gap in it:
  `half_life`, `skin_reaction` and `cyp2c9_substrate`.
- **The cross-validation score flattered the model at 26 of 29 endpoints**, by a median
  of 0.053. It selects; it does not report.

Prediction intervals over-cover at eight of the nine regression endpoints, median
97.0% against a nominal 90%. Every bundle records its measured coverage. One official
result is marked partial: 1,995 of solubility's 1,997 published test rows could be
featurized, so its mean is over a subset.

Full tables, the feature-set study behind the per-endpoint choices, and the
engineering measurements are in [docs/benchmark.md](docs/benchmark.md).

## Configuration

Dataset-dependent choices are computed rather than configured, so they do not have to be
maintained in 29 places: which feature set to use, how aggressively to prune columns, and whether a
dataset is too small for its metric to be read as a point estimate all follow from the number of
rows.

Three feature sets exist, and `--feature-set` overrides the choice.

| Feature set | Content | Width |
|---|---|---|
| `morgan` | Morgan radius 2, 2048 bits. The field-standard baseline | 2048 |
| `morgan_p2` | Morgan radius 6 plus the 2-point pharmacophore signature | 2372 |
| `morgan_p23` | Morgan radius 6 plus the 2- and 3-point signature | 89852 |

`morgan_p23` is the default above 1000 training rows. Below that the full space is overkill and
`morgan_p2` is used, which is also 10 to 30 times cheaper to compute.

## Limitations

- **No published models yet.** Train locally.
- **2D only.** No conformer generation, so stereochemistry is invisible to the model. Enantiomers
  receive identical predictions, and the training pipeline treats them as duplicates deliberately
  rather than pretending otherwise.
- **`<key>_in_domain` is a heuristic**, not a guarantee. It says a similar training molecule
  exists, which the bundle relates to expected error; it does not say the prediction is right.
- **The cross-validation score is biased** and is recorded as such. Only the held-out numbers are
  reported.
- **Small datasets.** Four endpoints have fewer than 600 compounds, so their test sets are under
  120 molecules and their metrics have wide intervals. They are flagged `low_confidence`.
- **The automatic tuner owns the search space.** Its `feature_fraction` grid starts at 0.4, which
  is high for a very sparse binary matrix, and it disables LightGBM's own rare-column filter. The
  explicit column pruning compensates and is recorded per endpoint.

## Development

```bash
uv sync --all-extras
```

```bash
uv run ruff check . && uv run pytest
```

The test suite needs no network: datasets are fetched rather than vendored, and the tests use
fixtures. Tests marked `slow` train small models and exercise the tuner.

```bash
uv run pytest -m 'not slow'
```

```bash
python scripts/check_readme_pair.py
```

## References

Data comes from Therapeutics Data Commons, retrieved from Harvard Dataverse. Dataset sources,
citations and licenses are listed in [docs/data-sources.md](docs/data-sources.md); most are
CC BY 4.0.

- Huang et al., *Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery
  and Development*, NeurIPS 2021. [arXiv:2102.09548](https://arxiv.org/abs/2102.09548)
- Huang et al., *Artificial intelligence foundation for therapeutic science*, Nature Chemical
  Biology, 2022. [doi:10.1038/s41589-022-01131-2](https://doi.org/10.1038/s41589-022-01131-2)
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
  [arXiv:1907.10902](https://arxiv.org/abs/1907.10902)
- Bemis and Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, Journal of Medicinal
  Chemistry, 1996.
- Rogers and Hahn, *Extended-Connectivity Fingerprints*, Journal of Chemical Information and
  Modeling, 2010.
- [RDKit](https://www.rdkit.org/) for cheminformatics, including the 2D pharmacophore signatures.

## License

Apache-2.0. See [LICENSE](LICENSE). The datasets carry their own licenses, listed in
[docs/data-sources.md](docs/data-sources.md).
