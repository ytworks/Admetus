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
- **Automatic hyperparameter tuning**, over the same folds the model is
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
3. **Hold out 20% at random**, seeded, and split the rest into folds. One model per fold,
   each stopping early on its own validation rows; the ensemble is their mean and the spread
   across them is reported beside the prediction.
4. **Tune and fit each fold on its own.** The automatic tuner is run once per fold, with that
   fold's validation rows as the watchlist, and the booster it settles on is the ensemble member.
   So the folds end up with different parameters, and the trial count is per fold.
5. **Measure once.** The held-out 20% is scored exactly once, with the endpoint's own metric and
   a bootstrap interval.

Nothing else happens to the data on the way. There is no standardization, no duplicate
collapsing, no label transform, no output clamp, no column pruning and no probability
calibration: the label is the published one, and LightGBM gets the whole 89,852-column vector.
Each of those was tried here and removed -- several of them measurably harmful, one of them
badly so. `docs/benchmark.md` records what each cost.

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
`admetus evaluate` reports this project's own random holdout instead.

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
| `<key>` | The prediction: the mean across the fold models. LightGBM's own probability for classification, **not** calibrated, or a value on the published scale for regression |
| `<key>_ens_sd` | Disagreement across the fold ensemble. On a log-scale endpoint this is named `<key>_sd_log10`, because a standard deviation of log values is not a standard deviation in the published unit |

One further column, `admetus_error`, carries the reason when a molecule could not be used.

`--column-prefix` namespaces everything, which makes a collision with your own columns
impossible.

An applicability-domain flag and a conformal prediction interval used to be here. Both were
removed: measured, the flag carried no information at 16 of the 29 endpoints -- `dili` reported
water as in-domain -- and the interval over-covered at 8 of 9 regression endpoints by enough to
be unusable. Neither was asked for, and a column that looks like a safeguard without being one
is worse than no column.

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

**No accuracy numbers here right now, on purpose.** The split changed: a scaffold-grouped
holdout was replaced by the reference implementation's random 20% holdout with folds over the
rest, so every number measured before it describes a pipeline this code no longer is. They were
deleted rather than relabelled, and all 29 endpoints are being retrained.

What will go here, once that finishes:

- **Held-out**, on the random 20%, with a bootstrap interval.
- **Official protocol**, on the benchmark group's published split, mean and standard deviation
  over its five seeds, for the 22 endpoints that have one.

Regenerate both with `python scripts/benchmark_tables.py`, which reads the bundles rather than
being typed into: the two READMEs have to carry the same numbers and CI compares every numeric
cell.

The feature-set study behind the per-endpoint choices and the engineering measurements are in
[docs/benchmark.md](docs/benchmark.md) and are unaffected by the split change, except where they
quote accuracy.

## Configuration

One feature space, for every endpoint: Morgan radius 6 over 2,048 bits concatenated with the
2- and 3-point pharmacophore signature, 89,852 columns. There is nothing to choose and no flag
to choose it with.

Two things still follow from dataset size, and neither touches the feature space: how many folds
the training portion can support, and whether a dataset is too small for its metric to be read as
a point estimate (`low_confidence`).

| Flag | Default | What it changes |
|---|---|---|
| `--folds` | 3 | Folds over the training portion, and so ensemble members |
| `--time-budget` | 900 | Wall-clock cap for tuning **one fold**, in seconds |
| `--seed` | 0 | The holdout partition and the tuner's sampler |
| `--jobs` | 0 | LightGBM threads; 0 lets LightGBM decide |

## Limitations

- **No published models yet.** Train locally.
- **2D only.** No conformer generation, so stereochemistry is invisible to the model. Enantiomers
  receive identical predictions.
- **The held-out score is a held-out-row score, not an unseen-structure score.** Rows are split as
  published, duplicates and all, as the reference implementation does. So a molecule can appear on
  both sides: on Caco-2 at seed 0, **26 of the 182 test rows have a feature vector identical to a
  training row**. Read the number as "how well it does on rows it was not fitted on", which is
  what the reference measures, and not as "how well it does on chemistry it has never seen".
- **No applicability-domain flag.** There was one; measured, it carried no information at 16 of
  the 29 endpoints, so it was removed rather than left looking like a safeguard. Nothing in the
  output says whether a molecule resembles the training set, so a caller working outside
  drug-like chemistry has to judge that themselves.
- **No prediction intervals.** There were conformal ones; they over-covered at 8 of 9 regression
  endpoints by enough to be useless. `<key>_ens_sd` is what is left, and it is a disagreement
  measure, not an error bar.
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
