# Progress

Cross-session memory for Admetus: what exists, what is measured, what is not done,
and what to pick up next. Read this first, then `CLAUDE.md`.

Last updated: 2026-11-27. Branch `feat/core-featurizer-and-data`, draft PR #1.
217 tests pass locally and in CI on Ubuntu and macOS for Python 3.12 and 3.13.

## State

Every layer is implemented and tested. 217 tests pass, `ruff check .` is clean, and
the full path works end to end on real data: fetch, train with automatic tuning,
save, load, predict into a CSV. The official protocol and the release round trip
are both implemented and exercised.

**The accuracy tables are the remaining work.** The feature set per endpoint is
now decided by measurement rather than by the size rule alone, and training all 29
endpoints is what fills in the numbers. Nothing is estimated in the meantime.

| Layer | State |
|---|---|
| `chem/` featurizer, standardization, similarity, parallel | done, tested |
| `data/` registry, download with pinned digests, loader, leakage audit, splits | done, tested |
| `train/` metrics, transforms, pruning, folds, tuning, calibration, conformal, trainer | done, tested |
| `bundle/` save and load with spec verification | done, tested |
| `predict/` byte-preserving CSV, runner | done, tested |
| `cli/` doctor, endpoints, data, train, evaluate, benchmark, predict, models | done |
| Feature set per endpoint, by measurement | done: 174 runs, 3 overrides, in `docs/benchmark.md` |
| Official five-seed protocol on the published splits | done: `admetus benchmark fetch` and `run` |
| Release packaging and install | done: `admetus models package` and `pull`, with a manifest of digests |
| CI and Docker | green on the branch: lint, the README pair check, the Ubuntu and macOS test matrices for Python 3.12 and 3.13, and the Docker image with its smoke tests |
| README in English and Japanese | done, pair check passes |
| Trained models and accuracy numbers | **not done** |

## What to do next

1. **Finish training and fill in the numbers.** `admetus data fetch --all` then
   `admetus train --all --time-budget 900`. Expect hours: the tuner runs 65 to 68
   trials per endpoint and the largest datasets are 13,445 rows. `train --all`
   skips endpoints that already have a valid bundle, so it resumes, and three
   `--endpoint` runs in parallel with `--jobs 4` keep a 12-core machine busy.
   Then regenerate the tables with `python scripts/benchmark_tables.py` and paste
   them into the benchmark section of **both** READMEs. Generated, not transcribed:
   CI compares every numeric cell of the two documents.
2. **Run the official protocol with tuning on.** `admetus benchmark run --all`
   works and the splits are pinned, but five seeds per endpoint means five tuning
   runs, so the numbers recorded first were taken with `--no-tune`. They are
   labelled as a floor, not a best effort. Re-running with a budget is the
   remaining accuracy work after item 1.
3. **Publish a release.** `admetus models package --out dist/models` writes the
   archives and the manifest, and `.github/workflows/release-models.yml` verifies
   that a tagged release installs from scratch. Nothing has been uploaded yet, so
   `admetus models pull` reports a download failure until a release exists. That
   step is deliberately a person's decision, not a script's: it publishes bytes
   under the project's name.

## Decisions worth not relitigating

Each of these was measured, and several contradict what the plan originally said.
`docs/benchmark.md` has the numbers.

- **Featurization is fast enough.** 3.73 ms per molecule, 848 molecules per second
  on nine workers, about two minutes per 100,000 unique molecules. An early
  estimate of 28 minutes was wrong by roughly 14 times. The vendored fast
  pharmacophore path that estimate justified was therefore **not** written; the
  profile that argues against it is recorded.
- **Mean non-zeros per molecule is 426**, not the 2,500 assumed, so memory and
  inter-process traffic are about six times smaller than planned for.
- **scipy's column indexing beats a hand-rolled reindex** (11.5 ms against
  20.7 ms on a 10,000 by 89,852 matrix), so `apply_basis` uses it and there is no
  shared intermediate basis. Serving all 29 endpoints from one raw matrix costs
  about 330 ms per chunk.
- **`Chem.MurckoDecompose` and `rdScaffoldNetwork` are broken** in RDKit 2026.09.2,
  raising `RuntimeError: bad any cast` for every molecule. The Bemis-Murcko
  framework reduction is implemented directly in `data/splits.py` and asserted
  against known scaffolds. Revisit if a later RDKit fixes it.
- **Isotonic calibration is not free in ranking terms.** Its step function creates
  ties: measured, it moved average precision from 0.6922 to 0.6348. Calibrators
  are therefore chosen by measurement, and one that would cost more than a small
  tolerance of the endpoint's own metric is rejected.
- **LightGBM 4.7.0 does not raise** when `min_data_in_leaf` changes on a reused
  `Dataset`, and reuse gives bit-identical predictions. The per-trial rebuild the
  plan called for is unnecessary. Moot on the tuning path anyway, since the tuner
  manages `feature_pre_filter` itself.
- **Most of the pharmacophore value is in the 2-point block.** Measured over all 29
  endpoints: the 324-bit 2-point block closes most of the gap from plain Morgan, and
  the 87,480-bit 3-point block adds a further 0.01 or less almost everywhere. The
  size rule survived at 26 of 29 endpoints; the three overrides are in
  `endpoints.yaml` with their margins. `tox21_nr_ar` is the one endpoint where the
  pharmacophore blocks measurably **hurt** -- 0.5074 against 0.4834 average
  precision -- because at 4.3% positive its 310 actives cannot pay for 89,852
  columns.
- **A feature set is chosen on the cross-validation score, never the held-out set.**
  Choosing against held-out data would turn the reported number into a selection
  score. The study therefore ranks on pooled out-of-fold predictions and runs two
  seeds, so a challenger has to beat the default by more than the fold noise.
- **The automatic tuner owns the search space**, by instruction. Two costs are
  real and recorded in every bundle: its `feature_fraction` grid starts at 0.4,
  and it sets `feature_pre_filter` to `False`. The explicit prevalence pruning is
  consequently the only rare-column guard, so do not thin it out.

## CI notes

- `admetus doctor` exits **3** when the install is sound but no model bundle is
  present, which is the normal state of a fresh checkout. The CI self-check
  accepts 0 and 3 and fails on anything else, since 1 means a broken install.
  A naive `run: admetus doctor` fails the build.
- A runner label that cannot be scheduled cancels its jobs before any step runs,
  with no log. That looks nothing like a test failure; `macos-14` did this and
  `macos-latest` fixed it.
- `concurrency: cancel-in-progress: true` cancels the previous run on every
  push, so a run marked cancelled after a quick follow-up push is expected.
- Remediation strings must name commands that exist. `doctor` and the bundle
  loader used to suggest `admetus models pull`, which is not implemented;
  pointing an automated caller at a missing command is worse than saying nothing.

## Things the code refuses to do, on purpose

Changing any of these is a behaviour change, not a cleanup.

- A molecule that fails to parse is never replaced by another one. It gets empty
  prediction cells and a reason.
- A leakage-audit collision drops the **training** row, never the held-out one, so
  the metric stays measured on the same compounds as the published benchmark.
- A dataset whose digest does not match the registry is refused, and a cached file
  that fails is moved aside so nothing reads it.
- A model bundle whose column space or golden fingerprint disagrees with the
  install is refused rather than used.
- Cross-validation scores are recorded as biased and are never reported as
  performance.
- Scaffold-grouped folds too thin to score are refused loudly rather than tuned
  against noise.

## Verified by hand on real data

- The published split archive fetches, verifies against its pin and unpacks all 22
  official endpoints; `dili` on the published split scores AUROC 0.9037 +/- 0.0046
  over two untuned seeds, against a leaderboard best near 0.925.
- `models package` then `models pull` round-trips a bundle through a manifest, and
  packaging the same bundle twice gives byte-identical archives.

- DILI: 68 tuner trials, held-out AUROC 0.8338 [0.7491, 0.9133] on 95 compounds.
- VDss: held-out Spearman 0.6037 [0.5080, 0.6818], and the conformal interval
  covered 93.2% of held-out labels at a nominal 90%.
- All 29 datasets fetch, normalize to the row counts pinned in the registry, and
  verify against their digests.
- The leakage audit finds 78 indistinguishable groups in `herg`, 5 of which span
  the published split. SMILES deduplication does not catch these.
- The CSV round trip is byte-identical through a file carrying a byte-order mark,
  duplicate column names, CRLF, an embedded newline in a quoted field, escaped
  quotes, cp932 text and no trailing newline.
