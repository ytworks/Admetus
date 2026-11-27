# Progress

Cross-session memory for Admetus: what exists, what is measured, what is not done,
and what to pick up next. Read this first, then `CLAUDE.md`.

Last updated: 2026-11-27. Branch `feat/core-featurizer-and-data`, draft PR #1.
237 tests pass locally and in CI on Ubuntu and macOS for Python 3.12 and 3.13.

## State

The pipeline now follows the reference implementation (`models.py` and the
`Carcinogens_Lagunin` notebook) rather than extending it:

| Stage | What it does |
|---|---|
| Featurization | Morgan r=6, 2,048 bits, concatenated with the 2- and 3-point pharmacophore signature. One space for every endpoint |
| Holdout | Random 20%, seeded, via `train_test_split`. Looked at once |
| Folds | `KFold` over the remaining 80%, default 3, `--folds` to raise it |
| Ensemble | One model per fold, each early-stopped on its own validation rows. Those models ship; there is no refit |
| Output | The prediction and the spread across the fold models, which is the reference's `Confidence` |
| Kept beyond the reference | Probability calibration and prevalence-based column pruning, both measured harmless and the second needed because the tuner disables LightGBM's own rare-column guard |

**No accuracy numbers are recorded right now.** Everything measured before described a
scaffold-grouped split that has been removed, so the tables were deleted rather than
relabelled. All 29 endpoints need retraining, and `scripts/benchmark_tables.py`
regenerates the tables from the bundles.

216 tests pass, `ruff check .` is clean, and the README pair check passes.

## What was removed, and why

Each of these was built here, not asked for, and measured before being removed:

- **Applicability-domain flag** (`in_domain`, `ad_score`, reference set, packed
  Tanimoto): uninformative at 16 of 29 endpoints. `dili` reported water as
  in-domain, because a flat similarity-to-error curve collapsed its threshold to
  0.0139.
- **Conformal prediction intervals**: over-covered at 8 of 9 regression endpoints,
  median 97.0% against a nominal 90%. Aspirin's Caco-2 interval was
  [-17.44, 8.18] on labels spanning -7.76 to -3.51.
- **Scaffold-grouped splitting and the feature-level leakage audit**: the reference
  splits at random. The scaffold machinery also produced a real leak -- grouping by
  input string rather than standardized structure put 9 of 129 held-out hERG
  molecules in a training scaffold -- and cost two full retrains to fix.
- **Feature-set presets, the size rule and three per-endpoint overrides**, with the
  174-run study that chose them. The reference featurizes everything identically.
- **The refit at a pooled round count**: the reference ships the fold models.

The lesson worth keeping: of 25 review findings across three rounds, **none were in
the core featurize-train-predict path**. All of them were in these added layers or in
the CLI and release machinery built to support them.

## Decisions worth not relitigating

Measured, and still true after the simplification. `docs/benchmark.md` has the numbers.

- **Featurization is fast enough.** 3.73 ms per molecule, 848 molecules per second on
  nine workers, about two minutes per 100,000 unique molecules. An early estimate of
  28 minutes was wrong by roughly 14 times, so the vendored fast pharmacophore path
  that estimate justified was never written.
- **Mean non-zeros per molecule is 426**, not the 2,500 assumed.
- **scipy's column indexing beats a hand-rolled reindex** (11.5 ms against 20.7 ms),
  so `apply_basis` uses it.
- **`Chem.MurckoDecompose` and `rdScaffoldNetwork` are broken** in RDKit 2026.09.2,
  raising `RuntimeError: bad any cast`. No longer load-bearing, since scaffolds are
  not used, but worth knowing before anyone reaches for them.
- **Isotonic calibration is not free in ranking terms.** Its ties moved average
  precision from 0.6922 to 0.6348, which is why the calibrator is chosen by
  measurement and one costing more than a small tolerance is rejected.
- **The automatic tuner owns the search space**, by instruction. It sets
  `feature_pre_filter` to `False`, which is why the explicit prevalence pruning is
  load-bearing rather than decorative.
- **The pruning band is the only size-dependent choice left.** It reads a row count;
  nothing else does.

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
- A dataset whose digest does not match the registry is refused, and a cached file
  that fails is moved aside so nothing reads it.
- A model bundle whose column space or golden fingerprint disagrees with the
  install is refused rather than used.
- Cross-validation scores are recorded as biased and are never reported as
  performance.
- Folds too thin to score are refused loudly rather than tuned against noise.
- A manifest's endpoint key must be one the registry publishes and must not name a
  directory the project keeps for itself.
- A booster that cannot be loaded, or whose width disagrees with the bundle's basis,
  stops an install before it replaces a working model.

## Verified on real data

- `dili` under the new scheme: 3 models, early-stopped at 25, 309 and 256 rounds,
  held-out AUROC 0.8665 on the random 20%.
- All 29 datasets fetch, normalize to the row counts pinned in the registry, and
  verify against their digests.
- The published split archive fetches, verifies and unpacks all 22 official endpoints.
- `models package` then `models pull` round-trips a bundle through a manifest, and
  packaging the same bundle twice gives byte-identical archives.
- The CSV round trip is byte-identical through a file carrying a byte-order mark,
  duplicate column names, CRLF, an embedded newline inside a quoted field, escaped
  quotes, cp932 text and no trailing newline.
