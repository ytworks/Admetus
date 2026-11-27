# Progress

Cross-session memory for Admetus: what exists, what is measured, what is not done,
and what to pick up next. Read this first, then `CLAUDE.md`.

Last updated: 2026-11-27. Branch `feat/core-featurizer-and-data`, draft PR #1.

## State

Every layer is implemented and tested. 186 tests pass, `ruff check .` is clean, and
the full path works end to end on real data: fetch, train with automatic tuning,
save, load, predict into a CSV.

**No models are published yet, and no accuracy table is filled in.** That is the
main remaining work, and it is deliberate that the README's benchmark section is
empty rather than estimated.

| Layer | State |
|---|---|
| `chem/` featurizer, standardization, similarity, parallel | done, tested |
| `data/` registry, download with pinned digests, loader, leakage audit, splits | done, tested |
| `train/` metrics, transforms, pruning, folds, tuning, calibration, conformal, trainer | done, tested |
| `bundle/` save and load with spec verification | done, tested |
| `predict/` byte-preserving CSV, runner | done, tested |
| `cli/` doctor, endpoints, data, train, evaluate, predict, models | done |
| CI and Docker | written, not yet exercised (no Docker on the development machine) |
| README in English and Japanese | done, pair check passes |
| Trained models and accuracy numbers | **not done** |

## What to do next

1. **Train everything and fill in the numbers.** `admetus data fetch --all` then
   `admetus train --all --time-budget 900`. Expect hours: the tuner runs 65 to 68
   trials per endpoint and the largest datasets are 13,445 rows. `train --all`
   skips endpoints that already have a valid bundle, so it resumes. Then put the
   held-out metrics with their intervals into the README benchmark section of
   **both** languages, and into `docs/benchmark.md`.
2. **Decide the feature set per endpoint by measurement.** Section 1.1 of the plan
   calls for running all three feature sets at a reduced budget and recording the
   result, rather than assuming `morgan_p23` is right everywhere. Not started.
3. **The official five-seed protocol.** `admetus evaluate` currently reports the
   metrics recorded at training time on our own scaffold split. The benchmark
   group's own splits are published as one archive (`admet_group`, Dataverse id
   4426004, a 1.4 MB zip readable with `zipfile`), and
   `download.extract_benchmark_archive` exists but nothing uses it yet. Without
   this, our numbers are not comparable with the published leaderboard.
4. **Publish model bundles.** The release side of section 7 is unwritten: no
   `models pull`, no manifest, no release workflow for bundles. `admetus models`
   currently only lists and shows what is local.
5. **Exercise CI and the Docker build.** Both are written but unverified, because
   this machine has no Docker daemon. The first CI run on the PR will tell.

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
- **The automatic tuner owns the search space**, by instruction. Two costs are
  real and recorded in every bundle: its `feature_fraction` grid starts at 0.4,
  and it sets `feature_pre_filter` to `False`. The explicit prevalence pruning is
  consequently the only rare-column guard, so do not thin it out.

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
