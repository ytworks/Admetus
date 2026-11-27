# Admetus

Admetus predicts ADMET properties from SMILES. One LightGBM fold ensemble per public dataset,
over a Morgan fingerprint concatenated with a 2D pharmacophore signature, served through a CLI
built for programs. Package: `src/admetus/`.

## Session start

ALWAYS read `PROGRESS.md` first. It is the cross-session memory: current state, what to do next,
the decisions that were settled by measurement, and the refusals that are deliberate.

Then read `docs/benchmark.md` before making any performance claim. Several figures in it
contradict plausible-sounding estimates, and the measurements are what the design rests on.

## Where things live

- `src/admetus/chem/` — the feature space. `spec.py` defines what a feature vector *is* and
  digests it; `standardize.py`, `morgan.py`, `pharm2d.py`, `featurizer.py` build it;
  `similarity.py` does packed-bit Tanimoto; `parallel.py` runs the spawn pool.
- `src/admetus/data/` — `endpoints.yaml` is the registry and the source of truth for dataset
  digests. `download.py`, `loader.py`, `audit.py` (feature-level leakage), `splits.py` (scaffold
  grouping and folds).
- `src/admetus/train/` — `metrics.py`, `transforms.py`, `prune.py`, `cv.py`, `objectives.py`,
  `tuning.py` (the automatic tuner), `calibrate.py`, `conformal.py`, `trainer.py` (orchestration).
- `src/admetus/bundle/` — persistence, with verification that refuses a mismatched column space.
- `src/admetus/predict/` — `csvio.py` holds the byte-preserving CSV contract; `runner.py` drives it.
- `src/admetus/cli/` — one module per command.

## Rules specific to this project

1. **Never report a number you have not measured.** Every figure in the README's benchmark
   section and in `docs/benchmark.md` came from a run that is reproducible by the script named
   beside it. Regenerate the accuracy tables with `scripts/benchmark_tables.py` rather than
   editing a cell; both READMEs carry the same numbers and CI compares every numeric cell.
   Do not fill a gap with an estimate, and do not leave placeholders.
2. **Cross-validation scores are not performance.** They select; they do not report. Every record
   carries `cv_score_biased: true`, and it is measured: out-of-fold flattered the model at 26 of
   the 29 endpoints, by a median of 0.045. Headline numbers come from the held-out set, with a
   bootstrap interval, or from the official protocol.
3. **Never substitute a molecule.** A parse failure is reported with a reason. There is a test
   asserting no substitution happens; do not relax it.
4. **The held-out set is sacred.** It is looked at once, at training time. Leakage-audit
   collisions drop training rows, never held-out rows.
5. **Passthrough CSV bytes are not yours to reformat.** Do not route the prediction path through
   pandas. `csvio.py` explains why, with the list of things that break.
6. **The registry is the source of truth for dataset digests.** A mismatch is refused.
   `checksums.json` is an observed-state record, nothing more.
7. **Both READMEs change together.** `scripts/check_readme_pair.py` compares the heading outline,
   the commands in code blocks, the links and the numbers in tables. CI runs it.
8. **Honesty in docstrings.** Where a guarantee is partial, say so. The digest pinning says it is
   still trust on first use; `in_domain` says it is a heuristic; the tuner's record names the two
   costs of delegating the search space.

## Test policy

- `pytest` runs everything; `-m 'not slow'` skips the tests that train models or run the tuner.
- No test may touch the network. Datasets are fetched, not vendored, and tests use fixtures.
- A test asserting current behaviour is a contract. If a change makes one fail, decide whether the
  behaviour or the test is wrong, and say which in the commit message.
- Fixtures for training need a label that is a function of the structure but **not** of the
  scaffold. A random label makes the leakage audit discard the training set as contradictory; a
  scaffold-determined label makes grouped stratification impossible. Both were found by the code
  correctly refusing a bad fixture.

## Style

Per the user's global rules: immutable data, small focused files (200-400 lines, 800 maximum),
explicit error handling, no silently swallowed exceptions, named constants rather than magic
numbers. `ruff` with a 120-column line length. Commit messages follow
`<type>: <description>`.
