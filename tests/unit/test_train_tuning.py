"""The automatic tuner path.

Tuning is delegated to Optuna's stepwise LightGBM tuner, so what is worth testing
is the integration: that our sparse matrices and the folds over the training
portion go through it, that the metric we align per endpoint survives, that the per-fold
boosters come back, and that the study resumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from admetus.data.splits import Split
from admetus.train import tuning


def toy(n_rows: int = 160, n_cols: int = 300, seed: int = 0, classification: bool = True):
    """A small sparse problem with real signal, so tuning has something to find."""
    rng = np.random.default_rng(seed)
    rows = [np.unique(rng.choice(n_cols, size=25, replace=False)).astype(np.int32) for _ in range(n_rows)]
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    matrix = sparse.csr_matrix(
        (np.ones(indptr[-1], np.float32), np.concatenate(rows), indptr), shape=(n_rows, n_cols)
    )
    signal = np.asarray(matrix[:, :5].sum(axis=1)).ravel()
    y = (
        (signal >= np.median(signal)).astype(float)
        if classification
        else signal + rng.normal(0, 0.2, n_rows)
    )
    return matrix, y


def folds_of(n_rows: int, k: int = 3) -> list[Split]:
    """Contiguous folds, as ``KFold`` without shuffling cuts them."""
    bounds = np.array_split(np.arange(n_rows), k)
    return [
        Split(np.sort(np.setdiff1d(np.arange(n_rows), block)), np.sort(block)) for block in bounds
    ]


def test_the_import_path_is_the_supported_one():
    """The old alias emits a FutureWarning; this project must not depend on it."""
    import optuna_integration.lightgbm as tuner_module

    assert hasattr(tuner_module, "LightGBMTunerCV")
    # Check the import statements, not the prose: the module docstring mentions the
    # deprecated alias deliberately, to say why it is not used.
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path("src/admetus/train/tuning.py").read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "optuna_integration.lightgbm" in imported
    assert "optuna.integration.lightgbm" not in imported


@pytest.mark.parametrize(
    ("task", "metric", "expected_objective"),
    [
        ("classification", "auroc", "binary"),
        ("classification", "auprc", "binary"),
        ("regression", "mae", "regression"),
        ("regression", "spearman", "regression"),
    ],
)
def test_the_objective_and_metric_are_the_reference_implementation_s(task, metric, expected_objective):
    """Four keys, and the endpoint's own metric is not among them.

    These used to be aligned with the metric the endpoint is judged on -- `auc`,
    `average_precision`, `l1` -- on the argument that a 0/1 error rate at a fixed
    0.5 threshold measures something nobody reports. True, and not what was asked
    for: the reference tunes against `binary_error` and `rmse`, so that is what the
    tuner optimizes. The endpoint's metric still decides what is reported.
    """
    params = tuning.fixed_params(task, metric)
    assert params["objective"] == expected_objective
    assert params["metric"] == ("binary_error" if task == "classification" else "rmse")
    assert params["boosting_type"] == "gbdt"
    assert params["verbosity"] == -1


def test_nothing_is_added_beyond_the_reference_except_knobs():
    """What is set has to be the reference's, plus things that cannot change a model."""
    params = tuning.fixed_params("classification", "auroc", seed=7)

    assert params["seed"] == 7
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    # Exact for a 0/1 feature space rather than an approximation of it.
    assert params["max_bin"] == 2
    # Each of these was an unrequested addition and is gone.
    for absent in ("learning_rate", "scale_pos_weight", "bagging_fraction", "feature_pre_filter"):
        assert absent not in params


def test_study_path_sits_beside_the_bundles_not_inside_one(tmp_path):
    """This asserted the old location, which was the defect: `save` replaces the
    endpoint directory atomically, so a study kept inside it was deleted by the
    successful run that built it."""
    assert tuning.study_path(tmp_path, "dili") == tmp_path / "studies" / "dili.sqlite"


def tune_one(matrix, y, params, tmp_path, name="t", seed=0, budget=20):
    """Tune the first of three folds, which is how the trainer calls it."""
    fold = folds_of(matrix.shape[0])[0]
    return tuning.tune_fold(
        matrix[fold.train], y[fold.train], matrix[fold.test], y[fold.test],
        params, seed=seed, time_budget=budget,
        storage=tmp_path / f"{name}.sqlite", study_name=name,
    )


@pytest.mark.slow
def test_the_tuner_fits_one_fold_and_returns_its_model(tmp_path):
    """One call per fold, and the booster it returns is the ensemble member.

    This replaced a single run across all folds, which tuned one parameter set
    against the pooled fold score and then needed a separate fit. The reference
    implementation calls the tuner once per fold, so the folds end up with different
    parameters and the trial count is per fold.
    """
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    result = tune_one(matrix, y, params, tmp_path)

    assert result.n_trials > 0
    assert np.isfinite(result.score)
    assert len(result.boosters) == 1
    assert result.boosters[0].predict(matrix[:5]).shape == (5,)
    # Early-stopped on its own validation rows, so it carries a best iteration.
    assert result.boosters[0].best_iteration >= 1


@pytest.mark.slow
def test_the_tuner_leaves_our_fixed_parameters_alone(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auprc", num_threads=2)
    result = tune_one(matrix, y, params, tmp_path, name="s")
    assert result.params["metric"] == "binary_error"
    assert result.params["objective"] == "binary"
    assert result.params["deterministic"] is True
    # It does choose these, and the record should say so rather than claiming
    # credit for everything in the merged set.
    assert set(result.tuned_keys) <= {
        "feature_fraction", "num_leaves", "bagging_fraction", "bagging_freq",
        "lambda_l1", "lambda_l2", "min_child_samples", "feature_pre_filter",
    }


@pytest.mark.slow
def test_the_study_resumes_from_sqlite(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    first = tune_one(matrix, y, params, tmp_path, name="resume", budget=10)
    second = tune_one(matrix, y, params, tmp_path, name="resume", budget=10)
    assert second.n_trials >= first.n_trials
    assert (tmp_path / "resume.sqlite").is_file()


@pytest.mark.slow
def test_a_seed_makes_the_run_reproducible(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    runs = [
        tune_one(matrix, y, params, tmp_path, name=f"seeded_{i}", seed=11, budget=15)
        for i in range(2)
    ]
    assert runs[0].score == runs[1].score
    assert runs[0].params["num_leaves"] == runs[1].params["num_leaves"]
    assert runs[0].seed == runs[1].seed == 11


def test_the_record_states_what_the_tuner_owns_and_that_cv_is_biased():
    result = tuning.TuneResult(
        params={"num_leaves": 7, "num_threads": 4}, score=0.8, n_trials=68,
        seed=0, time_budget=900, tuned_keys=["num_leaves"],
    )
    record = result.to_dict()
    assert record["cv_score_biased"] is True
    assert record["search_space_owner"] == "tuner"
    # The two measured costs of delegating the search space, written down.
    assert record["feature_fraction_grid_floor"] == 0.4
    assert "feature_pre_filter" not in record
    assert record["tuned_by_tuner"] == ["num_leaves"]
    # Thread count is a machine detail, not a hyperparameter.
    assert "num_threads" not in record["best_params"]


def test_untuned_defaults_respect_the_tuner_grid_floor():
    """Without tuning we still pick a feature_fraction the tuner could have chosen."""
    params = tuning.default_params("classification", "auroc", 400)
    assert params["feature_fraction"] >= 0.4
    # No learning rate is set any more: the reference leaves it to LightGBM.
    assert "learning_rate" not in params


def test_the_study_lives_outside_the_bundle_that_gets_replaced():
    """`save` replaces the endpoint directory atomically, study and all.

    A real `skin_reaction` run recorded 61 completed trials and left no database
    behind, because the study sat inside the directory `save` swaps out. Raising the
    budget with `--force` then repeated the search from scratch.
    """
    from pathlib import Path

    from admetus.train.tuning import study_path

    models = Path("/models")
    path = study_path(models, "dili")
    assert models / "dili" not in path.parents
    assert path.parent == models / "studies"


def test_a_study_is_named_after_the_settings_its_trials_depend_on():
    """Trials only transfer while the search is over the same thing."""
    from admetus.train.tuning import fixed_params, study_name

    auroc = fixed_params("classification", "auroc")
    # Both classification endpoints now tune against binary_error, so the objective
    # no longer distinguishes them; a regression endpoint does.
    other = fixed_params("regression", "rmse")

    base = study_name("dili", "morgan_p2", 1632, 5, auroc)
    assert base == study_name("dili", "morgan_p2", 1632, 5, auroc)          # stable
    assert base != study_name("dili", "morgan_p23", 7096, 5, auroc)         # feature set
    assert base != study_name("dili", "morgan_p2", 1632, 3, auroc)          # fold count
    assert base != study_name("dili", "morgan_p2", 1632, 5, other)          # objective
    assert base != study_name("herg", "morgan_p2", 1632, 5, auroc)          # endpoint


def test_the_full_winning_parameter_set_is_recorded_for_rebuilds():
    """A trial records only the stage it varied, so the whole set must be stored.

    The winning `ames` trial's `params` are `bagging_fraction` and `bagging_freq`
    alone; `feature_fraction` and the rest were fixed in earlier stages and recorded
    nowhere. Rebuilding from the trial alone silently drops them, which on DILI fold
    1 gave validation binary error 0.173228 where the winner scored 0.149606.
    """
    assert tuning.BEST_PARAMS_ATTR == "admetus_best_params"


@pytest.mark.slow
def test_a_resumed_study_still_yields_a_usable_booster(tmp_path):
    """Copying a study into a fresh model directory must not fail the run.

    The boosters live beside the study, so a study moved on its own has trials and no
    model. `get_best_booster` then raises and the endpoint used to exit 2. It now
    rebuilds from the recorded parameters and says so.
    """
    import shutil

    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    fold = folds_of(matrix.shape[0])[0]

    first = tuning.tune_fold(
        matrix[fold.train], y[fold.train], matrix[fold.test], y[fold.test],
        params, seed=0, time_budget=15,
        storage=tmp_path / "s.sqlite", study_name="s",
        model_dir=tmp_path / "s-boosters",
    )
    assert first.boosters and not first.tuner_failed

    # The study alone, as if it had been copied to another machine.
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(tmp_path / "s.sqlite", moved / "s.sqlite")
    second = tuning.tune_fold(
        matrix[fold.train], y[fold.train], matrix[fold.test], y[fold.test],
        params, seed=0, time_budget=5,
        storage=moved / "s.sqlite", study_name="s",
        model_dir=moved / "s-boosters",
    )
    assert second.boosters, "a resumed study must still produce a model"
    assert second.boosters[0].predict(matrix[:3]).shape == (3,)
    if second.tuner_failed:
        assert "full parameters recorded" in second.rebuilt_from


def _rebuild(full_params, study_attr, trial_params):
    """Rebuild one fold from a stand-in study, and report where it took its parameters."""
    import optuna

    matrix, y = toy()
    fold = folds_of(matrix.shape[0])[0]
    params = tuning.fixed_params("classification", "auroc", num_threads=2)

    study = optuna.create_study(direction="minimize")
    study.add_trial(
        optuna.trial.create_trial(
            params=dict(trial_params),
            distributions={k: optuna.distributions.FloatDistribution(0.1, 1.0) for k in trial_params},
            value=0.2,
        )
    )
    if study_attr is not None:
        study.set_user_attr(tuning.BEST_PARAMS_ATTR, dict(study_attr))

    import lightgbm as lgb

    return tuning._rebuild_fold(
        lgb, matrix[fold.train], y[fold.train], matrix[fold.test], y[fold.test],
        params, study, seed=0, time_budget=None, reason="a test",
        full_params=full_params,
    )


def test_a_rebuild_prefers_the_tuner_s_own_accumulated_parameters():
    """The tuner holds every stage's winner; the study and the trial hold less.

    When LightGBMTuner raises partway through, `tuner.best_params` is still the only
    place the whole winning set exists: the winning trial records just the stage it
    varied, and the study attribute is written only on a successful run. Rebuilding
    from the trial alone took DILI fold 1's validation error from 14.96% to 17.32%,
    because `feature_fraction=0.6` -- fixed two stages earlier -- was dropped.
    """
    result = _rebuild(
        full_params={"feature_fraction": 0.6, "bagging_fraction": 0.8},
        study_attr={"feature_fraction": 0.9},
        trial_params={"bagging_fraction": 0.8},
    )
    assert result.params["feature_fraction"] == 0.6
    assert result.params["bagging_fraction"] == 0.8
    assert "accumulated best parameters" in result.rebuilt_from
    assert result.boosters


def test_a_rebuild_falls_back_to_the_study_then_to_the_trial():
    """And says which, because the three are not equally complete."""
    from_study = _rebuild(
        full_params=None,
        study_attr={"feature_fraction": 0.9},
        trial_params={"bagging_fraction": 0.8},
    )
    assert from_study.params["feature_fraction"] == 0.9
    assert "recorded for the study" in from_study.rebuilt_from

    from_trial = _rebuild(full_params=None, study_attr=None, trial_params={"bagging_fraction": 0.7})
    assert from_trial.params["bagging_fraction"] == 0.7
    assert "earlier stages' choices are lost" in from_trial.rebuilt_from


def test_superseded_studies_are_counted_not_adopted(tmp_path):
    """A resumed run says how many trials it left behind, and takes none of them.

    Correcting the fold signature to identify folds by dataset row renamed every
    study, so a run interrupted before that change finds its own trials under a name
    it no longer computes. They are reported, not migrated: the old name cannot
    establish which partition it measured, which is the defect it had.
    """
    import optuna

    storage = tmp_path / "dili.sqlite"
    url = f"sqlite:///{storage}"
    for name in ("admetus-dili-old0", "admetus-dili-old1"):
        study = optuna.create_study(study_name=name, storage=url, direction="minimize")
        for value in (0.3, 0.2, 0.1):
            study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=value))

    stranded = tuning.unreachable_trials(
        storage,
        current=["admetus-dili-new0", "admetus-dili-new1"],
        superseded=["admetus-dili-old0", "admetus-dili-old1"],
    )
    assert stranded == 6

    # Once this version's own studies exist, there is nothing to report any more.
    optuna.create_study(study_name="admetus-dili-new0", storage=url, direction="minimize")
    assert tuning.unreachable_trials(
        storage,
        current=["admetus-dili-new0", "admetus-dili-new1"],
        superseded=["admetus-dili-old0", "admetus-dili-old1"],
    ) == 0


def test_counting_superseded_studies_never_fails_a_training_run():
    """Bookkeeping must not be the reason a run dies."""
    assert tuning.unreachable_trials(None, current=["a"], superseded=["b"]) == 0
    assert tuning.unreachable_trials(Path("/nonexistent/x.sqlite"), current=["a"], superseded=["b"]) == 0
