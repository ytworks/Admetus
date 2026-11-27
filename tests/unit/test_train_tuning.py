"""The automatic tuner path.

Tuning is delegated to Optuna's stepwise LightGBM tuner, so what is worth testing
is the integration: that our sparse matrices and scaffold-grouped folds go
through it, that the metric we align per endpoint survives, that the per-fold
boosters come back, and that the study resumes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from admetus.data.splits import Split
from admetus.train import cv, tuning


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
    """Contiguous folds, standing in for the scaffold-grouped ones."""
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
    ("task", "metric", "expected_metric", "expected_objective"),
    [
        ("classification", "auroc", "auc", "binary"),
        ("classification", "auprc", "average_precision", "binary"),
        ("regression", "mae", "l1", "regression_l1"),
        ("regression", "spearman", "l2", "regression"),
    ],
)
def test_metric_is_aligned_with_the_endpoint(task, metric, expected_metric, expected_objective):
    """A 0/1 error rate at a fixed threshold measures something nobody reports."""
    params = tuning.fixed_params(task, metric)
    assert params["metric"] == expected_metric
    assert params["objective"] == expected_objective
    assert params["metric"] != "binary_error"


def test_fixed_params_pin_what_the_tuner_cannot_search():
    params = tuning.fixed_params("classification", "auroc", scale_pos_weight=3.0, seed=7)
    assert params["learning_rate"] == cv.LEARNING_RATE == 0.03
    assert params["scale_pos_weight"] == 3.0
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    assert params["seed"] == 7
    # The tuner varies min_child_samples and turns this off itself; recording it
    # here keeps the bundle's record honest about what ran.
    assert params["feature_pre_filter"] is False


def test_no_weight_is_applied_to_a_balanced_endpoint():
    assert tuning.class_weight(0.45, 450, 550) == 1.0
    assert tuning.class_weight(0.33, 330, 670) == 1.0


def test_moderate_imbalance_gets_the_square_root_of_the_ratio():
    weight = tuning.class_weight(0.10, 100, 900)
    assert weight == pytest.approx(3.0)


def test_severe_imbalance_gets_the_full_ratio():
    weight = tuning.class_weight(0.04, 40, 960)
    assert weight == pytest.approx(24.0)


def test_degenerate_counts_do_not_produce_a_weight():
    assert tuning.class_weight(0.0, 0, 100) == 1.0
    assert tuning.class_weight(1.0, 100, 0) == 1.0


def test_study_direction_follows_the_metric():
    assert tuning._direction("auc") == "maximize"
    assert tuning._direction("average_precision") == "maximize"
    assert tuning._direction("l1") == "minimize"
    assert tuning._direction("l2") == "minimize"


def test_study_path_sits_beside_the_bundles_not_inside_one(tmp_path):
    """This asserted the old location, which was the defect: `save` replaces the
    endpoint directory atomically, so a study kept inside it was deleted by the
    successful run that built it."""
    assert tuning.study_path(tmp_path, "dili") == tmp_path / "studies" / "dili.sqlite"


@pytest.mark.slow
def test_tuner_accepts_sparse_input_and_our_folds(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    result = tuning.auto_tune(
        matrix, y, folds_of(matrix.shape[0]), params, seed=0, time_budget=20,
        storage=tmp_path / "study.sqlite", study_name="t",
    )
    assert result.n_trials > 0
    assert np.isfinite(result.score)
    # One booster per fold, which is what the ensemble and the out-of-fold
    # predictions both need.
    assert len(result.boosters) == 3
    # Predicting through a fold booster must work on a CSR slice.
    assert result.boosters[0].predict(matrix[:5]).shape == (5,)


@pytest.mark.slow
def test_the_tuner_leaves_our_fixed_parameters_alone(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auprc", scale_pos_weight=2.5, num_threads=2)
    result = tuning.auto_tune(
        matrix, y, folds_of(matrix.shape[0]), params, seed=0, time_budget=20,
        storage=tmp_path / "s.sqlite", study_name="t",
    )
    assert result.params["learning_rate"] == 0.03
    assert result.params["scale_pos_weight"] == 2.5
    assert result.params["metric"] == "average_precision"
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
    storage = tmp_path / "resume.sqlite"
    first = tuning.auto_tune(
        matrix, y, folds_of(matrix.shape[0]), params, seed=0, time_budget=10,
        storage=storage, study_name="resume",
    )
    second = tuning.auto_tune(
        matrix, y, folds_of(matrix.shape[0]), params, seed=0, time_budget=10,
        storage=storage, study_name="resume",
    )
    assert second.n_trials >= first.n_trials
    assert storage.is_file()


@pytest.mark.slow
def test_a_seed_makes_the_run_reproducible(tmp_path):
    matrix, y = toy()
    params = tuning.fixed_params("classification", "auroc", num_threads=2)
    runs = [
        tuning.auto_tune(
            matrix, y, folds_of(matrix.shape[0]), params, seed=11, time_budget=15,
            storage=tmp_path / f"seeded_{i}.sqlite", study_name="s",
        )
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
    assert record["feature_pre_filter"] is False
    assert record["tuned_by_tuner"] == ["num_leaves"]
    # Thread count is a machine detail, not a hyperparameter.
    assert "num_threads" not in record["best_params"]


def test_untuned_defaults_respect_the_tuner_grid_floor():
    """Without tuning we still pick a feature_fraction the tuner could have chosen."""
    params = tuning.default_params("classification", "auroc", 400)
    assert params["feature_fraction"] >= 0.4
    assert params["learning_rate"] == 0.03


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
    auprc = fixed_params("classification", "auprc")

    base = study_name("dili", "morgan_p2", 1632, 5, auroc)
    assert base == study_name("dili", "morgan_p2", 1632, 5, auroc)          # stable
    assert base != study_name("dili", "morgan_p23", 7096, 5, auroc)         # feature set
    assert base != study_name("dili", "morgan_p2", 1632, 3, auroc)          # fold count
    assert base != study_name("dili", "morgan_p2", 1632, 5, auprc)          # objective
    assert base != study_name("herg", "morgan_p2", 1632, 5, auroc)          # endpoint
