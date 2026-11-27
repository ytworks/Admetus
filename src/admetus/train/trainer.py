"""Training one endpoint, end to end.

This follows the reference implementation's arrangement:

1. Standardize and featurize **before** splitting, so two rows that collapse to
   the same molecule cannot end up on opposite sides.
2. Hold out 20% at random, seeded. It is not looked at again until step 6.
3. Split the remaining 80% into folds, and fit one model per fold, each stopping
   early on its own validation rows.
4. Fit the label transform and the column basis on the training rows only.
5. Fit calibration and the decision threshold on the out-of-fold predictions.
6. Touch the held-out 20% exactly once, and report it with a bootstrap interval.

The ensemble is the fold models themselves, averaged, with their spread reported
beside the prediction -- not a refit at a pooled round count. Each member therefore
early-stopped on the rows it predicts within cross-validation, which makes the
out-of-fold score optimistic on top of being a selection score. It is recorded as
biased and is never reported as performance; the held-out 20% is.

Two things this deliberately does **not** do, both removed after they were measured:
an applicability-domain flag, which was uninformative at 16 of 29 endpoints, and a
conformal prediction interval, which over-covered at 8 of 9 regression endpoints by
enough to be useless. Neither was asked for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from admetus.chem.featurizer import featurize_many
from admetus.chem.spec import FeaturizerSpec
from admetus.data import splits
from admetus.data.loader import Dataset
from admetus.data.registry import Endpoint, prevalence_range
from admetus.errors import InputDataError
from admetus.logging import get_logger
from admetus.train import calibrate, cv, metrics, objectives, prune, transforms, tuning

log = get_logger("train")

@dataclass
class TrainedModel:
    """Everything a bundle needs after training one endpoint."""

    endpoint: Endpoint
    spec: FeaturizerSpec
    basis: prune.ColumnBasis
    boosters: list[Any]
    rounds: int
    params: dict[str, Any]
    transform: transforms.Transform
    calibrator: calibrate.Calibrator | None
    best_threshold: float | None
    test_metrics: dict[str, Any]
    cv_record: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_boosters(self) -> int:
        return len(self.boosters)


#: Regression objective per endpoint, where squared error is not the aligned loss.
#: The rank-based endpoints get a robust loss, because their metric is Spearman and
#: the tuner cannot search the objective itself.
REGRESSION_OBJECTIVE = {
    "vdss": "regression_l1",
    "half_life": "regression_l1",
    "clearance_hepatocyte": "regression_l1",
    "clearance_microsome": "regression_l1",
    "ppbr": "regression_l1",
    "caco2": "regression_l1",
    "lipophilicity": "regression_l1",
    "solubility": "regression_l1",
    "ld50": "regression_l1",
}


def _predict_ensemble(boosters: list[Any], matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread across the fold ensemble, in the fitted space."""
    stacked = np.column_stack([b.predict(matrix) for b in boosters])
    return stacked.mean(axis=1), stacked.std(axis=1)


def train_endpoint(
    endpoint: Endpoint,
    dataset: Dataset,
    folds: int = splits.DEFAULT_FOLDS,
    seed: int = 0,
    tune: bool = True,
    time_budget: int | None = tuning.DEFAULT_TIME_BUDGET_SECONDS,
    num_threads: int = 0,
    test_fraction: float = splits.DEFAULT_TEST_FRACTION,
    bootstrap_resamples: int = 2000,
    models_dir: Path | None = None,
    holdout: np.ndarray | None = None,
) -> TrainedModel:
    """Train one endpoint and evaluate it once on held-out data.

    Args:
        endpoint: Which endpoint to train.
        dataset: Its normalized data.
        folds: Requested cross-validation folds.
        seed: Seed for splits and the sampler.
        tune: Run the automatic tuner. Off uses fixed parameters, for tests.
        time_budget: Wall-clock cap for tuning, in seconds. The tuner's trial
            count is effectively fixed, so this is the only way to bound it.
        num_threads: LightGBM threads. ``0`` lets LightGBM decide.
        test_fraction: Share held out.
        bootstrap_resamples: Resamples for the reported confidence interval.
        models_dir: Where to keep the resumable study database.
        holdout: Boolean mask over ``dataset`` rows naming the evaluation set. Given,
            it replaces the scaffold split, which is how the official benchmark
            protocol feeds in its own published test set. ``test_fraction`` is then
            recorded but unused. Rows whose molecule fails to featurize drop out of
            whichever side they were on, as they do on the scaffold path.

    Returns:
        The :class:`TrainedModel`.

    Raises:
        InputDataError: The dataset cannot support training or evaluation.
    """
    started = time.time()
    spec = endpoint.featurizer_spec()
    log.info("%s: %s", endpoint.key, spec.describe())

    featurized = featurize_many(dataset.smiles, spec)
    if featurized.n_ok < 50:
        raise InputDataError(
            f"{endpoint.key}: only {featurized.n_ok} of {len(dataset)} molecules could be "
            "featurized, which is too few to train on"
        )
    if featurized.failures:
        log.warning(
            "%s: %d molecule(s) could not be featurized and were excluded",
            endpoint.key, len(featurized.failures),
        )

    matrix = featurized.matrix
    y_published = dataset.y[featurized.positions]
    # --- split -------------------------------------------------------------
    if holdout is None:
        split = splits.random_holdout(matrix.shape[0], test_fraction=test_fraction, seed=seed)
        split_kind = f"random, {test_fraction:.0%} held out, seed {seed}"
    else:
        mask = np.asarray(holdout, dtype=bool)
        if mask.shape != (len(dataset),):
            raise InputDataError(
                f"{endpoint.key}: the holdout mask has shape {mask.shape}, "
                f"but the dataset has {len(dataset)} rows"
            )
        # featurized.positions indexes the dataset, so the mask follows the rows
        # that survived featurization rather than the original numbering.
        kept = mask[featurized.positions]
        split = splits.Split(train=np.flatnonzero(~kept), test=np.flatnonzero(kept))
        if split.test.size == 0 or split.train.size == 0:
            raise InputDataError(
                f"{endpoint.key}: the given holdout leaves {split.train.size} training and "
                f"{split.test.size} evaluation rows, so nothing can be measured"
            )
        split_kind = "given (published benchmark split)"
    train_rows = split.train
    test_rows = split.test

    if train_rows.size < 50:
        raise InputDataError(
            f"{endpoint.key}: only {train_rows.size} training rows, which is too few to train on"
        )

    # --- transform and basis, fitted on training rows only ----------------
    transform = transforms.fit(endpoint.transform, y_published[train_rows])
    y_fitted = transform.forward(y_published)

    x_train_raw = matrix[train_rows]
    basis = prune.fit_basis(x_train_raw, prevalence_range(train_rows.size))
    x_train = prune.apply_basis(x_train_raw, basis)
    y_train = y_fitted[train_rows]

    # --- folds, over the training portion only -----------------------------
    fold_sets = splits.cv_folds(train_rows.size, n_splits=folds)

    # --- parameters we set by rule, which the tuner then builds on ---------
    n_pos = int(np.count_nonzero(y_train == 1)) if endpoint.is_classification else 0
    n_neg = int(y_train.size - n_pos)
    weight = (
        tuning.class_weight(n_pos / y_train.size, n_pos, n_neg)
        if endpoint.is_classification
        else 1.0
    )
    objective = REGRESSION_OBJECTIVE.get(endpoint.key) if not endpoint.is_classification else None
    params = tuning.fixed_params(
        endpoint.task, endpoint.metric, scale_pos_weight=weight,
        objective=objective, num_threads=num_threads, seed=seed,
    )
    if weight != 1.0:
        log.info(
            "%s: %.1f%% positive, so scale_pos_weight=%.2f", endpoint.key,
            100 * n_pos / y_train.size, weight,
        )

    # --- automatic tuning --------------------------------------------------
    tune_record: dict[str, Any]
    if tune:
        log.info(
            "%s: tuning over %d fold(s), %d train rows, %d columns (budget %ss)",
            endpoint.key, len(fold_sets), train_rows.size, basis.n_cols, time_budget,
        )
        study = tuning.study_name(
            endpoint.key, spec.feature_set, basis.n_cols, len(fold_sets), params,
            seed=seed, fold_signature=splits.fold_signature(fold_sets),
        )
        tuned = tuning.auto_tune(
            x_train, y_train, fold_sets, params, seed=seed, time_budget=time_budget,
            storage=tuning.study_path(models_dir, endpoint.key) if models_dir else None,
            model_dir=tuning.booster_dir(models_dir, study) if models_dir else None,
            study_name=study,
        )
        merged = tuned.params
        tune_record = tuned.to_dict()
    else:
        merged = tuning.default_params(
            endpoint.task, endpoint.metric, train_rows.size, num_threads=num_threads, seed=seed
        )
        merged.update({k: v for k, v in params.items() if k not in merged})
        tune_record = {"tuner": None, "best_params": merged, "cv_score_biased": True}

    # --- fit the ensemble: one model per fold, each stopping on its own fold ---
    # These are the models that ship. The reference implementation keeps exactly
    # the models its folds produced, so there is no refit at a pooled round count.
    final = [
        cv.fit_fold(x_train, y_train, f, merged, seed=seed, stop_on_fold=True)
        for f in fold_sets
    ]
    boosters = [f.booster for f in final]
    rounds = max((f.best_iteration for f in final), default=0)
    log.info(
        "%s: fitted %d booster(s), early-stopped at %s rounds",
        endpoint.key, len(boosters), [f.best_iteration for f in final],
    )

    pooled = objectives.pool(final, x_train.shape[0], endpoint.task, endpoint.metric, y_train)
    oof = pooled.prediction
    oof_spread = np.zeros_like(oof)
    for fit in final:
        others = [b for b in boosters if b is not fit.booster]
        if others:
            stacked = np.column_stack([b.predict(x_train[fit.valid_rows]) for b in others])
            oof_spread[fit.valid_rows] = stacked.std(axis=1)

    # --- post-fit calibration and the decision threshold -------------------
    calibrator: calibrate.Calibrator | None = None
    best_threshold: float | None = None
    covered = pooled.covered & np.isfinite(oof)

    if endpoint.is_classification:
        weight = float(merged.get("scale_pos_weight", 1.0))
        calibrator = calibrate.fit(
            y_train[covered], oof[covered], scale_pos_weight=weight, metric=endpoint.metric
        )
        best_threshold, _ = metrics.best_threshold(y_train[covered], calibrator.apply(oof[covered]))

    # --- the one look at the held-out set ----------------------------------
    x_test = prune.apply_basis(matrix[test_rows], basis)
    mean_fitted, spread_fitted = _predict_ensemble(boosters, x_test)
    y_test = y_published[test_rows]

    if endpoint.is_classification:
        prediction = calibrator.apply(mean_fitted)
    else:
        prediction = transforms.clamp(transform.inverse(mean_fitted), endpoint.output_range)

    test_metrics: dict[str, Any] = {
        k: float(v)
        for k, v in metrics.evaluate(
            endpoint.task, y_test, prediction, shipped_threshold=best_threshold
        ).items()
    }
    primary = metrics.bootstrap(
        endpoint.metric, y_test, prediction, n_resamples=bootstrap_resamples, seed=seed
    )
    test_metrics["primary"] = primary.to_dict()
    test_metrics["primary_metric"] = endpoint.metric
    test_metrics["n_test"] = int(test_rows.size)
    test_metrics["low_confidence"] = endpoint.low_confidence(len(dataset))

    cv_record = dict(tune_record)
    # Scored on the production ensemble's own out-of-fold predictions, so it is
    # comparable across runs that differ in feature set or parameters. Biased by
    # construction -- the folds chose the model -- and flagged as such by
    # ``cv_score_biased``. It selects; it never reports.
    cv_record["pooled_oof_score"] = objectives.score(pooled, y_train, endpoint.metric)
    cv_record["pooled_oof_n"] = pooled.n_covered
    cv_record["fold_diagnostics"] = objectives.fold_diagnostics(pooled)
    cv_record["fold_report"] = splits.fold_report(fold_sets, y_train, endpoint.task)
    cv_record["deployment_rounds"] = rounds
    cv_record["n_folds"] = len(fold_sets)
    cv_record["n_columns_into_tuner"] = basis.n_cols

    provenance = {
        "n_dataset_rows": len(dataset),
        "n_featurized": featurized.n_ok,
        "n_featurize_failures": len(featurized.failures),
        "n_train": int(train_rows.size),
        "n_test": int(test_rows.size),
        "test_fraction": test_fraction if holdout is None else None,
        "split": split_kind,
        "dataset": dataset.provenance,
        "seed": seed,
        "num_threads": num_threads,
        "train_seconds": round(time.time() - started, 1),
    }
    log.info(
        "%s: held-out %s = %s (n=%d)%s",
        endpoint.key, endpoint.metric, primary, test_rows.size,
        "  [small test set]" if test_metrics["low_confidence"] else "",
    )

    return TrainedModel(
        endpoint=endpoint,
        spec=spec,
        basis=basis,
        boosters=boosters,
        rounds=rounds,
        params=merged,
        transform=transform,
        calibrator=calibrator,
        best_threshold=best_threshold,
        test_metrics=test_metrics,
        cv_record=cv_record,
        provenance=provenance,
    )


__all__ = ["TrainedModel", "train_endpoint"]
