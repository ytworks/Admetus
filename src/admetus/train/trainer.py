"""Training one endpoint, end to end.

This follows the reference implementation's arrangement:

1. Parse and featurize each published row as it is. Nothing is standardized and
   nothing is deduplicated, so two rows describing the same molecule stay two rows
   and can land on opposite sides of the split.
2. Hold out 20% at random, seeded, with ``train_test_split``. It is not looked at
   again until step 5.
3. ``KFold`` over the training portion, in the order the holdout left it, and one
   model per fold, each stopping early on its own validation rows.
4. Tune each fold on its own, so the folds end up with different parameters.
5. Touch the held-out 20% exactly once, and report it with a bootstrap interval.

The ensemble is the fold models themselves, averaged, with their spread reported
beside the prediction -- not a refit at a pooled round count.

The pooled out-of-fold score is recorded because it is what selected the model, and
it is recorded as biased for that reason alone. It is not reliably optimistic: each
member early-stopped on the rows it predicts within cross-validation, which pushes
it up, but it is one booster per row where the shipped prediction is the mean of all
of them, which pushes it down harder. Measured on the 29 shipped bundles it is the
more favourable of the two at 6 endpoints and the less favourable at 23. Either way
it selects and never reports; the held-out 20% does.

Nothing is fitted on top of the ensemble. No label transform, no column basis, no
probability calibration, no decision threshold, no applicability-domain flag and no
conformal interval: each was an addition, each was measured, and each is gone.
Calibration was the costly one -- it inverted the ranking on `carcinogens`, taking
held-out AUROC from 0.7506 to 0.2494.
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
from admetus.data.registry import Endpoint
from admetus.errors import InputDataError
from admetus.logging import get_logger
from admetus.train import cv, metrics, objectives, prune, transforms, tuning

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
    test_metrics: dict[str, Any]
    cv_record: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_boosters(self) -> int:
        return len(self.boosters)


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
        # The published training rows arrive in file order, and `cv_folds` cuts
        # contiguous blocks, so without this every seed validated on exactly the same
        # compounds: five DILI seeds gave identical folds, and the five-seed spread
        # reported none of the fold variation the protocol is asking for. The test
        # rows stay exactly as published.
        shuffled = np.random.default_rng(seed).permutation(split.train.size)
        split = splits.Split(train=split.train[shuffled], test=split.test)
    train_rows = split.train
    test_rows = split.test

    if train_rows.size < 50:
        raise InputDataError(
            f"{endpoint.key}: only {train_rows.size} training rows, which is too few to train on"
        )

    # Labels as published, and every column. The log10 transform, the output clamp
    # and the prevalence pruning were all additions: the reference trains on the
    # label it was given and hands LightGBM the whole 89,852-column vector.
    transform = transforms.fit(None, y_published[train_rows])
    basis = prune.identity_basis(matrix.shape[1])
    x_train = matrix[train_rows]
    y_train = y_published[train_rows]

    # --- folds, over the training portion only -----------------------------
    fold_sets = splits.cv_folds(train_rows.size, n_splits=folds)

    # --- the reference implementation's parameters; the tuner builds on them ---
    params = tuning.fixed_params(
        endpoint.task, endpoint.metric, num_threads=num_threads, seed=seed
    )

    # --- one tuned model per fold, as the reference implementation does -------
    # Each fold is tuned on its own, with its own validation rows as the watchlist,
    # and the booster the tuner settles on is the ensemble member. So the folds end
    # up with different parameters, and the trial count is per fold -- 68 each,
    # 204 for three, which is what the reference's notebooks show.
    tune_record: dict[str, Any]
    final: list[cv.FoldFit] = []
    per_fold: list[dict[str, Any]] = []
    if tune:
        log.info(
            "%s: tuning %d fold(s) separately, %d train rows, %d columns (budget %ss each)",
            endpoint.key, len(fold_sets), train_rows.size, basis.n_cols, time_budget,
        )
        # A study is named after the settings its trials depend on, the fold signature
        # among them, so correcting that signature renamed every study. An
        # interrupted run resumed after the upgrade finds its own trials on disk
        # under a name it no longer computes; they are not adopted, because the old
        # name cannot establish which partition it measured. Say so once, rather than
        # letting a resumed run look as though it had lost its history silently.
        def study_for(index: int, row_ids: Any) -> str:
            return tuning.study_name(
                endpoint.key, spec.feature_set, basis.n_cols, len(fold_sets), params,
                seed=seed,
                fold_signature=f"{splits.fold_signature(fold_sets, row_ids)}-{index}",
            )

        storage = tuning.study_path(models_dir, endpoint.key) if models_dir else None
        stranded = tuning.unreachable_trials(
            storage,
            current=[study_for(i, train_rows) for i in range(len(fold_sets))],
            superseded=[study_for(i, None) for i in range(len(fold_sets))],
        )
        if stranded:
            log.warning(
                "%s: %d trial(s) in this endpoint's study database were tuned under the "
                "previous fold-signature scheme and cannot be shown to belong to these "
                "folds, so they are not resumed. This run tunes from zero, once.",
                endpoint.key, stranded,
            )

        for index, fold in enumerate(fold_sets):
            study = study_for(index, train_rows)
            tuned = tuning.tune_fold(
                x_train[fold.train], y_train[fold.train],
                x_train[fold.test], y_train[fold.test],
                params, seed=seed, time_budget=time_budget,
                storage=storage,
                study_name=study,
                model_dir=tuning.booster_dir(models_dir, study) if models_dir else None,
            )
            if not tuned.boosters:
                raise InputDataError(
                    f"{endpoint.key}: the tuner returned no model for fold {index}"
                )
            booster = tuned.boosters[0]
            best_iteration = int(getattr(booster, "best_iteration", 0) or booster.num_trees())
            final.append(
                cv.FoldFit(
                    booster=booster,
                    valid_pred=np.asarray(
                        booster.predict(x_train[fold.test], num_iteration=best_iteration),
                        dtype=np.float64,
                    ),
                    valid_rows=fold.test,
                    best_iteration=best_iteration,
                    used_early_stopping=True,
                )
            )
            per_fold.append(tuned.to_dict())
        merged = per_fold[0]["best_params"]
        tune_record = {
            "tuner": per_fold[0]["tuner"],
            "per_fold": per_fold,
            "best_params": merged,
            "cv_score_biased": True,
            "n_trials": sum(r["n_trials"] for r in per_fold),
            "sampler_seed": seed,
            # Per fold, not for the endpoint: every fold gets the whole budget, so
            # the tuning wall clock is this times `n_folds`.
            "time_budget_seconds": time_budget,
            "search_space_owner": "tuner",
            "feature_fraction_grid_floor": 0.4,
        }
    else:
        merged = tuning.default_params(
            endpoint.task, endpoint.metric, train_rows.size, num_threads=num_threads, seed=seed
        )
        merged.update({k: v for k, v in params.items() if k not in merged})
        tune_record = {"tuner": None, "best_params": merged, "cv_score_biased": True}
        final = [
            cv.fit_fold(x_train, y_train, f, merged, seed=seed, stop_on_fold=True)
            for f in fold_sets
        ]

    boosters = [f.booster for f in final]
    rounds = max((f.best_iteration for f in final), default=0)
    log.info(
        "%s: %d booster(s), early-stopped at %s rounds",
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


    # --- the one look at the held-out set ----------------------------------
    x_test = matrix[test_rows]
    mean_fitted, spread_fitted = _predict_ensemble(boosters, x_test)
    y_test = y_published[test_rows]

    # The mean across the fold models, as published, with no calibration and no
    # clamp. Calibration was an addition and a harmful one: fitted on out-of-fold
    # predictions, Platt chose a negative slope on `carcinogens` and inverted the
    # ranking, turning a held-out AUROC of 0.7506 into 0.2494.
    prediction = mean_fitted

    test_metrics: dict[str, Any] = {
        k: float(v)
        for k, v in metrics.evaluate(
            endpoint.task, y_test, prediction
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
        test_metrics=test_metrics,
        cv_record=cv_record,
        provenance=provenance,
    )


__all__ = ["TrainedModel", "train_endpoint"]
