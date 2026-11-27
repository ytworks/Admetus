"""Training one endpoint, end to end.

The order of operations is what makes the reported number mean something:

1. Standardize and featurize **before** splitting, so two rows that collapse to
   the same molecule cannot end up on opposite sides.
2. Split by scaffold, largest groups into train, which puts the rare chemistry in
   the held-out set on purpose.
3. Audit at the level of the feature vector and drop the offending **training**
   rows. Held-out rows are never dropped, so the metric stays measured on the same
   compounds the published benchmark uses.
4. Fit the label transform and the column basis on the training rows only.
5. Search hyperparameters against pooled out-of-fold predictions.
6. Refit the fold ensemble at a fixed round count, so nothing that ships was
   stopped on data it was then scored against.
7. Fit calibration, the decision threshold and the conformal width on the
   out-of-fold predictions that already exist.
8. Touch the held-out set exactly once, and report it with a bootstrap interval.

The cross-validation score is used to rank candidates and is recorded as biased.
It is never reported as performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from admetus.chem.featurizer import featurize_many
from admetus.chem.similarity import Reference, nearest_neighbour_similarity
from admetus.chem.spec import FeaturizerSpec
from admetus.data import audit, splits
from admetus.data.loader import Dataset
from admetus.data.registry import Endpoint, prevalence_range
from admetus.errors import InputDataError
from admetus.logging import get_logger
from admetus.train import calibrate, conformal, cv, metrics, objectives, prune, transforms, tuning

log = get_logger("train")

#: Nominal coverage of the prediction interval fitted for regression endpoints.
INTERVAL_LEVEL = 0.9

#: Cap on reference fingerprints kept per endpoint for the applicability domain.
MAX_AD_REFERENCE = 4_096

#: Percentile of the training set's own nearest-neighbour similarity used as the
#: starting point for the domain threshold, before error calibration adjusts it.
AD_PERCENTILE = 5.0

#: How much worse than the overall error a similarity band may be before it is
#: treated as outside the domain.
AD_ERROR_MULTIPLE = 1.5


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
    interval: conformal.ConformalInterval | None
    reference: Reference
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


def _rounds_from(boosters: list[Any]) -> int:
    """Pick the production round count from the tuner's per-fold boosters.

    The median of what the folds settled on, with 10% headroom because the
    refitted models see slightly more data than the early-stopped ones did.
    """
    counts = [int(getattr(b, "best_iteration", 0) or b.num_trees()) for b in boosters]
    counts = [c for c in counts if c > 0]
    if not counts:
        return 300
    return max(10, int(round(float(np.median(counts)) * 1.1)))


def _predict_ensemble(boosters: list[Any], matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread across the fold ensemble, in the fitted space."""
    stacked = np.column_stack([b.predict(matrix) for b in boosters])
    return stacked.mean(axis=1), stacked.std(axis=1)


def _domain_threshold(
    matrix: sparse.csr_matrix,
    n_bits: int,
    y_true: np.ndarray,
    oof: np.ndarray,
    task: str,
    metric: str,
) -> tuple[float, list[dict[str, float]]]:
    """Choose the domain threshold from where error actually degrades.

    The training set's own nearest-neighbour distribution gives a starting point,
    but on its own it behaves backwards: a congeneric dataset has a high
    percentile and would reject almost any external compound, while a diverse one
    has a low percentile and would reject almost nothing -- strictness ends up
    inversely related to need.

    So the out-of-fold predictions are stratified by neighbour similarity and the
    threshold is placed where error exceeds a multiple of the overall error. The
    whole curve goes into the bundle, which gives ``in_domain`` a stated meaning.
    """
    similarity = nearest_neighbour_similarity(matrix, n_bits)
    fallback = float(np.percentile(similarity, AD_PERCENTILE))

    mask = np.isfinite(oof)
    if np.count_nonzero(mask) < 50:
        return fallback, []

    error = (
        np.abs(y_true[mask] - oof[mask])
        if task == "regression"
        else np.abs(y_true[mask] - oof[mask])
    )
    sim = similarity[mask]
    overall = float(np.median(error))

    edges = np.quantile(sim, np.linspace(0, 1, 11))
    curve: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        band = (sim >= lo) & (sim <= hi if i == len(edges) - 2 else sim < hi)
        if np.count_nonzero(band) < 5:
            continue
        curve.append(
            {
                "similarity_low": lo,
                "similarity_high": hi,
                "n": int(np.count_nonzero(band)),
                "median_error": float(np.median(error[band])),
            }
        )

    threshold = fallback
    if overall > 0:
        for entry in curve:
            if entry["median_error"] <= overall * AD_ERROR_MULTIPLE:
                threshold = entry["similarity_low"]
                break
    # Never claim a domain so wide that the training set itself sits outside it.
    threshold = float(min(max(threshold, 0.0), 0.9))
    _ = metric  # recorded by the caller; kept in the signature for clarity
    return threshold, curve


def train_endpoint(
    endpoint: Endpoint,
    dataset: Dataset,
    folds: int = 5,
    seed: int = 0,
    tune: bool = True,
    time_budget: int | None = tuning.DEFAULT_TIME_BUDGET_SECONDS,
    num_threads: int = 0,
    test_fraction: float = 0.2,
    bootstrap_resamples: int = 2000,
    models_dir: Path | None = None,
    feature_set: str | None = None,
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
        feature_set: Override the size-derived feature set.
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
    spec = endpoint.featurizer_spec(len(dataset))
    if feature_set:
        from admetus.chem.spec import get_spec

        base = get_spec(feature_set)
        spec = FeaturizerSpec(
            feature_set=base.feature_set,
            morgan_radius=base.morgan_radius,
            morgan_bits=base.morgan_bits,
            pharm_max_points=base.pharm_max_points,
            standardize=tuple(endpoint.standardize),
        )
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
    # Scaffold grouping has to see the molecule the model sees. Grouping the input
    # strings instead lets two rows that standardize to the same structure -- a
    # salt and its free base, two spellings of one tautomer -- land on opposite
    # sides of the split, which is the leak the split exists to prevent. Measured
    # on the shipped hERG split, 9 of 129 held-out molecules shared a standardized
    # scaffold with a surviving training row.
    smiles_ok = list(featurized.canonical)

    # --- split, then audit at the level the model can actually see ---------
    if holdout is None:
        split = splits.scaffold_split(smiles_ok, test_fraction=test_fraction)
        split_kind = f"scaffold, {test_fraction:.0%} held out"
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
    leakage = audit.audit_split(matrix, y_published, split.train, split.test)
    train_rows = np.setdiff1d(split.train, leakage.drop_from_train, assume_unique=False)
    test_rows = split.test

    keep, y_resolved, within = audit.resolve_within_train(matrix, y_published, train_rows, endpoint.task)
    train_rows = keep
    y_published = y_resolved

    if train_rows.size < 50:
        raise InputDataError(f"{endpoint.key}: only {train_rows.size} training rows survive auditing")

    # --- transform and basis, fitted on training rows only ----------------
    transform = transforms.fit(endpoint.transform, y_published[train_rows])
    y_fitted = transform.forward(y_published)

    x_train_raw = matrix[train_rows]
    basis = prune.fit_basis(x_train_raw, prevalence_range(train_rows.size))
    x_train = prune.apply_basis(x_train_raw, basis)
    y_train = y_fitted[train_rows]

    # --- folds -------------------------------------------------------------
    fold_sets = splits.cv_folds(
        [smiles_ok[i] for i in train_rows], y_train, endpoint.task, n_splits=folds, seed=seed
    )

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
        # The tuner's own boosters are used only to pick the round count. They were
        # early-stopped on the folds they were scored against, so they are not what
        # ships; the production ensemble is refitted below at a fixed count.
        rounds = _rounds_from(tuned.boosters)
        tune_record["deployment_rounds_source"] = (
            "median of the tuner's folds, with 10% headroom" if tuned.boosters
            else "DEFAULT: the tuner returned no folds, so the round count was not tuned"
        )
        if not tuned.boosters:
            log.warning(
                "%s: the tuner returned no per-fold boosters, so the production round count is "
                "the %d-round default rather than what the folds settled on", endpoint.key, rounds,
            )
    else:
        merged = tuning.default_params(
            endpoint.task, endpoint.metric, train_rows.size, num_threads=num_threads, seed=seed
        )
        merged.update({k: v for k, v in params.items() if k not in merged})
        tune_record = {"tuner": None, "best_params": merged, "cv_score_biased": True}
        rounds = 300

    # --- refit the production ensemble at a fixed round count ---------------
    final = [
        cv.fit_fold(x_train, y_train, f, merged, num_boost_round=rounds, seed=seed)
        for f in fold_sets
    ]
    boosters = [f.booster for f in final]
    log.info("%s: refitted %d booster(s) at %d rounds", endpoint.key, len(boosters), rounds)

    pooled = objectives.pool(final, x_train.shape[0], endpoint.task, endpoint.metric, y_train)
    oof = pooled.prediction
    oof_spread = np.zeros_like(oof)
    for fit in final:
        others = [b for b in boosters if b is not fit.booster]
        if others:
            stacked = np.column_stack([b.predict(x_train[fit.valid_rows]) for b in others])
            oof_spread[fit.valid_rows] = stacked.std(axis=1)

    # --- post-fit calibration, threshold and interval ----------------------
    calibrator: calibrate.Calibrator | None = None
    interval: conformal.ConformalInterval | None = None
    best_threshold: float | None = None
    covered = pooled.covered & np.isfinite(oof)

    if endpoint.is_classification:
        weight = float(merged.get("scale_pos_weight", 1.0))
        calibrator = calibrate.fit(
            y_train[covered], oof[covered], scale_pos_weight=weight, metric=endpoint.metric
        )
        best_threshold, _ = metrics.best_threshold(y_train[covered], calibrator.apply(oof[covered]))
    else:
        interval = conformal.fit(
            y_train[covered], oof[covered], oof_spread[covered],
            level=INTERVAL_LEVEL, is_log=transform.is_log,
        )

    # --- applicability domain ---------------------------------------------
    morgan_block = x_train_raw[:, : spec.morgan_bits]
    threshold, curve = _domain_threshold(
        morgan_block, spec.morgan_bits, y_train[covered], oof[covered], endpoint.task, endpoint.metric
    )
    reference = Reference.build(
        morgan_block, spec.morgan_bits, threshold, max_reference=MAX_AD_REFERENCE, seed=seed
    )

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

    if interval is not None:
        low, high = interval.bounds(mean_fitted, spread_fitted)
        achieved = conformal.coverage(
            y_test,
            transforms.clamp(transform.inverse(low), endpoint.output_range),
            transforms.clamp(transform.inverse(high), endpoint.output_range),
        )
        test_metrics["interval_level"] = interval.level
        test_metrics["interval_coverage"] = achieved
        log.info(
            "%s: interval covers %.1f%% of held-out labels at a nominal %.0f%%",
            endpoint.key, 100 * achieved, 100 * interval.level,
        )

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
        "leakage_audit": leakage.to_dict(),
        "within_train_resolution": within,
        "dataset": dataset.provenance,
        "seed": seed,
        "num_threads": num_threads,
        "domain_threshold": threshold,
        "domain_error_curve": curve,
        "n_reference": reference.n_reference,
        "reference_subsampled": reference.subsampled,
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
        interval=interval,
        reference=reference,
        best_threshold=best_threshold,
        test_metrics=test_metrics,
        cv_record=cv_record,
        provenance=provenance,
    )


__all__ = [
    "AD_ERROR_MULTIPLE",
    "AD_PERCENTILE",
    "INTERVAL_LEVEL",
    "MAX_AD_REFERENCE",
    "TrainedModel",
    "train_endpoint",
]
