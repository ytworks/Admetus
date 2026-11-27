"""``admetus train`` and ``admetus evaluate``."""

from __future__ import annotations

import time
from pathlib import Path

import click

from admetus.bundle import save as bundle_save
from admetus.bundle.load import available, read_metadata, verify
from admetus.cli.options import (
    all_option,
    data_dir_option,
    default_data_dir,
    default_models_dir,
    emit,
    endpoint_option,
    json_option,
    models_dir_option,
    run_guarded,
    setup_logging,
)
from admetus.data import loader, splits
from admetus.data.registry import resolve_keys
from admetus.errors import EXIT_NO_MODEL, AdmetusError
from admetus.logging import get_logger
from admetus.train import tuning
from admetus.train.trainer import train_endpoint

log = get_logger("cli.train")


@click.command("train")
@endpoint_option
@all_option
@data_dir_option
@models_dir_option
@click.option(
    "--folds", type=int, default=splits.DEFAULT_FOLDS, show_default=True,
    help="Cross-validation folds over the training portion, and so ensemble members.",
)
@click.option(
    "--time-budget", type=int, default=None,
    help="Wall-clock cap for tuning ONE FOLD, in seconds. Each fold is tuned separately, so "
         "an endpoint costs this many times --folds.",
)
@click.option(
    "--tune/--no-tune", default=True, show_default=True,
    help="Run the automatic LightGBM tuner. --no-tune uses fixed parameters.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Seed for splits and the sampler.")
@click.option("--jobs", type=int, default=0, help="LightGBM threads. 0 lets LightGBM decide.")
@click.option("--bootstrap", type=int, default=2000, show_default=True, help="Resamples for the interval.")
@click.option("--force", is_flag=True, help="Retrain endpoints that already have a valid bundle.")
@json_option
@click.pass_context
def train_command(
    ctx: click.Context,
    endpoints: tuple[str, ...],
    select_all: bool,
    data_dir: Path | None,
    models_dir: Path | None,
    folds: int,
    time_budget: int | None,
    tune: bool,
    seed: int,
    jobs: int,
    bootstrap: int,
    force: bool,
    as_json: bool,
) -> None:
    """Train one or more endpoints and write their bundles.

    Hyperparameters are chosen by Optuna's stepwise LightGBM tuner, over the same
    scaffold-grouped folds the model is cross-validated on. Its trial count is
    effectively fixed at 65 to 68, so --time-budget is how a long run is bounded.

    Endpoints that already have a valid bundle are skipped unless --force is
    given, so a long run can be resumed. The tuning study is kept in SQLite
    beside the bundle and resumes too.
    """
    setup_logging(ctx)

    def body() -> None:
        selected = resolve_keys(endpoints, select_all)
        data = Path(data_dir or default_data_dir())
        models = Path(models_dir or default_models_dir())
        started = time.time()

        results = []
        skipped = []
        failed = []
        for endpoint in selected:
            target = models / endpoint.key
            if not force and target.is_dir():
                ok, _ = verify(target, check_fingerprint=False)
                if ok:
                    skipped.append(endpoint.key)
                    log.info("%s: a valid bundle exists; use --force to retrain", endpoint.key)
                    continue
            try:
                dataset = loader.load(endpoint, data)
                model = train_endpoint(
                    endpoint, dataset, folds=folds, seed=seed, tune=tune,
                    time_budget=time_budget if time_budget is not None
                    else tuning.DEFAULT_TIME_BUDGET_SECONDS,
                    num_threads=jobs, bootstrap_resamples=bootstrap,
                    models_dir=models,
                )
                path = bundle_save.save(model, models)
                results.append({
                    "endpoint": endpoint.key,
                    "path": str(path),
                    "metric": endpoint.metric,
                    "primary": model.test_metrics["primary"],
                    "n_features": model.basis.n_cols,
                    # Biased by construction; it ranks candidates and is not performance.
                    "pooled_oof_score": model.cv_record["pooled_oof_score"],
                    "seconds": model.provenance["train_seconds"],
                })
            except AdmetusError as exc:
                failed.append({"endpoint": endpoint.key, "error": str(exc)})
                log.error("%s: %s", endpoint.key, exc)

        payload = {
            "ok": not failed,
            "models_dir": str(models),
            "n_trained": len(results),
            "n_skipped": len(skipped),
            "n_failed": len(failed),
            "skipped": skipped,
            "trained": results,
            "failed": failed,
            "seconds": round(time.time() - started, 1),
        }
        if as_json:
            emit(payload)
        else:
            for entry in results:
                primary = entry["primary"]
                click.echo(
                    f"{entry['endpoint']:<22} {entry['metric']:<9} "
                    f"{primary['value']:.4f} [{primary['ci_low']:.4f}, {primary['ci_high']:.4f}]  "
                    f"n={primary['n']}  {entry['seconds']:.0f}s"
                )
            if skipped:
                click.echo(f"\nskipped {len(skipped)} endpoint(s) that already have a bundle")
            if failed:
                for entry in failed:
                    click.echo(f"FAILED {entry['endpoint']}: {entry['error']}", err=True)
        if failed:
            raise SystemExit(2)

    run_guarded(body, as_json=as_json)


@click.command("evaluate")
@endpoint_option
@all_option
@models_dir_option
@json_option
@click.pass_context
def evaluate_command(
    ctx: click.Context, endpoints: tuple[str, ...], select_all: bool,
    models_dir: Path | None, as_json: bool,
) -> None:
    """Report the held-out metrics recorded when each model was trained.

    These are read from the bundles rather than recomputed: the held-out set is
    looked at exactly once, at training time, and the number is stored with its
    bootstrap interval. The cross-validation score is also stored, flagged as
    biased, and is not reported as performance.
    """
    setup_logging(ctx)

    def body() -> None:
        directory = Path(models_dir or default_models_dir())
        selected = resolve_keys(endpoints, select_all)
        # A directory can exist without holding a bundle: an interrupted training
        # run leaves its tuning study behind. Presence of the record is the test,
        # which is the same one `models list` applies.
        present = set(available(directory))
        rows = []
        missing = []
        for endpoint in selected:
            if endpoint.key not in present:
                missing.append(endpoint.key)
                continue
            path = directory / endpoint.key
            meta = read_metadata(path)
            metrics = meta.get("metrics") or {}
            # A record written before the shipped cutoff was evaluated carries only
            # `mcc_at_best_threshold`, which was chosen by searching the held-out
            # labels. Reporting it unlabelled invites reading an oracle's score as
            # the model's: on hERG it reads 0.63 where the shipped cutoff gives 0.56.
            # The current trainer ships probabilities and no operating threshold, so
            # `mcc_at_best_threshold` with no `mcc` is the normal state, not a stale
            # record. Telling the operator to retrain was wrong: another full run
            # produces exactly the same fields.
            retrospective_mcc = "mcc_at_best_threshold" in metrics
            rows.append({
                "endpoint": endpoint.key,
                "metric": endpoint.metric,
                "in_official_group": endpoint.in_official_group,
                "primary": metrics.get("primary"),
                "all_metrics": {k: v for k, v in metrics.items() if k != "primary"},
                "mcc_is_retrospective_only": retrospective_mcc,
                "mcc_note": (
                    "mcc_at_best_threshold was obtained by searching the held-out labels for the "
                    "cutoff that maximizes it, so it is a retrospective diagnostic and not a "
                    "score this model can deliver: no operating threshold ships, because the "
                    "reference implementation reports probabilities and chooses none."
                ) if retrospective_mcc else None,
                "cv_score_biased": (meta.get("cross_validation") or {}).get("cv_score_biased", True),
                "low_confidence": metrics.get("low_confidence"),
            })

        if as_json:
            # The outcome is the same in both modes. Reporting ok:true with no
            # result and exit 0 told an automated caller that an endpoint with no
            # model had been evaluated, which is the one thing it needs to know
            # is false.
            emit({
                "ok": bool(rows),
                "models_dir": str(directory),
                "n_models": len(rows),
                "n_missing": len(missing),
                "missing": missing,
                "results": rows,
                "remediation": (
                    None if rows else f"admetus train --endpoint {missing[0]}" if missing else None
                ),
            })
            if not rows:
                raise SystemExit(EXIT_NO_MODEL)
            return
        if not rows:
            click.echo(f"no trained models in {directory}")
            if missing:
                click.echo(f"train them with: admetus train --endpoint {missing[0]}")
            raise SystemExit(EXIT_NO_MODEL)
        click.echo(f"{'endpoint':<22} {'metric':<9} {'held-out (95% CI)':<26} {'n':>5}  notes")
        for row in rows:
            primary = row["primary"] or {}
            score = (
                f"{primary.get('value', float('nan')):.4f} "
                f"[{primary.get('ci_low', float('nan')):.4f}, {primary.get('ci_high', float('nan')):.4f}]"
            )
            notes = []
            if row["mcc_is_retrospective_only"]:
                notes.append("MCC retrospective (threshold picked on the test labels)")
            if row["low_confidence"]:
                notes.append("small dataset")
            if not row["in_official_group"]:
                notes.append("not in the official group")
            click.echo(
                f"{row['endpoint']:<22} {row['metric']:<9} {score:<26} "
                f"{primary.get('n', 0):>5}  {', '.join(notes)}"
            )
        if missing:
            click.echo(
                f"\n{len(missing)} endpoint(s) have no trained model yet: "
                f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}"
            )

    run_guarded(body, as_json=as_json)


__all__ = ["evaluate_command", "train_command"]
