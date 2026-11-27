"""``admetus benchmark``: the official protocol, on the published splits.

Separate from ``admetus evaluate`` on purpose. ``evaluate`` reads the number this
project measured on its own scaffold split; this command produces the number that can
sit beside the published leaderboard. Conflating them under one flag would make it
easy to quote one as the other.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from admetus.cli.options import (
    all_option,
    data_dir_option,
    default_data_dir,
    emit,
    endpoint_option,
    json_option,
    run_guarded,
    setup_logging,
    write_report,
)
from admetus.data import benchmark as bench
from admetus.data import splits as bench_splits
from admetus.data.registry import get_endpoint, official_endpoints
from admetus.errors import AdmetusError, UsageError
from admetus.logging import get_logger
from admetus.train import tuning
from admetus.train.official import run_endpoint

log = get_logger("cli.benchmark")


@click.group("benchmark")
def benchmark_group() -> None:
    """Run the official benchmark protocol on the published train/test splits."""


def _selected(endpoints: tuple[str, ...], select_all: bool):
    """Resolve endpoint selectors, refusing the ones with no published split."""
    if select_all:
        return official_endpoints()
    if not endpoints:
        raise UsageError(
            "no endpoint selected; pass --endpoint KEY (repeatable) or --all. "
            f"--all means the {len(official_endpoints())} endpoints of the official group."
        )
    chosen = [get_endpoint(key) for key in endpoints]
    outside = [e.key for e in chosen if not e.in_official_group]
    if outside:
        raise UsageError(
            f"no published split exists for {', '.join(outside)}, so the official protocol "
            "cannot be run on them. 'admetus evaluate' reports this project's own scaffold "
            "split for every endpoint."
        )
    return chosen


@benchmark_group.command("fetch")
@data_dir_option
@click.option("--overwrite", is_flag=True, help="Download again even if the archive is present.")
@json_option
@click.pass_context
def fetch_command(ctx: click.Context, data_dir: Path | None, overwrite: bool, as_json: bool) -> None:
    """Download and unpack the published split archive.

    One 1.4 MB zip covering all 22 endpoints of the official group, pinned by
    digest. A mismatch is refused rather than adopted.
    """
    setup_logging(ctx)

    def body() -> None:
        directory = Path(data_dir or default_data_dir())
        root = bench.fetch_archive(directory, overwrite=overwrite)
        present = [e.key for e in official_endpoints() if (bench.split_directory(directory, e) / "test.csv").is_file()]
        payload = {
            "ok": len(present) == len(official_endpoints()),
            "archive": str(bench.archive_path(directory)),
            "archive_sha256": bench.ARCHIVE_SHA256,
            "unpacked_into": str(root),
            "n_endpoints": len(present),
            "endpoints": present,
        }
        if as_json:
            emit(payload)
        else:
            click.echo(f"unpacked the published splits for {len(present)} endpoint(s) into {root}")

    run_guarded(body, as_json=as_json)


@benchmark_group.command("run")
@endpoint_option
@all_option
@data_dir_option
@click.option(
    "--folds", type=int, default=bench_splits.DEFAULT_FOLDS, show_default=True,
    help="Cross-validation folds inside train_val.",
)
@click.option(
    "--seeds", type=int, default=len(bench.OFFICIAL_SEEDS), show_default=True,
    help="How many of the protocol's seeds to run. Fewer than five is not a comparable number.",
)
@click.option(
    "--tune/--no-tune", default=True, show_default=True,
    help="Tune within each seed's run. --no-tune is cheaper and reported as a floor.",
)
@click.option(
    "--time-budget", type=int, default=None,
    help="Wall-clock cap for tuning ONE FOLD, in seconds. An endpoint costs this many "
         "times folds times seeds: at the default 3 folds and 5 seeds, 15 of them.",
)
@click.option("--jobs", type=int, default=0, help="LightGBM threads. 0 lets LightGBM decide.")
@click.option(
    "--work-dir", type=click.Path(path_type=Path), default=None,
    help="Where each seed's tuning study is kept, so an interrupted run resumes.",
)
@click.option(
    "--report", type=click.Path(path_type=Path), default=None,
    help="Write the full result to this JSON file, even if a later endpoint fails.",
)
@json_option
@click.pass_context
def run_command(
    ctx: click.Context,
    endpoints: tuple[str, ...],
    select_all: bool,
    data_dir: Path | None,
    folds: int,
    seeds: int,
    tune: bool,
    time_budget: int | None,
    jobs: int,
    work_dir: Path | None,
    report: Path | None,
    as_json: bool,
) -> None:
    """Train and score on the published split, once per seed.

    The result is a mean and a standard deviation over seeds, which is the form the
    leaderboard reports. Nothing is written to the model directory: this command
    measures the protocol, and the bundles that ship come from 'admetus train'.

    Expect this to be expensive. --time-budget caps the tuner on ONE FOLD, and a
    fold is tuned for every seed: at the default 3 folds and 5 seeds that is fifteen
    tuning runs per endpoint, so the wall clock is fifteen times the budget, not
    five. --no-tune skips tuning entirely.
    """
    setup_logging(ctx)

    def body() -> None:
        selected = _selected(endpoints, select_all)
        if not 1 <= seeds <= len(bench.OFFICIAL_SEEDS):
            raise UsageError(f"--seeds must be between 1 and {len(bench.OFFICIAL_SEEDS)}")
        directory = Path(data_dir or default_data_dir())
        chosen_seeds = bench.OFFICIAL_SEEDS[:seeds]
        budget = time_budget if time_budget is not None else tuning.DEFAULT_TIME_BUDGET_SECONDS
        started = time.time()

        results = []
        failed = []
        for endpoint in selected:
            try:
                split = bench.load_official_split(endpoint, directory)
                result = run_endpoint(
                    endpoint, split, seeds=chosen_seeds, folds=folds, tune=tune,
                    time_budget=budget if tune else None, num_threads=jobs,
                    work_dir=Path(work_dir) / endpoint.key if work_dir else None,
                )
            except AdmetusError as exc:
                log.error("%s: %s", endpoint.key, exc)
                failed.append({"endpoint": endpoint.key, "error": str(exc)})
                continue
            results.append(result.to_dict())
            if not as_json:
                click.echo(result.describe())

        # Reported, not requested. `complete_protocol` used to read the five seeds off
        # the command line, so a run that asked for five and completed two still
        # claimed the protocol was complete.
        all_seeds_ran = bool(results) and all(r["n_seeds"] == len(bench.OFFICIAL_SEEDS) for r in results)
        payload = {
            "ok": not failed and all(r["n_seeds"] == len(chosen_seeds) for r in results),
            "n_comparable": sum(1 for r in results if r["comparable_with_leaderboard"]),
            "protocol": "official benchmark group: published train_val/test split per endpoint",
            "archive_sha256": bench.ARCHIVE_SHA256,
            "seeds": list(chosen_seeds),
            "seeds_requested": len(chosen_seeds),
            "complete_protocol": not failed and all_seeds_ran,
            "tuned_per_seed": tune,
            "time_budget_seconds_per_fold": budget if tune else None,
            "tuning_runs_per_endpoint": folds * len(chosen_seeds) if tune else 0,
            "n_endpoints": len(results),
            "results": results,
            "failed": failed,
            "seconds": round(time.time() - started, 1),
        }
        write_report(report, payload)
        if as_json:
            emit(payload)
        elif failed:
            for entry in failed:
                click.echo(f"FAILED {entry['endpoint']}: {entry['error']}", err=True)
        # A seed that did not complete is a failure of the request, whichever list it
        # landed in. Checking only the outer list reported ok:false and exited 0.
        short = [r for r in results if r["n_seeds"] != len(chosen_seeds)]
        if failed or short:
            for entry in short:
                click.echo(
                    f"INCOMPLETE {entry['endpoint']}: {entry['n_seeds']} of "
                    f"{len(chosen_seeds)} seed(s) completed",
                    err=True,
                )
            raise SystemExit(2)

    run_guarded(body, as_json=as_json)


__all__ = ["benchmark_group"]
