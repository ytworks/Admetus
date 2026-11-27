"""``admetus data``: fetching and checking the public datasets."""

from __future__ import annotations

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
)
from admetus.data import loader
from admetus.data.download import clear_partials, fetch_dataset, human_bytes, verify_dataset
from admetus.data.registry import resolve_keys
from admetus.errors import NetworkError


@click.group("data")
def data_group() -> None:
    """Fetch and check the public datasets the models are trained on."""


@data_group.command("fetch")
@endpoint_option
@all_option
@data_dir_option
@click.option("--overwrite", is_flag=True, help="Fetch again even when the file is present.")
@click.option("--verify", "verify_only", is_flag=True, help="Only check what is already there.")
@click.option(
    "--accept-upstream-change", is_flag=True,
    help="Accept a dataset whose digest no longer matches the registry. Deliberate: any "
         "model trained on the pinned bytes becomes stale and must be retrained.",
)
@json_option
@click.pass_context
def fetch_command(
    ctx: click.Context,
    endpoints: tuple[str, ...],
    select_all: bool,
    data_dir: Path | None,
    overwrite: bool,
    accept_upstream_change: bool,
    verify_only: bool,
    as_json: bool,
) -> None:
    """Download datasets from Harvard Dataverse into a local directory.

    The files are published one per dataset, so this needs nothing beyond HTTP.

    Each dataset's digest is pinned in the registry, and a file that does not match
    is refused rather than adopted, so an upstream revision cannot silently become
    the new baseline and make a model's recorded provenance false. Taking a new
    version is deliberate: --accept-upstream-change.
    """
    setup_logging(ctx)

    def body() -> None:
        selected = resolve_keys(endpoints, select_all)
        directory = Path(data_dir or default_data_dir()).resolve()
        clear_partials(directory)

        results = []
        total = 0
        failures = 0
        for endpoint in selected:
            if verify_only:
                ok, detail = verify_dataset(endpoint.key, directory, endpoint.sha256)
                results.append({"endpoint": endpoint.key, "ok": ok, "detail": detail})
                failures += 0 if ok else 1
                if not as_json:
                    click.echo(f"{'ok  ' if ok else 'FAIL'} {endpoint.key:<22} {detail}")
                continue
            try:
                download = fetch_dataset(
                    endpoint.key, endpoint.download_url, directory, overwrite=overwrite,
                    expected_sha256=endpoint.sha256,
                    accept_upstream_change=accept_upstream_change,
                )
            except NetworkError as exc:
                failures += 1
                results.append({"endpoint": endpoint.key, "ok": False, "detail": str(exc)})
                if not as_json:
                    click.echo(f"FAIL {endpoint.key:<22} {exc}", err=True)
                continue
            total += download.n_bytes
            results.append({
                "endpoint": endpoint.key,
                "ok": True,
                "path": str(download.path),
                "bytes": download.n_bytes,
                "sha256": download.sha256,
                "pinned_sha256": endpoint.sha256,
                "from_cache": download.from_cache,
            })
            if not as_json:
                state = "cached " if download.from_cache else "fetched"
                click.echo(f"{state} {endpoint.key:<22} {human_bytes(download.n_bytes):>10}")

        if as_json:
            emit({
                "ok": failures == 0,
                "directory": str(directory),
                "n_endpoints": len(selected),
                "n_failed": failures,
                "total_bytes": total,
                "results": results,
            })
        elif not verify_only:
            click.echo(f"\n{human_bytes(total)} into {directory}")
        if failures:
            raise SystemExit(5 if not verify_only else 2)

    run_guarded(body, as_json=as_json)


@data_group.command("info")
@endpoint_option
@all_option
@data_dir_option
@json_option
@click.pass_context
def info_command(
    ctx: click.Context,
    endpoints: tuple[str, ...],
    select_all: bool,
    data_dir: Path | None,
    as_json: bool,
) -> None:
    """Describe the fetched datasets after normalization."""
    setup_logging(ctx)

    def body() -> None:
        selected = resolve_keys(endpoints, select_all)
        directory = Path(data_dir or default_data_dir())
        entries = []
        for endpoint in selected:
            dataset = loader.load(endpoint, directory)
            entry = {
                "endpoint": endpoint.key,
                "n_rows": len(dataset),
                "expected_rows": endpoint.expected_rows,
                "task": endpoint.task,
                "summary": dataset.summary(),
                "provenance": dataset.provenance,
            }
            if endpoint.is_classification:
                entry["pos_rate"] = dataset.pos_rate
            entries.append(entry)
            if not as_json:
                click.echo(dataset.summary())
        if as_json:
            emit({"ok": True, "directory": str(directory), "datasets": entries})

    run_guarded(body, as_json=as_json)


__all__ = ["data_group"]
