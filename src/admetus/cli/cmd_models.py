"""``admetus models``: what is installed, and whether it is intact."""

from __future__ import annotations

from pathlib import Path

import click

from admetus.bundle.load import available, read_metadata, verify
from admetus.cli.options import (
    all_option,
    default_models_dir,
    emit,
    endpoint_option,
    json_option,
    models_dir_option,
    run_guarded,
    setup_logging,
)
from admetus.errors import EXIT_NO_MODEL


@click.group("models")
def models_group() -> None:
    """Inspect and check the trained model bundles."""


@models_group.command("list")
@models_dir_option
@click.option("--verify", "do_verify", is_flag=True, help="Also check each bundle's integrity.")
@json_option
@click.pass_context
def list_command(ctx: click.Context, models_dir: Path | None, do_verify: bool, as_json: bool) -> None:
    """List the model bundles that are present.

    Works with no network: it reads the local directory only.
    """
    setup_logging(ctx)

    def body() -> None:
        directory = Path(models_dir or default_models_dir())
        keys = available(directory)
        entries = []
        problems = 0
        for key in keys:
            path = directory / key
            meta = read_metadata(path)
            entry = {
                "endpoint": key,
                "created_at": meta.get("created_at"),
                "task": meta["endpoint"]["task"],
                "metric": meta["endpoint"]["metric"],
                "n_boosters": meta["model"]["n_boosters"],
                "n_features": meta["featurizer"]["n_features"],
                "feature_set": meta["featurizer"]["spec"]["feature_set"],
                "primary": (meta.get("metrics") or {}).get("primary"),
                "spec_sha256": meta["featurizer"]["spec_sha256"],
            }
            if do_verify:
                ok, found = verify(path)
                entry["verified"] = ok
                entry["problems"] = found
                problems += 0 if ok else 1
            entries.append(entry)

        if as_json:
            emit({
                "ok": problems == 0,
                "models_dir": str(directory),
                "n_models": len(entries),
                "n_problems": problems,
                "models": entries,
            })
        else:
            if not entries:
                click.echo(f"no model bundles in {directory}")
                click.echo("train one with: admetus train --endpoint dili")
                raise SystemExit(EXIT_NO_MODEL)
            click.echo(f"{'endpoint':<22} {'metric':<9} {'score':<24} {'features':>9} {'set':<12} created")
            for entry in entries:
                primary = entry["primary"] or {}
                score = (
                    f"{primary.get('value', float('nan')):.4f} "
                    f"[{primary.get('ci_low', float('nan')):.4f}, {primary.get('ci_high', float('nan')):.4f}]"
                    if primary else "-"
                )
                mark = "" if not do_verify else ("  ok" if entry["verified"] else "  PROBLEM")
                click.echo(
                    f"{entry['endpoint']:<22} {entry['metric']:<9} {score:<24} "
                    f"{entry['n_features']:>9} {entry['feature_set']:<12} "
                    f"{str(entry['created_at'])[:10]}{mark}"
                )
            click.echo(f"\n{len(entries)} bundle(s) in {directory}")
        if problems:
            raise SystemExit(EXIT_NO_MODEL)

    run_guarded(body, as_json=as_json)


@models_group.command("show")
@endpoint_option
@all_option
@models_dir_option
@json_option
@click.pass_context
def show_command(
    ctx: click.Context, endpoints: tuple[str, ...], select_all: bool,
    models_dir: Path | None, as_json: bool,
) -> None:
    """Print a bundle's full record, including how it was trained and measured."""
    setup_logging(ctx)

    def body() -> None:
        from admetus.data.registry import resolve_keys

        directory = Path(models_dir or default_models_dir())
        selected = resolve_keys(endpoints, select_all)
        records = {e.key: read_metadata(directory / e.key) for e in selected}
        if as_json:
            emit({"ok": True, "models_dir": str(directory), "models": records})
        else:
            import json as _json

            click.echo(_json.dumps(records, indent=2, sort_keys=True, default=str))

    run_guarded(body, as_json=as_json)


__all__ = ["models_group"]
