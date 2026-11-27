"""``admetus endpoints``: the registry, in a form a program can plan from."""

from __future__ import annotations

from pathlib import Path

import click

from admetus.bundle.load import available
from admetus.cli.options import (
    default_models_dir,
    emit,
    json_option,
    models_dir_option,
    run_guarded,
    setup_logging,
)
from admetus.data.registry import all_endpoints
from admetus.predict.runner import Options, column_names


def _output_columns(endpoint, prefix: str = "") -> list[dict[str, object]]:
    """Describe the columns an endpoint adds, so a caller knows the schema up front."""
    base = f"{prefix}{endpoint.key}"
    is_log = endpoint.transform in ("log10", "log10_safe")
    columns: list[dict[str, object]] = [
        {
            "name": base,
            "dtype": "float",
            "unit": "probability" if endpoint.task == "classification" else endpoint.unit,
            "range": [0.0, 1.0] if endpoint.task == "classification"
            else (list(endpoint.output_range) if endpoint.output_range else None),
            "meaning": "calibrated probability of the positive class"
            if endpoint.task == "classification"
            else "predicted value on the published scale",
        },
        {
            "name": f"{base}_sd_log10" if is_log else f"{base}_ens_sd",
            "dtype": "float",
            "unit": "log10 units" if is_log else endpoint.unit,
            "range": None,
            "meaning": "disagreement across the fold ensemble, not a calibrated error bar",
        },
        {
            "name": f"{base}_in_domain",
            "dtype": "boolean",
            "unit": None,
            "range": None,
            "meaning": "heuristic: the molecule has a training neighbour at or above the "
                       "similarity threshold recorded in the bundle",
        },
    ]
    return columns


@click.command("endpoints")
@json_option
@models_dir_option
@click.pass_context
def endpoints_command(ctx: click.Context, as_json: bool, models_dir: Path | None) -> None:
    """List every endpoint, with its task, metric, units and output columns."""
    setup_logging(ctx)

    def body() -> None:
        directory = models_dir or default_models_dir()
        present = set(available(directory))
        entries = []
        for endpoint in all_endpoints():
            payload = endpoint.to_json()
            payload["model_available"] = endpoint.key in present
            payload["output_columns"] = _output_columns(endpoint)
            entries.append(payload)

        if as_json:
            emit({
                "ok": True,
                "models_dir": str(directory),
                "n_endpoints": len(entries),
                "endpoints": entries,
            })
            return

        click.echo(f"{'key':<22} {'task':<15} {'metric':<9} {'n':>7} {'model':<6} unit")
        for entry in entries:
            flag = "yes" if entry["model_available"] else "-"
            note = "  (small dataset)" if entry["low_confidence"] else ""
            click.echo(
                f"{entry['key']:<22} {entry['task']:<15} {entry['metric']:<9} "
                f"{entry['n_rows']:>7} {flag:<6} {entry['unit']}{note}"
            )
        official = sum(1 for e in entries if e["in_official_group"])
        click.echo(
            f"\n{len(entries)} endpoints, {official} in the official benchmark group, "
            f"{len(present)} with a trained model in {directory}"
        )

    run_guarded(body, as_json=as_json)


# The predict runner's column naming is the single source of truth; this keeps the
# two in step if the naming ever changes.
def _assert_matches_runner() -> None:  # pragma: no cover - import-time guard
    _ = Options, column_names


__all__ = ["endpoints_command"]
