"""``admetus doctor``: can this install predict, and if not, what fixes it.

One call should let a caller decide between proceeding, fetching models, and
giving up, so every check that can block a run is reported here with its
remediation attached.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import click

from admetus._compat import lightgbm_status, python_status
from admetus.bundle.load import available
from admetus.cli.options import (
    cache_root,
    default_data_dir,
    default_models_dir,
    emit,
    json_option,
    models_dir_option,
    run_guarded,
    setup_logging,
)
from admetus.data.registry import all_endpoints
from admetus.errors import EXIT_INTERNAL, EXIT_NO_MODEL, EXIT_OK


def _chemistry_status() -> dict[str, object]:
    """Check that the feature space this install builds is the expected one."""
    try:
        import rdkit

        from admetus.bundle.save import golden_fingerprint
        from admetus.chem.pharm2d import get_feature_factory, get_sig_factory
        from admetus.chem.spec import EXPECTED_FAMILY_COUNT, PRESETS

        families = get_feature_factory().GetFeatureFamilies()
        checks = {
            "rdkit_version": rdkit.__version__,
            "n_feature_families": len(families),
            "families_expected": len(families) == EXPECTED_FAMILY_COUNT,
            "signature_sizes": {
                name: get_sig_factory(spec).GetSigSize()
                for name, spec in PRESETS.items()
                if spec.has_pharmacophore
            },
            "feature_widths": {name: spec.n_features for name, spec in PRESETS.items()},
            "golden_fingerprints": {
                name: golden_fingerprint(spec) for name, spec in PRESETS.items()
            },
        }
        sizes_ok = all(
            checks["signature_sizes"][name] == PRESETS[name].pharm_size
            for name in checks["signature_sizes"]
        )
        checks["ok"] = bool(checks["families_expected"] and sizes_ok)
        checks["error"] = None if checks["ok"] else "the pharmacophore feature space is not as expected"
        return checks
    except Exception as exc:  # noqa: BLE001 - doctor must report, not die
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "remediation": None}


def _cpu_status() -> dict[str, object]:
    """Report the core count the featurizer will plan around."""
    counter = getattr(os, "process_cpu_count", None)
    usable = counter() if counter else os.cpu_count()
    info: dict[str, object] = {"cpu_count": os.cpu_count(), "usable_cpus": usable}
    if platform.system() == "Darwin":
        # Apple silicon reports its core types as performance levels. A virtual
        # machine may expose one level or none, so each is probed independently
        # and a missing one is simply absent rather than an error.
        import subprocess

        for level, label in ((0, "performance_cores"), (1, "efficiency_cores")):
            try:
                out = subprocess.run(
                    ["sysctl", "-n", f"hw.perflevel{level}.logicalcpu"],
                    capture_output=True, text=True, check=False, timeout=5,
                )
            except (OSError, subprocess.SubprocessError):  # pragma: no cover - informational
                break
            if out.returncode == 0 and out.stdout.strip().isdigit():
                info[label] = int(out.stdout.strip())
    return info


def _directory_status(path: Path) -> dict[str, object]:
    """Describe a directory and how much it holds."""
    path = Path(path)
    if not path.is_dir():
        return {"path": str(path), "exists": False, "bytes": 0}
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return {"path": str(path), "exists": True, "bytes": total}


@click.command("doctor")
@json_option
@models_dir_option
@click.pass_context
def doctor_command(ctx: click.Context, as_json: bool, models_dir: Path | None) -> None:
    """Check whether this install can train and predict, and say what to fix."""
    setup_logging(ctx)

    def body() -> None:
        directory = models_dir or default_models_dir()
        lightgbm = lightgbm_status()
        chemistry = _chemistry_status()
        present = available(directory)
        registry = [e.key for e in all_endpoints()]

        report = {
            "python": python_status(),
            "lightgbm": lightgbm,
            "chemistry": chemistry,
            "cpu": _cpu_status(),
            "models": {
                **_directory_status(directory),
                "n_available": len(present),
                "n_registry": len(registry),
                "available": present,
                "missing": [k for k in registry if k not in set(present)],
            },
            "data": _directory_status(default_data_dir()),
            "cache": _directory_status(cache_root()),
        }

        broken = not lightgbm["ok"] or not chemistry.get("ok")
        if broken:
            code = EXIT_INTERNAL
            verdict = "install is not usable"
        elif not present:
            code = EXIT_NO_MODEL
            verdict = "usable, but no trained model is available"
        else:
            code = EXIT_OK
            verdict = "ready to predict"
        report["verdict"] = verdict
        report["ok"] = code == EXIT_OK

        if as_json:
            emit(report)
        else:
            click.echo(f"verdict: {verdict}\n")
            click.echo(f"python     {report['python']['version']}  {report['python']['machine']}")
            if lightgbm["ok"]:
                click.echo(f"lightgbm   {lightgbm['version']}  ok")
            else:
                click.echo(f"lightgbm   UNUSABLE: {lightgbm['error']}")
                if lightgbm["remediation"]:
                    click.echo(f"           fix with: {lightgbm['remediation']}")
            if chemistry.get("ok"):
                widths = ", ".join(f"{k}={v}" for k, v in chemistry["feature_widths"].items())
                click.echo(f"chemistry  rdkit {chemistry['rdkit_version']}  ok  ({widths})")
            else:
                click.echo(f"chemistry  PROBLEM: {chemistry.get('error')}")
            cpu = report["cpu"]
            cores = f"{cpu['usable_cpus']} usable of {cpu['cpu_count']}"
            # A macOS VM may expose one performance level and no efficiency level,
            # so neither key can be assumed from the presence of the other.
            split = [
                f"{cpu[key]} {label}"
                for key, label in (("performance_cores", "performance"), ("efficiency_cores", "efficiency"))
                if key in cpu
            ]
            if split:
                cores += f" ({', '.join(split)})"
            click.echo(f"cpu        {cores}")
            click.echo(
                f"models     {len(present)} of {len(registry)} available in {directory}"
            )
            if not present:
                click.echo("           train one with: admetus train --endpoint dili")
        if code != EXIT_OK:
            raise SystemExit(code)

    run_guarded(body, as_json=as_json)


__all__ = ["doctor_command"]
