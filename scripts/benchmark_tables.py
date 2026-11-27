"""Render the accuracy tables from the bundles, rather than transcribing them.

The two READMEs have to carry the same numbers -- CI compares every numeric cell of
their tables -- and 29 endpoints copied by hand across two documents is a lot of
chances to be wrong in a way nobody notices. So the numbers are generated from the
bundles themselves, and from the official protocol's report where one exists.

Three things this will not do:

* It never recomputes a metric. The held-out number is read from the bundle, where it
  was written the one time the held-out set was looked at.
* It never fills a gap. An endpoint with no bundle is a dash, and an endpoint with no
  official run is absent from the official table.
* It never quotes a cross-validation score. Those rank candidates; `docs/benchmark.md`
  reports them under the feature-set study, labelled, and nowhere else.

Usage::

    python scripts/benchmark_tables.py --models-dir models --official official.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admetus.bundle.load import available, read_metadata
from admetus.data.registry import all_endpoints

#: How many decimals each metric is quoted to. MAE on `ppbr` is a percentage point
#: and on `caco2` a log unit, so a single width would be wrong for one of them.
DECIMALS = {"auroc": 4, "auprc": 4, "spearman": 4, "mae": 4}

AREAS = (
    ("Absorption", ("caco2", "pampa", "hia", "pgp", "bioavailability")),
    ("Distribution", ("lipophilicity", "solubility", "bbb", "ppbr", "vdss")),
    (
        "Metabolism",
        (
            "cyp1a2_veith", "cyp2c9_veith", "cyp2c19_veith", "cyp2d6_veith", "cyp3a4_veith",
            "cyp2c9_substrate", "cyp2d6_substrate", "cyp3a4_substrate",
        ),
    ),
    ("Excretion", ("half_life", "clearance_hepatocyte", "clearance_microsome")),
    (
        "Toxicity",
        ("ld50", "herg", "herg_karim", "ames", "dili", "carcinogens", "skin_reaction", "tox21_nr_ar"),
    ),
)


def _format(value: float | None, metric: str) -> str:
    if value is None:
        return "-"
    return f"{value:.{DECIMALS.get(metric, 4)}f}"


def held_out_table(models_dir: Path, official_only: bool = False) -> str:
    """One row per endpoint: the held-out metric with its bootstrap interval."""
    present = set(available(models_dir))
    lines = [
        "| Endpoint | Task | Metric | Held-out | 95% CI | n | Feature set |",
        "|---|---|---|---|---|---|---|",
    ]
    for endpoint in all_endpoints():
        if official_only and not endpoint.in_official_group:
            continue
        task = "classification" if endpoint.is_classification else "regression"
        if endpoint.key not in present:
            lines.append(f"| `{endpoint.key}` | {task} | {endpoint.metric} | - | - | - | - |")
            continue
        meta = read_metadata(models_dir / endpoint.key)
        primary = (meta.get("metrics") or {}).get("primary") or {}
        spec = (meta.get("featurizer") or {}).get("spec") or {}
        lines.append(
            f"| `{endpoint.key}` | {task} | {endpoint.metric} | "
            f"{_format(primary.get('value'), endpoint.metric)} | "
            f"[{_format(primary.get('ci_low'), endpoint.metric)}, "
            f"{_format(primary.get('ci_high'), endpoint.metric)}] | "
            f"{primary.get('n', '-')} | `{spec.get('feature_set', '-')}` |"
        )
    return "\n".join(lines)


def official_table(report: Path) -> str:
    """One row per endpoint of the official group: mean and spread over the seeds."""
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    by_key = {r["endpoint"]: r for r in payload.get("results", [])}
    lines = [
        "| Endpoint | Metric | Official protocol | Seeds | Test rows |",
        "|---|---|---|---|---|",
    ]
    for endpoint in all_endpoints():
        entry = by_key.get(endpoint.key)
        if not entry:
            continue
        seeds = entry.get("seeds") or []
        n_test = seeds[0]["n_test"] if seeds else "-"
        lines.append(
            f"| `{endpoint.key}` | {endpoint.metric} | "
            f"{_format(entry['mean'], endpoint.metric)} +/- "
            f"{_format(entry['sd'], endpoint.metric)} | "
            f"{entry['n_seeds']} | {n_test} |"
        )
    return "\n".join(lines)


def coverage_table(models_dir: Path) -> str:
    """Interval coverage for the regression endpoints, against the nominal level."""
    present = set(available(models_dir))
    rows = []
    for endpoint in all_endpoints():
        if endpoint.is_classification or endpoint.key not in present:
            continue
        meta = read_metadata(models_dir / endpoint.key)
        metrics = meta.get("metrics") or {}
        achieved = metrics.get("interval_coverage")
        if achieved is None:
            continue
        nominal = metrics.get("interval_level", 0.9)
        rows.append(f"| `{endpoint.key}` | {nominal:.0%} | {achieved:.1%} | {metrics['n_test']} |")
    if not rows:
        return ""
    return "\n".join(
        ["| Endpoint | Nominal | Measured coverage | Held-out rows |", "|---|---|---|---|", *rows]
    )


def summary(models_dir: Path) -> str:
    """A line naming how much of the registry is actually trained."""
    present = available(models_dir)
    registry = all_endpoints()
    official = [e for e in registry if e.in_official_group and e.key in set(present)]
    return (
        f"{len(present)} of {len(registry)} endpoints trained, "
        f"{len(official)} of 22 in the official group."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--official", type=Path, default=None, help="Report from 'admetus benchmark run'.")
    parser.add_argument("--areas", action="store_true", help="Also print the per-area grouping.")
    args = parser.parse_args()

    print(f"<!-- {summary(args.models_dir)} -->\n")
    print("### Held-out scaffold split\n")
    print(held_out_table(args.models_dir))
    coverage = coverage_table(args.models_dir)
    if coverage:
        print("\n### Prediction interval coverage\n")
        print(coverage)
    if args.official:
        print("\n### Official protocol\n")
        print(official_table(args.official))
    if args.areas:
        print("\n<!-- per area -->")
        for area, keys in AREAS:
            print(f"<!-- {area}: {', '.join(keys)} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
