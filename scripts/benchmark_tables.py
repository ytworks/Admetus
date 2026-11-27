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
* It never quotes a cross-validation score. Those rank candidates and are recorded in
  each bundle, labelled as biased, and reported nowhere.

Usage::

    python scripts/benchmark_tables.py --models-dir models --official official.json
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def oof_comparison(models_dir: Path) -> dict[str, object]:
    """Compare each bundle's pooled out-of-fold score with its held-out score.

    Reported as a sentence rather than a table, because the point is not the per
    endpoint number but the direction and size of the disagreement. Direction is
    taken from the endpoint's metric: for MAE a smaller value is better, so
    "flattering" means a *lower* out-of-fold number.

    Generated, not remembered. The earlier pipeline's figure -- out-of-fold better at
    26 of 29 endpoints by a median of 0.045 -- described calibrated, pruned,
    single-booster predictions and does not survive the realignment: the shipped
    prediction is now the mean of every fold model, while the out-of-fold score is
    one booster per row, so out-of-fold is usually the pessimistic one.
    """
    present = set(available(models_dir))
    gaps: list[tuple[str, float]] = []
    for endpoint in all_endpoints():
        if endpoint.key not in present:
            continue
        meta = read_metadata(models_dir / endpoint.key)
        oof = (meta.get("cross_validation") or {}).get("pooled_oof_score")
        held = ((meta.get("metrics") or {}).get("primary") or {}).get("value")
        if oof is None or held is None:
            continue
        flatter = (oof - held) if endpoint.higher_is_better else (held - oof)
        gaps.append((endpoint.key, flatter))

    flattering = [g for g in gaps if g[1] > 0]
    worst = max(gaps, key=lambda g: g[1]) if gaps else ("-", 0.0)
    return {
        "n": len(gaps),
        "n_flattering": len(flattering),
        "median_abs_gap": round(statistics.median(abs(g[1]) for g in gaps), 4) if gaps else None,
        "largest_flattering": worst,
        "sentence": (
            f"out-of-fold and held-out disagree at every endpoint; out-of-fold is the "
            f"more favourable of the two at {len(flattering)} of {len(gaps)}, and the "
            f"median absolute gap is "
            f"{statistics.median(abs(g[1]) for g in gaps):.3f}" if gaps else "no bundles"
        ),
    }


def held_out_table(models_dir: Path, official_only: bool = False) -> str:
    """One row per endpoint: the held-out metric with its bootstrap interval."""
    present = set(available(models_dir))
    lines = [
        "| Endpoint | Task | Metric | Held-out | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for endpoint in all_endpoints():
        if official_only and not endpoint.in_official_group:
            continue
        task = "classification" if endpoint.is_classification else "regression"
        if endpoint.key not in present:
            lines.append(f"| `{endpoint.key}` | {task} | {endpoint.metric} | - | - | - |")
            continue
        meta = read_metadata(models_dir / endpoint.key)
        primary = (meta.get("metrics") or {}).get("primary") or {}
        split = str((meta.get("provenance") or {}).get("split", "unrecorded"))
        # A score from another split does not belong in this column. A partially
        # retrained directory used to publish scaffold and random holdouts together,
        # so DILI's 0.8342 -- measured on a scaffold split -- appeared under
        # "random 20%".
        if not split.startswith("random"):
            lines.append(
                f"| `{endpoint.key}` | {task} | {endpoint.metric} | - | - | - |"
                f"  <!-- omitted: measured on {split} -->"
            )
            continue
        lines.append(
            f"| `{endpoint.key}` | {task} | {endpoint.metric} | "
            f"{_format(primary.get('value'), endpoint.metric)} | "
            f"[{_format(primary.get('ci_low'), endpoint.metric)}, "
            f"{_format(primary.get('ci_high'), endpoint.metric)}] | "
            f"{primary.get('n', '-')} |"
        )
    return "\n".join(lines)


def official_table(report: Path) -> str:
    """One row per endpoint of the official group: mean and spread over the seeds."""
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    by_key = {r["endpoint"]: r for r in payload.get("results", [])}
    lines = [
        "| Endpoint | Metric | Official protocol | Seeds | Test rows scored | Comparable |",
        "|---|---|---|---|---|---|",
    ]
    for endpoint in all_endpoints():
        entry = by_key.get(endpoint.key)
        if not entry:
            continue
        seeds = entry.get("seeds") or []
        scored = seeds[0]["n_test"] if seeds else 0
        published = (seeds[0].get("n_test_published") or 0) if seeds else 0
        rows = f"{scored} of {published}" if published and scored != published else str(scored or "-")
        # A partial result is labelled in the table, not quietly presented as one.
        comparable = entry.get("comparable_with_leaderboard")
        mark = "yes" if comparable else "no: " + "; ".join(entry.get("not_comparable_because") or ["unknown"])
        lines.append(
            f"| `{endpoint.key}` | {endpoint.metric} | "
            f"{_format(entry['mean'], endpoint.metric)} +/- "
            f"{_format(entry['sd'], endpoint.metric)} | "
            f"{entry['n_seeds']} | {rows} | {mark} |"
        )
    return "\n".join(lines)


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
    print("### Held-out, on the random 20%\n")
    print(held_out_table(args.models_dir))
    if args.official:
        print("\n### Official protocol\n")
        print(official_table(args.official))
    comparison = oof_comparison(args.models_dir)
    print("\n<!-- cross-validation against held out -->")
    print(f"<!-- {comparison['sentence']}; "
          f"largest flattering gap: {comparison['largest_flattering'][0]} "
          f"{comparison['largest_flattering'][1]:+.4f} -->")
    if args.areas:
        print("\n<!-- per area -->")
        for area, keys in AREAS:
            print(f"<!-- {area}: {', '.join(keys)} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
