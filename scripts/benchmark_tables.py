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


#: The split every published number is measured on. A bundle from any other split
#: is excluded from both the table and the summary rather than quietly mixed in.
PUBLISHED_SPLIT_PREFIX = "random"


def _split_of(meta: dict) -> str:
    """How the bundle says its held-out set was chosen."""
    return str((meta.get("provenance") or {}).get("split", "unrecorded"))


def eligible(models_dir: Path) -> tuple[dict, dict[str, str]]:
    """Split the directory into bundles that may be published and bundles that may not.

    One rule, used by the table and by the summary beside it. They disagreed before:
    the table excluded a bundle measured on a scaffold split and the summary read it
    anyway, so a directory holding one legacy bundle and one current one published a
    one-row table under a sentence describing two endpoints -- and named the excluded
    one as the largest gap. With only legacy bundles the table had no measured row at
    all while the sentence still quoted figures.

    Returns the eligible bundles' metadata keyed by endpoint, and the excluded keys
    with the split each was measured on.
    """
    present = set(available(models_dir))
    keep: dict[str, dict] = {}
    excluded: dict[str, str] = {}
    for endpoint in all_endpoints():
        if endpoint.key not in present:
            continue
        meta = read_metadata(models_dir / endpoint.key)
        split = _split_of(meta)
        if split.startswith(PUBLISHED_SPLIT_PREFIX):
            keep[endpoint.key] = meta
        else:
            excluded[endpoint.key] = split
    return keep, excluded


def oof_comparison(models_dir: Path) -> dict[str, object]:
    """Compare each published bundle's pooled out-of-fold score with its held-out score.

    Reported as a sentence rather than a table, because the point is not the per
    endpoint number but the direction and size of the disagreement. Direction is
    taken from the endpoint's metric: for MAE a smaller value is better, so
    "flattering" means a *lower* out-of-fold number.

    Only the bundles the held-out table publishes are read, so the sentence always
    describes the rows printed above it.

    Generated, not remembered. The earlier pipeline's figure -- out-of-fold better at
    26 of 29 endpoints by a median of 0.045 -- described calibrated, pruned,
    single-booster predictions and does not survive the realignment: the shipped
    prediction is now the mean of every fold model, while the out-of-fold score is
    one booster per row, so out-of-fold is usually the pessimistic one.
    """
    keep, excluded = eligible(models_dir)
    gaps: list[tuple[str, float]] = []
    for endpoint in all_endpoints():
        meta = keep.get(endpoint.key)
        if meta is None:
            continue
        oof = (meta.get("cross_validation") or {}).get("pooled_oof_score")
        held = ((meta.get("metrics") or {}).get("primary") or {}).get("value")
        if oof is None or held is None:
            continue
        flatter = (oof - held) if endpoint.higher_is_better else (held - oof)
        gaps.append((endpoint.key, flatter))

    note = (
        f"; {len(excluded)} bundle(s) excluded, measured on another split"
        if excluded else ""
    )
    if not gaps:
        return {
            "n": 0,
            "n_flattering": 0,
            "median_abs_gap": None,
            "largest_flattering": None,
            "sentence": (
                "no bundle measured on the random holdout, so there is nothing to "
                f"compare{note}"
            ),
        }

    flattering = [g for g in gaps if g[1] > 0]
    median_gap = statistics.median(abs(g[1]) for g in gaps)
    # Only a positive gap is a flattering one. Reporting the maximum unconditionally
    # labelled a run where nothing flattered -- every out-of-fold score below its
    # held-out one -- as having a "largest flattering gap" of -0.0208.
    worst = max(gaps, key=lambda g: g[1])
    return {
        "n": len(gaps),
        "n_flattering": len(flattering),
        "median_abs_gap": round(median_gap, 4),
        "largest_flattering": worst if worst[1] > 0 else None,
        "sentence": (
            f"out-of-fold and held-out disagree at every endpoint; out-of-fold is the "
            f"more favourable of the two at {len(flattering)} of {len(gaps)}, and the "
            f"median absolute gap is {median_gap:.3f}{note}"
        ),
    }


def held_out_table(models_dir: Path, official_only: bool = False) -> str:
    """One row per endpoint: the held-out metric with its bootstrap interval."""
    keep, excluded = eligible(models_dir)
    lines = [
        "| Endpoint | Task | Metric | Held-out | 95% CI | n |",
        "|---|---|---|---|---|---|",
    ]
    for endpoint in all_endpoints():
        if official_only and not endpoint.in_official_group:
            continue
        task = "classification" if endpoint.is_classification else "regression"
        meta = keep.get(endpoint.key)
        if meta is None:
            # A score from another split does not belong in this column. A partially
            # retrained directory used to publish scaffold and random holdouts
            # together, so DILI's 0.8342 -- measured on a scaffold split -- appeared
            # under "random 20%".
            why = excluded.get(endpoint.key)
            suffix = f"  <!-- omitted: measured on {why} -->" if why else ""
            lines.append(f"| `{endpoint.key}` | {task} | {endpoint.metric} | - | - | - |{suffix}")
            continue
        primary = (meta.get("metrics") or {}).get("primary") or {}
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
    """A line naming how much of the registry is publishable, and what was left out.

    Counts the bundles the table prints, for the same reason the comparison does: a
    directory part-way through a retrain holds bundles from two different splits, and
    a count of everything present describes neither.
    """
    keep, excluded = eligible(models_dir)
    registry = all_endpoints()
    official = [e for e in registry if e.in_official_group and e.key in keep]
    line = (
        f"{len(keep)} of {len(registry)} endpoints measured on the random holdout, "
        f"{len(official)} of 22 in the official group."
    )
    if excluded:
        line += f" {len(excluded)} excluded as measured on another split: {', '.join(sorted(excluded))}."
    return line


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
    worst = comparison["largest_flattering"]
    tail = f"; largest flattering gap: {worst[0]} {worst[1]:+.4f}" if worst else ""
    print(f"<!-- {comparison['sentence']}{tail} -->")
    if args.areas:
        print("\n<!-- per area -->")
        for area, keys in AREAS:
            print(f"<!-- {area}: {', '.join(keys)} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
