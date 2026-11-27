"""Combine the reports of a sharded ``admetus benchmark run`` into one.

Five seeds over 22 endpoints is hours of wall clock, so the run is split across
parallel processes, one subset of endpoints each. `benchmark_tables.py --official`
reads a single report, and the table it prints has to be reproducible from a named
script rather than assembled by hand.

What this does **not** do:

* It never merges two reports that scored the same endpoint. That would mean picking
  one of two measurements, and picking is what the whole project refuses to do with
  numbers. It is reported as a conflict and nothing is written.
* It never merges runs with different protocol settings. The seeds, the tuning flag,
  the per-fold budget and the split archive's digest all have to agree, because a
  table whose rows were measured under different budgets is one number per row and
  no way to tell which.
* It recomputes every summary field from the merged results rather than copying one
  shard's, so `ok` and `complete_protocol` describe the whole run.

Usage::

    python scripts/merge_benchmark_reports.py a.json b.json c.json -o official.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Settings every shard must agree on for their rows to belong in one table.
SHARED_SETTINGS = (
    "protocol",
    "archive_sha256",
    "seeds",
    "tuned_per_seed",
    "time_budget_seconds_per_fold",
)


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source"] = str(path)
        reports.append(payload)
    return reports


def _disagreements(reports: list[dict[str, Any]]) -> list[str]:
    """Settings on which the shards do not agree, and endpoints scored twice."""
    problems = []
    first = reports[0]
    for key in SHARED_SETTINGS:
        values = {json.dumps(r.get(key), sort_keys=True) for r in reports}
        if len(values) > 1:
            problems.append(
                f"{key} differs between shards: " + ", ".join(
                    f"{Path(r['_source']).name}={json.dumps(r.get(key), sort_keys=True)}"
                    for r in reports
                )
            )
    _ = first
    seen: dict[str, str] = {}
    for report in reports:
        for result in report.get("results", []):
            key = result["endpoint"]
            if key in seen:
                problems.append(
                    f"{key} was scored by both {Path(seen[key]).name} and "
                    f"{Path(report['_source']).name}; merging would mean choosing one"
                )
            seen[key] = report["_source"]
    return problems


def merge(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """One report describing every shard's work, with the summary recomputed."""
    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for report in reports:
        results.extend(report.get("results", []))
        failed.extend(report.get("failed", []))

    n_seeds = len(reports[0].get("seeds") or [])
    all_seeds_ran = bool(results) and all(r["n_seeds"] == n_seeds for r in results)
    merged = {
        key: reports[0].get(key) for key in SHARED_SETTINGS
    }
    merged.update({
        "ok": not failed and all_seeds_ran,
        "n_comparable": sum(1 for r in results if r.get("comparable_with_leaderboard")),
        "seeds_requested": n_seeds,
        "complete_protocol": not failed and all_seeds_ran,
        "n_endpoints": len(results),
        "results": results,
        "failed": failed,
        "seconds": round(sum(float(r.get("seconds") or 0) for r in reports), 1),
        "merged_from": [Path(r["_source"]).name for r in reports],
        "merged_note": (
            "wall clock is the sum of the shards, which ran in parallel; it is the "
            "compute spent, not the time taken"
        ),
        "schema_version": reports[0].get("schema_version"),
        "admetus_version": reports[0].get("admetus_version"),
    })
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="Shard reports to merge.")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    reports = _load(args.reports)
    problems = _disagreements(reports)
    if problems:
        for problem in problems:
            print(f"refused: {problem}", file=sys.stderr)
        return 2

    merged = merge(reports)
    args.out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.out}: {merged['n_endpoints']} endpoint(s) from {len(reports)} shard(s), "
        f"{merged['n_comparable']} leaderboard-comparable, "
        f"complete_protocol={merged['complete_protocol']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
