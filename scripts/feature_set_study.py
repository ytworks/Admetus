"""Decide the feature set per endpoint by measurement rather than by the size rule alone.

The registry picks ``morgan_p2`` below 1,000 rows and ``morgan_p23`` above it. That
rule is a guess about where the 87,480-bit 3-point pharmacophore block starts paying
for itself. This script measures it: every endpoint is trained once per feature set and
ranked on its **pooled out-of-fold** score.

Three things make the comparison mean something:

* Selection uses the cross-validation score, never the held-out set. The held-out set
  is looked at once, when the endpoint is finally trained, and a feature set chosen
  against it would make that number a selection score instead of a measurement.
* Tuning is off. A fixed parameter set is the same for all three candidates, so the
  comparison is of feature spaces and not of how much tuner budget each one happened
  to get. The chosen set is tuned afterwards, in the real training run.
* Each candidate runs at two seeds. The seed moves the fold assignment, so the spread
  between seeds is a floor on how much of any difference is noise. A challenger has to
  beat the size rule's default by more than that floor to be adopted, which is what
  stops the study from chasing fold luck.

Writes a JSON record and a Markdown table to the output directory. The table goes into
``docs/benchmark.md``; the adopted overrides go into ``endpoints.yaml`` by hand, so the
decision stays reviewable in a diff.

Usage::

    python scripts/feature_set_study.py --out study/ --workers 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from admetus.data.registry import HIGHER_IS_BETTER, all_endpoints

#: Every candidate feature space, narrowest first.
FEATURE_SETS = ("morgan", "morgan_p2", "morgan_p23")

#: Seeds per candidate. Two is enough for a floor on fold noise and keeps the study
#: at 174 training runs rather than 261.
SEEDS = (0, 1)

#: A challenger must beat the default by more than the measured seed spread *and* by
#: this much, so that an endpoint whose two seeds happen to agree closely cannot be
#: switched on a third-decimal difference.
MIN_MARGIN = 0.005


def run_one(args: tuple[str, str, int, str, str, int]) -> dict:
    """Train one (endpoint, feature set, seed) and return its record.

    Runs as a subprocess so that a crash in one candidate cannot take the study with
    it, and so RDKit's and LightGBM's thread pools start clean each time.
    """
    key, feature_set, seed, data_dir, out_dir, jobs = args
    models = Path(out_dir) / "models" / f"{feature_set}-seed{seed}"
    started = time.time()
    completed = subprocess.run(
        [
            sys.executable, "-m", "admetus", "--quiet", "train",
            "--endpoint", key,
            "--feature-set", feature_set,
            "--seed", str(seed),
            "--no-tune",
            "--force",
            "--jobs", str(jobs),
            "--bootstrap", "200",
            "--data-dir", data_dir,
            "--models-dir", str(models),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "endpoint": key,
        "feature_set": feature_set,
        "seed": seed,
        "wall_seconds": round(time.time() - started, 1),
        "ok": False,
        "score": None,
        "n_features": None,
        "error": None,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        record["error"] = (completed.stderr or completed.stdout or "no output")[-400:]
        return record
    trained = payload.get("trained") or []
    if not trained:
        failed = payload.get("failed") or [{}]
        record["error"] = str(failed[0].get("error") or payload.get("error") or "no result")[:400]
        return record
    entry = trained[0]
    record.update(
        ok=True,
        score=entry["pooled_oof_score"],
        n_features=entry["n_features"],
        train_seconds=entry["seconds"],
    )
    return record


def decide(key: str, default_set: str, scores: dict[str, list[float]], metric: str) -> dict:
    """Pick a feature set for one endpoint from its per-seed scores.

    Returns a record naming the winner, whether it displaces the default, and the
    margin and noise floor the decision rests on.
    """
    higher = HIGHER_IS_BETTER[metric]
    sign = 1.0 if higher else -1.0

    means: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for name, values in scores.items():
        usable = [v for v in values if v is not None]
        if not usable:
            continue
        means[name] = sum(usable) / len(usable)
        spreads[name] = max(usable) - min(usable) if len(usable) > 1 else float("nan")

    if default_set not in means:
        return {
            "endpoint": key, "metric": metric, "default": default_set, "chosen": default_set,
            "changed": False, "reason": "the default did not complete; left as it is",
            "means": means, "seed_spreads": spreads,
        }

    best = max(means, key=lambda name: sign * means[name])
    margin = sign * (means[best] - means[default_set])
    noise = max(
        [s for s in (spreads.get(best), spreads.get(default_set)) if s is not None and s == s],
        default=0.0,
    )
    floor = max(noise, MIN_MARGIN)
    changed = best != default_set and margin > floor
    if best == default_set:
        reason = "the size rule's default is also the best measured"
    elif changed:
        reason = f"beats the default by {margin:.4f}, above the {floor:.4f} noise floor"
    else:
        reason = f"ahead by only {margin:.4f}, within the {floor:.4f} noise floor"
    return {
        "endpoint": key, "metric": metric, "default": default_set,
        "best_measured": best, "chosen": best if changed else default_set,
        "changed": changed, "margin": margin, "noise_floor": floor,
        "reason": reason, "means": means, "seed_spreads": spreads,
    }


def markdown(decisions: list[dict]) -> str:
    """Render the study as the table that goes into the benchmark document."""
    lines = [
        "| endpoint | metric | default | morgan | morgan_p2 | morgan_p23 | chosen | why |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in decisions:
        cells = []
        for name in FEATURE_SETS:
            value = entry["means"].get(name)
            text = "-" if value is None else f"{value:.4f}"
            if entry.get("best_measured") == name:
                text = f"**{text}**"
            cells.append(text)
        lines.append(
            f"| `{entry['endpoint']}` | {entry['metric']} | `{entry['default']}` | "
            + " | ".join(cells)
            + f" | `{entry['chosen']}` | {entry['reason']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Where to write the study.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=3, help="Concurrent training runs.")
    parser.add_argument("--jobs", type=int, default=4, help="LightGBM threads per run.")
    parser.add_argument("--endpoint", action="append", default=None, help="Limit to these keys.")
    args = parser.parse_args()

    endpoints = all_endpoints()
    if args.endpoint:
        wanted = set(args.endpoint)
        endpoints = [e for e in endpoints if e.key in wanted]
    args.out.mkdir(parents=True, exist_ok=True)

    tasks = [
        (e.key, feature_set, seed, str(args.data_dir), str(args.out), args.jobs)
        for e in endpoints
        for feature_set in FEATURE_SETS
        for seed in SEEDS
    ]
    print(f"{len(tasks)} runs: {len(endpoints)} endpoints x {len(FEATURE_SETS)} feature sets x {len(SEEDS)} seeds")

    records: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_one, tasks):
            records.append(record)
            status = f"{record['score']:.4f}" if record["ok"] else f"FAILED {record['error']}"
            print(
                f"[{len(records):>3}/{len(tasks)}] {record['endpoint']:<22} "
                f"{record['feature_set']:<11} seed={record['seed']} "
                f"{record['wall_seconds']:>6.1f}s  {status}",
                flush=True,
            )

    decisions = []
    for endpoint in endpoints:
        scores = {
            name: [
                r["score"] for r in records
                if r["endpoint"] == endpoint.key and r["feature_set"] == name
            ]
            for name in FEATURE_SETS
        }
        decisions.append(
            decide(endpoint.key, endpoint.feature_set(endpoint.expected_rows), scores, endpoint.metric)
        )

    payload = {
        "selection_metric": "pooled out-of-fold score, biased by construction",
        "tuning": "off, fixed parameters shared by every candidate",
        "seeds": list(SEEDS),
        "min_margin": MIN_MARGIN,
        "wall_seconds": round(time.time() - started, 1),
        "runs": records,
        "decisions": decisions,
    }
    (args.out / "feature_set_study.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "feature_set_study.md").write_text(markdown(decisions), encoding="utf-8")

    changed = [d for d in decisions if d["changed"]]
    print(f"\n{len(changed)} endpoint(s) would change feature set:")
    for entry in changed:
        print(f"  {entry['endpoint']:<22} {entry['default']} -> {entry['chosen']}  ({entry['reason']})")
    failures = [r for r in records if not r["ok"]]
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for record in failures[:10]:
            print(f"  {record['endpoint']} {record['feature_set']} seed={record['seed']}: {record['error']}")
    print(f"\nwrote {args.out / 'feature_set_study.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
