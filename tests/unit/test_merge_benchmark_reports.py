"""Merging a sharded protocol run: what it refuses rather than resolves.

Five seeds over 22 endpoints is hours of wall clock, so the run is split across
parallel processes and their reports are combined to make one table. Two ways that
could quietly produce a table nobody could defend: rows measured under different
budgets sitting in one column, and an endpoint scored twice with one of the two
measurements chosen. Both are refused.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_benchmark_reports.py"


@pytest.fixture(scope="module")
def merger():
    spec = importlib.util.spec_from_file_location("merge_benchmark_reports", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_benchmark_reports"] = module
    spec.loader.exec_module(module)
    return module


def _report(tmp_path: Path, name: str, endpoints: list[str], *, budget: int = 60,
            n_seeds: int = 5, failed: list[str] | None = None) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({
        "ok": True,
        "protocol": "official benchmark group: published train_val/test split per endpoint",
        "archive_sha256": "abc",
        "seeds": [1, 2, 3, 4, 5],
        "tuned_per_seed": True,
        "time_budget_seconds_per_fold": budget,
        "schema_version": 1,
        "admetus_version": "0.1.0.dev0",
        "seconds": 10.0,
        "results": [
            {"endpoint": key, "n_seeds": n_seeds, "mean": 0.9, "sd": 0.01,
             "comparable_with_leaderboard": True}
            for key in endpoints
        ],
        "failed": [{"endpoint": key, "error": "no split"} for key in (failed or [])],
    }), encoding="utf-8")
    return path


def test_shards_covering_different_endpoints_merge(merger, tmp_path):
    a = _report(tmp_path, "a.json", ["dili", "herg"])
    b = _report(tmp_path, "b.json", ["caco2"])

    reports = merger._load([a, b])
    assert merger._disagreements(reports) == []

    merged = merger.merge(reports)
    assert [r["endpoint"] for r in merged["results"]] == ["dili", "herg", "caco2"]
    assert merged["n_endpoints"] == 3
    assert merged["n_comparable"] == 3
    assert merged["ok"] is True
    assert merged["complete_protocol"] is True
    assert merged["merged_from"] == ["a.json", "b.json"]
    # The shards ran in parallel, so the sum is compute spent and the report says so.
    assert merged["seconds"] == 20.0
    assert "not the time taken" in merged["merged_note"]


def test_a_different_budget_between_shards_is_refused(merger, tmp_path):
    a = _report(tmp_path, "a.json", ["dili"], budget=60)
    b = _report(tmp_path, "b.json", ["caco2"], budget=240)

    problems = merger._disagreements(merger._load([a, b]))
    assert any("time_budget_seconds_per_fold differs" in p for p in problems)


def test_an_endpoint_scored_twice_is_refused_not_chosen_between(merger, tmp_path):
    a = _report(tmp_path, "a.json", ["dili", "herg"])
    b = _report(tmp_path, "b.json", ["dili"])

    problems = merger._disagreements(merger._load([a, b]))
    assert any("dili was scored by both" in p for p in problems)
    assert any("choosing one" in p for p in problems)


def test_the_summary_is_recomputed_not_copied(merger, tmp_path):
    """One shard's `ok: true` must not describe a run where another shard failed."""
    a = _report(tmp_path, "a.json", ["dili"])
    b = _report(tmp_path, "b.json", ["caco2"], failed=["solubility"])

    merged = merger.merge(merger._load([a, b]))
    assert merged["ok"] is False
    assert merged["complete_protocol"] is False
    assert [f["endpoint"] for f in merged["failed"]] == ["solubility"]


def test_a_shard_that_ran_short_makes_the_protocol_incomplete(merger, tmp_path):
    a = _report(tmp_path, "a.json", ["dili"])
    b = _report(tmp_path, "b.json", ["caco2"], n_seeds=2)

    merged = merger.merge(merger._load([a, b]))
    assert merged["complete_protocol"] is False
    assert merged["ok"] is False
