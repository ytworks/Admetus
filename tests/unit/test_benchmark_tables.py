"""The table generator: one eligibility rule for the table, the summary and the note.

The script exists so the accuracy numbers are generated from the bundles rather than
typed into two READMEs, which only helps if everything it prints describes the same
set of bundles. A directory part-way through a retrain holds bundles from two
different splits, and that is where the three disagreed: the table dropped a bundle
measured on another split, while the summary counted it and the cross-validation note
read its scores and named it as the largest gap.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tables.py"


@pytest.fixture(scope="module")
def tables():
    spec = importlib.util.spec_from_file_location("benchmark_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_tables"] = module
    spec.loader.exec_module(module)
    return module


def _bundle(directory: Path, key: str, split: str, oof: float, held: float) -> None:
    """The fields the generator reads, and nothing else."""
    path = directory / key
    path.mkdir(parents=True, exist_ok=True)
    (path / "bundle.json").write_text(
        json.dumps({
            "provenance": {"split": split},
            "cross_validation": {"pooled_oof_score": oof},
            "metrics": {"primary": {"value": held, "ci_low": held - 0.05, "ci_high": held + 0.05, "n": 100}},
        }),
        encoding="utf-8",
    )


def test_a_bundle_from_another_split_is_excluded_everywhere(tables, tmp_path):
    # hERG on the published split, DILI left over from the scaffold-grouped one.
    _bundle(tmp_path, "herg", "random, 20% held out, seed 0", oof=0.8458, held=0.8667)
    _bundle(tmp_path, "dili", "scaffold-grouped, 20% held out", oof=0.95, held=0.70)

    keep, excluded = tables.eligible(tmp_path)
    assert set(keep) == {"herg"}
    assert set(excluded) == {"dili"}

    table = tables.held_out_table(tmp_path)
    assert "| `herg` | classification | auroc | 0.8667 |" in table
    assert "omitted: measured on scaffold-grouped" in table

    comparison = tables.oof_comparison(tmp_path)
    # One row published, so one row compared -- and the excluded DILI, whose 0.95
    # against 0.70 is by far the largest gap in the directory, is not the answer.
    assert comparison["n"] == 1
    assert comparison["largest_flattering"] is None
    assert "0 of 1" in comparison["sentence"]
    assert "1 bundle(s) excluded" in comparison["sentence"]

    assert "1 of 29 endpoints measured on the random holdout" in tables.summary(tmp_path)
    assert "excluded as measured on another split: dili" in tables.summary(tmp_path)


def test_a_directory_with_no_publishable_bundle_says_so_rather_than_quoting_figures(tables, tmp_path):
    """The table has no measured row, so the sentence beside it must have no numbers."""
    _bundle(tmp_path, "dili", "scaffold-grouped, 20% held out", oof=0.95, held=0.70)
    _bundle(tmp_path, "herg", "scaffold-grouped, 20% held out", oof=0.91, held=0.80)

    comparison = tables.oof_comparison(tmp_path)
    assert comparison["n"] == 0
    assert comparison["median_abs_gap"] is None
    assert comparison["largest_flattering"] is None
    assert "nothing to compare" in comparison["sentence"]
    assert "of 2" not in comparison["sentence"]


def test_the_direction_of_the_gap_follows_the_metric(tables, tmp_path):
    """For MAE a smaller value is better, so a lower out-of-fold number flatters."""
    # caco2 is MAE: out-of-fold 0.28 against held-out 0.31 is the favourable one.
    _bundle(tmp_path, "caco2", "random, 20% held out, seed 0", oof=0.28, held=0.31)
    assert tables.oof_comparison(tmp_path)["n_flattering"] == 1

    # And the other way round.
    _bundle(tmp_path, "caco2", "random, 20% held out, seed 0", oof=0.34, held=0.31)
    assert tables.oof_comparison(tmp_path)["n_flattering"] == 0
