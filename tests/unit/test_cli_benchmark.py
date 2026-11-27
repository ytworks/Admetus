"""``admetus benchmark``: what it refuses, before anything expensive starts.

Five seeds per endpoint is the most costly thing this project can be asked to do, so
the checks that stop a run must happen before the first fit, and they must say what
to do instead. None of these tests trains anything or touches the network.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from admetus.cli.main import cli
from admetus.data.benchmark import OFFICIAL_SEEDS
from admetus.data.registry import get_endpoint
from admetus.errors import EXIT_USAGE
from tests.unit.test_data_benchmark import write_split


@pytest.fixture
def runner():
    return CliRunner()


def test_selecting_nothing_says_what_all_would_mean(runner):
    result = runner.invoke(cli, ["benchmark", "run"])
    assert result.exit_code == EXIT_USAGE
    assert "--all" in result.output
    assert "22" in result.output


def test_an_endpoint_outside_the_official_group_is_refused_and_redirected(runner):
    result = runner.invoke(cli, ["benchmark", "run", "--endpoint", "pampa"])
    assert result.exit_code == EXIT_USAGE
    assert "no published split exists for pampa" in result.output
    # The refusal has to name the thing that does work for that endpoint.
    assert "admetus evaluate" in result.output


def test_an_unknown_endpoint_is_refused_before_any_download(runner):
    result = runner.invoke(cli, ["benchmark", "run", "--endpoint", "not_an_endpoint"])
    assert result.exit_code == EXIT_USAGE
    assert "unknown endpoint" in result.output


@pytest.mark.parametrize("seeds", ["0", "6"])
def test_a_seed_count_outside_the_protocol_is_refused(runner, seeds):
    result = runner.invoke(cli, ["benchmark", "run", "--endpoint", "dili", "--seeds", seeds])
    assert result.exit_code == EXIT_USAGE
    assert f"between 1 and {len(OFFICIAL_SEEDS)}" in result.output


def test_a_missing_split_file_is_reported_per_endpoint(runner, tmp_path):
    """One endpoint without its split must not abort the others.

    Reading ``stdout`` rather than the combined output is part of the check: logs go
    to standard error so that ``--json`` leaves standard output parseable, and this
    command logs the failure as it happens.
    """
    write_split(tmp_path, get_endpoint("dili"))
    result = runner.invoke(
        cli,
        [
            "--quiet", "benchmark", "run", "--endpoint", "herg", "--seeds", "1",
            "--no-tune", "--data-dir", str(tmp_path), "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["failed"][0]["endpoint"] == "herg"
    assert "admetus benchmark fetch" in payload["failed"][0]["error"]


def test_the_report_says_whether_the_protocol_was_complete(runner, tmp_path):
    """A mean over fewer than five seeds is not the leaderboard's number."""
    write_split(tmp_path, get_endpoint("dili"))
    result = runner.invoke(
        cli,
        [
            "--quiet", "benchmark", "run", "--endpoint", "dili", "--seeds", "2",
            "--no-tune", "--data-dir", str(tmp_path), "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["seeds"] == [1, 2]
    assert payload["complete_protocol"] is False
    assert payload["tuned_per_seed"] is False
    # Three train_val rows cannot be trained on, so this fails -- loudly, per endpoint.
    assert payload["failed"] or payload["results"]


def test_complete_protocol_is_what_ran_not_what_was_asked_for(runner, tmp_path):
    """It used to compare the request with five, and never the outcome.

    Asking for all five seeds and completing none of them reported
    ``complete_protocol: true``, which is the one field a caller reads to decide
    whether a number is comparable with the leaderboard.
    """
    write_split(tmp_path, get_endpoint("dili"))
    result = runner.invoke(
        cli,
        [
            "--quiet", "benchmark", "run", "--endpoint", "dili", "--seeds", "5",
            "--no-tune", "--data-dir", str(tmp_path), "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["seeds_requested"] == 5
    assert payload["seeds"] == list(OFFICIAL_SEEDS)
    # Three train_val rows cannot be trained on, so no seed completed. The endpoint
    # is still in `results`, which is exactly the case the old check missed.
    assert [r["n_seeds"] for r in payload["results"]] == [0]
    assert payload["complete_protocol"] is False
    assert payload["ok"] is False


def test_the_budget_is_reported_per_fold_with_the_number_of_runs_it_buys(runner, tmp_path):
    """`--time-budget` caps one fold, and every fold of every seed is tuned.

    Reported as `time_budget_seconds`, it read as the cost of the whole endpoint; at
    3 folds and 5 seeds the wall clock is fifteen times that.
    """
    write_split(tmp_path, get_endpoint("dili"))
    result = runner.invoke(
        cli,
        [
            "--quiet", "benchmark", "run", "--endpoint", "dili", "--seeds", "5",
            "--time-budget", "30", "--data-dir", str(tmp_path), "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["time_budget_seconds_per_fold"] == 30
    assert payload["tuning_runs_per_endpoint"] == 3 * 5
