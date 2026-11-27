"""The report file has to say what the command returned, on every path.

`--report` exists so an automated caller can read the outcome without parsing text,
which only works if the file and the exit status agree. Three ways they used to
disagree, all of them silent:

* a refusal found before any work wrote nothing at all, so a report left behind by an
  earlier successful run stayed byte-for-byte in place and read as this run's result;
* ``--strict --json`` settled the status after the JSON document had gone to standard
  output, so stdout said ``ok: true`` while the report said ``ok: false``;
* ``admetus evaluate`` keyed "the MCC is retrospective only" on the presence of the
  searched value rather than the absence of an operating-point one.

None of these train anything.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from admetus.cli.cmd_train import _mcc_note
from admetus.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


def _stale_success(path) -> None:
    """Leave behind what an earlier successful run would have written."""
    path.write_text(json.dumps({"ok": True, "n_rows": 5, "n_failed": 0}), encoding="utf-8")


def test_a_missing_input_file_replaces_a_stale_success_report(runner, tmp_path):
    report = tmp_path / "report.json"
    _stale_success(report)

    result = runner.invoke(
        cli,
        ["predict", "--models-dir", str(tmp_path / "models"), "--endpoint", "dili",
         "-i", str(tmp_path / "absent.csv"), "--report", str(report)],
    )

    assert result.exit_code != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["error"]


def test_an_empty_model_directory_replaces_a_stale_success_report(runner, tmp_path):
    report = tmp_path / "report.json"
    _stale_success(report)
    empty = tmp_path / "models"
    empty.mkdir()
    source = tmp_path / "in.csv"
    source.write_text("smiles\nCCO\n", encoding="utf-8")

    result = runner.invoke(
        cli,
        ["predict", "--models-dir", str(empty), "--endpoint", "dili",
         "-i", str(source), "--report", str(report)],
    )

    assert result.exit_code != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert "no model bundles" in payload["error"]


def test_the_mcc_flag_reads_the_absence_of_an_operating_point_score():
    """"Only" means there is nothing else.

    Keyed on the presence of ``mcc_at_best_threshold`` alone, a record carrying both
    values was described as having only the retrospective one.
    """
    assert _mcc_note(has_searched_mcc=False, retrospective_only=False) is None

    only = _mcc_note(has_searched_mcc=True, retrospective_only=True)
    assert "no operating threshold ships" in only

    both = _mcc_note(has_searched_mcc=True, retrospective_only=False)
    assert "retrospective diagnostic" in both
    assert "`mcc` is the score at the operating point" in both
