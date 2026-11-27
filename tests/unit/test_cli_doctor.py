"""``admetus doctor``: the contract an automated caller branches on.

The exit codes matter more here than the text. A fresh checkout has no bundles and
must report 3, which CI accepts; 1 means the install itself is broken, and CI fails
on it. The regression below is why there are tests here at all: a macOS runner that
reports only one CPU performance level made the text renderer read a key that was
never set, and a cosmetic line turned a usable install into exit 1.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from click.testing import CliRunner

from admetus.cli import cmd_doctor
from admetus.cli.main import cli
from admetus.errors import EXIT_NO_MODEL, EXIT_OK


@pytest.fixture
def runner():
    return CliRunner()


def test_a_fresh_checkout_reports_no_model_rather_than_a_broken_install(runner, tmp_path):
    result = runner.invoke(cli, ["doctor", "--models-dir", str(tmp_path / "absent")])
    assert result.exit_code == EXIT_NO_MODEL, result.output
    assert "no trained model" in result.output


def test_the_json_report_names_what_a_caller_checks(runner, tmp_path):
    result = runner.invoke(cli, ["doctor", "--json", "--models-dir", str(tmp_path)])
    assert result.exit_code in (EXIT_OK, EXIT_NO_MODEL), result.output
    report = json.loads(result.output)
    assert report["lightgbm"]["ok"] is True
    assert report["chemistry"]["ok"] is True
    assert report["cpu"]["usable_cpus"] >= 1


def _fake_sysctl(answers: dict[str, str]):
    """Stand in for sysctl, answering only the keys given."""

    def run(argv, **kwargs):
        key = argv[-1]
        if key in answers:
            return subprocess.CompletedProcess(argv, 0, stdout=answers[key], stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown oid")

    return run


def test_a_machine_reporting_one_performance_level_still_reports(runner, monkeypatch, tmp_path):
    """The virtualized-Apple-Silicon case: hw.perflevel0 exists, hw.perflevel1 does not."""
    monkeypatch.setattr(cmd_doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess, "run", _fake_sysctl({"hw.perflevel0.logicalcpu": "4\n"})
    )

    result = runner.invoke(cli, ["doctor", "--models-dir", str(tmp_path)])

    assert result.exit_code == EXIT_NO_MODEL, result.output
    assert "internal error" not in result.output
    # Whichever level was reported is reported, and the absent one is simply absent.
    assert "4 performance" in result.output
    assert "efficiency" not in result.output


def test_both_halves_of_the_core_split_are_reported_together(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_sysctl({"hw.perflevel0.logicalcpu": "4\n", "hw.perflevel1.logicalcpu": "6\n"}),
    )

    result = runner.invoke(cli, ["doctor", "--models-dir", str(tmp_path)])

    assert result.exit_code == EXIT_NO_MODEL, result.output
    assert "4 performance, 6 efficiency" in result.output


def test_remediation_only_names_commands_that_exist(runner, tmp_path):
    """Pointing an automated caller at a command that does not exist is worse than silence.

    ``doctor`` and the bundle loader used to suggest ``admetus models pull`` while no
    such command existed. This reads every ``admetus <word>`` the output suggests and
    checks it against the command line as actually built.
    """
    import re

    result = runner.invoke(cli, ["doctor", "--models-dir", str(tmp_path)])
    suggested = set(re.findall(r"admetus ([a-z][a-z_-]*)", result.output))
    assert suggested, result.output
    unknown = suggested - set(cli.commands)
    assert not unknown, f"doctor suggests commands that do not exist: {sorted(unknown)}"
