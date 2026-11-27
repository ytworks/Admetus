"""The endpoint registry and the rules that follow from dataset size."""

from __future__ import annotations

import pytest

from admetus.data.registry import (
    LOW_CONFIDENCE_ROWS,
    all_endpoints,
    get_endpoint,
    load_registry,
    official_endpoints,
    resolve_keys,
)
from admetus.errors import UsageError


def test_registry_holds_the_expected_endpoints():
    assert len(all_endpoints()) == 29
    assert len(official_endpoints()) == 22


def test_keys_and_dataset_ids_are_unique():
    endpoints = all_endpoints()
    assert len({e.key for e in endpoints}) == len(endpoints)
    assert len({e.dataverse_id for e in endpoints}) == len(endpoints)


def test_every_endpoint_is_fully_described():
    for endpoint in all_endpoints():
        assert endpoint.task in ("classification", "regression")
        assert endpoint.metric in ("auroc", "auprc", "mae", "spearman")
        assert endpoint.unit
        assert endpoint.description
        assert endpoint.expected_rows and endpoint.expected_rows > 0
        assert endpoint.download_url.endswith(str(endpoint.dataverse_id))


def test_transforms_only_on_regression_and_only_known_ones():
    for endpoint in all_endpoints():
        if endpoint.transform is not None:
            assert endpoint.task == "regression"
            assert endpoint.transform in ("log10", "log10_safe")


def test_metric_direction():
    assert get_endpoint("dili").higher_is_better is True
    assert get_endpoint("caco2").higher_is_better is False


def test_correlated_endpoints_exist_and_are_mutual():
    registry = load_registry()
    for endpoint in registry.values():
        for other in endpoint.correlated_with:
            assert other in registry
            assert endpoint.key in registry[other].correlated_with, (
                f"{endpoint.key} says it correlates with {other}, but not the other way round"
            )


def test_the_two_herg_curations_declare_each_other():
    """Two models over the same assay must not read as independent evidence."""
    assert "herg_karim" in get_endpoint("herg").correlated_with
    assert "herg" in get_endpoint("herg_karim").correlated_with


def test_species_filter_only_where_the_file_has_one():
    with_filter = [e.key for e in all_endpoints() if e.species_filter]
    assert with_filter == ["ppbr"]


def test_every_endpoint_gets_the_same_feature_space():
    """The size rule and the three measured overrides are gone.

    The reference implementation featurizes every dataset the same way, including
    the 280-compound one its own notebook runs, so there is nothing to choose.
    """
    spaces = {e.featurizer_spec().feature_set for e in all_endpoints()}
    assert spaces == {"morgan_p23"}
    # And the row count no longer changes the answer.
    small, large = get_endpoint("carcinogens"), get_endpoint("herg_karim")
    assert small.featurizer_spec(220).feature_set == large.featurizer_spec(13_445).feature_set

def test_low_confidence_is_flagged_for_small_datasets():
    assert get_endpoint("carcinogens").low_confidence() is True
    assert get_endpoint("solubility").low_confidence() is False
    assert get_endpoint("dili").expected_rows < LOW_CONFIDENCE_ROWS


def test_json_description_is_complete():
    payload = get_endpoint("vdss").to_json()
    for field in ("key", "task", "metric", "direction", "unit", "transform", "n_rows",
                  "low_confidence", "correlated_with", "in_official_group"):
        assert field in payload
    assert payload["direction"] == "higher_is_better"
    # No endpoint declares a transform any more; the field stays in the description
    # so a caller can see that none is applied.
    assert payload["transform"] is None


def test_unknown_endpoint_names_the_remedy():
    with pytest.raises(UsageError, match="admetus endpoints"):
        get_endpoint("no_such_endpoint")


def test_resolve_keys():
    assert len(resolve_keys([], select_all=True)) == 29
    assert [e.key for e in resolve_keys(["dili", "caco2"])] == ["dili", "caco2"]
    with pytest.raises(UsageError, match="no endpoint selected"):
        resolve_keys([])


def test_evaluate_requires_a_bundle_not_just_a_directory(tmp_path):
    """An interrupted training run leaves its tuning study behind with no bundle.

    Presence of the record is the test, which is what `models list` applies too.
    """
    from admetus.bundle.load import available

    (tmp_path / "dili").mkdir()
    (tmp_path / "dili" / "optuna.sqlite").write_bytes(b"not a bundle")
    assert available(tmp_path) == []


def test_doctor_reports_cores_when_a_level_is_missing():
    """A macOS VM can expose one performance level and no efficiency level.

    doctor is the first command a new machine runs, so it must not crash on a
    partially reported core topology. This failed in CI with
    KeyError: 'efficiency_cores'.
    """
    from click.testing import CliRunner

    from admetus.cli import cmd_doctor
    from admetus.cli.main import cli

    for cpu in (
        {"cpu_count": 4, "usable_cpus": 4},
        {"cpu_count": 4, "usable_cpus": 4, "performance_cores": 4},
        {"cpu_count": 4, "usable_cpus": 4, "efficiency_cores": 4},
        {"cpu_count": 10, "usable_cpus": 9, "performance_cores": 4, "efficiency_cores": 6},
    ):
        original = cmd_doctor._cpu_status
        cmd_doctor._cpu_status = lambda c=cpu: dict(c)
        try:
            result = CliRunner().invoke(cli, ["doctor"])
        finally:
            cmd_doctor._cpu_status = original
        assert "internal error" not in result.output, (cpu, result.output)
        assert "cpu " in result.output


def test_no_endpoint_declares_a_transform_a_range_or_a_pipeline():
    """Three per-endpoint knobs the reference implementation does not have.

    It trains on the label it was given, clamps nothing, and fingerprints the parsed
    molecule without standardizing it. A log10 transform, an output range and a
    standardization pipeline were all additions, and each of them changed either the
    target or the molecule the model sees.
    """
    for endpoint in all_endpoints():
        assert endpoint.transform is None, endpoint.key
        assert endpoint.output_range is None, endpoint.key
        assert endpoint.standardize == (), endpoint.key
        assert endpoint.featurizer_spec().standardize == (), endpoint.key
