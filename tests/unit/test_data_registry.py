"""The endpoint registry and the rules that follow from dataset size."""

from __future__ import annotations

import pytest

from admetus.data.registry import (
    LOW_CONFIDENCE_ROWS,
    all_endpoints,
    get_endpoint,
    load_registry,
    official_endpoints,
    prevalence_range,
    resolve_keys,
    search_budget,
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


def test_solubility_keeps_salts():
    """Aqueous solubility is measured on the salt, so the fragment step is omitted."""
    assert "largest_fragment" not in get_endpoint("solubility").standardize
    assert "largest_fragment" in get_endpoint("dili").standardize


def test_bounded_targets_declare_their_range():
    assert get_endpoint("ppbr").output_range == (0.0, 100.0)
    assert get_endpoint("vdss").output_range == (0.0, None)


def test_species_filter_only_where_the_file_has_one():
    with_filter = [e.key for e in all_endpoints() if e.species_filter]
    assert with_filter == ["ppbr"]


def test_feature_set_follows_dataset_size():
    endpoint = get_endpoint("dili")
    assert endpoint.feature_set(400) == "morgan_p2"
    assert endpoint.feature_set(5000) == "morgan_p23"


def test_featurizer_spec_carries_the_endpoint_pipeline():
    spec = get_endpoint("solubility").featurizer_spec(9982)
    assert spec.standardize == ("cleanup", "uncharge")
    assert get_endpoint("dili").featurizer_spec(5000).standardize == (
        "cleanup", "largest_fragment", "uncharge",
    )


def test_prevalence_band_is_two_sided_and_tightens_with_size():
    small = prevalence_range(300)
    medium = prevalence_range(2000)
    large = prevalence_range(13000)
    assert small.low > medium.low > large.low
    assert small.high < medium.high < large.high
    for band in (small, medium, large):
        assert 0 < band.low < band.high < 1


def test_budget_trades_trials_for_repeats_on_small_data():
    small = search_budget(400)
    large = search_budget(13000)
    assert small.cv_repeats > large.cv_repeats
    assert small.trials < large.trials


def test_budget_overrides_win():
    assert search_budget(400, trials=7, cv_repeats=1) == search_budget(99999, trials=7, cv_repeats=1)


def test_low_confidence_is_flagged_for_small_datasets():
    assert get_endpoint("carcinogens").low_confidence() is True
    assert get_endpoint("solubility").low_confidence() is False
    assert get_endpoint("dili").expected_rows < LOW_CONFIDENCE_ROWS


def test_json_description_is_complete():
    payload = get_endpoint("vdss").to_json()
    for field in ("key", "task", "metric", "direction", "unit", "transform", "n_rows",
                  "low_confidence", "feature_set", "correlated_with", "in_official_group"):
        assert field in payload
    assert payload["direction"] == "higher_is_better"
    assert payload["transform"] == "log10_safe"


def test_unknown_endpoint_names_the_remedy():
    with pytest.raises(UsageError, match="admetus endpoints"):
        get_endpoint("no_such_endpoint")


def test_resolve_keys():
    assert len(resolve_keys([], select_all=True)) == 29
    assert [e.key for e in resolve_keys(["dili", "caco2"])] == ["dili", "caco2"]
    with pytest.raises(UsageError, match="no endpoint selected"):
        resolve_keys([])
