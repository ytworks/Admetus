"""Normalizing published dataset files, and the assertions that guard it."""

from __future__ import annotations

import numpy as np
import pytest

from admetus.data import loader
from admetus.data.registry import Endpoint
from admetus.errors import InputDataError


def make_endpoint(**overrides) -> Endpoint:
    """A minimal endpoint for exercising the loader."""
    base = {
        "key": "fixture",
        "tdc_name": "Fixture",
        "dataverse_id": 1,
        "task": "classification",
        "metric": "auroc",
        "unit": "binary",
        "in_official_group": False,
        "description": "test fixture",
    }
    base.update(overrides)
    return Endpoint(**base)


def write_tab(tmp_path, rows: list[str], key: str = "fixture"):
    path = tmp_path / f"{key}.tab"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_modern_header(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCN\t0"])
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.smiles == ["CCO", "CCN"]
    assert dataset.ids == ["a", "b"]
    assert dataset.y.tolist() == [1.0, 0.0]


def test_legacy_header_is_accepted(tmp_path):
    """Older published files use ID / X / Y instead of Drug_ID / Drug / Y."""
    directory = write_tab(tmp_path, ["ID\tX\tY", "a\tCCO\t1", "b\tCCN\t0"])
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.smiles == ["CCO", "CCN"]


def test_quoted_fields_are_unquoted(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", '"a"\t"CCO"\t1', '"b"\t"CCN"\t0'])
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.smiles == ["CCO", "CCN"]
    assert dataset.ids == ["a", "b"]


def test_species_filter_keeps_only_the_named_species(tmp_path):
    directory = write_tab(
        tmp_path,
        [
            "Drug_ID\tDrug\tY\tSpecies",
            "a\tCCO\t10.0\tHomo sapiens",
            "b\tCCN\t20.0\tCanis lupus familiaris",
            "c\tCCC\t30.0\tHomo sapiens",
        ],
    )
    endpoint = make_endpoint(task="regression", metric="mae", species_filter="Homo sapiens")
    dataset = loader.load(endpoint, directory)
    assert dataset.smiles == ["CCO", "CCC"]
    assert dataset.provenance["n_wrong_species"] == 1


def test_species_filter_without_the_column_is_refused(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1"])
    endpoint = make_endpoint(species_filter="Homo sapiens")
    with pytest.raises(InputDataError, match="no Species column"):
        loader.load(endpoint, directory)


def test_wide_file_selects_one_label_column_and_drops_its_blanks(tmp_path):
    directory = write_tab(
        tmp_path,
        [
            "NR-AR\tSR-MMP\tID\tX",
            "1\t0\ta\tCCO",
            "\t1\tb\tCCN",
            "0\t1\tc\tCCC",
        ],
    )
    endpoint = make_endpoint(label_name="NR-AR")
    dataset = loader.load(endpoint, directory)
    assert dataset.smiles == ["CCO", "CCC"]
    assert dataset.y.tolist() == [1.0, 0.0]
    assert dataset.provenance["n_missing_label"] == 1


def test_missing_label_column_lists_what_is_available(tmp_path):
    directory = write_tab(tmp_path, ["NR-AR\tID\tX", "1\ta\tCCO"])
    with pytest.raises(InputDataError, match="SR-MMP|no label column"):
        loader.load(make_endpoint(label_name="SR-MMP"), directory)


def test_duplicate_structures_are_collapsed_for_regression_by_median(tmp_path):
    directory = write_tab(
        tmp_path,
        ["Drug_ID\tDrug\tY", "a\tCCO\t1.0", "b\tCCO\t3.0", "c\tCCO\t11.0", "d\tCCN\t5.0"],
    )
    endpoint = make_endpoint(task="regression", metric="mae")
    dataset = loader.load(endpoint, directory)
    assert dataset.smiles == ["CCO", "CCN"]
    assert dataset.y.tolist() == [3.0, 5.0]
    assert dataset.provenance["n_duplicate_removed"] == 2


def test_contradictory_classification_duplicates_are_dropped(tmp_path):
    """A structure labelled both 0 and 1 teaches nothing, so the group goes."""
    directory = write_tab(
        tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCO\t0", "c\tCCN\t1", "d\tCCC\t0"]
    )
    dataset = loader.load(make_endpoint(), directory)
    assert "CCO" not in dataset.smiles
    assert dataset.provenance["n_label_conflict"] == 1


def test_unanimous_classification_duplicates_are_kept_once(tmp_path):
    directory = write_tab(
        tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCO\t1", "c\tCCN\t0"]
    )
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.smiles == ["CCO", "CCN"]
    assert dataset.provenance["n_duplicate_removed"] == 1


def test_non_binary_label_in_a_classification_file_is_dropped(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCN\t0.5", "c\tCCC\t0"])
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.y.tolist() == [1.0, 0.0]


def test_single_class_labels_are_refused(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCN\t1"])
    with pytest.raises(InputDataError, match="all one class"):
        loader.load(make_endpoint(), directory)


def test_negative_labels_contradict_a_log_transform(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t-1.0", "b\tCCN\t2.0"])
    endpoint = make_endpoint(task="regression", metric="mae", transform="log10_safe")
    with pytest.raises(InputDataError, match="negative labels"):
        loader.load(endpoint, directory)


def test_short_rows_are_skipped_not_padded(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCN", "c\tCCC\t0"])
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.smiles == ["CCO", "CCC"]
    assert dataset.provenance["n_short_row"] == 1


def test_missing_file_names_the_remedy(tmp_path):
    with pytest.raises(InputDataError, match="admetus data fetch"):
        loader.load(make_endpoint(), tmp_path)


def test_empty_file_is_refused(tmp_path):
    (tmp_path / "fixture.tab").write_text("", encoding="utf-8")
    with pytest.raises(InputDataError, match="empty"):
        loader.load(make_endpoint(), tmp_path)


def test_unrecognizable_header_is_refused(tmp_path):
    directory = write_tab(tmp_path, ["alpha\tbeta", "1\t2"])
    with pytest.raises(InputDataError, match="structure column"):
        loader.load(make_endpoint(), directory)


def test_summary_and_rates(tmp_path):
    directory = write_tab(
        tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1", "b\tCCN\t0", "c\tCCC\t0", "d\tCCCC\t0"]
    )
    dataset = loader.load(make_endpoint(), directory)
    assert dataset.n_positive == 1
    assert dataset.pos_rate == pytest.approx(0.25)
    assert "25.0%" in dataset.summary()


def test_regression_summary_reports_the_range(tmp_path):
    directory = write_tab(tmp_path, ["Drug_ID\tDrug\tY", "a\tCCO\t1.5", "b\tCCN\t9.5"])
    dataset = loader.load(make_endpoint(task="regression", metric="mae"), directory)
    assert "1.5" in dataset.summary()
    assert np.isnan(dataset.pos_rate)
