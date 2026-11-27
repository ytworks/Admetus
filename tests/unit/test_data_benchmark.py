"""The published split archive: where its files are, and what a bad one does.

No network. The archive layout is reproduced on disk from the directory names the
registry implies, which is also the test that the mapping from ``tdc_name`` to the
archive's directory name is real rather than assumed.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from admetus.data import benchmark
from admetus.data.registry import get_endpoint, official_endpoints
from admetus.errors import InputDataError, NetworkError, UsageError

#: The 22 directory names the archive actually carries, recorded here so a change to
#: `tdc_name` in the registry cannot quietly break the mapping.
ARCHIVE_DIRECTORIES = frozenset({
    "ames", "bbb_martins", "bioavailability_ma", "caco2_wang", "clearance_hepatocyte_az",
    "clearance_microsome_az", "cyp2c9_substrate_carbonmangels", "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels", "cyp2d6_veith", "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith", "dili", "half_life_obach", "herg", "hia_hou", "ld50_zhu",
    "lipophilicity_astrazeneca", "pgp_broccatelli", "ppbr_az", "solubility_aqsoldb",
    "vdss_lombardo",
})

TRAIN_VAL = (
    "Drug_ID,Drug,Y\n"
    "Drug 1,CCO,0\n"
    "Drug 2,CC(=O)Oc1ccccc1C(=O)O,1\n"
    "Drug 3,c1ccccc1,0\n"
)
TEST = "Drug_ID,Drug,Y\nDrug 4,CCN,1\nDrug 5,c1ccncc1,0\n"


def write_split(root, endpoint, train_val=TRAIN_VAL, test=TEST):
    directory = benchmark.split_directory(root, endpoint)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_val.csv").write_text(train_val, encoding="utf-8")
    (directory / "test.csv").write_text(test, encoding="utf-8")
    return directory


def test_every_official_endpoint_maps_to_a_directory_the_archive_has():
    official = official_endpoints()
    assert len(official) == 22
    assert {e.tdc_name.lower() for e in official} == ARCHIVE_DIRECTORIES


def test_the_split_is_read_with_the_published_rows_marked_as_the_holdout(tmp_path):
    endpoint = get_endpoint("dili")
    write_split(tmp_path, endpoint)

    split = benchmark.load_official_split(endpoint, tmp_path)

    assert split.n_train_val == 3
    assert split.n_test == 2
    assert len(split.dataset) == 5
    # Training rows come first, and exactly the test rows are flagged.
    assert split.holdout.tolist() == [False, False, False, True, True]
    assert split.dataset.smiles[3:] == ["CCN", "c1ccncc1"]
    assert np.array_equal(split.dataset.y, np.asarray([0, 1, 0, 1, 0], dtype=float))
    assert split.dataset.provenance["archive_sha256"] == benchmark.ARCHIVE_SHA256


def test_an_endpoint_outside_the_official_group_is_refused_not_invented(tmp_path):
    with pytest.raises(UsageError, match="not in the official benchmark group"):
        benchmark.load_official_split(get_endpoint("pampa"), tmp_path)


def test_a_missing_split_file_says_how_to_get_it(tmp_path):
    with pytest.raises(InputDataError, match="admetus benchmark fetch"):
        benchmark.load_official_split(get_endpoint("dili"), tmp_path)


def test_a_label_that_is_not_a_number_is_refused(tmp_path):
    endpoint = get_endpoint("dili")
    write_split(tmp_path, endpoint, test="Drug_ID,Drug,Y\nDrug 4,CCN,not_a_number\n")
    with pytest.raises(InputDataError, match="is not a number"):
        benchmark.load_official_split(endpoint, tmp_path)


def test_a_missing_column_is_named(tmp_path):
    endpoint = get_endpoint("dili")
    write_split(tmp_path, endpoint, test="Drug_ID,Drug\nDrug 4,CCN\n")
    with pytest.raises(InputDataError, match="'Y'"):
        benchmark.load_official_split(endpoint, tmp_path)


def test_an_archive_entry_that_would_escape_the_destination_is_refused(tmp_path):
    from admetus.data.download import extract_benchmark_archive

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.csv", "Drug_ID,Drug,Y\n")
    with pytest.raises(NetworkError, match="unsafe path"):
        extract_benchmark_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.csv").exists()


def test_the_protocol_runs_five_seeds():
    # Not a detail: a mean over fewer seeds is not the leaderboard's number, and the
    # CLI bounds --seeds by this tuple.
    assert benchmark.OFFICIAL_SEEDS == (1, 2, 3, 4, 5)
