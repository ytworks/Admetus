"""Train a model and predict with it, checking the contract end to end.

Deliberately small and tuning-free so it runs in CI, but it exercises the whole
path: load, standardize, featurize, split, audit, prune, fit, save, load, predict
and append.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from admetus.bundle import load as bundle_load
from admetus.bundle import save as bundle_save
from admetus.data.loader import Dataset
from admetus.data.registry import Endpoint
from admetus.errors import InputDataError
from admetus.predict import csvio, runner
from admetus.train.trainer import train_endpoint

pytestmark = pytest.mark.slow

#: A fixture input carrying every CSV hazard at once.
TRICKY = (
    "﻿compound_id,smiles,project,note\r\n"
    '001,"CC(=O)Oc1ccccc1C(=O)O",alpha,aspirin\r\n'
    '"0,2",CCO,beta,"multi\nline note"\r\n'
    "003,CC(=O)[O-].[Na+],alpha,salt\r\n"
    '004,not_a_molecule,gamma,"should fail"\r\n'
    "005,CCO,beta,duplicate of row 2"
)


def synthetic_endpoint() -> Endpoint:
    return Endpoint(
        key="fixture",
        tdc_name="Fixture",
        dataverse_id=1,
        task="classification",
        metric="auroc",
        unit="binary",
        in_official_group=False,
        description="synthetic fixture",
    )


def synthetic_dataset(n: int = 420) -> Dataset:
    """Distinct molecules over many scaffolds, with a learnable label.

    Two properties the fixture has to have, both learned the hard way:

    The label must be a function of the structure, not a random draw. Two
    molecules the featurizer cannot distinguish must not carry different labels,
    or the leakage audit correctly discards the training set as contradictory.

    The label must not be determined by the scaffold either. Scaffold-grouped
    cross-validation cannot stratify when the group decides the class, and the
    splitter rightly refuses. So the scaffold varies independently of the label,
    and the substituent carries the signal.
    """
    cores = [
        "c1ccccc1", "c1ccncc1", "c1ccsc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CCNCC1",
        "c1cc[nH]c1", "C1CCOCC1", "c1ccc(cc1)", "C1CCCC1", "c1cnc[nH]1", "C1CCCNC1",
    ]
    #: Polar substituents are the positive class; the scaffold says nothing.
    polar = ["O", "N", "C(=O)O", "C(=O)N", "S(=O)(=O)N", "OC"]
    apolar = ["C", "CC", "F", "Cl", "C(C)C", "CCC"]

    smiles: list[str] = []
    labels: list[float] = []
    seen: set[str] = set()
    index = 0
    while len(smiles) < n and index < 10_000:
        core = cores[index % len(cores)]
        positive = (index // len(cores)) % 2 == 0
        pool = polar if positive else apolar
        group = pool[(index // (len(cores) * 2)) % len(pool)]
        chain = "C" * ((index // (len(cores) * 2 * len(pool))) % 3 + 1)
        candidate = core + chain + group
        index += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        smiles.append(candidate)
        labels.append(1.0 if positive else 0.0)
    return Dataset(
        key="fixture",
        ids=[str(i) for i in range(len(smiles))],
        smiles=smiles,
        y=np.asarray(labels, dtype=np.float64),
        task="classification",
        provenance={"n_rows": len(smiles), "source": "synthetic"},
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once and reuse, since even an untuned fit is not free."""
    directory = tmp_path_factory.mktemp("models")
    endpoint = synthetic_endpoint()
    model = train_endpoint(
        synthetic_endpoint(), synthetic_dataset(), folds=3, seed=0, tune=False,
        num_threads=2, bootstrap_resamples=50,
    )
    path = bundle_save.save(model, directory)
    return endpoint, model, path


def test_training_produces_a_complete_bundle(trained):
    _, model, path = trained
    assert (path / "bundle.json").is_file()
    assert (path / "columns.npz").is_file()
    # No applicability-domain reference: the flag it served was removed after it
    # measured as uninformative at 16 of 29 endpoints.
    assert not (path / "ad_reference.npz").exists()
    assert len(list(path.glob("fold_*.txt"))) == model.n_boosters == 3

    record = json.loads((path / "bundle.json").read_text())
    assert record["featurizer"]["spec_sha256"]
    assert record["featurizer"]["column_sha256"]
    assert record["featurizer"]["golden_fingerprint"]
    # The cross-validation score must be marked as what it is.
    assert record["cross_validation"]["cv_score_biased"] is True
    # The held-out metric must carry its uncertainty.
    primary = record["metrics"]["primary"]
    assert primary["ci_low"] <= primary["value"] <= primary["ci_high"]


def test_the_bundle_verifies_and_reloads_identically(trained):
    _, model, path = trained
    ok, problems = bundle_load.verify(path)
    assert ok, problems

    bundle_load.load.cache_clear()
    loaded = bundle_load.load(str(path))
    assert loaded.n_boosters == model.n_boosters
    assert loaded.basis.n_cols == model.basis.n_cols
    assert loaded.spec.spec_sha256 == model.spec.spec_sha256


def test_a_bundle_from_a_different_column_space_is_refused(trained, tmp_path):
    import shutil

    _, _, path = trained
    copy = tmp_path / "tampered"
    shutil.copytree(path, copy)
    record = json.loads((copy / "bundle.json").read_text())
    record["featurizer"]["column_sha256"] = "0" * 64
    (copy / "bundle.json").write_text(json.dumps(record))

    bundle_load.load.cache_clear()
    with pytest.raises(bundle_load.SpecMismatchError, match="retained columns"):
        bundle_load.load(str(copy))


def _predict(path, tmp_path, **overrides):
    source = tmp_path / "in.csv"
    source.write_bytes(TRICKY.encode("utf-8"))
    destination = tmp_path / "out.csv"
    bundle_load.load.cache_clear()
    model = bundle_load.load(str(path))
    options = runner.Options(jobs=1, **overrides)
    report = runner.run(source, destination, [model], options)
    return source, destination, report


def test_predict_appends_columns_and_keeps_everything_else(trained, tmp_path):
    _, _, path = trained
    source, destination, report = _predict(path, tmp_path)

    original = list(csvio.read_records(source, csvio.sniff(source)))
    produced = list(csvio.read_records(destination, csvio.sniff(destination)))

    # Six records: the header plus five data rows.
    assert len(produced) == len(original) == 6
    for new, old in zip(produced, original):
        assert new.fields[: len(old.fields)] == old.fields

    header = produced[0].fields
    assert header[4:] == ["fixture", "fixture_ens_sd", "admetus_error"]
    assert report.added_columns == header[4:]


def test_the_passthrough_bytes_are_unchanged(trained, tmp_path):
    """Strip the appended fields and the file must be byte-identical to the input."""
    _, _, path = trained
    source, destination, _ = _predict(path, tmp_path)
    original = source.read_bytes()

    dialect = csvio.sniff(destination)
    # Three appended fields now: the prediction, the ensemble spread and the error
    # column. The domain flag went with the applicability-domain machinery.
    n_appended = 3
    rebuilt = b""
    for record in csvio.read_records(destination, dialect):
        body = record.body.rsplit(",", n_appended)[0]
        rebuilt += (body + record.terminator).encode("utf-8")
    assert rebuilt == original


def test_an_unparseable_molecule_is_reported_not_substituted(trained, tmp_path):
    _, _, path = trained
    _, destination, report = _predict(path, tmp_path)
    rows = list(csvio.read_records(destination, csvio.sniff(destination)))
    bad = rows[4]
    assert bad.fields[1] == "not_a_molecule"
    assert bad.fields[4] == ""           # no prediction
    assert "could not parse" in bad.fields[-1]
    assert report.n_failed == 1
    # The fixture has five data rows; one bad row is not a failed run.
    assert report.n_rows == 5


def test_a_repeated_molecule_is_featurized_once_and_predicted_the_same(trained, tmp_path):
    _, _, path = trained
    _, destination, report = _predict(path, tmp_path)
    rows = list(csvio.read_records(destination, csvio.sniff(destination)))
    assert rows[2].fields[1] == rows[5].fields[1] == "CCO"
    assert rows[2].fields[4] == rows[5].fields[4]
    # Four distinct strings across five rows, three of which featurize.
    assert report.n_unique_raw == 4
    assert report.n_featurized == 3


def test_salts_are_standardized_and_counted(trained, tmp_path):
    _, _, path = trained
    _, _, report = _predict(path, tmp_path)
    assert report.n_standardization_changed >= 1


def test_classification_output_is_a_probability(trained, tmp_path):
    _, _, path = trained
    _, destination, _ = _predict(path, tmp_path)
    for record in list(csvio.read_records(destination, csvio.sniff(destination)))[1:]:
        if record.fields[4]:
            assert 0.0 <= float(record.fields[4]) <= 1.0


def test_worker_count_does_not_change_the_output(trained, tmp_path):
    _, _, path = trained
    source = tmp_path / "in.csv"
    source.write_bytes(TRICKY.encode("utf-8"))
    bundle_load.load.cache_clear()
    model = bundle_load.load(str(path))

    outputs = []
    for jobs in (1, 2):
        destination = tmp_path / f"out_{jobs}.csv"
        runner.run(source, destination, [model], runner.Options(jobs=jobs, batch_size=2))
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


def test_a_column_prefix_namespaces_everything(trained, tmp_path):
    _, _, path = trained
    _, _, report = _predict(path, tmp_path, column_prefix="admet_")
    assert all(name.startswith("admet_") for name in report.added_columns)


def test_a_colliding_column_name_is_renamed(trained, tmp_path):
    _, _, path = trained
    source = tmp_path / "collide.csv"
    source.write_text("id,smiles,fixture\n1,CCO,existing\n", encoding="utf-8")
    destination = tmp_path / "out.csv"
    bundle_load.load.cache_clear()
    report = runner.run(
        source, destination, [bundle_load.load(str(path))], runner.Options(jobs=1)
    )
    assert "fixture_admetus" in report.added_columns
    assert destination.read_text(encoding="utf-8").splitlines()[1].startswith("1,CCO,existing,")


def test_a_given_holdout_replaces_the_scaffold_split():
    """The official protocol's path: the published test rows are the ones scored.

    Checks what the protocol depends on -- that the rows handed in as the holdout are
    exactly the rows measured, and that the record says the split was given rather
    than derived, so a number from this path cannot be mistaken for a scaffold one.
    """
    dataset = synthetic_dataset()
    holdout = np.zeros(len(dataset), dtype=bool)
    holdout[-120:] = True

    model = train_endpoint(
        synthetic_endpoint(), dataset, folds=3, seed=0, tune=False,
        num_threads=2, bootstrap_resamples=0, holdout=holdout,
    )

    assert model.provenance["split"] == "given (published benchmark split)"
    assert model.provenance["test_fraction"] is None
    # The scored rows are the holdout's, less any molecule the featurizer could not
    # read -- which drops out of whichever side it was on rather than moving sides.
    n_failed = model.provenance["n_featurize_failures"]
    assert model.provenance["n_test"] == model.test_metrics["n_test"]
    assert 120 - n_failed <= model.provenance["n_test"] <= 120
    assert model.provenance["n_train"] <= len(dataset) - model.provenance["n_test"]


def test_a_holdout_of_the_wrong_shape_is_refused():
    dataset = synthetic_dataset()
    with pytest.raises(InputDataError, match="holdout mask has shape"):
        train_endpoint(
            synthetic_endpoint(), dataset, folds=3, seed=0, tune=False,
            holdout=np.zeros(len(dataset) - 1, dtype=bool),
        )


def test_a_holdout_that_leaves_nothing_to_measure_is_refused():
    dataset = synthetic_dataset()
    with pytest.raises(InputDataError, match="nothing can be measured"):
        train_endpoint(
            synthetic_endpoint(), dataset, folds=3, seed=0, tune=False,
            holdout=np.zeros(len(dataset), dtype=bool),
        )


