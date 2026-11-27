"""Packaging and installing bundles.

No network: the manifest is read from a local directory, which is the same code path
an installer takes once the base is a URL prefix instead of a folder.

The bundle here is a hand-built fixture rather than a trained model. That is the
point of the unit: these tests are about digests, archive safety and refusal, and a
real fit would make them slow without making them say more. The round trip through a
genuinely trained bundle is covered in the integration tests.
"""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from admetus.bundle import manifest as mf
from admetus.errors import ModelError, NetworkError


def _real_booster(n_features: int) -> str:
    """A genuine one-tree LightGBM model of the given width.

    Has to be genuine: `verify` loads every booster in a child process and compares
    its width with the bundle's basis, because a hand-written stub is exactly what a
    damaged file looks like, and an intact model of the wrong width is exactly what a
    bundle from another feature preset looks like. Both are things the verifier is
    for.
    """
    import lightgbm as lgb
    import numpy as _np

    rng = _np.random.default_rng(0)
    x = rng.random((40, n_features))
    y = (x[:, 0] > 0.5).astype(float)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 2, "min_data_in_leaf": 1,
         "feature_pre_filter": False, "seed": 0},
        lgb.Dataset(x, label=y),
        num_boost_round=1,
    )
    return booster.model_to_string()


def make_bundle(models_dir, key="dili", n_boosters=1, golden=None):
    """Write a directory that verifies, fingerprint included unless told otherwise.

    Keyed to a real endpoint, because a manifest may only name one the registry
    publishes: the models directory also holds the tuning store, and a well-shaped
    key like `studies` would have the installer delete its own search history.
    """
    from admetus.bundle.save import golden_fingerprint
    from admetus.chem.spec import get_spec

    spec = get_spec()
    golden = golden_fingerprint(spec) if golden is None else golden
    directory = models_dir / key
    directory.mkdir(parents=True)
    kept = np.asarray([1, 5, 9, 17], dtype=np.int32)
    np.savez_compressed(directory / "columns.npz", kept_indices=kept)
    np.savez_compressed(
        directory / "ad_reference.npz",
        packed=np.zeros((2, spec.morgan_bits // 8), dtype=np.uint8),
        popcount=np.zeros(2, dtype=np.int32),
    )
    for index in range(n_boosters):
        (directory / f"fold_{index}.txt").write_text(
            _real_booster(int(kept.size)), encoding="utf-8"
        )
    # A training by-product that must not be published.
    (directory / "optuna.sqlite").write_bytes(b"not a model")
    (directory / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-11-27T00:00:00+00:00",
                "endpoint": {
                    "key": key, "task": "classification", "metric": "auroc",
                    "unit": "binary", "in_official_group": False,
                },
                "featurizer": {
                    "spec": spec.to_dict(),
                    "spec_sha256": spec.spec_sha256,
                    "column_sha256": spec.column_sha256(kept),
                    "n_raw_features": spec.n_features,
                    "n_features": int(kept.size),
                    "golden_fingerprint": golden,
                },
                "model": {"n_boosters": n_boosters, "num_boost_round": 10, "params": {}},
                "metrics": {"primary": {"value": 0.9, "ci_low": 0.8, "ci_high": 0.95, "n": 50}},
                "libraries": {"admetus": "0.0.0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def test_packaging_writes_a_manifest_naming_every_file_and_its_digest(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    manifest = mf.build(models, tmp_path / "dist")

    assert manifest["n_bundles"] == 1
    entry = manifest["bundles"][0]
    assert entry["endpoint"] == "dili"
    assert entry["archive_sha256"] and len(entry["archive_sha256"]) == 64
    names = sorted(f["name"] for f in entry["files"])
    assert names == ["ad_reference.npz", "bundle.json", "columns.npz", "fold_0.txt"]
    # The record a caller reads before downloading anything.
    assert entry["held_out"]["value"] == 0.9
    assert entry["feature_set"] == "morgan_p23"
    assert (tmp_path / "dist" / mf.MANIFEST_FILE).is_file()


def test_the_study_database_is_not_published(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    with tarfile.open(tmp_path / "dist" / "dili.tar.gz") as tar:
        assert not any(name.endswith(".sqlite") for name in tar.getnames())


def test_packaging_twice_gives_identical_bytes(tmp_path):
    """A digest in the manifest describes the model, not when tar ran."""
    models = tmp_path / "models"
    make_bundle(models)
    first = mf.build(models, tmp_path / "one")["bundles"][0]["archive_sha256"]
    second = mf.build(models, tmp_path / "two")["bundles"][0]["archive_sha256"]
    assert first == second


def test_a_bundle_that_does_not_verify_is_refused(tmp_path):
    models = tmp_path / "models"
    directory = make_bundle(models)
    (directory / "fold_0.txt").unlink()
    with pytest.raises(ModelError, match="does not verify"):
        mf.build(models, tmp_path / "dist")


def test_an_empty_models_directory_is_refused(tmp_path):
    with pytest.raises(ModelError, match="nothing to publish"):
        mf.build(tmp_path / "models", tmp_path / "dist")


def test_asking_for_an_endpoint_with_no_bundle_names_it(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    with pytest.raises(ModelError, match="no bundle for herg"):
        mf.build(models, tmp_path / "dist", keys=["herg"])


def test_install_round_trips_the_bundle(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")

    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")
    results = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")

    assert [r["installed"] for r in results] == [True]
    assert results[0]["ok"] is True
    installed = tmp_path / "installed" / "dili"
    assert sorted(p.name for p in installed.iterdir()) == [
        "ad_reference.npz", "bundle.json", "columns.npz", "fold_0.txt"
    ]


def test_an_already_installed_bundle_is_left_alone(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")

    mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")
    again = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")
    assert again == [{"endpoint": "dili", "installed": False, "reason": "already present"}]


def test_an_archive_whose_digest_disagrees_is_not_unpacked(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    archive = tmp_path / "dist" / "dili.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")
    results = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")

    assert results[0]["installed"] is False
    assert "digest" in results[0]["reason"]
    assert not (tmp_path / "installed" / "dili").exists()


def test_an_archive_member_outside_the_bundle_is_refused(tmp_path):
    """A manifest can be honest about bytes that still hold a hostile path."""
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "dili.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = tmp_path / "payload"
        payload.write_text("x", encoding="utf-8")
        tar.add(payload, arcname="../escaped.txt")

    from admetus.data.download import sha256_file

    manifest = {
        "manifest_version": 1, "n_bundles": 1,
        "bundles": [{
            "endpoint": "dili", "archive": "dili.tar.gz",
            "archive_sha256": sha256_file(archive), "archive_bytes": archive.stat().st_size,
            "files": [],
        }],
    }
    (dist / mf.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    read, base = mf.read_manifest(str(dist), tmp_path / "cache")
    results = mf.install(read, base, tmp_path / "installed", tmp_path / "cache")
    assert results[0]["installed"] is False
    assert "not a file of the" in results[0]["reason"]
    assert not (tmp_path / "escaped.txt").exists()


def test_a_manifest_that_is_not_one_is_refused(tmp_path):
    path = tmp_path / mf.MANIFEST_FILE
    path.write_text('{"something": "else"}', encoding="utf-8")
    with pytest.raises(NetworkError, match="does not look like a model manifest"):
        mf.read_manifest(str(path), tmp_path)


def test_a_missing_manifest_says_where_it_looked(tmp_path):
    with pytest.raises(NetworkError, match="no manifest at"):
        mf.read_manifest(str(tmp_path / "absent"), tmp_path)


def test_asking_the_manifest_for_an_endpoint_it_lacks_is_refused(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")
    with pytest.raises(ModelError, match="no bundle for herg"):
        mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache", keys=["herg"])


def test_a_candidate_that_does_not_verify_never_replaces_a_working_bundle(tmp_path):
    """The working copy is wanted most exactly when the incoming one is broken.

    `models pull --overwrite` used to delete the target and then verify what it had
    unpacked, so an incompatible bundle cost the operator a working one. Worse, the
    retry without --overwrite then said "already present", because that check
    skipped the fingerprint.
    """
    models = tmp_path / "models"
    make_bundle(models)                      # the working copy, fingerprint and all
    mf.build(models, tmp_path / "dist")      # a release of it

    # Now publish a release whose fingerprint belongs to a different install.
    other = tmp_path / "other"
    make_bundle(other, golden="deadbeef")
    mf.build(other, tmp_path / "bad")

    installed = tmp_path / "installed"
    good, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")
    mf.install(good, base, installed, tmp_path / "cache")
    working = (installed / "dili" / "bundle.json").read_text()

    bad, bad_base = mf.read_manifest(str(tmp_path / "bad"), tmp_path / "cache")
    results = mf.install(bad, bad_base, installed, tmp_path / "cache", overwrite=True)

    assert results[0]["installed"] is False
    assert results[0]["ok"] is False
    assert "nothing was replaced" in results[0]["reason"]
    # The working bundle is untouched, and no staging directory is left behind.
    assert (installed / "dili" / "bundle.json").read_text() == working
    assert not list(installed.glob(".*.incoming"))


def test_an_existing_bundle_that_does_not_verify_here_is_not_called_present(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")

    installed = tmp_path / "installed"
    make_bundle(installed, golden="deadbeef")   # a local bundle from another install

    results = mf.install(manifest, base, installed, tmp_path / "cache")

    # Not "already present": the local copy does not work here, so the published
    # one is installed over it.
    assert results[0]["installed"] is True
    assert results[0]["ok"] is True


def test_a_manifest_endpoint_key_that_is_a_path_is_refused(tmp_path):
    """The key becomes a directory name under the models directory."""
    import json as _json

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / mf.MANIFEST_FILE).write_text(
        _json.dumps({
            "manifest_version": 1, "n_bundles": 1,
            "bundles": [{"endpoint": "../../escaped", "archive": "x.tar.gz",
                         "archive_sha256": "0" * 64, "archive_bytes": 1, "files": []}],
        }),
        encoding="utf-8",
    )
    manifest, base = mf.read_manifest(str(dist), tmp_path / "cache")
    with pytest.raises(NetworkError, match="not a usable key"):
        mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")


def test_a_manifest_archive_name_that_is_a_path_is_refused(tmp_path):
    import json as _json

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / mf.MANIFEST_FILE).write_text(
        _json.dumps({
            "manifest_version": 1, "n_bundles": 1,
            "bundles": [{"endpoint": "dili", "archive": "../../etc/passwd",
                         "archive_sha256": "0" * 64, "archive_bytes": 1, "files": []}],
        }),
        encoding="utf-8",
    )
    manifest, base = mf.read_manifest(str(dist), tmp_path / "cache")
    with pytest.raises(NetworkError, match="not a plain filename"):
        mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")


def test_a_truncated_fold_is_caught_before_it_replaces_anything(tmp_path):
    """Counting fold files is not checking them.

    A bundle with one truncated fold packaged and installed cleanly, both commands
    exiting 0, and then the prediction that worked before the update failed with
    `LightGBMError: Unknown model format` -- after the working bundle was gone.
    """
    models = tmp_path / "models"
    directory = make_bundle(models)
    fold = directory / "fold_0.txt"
    fold.write_text(fold.read_text().split("end of trees")[0], encoding="utf-8")

    with pytest.raises(ModelError, match="cut short"):
        mf.build(models, tmp_path / "dist")


def test_a_fold_missing_its_last_bytes_is_caught_too(tmp_path):
    """The marker a complete model ends with is not proof the model is complete.

    A real fold short by two bytes keeps that marker, passed the structural check,
    packaged and installed with both commands exiting 0, and then failed the first
    prediction when LightGBM read its trailing metadata. Only loading it settles it.
    """
    models = tmp_path / "models"
    directory = make_bundle(models)
    fold = directory / "fold_0.txt"
    fold.write_bytes(fold.read_bytes()[:-2])

    with pytest.raises(ModelError, match="could not be loaded by LightGBM"):
        mf.build(models, tmp_path / "dist")


def test_a_fold_of_the_wrong_width_is_caught_though_it_loads(tmp_path):
    """An intact booster from another feature preset is not a damaged file.

    Nothing about reading it says anything is wrong; it has to be compared with the
    basis the bundle ships, or it loads cleanly and predicts on the wrong width.
    """
    models = tmp_path / "models"
    directory = make_bundle(models)
    (directory / "fold_0.txt").write_text(_real_booster(7), encoding="utf-8")

    with pytest.raises(ModelError, match="different feature space"):
        mf.build(models, tmp_path / "dist")


def test_the_tuning_store_cannot_be_named_as_an_endpoint(tmp_path):
    """`studies` is a well-shaped key and a directory this project owns."""
    import json as _json

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / mf.MANIFEST_FILE).write_text(
        _json.dumps({
            "manifest_version": 1, "n_bundles": 1,
            "bundles": [{"endpoint": "studies", "archive": "studies.tar.gz",
                         "archive_sha256": "0" * 64, "archive_bytes": 1, "files": []}],
        }),
        encoding="utf-8",
    )
    manifest, base = mf.read_manifest(str(dist), tmp_path / "cache")
    with pytest.raises(NetworkError, match="keeps for itself"):
        mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")


def test_an_archive_whose_bundle_is_for_another_endpoint_is_refused(tmp_path):
    """Correct member names are not proof of correct contents.

    The member-path check catches an archive whose files sit under the wrong
    directory. This is the case it cannot catch: members named for the endpoint being
    installed, holding a record that says it is a different model.
    """
    import gzip
    import io
    import json as _json

    from admetus.data.download import sha256_file

    models = tmp_path / "models"
    directory = make_bundle(models, key="dili")

    # Repack dili's files as if they were herg's, member names included.
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for path in sorted(directory.iterdir()):
            if path.suffix not in (".json", ".npz", ".txt"):
                continue
            info = tar.gettarinfo(str(path), arcname=f"herg/{path.name}")
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "herg.tar.gz"
    with archive.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())

    (dist / mf.MANIFEST_FILE).write_text(
        _json.dumps({
            "manifest_version": 1, "n_bundles": 1,
            "bundles": [{"endpoint": "herg", "archive": "herg.tar.gz",
                         "archive_sha256": sha256_file(archive),
                         "archive_bytes": archive.stat().st_size, "files": []}],
        }),
        encoding="utf-8",
    )

    manifest, base = mf.read_manifest(str(dist), tmp_path / "cache")
    results = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")

    assert results[0]["installed"] is False
    assert "not 'herg'" in results[0]["problems"][0]
    assert not (tmp_path / "installed" / "herg").exists()
