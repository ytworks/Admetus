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


def make_bundle(models_dir, key="fixture", n_boosters=1):
    """Write a directory that passes ``verify(check_fingerprint=False)``."""
    from admetus.chem.spec import get_spec

    spec = get_spec("morgan")
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
        (directory / f"fold_{index}.txt").write_text("tree\n", encoding="utf-8")
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
                    "golden_fingerprint": "deadbeef",
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
    assert entry["endpoint"] == "fixture"
    assert entry["archive_sha256"] and len(entry["archive_sha256"]) == 64
    names = sorted(f["name"] for f in entry["files"])
    assert names == ["ad_reference.npz", "bundle.json", "columns.npz", "fold_0.txt"]
    # The record a caller reads before downloading anything.
    assert entry["held_out"]["value"] == 0.9
    assert entry["feature_set"] == "morgan"
    assert (tmp_path / "dist" / mf.MANIFEST_FILE).is_file()


def test_the_study_database_is_not_published(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    with tarfile.open(tmp_path / "dist" / "fixture.tar.gz") as tar:
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
    installed = tmp_path / "installed" / "fixture"
    assert sorted(p.name for p in installed.iterdir()) == [
        "ad_reference.npz", "bundle.json", "columns.npz", "fold_0.txt"
    ]
    # The golden fingerprint of the fixture is nonsense on purpose, so a real
    # install reports the mismatch rather than hiding it.
    assert results[0]["ok"] is False
    assert any("fingerprint" in problem for problem in results[0]["problems"])


def test_an_already_installed_bundle_is_left_alone(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")

    mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")
    again = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")
    assert again == [{"endpoint": "fixture", "installed": False, "reason": "already present"}]


def test_an_archive_whose_digest_disagrees_is_not_unpacked(tmp_path):
    models = tmp_path / "models"
    make_bundle(models)
    mf.build(models, tmp_path / "dist")
    archive = tmp_path / "dist" / "fixture.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    manifest, base = mf.read_manifest(str(tmp_path / "dist"), tmp_path / "cache")
    results = mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache")

    assert results[0]["installed"] is False
    assert "digest" in results[0]["reason"]
    assert not (tmp_path / "installed" / "fixture").exists()


def test_an_archive_member_outside_the_bundle_is_refused(tmp_path):
    """A manifest can be honest about bytes that still hold a hostile path."""
    dist = tmp_path / "dist"
    dist.mkdir()
    archive = dist / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = tmp_path / "payload"
        payload.write_text("x", encoding="utf-8")
        tar.add(payload, arcname="../escaped.txt")

    from admetus.data.download import sha256_file

    manifest = {
        "manifest_version": 1, "n_bundles": 1,
        "bundles": [{
            "endpoint": "fixture", "archive": "fixture.tar.gz",
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
    with pytest.raises(ModelError, match="no bundle for dili"):
        mf.install(manifest, base, tmp_path / "installed", tmp_path / "cache", keys=["dili"])
