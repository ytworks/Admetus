"""Packaging bundles for release, and installing them again.

A trained bundle is a directory of LightGBM text models plus a record. Publishing
one means answering two questions for a caller who was not there when it was trained:
*which bytes am I getting*, and *are they the ones that were published*. So a release
is a manifest plus one archive per endpoint, and the manifest carries the digest of
every archive and of every file inside it.

Three decisions worth stating:

* **One archive per endpoint, not one for everything.** A caller usually wants two or
  three endpoints, and the largest bundles here are an order of magnitude bigger than
  the smallest. A single tarball would make every install pay for all 29.
* **The archives are reproducible.** Entry order, mtime, mode, uid and gid are all
  fixed, and gzip is written with no timestamp. Packaging the same bundles twice gives
  the same bytes, so a digest in the manifest is a statement about the model rather
  than about when the tar ran.
* **Install verifies before it unpacks, and again afterwards.** The archive's digest
  is checked at the temporary path, member paths are refused if they would escape the
  destination, and the installed bundle goes through the same ``verify`` the loader
  uses. A bundle that arrives intact but does not match this install's feature space
  is a failure at install time, not at prediction time.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from admetus.bundle.load import available, read_metadata, verify
from admetus.bundle.save import BUNDLE_FILE
from admetus.data.download import fetch, sha256_file
from admetus.errors import ModelError, NetworkError
from admetus.logging import get_logger
from admetus.version import SCHEMA_VERSION, get_version

log = get_logger("bundle.manifest")

MANIFEST_FILE = "models-manifest.json"

#: Fixed timestamp for every archive member, so packaging is reproducible.
FIXED_MTIME = 0

#: Files that belong to a bundle. Anything else in the directory -- an Optuna study
#: database, for one -- is a by-product of training and is not published.
PUBLISHED_SUFFIXES = (".json", ".npz", ".txt")

#: Where a release's manifest lives, by convention, when no source is given.
DEFAULT_MANIFEST_URL = (
    f"https://github.com/ytworks/Admetus/releases/latest/download/{MANIFEST_FILE}"
)


@dataclass(frozen=True)
class PublishedFile:
    """One file inside a published archive."""

    name: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class PublishedBundle:
    """One endpoint's published archive."""

    endpoint: str
    archive: str
    archive_sha256: str
    archive_bytes: int
    files: list[PublishedFile]
    record: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "archive": self.archive,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "files": [f.to_dict() for f in self.files],
            **self.record,
        }


def _published_files(directory: Path) -> list[Path]:
    """The files of a bundle, in a fixed order."""
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix in PUBLISHED_SUFFIXES),
        key=lambda p: p.name,
    )


def _summary(directory: Path) -> dict[str, Any]:
    """The part of a bundle's record a caller needs before downloading it."""
    meta = read_metadata(directory)
    metrics = meta.get("metrics") or {}
    featurizer = meta.get("featurizer") or {}
    return {
        "task": meta["endpoint"]["task"],
        "metric": meta["endpoint"]["metric"],
        "unit": meta["endpoint"].get("unit"),
        "in_official_group": meta["endpoint"].get("in_official_group"),
        "created_at": meta.get("created_at"),
        "schema_version": meta.get("schema_version"),
        "feature_set": (featurizer.get("spec") or {}).get("feature_set"),
        "n_features": featurizer.get("n_features"),
        "spec_sha256": featurizer.get("spec_sha256"),
        "golden_fingerprint": featurizer.get("golden_fingerprint"),
        "held_out": metrics.get("primary"),
        "libraries": meta.get("libraries"),
    }


def pack_bundle(directory: Path, archive: Path) -> PublishedBundle:
    """Write one bundle to a reproducible ``.tar.gz``.

    Raises:
        ModelError: The directory does not hold a usable bundle.
    """
    directory = Path(directory)
    if not (directory / BUNDLE_FILE).is_file():
        raise ModelError(f"{directory} does not hold a model bundle")
    ok, problems = verify(directory, check_fingerprint=False)
    if not ok:
        raise ModelError(
            f"refusing to package {directory.name}: the bundle does not verify:\n  - "
            + "\n  - ".join(problems)
        )

    members = _published_files(directory)
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)

    # gzip's own header carries an mtime, so the stream is built with one fixed at
    # zero rather than letting tarfile stamp "now" into the bytes.
    import gzip

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in members:
            info = tar.gettarinfo(str(path), arcname=f"{directory.name}/{path.name}")
            info.mtime = FIXED_MTIME
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    with archive.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=FIXED_MTIME) as gz:
        gz.write(raw.getvalue())

    files = [
        PublishedFile(name=p.name, sha256=sha256_file(p), bytes=p.stat().st_size) for p in members
    ]
    log.info("packaged %s: %d file(s), %d bytes", directory.name, len(files), archive.stat().st_size)
    return PublishedBundle(
        endpoint=directory.name,
        archive=archive.name,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        files=files,
        record=_summary(directory),
    )


def build(models_dir: Path, out_dir: Path, keys: list[str] | None = None) -> dict[str, Any]:
    """Package every bundle in ``models_dir`` and write the manifest beside them.

    Args:
        models_dir: Where the trained bundles live.
        out_dir: Where the archives and the manifest are written.
        keys: Endpoints to publish. ``None`` publishes everything present.

    Returns:
        The manifest, as written.

    Raises:
        ModelError: No bundle was found, or one does not verify.
    """
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    present = available(models_dir)
    selected = [k for k in present if keys is None or k in set(keys)]
    if keys:
        unknown = sorted(set(keys) - set(present))
        if unknown:
            raise ModelError(
                f"no bundle for {', '.join(unknown)} in {models_dir}. "
                "Train them first, or leave them out of the release."
            )
    if not selected:
        raise ModelError(f"no model bundles in {models_dir}, so there is nothing to publish")

    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = [
        pack_bundle(models_dir / key, out_dir / f"{key}.tar.gz").to_dict() for key in selected
    ]
    manifest = {
        "manifest_version": 1,
        "schema_version": SCHEMA_VERSION,
        "admetus_version": get_version(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_bundles": len(bundles),
        "bundles": bundles,
    }
    path = out_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote %s describing %d bundle(s)", path, len(bundles))
    return manifest


def read_manifest(source: str, directory: Path) -> tuple[dict[str, Any], str]:
    """Read a manifest from a URL or a local path.

    Args:
        source: A URL ending in the manifest file, or a local path to one.
        directory: Where a downloaded manifest is cached.

    Returns:
        ``(manifest, base)`` where ``base`` is the URL prefix or directory the
        archives are fetched from, relative to the manifest.

    Raises:
        NetworkError: The manifest could not be read or is not usable JSON.
    """
    if source.startswith(("http://", "https://")):
        destination = Path(directory) / MANIFEST_FILE
        result = fetch(source, destination, overwrite=True)
        text = result.path.read_text(encoding="utf-8")
        base = source.rsplit("/", 1)[0]
    else:
        path = Path(source)
        if path.is_dir():
            path = path / MANIFEST_FILE
        if not path.is_file():
            raise NetworkError(f"no manifest at {path}")
        text = path.read_text(encoding="utf-8")
        base = str(path.parent)
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NetworkError(f"{source} is not a readable manifest: {exc}") from exc
    if not isinstance(manifest, dict) or "bundles" not in manifest:
        raise NetworkError(f"{source} does not look like a model manifest")
    return manifest, base


#: An endpoint key as the registry spells one: lowercase, digits, underscores. The
#: manifest comes off the network and its keys name directories, so the shape is
#: checked rather than trusted.
ENDPOINT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _checked_endpoint(value: Any) -> str:
    """Return a manifest endpoint key, or refuse it.

    The key becomes a directory name under the models directory, so a key like
    ``../../etc`` would write outside it. Refusing the shape is cheaper and more
    obvious than normalizing and then hoping the containment check catches it.
    """
    if not isinstance(value, str) or not ENDPOINT_KEY.match(value):
        raise NetworkError(
            f"the manifest names an endpoint {value!r} that is not a usable key. "
            "A key is lowercase letters, digits and underscores."
        )
    return value


def _checked_archive(value: Any, endpoint: str) -> str:
    """Return a manifest archive filename, or refuse it.

    A bare filename, nothing else: the name is joined to a download directory and to
    the release's base URL, and a path in it would reach outside either.
    """
    if not isinstance(value, str) or not value or value != Path(value).name:
        raise NetworkError(
            f"the manifest gives {endpoint} an archive {value!r} that is not a plain filename"
        )
    if value in (".", "..") or value.startswith("."):
        raise NetworkError(f"the manifest gives {endpoint} an archive name that is not usable: {value!r}")
    return value


def _contained(path: Path, root: Path, what: str) -> Path:
    """Return ``path`` resolved, having checked it stays under ``root``."""
    root = Path(root).resolve()
    resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise NetworkError(f"{what} would land outside {root}: {resolved}")
    return resolved


def _unpack(archive: Path, models_dir: Path, endpoint: str) -> Path:
    """Unpack one archive to a staging directory beside the bundle it would replace.

    Member paths come off the network, so each one is checked before anything is
    written: every entry must be a regular file directly under ``<endpoint>/``.

    The staging directory is returned rather than moved into place. Replacing a
    working bundle before the incoming one has been verified loses the working copy
    when the incoming one turns out not to load here, which is exactly when the old
    one is most wanted.
    """
    models_dir = Path(models_dir)
    staging = models_dir / f".{endpoint}.incoming"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise NetworkError(f"{archive.name} holds {member.name!r}, which is not a regular file")
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or path.parts[0] != endpoint:
                raise NetworkError(
                    f"{archive.name} holds {member.name!r}, which is not a file of the "
                    f"{endpoint!r} bundle"
                )
            member.name = path.parts[1]
            tar.extract(member, staging, filter="data")

    return staging


def install(
    manifest: dict[str, Any],
    base: str,
    models_dir: Path,
    cache_dir: Path,
    keys: list[str] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Download and install the bundles a manifest describes.

    Args:
        manifest: As returned by :func:`read_manifest`.
        base: URL prefix or directory the archives sit in.
        models_dir: Where bundles are installed.
        cache_dir: Where archives are downloaded to.
        keys: Endpoints to install. ``None`` installs everything in the manifest.
        overwrite: Reinstall an endpoint that already has a bundle.

    Returns:
        One record per endpoint, saying what happened and why.
    """
    # Every name the manifest supplies is checked before anything is written or
    # removed, because these names become directories under the models directory and
    # files under the download directory.
    bundles = []
    for entry in manifest["bundles"]:
        endpoint = _checked_endpoint(entry.get("endpoint"))
        bundles.append({**entry, "endpoint": endpoint,
                        "archive": _checked_archive(entry.get("archive"), endpoint)})

    wanted = set(keys) if keys else None
    entries = [b for b in bundles if wanted is None or b["endpoint"] in wanted]
    if wanted:
        missing = sorted(wanted - {b["endpoint"] for b in bundles})
        if missing:
            raise ModelError(
                f"the manifest has no bundle for {', '.join(missing)}. It publishes "
                f"{manifest['n_bundles']} endpoint(s)."
            )

    models_dir = Path(models_dir)
    cache_dir = Path(cache_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []

    for entry in entries:
        endpoint = entry["endpoint"]
        target = Path(_contained(Path(endpoint), models_dir, f"the bundle directory for {endpoint}"))
        if target.is_dir() and not overwrite:
            # The fingerprint is part of the test here, not just the file list. It
            # used to be skipped, so after a failed --overwrite a retry without the
            # flag reported "already present" for a bundle that does not work on
            # this install -- the one case where saying nothing is worst.
            ok, _ = verify(target, check_fingerprint=True)
            if ok:
                out.append({"endpoint": endpoint, "installed": False, "reason": "already present"})
                continue

        source = f"{base}/{entry['archive']}"
        archive = Path(
            _contained(Path(entry["archive"]), cache_dir, f"the download for {endpoint}")
        )
        try:
            if source.startswith(("http://", "https://")):
                fetch(source, archive, overwrite=True, expected_sha256=entry["archive_sha256"])
            else:
                local = Path(base) / entry["archive"]
                digest = sha256_file(local) if local.is_file() else None
                if digest != entry["archive_sha256"]:
                    raise NetworkError(
                        f"{local} does not match the digest the manifest records for it"
                    )
                archive = local
            staging = _unpack(archive, models_dir, endpoint)
        except (NetworkError, OSError) as exc:
            out.append({"endpoint": endpoint, "installed": False, "reason": str(exc), "ok": False})
            continue

        # Verified at the staging path, so a candidate that does not load here never
        # displaces a bundle that does.
        ok, problems = verify(staging, check_fingerprint=True)
        if not ok:
            shutil.rmtree(staging, ignore_errors=True)
            log.error("%s does not verify here, so it was not installed: %s", endpoint, "; ".join(problems))
            out.append({
                "endpoint": endpoint,
                "installed": False,
                "ok": False,
                "problems": problems,
                "reason": (
                    "the published bundle does not verify on this install, so nothing was "
                    "replaced" if target.is_dir() else
                    "the published bundle does not verify on this install"
                ),
            })
            continue

        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        out.append({
            "endpoint": endpoint,
            "installed": True,
            "path": str(target),
            "ok": True,
            "problems": [],
            "reason": "installed",
        })
    return out


__all__ = [
    "DEFAULT_MANIFEST_URL",
    "MANIFEST_FILE",
    "PublishedBundle",
    "PublishedFile",
    "build",
    "install",
    "pack_bundle",
    "read_manifest",
]
