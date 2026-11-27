"""Fetching published datasets.

The datasets live on Harvard Dataverse, one file per id, and are retrieved with
``urllib`` from the standard library. Nothing here needs a heavyweight client, and
keeping the dependency out means the inference install stays small.

Downloads are atomic: a file is written to ``<name>.part`` and renamed only once
it is complete and its digest recorded. A half-written dataset is never mistaken
for a whole one.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from admetus.errors import NetworkError
from admetus.logging import get_logger

log = get_logger("data.download")

#: A plain client name. Dataverse serves anonymous requests, and the user's own
#: identity is never attached to an outbound request.
USER_AGENT = "admetus/dataset-fetch"

CHUNK_BYTES = 1 << 16
DEFAULT_TIMEOUT = 120.0
CHECKSUM_FILE = "checksums.json"


@dataclass(frozen=True)
class Download:
    """A dataset file on disk.

    Attributes:
        path: Where the file landed.
        sha256: Digest of its contents.
        n_bytes: Size in bytes.
        from_cache: Whether it was already present and left untouched.
    """

    path: Path
    sha256: str
    n_bytes: int
    from_cache: bool


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(directory: Path) -> dict[str, dict[str, object]]:
    """Load the recorded digests for a dataset directory, or an empty mapping."""
    path = directory / CHECKSUM_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read %s; treating it as empty", path)
        return {}


def write_checksums(directory: Path, checksums: dict[str, dict[str, object]]) -> None:
    """Write the digest record atomically."""
    path = directory / CHECKSUM_FILE
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def fetch(url: str, destination: Path, timeout: float = DEFAULT_TIMEOUT, overwrite: bool = False) -> Download:
    """Download ``url`` to ``destination``.

    Args:
        url: What to fetch.
        destination: Where to put it. Parent directories are created.
        timeout: Socket timeout in seconds.
        overwrite: Fetch again even when the file is already present.

    Returns:
        A :class:`Download` describing the file on disk.

    Raises:
        NetworkError: The request failed. Distinct from a missing-model error so
            a caller can tell whether retrying is worthwhile.
    """
    destination = Path(destination)
    if destination.is_file() and not overwrite:
        log.debug("already present: %s", destination)
        return Download(destination, sha256_file(destination), destination.stat().st_size, from_cache=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    digest = hashlib.sha256()
    n_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                n_bytes += len(block)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise NetworkError(f"could not download {url}: {exc}") from exc

    if n_bytes == 0:
        partial.unlink(missing_ok=True)
        raise NetworkError(f"{url} returned an empty response")

    partial.replace(destination)
    log.debug("fetched %s (%d bytes) -> %s", url, n_bytes, destination)
    return Download(destination, digest.hexdigest(), n_bytes, from_cache=False)


def fetch_dataset(key: str, url: str, directory: Path, overwrite: bool = False) -> Download:
    """Download one dataset and record its digest alongside the others."""
    directory = Path(directory)
    result = fetch(url, directory / f"{key}.tab", overwrite=overwrite)

    checksums = read_checksums(directory)
    previous = checksums.get(key)
    if previous and previous.get("sha256") != result.sha256:
        log.warning(
            "%s changed upstream: recorded %s, now %s. The published dataset was revised.",
            key, str(previous.get("sha256"))[:16], result.sha256[:16],
        )
    checksums[key] = {"sha256": result.sha256, "bytes": result.n_bytes, "url": url}
    write_checksums(directory, checksums)
    return result


def verify_dataset(key: str, directory: Path) -> tuple[bool, str]:
    """Check one dataset against its recorded digest.

    Returns:
        ``(ok, detail)``.
    """
    directory = Path(directory)
    path = directory / f"{key}.tab"
    if not path.is_file():
        return False, "file is missing"
    recorded = read_checksums(directory).get(key)
    if not recorded:
        return False, "no digest was recorded"
    actual = sha256_file(path)
    if actual != recorded.get("sha256"):
        return False, f"digest mismatch: recorded {str(recorded.get('sha256'))[:16]}, found {actual[:16]}"
    return True, "ok"


def extract_benchmark_archive(archive: Path, destination: Path) -> list[str]:
    """Unpack the official benchmark split archive.

    The archive comes off the network, so entries that would escape the
    destination are refused before anything is written.

    Returns:
        The names that were extracted.

    Raises:
        NetworkError: The archive is not a usable zip, or holds an unsafe path.
    """
    import zipfile

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            for name in names:
                target = Path(name)
                if target.is_absolute() or ".." in target.parts:
                    raise NetworkError(f"benchmark archive holds an unsafe path: {name!r}")
            bundle.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise NetworkError(f"{archive} is not a readable zip archive: {exc}") from exc
    return names


def clear_partials(directory: Path) -> int:
    """Remove leftover ``.part`` files from an interrupted run."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.part"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def human_bytes(n: int) -> str:
    """Format a byte count for a log line."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover


__all__ = [
    "CHECKSUM_FILE",
    "Download",
    "clear_partials",
    "extract_benchmark_archive",
    "fetch",
    "fetch_dataset",
    "human_bytes",
    "read_checksums",
    "sha256_file",
    "verify_dataset",
    "write_checksums",
]
