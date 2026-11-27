"""Dataset digest pinning.

The registry is the source of truth for which bytes a model was trained on. A
file that disagrees is refused, and the pin is never rewritten by a fetch.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from admetus.data.download import (
    CHECKSUM_FILE,
    fetch_dataset,
    read_checksums,
    sha256_file,
    verify_dataset,
)
from admetus.data.registry import get_endpoint
from admetus.errors import NetworkError

CONTENT = b"Drug_ID\tDrug\tY\na\tCCO\t1\nb\tCCN\t0\n"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
WRONG = "0" * 64


@pytest.fixture
def staged(tmp_path):
    """A dataset file already on disk, as a cached fetch would leave it."""
    (tmp_path / "fixture.tab").write_bytes(CONTENT)
    return tmp_path


def test_matching_digest_is_accepted(staged):
    result = fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=DIGEST)
    assert result.from_cache
    assert result.sha256 == DIGEST


def test_mismatched_digest_is_refused_not_adopted(staged):
    with pytest.raises(NetworkError, match="does not match the digest pinned"):
        fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=WRONG)


def test_refusal_names_both_digests_and_the_remedy(staged):
    with pytest.raises(NetworkError) as caught:
        fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=WRONG)
    message = str(caught.value)
    assert WRONG in message
    assert DIGEST in message
    assert "--accept-upstream-change" in message


def test_a_cached_file_that_fails_its_digest_is_moved_aside(staged):
    """The canonical path must be cleared so nothing later reads the bad bytes.

    The file is renamed rather than deleted: clearing the path is what matters,
    and keeping the bytes lets someone see what actually arrived.
    """
    before = (staged / "fixture.tab").read_bytes()
    with pytest.raises(NetworkError, match="moved aside"):
        fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=WRONG)

    assert not (staged / "fixture.tab").exists()
    quarantined = staged / "fixture.tab.mismatch"
    assert quarantined.read_bytes() == before
    # And nothing was recorded for a fetch that did not happen.
    assert "fixture" not in read_checksums(staged)


def test_a_refusal_leaves_no_temporary_file(staged):
    """Mismatched bytes must not be left behind at the partial path either."""
    with pytest.raises(NetworkError):
        fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=WRONG)
    assert not list(staged.glob("*.part"))


def test_a_loader_will_not_read_a_file_that_fails_its_pin(tmp_path):
    """The fetcher is not the only guard: a file can be altered after it is fetched."""
    from admetus.data import loader
    from admetus.data.registry import Endpoint
    from admetus.errors import InputDataError

    (tmp_path / "fixture.tab").write_bytes(CONTENT + b"c\tCCC\t1\n")
    endpoint = Endpoint(
        key="fixture", tdc_name="F", dataverse_id=1, task="classification",
        metric="auroc", unit="binary", in_official_group=False,
        description="x", sha256=DIGEST,
    )
    with pytest.raises(InputDataError, match="does not match the digest pinned"):
        loader.load(endpoint, tmp_path)

    # The same file reads fine once the check is deliberately skipped.
    assert len(loader.load(endpoint, tmp_path, verify_digest=False)) == 3


def test_the_accept_path_is_pinned_to_the_digest_it_reported(staged, monkeypatch):
    """Accepting a change must adopt the bytes the warning named, not a later fetch.

    Re-fetching unpinned would let a second response differ from what the operator
    was shown and agreed to, which is the gap the pin exists to close.
    """
    import admetus.data.download as module

    calls: list[str | None] = []
    real = module.fetch

    def spy(url, destination, timeout=120.0, overwrite=False, expected_sha256=None):
        calls.append(expected_sha256)
        return real(url, destination, timeout, overwrite, expected_sha256)

    monkeypatch.setattr(module, "fetch", spy)
    result = module.fetch_dataset(
        "fixture", "https://example.invalid/x", staged,
        expected_sha256=WRONG, accept_upstream_change=True,
    )
    assert result.sha256 == DIGEST
    # First attempt pinned to the registry value, retry pinned to what was found.
    assert calls == [WRONG, DIGEST]


def test_the_pinned_value_is_not_rewritten_by_a_fetch(staged):
    """A fetch records what it observed; it never edits the pin."""
    before = get_endpoint("dili").sha256
    with pytest.raises(NetworkError):
        fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=WRONG)
    from admetus.data.registry import load_registry

    load_registry.cache_clear()
    assert get_endpoint("dili").sha256 == before


def test_the_escape_hatch_accepts_deliberately(staged):
    result = fetch_dataset(
        "fixture", "https://example.invalid/x", staged,
        expected_sha256=WRONG, accept_upstream_change=True,
    )
    assert result.sha256 == DIGEST
    record = read_checksums(staged)["fixture"]
    assert record["matches_pin"] is False
    assert record["pinned"] == WRONG


def test_the_observed_record_is_not_the_source_of_truth(staged):
    """A stale observed record must not override the pin."""
    (staged / CHECKSUM_FILE).write_text(
        json.dumps({"fixture": {"sha256": WRONG, "bytes": 1, "url": "x"}}), encoding="utf-8"
    )
    result = fetch_dataset("fixture", "https://example.invalid/x", staged, expected_sha256=DIGEST)
    assert result.sha256 == DIGEST
    assert read_checksums(staged)["fixture"]["matches_pin"] is True


def test_verify_checks_the_cached_file_against_the_pin(staged):
    ok, detail = verify_dataset("fixture", staged, expected_sha256=DIGEST)
    assert ok
    assert "pinned" in detail

    (staged / "fixture.tab").write_bytes(CONTENT + b"tampered\n")
    ok, detail = verify_dataset("fixture", staged, expected_sha256=DIGEST)
    assert not ok
    assert "does not match the pin" in detail


def test_verify_reports_a_missing_file(tmp_path):
    ok, detail = verify_dataset("fixture", tmp_path, expected_sha256=DIGEST)
    assert not ok
    assert "missing" in detail


def test_unpinned_endpoint_falls_back_to_the_observed_record(staged):
    ok, detail = verify_dataset("fixture", staged)
    assert not ok and "none was recorded" in detail
    fetch_dataset("fixture", "https://example.invalid/x", staged)
    ok, detail = verify_dataset("fixture", staged)
    assert ok and "unpinned" in detail


def test_every_registry_endpoint_carries_a_pin():
    from admetus.data.registry import all_endpoints

    unpinned = [e.key for e in all_endpoints() if not e.sha256]
    assert unpinned == [], f"these endpoints have no pinned digest: {unpinned}"


def test_pinned_digests_are_well_formed():
    from admetus.data.registry import all_endpoints

    for endpoint in all_endpoints():
        assert len(endpoint.sha256) == 64
        assert all(c in "0123456789abcdef" for c in endpoint.sha256)


def test_sha256_file_matches_hashlib(staged):
    assert sha256_file(staged / "fixture.tab") == DIGEST
