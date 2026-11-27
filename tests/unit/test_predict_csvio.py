"""The CSV contract: only the added columns may differ.

The fixtures here are deliberately awkward, because every one of these features
is something a real input file turns out to have.
"""

from __future__ import annotations

import pytest

from admetus.errors import InputDataError
from admetus.predict import csvio

#: One file carrying every hazard at once: a byte-order mark, duplicate column
#: names, CRLF endings, a comma inside a quoted field, a newline inside a quoted
#: field, escaped double quotes, a zero-padded identifier, and no final newline.
NASTY = (
    "﻿id,smiles,note,note\r\n"
    '001,"CC(=O)O",hello,x\r\n'
    '"0,2",CCO,"multi\nline",y\r\n'
    '003,CCN,"say ""hi""",z\r\n'
    "005,CCCC,tail,v"
)


@pytest.fixture
def nasty(tmp_path):
    path = tmp_path / "nasty.csv"
    path.write_bytes(NASTY.encode("utf-8", "surrogateescape"))
    return path


def test_round_trip_with_no_new_columns_is_byte_identical(nasty, tmp_path):
    original = nasty.read_bytes()
    dialect = csvio.sniff(nasty)
    out = tmp_path / "same.csv"
    with csvio.Appender(out, dialect) as appender:
        for record in csvio.read_records(nasty, dialect):
            appender.write(record, [])
    assert out.read_bytes() == original


def test_duplicate_column_names_survive(nasty):
    header = next(csvio.read_records(nasty, csvio.sniff(nasty))).fields
    assert header.count("note") == 2


def test_bom_is_detected_and_does_not_break_column_matching(nasty):
    dialect = csvio.sniff(nasty)
    assert dialect.has_bom
    header = next(csvio.read_records(nasty, dialect)).fields
    assert header[0].startswith(csvio.BOM)
    # Matching strips it, so the first column is still findable by name.
    assert csvio.find_column(header, "id") == 0


def test_a_quoted_field_containing_a_newline_is_one_record(nasty):
    records = list(csvio.read_records(nasty, csvio.sniff(nasty)))
    assert len(records) == 5
    assert records[2].fields[2] == "multi\nline"


def test_smiles_are_parsed_despite_the_quoting(nasty):
    records = list(csvio.read_records(nasty, csvio.sniff(nasty)))
    assert [r.fields[1] for r in records[1:]] == ["CC(=O)O", "CCO", "CCN", "CCCC"]


def test_appending_leaves_every_original_field_untouched(nasty, tmp_path):
    dialect = csvio.sniff(nasty)
    records = list(csvio.read_records(nasty, dialect))
    out = tmp_path / "plus.csv"
    with csvio.Appender(out, dialect) as appender:
        for index, record in enumerate(records):
            appender.write(record, ["p", "q"] if index == 0 else ["0.5", "ok"])

    back = list(csvio.read_records(out, csvio.sniff(out)))
    assert len(back) == len(records)
    for new, old in zip(back, records):
        assert new.fields[: len(old.fields)] == old.fields
        assert len(new.fields) == len(old.fields) + 2


def test_no_newline_is_added_at_the_end(nasty, tmp_path):
    dialect = csvio.sniff(nasty)
    out = tmp_path / "plus.csv"
    with csvio.Appender(out, dialect) as appender:
        for index, record in enumerate(csvio.read_records(nasty, dialect)):
            appender.write(record, ["p"] if index == 0 else ["0.5"])
    assert not out.read_bytes().endswith(b"\n")


def test_crlf_endings_are_preserved(nasty, tmp_path):
    original = nasty.read_bytes()
    dialect = csvio.sniff(nasty)
    out = tmp_path / "plus.csv"
    with csvio.Appender(out, dialect) as appender:
        for index, record in enumerate(csvio.read_records(nasty, dialect)):
            appender.write(record, ["p"] if index == 0 else ["0.5"])
    assert out.read_bytes().count(b"\r\n") == original.count(b"\r\n")


@pytest.mark.parametrize("encoding", ["utf-8", "cp932", "latin-1"])
def test_text_in_other_encodings_survives_without_being_declared(tmp_path, encoding):
    """Only the SMILES column is interpreted, so the file's encoding need not be known."""
    text = "id,smiles,memo\n1,CCO,日本語\n2,CCN,テスト\n"
    source = tmp_path / f"{encoding}.csv"
    try:
        source.write_bytes(text.encode(encoding))
    except UnicodeEncodeError:
        pytest.skip(f"{encoding} cannot represent the fixture")
    original = source.read_bytes()

    dialect = csvio.sniff(source)
    records = list(csvio.read_records(source, dialect))
    assert [r.fields[1] for r in records[1:]] == ["CCO", "CCN"]

    out = tmp_path / f"{encoding}_out.csv"
    with csvio.Appender(out, dialect) as appender:
        for record in records:
            appender.write(record, [])
    assert out.read_bytes() == original


@pytest.mark.parametrize(("sep", "name"), [("\t", "a.tsv"), (";", "b.csv"), ("|", "c.csv")])
def test_delimiters_are_detected_and_used_for_the_new_fields(tmp_path, sep, name):
    source = tmp_path / name
    source.write_text(f"id{sep}smiles{sep}x\n1{sep}CCO{sep}a\n", encoding="utf-8")
    dialect = csvio.sniff(source)
    assert dialect.sep == sep
    out = tmp_path / f"out_{name}"
    with csvio.Appender(out, dialect) as appender:
        for index, record in enumerate(csvio.read_records(source, dialect)):
            appender.write(record, ["p"] if index == 0 else ["0.5"])
    assert out.read_text(encoding="utf-8") == f"id{sep}smiles{sep}x{sep}p\n1{sep}CCO{sep}a{sep}0.5\n"


def test_an_explicit_separator_overrides_detection(tmp_path):
    source = tmp_path / "x.csv"
    source.write_text("id;smiles\n1;CCO\n", encoding="utf-8")
    assert csvio.sniff(source, sep=";").sep == ";"


def test_a_new_field_needing_quotes_gets_them(tmp_path):
    source = tmp_path / "x.csv"
    source.write_text("id,smiles\n1,CCO\n", encoding="utf-8")
    dialect = csvio.sniff(source)
    out = tmp_path / "out.csv"
    with csvio.Appender(out, dialect) as appender:
        for index, record in enumerate(csvio.read_records(source, dialect)):
            appender.write(record, ["note"] if index == 0 else ['a,b "quoted"'])
    assert '"a,b ""quoted"""' in out.read_text(encoding="utf-8")


def test_ragged_rows_are_refused_by_default(tmp_path):
    source = tmp_path / "ragged.csv"
    source.write_text("id,smiles,x\n1,CCO\n", encoding="utf-8")
    dialect = csvio.sniff(source)
    records = list(csvio.read_records(source, dialect))
    policy = csvio.RaggedPolicy("error")
    with pytest.raises(InputDataError, match="misalign"):
        policy.check(records[1], len(records[0].fields), source)


def test_ragged_rows_can_be_allowed_deliberately(tmp_path):
    source = tmp_path / "ragged.csv"
    source.write_text("id,smiles,x\n1,CCO\n", encoding="utf-8")
    dialect = csvio.sniff(source)
    records = list(csvio.read_records(source, dialect))
    policy = csvio.RaggedPolicy("pad")
    policy.check(records[1], len(records[0].fields), source)
    assert policy.seen == 1


def test_column_detection_by_convention_and_by_name(tmp_path):
    assert csvio.find_column(["id", "SMILES"], None) == 1
    assert csvio.find_column(["id", "structure"], None) == 1
    assert csvio.find_column(["id", "Canonical_SMILES"], "canonical_smiles") == 1
    with pytest.raises(InputDataError, match="--smiles-column"):
        csvio.find_column(["id", "name"], None)
    with pytest.raises(InputDataError, match="no column named"):
        csvio.find_column(["id", "smiles"], "nope")


def test_collision_policies():
    header = ["id", "dili"]
    assert csvio.unique_names(header, ["dili"], "suffix") == ["dili_admetus"]
    assert csvio.unique_names(header, ["dili"], "replace") == ["dili"]
    with pytest.raises(InputDataError, match="--column-prefix"):
        csvio.unique_names(header, ["dili"], "error")
    # Repeated collisions keep counting rather than colliding again.
    assert csvio.unique_names(["dili", "dili_admetus"], ["dili"], "suffix") == ["dili_admetus2"]


def test_an_interrupted_write_leaves_no_output(tmp_path):
    """A truncated file that looks like a result is worse than no file."""
    source = tmp_path / "x.csv"
    source.write_text("id,smiles\n1,CCO\n", encoding="utf-8")
    dialect = csvio.sniff(source)
    out = tmp_path / "out.csv"
    with pytest.raises(RuntimeError), csvio.Appender(out, dialect) as appender:
        appender.write(next(csvio.read_records(source, dialect)), ["p"])
        raise RuntimeError("interrupted")
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


def test_missing_and_empty_inputs_are_refused(tmp_path):
    with pytest.raises(InputDataError, match="not found"):
        csvio.sniff(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(InputDataError, match="empty"):
        csvio.sniff(empty)
