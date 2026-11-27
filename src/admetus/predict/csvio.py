"""Reading a CSV, adding columns, and leaving everything else byte for byte intact.

The requirement is that only the new prediction columns differ between input and
output. Round-tripping through a dataframe does not achieve that, and the ways it
fails are not exotic:

===========================  ==================================================
Input feature                What a dataframe round-trip does to it
===========================  ==================================================
Duplicate column names       Renames them, so header content is lost
Quoted fields                ``a,"b",c`` is rewritten as ``a,b,c``
Short rows / trailing comma  Missing fields become empty strings
CRLF line endings            Rewritten to the platform's newline
Byte-order mark              Sticks to the first column name, so it stops matching
Non-UTF-8 text               Decode error, or mojibake
Final line without newline   Gains one
Zero-padded identifiers      ``"0001"`` becomes ``1``
===========================  ==================================================

So the passthrough fields are never re-serialized. Each record's **raw text** is
kept, the new fields are appended to it, and the original line terminator is put
back. Only the SMILES column is ever interpreted.

Opening with ``errors="surrogateescape"`` means the input's encoding does not even
need to be known: bytes that are not valid UTF-8 survive the decode and the encode
unchanged, while the SMILES column -- ASCII by construction -- still parses.
"""

from __future__ import annotations

import contextlib
import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from admetus.errors import InputDataError

BOM = "﻿"
DELIMITERS = ",\t;|"

#: csv's default field limit is too small for a long SMILES-plus-metadata row.
csv.field_size_limit(min(2**31 - 1, 10_000_000))


@dataclass
class Dialect:
    """How a particular file is laid out.

    Attributes:
        sep: Field delimiter.
        has_bom: Whether the file starts with a byte-order mark.
        encoding: Text encoding used for both reading and writing.
    """

    sep: str = ","
    has_bom: bool = False
    encoding: str = "utf-8"


class _LineCapture:
    """An iterator that remembers the physical lines ``csv.reader`` consumed.

    A record can span several lines when a quoted field contains a newline, so the
    reader is fed through this and asked afterwards what it used.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer: list[str] = []
        self._first = True

    def __iter__(self) -> _LineCapture:
        return self

    def __next__(self) -> str:
        line = next(self._stream)
        self._buffer.append(line)
        if self._first:
            self._first = False
            # The parser must not see the byte-order mark, while the raw text keeps
            # it: a BOM sitting before an opening quote stops csv from recognizing
            # the field as quoted, so `"smiles","id"` parsed as a column literally
            # called '"smiles"' and the SMILES column could not be found. Stripping
            # it afterwards is too late. `take()` still returns the mark, because
            # the output is a byte-for-byte passthrough of the input.
            return line.lstrip(BOM)
        return line

    def take(self) -> str:
        """Return and clear the raw text of the record just parsed."""
        text = "".join(self._buffer)
        self._buffer.clear()
        return text


@dataclass
class Record:
    """One parsed record and the text it came from.

    Attributes:
        fields: Parsed fields.
        body: Raw text with the line terminator removed.
        terminator: The terminator that followed it, possibly empty on the last line.
        number: 1-based record number, the header being record 1.
    """

    fields: list[str]
    body: str
    terminator: str
    number: int


def sniff(path: Path, encoding: str = "utf-8", sep: str | None = None) -> Dialect:
    """Work out a file's delimiter and whether it carries a byte-order mark.

    Raises:
        InputDataError: The file is missing or empty.
    """
    path = Path(path)
    if not path.is_file():
        raise InputDataError(f"input file not found: {path}")
    with path.open("r", encoding=encoding, errors="surrogateescape", newline="") as handle:
        first = handle.readline()
    if not first:
        raise InputDataError(f"{path} is empty")

    has_bom = first.startswith(BOM)
    if sep:
        return Dialect(sep=sep, has_bom=has_bom, encoding=encoding)

    sample = first.lstrip(BOM)
    try:
        detected = csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        # One column, or a header the sniffer cannot read. Comma is the safe guess
        # and the delimiter only matters for finding the SMILES column.
        detected = ","
    return Dialect(sep=detected, has_bom=has_bom, encoding=encoding)


def find_column(header: list[str], name: str | None) -> int:
    """Locate the SMILES column.

    A byte-order mark is stripped before matching, since otherwise it silently
    glues itself to the first column's name. Matching is case-insensitive, and
    falls back to the conventional names when none is given.

    Raises:
        InputDataError: No column matches, with the available names listed.
    """
    cleaned = [h.lstrip(BOM).strip() for h in header]
    if name:
        for index, column in enumerate(cleaned):
            if column == name:
                return index
        for index, column in enumerate(cleaned):
            if column.casefold() == name.casefold():
                return index
        raise InputDataError(
            f"no column named {name!r}. The file has: {cleaned}"
        )

    for candidate in ("smiles", "SMILES", "structure", "canonical_smiles", "drug", "X"):
        for index, column in enumerate(cleaned):
            if column.casefold() == candidate.casefold():
                return index
    raise InputDataError(
        f"could not find a SMILES column among {cleaned}. Pass --smiles-column NAME."
    )


def read_records(path: Path, dialect: Dialect) -> Iterator[Record]:
    """Yield every record, header included, with its raw text preserved."""
    path = Path(path)
    with path.open("r", encoding=dialect.encoding, errors="surrogateescape", newline="") as handle:
        capture = _LineCapture(handle)
        reader = csv.reader(capture, delimiter=dialect.sep)
        for number, fields in enumerate(reader, start=1):
            raw = capture.take()
            body = raw.rstrip("\r\n")
            yield Record(fields=fields, body=body, terminator=raw[len(body):], number=number)


def split_terminator(raw: str) -> tuple[str, str]:
    """Split a raw record into its body and its line terminator."""
    body = raw.rstrip("\r\n")
    return body, raw[len(body):]


class Appender:
    """Writes records back out with extra fields appended.

    The output is written to a sibling ``.part`` file and renamed at the end, so a
    truncated run leaves no file rather than a file that looks like a result.
    """

    def __init__(self, path: Path | None, dialect: Dialect, stdout: TextIO | None = None) -> None:
        self.dialect = dialect
        self._path = Path(path) if path is not None else None
        self._partial: Path | None = None
        self._stdout = stdout
        self._handle: TextIO | None = None

    def __enter__(self) -> Appender:
        if self._stdout is not None:
            # Standard output has to carry the input's encoding too. Writing through
            # the terminal's encoding turned a cp932 passthrough column into UTF-8,
            # so `predict --encoding cp932 > out.csv` and `-o out.csv` produced
            # different bytes from one input -- and only one of them was a
            # passthrough. Reconfiguring the stream is enough; it is the same
            # handle, so nothing about buffering or ownership changes.
            with contextlib.suppress(AttributeError, ValueError):
                self._stdout.reconfigure(
                    encoding=self.dialect.encoding, errors="surrogateescape", newline=""
                )
            self._handle = self._stdout
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._partial = self._path.with_name(self._path.name + ".part")
        self._handle = self._partial.open(
            "w", encoding=self.dialect.encoding, errors="surrogateescape", newline=""
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stdout is not None:
            self._handle.flush()
            return
        self._handle.close()
        if exc_type is None:
            self._partial.replace(self._path)
        else:
            self._partial.unlink(missing_ok=True)

    def write(self, record: Record, extra: list[str]) -> None:
        """Append ``extra`` fields to a record and write it unchanged otherwise.

        The record's own terminator is written back verbatim, including when it is
        empty because the file ended without a newline. Supplying one would change
        the output's bytes for no reason.
        """
        fields = self.dialect.sep.join(_quote(value, self.dialect.sep) for value in extra)
        joiner = self.dialect.sep if extra else ""
        self._handle.write(record.body + joiner + fields + record.terminator)


def _quote(value: str, sep: str) -> str:
    """Quote a new field only when it needs it.

    Only applied to fields this tool creates; passthrough text is never touched.
    """
    if value == "":
        return value
    if any(ch in value for ch in (sep, '"', "\n", "\r")):
        return '"' + value.replace('"', '""') + '"'
    return value


def unique_names(header: list[str], wanted: list[str], on_collision: str, suffix: str = "_admetus") -> list[str]:
    """Resolve new column names against the ones already present.

    Args:
        header: Existing column names.
        wanted: Names this run wants to add.
        on_collision: ``suffix`` to rename, ``error`` to refuse, ``replace`` to
            keep the wanted name and let the duplicate stand.
        suffix: Appended when renaming.

    Returns:
        The names actually used, in the order given.

    Raises:
        InputDataError: A name collides and ``on_collision`` is ``error``.
    """
    existing = {h.lstrip(BOM).strip() for h in header}
    out: list[str] = []
    for name in wanted:
        if name not in existing and name not in out:
            out.append(name)
            continue
        if on_collision == "error":
            raise InputDataError(
                f"the input already has a column named {name!r}. Use --column-prefix, or "
                "--on-collision suffix to rename the new columns."
            )
        if on_collision == "replace":
            out.append(name)
            continue
        candidate = f"{name}{suffix}"
        counter = 2
        while candidate in existing or candidate in out:
            candidate = f"{name}{suffix}{counter}"
            counter += 1
        out.append(candidate)
    return out


@dataclass
class RaggedPolicy:
    """What to do with a record whose field count differs from the header.

    Padding changes the passthrough bytes; not padding misaligns the appended
    columns. Neither is silently acceptable, so the default refuses.
    """

    mode: str = "error"
    seen: int = field(default=0)
    n_short: int = field(default=0)
    n_surplus: int = field(default=0)

    def check(self, record: Record, n_header: int, path: Path) -> int:
        """Raise, or return how many empty fields to insert before the new columns.

        ``pad`` used only to count, which left the appended columns shifted into the
        row's missing fields: with the header ``id,smiles,note`` and the row
        ``1,CCO``, the prediction landed in ``note`` and its ensemble spread landed
        in the prediction's own column. Reading the output by header name then gave
        confident wrong numbers, and the run still exited 0.

        A short row is therefore padded for real, which the mode's name has always
        promised. A row with *surplus* fields cannot be fixed by padding -- the
        extra fields are already past the header's last column -- so it is reported
        whatever the mode, because silently appending after them would put the
        prediction columns at a different offset than the header says.

        Returns:
            How many empty fields to insert before the appended columns.

        Raises:
            InputDataError: The record is ragged and the policy is ``error``, or it
                carries more fields than the header under any policy.
        """
        n_fields = len(record.fields)
        if n_fields == n_header:
            return 0
        self.seen += 1
        if n_fields > n_header:
            self.n_surplus += 1
            raise InputDataError(
                f"{path.name} line {record.number} has {n_fields} field(s) but the header has "
                f"{n_header}. The extra field(s) sit where the prediction columns would go, so "
                "the output could not be read by header name. Fix the row, or drop the surplus "
                "field(s) before predicting."
            )
        self.n_short += 1
        if self.mode == "error":
            raise InputDataError(
                f"{path.name} line {record.number} has {n_fields} field(s) but the "
                f"header has {n_header}. Appending columns to a ragged row would misalign them. "
                "Pass --on-ragged pad to append anyway."
            )
        return n_header - n_fields


__all__ = [
    "BOM",
    "Appender",
    "Dialect",
    "RaggedPolicy",
    "Record",
    "find_column",
    "read_records",
    "sniff",
    "split_terminator",
    "unique_names",
]
