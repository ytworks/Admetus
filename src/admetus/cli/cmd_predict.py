"""``admetus predict``: add prediction columns to a CSV."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import click

from admetus.bundle.load import available, load
from admetus.cli.options import (
    all_option,
    default_jobs,
    default_models_dir,
    emit,
    endpoint_option,
    json_option,
    models_dir_option,
    run_guarded,
    setup_logging,
    write_report,
)
from admetus.data.registry import resolve_keys
from admetus.errors import EXIT_PARTIAL_FAILURE, ModelError, UsageError
from admetus.predict import runner

#: Rows that `--json` will carry inline. A JSON document holding a ten-million-row
#: CSV is not a useful object, so a larger run is told to write a file instead of
#: being silently truncated.
MAX_JSON_ROWS = 10_000


@click.command("predict")
@click.option("-i", "--input", "source", type=click.Path(path_type=Path), default=None,
              help="Input CSV. All columns other than SMILES are passed through unchanged.")
@click.option("--smiles", "inline", multiple=True, metavar="SMILES",
              help="Predict these structures directly instead of reading a file. Repeatable.")
@click.option("-o", "--output", "destination", type=click.Path(path_type=Path), default=None,
              help="Output CSV. Omit to write to standard output.")
@endpoint_option
@all_option
@models_dir_option
@click.option("--smiles-column", default=None, help="Name of the SMILES column. Detected when omitted.")
@click.option("--jobs", type=int, default=None, help="Worker processes. Default scales with the input.")
@click.option("--chunksize", type=int, default=runner.DEFAULT_CHUNK_SIZE, show_default=True,
              help="Rows read, deduplicated and written per pass.")
@click.option("--batch-size", type=int, default=64, show_default=True,
              help="Molecules per worker task.")
@click.option("--memo-size", type=int, default=runner.DEFAULT_MEMO_SIZE, show_default=True,
              help="Distinct molecules remembered across chunks.")
@click.option("--ad/--no-ad", "applicability_domain", default=True, show_default=True,
              help="Compute the applicability-domain flag. Off is faster.")
@click.option("--ad-score", is_flag=True, help="Also emit the nearest-neighbour similarity itself.")
@click.option("--interval", is_flag=True, help="Also emit prediction intervals, where available.")
@click.option("--column-prefix", default="", help="Prefix every added column, e.g. admet_.")
@click.option("--on-collision", type=click.Choice(["suffix", "error", "replace"]), default="suffix",
              show_default=True, help="What to do when an added column name already exists.")
@click.option("--on-ragged", type=click.Choice(["error", "pad"]), default="error", show_default=True,
              help="What to do with a row whose field count differs from the header.")
@click.option("--strict", is_flag=True, help="Exit non-zero if any molecule fails.")
@click.option("--max-failures", type=int, default=None, help="Stop after this many failed molecules.")
@click.option("--sep", default=None, help="Field delimiter. Detected when omitted.")
@click.option("--encoding", default="utf-8", show_default=True, help="Text encoding of the input.")
@click.option("--report", "report_path", type=click.Path(path_type=Path), default=None,
              help="Write a JSON report here, even if the run fails.")
@json_option
@click.pass_context
def predict_command(
    ctx: click.Context,
    source: Path | None,
    inline: tuple[str, ...],
    destination: Path | None,
    endpoints: tuple[str, ...],
    select_all: bool,
    models_dir: Path | None,
    smiles_column: str | None,
    jobs: int | None,
    chunksize: int,
    batch_size: int,
    memo_size: int,
    applicability_domain: bool,
    ad_score: bool,
    interval: bool,
    column_prefix: str,
    on_collision: str,
    on_ragged: str,
    strict: bool,
    max_failures: int | None,
    sep: str | None,
    encoding: str,
    report_path: Path | None,
    as_json: bool,
) -> None:
    """Predict ADMET properties for every row of a CSV.

    Columns other than SMILES are written back byte for byte: quoting, line
    endings, encoding and column order are all preserved, and the prediction
    columns are appended at the end.

    A molecule that cannot be parsed yields empty prediction cells and a reason in
    the admetus_error column. That is not a failure of the run, so the exit code
    stays 0 unless --strict is given or every row failed.
    """
    setup_logging(ctx)

    def body() -> None:
        directory = Path(models_dir or default_models_dir())
        present = set(available(directory))
        if not present:
            raise ModelError(
                f"no model bundles in {directory}. Train one with "
                "'admetus train --endpoint dili', or point --models-dir at a directory "
                "that has them."
            )

        selected = resolve_keys(endpoints, select_all)
        missing = [e.key for e in selected if e.key not in present]
        if missing:
            raise ModelError(
                f"no bundle for {missing} in {directory}. "
                f"Available: {sorted(present)}"
            )

        models = [load(str(directory / e.key)) for e in selected]

        temporary: Path | None = None
        input_path = source
        if inline:
            if source is not None:
                raise UsageError("pass either -i/--input or --smiles, not both")
            fd, name = tempfile.mkstemp(suffix=".csv")
            temporary = Path(name)
            with open(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write("smiles\n")
                for text in inline:
                    escaped = text.replace('"', '""')
                    handle.write(f'"{escaped}"\n')
            input_path = temporary

        # `--json` with no -o has to hold every row to put it in the document, so the
        # cap belongs to the run rather than to the reporting afterwards. Checked at
        # the end, a 10,001-row input finished every prediction, wrote a report saying
        # ok with 10,001 rows, threw the predictions away and exited 2: work paid for,
        # nothing delivered, and a report that disagreed with the outcome.
        inline_json = as_json and destination is None
        options = runner.Options(
            smiles_column=smiles_column,
            jobs=jobs if jobs is not None else default_jobs(),
            chunk_size=chunksize,
            batch_size=batch_size,
            memo_size=memo_size,
            applicability_domain=applicability_domain,
            interval=interval,
            ad_score=ad_score,
            column_prefix=column_prefix,
            on_collision=on_collision,
            on_ragged=on_ragged,
            strict=strict,
            max_failures=max_failures,
            sep=sep,
            encoding=encoding,
            max_rows=MAX_JSON_ROWS if inline_json else None,
        )

        # `--json` has to produce one JSON document. It used to write the CSV to
        # standard output and then the report after it, so json.loads(stdout) failed
        # on the first character and the report alone carried no predictions. With
        # no -o, the CSV is written to a temporary file and its rows go into the
        # document; with -o, the file gets the CSV and standard output the report.
        out_path = destination
        json_csv: Path | None = None
        if inline_json:
            fd, name = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            json_csv = Path(name)
            out_path = json_csv

        report = None
        try:
            report = runner.run(input_path, out_path, models, options)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if report is not None:
                write_report(
                    report_path,
                    _report_payload(report, models, input_path, out_path if not inline_json else None),
                )

        payload = _report_payload(report, models, input_path, out_path if not inline_json else None)
        if inline_json:
            try:
                payload["predictions"] = _rows_from_csv(json_csv, report, options)
            finally:
                json_csv.unlink(missing_ok=True)
        if as_json:
            emit(payload)
        elif out_path is not None:
            click.echo(
                f"{report.n_rows} row(s), {report.n_featurized} distinct molecule(s), "
                f"{report.n_failed} failed -> {out_path}",
                err=True,
            )
        # Exit 4 covers both documented cases: --strict with any failure, and every
        # row failing. The report has already been written either way.
        if report.all_failed or (strict and report.n_failed):
            raise SystemExit(EXIT_PARTIAL_FAILURE)

    run_guarded(body, as_json=as_json)


def _coerce(value: str):
    """Type one generated cell for JSON: empty is null, the flag is a boolean.

    Only the columns this run added are typed. A passthrough column stays the exact
    string it was, because reformatting someone's data is the one thing the CSV path
    promises not to do.
    """
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _rows_from_csv(path: Path, report, options: runner.Options) -> list[dict]:
    """Read the written CSV back as JSON rows.

    A JSON object is keyed by name, so a duplicate column name cannot survive it: the
    second value overwrote the first, which turned the header ``smiles,note,note``
    into a row carrying only the last ``note``. Losing input data to make a
    convenience shape is worse than refusing the shape, and the CSV path keeps every
    column, so that is where such a file is sent.

    Raises:
        UsageError: The header repeats a column name.
    """
    from admetus.predict import csvio

    dialect = csvio.sniff(path, encoding=options.encoding, sep=options.sep)
    records = csvio.read_records(path, dialect)
    header = next(records).fields
    seen: dict[str, int] = {}
    for name in header:
        seen[name] = seen.get(name, 0) + 1
    repeated = sorted(name for name, count in seen.items() if count > 1)
    if repeated:
        raise UsageError(
            f"the input repeats the column name(s) {', '.join(repr(r) for r in repeated)}, which "
            "cannot be represented as JSON objects without dropping a value. Pass -o/--output to "
            "write the CSV, which keeps every column, and --json then prints the report alone."
        )
    added = set(report.added_columns)
    rows: list[dict] = []
    for record in records:
        rows.append({
            name: (_coerce(value) if name in added else value)
            for name, value in zip(header, record.fields)
        })
    return rows


def _report_payload(report, models, source, destination) -> dict:
    """Assemble the report document."""
    return {
        "ok": not report.all_failed,
        "input": str(source) if source else None,
        "output": str(destination) if destination else "-",
        "endpoints": [m.key for m in models],
        "added_columns": report.added_columns,
        "molecules": {
            "n_rows": report.n_rows,
            "n_unique_raw": report.n_unique_raw,
            "n_unique_canonical": report.n_unique_canonical,
            "n_featurized": report.n_featurized,
            "n_failed": report.n_failed,
            "n_standardization_changed": report.n_standardization_changed,
            "n_memo_hits": report.n_memo_hits,
        },
        "per_endpoint": report.endpoints,
        "failure_reasons": report.failure_reasons,
        "failures": report.failures,
        "failures_truncated": report.n_failed > len(report.failures),
        "all_failed": report.all_failed,
        "timings_s": report.timings,
        "throughput_rows_per_s": (
            round(report.n_rows / report.timings["total"], 1)
            if report.timings.get("total") else None
        ),
    }


__all__ = ["predict_command"]
