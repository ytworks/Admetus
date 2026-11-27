"""Running prediction over a CSV.

The shape of the work, and why:

* **Featurize each distinct molecule once and serve every endpoint from it.** The
  feature vector does not depend on the endpoint, so computing it per endpoint
  would repeat the expensive step 29 times.
* **Deduplicate within a chunk, plus a bounded memo across chunks.** A dictionary
  covering a ten-million-row file would need gigabytes; a chunk-local map plus a
  capped recent-molecules cache is bounded and catches the repeats that matter.
* **Stream in chunks and append as you go**, so memory follows the chunk size
  rather than the file size.
* **Write to a temporary file and rename at the end.** A truncated CSV that looks
  like a result is worse than no file.
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from admetus.bundle.load import LoadedModel
from admetus.chem.featurizer import build_csr
from admetus.chem.parallel import DEFAULT_BATCH_SIZE, Featurizer, resolve_jobs
from admetus.chem.similarity import in_domain, max_similarity, pack_bits
from admetus.errors import InputDataError
from admetus.logging import get_logger, progress_disabled
from admetus.predict import csvio
from admetus.train import prune, transforms

log = get_logger("predict")

#: Rows read, deduplicated and written per pass.
DEFAULT_CHUNK_SIZE = 50_000

#: Distinct molecules remembered across chunks.
DEFAULT_MEMO_SIZE = 200_000

#: What an unpredictable row is written as.
EMPTY = ""


@dataclass
class Options:
    """How a prediction run should behave."""

    smiles_column: str | None = None
    jobs: int | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    memo_size: int = DEFAULT_MEMO_SIZE
    applicability_domain: bool = True
    interval: bool = False
    ad_score: bool = False
    column_prefix: str = ""
    on_collision: str = "suffix"
    on_ragged: str = "error"
    strict: bool = False
    max_failures: int | None = None
    sep: str | None = None
    encoding: str = "utf-8"
    precision: int = 6


@dataclass
class Report:
    """What a run did, for ``--report``."""

    n_rows: int = 0
    n_unique_raw: int = 0
    n_unique_canonical: int = 0
    n_featurized: int = 0
    n_failed: int = 0
    n_standardization_changed: int = 0
    n_memo_hits: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    endpoints: list[dict[str, Any]] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def record_failure(self, row: int, smiles: str, reason: str, cap: int = 100) -> None:
        """Count a failure, keeping a capped sample of the detail."""
        self.n_failed += 1
        key = reason.split(":")[0].strip()[:80]
        self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
        if len(self.failures) < cap:
            self.failures.append({"row": row, "smiles": smiles, "reason": reason})


def column_names(model: LoadedModel, options: Options) -> list[str]:
    """The columns one endpoint contributes, in order.

    The spread column is named for the space it lives in. A standard deviation of
    log-scale predictions is not a standard deviation in L/kg or hours, and naming
    it ``_ens_sd`` on a log endpoint would invite exactly that misreading.
    """
    prefix = options.column_prefix
    base = f"{prefix}{model.key}"
    names = [base]
    names.append(f"{base}_sd_log10" if model.transform.is_log else f"{base}_ens_sd")
    if options.applicability_domain and model.reference is not None:
        names.append(f"{base}_in_domain")
        if options.ad_score:
            names.append(f"{base}_ad_score")
    if options.interval and model.interval is not None:
        names.extend([f"{base}_lo", f"{base}_hi"])
    return names


def _format(value: float, precision: int) -> str:
    """Render a number for the CSV, or an empty field when it is not finite."""
    if value is None or not np.isfinite(value):
        return EMPTY
    return f"{value:.{precision}g}"


@dataclass
class EndpointOutput:
    """One endpoint's columns for a chunk of distinct molecules."""

    values: list[list[str]]
    n_in_domain: int
    n_predicted: int


def predict_block(
    model: LoadedModel,
    matrix,
    packed: np.ndarray | None,
    popcount: np.ndarray | None,
    options: Options,
) -> EndpointOutput:
    """Predict one endpoint over a block of featurized molecules."""
    narrowed = prune.apply_basis(matrix, model.basis)
    stacked = np.column_stack([b.predict(narrowed) for b in model.boosters])
    mean_fitted = stacked.mean(axis=1)
    spread_fitted = stacked.std(axis=1)

    if model.is_classification:
        value = model.calibrator.apply(mean_fitted) if model.calibrator else mean_fitted
        spread = spread_fitted
    else:
        value = transforms.clamp(model.transform.inverse(mean_fitted), model.output_range)
        # Reported in the space it was computed in, which the column name states.
        spread = spread_fitted

    low = high = None
    if options.interval and model.interval is not None:
        lo_fitted, hi_fitted = model.interval.bounds(mean_fitted, spread_fitted)
        low = transforms.clamp(model.transform.inverse(lo_fitted), model.output_range)
        high = transforms.clamp(model.transform.inverse(hi_fitted), model.output_range)

    domain = score = None
    n_in_domain = 0
    if options.applicability_domain and model.reference is not None and packed is not None:
        if options.ad_score:
            score = max_similarity(packed, popcount, model.reference)
            domain = score >= model.reference.threshold
        else:
            domain = in_domain(packed, popcount, model.reference)
        n_in_domain = int(np.count_nonzero(domain))

    rows: list[list[str]] = []
    precision = options.precision
    for i in range(matrix.shape[0]):
        cells = [_format(value[i], precision), _format(spread[i], precision)]
        if domain is not None:
            cells.append("true" if domain[i] else "false")
            if score is not None:
                cells.append(_format(score[i], precision))
        if low is not None:
            cells.extend([_format(low[i], precision), _format(high[i], precision)])
        rows.append(cells)

    return EndpointOutput(values=rows, n_in_domain=n_in_domain, n_predicted=matrix.shape[0])


def run(
    source: Path | None,
    destination: Path | None,
    models: list[LoadedModel],
    options: Options,
) -> Report:
    """Predict every endpoint over a CSV and append the columns.

    Args:
        source: Input path, or ``None`` to read standard input.
        destination: Output path, or ``None`` to write standard output.
        models: Loaded models, in output column order.
        options: Run options.

    Returns:
        The :class:`Report`.

    Raises:
        InputDataError: The input is unusable, or every row failed.
    """
    if not models:
        raise InputDataError("no models selected")
    started = time.time()
    report = Report()

    specs = {m.spec.spec_sha256 for m in models}
    if len(specs) != 1:
        groups = {}
        for m in models:
            groups.setdefault(m.spec.feature_set, []).append(m.key)
        raise InputDataError(
            "the selected endpoints were trained on different feature spaces, so they cannot "
            f"share one featurization pass: {groups}. Predict them in separate runs."
        )
    spec = models[0].spec

    if source is None:
        raise InputDataError("reading from standard input is not supported yet; pass -i PATH")

    dialect = csvio.sniff(source, encoding=options.encoding, sep=options.sep)
    records = csvio.read_records(source, dialect)
    try:
        header = next(records)
    except StopIteration:
        raise InputDataError(f"{source} has no header row") from None

    smiles_at = csvio.find_column(header.fields, options.smiles_column)
    wanted: list[str] = []
    per_model: list[list[str]] = []
    for model in models:
        names = column_names(model, options)
        per_model.append(names)
        wanted.extend(names)
    wanted.append(f"{options.column_prefix}admetus_error")

    resolved = csvio.unique_names(header.fields, wanted, options.on_collision)
    report.added_columns = resolved
    n_header = len(header.fields)
    ragged = csvio.RaggedPolicy(options.on_ragged)

    jobs = resolve_jobs(options.jobs, options.chunk_size)
    memo: OrderedDict[str, tuple[list[str], str | None]] = OrderedDict()
    endpoint_stats = [{"key": m.key, "n_predicted": 0, "n_in_domain": 0} for m in models]
    blank = [EMPTY] * (len(resolved) - 1)

    t_featurize = 0.0
    t_predict = 0.0

    with (
        csvio.Appender(destination, dialect, stdout=sys.stdout if destination is None else None) as out,
        Featurizer(spec, jobs=jobs, batch_size=options.batch_size) as featurizer,
    ):
        out.write(header, resolved)

        for chunk in _chunks(records, options.chunk_size):
            for record in chunk:
                ragged.check(record, n_header, source)
            report.n_rows += len(chunk)

            raw = [
                record.fields[smiles_at].strip() if smiles_at < len(record.fields) else ""
                for record in chunk
            ]
            pending: dict[str, list[int]] = {}
            for position, text in enumerate(raw):
                if text in memo:
                    report.n_memo_hits += 1
                    memo.move_to_end(text)
                    continue
                pending.setdefault(text, []).append(position)

            report.n_unique_raw += len(pending)
            distinct = list(pending)

            t0 = time.time()
            featurized = featurizer.run(distinct) if distinct else None
            t_featurize += time.time() - t0

            rows: list[np.ndarray] = []
            keys: list[str] = []
            failed: dict[str, str] = {}
            if featurized is not None:
                for offset, text in enumerate(distinct):
                    indices = featurized.indices[offset]
                    if indices is None:
                        failed[text] = featurized.reasons[offset] or "unknown failure"
                        continue
                    rows.append(indices)
                    keys.append(text)
                    if featurized.changed[offset]:
                        report.n_standardization_changed += 1

            report.n_unique_canonical += len(keys)
            report.n_featurized += len(keys)

            if rows:
                t0 = time.time()
                matrix = build_csr(rows, spec.n_features)
                packed = popcount = None
                if options.applicability_domain and any(m.reference is not None for m in models):
                    packed, popcount = pack_bits(
                        [r[r < spec.morgan_bits] for r in rows], spec.morgan_bits
                    )
                outputs = []
                for index, model in enumerate(models):
                    produced = predict_block(model, matrix, packed, popcount, options)
                    outputs.append(produced)
                    endpoint_stats[index]["n_predicted"] += produced.n_predicted
                    endpoint_stats[index]["n_in_domain"] += produced.n_in_domain
                t_predict += time.time() - t0

                for offset, text in enumerate(keys):
                    cells: list[str] = []
                    for produced in outputs:
                        cells.extend(produced.values[offset])
                    memo[text] = (cells, None)
                    memo.move_to_end(text)

            for text, reason in failed.items():
                memo[text] = (list(blank), reason)
                memo.move_to_end(text)

            while len(memo) > options.memo_size:
                memo.popitem(last=False)

            for position, record in enumerate(chunk):
                text = raw[position]
                entry = memo.get(text)
                if entry is None:
                    cells, reason = list(blank), "molecule was evicted from the cache"
                else:
                    cells, reason = entry
                if reason:
                    report.record_failure(record.number, text, reason)
                    if options.max_failures is not None and report.n_failed > options.max_failures:
                        raise InputDataError(
                            f"stopped after {report.n_failed} failed molecule(s), over the "
                            f"--max-failures limit of {options.max_failures}"
                        )
                out.write(record, [*cells, reason or EMPTY])

            if not progress_disabled():
                log.info("processed %d row(s)", report.n_rows)

    report.endpoints = endpoint_stats
    report.timings = {
        "featurize": round(t_featurize, 3),
        "predict": round(t_predict, 3),
        "total": round(time.time() - started, 3),
    }
    if report.n_rows and report.n_failed == report.n_rows:
        raise InputDataError(
            f"every one of the {report.n_rows} row(s) failed. "
            f"Reasons: {report.failure_reasons}"
        )
    return report


def _chunks(records, size: int):
    """Group an iterator of records into lists of at most ``size``."""
    batch: list[csvio.Record] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MEMO_SIZE",
    "EndpointOutput",
    "Options",
    "Report",
    "column_names",
    "predict_block",
    "run",
]
