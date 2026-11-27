"""Typed access to the endpoint registry, and the rules that follow from dataset size.

Several training choices are functions of how much data an endpoint actually has,
so they are computed here rather than duplicated 29 times in the YAML:

* which feature set to use, because 89,852 columns against a few hundred rows is
  not a useful feature space;
* how aggressively to prune rare and ubiquitous columns, since a fixed absolute
  count means "present in 2.8% of molecules" at n=278 and "present in 0.06%" at
  n=13,445;
* how to split a fixed search budget between Optuna trials and repeated
  cross-validation, because past a point extra trials buy only winner's curse;
* whether to warn callers that a test set is too small to read confidently.

Anything stored in the YAML overrides the computed value.

The feature space is **not** one of these. Every endpoint gets the same one, as the
reference implementation does; the rule that used to pick by dataset size, and the
three measured overrides on top of it, were not asked for and are gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import yaml

from admetus.chem.spec import DEFAULT_STANDARDIZE, FeaturizerSpec, get_spec
from admetus.errors import UsageError

Task = Literal["classification", "regression"]
Metric = Literal["auroc", "auprc", "mae", "spearman"]

#: Whether a larger value of each metric is better.
HIGHER_IS_BETTER: dict[str, bool] = {"auroc": True, "auprc": True, "spearman": True, "mae": False}

#: Below this many labelled rows, a held-out test set is too small to read as a
#: point estimate, so callers are told so explicitly.
LOW_CONFIDENCE_ROWS = 600


@dataclass(frozen=True)
class Endpoint:
    """One public dataset and everything needed to train and serve a model for it."""

    key: str
    tdc_name: str
    dataverse_id: int
    task: Task
    metric: Metric
    unit: str
    in_official_group: bool
    description: str
    expected_rows: int | None = None
    sha256: str | None = None
    label_name: str | None = None
    transform: str | None = None
    output_range: tuple[float | None, float | None] | None = None
    species_filter: str | None = None
    standardize: tuple[str, ...] = DEFAULT_STANDARDIZE
    correlated_with: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def higher_is_better(self) -> bool:
        """Whether a larger metric value is better."""
        return HIGHER_IS_BETTER[self.metric]

    @property
    def download_url(self) -> str:
        """Where the published dataset file lives."""
        return f"https://dataverse.harvard.edu/api/access/datafile/{self.dataverse_id}"

    def featurizer_spec(self, n_train: int | None = None) -> FeaturizerSpec:
        """Return the featurizer spec for this endpoint, with its standardization pipeline.

        Every endpoint gets the same feature space; ``n_train`` is accepted and
        ignored so callers that still pass it keep working. The size rule that used
        to read it, and the three per-endpoint overrides on top of it, were not asked
        for and are gone.
        """
        _ = n_train
        base = get_spec()
        if tuple(self.standardize) == tuple(base.standardize):
            return base
        # Datasets whose label describes the salt use a narrower pipeline.
        return FeaturizerSpec(
            feature_set=base.feature_set,
            morgan_radius=base.morgan_radius,
            morgan_bits=base.morgan_bits,
            pharm_max_points=base.pharm_max_points,
            pharm_min_points=base.pharm_min_points,
            pharm_bins=base.pharm_bins,
            pharm_fdef=base.pharm_fdef,
            triangle_prune_bins=base.triangle_prune_bins,
            standardize=tuple(self.standardize),
        )

    def low_confidence(self, n_rows: int | None = None) -> bool:
        """Whether the dataset is too small for its test metric to be read as a point estimate."""
        n = n_rows if n_rows is not None else self.expected_rows
        return n is not None and n < LOW_CONFIDENCE_ROWS

    def to_json(self, n_rows: int | None = None) -> dict[str, Any]:
        """Return the machine-readable description ``admetus endpoints --json`` serves."""
        n = n_rows if n_rows is not None else self.expected_rows
        return {
            "key": self.key,
            "dataset": self.tdc_name,
            "label_name": self.label_name,
            "task": self.task,
            "metric": self.metric,
            "direction": "higher_is_better" if self.higher_is_better else "lower_is_better",
            "unit": self.unit,
            "transform": self.transform,
            "output_range": list(self.output_range) if self.output_range else None,
            "in_official_group": self.in_official_group,
            "n_rows": n,
            "low_confidence": self.low_confidence(n),
            "correlated_with": list(self.correlated_with),
            "description": self.description,
            "notes": self.notes,
        }


def _coerce(raw: dict[str, Any], defaults: dict[str, Any]) -> Endpoint:
    """Build an :class:`Endpoint` from one YAML mapping."""
    known = {
        "key", "tdc_name", "dataverse_id", "sha256", "task", "metric", "unit",
        "in_official_group", "description", "expected_rows", "label_name", "transform",
        "output_range", "species_filter", "standardize", "correlated_with", "notes",
    }
    unknown = set(raw) - known
    if unknown:
        raise UsageError(f"endpoint {raw.get('key')!r} has unknown fields: {sorted(unknown)}")
    if raw["task"] not in ("classification", "regression"):
        raise UsageError(f"endpoint {raw['key']!r} has unknown task {raw['task']!r}")
    if raw["metric"] not in HIGHER_IS_BETTER:
        raise UsageError(f"endpoint {raw['key']!r} has unknown metric {raw['metric']!r}")
    if raw.get("transform") not in (None, "log10", "log10_safe"):
        raise UsageError(f"endpoint {raw['key']!r} has unknown transform {raw['transform']!r}")
    if raw["task"] == "classification" and raw.get("transform"):
        raise UsageError(f"endpoint {raw['key']!r} is classification but declares a transform")

    output_range = raw.get("output_range")
    return Endpoint(
        key=raw["key"],
        tdc_name=raw["tdc_name"],
        dataverse_id=int(raw["dataverse_id"]),
        task=raw["task"],
        metric=raw["metric"],
        unit=raw["unit"],
        in_official_group=bool(raw["in_official_group"]),
        description=raw["description"],
        expected_rows=raw.get("expected_rows"),
        sha256=raw.get("sha256"),
        label_name=raw.get("label_name"),
        transform=raw.get("transform"),
        output_range=tuple(output_range) if output_range else None,
        species_filter=raw.get("species_filter"),
        standardize=tuple(raw.get("standardize") or defaults.get("standardize") or DEFAULT_STANDARDIZE),
        correlated_with=tuple(raw.get("correlated_with") or ()),
        notes=raw.get("notes"),
    )


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Endpoint]:
    """Load and validate the packaged endpoint registry, keyed by endpoint key."""
    text = resources.files("admetus.data").joinpath("endpoints.yaml").read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    defaults = raw.get("defaults") or {}

    registry: dict[str, Endpoint] = {}
    for entry in raw["endpoints"]:
        endpoint = _coerce(entry, defaults)
        if endpoint.key in registry:
            raise UsageError(f"duplicate endpoint key {endpoint.key!r} in the registry")
        registry[endpoint.key] = endpoint

    seen_ids: dict[int, str] = {}
    for endpoint in registry.values():
        if endpoint.dataverse_id in seen_ids:
            raise UsageError(
                f"endpoints {seen_ids[endpoint.dataverse_id]!r} and {endpoint.key!r} "
                f"share dataverse id {endpoint.dataverse_id}"
            )
        seen_ids[endpoint.dataverse_id] = endpoint.key

    for endpoint in registry.values():
        for other in endpoint.correlated_with:
            if other not in registry:
                raise UsageError(
                    f"endpoint {endpoint.key!r} says it correlates with unknown endpoint {other!r}"
                )
    return registry


def get_endpoint(key: str) -> Endpoint:
    """Look up one endpoint.

    Raises:
        UsageError: No such endpoint, with the available keys listed.
    """
    registry = load_registry()
    try:
        return registry[key]
    except KeyError:
        raise UsageError(
            f"unknown endpoint {key!r}. Run 'admetus endpoints' to list the {len(registry)} available."
        ) from None


def all_endpoints() -> list[Endpoint]:
    """Every endpoint, in registry order."""
    return list(load_registry().values())


def official_endpoints() -> list[Endpoint]:
    """Only the endpoints of the official ADMET benchmark group."""
    return [e for e in all_endpoints() if e.in_official_group]


def resolve_keys(keys: tuple[str, ...] | list[str], select_all: bool = False) -> list[Endpoint]:
    """Turn CLI endpoint selectors into endpoints.

    Raises:
        UsageError: Nothing was selected, or a key is unknown.
    """
    if select_all:
        return all_endpoints()
    if not keys:
        raise UsageError("no endpoint selected; pass --endpoint KEY (repeatable) or --all")
    return [get_endpoint(k) for k in keys]


__all__ = [
    "HIGHER_IS_BETTER",
    "LOW_CONFIDENCE_ROWS",
    "Endpoint",
    "all_endpoints",
    "get_endpoint",
    "load_registry",
    "official_endpoints",
    "resolve_keys",
]
