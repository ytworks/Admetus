"""Load a bundle's boosters in a child process, and say whether they serve.

Run as ``python -m admetus.bundle._loadcheck <bundle dir>``. Prints one JSON object
and exits 0 when every booster loads and matches the bundle's column count.

**Why a child process.** LightGBM's text parser does not always raise where Python
can catch it. On a model truncated mid-tree it raises from C++ through
``std::runtime_error`` and the process aborts: ``libc++abi: terminating due to
uncaught exception``. A verifier that dies while checking is worse than no verifier,
so the load happens somewhere whose death is an answer rather than an outage.

**Why a load rather than a structural check.** The structural check that preceded
this one read the end of the file for the marker a complete model carries. It caught
a file cut in half and accepted a real model missing its last two bytes, because the
marker was still there and the damage was in the trailing metadata LightGBM reads
after it. Only the parser knows whether the parser can read a file.

**Why the column count too.** An intact booster from the same endpoint under a
different feature preset loads perfectly and then predicts on the wrong width. That
is not a corrupt file, so no amount of reading it will say anything is wrong; it has
to be compared with the basis the bundle ships.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check(directory: Path) -> dict[str, object]:
    """Load every booster and compare its width with the bundle's record."""
    from admetus._compat import load_lightgbm
    from admetus.bundle.save import BOOSTER_PATTERN as PATTERN

    lgb = load_lightgbm()
    meta = json.loads((Path(directory) / "bundle.json").read_text(encoding="utf-8"))
    expected_columns = int(meta["featurizer"]["n_features"])
    n_boosters = int(meta["model"]["n_boosters"])

    problems: list[str] = []
    for index in range(n_boosters):
        path = Path(directory) / PATTERN.format(index=index)
        try:
            booster = lgb.Booster(model_file=str(path))
        except Exception as exc:  # noqa: BLE001 - any failure means it cannot serve
            # LightGBM reads the trailing metadata as JSON, so a file short by a
            # couple of bytes surfaces as a JSON error rather than a model error.
            # Said plainly here, because "Expecting value: line 1 column 1" tells an
            # operator nothing about a model file.
            problems.append(
                f"{path.name} could not be loaded by LightGBM ({type(exc).__name__}: {exc}). "
                "The file is damaged or incomplete."
            )
            continue
        width = int(booster.num_feature())
        if width != expected_columns:
            problems.append(
                f"{path.name} expects {width} feature(s) but the bundle's basis has "
                f"{expected_columns}; it was trained against a different feature space"
            )
    return {"ok": not problems, "problems": problems}


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "problems": ["usage: _loadcheck <bundle dir>"]}))
        return 2
    try:
        result = check(Path(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001 - reported, not raised, to the parent
        result = {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
