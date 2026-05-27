"""Storage and partitioning helpers for the Phase 0 acquisition layer.

Two responsibilities, one each:

1. ``split_for_date`` — given a date, return ``"train" | "validation" | "holdout"``
   per the locked windows in ``locked_spec.SPEC``. The function fails closed:
   dates outside the locked windows raise. Sample-construction code should
   never silently fall back to a default split.

2. ``write_partitioned_parquet`` — split a frame by calendar month and write
   one parquet per (split, source, YYYY-MM). Updates ``data/MANIFEST.json``
   with row counts and SHA-256 of each file so the Week-3 holdout-hash
   commitment in ``pre-registration.md`` §4 can be verified later.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)

Split = Literal["train", "validation", "holdout"]


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def split_for_date(d: date | str) -> Split:
    """Return the locked split that contains ``d``. Raises if ``d`` is outside
    every locked window — this is intentional, not an oversight."""
    if isinstance(d, str):
        d = _parse(d)
    train = (_parse(SPEC.train_start), _parse(SPEC.train_end))
    validation = (_parse(SPEC.validation_start), _parse(SPEC.validation_end))
    holdout = (_parse(SPEC.holdout_start), _parse(SPEC.holdout_end))
    if train[0] <= d <= train[1]:
        return "train"
    if validation[0] <= d <= validation[1]:
        return "validation"
    if holdout[0] <= d <= holdout[1]:
        return "holdout"
    raise ValueError(
        f"date {d} is outside all locked splits "
        f"(train {train}, validation {validation}, holdout {holdout})"
    )


@dataclass(frozen=True)
class WriteResult:
    path: Path
    row_count: int
    sha256: str


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_partitioned_parquet(
    df,  # pandas.DataFrame; not type-annotated to avoid forcing import at module load
    data_root: Path,
    source: str,
    timestamp_col: str = "timestamp",
) -> list[WriteResult]:
    """Write ``df`` partitioned by (split, YYYY-MM) into
    ``data_root/{split}/{source}/YYYY-MM.parquet`` and update
    ``data_root/MANIFEST.json``. Returns one ``WriteResult`` per file written.

    ``df`` must have a column ``timestamp_col`` parseable as a date (or
    datetime-like). The split is determined by the *date* portion of the
    timestamp.
    """
    import pandas as pd  # local import keeps module load light

    if df.empty:
        logger.info("write_partitioned_parquet: empty frame, nothing to write")
        return []

    ts = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.assign(_date=ts.dt.date, _year_month=ts.dt.strftime("%Y-%m"))

    results: list[WriteResult] = []
    for (year_month,), group in df.groupby(["_year_month"]):
        # All rows in a month share a split (each split aligns on month
        # boundaries in SPEC — verified by tests/test_storage_split.py).
        sample_date = group["_date"].iloc[0]
        split = split_for_date(sample_date)
        out_dir = data_root / split / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year_month}.parquet"
        group_clean = group.drop(columns=["_date", "_year_month"])
        group_clean.to_parquet(out_path, index=False)
        sha = _sha256_of(out_path)
        results.append(WriteResult(out_path, len(group_clean), sha))
        logger.info(
            "wrote %s rows=%d sha256=%s",
            out_path.relative_to(data_root),
            len(group_clean),
            sha[:12],
        )

    _update_manifest(data_root, results)
    return results


def _update_manifest(data_root: Path, results: list[WriteResult]) -> None:
    manifest_path = data_root / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"files": {}}
    for r in results:
        key = str(r.path.relative_to(data_root)).replace("\\", "/")
        manifest["files"][key] = {
            "rows": r.row_count,
            "sha256": r.sha256,
            "written_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
