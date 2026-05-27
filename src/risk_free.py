"""Risk-free rate acquisition: FRED ``DGS3MO`` daily yield.

Source pinned in ``locked_spec.SPEC.risk_free_rate_source`` (``FRED:DGS3MO``).
Public CSV endpoint — no auth required. Used for Black-Scholes IV inversion
during sample construction.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import requests

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _series_id() -> str:
    src = SPEC.risk_free_rate_source
    if not src.startswith("FRED:"):
        raise ValueError(f"unexpected risk-free source {src!r}; expected FRED:<id>")
    return src.removeprefix("FRED:")


def fetch_risk_free(start: date, end: date, timeout_seconds: int = 30):
    """Fetch the locked FRED series between ``start`` and ``end`` inclusive.

    Returns a pandas DataFrame with two columns:

    - ``timestamp`` — UTC midnight of the FRED observation date
    - ``yield_pct`` — annualized yield in percent (FRED's native unit)

    FRED returns ``.`` for missing observations (weekends, holidays); those
    rows are dropped here. Sample-construction code should forward-fill if
    it needs a value at intra-day timestamps.
    """
    import pandas as pd

    series_id = _series_id()
    params = {
        "id": series_id,
        "cosd": start.isoformat(),
        "coed": end.isoformat(),
    }
    logger.info(
        "fetching FRED %s from %s to %s", series_id, start.isoformat(), end.isoformat()
    )
    response = requests.get(FRED_CSV_URL, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    if list(raw.columns) != ["observation_date", series_id]:
        raise ValueError(
            f"unexpected FRED CSV columns for {series_id}: {list(raw.columns)}"
        )
    cleaned = raw[raw[series_id] != "."].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["observation_date"], utc=True)
    cleaned["yield_pct"] = cleaned[series_id].astype(float)
    return cleaned[["timestamp", "yield_pct"]].reset_index(drop=True)


def fetch_risk_free_to_csv(start: date, end: date, out_path: Path) -> Path:
    """Lightweight smoke-test path: write raw FRED CSV to disk without
    requiring pandas. Returns the output path."""
    series_id = _series_id()
    params = {
        "id": series_id,
        "cosd": start.isoformat(),
        "coed": end.isoformat(),
    }
    response = requests.get(FRED_CSV_URL, params=params, timeout=30)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(response.text, encoding="utf-8")
    logger.info("wrote raw FRED CSV to %s (%d bytes)", out_path, len(response.text))
    return out_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 else date(2026, 5, 1)
    end = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else date(2026, 5, 27)
    out = Path("data/_smoke") / f"DGS3MO_{start.isoformat()}_{end.isoformat()}.csv"
    fetch_risk_free_to_csv(start, end, out)
    print(f"wrote {out}")
