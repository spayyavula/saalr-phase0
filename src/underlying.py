"""Underlying acquisition: SPY 1-minute aggregates.

Per ``pre-registration.md`` §3:

- Symbol: ``SPEC.underlying_symbol`` (SPY).
- Granularity: ``SPEC.underlying_granularity_minutes`` (1).
- Use: spot price for Black-Scholes IV inversion during sample construction.
"""
from __future__ import annotations

import logging
import os
from datetime import date

from polygon import RESTClient

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


def _client() -> RESTClient:
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY (or POLYGON_API_KEY) is not set; see .env.example"
        )
    return RESTClient(api_key=key)


def fetch_underlying(start: date, end: date, symbol: str | None = None):
    """Fetch 1-minute aggregates for ``symbol`` (default: locked SPY) over
    [start, end). Returns a pandas DataFrame with:

    - ``timestamp`` — UTC bar-open time
    - ``open``, ``high``, ``low``, ``close``, ``vwap`` — float
    - ``volume`` — int
    - ``trade_count`` — int (named ``n`` in the raw response)
    """
    import pandas as pd

    symbol = symbol or SPEC.underlying_symbol
    granularity = SPEC.underlying_granularity_minutes
    logger.info(
        "underlying pull: symbol=%s [%s, %s) granularity=%d-min",
        symbol,
        start.isoformat(),
        end.isoformat(),
        granularity,
    )
    rows: list[dict] = []
    for bar in _client().list_aggs(
        ticker=symbol,
        multiplier=granularity,
        timespan="minute",
        from_=start.isoformat(),
        to=end.isoformat(),
        adjusted=True,
        sort="asc",
        limit=50000,
    ):
        bar_dict = bar if isinstance(bar, dict) else bar.__dict__
        rows.append(
            {
                "timestamp": pd.to_datetime(bar_dict["timestamp"], unit="ms", utc=True),
                "open": float(bar_dict["open"]),
                "high": float(bar_dict["high"]),
                "low": float(bar_dict["low"]),
                "close": float(bar_dict["close"]),
                "vwap": float(bar_dict.get("vwap", 0.0)),
                "volume": int(bar_dict.get("volume", 0)),
                "trade_count": int(bar_dict.get("transactions", 0)),
            }
        )
    return pd.DataFrame.from_records(rows)
