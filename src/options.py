"""Options acquisition: SPY weekly contracts near ATM at 1-minute granularity.

Per ``pre-registration.md`` §3:

- Underlying: SPY weekly (Friday expirations).
- Strikes: ATM ±2 — locked in ``SPEC.options_strike_window``.
- Granularity: 1-minute, mid-quote basis.
- Spread filter: exclude (strike, expiry) where the 1-minute mid-quote
  bid-ask spread exceeds ``SPEC.options_max_spread_pct_of_mid`` (10 %% of mid).

Two open implementation questions to resolve in Week 2 before bulk ingest
(both will be answered in a decisions/ entry once verified empirically):

Q1. **Strike window at ingest vs sample construction.**
    Spot moves intraday, so the "ATM" strike at minute t differs from the
    "ATM" strike at minute t+30. To guarantee we have data for every
    minute's true ATM ±2 at sample-construction time, we ingest a wider
    window (e.g. ATM±10 by daily-open spot) and post-filter to ATM±2 per
    event. Storing every contract every day is wasteful; storing too few
    will silently drop events. The fix-window-at-ingest tradeoff is
    documented here so the post-filter step in sample construction can
    assert coverage.

Q2. **Mid-quote source.**
    The vendor's minute aggregates are OHLC of *trades*, not quotes. The
    pre-reg locks "mid-quote" specifically (``SPEC.iv_quote_basis ==
    "call_put_mid"``). Three candidate paths:

      a. Use the per-minute *quote* aggregates endpoint (if it exists).
      b. Pull tick-level quotes and bucket to 1-minute mid ourselves.
      c. Use the per-strike snapshot endpoint at the exact event times
         (cleaner but requires re-pulling for any reprocessing).

    None are picked here. ``fetch_option_aggregates`` below pulls trade-OHLC
    today purely to confirm the contract-listing path; mid-quote arrives in
    a follow-on commit once Q2 is decided.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Iterator

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


def iter_weekly_fridays(start: date, end: date) -> Iterator[date]:
    """Yield every Friday between ``start`` and ``end`` inclusive. SPY weekly
    options use Friday expirations. SPY also has Mon/Wed weeklies; those are
    out of scope per the locked pre-registration ("weekly Friday")."""
    days_ahead = (4 - start.weekday()) % 7  # weekday 4 == Friday
    cursor = start + timedelta(days=days_ahead)
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=7)


def list_contracts_for_expiry(
    expiry: date,
    spot_min: float,
    spot_max: float,
    contract_types: tuple[str, ...] = ("call", "put"),
) -> list[dict]:
    """List SPY option contracts expiring on ``expiry`` whose strikes fall in
    [spot_min, spot_max]. The bounds are wide on purpose — see Q1 in the
    module docstring."""
    rows: list[dict] = []
    for kind in contract_types:
        for contract in _client().list_options_contracts(
            underlying_ticker=SPEC.options_symbol,
            expiration_date=expiry.isoformat(),
            contract_type=kind,
            strike_price_gte=spot_min,
            strike_price_lte=spot_max,
            expired=True,
            limit=1000,
        ):
            contract_dict = (
                contract if isinstance(contract, dict) else contract.__dict__
            )
            rows.append(contract_dict)
    return rows


def fetch_option_aggregates(occ_ticker: str, start: date, end: date):
    """Fetch 1-minute trade-OHLC aggregates for a single OCC-formatted option
    ticker (e.g. ``O:SPY260605C00580000``).

    NOTE: trade-OHLC, not mid-quote. See Q2 in the module docstring.
    """
    import pandas as pd

    granularity = SPEC.options_granularity_minutes
    rows: list[dict] = []
    for bar in _client().list_aggs(
        ticker=occ_ticker,
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
                "ticker": occ_ticker,
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
