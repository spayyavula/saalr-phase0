"""Options acquisition: SPY weekly contracts near ATM at 1-minute granularity.

Per ``pre-registration.md`` §3:

- Underlying: SPY weekly (Friday expirations).
- Sample strikes: ATM ±2 — locked in ``SPEC.options_strike_window``.
- Granularity: 1-minute.
- Quote basis: NBBO mid — locked in ``SPEC.options_quote_basis``.
- Spread filter: drop (strike, expiry, time) where the bid-ask spread
  exceeds ``SPEC.options_max_spread_pct_of_mid`` (10 % of mid).

The ingest strike window and the mid-quote source were resolved
2026-05-27 in
``decisions/2026-05-27_q1-strike-window-and-q2-mid-quote.md``:

- Ingest ``ATM_open ± INGEST_STRIKE_HALF_WIDTH`` strikes per daily
  pull; sample construction post-filters to the locked ATM-event ±2
  window and asserts coverage.
- Mid quotes come from windowed NBBO ticks via
  ``fetch_option_mid_quote_at`` with "as-of" semantics at each sample
  time.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Optional

from polygon import RESTClient

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


INGEST_STRIKE_HALF_WIDTH: float = 25.0
MID_QUOTE_WINDOW_SECONDS: int = 30


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
    ``[spot_min, spot_max]``. Use ``list_contracts_for_ingest`` for the
    daily-pull case; this lower-level helper is exposed for explicit
    coverage-check re-pulls."""
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


def list_contracts_for_ingest(
    expiry: date,
    daily_open_spot: float,
    half_width: float = INGEST_STRIKE_HALF_WIDTH,
) -> list[dict]:
    """Daily-ingest contract listing: ``ATM_open ± half_width`` calls + puts.

    Sample construction post-filters this set to
    ``SPEC.options_strike_window`` strikes around the event-time ATM and
    asserts coverage; failures land in ``coverage_failures.parquet``."""
    return list_contracts_for_expiry(
        expiry=expiry,
        spot_min=daily_open_spot - half_width,
        spot_max=daily_open_spot + half_width,
    )


def fetch_option_mid_quote_at(
    occ_ticker: str,
    sample_time: datetime,
    window_seconds: int = MID_QUOTE_WINDOW_SECONDS,
) -> Optional[dict]:
    """Return the as-of NBBO mid-quote for ``occ_ticker`` at ``sample_time``.

    Pulls NBBO ticks in ``[sample - window, sample + window]`` sorted
    descending and returns the first tick with
    ``sip_timestamp <= sample_time``. ``None`` if no qualifying tick is in
    the window or if bid/ask is missing/invalid.

    The 30-second window is conservative for an ATM SPY weekly (sub-second
    NBBO update rates during RTH) and protects against feed gaps without
    bloating the query."""
    if sample_time.tzinfo is None:
        sample_time = sample_time.replace(tzinfo=timezone.utc)
    sample_ns = int(sample_time.timestamp() * 1_000_000_000)
    lo_ns = sample_ns - window_seconds * 1_000_000_000
    hi_ns = sample_ns + window_seconds * 1_000_000_000

    quotes = _client().list_quotes(
        ticker=occ_ticker,
        timestamp_gte=lo_ns,
        timestamp_lte=hi_ns,
        limit=500,
        sort="timestamp",
        order="desc",
    )

    chosen: tuple[int, float, float] | None = None
    for q in quotes:
        ts = getattr(q, "sip_timestamp", None) or getattr(q, "participant_timestamp", None)
        if ts is None or ts > sample_ns:
            continue
        bid = getattr(q, "bid_price", None)
        ask = getattr(q, "ask_price", None)
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            continue
        chosen = (int(ts), float(bid), float(ask))
        break

    if chosen is None:
        return None

    ts_ns, bid, ask = chosen
    mid = 0.5 * (bid + ask)
    spread = (ask - bid) / mid if mid > 0 else float("inf")
    return {
        "ticker": occ_ticker,
        "sample_time": sample_time,
        "tick_time": datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct_of_mid": spread,
        "passes_spread_filter": spread <= SPEC.options_max_spread_pct_of_mid,
    }


def fetch_option_aggregates(occ_ticker: str, start: date, end: date):
    """Fetch 1-minute trade-OHLC aggregates for a single OCC-formatted option
    ticker (e.g. ``O:SPY260605C00580000``).

    NOTE: trade-OHLC only — NOT the spec-locked mid-quote. Kept as a
    sanity helper for spot-checking liquidity (``volume``,
    ``trade_count``). Sample construction must use
    ``fetch_option_mid_quote_at`` per the Q2 decision.
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
