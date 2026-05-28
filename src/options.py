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
import random
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Iterator, Optional
from zoneinfo import ZoneInfo

from polygon import RESTClient

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


def _is_transient_quote_error(exc: Exception) -> bool:
    """True for vendor/network failures worth retrying (502/503/504,
    connection resets, timeouts, rate-limit blips). False for programming
    errors, which should surface rather than be masked as a missing quote."""
    import urllib3.exceptions

    if isinstance(exc, (urllib3.exceptions.HTTPError, ConnectionError, TimeoutError, OSError)):
        return True
    try:
        from polygon.exceptions import BadResponse

        if isinstance(exc, BadResponse):
            return True
    except ImportError:
        pass
    # The polygon client wraps urllib3 MaxRetryError; the message reliably
    # carries these markers even when the surfaced type is generic.
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in ("max retries", "502", "503", "504", "timed out",
                       "timeout", "connection", "too many")
    )


INGEST_STRIKE_HALF_WIDTH: float = 25.0
MID_QUOTE_WINDOW_SECONDS: int = 30
EXPIRIES_PER_TRADING_DAY: int = 2
INTER_CALL_SLEEP_SECONDS: float = 0.2
QUOTE_RETRY_ATTEMPTS: int = 3
QUOTE_RETRY_SLEEP_SECONDS: float = 5.0
_ET = ZoneInfo("America/New_York")
_RTH_OPEN_ET = dtime(9, 30)
_RTH_CLOSE_ET = dtime(16, 0)


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
    limit: int = 100,
) -> Optional[dict]:
    """Return the as-of NBBO mid-quote for ``occ_ticker`` at ``sample_time``.

    Pulls NBBO ticks in ``[sample - window, sample]`` sorted descending
    and returns the first tick with valid bid/ask (the most recent quote
    at or before ``sample_time``). ``None`` if no qualifying tick is in
    the window.

    The upper bound is ``sample_time`` exactly (not ``sample + window``):
    we only ever pick ticks at or before the sample, so including the
    future side would waste the ``limit`` budget on ticks we discard —
    and on a liquid contract with hundreds of ticks in the +window side,
    could flood the page and produce a false negative. ``order=desc`` +
    ``timestamp_lte=sample`` means the first returned row is already the
    answer; ``limit`` only needs to be large enough to skip past a few
    crossed/locked quotes at the very top."""
    if sample_time.tzinfo is None:
        sample_time = sample_time.replace(tzinfo=timezone.utc)
    sample_ns = int(sample_time.timestamp() * 1_000_000_000)
    lo_ns = sample_ns - window_seconds * 1_000_000_000

    chosen: tuple[int, float, float] | None = None
    for attempt in range(QUOTE_RETRY_ATTEMPTS):
        try:
            quotes = _client().list_quotes(
                ticker=occ_ticker,
                timestamp_gte=lo_ns,
                timestamp_lte=sample_ns,
                limit=limit,
                sort="timestamp",
                order="desc",
            )
            chosen = None
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
            break  # query + iteration succeeded (chosen may legitimately be None)
        except Exception as exc:  # noqa: BLE001
            # Transient vendor failures (502s, connection resets, timeouts,
            # rate-limit blips) must NOT abort the caller's whole batch. Retry
            # with backoff; on persistent failure return None — the caller
            # treats that as a missing quote (auditable failure_reason), not a
            # crash. A non-transient error (a real bug) is re-raised so it
            # surfaces rather than being silently masked as a missing quote.
            # See decisions/2026-05-28_quote-fetch-resilience.md.
            if not _is_transient_quote_error(exc):
                raise
            if attempt == QUOTE_RETRY_ATTEMPTS - 1:
                logger.warning(
                    "quote fetch gave up for %s @ %s after %d attempt(s): %s",
                    occ_ticker, sample_time.isoformat(), attempt + 1, exc,
                )
                return None
            logger.info(
                "transient quote error for %s (attempt %d/%d), retrying in %.1fs: %s",
                occ_ticker, attempt + 1, QUOTE_RETRY_ATTEMPTS,
                QUOTE_RETRY_SLEEP_SECONDS, exc,
            )
            time.sleep(QUOTE_RETRY_SLEEP_SECONDS + random.uniform(0, 0.25))

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


def _daily_opens_from_underlying(underlying_df) -> dict[date, float]:
    """For each ET trading day in ``underlying_df``, return the open of the
    first regular-trading-hours (>= 09:30 ET) bar. Skips days with no RTH
    data. Pure pandas helper; safe to call without polygon access."""
    import pandas as pd

    if underlying_df is None or len(underlying_df) == 0:
        return {}

    ts_utc = pd.to_datetime(underlying_df["timestamp"], utc=True)
    ts_et = ts_utc.dt.tz_convert(_ET)
    et_dates = ts_et.dt.date
    et_times = ts_et.dt.time
    rth_mask = (et_times >= _RTH_OPEN_ET) & (et_times < _RTH_CLOSE_ET)

    rth = underlying_df.loc[rth_mask].assign(
        _et_date=et_dates[rth_mask].values,
        _ts_ns=ts_utc[rth_mask].astype("int64").values,
    )
    first_bars = (
        rth.sort_values("_ts_ns", kind="stable")
        .groupby("_et_date", sort=True)
        .first()
    )
    return {d: float(row["open"]) for d, row in first_bars.iterrows()}


def _expiries_to_pull(trading_day: date, horizon: int = EXPIRIES_PER_TRADING_DAY) -> list[date]:
    """Return the next ``horizon`` weekly Fridays from ``trading_day`` inclusive.

    If ``trading_day`` is itself a Friday, it is the first element.
    Used so per-event ``nearest_weekly_friday`` lookups can resolve to
    either today's Friday (events 09:35-15:30 ET) or next week's (Friday
    post-close + forward-horizon samples)."""
    fridays = list(
        iter_weekly_fridays(trading_day, trading_day + timedelta(days=7 * (horizon + 1)))
    )
    return fridays[:horizon]


def fetch_contracts_for_month(
    year_month: str,
    underlying_df,
    expiries_per_day: int = EXPIRIES_PER_TRADING_DAY,
    inter_call_sleep_seconds: float = INTER_CALL_SLEEP_SECONDS,
):
    """For each trading day in ``year_month``, list option contracts in
    ``ATM_open +/- INGEST_STRIKE_HALF_WIDTH`` for the next
    ``expiries_per_day`` weekly Friday expiries.

    Returns a DataFrame with one row per (date_observed, expiry, contract).
    The ``timestamp`` column carries the trading-day midnight UTC so
    ``storage.write_partitioned_parquet`` routes each row to the correct
    locked split. Empty if ``underlying_df`` has no RTH bars for the month.
    """
    import pandas as pd

    daily_opens = _daily_opens_from_underlying(underlying_df)
    yr, mo = (int(p) for p in year_month.split("-"))
    daily_opens = {
        d: spot for d, spot in daily_opens.items() if d.year == yr and d.month == mo
    }

    columns = [
        "timestamp", "ingest_spot", "expiry", "ticker", "strike",
        "contract_type", "primary_exchange", "expiration_date",
    ]
    if not daily_opens:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    client = _client()
    for trading_day in sorted(daily_opens):
        spot = daily_opens[trading_day]
        spot_min = spot - INGEST_STRIKE_HALF_WIDTH
        spot_max = spot + INGEST_STRIKE_HALF_WIDTH
        for expiry in _expiries_to_pull(trading_day, expiries_per_day):
            for kind in ("call", "put"):
                for contract in client.list_options_contracts(
                    underlying_ticker=SPEC.options_symbol,
                    expiration_date=expiry.isoformat(),
                    contract_type=kind,
                    strike_price_gte=spot_min,
                    strike_price_lte=spot_max,
                    expired=True,
                    limit=1000,
                ):
                    c = contract if isinstance(contract, dict) else contract.__dict__
                    rows.append({
                        "timestamp": pd.Timestamp(trading_day, tz="UTC"),
                        "ingest_spot": spot,
                        "expiry": expiry.isoformat(),
                        "ticker": c.get("ticker"),
                        "strike": float(c.get("strike_price", 0.0)),
                        "contract_type": c.get("contract_type", kind),
                        "primary_exchange": c.get("primary_exchange", ""),
                        "expiration_date": c.get("expiration_date", expiry.isoformat()),
                    })
            time.sleep(inter_call_sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(rows)


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
