"""Per-event sample construction for the §6 primary IC test.

Per ``pre-registration.md`` §4 / §5:

- An **event** is each news article whose ``published_utc`` falls in
  [09:35, 15:30) ET on a US trading day (the locked
  ``SPEC.event_window_open_et`` / ``event_window_close_et``, with
  half-open interval excluding the closing minute).
- The **signal at t** is the exponentially-weighted sum of all article
  scores published in [t - aggregation_lookback_hours, t] with weights
  ``exp(-(t - published_utc) / (halflife_minutes / ln 2))``. The
  current article is included.
- The **target at t** is the change in the ATM-closest-listed weekly
  call/put-mid IV from t to t + forward_horizon_minutes, with the
  contract identity (expiry + strike) **frozen at t** so both samples
  are measured on the same option.

The trading-day calendar is derived from the existence of
``data/{split}/underlying/{YYYY-MM}.parquet`` bars on a date, so the
project does not depend on an external exchange-calendar package.

Pure helpers (no I/O) live above ``build_events_frame_for_month``.
The frame-builder at the bottom drives the per-event mid-quote pulls
via ``src.options.fetch_option_mid_quote_at`` and returns the
events frame that ``compute_iv`` and ``evaluate_validation`` consume.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_EVENT_OPEN_ET = dtime.fromisoformat(SPEC.event_window_open_et)
_EVENT_CLOSE_ET = dtime.fromisoformat(SPEC.event_window_close_et)


# ---------------------------------------------------------------------------
# Event window filter
# ---------------------------------------------------------------------------


def is_in_event_window(ts_utc: datetime) -> bool:
    """Return True if ``ts_utc`` converted to ET falls in the
    half-open window [SPEC.event_window_open_et, SPEC.event_window_close_et).

    The trading-day check (weekday + holiday) is **not** done here —
    callers should additionally filter to dates with underlying bars.
    Doing so here would require a holiday calendar, which we instead
    derive from on-disk data. Weekends are excluded as a fast pre-filter
    because no underlying parquet would contain them anyway.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    et = ts_utc.astimezone(_ET)
    if et.weekday() >= 5:  # 5 == Saturday, 6 == Sunday
        return False
    t = et.time()
    return _EVENT_OPEN_ET <= t < _EVENT_CLOSE_ET


# ---------------------------------------------------------------------------
# Expiry selection
# ---------------------------------------------------------------------------


def nearest_weekly_friday(event_date: date) -> date:
    """Return the nearest weekly Friday expiry at or after ``event_date``.

    Per ``SPEC.expiry_rule = "nearest_weekly_friday"``. Today's Friday
    counts (an event Friday morning still trades the Friday expiry).
    """
    days_ahead = (4 - event_date.weekday()) % 7  # 4 == Friday
    return event_date + timedelta(days=days_ahead)


# ---------------------------------------------------------------------------
# ATM strike selection
# ---------------------------------------------------------------------------


def pick_atm_strike(spot: float, available_strikes) -> Optional[float]:
    """Return the strike in ``available_strikes`` closest to ``spot``.

    Per ``SPEC.strike_rule = "atm_closest_listed"`` — closest LISTED
    strike, not a synthetic round number. Ties broken toward the
    lower strike (deterministic and matches the convention of
    "round-to-even" being avoided in financial pricing).

    Returns ``None`` if ``available_strikes`` is empty.
    """
    if not available_strikes:
        return None
    sorted_strikes = sorted(set(float(k) for k in available_strikes))
    # Linear scan; the locked ingest window is at most ~50 strikes.
    best = sorted_strikes[0]
    best_diff = abs(best - spot)
    for k in sorted_strikes[1:]:
        diff = abs(k - spot)
        if diff < best_diff:
            best = k
            best_diff = diff
        # Ties (diff == best_diff) keep the smaller strike, which was
        # encountered first because we sorted ascending.
    return best


# ---------------------------------------------------------------------------
# EWMA sentiment aggregation
# ---------------------------------------------------------------------------


def compute_aggregated_signal(
    article_times,
    article_scores,
    eval_time: datetime,
    lookback_hours: int = SPEC.aggregation_lookback_hours,
    halflife_minutes: int = SPEC.ewma_halflife_minutes,
) -> float:
    """Compute the EWMA signal AT ``eval_time`` from articles in
    [eval_time - lookback, eval_time].

    Per ``pre-registration.md`` §5:
    - Window: [eval_time - lookback_hours, eval_time]
    - Weight: exp(-(eval_time - published_utc) / lambda) where
      lambda = halflife_minutes / ln(2)
    - Articles outside the window: dropped.
    - The article published AT eval_time contributes weight 1.

    Returns 0.0 if no articles in the window — that's the "no recent
    news" signal value, not NaN.

    ``article_times`` is an iterable of tz-aware ``datetime`` (or
    pandas Timestamp); ``article_scores`` is a parallel iterable of
    floats.
    """
    if eval_time.tzinfo is None:
        eval_time = eval_time.replace(tzinfo=timezone.utc)
    lambda_seconds = (halflife_minutes * 60.0) / math.log(2.0)
    lookback_seconds = lookback_hours * 3600.0
    weighted_sum = 0.0
    for t, s in zip(article_times, article_scores):
        if t is None or s is None:
            continue
        t_dt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(tzinfo=timezone.utc)
        delta = (eval_time - t_dt).total_seconds()
        if delta < 0 or delta > lookback_seconds:
            continue
        weight = math.exp(-delta / lambda_seconds)
        weighted_sum += weight * float(s)
    return weighted_sum


# ---------------------------------------------------------------------------
# Spot lookup
# ---------------------------------------------------------------------------


def find_spot_at_minute(underlying_df, ts_utc: datetime) -> Optional[float]:
    """Return the SPY ``close`` of the 1-minute bar containing ``ts_utc``,
    or the bar immediately preceding it if ``ts_utc`` falls between two
    bars (e.g., bar-open time exactly equals ts).

    Returns ``None`` if no bar on or before ``ts_utc`` exists in
    ``underlying_df``. Uses ``close`` as the as-of price for the minute
    (consistent with treating the bar as "the minute ending here").
    """
    import pandas as pd

    if underlying_df is None or len(underlying_df) == 0:
        return None
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ts_ns = pd.Timestamp(ts_utc).value
    bar_times = pd.to_datetime(underlying_df["timestamp"], utc=True).astype("int64")
    mask = bar_times <= ts_ns
    if not mask.any():
        return None
    idx = bar_times[mask].idxmax()
    return float(underlying_df.loc[idx, "close"])


# ---------------------------------------------------------------------------
# Risk-free rate lookup
# ---------------------------------------------------------------------------


EVENTS_FRAME_COLUMNS: tuple[str, ...] = (
    "article_id",
    "timestamp",            # event time = published_utc (UTC, tz-aware)
    "event_date_et",        # ET trading day
    "sentiment_score",      # this article's raw P_pos - P_neg
    "aggregated_signal",    # EWMA per SPEC.aggregation_lookback_hours/halflife
    "expiry",               # ISO date string, nearest weekly Friday
    "atm_strike",           # closest listed strike to spot_at_t
    "spot_at_tminus30",
    "spot_at_t",
    "spot_at_t30",
    "rfr",                  # decimal, e.g. 0.0365
    "call_ticker",
    "put_ticker",
    "call_mid_tminus30", "call_spread_pct_tminus30",
    "put_mid_tminus30",  "put_spread_pct_tminus30",
    "call_mid_t", "call_spread_pct_t",
    "put_mid_t",  "put_spread_pct_t",
    "call_mid_t30", "call_spread_pct_t30",
    "put_mid_t30",  "put_spread_pct_t30",
    "passes_spread_filter",
    "failure_reason",       # empty if usable for IV; descriptive string otherwise
)


def find_rfr_at_date(risk_free_df, d: date) -> Optional[float]:
    """Return the FRED DGS3MO observation on date ``d`` (or the most
    recent prior observation; weekends/holidays carry forward) as an
    annualized **decimal** rate suitable for BS — e.g. ``0.0365`` for
    3.65 %.

    ``src/risk_free.py`` writes the ``yield_pct`` column in percent
    (FRED's native unit); this helper divides by 100 so callers don't
    forget. Returns ``None`` if no observation on or before ``d`` exists.
    """
    import pandas as pd

    if risk_free_df is None or len(risk_free_df) == 0:
        return None
    ts = pd.to_datetime(risk_free_df["timestamp"], utc=True).dt.date
    mask = ts <= d
    if not mask.any():
        return None
    idx = ts[mask].idxmax()
    return float(risk_free_df.loc[idx, "yield_pct"]) / 100.0


# ---------------------------------------------------------------------------
# Contract lookups from the backfill_options universe
# ---------------------------------------------------------------------------


def _pull_six_quotes(call_ticker, put_ticker, tminus30_dt, t_dt, t30_dt):
    """Fetch the six mid-quotes (call/put at t-30, t, t+30) concurrently.

    The calls are independent and network-bound, so a 6-worker thread
    pool collapses ~6 sequential round-trips into ~1. ``_client()`` in
    ``src.options`` builds a fresh ``RESTClient`` per call, so each
    thread gets its own client — no shared mutable state.

    The t-30 pair is best-effort (events early in the session have no
    pre-open quote); callers must treat a ``None`` t-30 result as a
    null prior sample, not a failure.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.options import fetch_option_mid_quote_at

    specs = {
        ("call", "tminus30"): (call_ticker, tminus30_dt),
        ("put", "tminus30"): (put_ticker, tminus30_dt),
        ("call", "t"): (call_ticker, t_dt),
        ("put", "t"): (put_ticker, t_dt),
        ("call", "t30"): (call_ticker, t30_dt),
        ("put", "t30"): (put_ticker, t30_dt),
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            key: pool.submit(fetch_option_mid_quote_at, ticker, when)
            for key, (ticker, when) in specs.items()
        }
        return {key: fut.result() for key, fut in futures.items()}


def _build_contract_lookups(options_df):
    """Index the options contract universe for O(1) per-event lookup.

    Returns ``(ticker_by_key, strikes_by_de)`` where:
    - ``ticker_by_key[(date, expiry, strike, kind)] -> occ_ticker``
    - ``strikes_by_de[(date, expiry)] -> sorted list of strikes that
      have BOTH a call and a put listed`` (call_put_mid needs both legs)
    """
    import pandas as pd

    ticker_by_key: dict = {}
    legs_by_des: dict = {}  # (date, expiry, strike) -> set of kinds

    dates = pd.to_datetime(options_df["timestamp"], utc=True).dt.date
    for d, expiry, strike, kind, ticker in zip(
        dates,
        options_df["expiry"],
        options_df["strike"],
        options_df["contract_type"],
        options_df["ticker"],
    ):
        strike = float(strike)
        ticker_by_key[(d, expiry, strike, kind)] = ticker
        legs_by_des.setdefault((d, expiry, strike), set()).add(kind)

    strikes_by_de: dict = {}
    for (d, expiry, strike), kinds in legs_by_des.items():
        if "call" in kinds and "put" in kinds:
            strikes_by_de.setdefault((d, expiry), []).append(strike)
    for key in strikes_by_de:
        strikes_by_de[key].sort()

    return ticker_by_key, strikes_by_de


def _empty_row(article_id, t, et_d, raw_score, signal, failure_reason):
    """A row with metadata populated and all market columns null."""
    row = {c: None for c in EVENTS_FRAME_COLUMNS}
    row.update({
        "article_id": article_id,
        "timestamp": t,
        "event_date_et": et_d.isoformat(),
        "sentiment_score": raw_score,
        "aggregated_signal": signal,
        "passes_spread_filter": False,
        "failure_reason": failure_reason,
    })
    return row


def _build_one_event_row(
    t, et_d, article_id, raw_score, signal,
    underlying_df, risk_free_df, strikes_by_de, ticker_by_key,
    forward_horizon_minutes, inter_call_sleep_seconds,
):
    """Build one event row (success or with a failure_reason). Pulls the six
    mid-quotes. Raises only on genuinely unexpected errors — transient quote
    failures are absorbed by fetch_option_mid_quote_at (returns None)."""
    t_dt = t.to_pydatetime()
    t30 = t_dt + timedelta(minutes=forward_horizon_minutes)
    tminus30 = t_dt - timedelta(minutes=forward_horizon_minutes)

    row = _empty_row(article_id, t, et_d, raw_score, signal, "")

    spot_t = find_spot_at_minute(underlying_df, t_dt)
    spot_t30 = find_spot_at_minute(underlying_df, t30)
    rfr = find_rfr_at_date(risk_free_df, et_d)
    row["spot_at_tminus30"] = find_spot_at_minute(underlying_df, tminus30)
    row["spot_at_t"] = spot_t
    row["spot_at_t30"] = spot_t30
    row["rfr"] = rfr

    if spot_t is None:
        row["failure_reason"] = "no_underlying_at_t"
        return row
    if spot_t30 is None:
        row["failure_reason"] = "no_underlying_at_t30"
        return row
    if rfr is None:
        row["failure_reason"] = "no_rfr_for_date"
        return row

    expiry = nearest_weekly_friday(et_d)
    row["expiry"] = expiry.isoformat()
    available = strikes_by_de.get((et_d, expiry.isoformat()))
    atm = pick_atm_strike(spot_t, available) if available else None
    if atm is None:
        row["failure_reason"] = "no_atm_contract_for_expiry"
        return row
    row["atm_strike"] = atm

    call_ticker = ticker_by_key.get((et_d, expiry.isoformat(), atm, "call"))
    put_ticker = ticker_by_key.get((et_d, expiry.isoformat(), atm, "put"))
    row["call_ticker"] = call_ticker
    row["put_ticker"] = put_ticker
    if call_ticker is None or put_ticker is None:
        row["failure_reason"] = "missing_call_or_put_leg"
        return row

    quotes = _pull_six_quotes(call_ticker, put_ticker, tminus30, t_dt, t30)
    if inter_call_sleep_seconds:
        import time as _time
        _time.sleep(inter_call_sleep_seconds)

    for when in ("tminus30", "t", "t30"):
        for leg in ("call", "put"):
            q = quotes.get((leg, when))
            row[f"{leg}_mid_{when}"] = q["mid"] if q else None
            row[f"{leg}_spread_pct_{when}"] = q["spread_pct_of_mid"] if q else None

    # Only t and t30 are REQUIRED (they define forward_iv_change, the
    # target). t-30 is best-effort: a missing pre-open prior quote just
    # nulls prior_iv_change, which B1 drops, while the event still counts
    # for the primary IC.
    required_missing = sorted(
        f"{leg}_{when}"
        for (leg, when), q in quotes.items()
        if q is None and when in ("t", "t30")
    )
    if required_missing:
        row["failure_reason"] = "no_quote_" + ",".join(required_missing)
        return row

    spreads = [
        row["call_spread_pct_t"], row["put_spread_pct_t"],
        row["call_spread_pct_t30"], row["put_spread_pct_t30"],
    ]
    row["passes_spread_filter"] = all(
        s is not None and s <= SPEC.options_max_spread_pct_of_mid for s in spreads
    )
    return row


def build_events_frame_for_month(
    year_month: str,
    sentiment_df,
    underlying_df,
    options_df,
    risk_free_df,
    forward_horizon_minutes: int = SPEC.forward_horizon_minutes,
    inter_call_sleep_seconds: float = 0.0,
    max_events: Optional[int] = None,
):
    """Build the per-event sample frame for ``year_month``.

    For each news article that is an event (in-window, trading day,
    ET date in ``year_month``):

    1. ``aggregated_signal`` = EWMA over recent articles (``sentiment_df``
       should include the prior month so the 4h look-back is complete).
    2. ``spot_at_t`` / ``spot_at_t30`` from ``underlying_df``.
    3. nearest weekly Friday expiry; ATM-closest-listed strike from the
       contract universe; the (expiry, strike) is frozen for both samples
       (decision D1).
    4. Four mid-quotes via ``fetch_option_mid_quote_at``: call+put at t and
       at t+``forward_horizon_minutes``.

    Rows that cannot be fully populated are still emitted with a
    ``failure_reason`` so downstream stages can count drops rather than
    have them vanish. ``passes_spread_filter`` is True only when all four
    legs are within ``SPEC.options_max_spread_pct_of_mid``.

    ``max_events`` caps the number of events processed (smoke testing).
    """
    import pandas as pd

    if sentiment_df is None or len(sentiment_df) == 0:
        return pd.DataFrame(columns=EVENTS_FRAME_COLUMNS)

    yr, mo = (int(p) for p in year_month.split("-"))

    # Article (time, score) arrays for the EWMA — reused across events.
    article_times = pd.to_datetime(sentiment_df["timestamp"], utc=True).tolist()
    article_scores = [float(s) for s in sentiment_df["sentiment_score"].tolist()]

    # Trading-day calendar from underlying (ET dates with any bar).
    trading_days: set = set()
    if underlying_df is not None and len(underlying_df) > 0:
        ud_et = pd.to_datetime(underlying_df["timestamp"], utc=True).dt.tz_convert(_ET)
        trading_days = set(ud_et.dt.date)

    ticker_by_key, strikes_by_de = _build_contract_lookups(options_df)

    # Candidate events: this month's in-window articles on trading days.
    sent_ts = pd.to_datetime(sentiment_df["timestamp"], utc=True)
    sent_et_date = sent_ts.dt.tz_convert(_ET).dt.date
    rows_out: list[dict] = []
    processed = 0

    for i in range(len(sentiment_df)):
        t = sent_ts.iloc[i]
        et_d = sent_et_date.iloc[i]
        if et_d.year != yr or et_d.month != mo:
            continue
        if not is_in_event_window(t.to_pydatetime()):
            continue
        if et_d not in trading_days:
            continue

        if max_events is not None and processed >= max_events:
            break
        processed += 1

        article_id = sentiment_df.iloc[i].get("id")
        raw_score = float(sentiment_df.iloc[i]["sentiment_score"])
        signal = compute_aggregated_signal(article_times, article_scores, t.to_pydatetime())

        try:
            row = _build_one_event_row(
                t, et_d, article_id, raw_score, signal,
                underlying_df, risk_free_df, strikes_by_de, ticker_by_key,
                forward_horizon_minutes, inter_call_sleep_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            # One unexpected event failure must not abort the whole month's
            # ~1,500 events. Record it structurally and move on.
            logger.warning("event @ %s raised, recording as failure: %s", t, exc)
            row = _empty_row(
                article_id, t, et_d, raw_score, signal,
                f"exception:{type(exc).__name__}",
            )
        rows_out.append(row)

    if not rows_out:
        return pd.DataFrame(columns=EVENTS_FRAME_COLUMNS)
    return pd.DataFrame.from_records(rows_out, columns=EVENTS_FRAME_COLUMNS)
