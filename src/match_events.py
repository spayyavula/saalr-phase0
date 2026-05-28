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

This module contains only the **pure helpers** for sample
construction. The I/O-heavy mid-quote pull at each ``(t, expiry,
strike)`` and the orchestrator stage land in a follow-on commit
once ``backfill_options`` has finished writing the contract universe.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from src.locked_spec import SPEC

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
