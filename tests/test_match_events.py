"""Tests for the pure helpers in src/match_events.py — no API calls."""
from __future__ import annotations

import importlib.util
import math
from datetime import date, datetime, timezone

import pytest

if importlib.util.find_spec("pandas") is None:
    pytest.skip("pandas required", allow_module_level=True)

import pandas as pd

from src.locked_spec import SPEC
from src.match_events import (
    compute_aggregated_signal,
    find_rfr_at_date,
    find_spot_at_minute,
    is_in_event_window,
    nearest_weekly_friday,
    pick_atm_strike,
)


# -- is_in_event_window -------------------------------------------------


def test_event_window_inside_winter_est():
    # 14:30 UTC on 2024-01-08 (Mon) = 09:30 EST. Below the 09:35 open -> False
    assert is_in_event_window(datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)) is False
    # 14:35 UTC = 09:35 EST -> True (inclusive open)
    assert is_in_event_window(datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)) is True
    # 20:29 UTC = 15:29 EST -> True
    assert is_in_event_window(datetime(2024, 1, 8, 20, 29, tzinfo=timezone.utc)) is True
    # 20:30 UTC = 15:30 EST -> False (exclusive close)
    assert is_in_event_window(datetime(2024, 1, 8, 20, 30, tzinfo=timezone.utc)) is False


def test_event_window_inside_summer_edt():
    # 13:35 UTC on 2024-07-15 (Mon) = 09:35 EDT -> True
    assert is_in_event_window(datetime(2024, 7, 15, 13, 35, tzinfo=timezone.utc)) is True
    # 19:29 UTC = 15:29 EDT -> True
    assert is_in_event_window(datetime(2024, 7, 15, 19, 29, tzinfo=timezone.utc)) is True
    # 19:30 UTC = 15:30 EDT -> False
    assert is_in_event_window(datetime(2024, 7, 15, 19, 30, tzinfo=timezone.utc)) is False


def test_event_window_excludes_weekends():
    # 2024-01-06 was a Saturday. 14:35 UTC = 09:35 EST on Sat -> False
    assert is_in_event_window(datetime(2024, 1, 6, 14, 35, tzinfo=timezone.utc)) is False
    # 2024-01-07 was a Sunday
    assert is_in_event_window(datetime(2024, 1, 7, 14, 35, tzinfo=timezone.utc)) is False


def test_event_window_naive_treated_as_utc():
    naive = datetime(2024, 1, 8, 14, 35)  # no tzinfo
    assert is_in_event_window(naive) is True


# -- nearest_weekly_friday ----------------------------------------------


def test_nearest_weekly_friday_from_monday():
    # 2024-01-08 is Monday -> Friday 2024-01-12
    assert nearest_weekly_friday(date(2024, 1, 8)) == date(2024, 1, 12)


def test_nearest_weekly_friday_from_friday_is_same_day():
    # 2024-01-12 is Friday -> 2024-01-12
    assert nearest_weekly_friday(date(2024, 1, 12)) == date(2024, 1, 12)


def test_nearest_weekly_friday_from_thursday():
    # 2024-01-11 (Thu) -> 2024-01-12 (Fri)
    assert nearest_weekly_friday(date(2024, 1, 11)) == date(2024, 1, 12)


def test_nearest_weekly_friday_from_saturday():
    # 2024-01-13 (Sat) -> 2024-01-19 (next Fri)
    assert nearest_weekly_friday(date(2024, 1, 13)) == date(2024, 1, 19)


# -- pick_atm_strike -----------------------------------------------------


def test_pick_atm_closest_integer():
    assert pick_atm_strike(471.30, [468, 469, 470, 471, 472, 473]) == 471.0


def test_pick_atm_handles_half_dollar_strikes():
    assert pick_atm_strike(471.30, [470.0, 470.5, 471.0, 471.5, 472.0]) == pytest.approx(471.5)


def test_pick_atm_tie_breaks_to_lower_strike():
    # Spot is exactly between 470 and 471 -> 470 (lower wins)
    assert pick_atm_strike(470.5, [470, 471]) == 470.0


def test_pick_atm_empty_returns_none():
    assert pick_atm_strike(500.0, []) is None


def test_pick_atm_ignores_duplicates():
    # Should treat as set; correct closest is still 471
    assert pick_atm_strike(471.30, [471, 471, 471, 472]) == 471.0


# -- compute_aggregated_signal -------------------------------------------


def test_signal_zero_when_no_articles_in_window():
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    s = compute_aggregated_signal([], [], eval_t)
    assert s == 0.0


def test_signal_drops_articles_older_than_lookback():
    eval_t = datetime(2024, 1, 8, 18, 0, tzinfo=timezone.utc)
    # 5h-old article (outside 4h window) is dropped
    too_old = eval_t - pd.Timedelta(hours=5)
    s = compute_aggregated_signal([too_old], [1.0], eval_t)
    assert s == 0.0


def test_signal_at_t_includes_article_at_t_with_unit_weight():
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    s = compute_aggregated_signal([eval_t], [0.7], eval_t)
    assert s == pytest.approx(0.7, abs=1e-9)


def test_signal_decays_at_halflife():
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    # One article exactly halflife_minutes (30) old -> weight 0.5
    at_halflife = eval_t - pd.Timedelta(minutes=SPEC.ewma_halflife_minutes)
    s = compute_aggregated_signal([at_halflife], [1.0], eval_t)
    assert s == pytest.approx(0.5, abs=1e-9)


def test_signal_sums_multiple_articles_with_their_weights():
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    # Two articles: at t (weight 1.0) and at t - 60min (weight 0.25 since
    # 60 min == 2 halflives -> exp(-2*ln2) == 0.25)
    times = [eval_t, eval_t - pd.Timedelta(minutes=60)]
    scores = [1.0, 1.0]
    s = compute_aggregated_signal(times, scores, eval_t)
    assert s == pytest.approx(1.0 + 0.25, abs=1e-9)


def test_signal_ignores_future_articles():
    """An article timestamped AFTER eval_time is dropped (no look-ahead)."""
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    future = eval_t + pd.Timedelta(minutes=10)
    s = compute_aggregated_signal([future], [1.0], eval_t)
    assert s == 0.0


def test_signal_handles_naive_article_times():
    eval_t = datetime(2024, 1, 8, 14, 35, tzinfo=timezone.utc)
    naive = datetime(2024, 1, 8, 14, 35)  # naive, treat as UTC
    s = compute_aggregated_signal([naive], [0.4], eval_t)
    assert s == pytest.approx(0.4, abs=1e-9)


# -- find_spot_at_minute -------------------------------------------------


def _bar(ts_iso, close):
    return {"timestamp": pd.Timestamp(ts_iso, tz="UTC"), "close": close}


def test_spot_picks_bar_at_or_before_ts():
    df = pd.DataFrame.from_records([
        _bar("2024-01-08 14:30:00", 470.0),
        _bar("2024-01-08 14:31:00", 470.5),
        _bar("2024-01-08 14:32:00", 470.8),
    ])
    # Exactly on the 14:31 bar -> use that bar's close
    spot = find_spot_at_minute(df, datetime(2024, 1, 8, 14, 31, tzinfo=timezone.utc))
    assert spot == pytest.approx(470.5)
    # 14:31:30 (between 14:31 and 14:32) -> still use 14:31
    spot = find_spot_at_minute(df, datetime(2024, 1, 8, 14, 31, 30, tzinfo=timezone.utc))
    assert spot == pytest.approx(470.5)


def test_spot_none_when_ts_predates_first_bar():
    df = pd.DataFrame.from_records([_bar("2024-01-08 14:30:00", 470.0)])
    spot = find_spot_at_minute(df, datetime(2024, 1, 8, 14, 29, tzinfo=timezone.utc))
    assert spot is None


def test_spot_none_for_empty_frame():
    df = pd.DataFrame(columns=["timestamp", "close"])
    assert find_spot_at_minute(df, datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)) is None
    assert find_spot_at_minute(None, datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)) is None


# -- find_rfr_at_date ----------------------------------------------------


def _rfr_row(d_iso, yield_pct):
    return {"timestamp": pd.Timestamp(d_iso, tz="UTC"), "yield_pct": yield_pct}


def test_rfr_returns_decimal_for_exact_match():
    df = pd.DataFrame.from_records([
        _rfr_row("2024-01-08", 3.65),
        _rfr_row("2024-01-09", 3.70),
    ])
    # 3.65 / 100 -> 0.0365
    assert find_rfr_at_date(df, date(2024, 1, 8)) == pytest.approx(0.0365)


def test_rfr_forward_fills_weekends_and_holidays():
    df = pd.DataFrame.from_records([
        _rfr_row("2024-01-05", 3.60),  # Friday
        _rfr_row("2024-01-08", 3.65),  # Monday
    ])
    # 2024-01-06 (Sat) and 2024-01-07 (Sun) should use Friday's value
    assert find_rfr_at_date(df, date(2024, 1, 6)) == pytest.approx(0.036)
    assert find_rfr_at_date(df, date(2024, 1, 7)) == pytest.approx(0.036)
    assert find_rfr_at_date(df, date(2024, 1, 8)) == pytest.approx(0.0365)


def test_rfr_none_when_date_predates_first_obs():
    df = pd.DataFrame.from_records([_rfr_row("2024-01-08", 3.65)])
    assert find_rfr_at_date(df, date(2024, 1, 1)) is None


def test_rfr_none_for_empty_frame():
    df = pd.DataFrame(columns=["timestamp", "yield_pct"])
    assert find_rfr_at_date(df, date(2024, 1, 8)) is None
    assert find_rfr_at_date(None, date(2024, 1, 8)) is None
