"""Tests for the pure helpers in src/options.py — no Polygon API calls.

The contract-listing path itself is exercised live in
``.phase0/smoke_options_backfill.py``; here we lock the date-arithmetic
and underlying-bar parsing that drive it.
"""
from __future__ import annotations

import importlib.util
from datetime import date

import pytest

if importlib.util.find_spec("pandas") is None:
    pytest.skip("pandas required", allow_module_level=True)

import pandas as pd

from src.options import (
    EXPIRIES_PER_TRADING_DAY,
    _daily_opens_from_underlying,
    _expiries_to_pull,
    iter_weekly_fridays,
)


# -- iter_weekly_fridays (defensive lock; already used by ingest helpers) ----


def test_iter_weekly_fridays_includes_start_if_friday():
    fri = date(2024, 1, 5)
    out = list(iter_weekly_fridays(fri, fri + pd.Timedelta(days=14).to_pytimedelta()))
    assert out[0] == fri
    assert out[1] == date(2024, 1, 12)


def test_iter_weekly_fridays_skips_to_next_friday_if_start_is_not():
    monday = date(2024, 1, 1)  # Mon
    out = list(iter_weekly_fridays(monday, date(2024, 1, 12)))
    assert out[0] == date(2024, 1, 5)
    assert out[1] == date(2024, 1, 12)


# -- _expiries_to_pull -------------------------------------------------------


def test_expiries_to_pull_from_monday_picks_next_two_fridays():
    monday = date(2024, 1, 1)  # Mon
    assert _expiries_to_pull(monday, horizon=2) == [
        date(2024, 1, 5),
        date(2024, 1, 12),
    ]


def test_expiries_to_pull_from_friday_includes_today():
    friday = date(2024, 1, 5)
    assert _expiries_to_pull(friday, horizon=2) == [
        date(2024, 1, 5),
        date(2024, 1, 12),
    ]


def test_expiries_to_pull_horizon_one():
    wednesday = date(2024, 1, 3)
    assert _expiries_to_pull(wednesday, horizon=1) == [date(2024, 1, 5)]


def test_expiries_to_pull_default_horizon_matches_module_constant():
    monday = date(2024, 1, 1)
    out = _expiries_to_pull(monday)
    assert len(out) == EXPIRIES_PER_TRADING_DAY


# -- _daily_opens_from_underlying --------------------------------------------


def _bar(ts_utc: str, open_: float) -> dict:
    return {"timestamp": pd.Timestamp(ts_utc, tz="UTC"), "open": open_}


def test_daily_opens_picks_first_rth_bar_per_et_date():
    """09:30 ET = 14:30 UTC (EST) / 13:30 UTC (EDT). For a January day
    (EST), the 14:30 UTC bar is the daily open. Pre-RTH bars (e.g. 09:00
    ET = 14:00 UTC) must be ignored."""
    df = pd.DataFrame.from_records([
        _bar("2024-01-02 14:00:00", 470.50),  # 09:00 ET pre-market
        _bar("2024-01-02 14:30:00", 471.30),  # 09:30 ET = first RTH
        _bar("2024-01-02 14:31:00", 471.42),
        _bar("2024-01-02 20:59:00", 472.10),  # 15:59 ET (last RTH)
        _bar("2024-01-02 21:30:00", 472.05),  # 16:30 ET after-hours
        _bar("2024-01-03 14:30:00", 472.80),
    ])
    opens = _daily_opens_from_underlying(df)
    assert opens[date(2024, 1, 2)] == pytest.approx(471.30)
    assert opens[date(2024, 1, 3)] == pytest.approx(472.80)
    assert date(2024, 1, 1) not in opens
    assert len(opens) == 2


def test_daily_opens_handles_dst_summer():
    """09:30 EDT = 13:30 UTC. Same logic, hour-shifted."""
    df = pd.DataFrame.from_records([
        _bar("2024-07-15 13:00:00", 540.00),  # 09:00 ET pre-market
        _bar("2024-07-15 13:30:00", 540.50),  # 09:30 EDT = first RTH
        _bar("2024-07-15 19:59:00", 541.10),  # 15:59 EDT
    ])
    opens = _daily_opens_from_underlying(df)
    assert opens[date(2024, 7, 15)] == pytest.approx(540.50)


def test_daily_opens_excludes_after_hours_only_days():
    """A day with only post-RTH bars (e.g., a partial-day collection
    artifact) returns no entry for that date."""
    df = pd.DataFrame.from_records([
        _bar("2024-01-02 21:00:00", 472.00),  # 16:00 ET (exact close — excluded)
        _bar("2024-01-02 21:30:00", 472.10),
    ])
    opens = _daily_opens_from_underlying(df)
    assert opens == {}


def test_daily_opens_empty_frame_returns_empty_dict():
    df = pd.DataFrame(columns=["timestamp", "open"])
    assert _daily_opens_from_underlying(df) == {}
    assert _daily_opens_from_underlying(None) == {}


def test_daily_opens_with_february_dst_boundary_uses_correct_et_date():
    """A bar at 00:30 UTC on 2026-02-17 = 19:30 ET on 2026-02-16
    (Presidents Day, market closed) — should NOT get assigned to
    2026-02-17 as if it were the open. After-hours bars on holidays
    fall outside RTH and are excluded."""
    df = pd.DataFrame.from_records([
        # 2026-02-17 00:30 UTC = 2026-02-16 19:30 ET (after Pres Day close)
        _bar("2026-02-17 00:30:00", 100.0),
        # 2026-02-17 14:30 UTC = 09:30 EST on 2026-02-17 (Tue, market open)
        _bar("2026-02-17 14:30:00", 101.5),
    ])
    opens = _daily_opens_from_underlying(df)
    assert opens == {date(2026, 2, 17): pytest.approx(101.5)}
