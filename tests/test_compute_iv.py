"""Tests for src/compute_iv.py.

Strategy: synthesize call/put mid prices from a KNOWN sigma via the
frozen iv_surface._bs_price, then assert compute_iv_for_events recovers
that sigma and wires prior_iv_change / forward_iv_change / signal
correctly. iv_surface itself is already tested in test_iv_surface.py;
here we test the column wiring, the three T computations, and the
change arithmetic.
"""
from __future__ import annotations

import importlib.util
import math
from datetime import date, datetime, timedelta, timezone

import pytest

if importlib.util.find_spec("scipy") is None or importlib.util.find_spec("pandas") is None:
    pytest.skip("scipy + pandas required", allow_module_level=True)

import pandas as pd

from src.compute_iv import compute_iv_for_events
from src.iv_surface import _bs_price, time_to_expiry_years
from src.locked_spec import SPEC


# A Monday event well inside the window; expiry that Friday.
_EVENT_T = datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc)  # 11:00 EDT
_EXPIRY = date(2024, 6, 7)
_STRIKE = 500.0
_SPOT = 500.0
_RFR = 0.04
_H = SPEC.forward_horizon_minutes  # 30


def _legs(sigma: float, sample_time: datetime, spot: float = _SPOT) -> tuple[float, float]:
    """call_mid, put_mid generated from a known sigma at sample_time."""
    T = time_to_expiry_years(sample_time, _EXPIRY)
    call = _bs_price(spot, _STRIKE, T, _RFR, sigma, "call")
    put = _bs_price(spot, _STRIKE, T, _RFR, sigma, "put")
    return call, put


def _row(sig_tminus30, sig_t, sig_t30, *, include_prior=True, aggregated_signal=0.5):
    tminus30 = _EVENT_T - timedelta(minutes=_H)
    t30 = _EVENT_T + timedelta(minutes=_H)
    call_t, put_t = _legs(sig_t, _EVENT_T)
    call_t30, put_t30 = _legs(sig_t30, t30)
    row = {
        "timestamp": pd.Timestamp(_EVENT_T),
        "aggregated_signal": aggregated_signal,
        "expiry": _EXPIRY.isoformat(),
        "atm_strike": _STRIKE,
        "spot_at_t": _SPOT,
        "spot_at_t30": _SPOT,
        "rfr": _RFR,
        "call_mid_t": call_t, "put_mid_t": put_t,
        "call_mid_t30": call_t30, "put_mid_t30": put_t30,
    }
    if include_prior:
        call_tm, put_tm = _legs(sig_tminus30, tminus30)
        row["spot_at_tminus30"] = _SPOT
        row["call_mid_tminus30"] = call_tm
        row["put_mid_tminus30"] = put_tm
    else:
        row["spot_at_tminus30"] = None
        row["call_mid_tminus30"] = None
        row["put_mid_tminus30"] = None
    return row


def test_forward_iv_change_is_iv_t30_minus_iv_t():
    df = pd.DataFrame([_row(0.18, 0.20, 0.25)])
    out = compute_iv_for_events(df)
    assert out["iv_t"].iloc[0] == pytest.approx(0.20, abs=1e-4)
    assert out["iv_t30"].iloc[0] == pytest.approx(0.25, abs=1e-4)
    assert out["forward_iv_change"].iloc[0] == pytest.approx(0.05, abs=1e-4)


def test_prior_iv_change_is_iv_t_minus_iv_tminus30():
    df = pd.DataFrame([_row(0.18, 0.20, 0.25)])
    out = compute_iv_for_events(df)
    assert out["iv_tminus30"].iloc[0] == pytest.approx(0.18, abs=1e-4)
    assert out["prior_iv_change"].iloc[0] == pytest.approx(0.02, abs=1e-4)


def test_signal_column_aliases_aggregated_signal():
    df = pd.DataFrame([_row(0.18, 0.20, 0.25, aggregated_signal=-0.73)])
    out = compute_iv_for_events(df)
    assert out["signal"].iloc[0] == pytest.approx(-0.73)


def test_missing_prior_leg_nulls_prior_change_but_keeps_forward():
    """Early-session events have no t-30 quote: prior_iv_change is NaN
    but iv_t / iv_t30 / forward_iv_change still compute."""
    df = pd.DataFrame([_row(0.18, 0.20, 0.25, include_prior=False)])
    out = compute_iv_for_events(df)
    assert math.isnan(out["iv_tminus30"].iloc[0])
    assert math.isnan(out["prior_iv_change"].iloc[0])
    assert out["iv_t"].iloc[0] == pytest.approx(0.20, abs=1e-4)
    assert out["forward_iv_change"].iloc[0] == pytest.approx(0.05, abs=1e-4)


def test_missing_required_t_leg_nulls_forward_change():
    df = pd.DataFrame([_row(0.18, 0.20, 0.25)])
    df.loc[0, "call_mid_t"] = None
    out = compute_iv_for_events(df)
    assert math.isnan(out["iv_t"].iloc[0])
    assert math.isnan(out["forward_iv_change"].iloc[0])


def test_empty_frame_returns_empty_with_columns():
    df = pd.DataFrame(columns=["timestamp", "aggregated_signal", "expiry",
                               "atm_strike", "spot_at_t", "spot_at_t30", "rfr",
                               "call_mid_t", "put_mid_t", "call_mid_t30",
                               "put_mid_t30", "spot_at_tminus30",
                               "call_mid_tminus30", "put_mid_tminus30"])
    out = compute_iv_for_events(df)
    assert len(out) == 0
    for col in ("iv_t", "iv_t30", "iv_tminus30", "prior_iv_change",
                "forward_iv_change", "signal"):
        assert col in out.columns
