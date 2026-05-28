"""Tests for src/iv_surface.py.

Requires scipy (for the inversion) and Python 3.9+ zoneinfo (for the
ET market-close conversion). Skips cleanly if scipy isn't installed.
"""
from __future__ import annotations

import importlib.util
import math
from datetime import date, datetime, timezone

import pytest

if importlib.util.find_spec("scipy") is None:
    pytest.skip("scipy required", allow_module_level=True)

from src.iv_surface import (
    _bs_price,
    _no_arbitrage_band,
    call_put_mid_iv,
    implied_volatility,
    time_to_expiry_years,
)


# -- round-trip ---------------------------------------------------------


@pytest.mark.parametrize("sigma_in", [0.10, 0.20, 0.35, 0.50, 0.80])
@pytest.mark.parametrize("moneyness", [-0.05, 0.0, 0.05])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_iv_roundtrip(sigma_in, moneyness, option_type):
    """BS(sigma) -> price -> IV(price) should recover sigma."""
    S = 100.0
    K = S * (1.0 + moneyness)
    T = 30.0 / 365.0
    r = 0.04
    price = _bs_price(S, K, T, r, sigma_in, option_type)
    sigma_out = implied_volatility(price, S, K, T, r, option_type)
    assert sigma_out == pytest.approx(sigma_in, abs=1e-5)


# -- put-call parity / IV equality from same-sigma synthetic prices ------


def test_call_and_put_iv_match_at_same_sigma():
    """If call and put prices come from the same sigma, inverted IVs match."""
    S, K, T, r, sigma = 100.0, 100.0, 7.0 / 365.0, 0.04, 0.25
    call_price = _bs_price(S, K, T, r, sigma, "call")
    put_price = _bs_price(S, K, T, r, sigma, "put")
    iv_call = implied_volatility(call_price, S, K, T, r, "call")
    iv_put = implied_volatility(put_price, S, K, T, r, "put")
    assert iv_call == pytest.approx(sigma, abs=1e-5)
    assert iv_put == pytest.approx(sigma, abs=1e-5)


# -- call_put_mid_iv -----------------------------------------------------


def test_call_put_mid_iv_averages_two_inverted_legs():
    S, K, T, r, sigma = 100.0, 100.0, 7.0 / 365.0, 0.04, 0.30
    call_price = _bs_price(S, K, T, r, sigma, "call")
    put_price = _bs_price(S, K, T, r, sigma, "put")
    mid = call_put_mid_iv(call_price, put_price, S, K, T, r)
    assert mid == pytest.approx(sigma, abs=1e-5)


def test_call_put_mid_iv_nan_when_either_leg_arbitrage():
    """Sub-intrinsic call price -> call IV is NaN -> mid is NaN."""
    S, K, T, r = 100.0, 100.0, 7.0 / 365.0, 0.04
    # ATM put at sigma=0.25 has a real price; we pair it with an
    # arbitrage-violating negative-priced call.
    put_price = _bs_price(S, K, T, r, 0.25, "put")
    mid = call_put_mid_iv(-1.0, put_price, S, K, T, r)
    assert math.isnan(mid)


# -- arbitrage rejection -------------------------------------------------


def test_iv_nan_for_price_below_intrinsic_call():
    S, K, T, r = 100.0, 90.0, 30.0 / 365.0, 0.04
    lo, _ = _no_arbitrage_band(S, K, T, r, "call")
    assert implied_volatility(lo - 0.5, S, K, T, r, "call") != implied_volatility(
        lo - 0.5, S, K, T, r, "call"
    ) or math.isnan(implied_volatility(lo - 0.5, S, K, T, r, "call"))
    assert math.isnan(implied_volatility(lo - 1.0, S, K, T, r, "call"))


def test_iv_nan_for_price_above_underlying_call():
    S, K, T, r = 100.0, 100.0, 30.0 / 365.0, 0.04
    assert math.isnan(implied_volatility(S + 1.0, S, K, T, r, "call"))


def test_iv_nan_for_price_above_discounted_strike_put():
    S, K, T, r = 100.0, 100.0, 30.0 / 365.0, 0.04
    _, hi = _no_arbitrage_band(S, K, T, r, "put")
    assert math.isnan(implied_volatility(hi + 1.0, S, K, T, r, "put"))


# -- degenerate inputs ---------------------------------------------------


def test_iv_nan_for_zero_time_to_expiry():
    assert math.isnan(implied_volatility(5.0, 100.0, 100.0, 0.0, 0.04, "call"))


def test_iv_nan_for_negative_time_to_expiry():
    assert math.isnan(implied_volatility(5.0, 100.0, 100.0, -0.01, 0.04, "call"))


def test_iv_nan_for_non_finite_price():
    assert math.isnan(implied_volatility(float("nan"), 100.0, 100.0, 0.1, 0.04, "call"))
    assert math.isnan(implied_volatility(float("inf"), 100.0, 100.0, 0.1, 0.04, "call"))


# -- BS price degeneracies (asserted directly because IV inversion masks them) --


def test_bs_price_intrinsic_at_expiry():
    assert _bs_price(110.0, 100.0, 0.0, 0.04, 0.25, "call") == pytest.approx(10.0)
    assert _bs_price(90.0, 100.0, 0.0, 0.04, 0.25, "put") == pytest.approx(10.0)
    assert _bs_price(95.0, 100.0, 0.0, 0.04, 0.25, "call") == pytest.approx(0.0)


def test_bs_price_zero_vol_is_discounted_intrinsic():
    S, K, T, r = 100.0, 95.0, 30.0 / 365.0, 0.04
    expected = S - K * math.exp(-r * T)
    assert _bs_price(S, K, T, r, 0.0, "call") == pytest.approx(expected)


# -- time_to_expiry_years ------------------------------------------------


def test_time_to_expiry_at_close_is_zero():
    # 2024-03-15 was a Friday. 15:30 ET = 19:30 UTC (during EDT? no,
    # March 15 2024 is between the second Sun of March (Mar 10) and
    # the first Sun of Nov, so EDT applies -> 15:30 EDT = 19:30 UTC).
    expiry = date(2024, 3, 15)
    sample = datetime(2024, 3, 15, 19, 30, 0, tzinfo=timezone.utc)
    T = time_to_expiry_years(sample, expiry)
    assert T == pytest.approx(0.0, abs=1e-9)


def test_time_to_expiry_seven_days_before_close():
    expiry = date(2024, 3, 15)
    sample = datetime(2024, 3, 8, 19, 30, 0, tzinfo=timezone.utc)  # 7 days earlier
    T = time_to_expiry_years(sample, expiry)
    assert T == pytest.approx(7.0 / 365.0, abs=1e-9)


def test_time_to_expiry_handles_dst_transition():
    """A sample on the EST-side of a DST transition still resolves to ET
    close on the expiry date. The week of 2024-03-10 -> 2024-03-15 crosses
    the spring-forward; the resulting T must be 5 days (calendar), not 5
    days minus an hour even though wall-clock hours differ by 23."""
    # Friday 2024-03-08 was pre-DST (EST = UTC-5); 15:30 EST = 20:30 UTC.
    # Friday 2024-03-15 was post-DST (EDT = UTC-4); 15:30 EDT = 19:30 UTC.
    sample = datetime(2024, 3, 8, 20, 30, 0, tzinfo=timezone.utc)  # 15:30 EST
    expiry = date(2024, 3, 15)
    T = time_to_expiry_years(sample, expiry)
    # Calendar = 7 days. Wall-time difference is 6 days 23 hours because of
    # spring-forward, so T should be (7*86400 - 3600) / (365*86400).
    expected_seconds = 7 * 86400 - 3600
    assert T == pytest.approx(expected_seconds / (365.0 * 86400.0), abs=1e-9)


def test_time_to_expiry_naive_sample_treated_as_utc():
    """A tz-naive datetime is assumed UTC (so we don't silently use the
    process-local timezone for an upstream input)."""
    naive = datetime(2024, 3, 15, 19, 30, 0)  # no tzinfo
    expiry = date(2024, 3, 15)
    T = time_to_expiry_years(naive, expiry)
    assert T == pytest.approx(0.0, abs=1e-9)
