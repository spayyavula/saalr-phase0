"""Black-Scholes implied-volatility inversion for SPY weekly options.

Per ``pre-registration.md`` §5 (locked in ``SPEC``):

- ``iv_inversion_method = "black_scholes"`` — vanilla non-dividend BS.
- ``iv_quote_basis = "call_put_mid"`` — the per-event IV is the average
  of the inverted call IV and the inverted put IV at the locked
  ATM-closest-listed strike.
- ``expiry_rule = "nearest_weekly_friday"``.
- ``strike_rule = "atm_closest_listed"``.

The day-count convention for ``T`` (years-to-expiry) is **ACT/365 with
expiry at SPEC.event_window_close_et (15:30 ET) on the expiry date**.
That choice is documented in
``decisions/2026-05-28_iv-day-count-and-arbitrage-handling.md``; it is
not pinned in the pre-registration because it does not affect the
locked Spearman rank IC (every event uses the same convention).

``scipy.optimize.brentq`` does the inversion on a wide bracket
``[1e-6, 5.0]`` with explicit arbitrage checks against the no-arbitrage
band ``[intrinsic, S]`` for calls and ``[intrinsic, K * exp(-r*T)]`` for
puts. Out-of-band prices return ``NaN`` rather than a brentq error;
this is the right semantic for a per-row computation that the
downstream pipeline will count and drop.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from scipy.optimize import brentq
from scipy.stats import norm

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)

OptionType = Literal["call", "put"]

_ET = ZoneInfo("America/New_York")
_DAYS_PER_YEAR = 365.0
_IV_BRACKET_LO = 1e-6
_IV_BRACKET_HI = 5.0


def _expiry_close_utc(expiry: date) -> datetime:
    """Market close at ET on the expiry date, returned in UTC.

    Uses ``SPEC.event_window_close_et`` (15:30 ET) as the cash-settlement
    moment. ``zoneinfo`` handles DST so the UTC offset is correct
    year-round.
    """
    close_t = time.fromisoformat(SPEC.event_window_close_et)
    naive = datetime.combine(expiry, close_t)
    aware_et = naive.replace(tzinfo=_ET)
    return aware_et.astimezone(timezone.utc)


def time_to_expiry_years(sample_time_utc: datetime, expiry: date) -> float:
    """Years from ``sample_time_utc`` to ET market close on ``expiry``.

    ACT/365 with sub-day resolution (seconds / (365 * 86400)). Returns
    a negative value if sample is past expiry; callers should treat
    ``T <= 0`` as a dead contract.
    """
    if sample_time_utc.tzinfo is None:
        sample_time_utc = sample_time_utc.replace(tzinfo=timezone.utc)
    expiry_utc = _expiry_close_utc(expiry)
    delta_seconds = (expiry_utc - sample_time_utc).total_seconds()
    return delta_seconds / (_DAYS_PER_YEAR * 86400.0)


def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    """Vanilla non-dividend Black-Scholes price for a European call or put."""
    if S <= 0 or K <= 0:
        return float("nan")
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if sigma <= 0:
        if option_type == "call":
            return max(S - K * math.exp(-r * T), 0.0)
        return max(K * math.exp(-r * T) - S, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _no_arbitrage_band(
    S: float, K: float, T: float, r: float, option_type: OptionType
) -> tuple[float, float]:
    """Lower/upper no-arbitrage bounds for the European option price.

    Lower: max(0, S - K*e^{-rT}) for a call, max(0, K*e^{-rT} - S) for a
    put. Upper: S for a call (worst case: free shares), K*e^{-rT} for a
    put (worst case: cash discounted from strike).
    """
    discounted_strike = K * math.exp(-r * T)
    if option_type == "call":
        return max(0.0, S - discounted_strike), S
    return max(0.0, discounted_strike - S), discounted_strike


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float = 1e-6,
) -> float:
    """Invert BS to find the volatility that prices ``option_type`` at ``price``.

    Returns ``NaN`` when:

    - ``T <= 0`` (contract is dead),
    - ``price`` is outside the no-arbitrage band, or
    - ``brentq`` cannot find a root in ``[1e-6, 5.0]`` (effectively a
      band-clipping that no positive-finite sigma can match).

    Tested in ``tests/test_iv_surface.py``.
    """
    if not (S > 0 and K > 0 and T > 0 and math.isfinite(price)):
        return float("nan")

    lo_band, hi_band = _no_arbitrage_band(S, K, T, r, option_type)
    # Strict arbitrage rejection beats a brentq sign-of-error gymnastics.
    if price < lo_band - tol or price > hi_band + tol:
        return float("nan")

    def err(sigma: float) -> float:
        return _bs_price(S, K, T, r, sigma, option_type) - price

    try:
        err_lo = err(_IV_BRACKET_LO)
        err_hi = err(_IV_BRACKET_HI)
        if err_lo * err_hi > 0:
            return float("nan")
        sigma = brentq(err, _IV_BRACKET_LO, _IV_BRACKET_HI, xtol=tol)
    except (ValueError, RuntimeError) as exc:
        logger.debug(
            "brentq failed for price=%s S=%s K=%s T=%s r=%s type=%s: %s",
            price, S, K, T, r, option_type, exc,
        )
        return float("nan")
    return float(sigma)


def call_put_mid_iv(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
) -> float:
    """Per-event IV per ``SPEC.iv_quote_basis = "call_put_mid"``.

    Returns the average of the call-implied and put-implied vols.
    If either inversion fails (returns NaN), the result is NaN — the
    pre-reg's "mid" basis means both legs must be valid.
    """
    sigma_call = implied_volatility(call_price, S, K, T, r, "call")
    sigma_put = implied_volatility(put_price, S, K, T, r, "put")
    if not (math.isfinite(sigma_call) and math.isfinite(sigma_put)):
        return float("nan")
    return 0.5 * (sigma_call + sigma_put)
