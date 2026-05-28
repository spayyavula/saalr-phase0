"""Black-Scholes IV inversion over a matched-events frame.

Applies the frozen ``iv_surface`` inversion to the ``match_events``
output, producing the columns ``evaluate_primary`` requires:
``signal``, ``prior_iv_change``, ``forward_iv_change`` (plus the raw
``iv_tminus30`` / ``iv_t`` / ``iv_t30`` for the writeup).

Per the same-contract policy (match_events decision D1), all three IV
samples use the contract identity frozen at t. The t-30 sample is
**best-effort**: events early in the session have no pre-open quote, so
``iv_tminus30`` / ``prior_iv_change`` are NaN for them while ``iv_t`` /
``iv_t30`` / ``forward_iv_change`` still compute. Those events still
count for the primary IC; B1 drops the NaN-prior rows via its own
NaN handling.
"""
from __future__ import annotations

import math
from datetime import date as _date, timedelta

from src.iv_surface import call_put_mid_iv, time_to_expiry_years
from src.locked_spec import SPEC

_IV_COLUMNS = (
    "iv_tminus30",
    "iv_t",
    "iv_t30",
    "prior_iv_change",
    "forward_iv_change",
    "signal",
)


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _iv_at(call_mid, put_mid, spot, strike, sample_time, expiry, rfr) -> float:
    """One-sample call+put-mid IV; NaN if any required input is missing."""
    if _is_missing(call_mid) or _is_missing(put_mid) or _is_missing(spot):
        return float("nan")
    T = time_to_expiry_years(sample_time, expiry)
    return call_put_mid_iv(
        float(call_mid), float(put_mid), float(spot), float(strike), T, float(rfr)
    )


def compute_iv_for_events(events_df):
    """Add IV columns to a match_events frame. Returns a copy.

    New columns: ``iv_tminus30``, ``iv_t``, ``iv_t30``,
    ``prior_iv_change`` (= iv_t - iv_tminus30), ``forward_iv_change``
    (= iv_t30 - iv_t), and ``signal`` (alias of ``aggregated_signal``,
    the name ``evaluate_primary`` consumes).
    """
    import pandas as pd

    out = events_df.copy()
    if len(out) == 0:
        for col in _IV_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    horizon = timedelta(minutes=SPEC.forward_horizon_minutes)
    iv_tm: list[float] = []
    iv_t: list[float] = []
    iv_t30: list[float] = []
    prior: list[float] = []
    forward: list[float] = []
    signal: list[float] = []

    for _, row in events_df.iterrows():
        t = pd.Timestamp(row["timestamp"]).to_pydatetime()
        expiry_raw = row.get("expiry")
        expiry = _date.fromisoformat(expiry_raw) if expiry_raw else None
        strike = row.get("atm_strike")
        rfr = row.get("rfr")
        contract_known = expiry is not None and not _is_missing(strike) and not _is_missing(rfr)

        def iv_at(call_key, put_key, spot_key, sample_time):
            if not contract_known:
                return float("nan")
            return _iv_at(
                row.get(call_key), row.get(put_key), row.get(spot_key),
                strike, sample_time, expiry, rfr,
            )

        v_tm = iv_at("call_mid_tminus30", "put_mid_tminus30", "spot_at_tminus30", t - horizon)
        v_t = iv_at("call_mid_t", "put_mid_t", "spot_at_t", t)
        v_t30 = iv_at("call_mid_t30", "put_mid_t30", "spot_at_t30", t + horizon)

        iv_tm.append(v_tm)
        iv_t.append(v_t)
        iv_t30.append(v_t30)
        prior.append(
            v_t - v_tm if not (math.isnan(v_t) or math.isnan(v_tm)) else float("nan")
        )
        forward.append(
            v_t30 - v_t if not (math.isnan(v_t30) or math.isnan(v_t)) else float("nan")
        )
        signal.append(float(row["aggregated_signal"]))

    out["iv_tminus30"] = iv_tm
    out["iv_t"] = iv_t
    out["iv_t30"] = iv_t30
    out["prior_iv_change"] = prior
    out["forward_iv_change"] = forward
    out["signal"] = signal
    return out
