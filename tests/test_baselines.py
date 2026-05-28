"""Tests for src/baselines.py against synthetic event frames."""
from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("pandas") is None:
    pytest.skip("pandas required", allow_module_level=True)

import numpy as np
import pandas as pd

from src.baselines import (
    b1_persistence,
    b2_random,
    b3_prior_day_same_time,
    compute_all_baselines,
)


def _synthetic_events(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    forward = rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal": rng.normal(size=n),
            "forward_iv_change": forward,
            "prior_iv_change": 0.5 * forward + rng.normal(scale=0.1, size=n),
        }
    )


def test_b1_persistence_picks_up_real_autocorr() -> None:
    events = _synthetic_events()
    result = b1_persistence(events)
    assert abs(result.ic) > 0.5
    assert result.p_value < 1e-6


def test_b2_random_kills_real_signal() -> None:
    rng = np.random.default_rng(99)
    n = 2000
    forward = rng.normal(size=n)
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"),
            "signal": forward + 0.3 * rng.normal(size=n),  # strong real signal
            "forward_iv_change": forward,
            "prior_iv_change": rng.normal(size=n),
        }
    )
    shuffled_result = b2_random(events, seed=42)
    assert abs(shuffled_result.ic) < 0.1


def test_compute_all_baselines_returns_three_independent_ics() -> None:
    events = _synthetic_events(n=500, seed=3)
    results = compute_all_baselines(events)
    assert results.b1_persistence.n > 0
    assert results.b2_random.n > 0
    assert results.b3_prior_day_same_time.n >= 0  # may have NaNs from reindex
    assert results.strongest_ic >= 0


def test_b3_handles_duplicate_timestamps_without_crashing() -> None:
    # Multiple news items can land in the same clock minute; the prior-day
    # join must not raise "cannot reindex on an axis with duplicate labels".
    base = pd.date_range("2025-01-01 14:00", periods=50, freq="D", tz="UTC")
    ts = base.append(base[:5])  # 5 same-minute duplicate timestamps
    n = len(ts)
    rng = np.random.default_rng(7)
    events = pd.DataFrame(
        {
            "timestamp": ts,
            "signal": rng.normal(size=n),
            "forward_iv_change": rng.normal(size=n),
            "prior_iv_change": rng.normal(size=n),
        }
    )
    result = b3_prior_day_same_time(events)
    assert result.n > 0  # daily spacing => prior-day matches exist
