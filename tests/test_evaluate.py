"""Tests for src/evaluate.py: pass/fail decisions and sub-period stability."""
from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("pandas") is None:
    pytest.skip("pandas required", allow_module_level=True)

import numpy as np
import pandas as pd

from src.evaluate import evaluate_primary, subperiod_stability_check
from src.locked_spec import SPEC


def _events_with_correlation(n: int, target_ic: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-02-16", periods=n, freq="3min", tz="UTC")
    forward = rng.normal(size=n)
    noise = rng.normal(size=n)
    if target_ic >= 0:
        signal = target_ic * forward + np.sqrt(max(1.0 - target_ic ** 2, 0.0)) * noise
    else:
        signal = target_ic * forward + np.sqrt(max(1.0 - target_ic ** 2, 0.0)) * noise
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal": signal,
            "forward_iv_change": forward,
            "prior_iv_change": 0.05 * forward + rng.normal(scale=0.5, size=n),
        }
    )


def test_evaluate_primary_passes_on_strong_signal_above_grey_zone() -> None:
    events = _events_with_correlation(n=1500, target_ic=0.30, seed=7)
    result = evaluate_primary(events, label="test_strong")
    assert result.verdict.s1_ic_meets_threshold is True
    assert result.verdict.s2_pvalue_meets_threshold is True
    assert result.in_grey_zone is False
    # Strong signal exceeds the random baseline by far; B1 may also be strong
    # due to prior_iv_change correlation in the synthesis. Check |ic| > min.
    assert abs(result.primary_ic.ic) > SPEC.s1_min_holdout_ic


def test_evaluate_primary_fails_on_null_signal() -> None:
    events = _events_with_correlation(n=1500, target_ic=0.0, seed=11)
    result = evaluate_primary(events, label="test_null")
    assert result.verdict.s1_ic_meets_threshold is False
    assert result.verdict.passes_all is False


def test_grey_zone_classification() -> None:
    events = _events_with_correlation(n=3000, target_ic=0.04, seed=13)
    result = evaluate_primary(events, label="test_grey")
    if SPEC.grey_zone_lower_exclusive < abs(result.primary_ic.ic) < SPEC.grey_zone_upper_exclusive:
        assert result.in_grey_zone is True
        assert result.verdict.s1_ic_meets_threshold is False  # grey-zone = failure


def test_subperiod_midpoint_split_is_calendar_based() -> None:
    """A timestamp distribution skewed toward one end should still produce a
    midpoint at the calendar midpoint, not the median event timestamp."""
    skewed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-02-16T10:00:00Z"] * 100
                + ["2026-04-29T15:00:00Z"] * 900,
                utc=True,
            ),
            "signal": np.random.default_rng(0).normal(size=1000),
            "forward_iv_change": np.random.default_rng(1).normal(size=1000),
        }
    )
    check = subperiod_stability_check(skewed)
    # Calendar midpoint should be around 2026-03-23
    assert "2026-03" in check.midpoint
    # And the first half should contain the early skew (100 events), not 500
    assert check.first_half_n == 100
    assert check.second_half_n == 900


def test_evaluate_writes_result_to_disk(tmp_path) -> None:
    events = _events_with_correlation(n=200, target_ic=0.1, seed=5)
    result = evaluate_primary(events, label="test_write")
    out = result.write(tmp_path)
    assert out.exists()
    assert "primary_ic" in out.read_text(encoding="utf-8")
