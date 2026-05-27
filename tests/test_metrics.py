"""Tests for src/metrics.py.

These tests require numpy + scipy + pandas (i.e. ``pip install -r
requirements.txt``); they do not require torch or polygon-api-client.
"""
from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("scipy") is None:
    pytest.skip("numpy + scipy required", allow_module_level=True)

import numpy as np

from src.locked_spec import SPEC
from src.metrics import bootstrap_ic_ci, hit_rate, sharpe_approx, spearman_ic


def test_spearman_ic_perfect_monotonic_is_one() -> None:
    x = np.arange(100, dtype=float)
    y = x ** 2  # monotonic increasing, non-linear -> Spearman == 1
    ic, p = spearman_ic(x, y)
    assert ic == pytest.approx(1.0, abs=1e-9)
    assert p < 1e-6


def test_spearman_ic_perfect_anti_monotonic_is_neg_one() -> None:
    x = np.arange(100, dtype=float)
    y = -np.exp(x / 10.0)
    ic, _ = spearman_ic(x, y)
    assert ic == pytest.approx(-1.0, abs=1e-9)


def test_spearman_ic_independent_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = rng.normal(size=2000)
    ic, p = spearman_ic(x, y)
    assert abs(ic) < 0.1
    assert p > 0.01  # not significant for genuinely independent data


def test_bootstrap_ci_contains_point_estimate_for_strong_signal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = x + 0.3 * rng.normal(size=500)
    result = bootstrap_ic_ci(x, y, resamples=500, seed=1)
    assert result.ci_low <= result.ic <= result.ci_high
    assert result.confidence_level == SPEC.confidence_level


def test_bootstrap_ci_uses_spec_defaults_when_unspecified() -> None:
    x = np.linspace(0, 1, 50)
    y = x + np.random.default_rng(0).normal(scale=0.1, size=50)
    result = bootstrap_ic_ci(x, y, seed=0)
    assert result.bootstrap_resamples == SPEC.bootstrap_resamples
    assert result.confidence_level == SPEC.confidence_level


def test_bootstrap_ci_handles_too_few_samples() -> None:
    result = bootstrap_ic_ci([1.0], [2.0])
    assert np.isnan(result.ic)
    assert np.isnan(result.ci_low) and np.isnan(result.ci_high)


def test_hit_rate_basic() -> None:
    pred = np.array([1, -1, 1, -1, 0])
    real = np.array([1, -1, -1, -1, 1])  # last has pred=0 -> excluded
    # matches over non-zero pred: 1,-1,-1 vs 1,-1,-1: idx 0 match, idx 1 match,
    # idx 2 no match, idx 3 match -> 3 of 4
    assert hit_rate(pred, real) == pytest.approx(0.75)


def test_sharpe_approx_zero_for_constant_returns() -> None:
    assert np.isnan(sharpe_approx([0.01] * 100))


def test_sharpe_approx_positive_for_positive_drift() -> None:
    rng = np.random.default_rng(2)
    returns = 0.001 + rng.normal(scale=0.01, size=252)
    s = sharpe_approx(returns)
    assert s > 0
