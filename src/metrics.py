"""Locked statistical metrics for the §§6-8 evaluation.

Functions here are the *only* metric implementations the harness uses.
Adding a new metric is a decision-log event — drift from one accidental
helper to "let's just look at Pearson too" is exactly what the
pre-registration discipline exists to prevent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.locked_spec import SPEC

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ICResult:
    """Result of a Spearman IC computation with bootstrap CI."""

    n: int
    ic: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence_level: float
    bootstrap_resamples: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < SPEC.significance_pvalue


def _drop_nan(signal: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~(np.isnan(signal) | np.isnan(target))
    return signal[mask], target[mask]


def spearman_ic(signal, target) -> tuple[float, float]:
    """Two-sided Spearman rank correlation between ``signal`` and ``target``.

    Returns ``(ic, p_value)``. The pre-reg locks Spearman (not Pearson) per
    §6 because IV-change distributions are heavy-tailed and the
    relationship is not assumed linear.
    """
    signal_arr = np.asarray(signal, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    signal_clean, target_clean = _drop_nan(signal_arr, target_arr)
    if signal_clean.size < 2:
        return (float("nan"), float("nan"))
    result = stats.spearmanr(signal_clean, target_clean, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def bootstrap_ic_ci(
    signal,
    target,
    *,
    resamples: int | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
) -> ICResult:
    """Bootstrap a confidence interval for the Spearman IC.

    Defaults pull from SPEC: ``bootstrap_resamples`` and ``confidence_level``.
    A fixed ``seed`` makes the result reproducible — pass the same seed
    you'll publish in the writeup.
    """
    resamples = resamples or SPEC.bootstrap_resamples
    confidence_level = confidence_level or SPEC.confidence_level

    signal_arr = np.asarray(signal, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    signal_clean, target_clean = _drop_nan(signal_arr, target_arr)
    n = signal_clean.size
    if n < 2:
        return ICResult(
            n=n,
            ic=float("nan"),
            p_value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=confidence_level,
            bootstrap_resamples=resamples,
        )

    point_ic, point_p = spearman_ic(signal_clean, target_clean)

    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        result = stats.spearmanr(signal_clean[idx], target_clean[idx])
        boot[i] = float(result.statistic)

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])

    return ICResult(
        n=n,
        ic=point_ic,
        p_value=point_p,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        bootstrap_resamples=resamples,
    )


def hit_rate(direction_pred, direction_realized) -> float:
    """Fraction of events where the predicted direction matched the realized
    direction. Used by Variant B success criterion E2."""
    pred = np.asarray(direction_pred)
    real = np.asarray(direction_realized)
    mask = (pred != 0) & ~np.isnan(real)
    if not mask.any():
        return float("nan")
    matches = (np.sign(pred[mask]) == np.sign(real[mask])).sum()
    return float(matches) / int(mask.sum())


def sharpe_approx(returns, *, periods_per_year: int = 252) -> float:
    """Annualized Sharpe-approx used by Variant B criteria E1 and E5.

    "Sharpe-approx" because the pre-reg §9b's execution model is event-driven
    (variable holding periods), not bar-aligned. We treat each closed
    position's net return as one sample, annualize with ``periods_per_year``.
    The result is comparable across variants but is *not* a true Sharpe.
    """
    returns_arr = np.asarray(returns, dtype=float)
    returns_clean = returns_arr[~np.isnan(returns_arr)]
    if returns_clean.size < 2:
        return float("nan")
    std = float(returns_clean.std(ddof=1))
    # Tolerance instead of `== 0.0`: numpy's std of a constant array
    # rounds to ~1e-18, not literally 0, so the strict-equality guard
    # let through "constant returns" and returned an astronomical Sharpe
    # (mean / float-noise * sqrt(252)).
    if std < 1e-12:
        return float("nan")
    mean = float(returns_clean.mean())
    return mean / std * np.sqrt(periods_per_year)
