"""The three locked baselines from §7 of the pre-registration.

Per §7, the candidate model's holdout IC must exceed the strongest
baseline IC by ``SPEC.baseline_margin_multiplier`` (1.5×). All three
baselines run on the same matched-events frame as the candidate.

Baselines:

- **B1 persistence** — predicts forward IV change = prior 30-min IV change.
  Trivial autocorrelation; if the candidate signal can't beat this, it's
  rediscovering volatility clustering, not extracting news content.
- **B2 random** — sentiment scores shuffled across timestamps. Expected
  IC ≈ 0 by construction; protects against multiple-comparisons hindsight
  bias (e.g. "well at least it beat random").
- **B3 prior-day-same-time** — yesterday's sentiment at the same clock
  minute. Cheap intra-day-pattern baseline; if the candidate can't beat
  it, it's rediscovering time-of-day effects, not news.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from src.locked_spec import SPEC
from src.metrics import bootstrap_ic_ci, ICResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineResults:
    """Holds the IC of every locked baseline. Field names match the
    pre-registration's IDs so the writeup can cite them directly."""

    b1_persistence: ICResult
    b2_random: ICResult
    b3_prior_day_same_time: ICResult

    @property
    def strongest_ic(self) -> float:
        """Strongest baseline IC by absolute value. The candidate's IC must
        exceed ``baseline_margin_multiplier × strongest_ic`` per §9 S3."""
        ics = [
            self.b1_persistence.ic,
            self.b2_random.ic,
            self.b3_prior_day_same_time.ic,
        ]
        return max((abs(ic) for ic in ics if not np.isnan(ic)), default=float("nan"))


def b1_persistence(events: pd.DataFrame) -> ICResult:
    """B1: predict forward IV change with the prior 30-min IV change.

    ``events`` columns required:

    - ``forward_iv_change`` — realized IV(t+30) − IV(t) (the target)
    - ``prior_iv_change`` — realized IV(t) − IV(t−30) (the predictor)
    """
    return bootstrap_ic_ci(
        events["prior_iv_change"], events["forward_iv_change"],
        seed=SPEC_RANDOM_SEED_B1,
    )


def b2_random(events: pd.DataFrame, *, seed: int = 42) -> ICResult:
    """B2: shuffle the candidate signal across timestamps, recompute IC.

    Single shuffle with a fixed seed for reproducibility. The pre-reg
    treats this as "trivial baseline" not as a Monte-Carlo null distribution.
    """
    rng = np.random.default_rng(seed)
    shuffled = events["signal"].to_numpy().copy()
    rng.shuffle(shuffled)
    return bootstrap_ic_ci(shuffled, events["forward_iv_change"], seed=seed)


def b3_prior_day_same_time(events: pd.DataFrame) -> ICResult:
    """B3: at each event time t on day d, use the candidate signal at the
    same clock minute on day d−1. Events where no prior-day match exists
    are dropped (not back-filled further)."""
    if not isinstance(events.index, pd.DatetimeIndex):
        df = events.set_index(pd.DatetimeIndex(events["timestamp"]))
    else:
        df = events

    target_signal = df["signal"].copy()
    target_signal.index = target_signal.index + timedelta(days=1)
    aligned = target_signal.reindex(df.index, method="nearest", tolerance=timedelta(minutes=1))
    mask = ~aligned.isna()
    if not mask.any():
        return bootstrap_ic_ci([], [])
    return bootstrap_ic_ci(
        aligned[mask].to_numpy(),
        df["forward_iv_change"][mask].to_numpy(),
        seed=SPEC_RANDOM_SEED_B3,
    )


def compute_all_baselines(events: pd.DataFrame) -> BaselineResults:
    return BaselineResults(
        b1_persistence=b1_persistence(events),
        b2_random=b2_random(events),
        b3_prior_day_same_time=b3_prior_day_same_time(events),
    )


# Fixed per-baseline seeds so bootstrap CIs are reproducible across runs.
# Different seeds per baseline avoid spurious cross-baseline correlation
# in the bootstrap resampling indexes.
SPEC_RANDOM_SEED_B1 = 1001
SPEC_RANDOM_SEED_B3 = 1003
