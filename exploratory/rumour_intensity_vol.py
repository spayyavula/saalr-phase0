"""EXPLORATORY (pre-registration.md §10 — cannot claim success).

Tests the hypothesis: "does volatility increase with a rumour?" — i.e. does
news INTENSITY (regardless of sentiment direction) coincide with IV rising?

Two intensity measures per event at time t:
  - count4h  = number of articles in [t - 4h, t]  (news arrival / clustering)
  - magsig   = |aggregated_signal|                (signed-EWMA magnitude)

Correlated (Spearman) against three IV quantities:
  - prior_iv_change  = IV(t) - IV(t-30)   -> the ramp INTO the event
  - forward_iv_change= IV(t+30) - IV(t)   -> the move AFTER (crush?)
  - iv_t             = the IV level at t

Expected if "vol rises with a rumour, then crushes": intensity correlates
POSITIVELY with prior_iv_change (ramp), NEGATIVELY with forward_iv_change
(crush). All correlations are contemporaneous/descriptive, NOT predictive.

ALL correlations are printed (no cherry-picking) — running several inflates
false-positive risk, so treat any result as hypothesis-generating only.
Train/validation splits only; never the holdout.

    python exploratory/rumour_intensity_vol.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.compute_iv import compute_iv_for_events
from src.locked_spec import SPEC

SPLITS = ("train", "validation")  # holdout deliberately excluded
_LOOKBACK_NS = SPEC.aggregation_lookback_hours * 3600 * 1_000_000_000


def _load(split: str, source: str) -> pd.DataFrame:
    d = Path("data") / split / source
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _count_in_lookback(event_ns: np.ndarray, article_ns: np.ndarray) -> np.ndarray:
    """For each event time, count articles in [t - lookback, t]."""
    article_ns = np.sort(article_ns)
    hi = np.searchsorted(article_ns, event_ns, side="right")
    lo = np.searchsorted(article_ns, event_ns - _LOOKBACK_NS, side="left")
    return hi - lo


def _spearman(x: pd.Series, y: pd.Series) -> str:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return f"rho=   n/a   p=  n/a   n={int(mask.sum())}"
    r = stats.spearmanr(x[mask], y[mask])
    return f"rho={r.statistic:+.4f}  p={r.pvalue:.3g}  n={int(mask.sum())}"


def main() -> int:
    print("=" * 72)
    print("EXPLORATORY (pre-reg §10) — cannot claim success. Train/validation only.")
    print("Hypothesis: does IV rise with rumour intensity (then crush after)?")
    print("=" * 72)
    for split in SPLITS:
        events = _load(split, "events")
        if events.empty:
            print(f"\n[{split}] no events on disk yet — skipping")
            continue
        sentiment = _load(split, "sentiment")
        iv = compute_iv_for_events(events).copy()

        iv["magsig"] = iv["aggregated_signal"].abs()
        if not sentiment.empty:
            ev_ns = pd.to_datetime(iv["timestamp"], utc=True).astype("int64").to_numpy()
            art_ns = pd.to_datetime(sentiment["timestamp"], utc=True).astype("int64").to_numpy()
            iv["count4h"] = _count_in_lookback(ev_ns, art_ns)
        else:
            iv["count4h"] = np.nan

        print(f"\n[{split}]  events={len(iv)}")
        for intensity in ("count4h", "magsig"):
            print(f"  intensity = {intensity}")
            print(f"    vs prior_iv_change   (ramp into t):   {_spearman(iv[intensity], iv['prior_iv_change'])}")
            print(f"    vs forward_iv_change (after t):       {_spearman(iv[intensity], iv['forward_iv_change'])}")
            print(f"    vs iv_t              (level at t):    {_spearman(iv[intensity], iv['iv_t'])}")

    print("\nContemporaneous/descriptive, NOT predictive. Exploratory only —")
    print("hypothesis-generating; the locked test is the Week-7 holdout primary IC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
