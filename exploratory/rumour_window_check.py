"""EXPLORATORY (pre-registration.md §10 — cannot claim success).

Compares the locked post-news relationship against the anticipation/run-up
relationship, to probe the "buy the rumour, sell the news" interpretation
(decisions/2026-05-28_buy-rumour-sell-news-interpretation.md):

    forward:  Spearman(signal, forward_iv_change)   [= IV(t+30) - IV(t)]
    prior:    Spearman(signal, prior_iv_change)      [= IV(t)   - IV(t-30)]

If the informative move lives in the rumour phase, the PRIOR correlation
would be the stronger of the two.

Runs on the TRAIN and VALIDATION splits only — NEVER the holdout. Reads
match_events output (data/{split}/events/*.parquet) and applies the frozen
IV inversion in-process, so it works on whatever months have completed.

    python exploratory/rumour_window_check.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from src.compute_iv import compute_iv_for_events

SPLITS = ("train", "validation")  # holdout deliberately excluded


def _load_events(split: str) -> pd.DataFrame:
    d = Path("data") / split / "events"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _spearman(x, y) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan"), float("nan"), int(mask.sum())
    r = stats.spearmanr(x[mask], y[mask])
    return float(r.statistic), float(r.pvalue), int(mask.sum())


def main() -> int:
    print("=" * 70)
    print("EXPLORATORY (pre-reg §10) — cannot claim success. Train/validation only.")
    print("=" * 70)
    for split in SPLITS:
        events = _load_events(split)
        if events.empty:
            print(f"\n[{split}] no events on disk yet — skipping")
            continue
        iv = compute_iv_for_events(events)
        fwd_r, fwd_p, fwd_n = _spearman(iv["signal"], iv["forward_iv_change"])
        pri_r, pri_p, pri_n = _spearman(iv["signal"], iv["prior_iv_change"])
        print(f"\n[{split}]  events={len(iv)}")
        print(f"  forward  Spearman(signal, IV(t+30)-IV(t)):  rho={fwd_r:+.4f}  p={fwd_p:.3g}  n={fwd_n}")
        print(f"  prior    Spearman(signal, IV(t)-IV(t-30)):  rho={pri_r:+.4f}  p={pri_p:.3g}  n={pri_n}")
        if not (pd.isna(fwd_r) or pd.isna(pri_r)):
            stronger = "PRIOR (rumour-phase)" if abs(pri_r) > abs(fwd_r) else "FORWARD (post-news)"
            print(f"  stronger |rho|: {stronger}")
    print("\nReminder: exploratory, on train/validation, often partial months.")
    print("This is diagnostic only — NOT a result. The locked test is the")
    print("one-shot holdout primary IC in Week 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
