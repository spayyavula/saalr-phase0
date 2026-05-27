"""Locked specifications for the Saalr Phase 0 signal-validation experiment.

These values mirror ``pre-registration.md`` (filed 2026-05-27, public gist
at https://gist.github.com/spayyavula/2c42dceaf372a839ddc1fb7ceb9e4a04).
Editing any value breaks ``tests/test_locked_spec.py`` by design.

To legitimately update (e.g., the Week 3 addendum that locks the LSTM
and GARCH model specs and the holdout SHA-256 hash):

1. Add or change the value here.
2. Recompute the fingerprint and update the pin in ``tests/test_locked_spec.py``.
3. Add a decision-log entry under ``decisions/`` describing the change and
   the public addendum that accompanies it (per pre-registration.md §10
   on multiple-comparisons discipline and §13 on publication commitments).

Items explicitly deferred to a Week 3 addendum and therefore absent here:

- LSTM architecture, training window, and hyperparameters (§9b).
- GARCH(1,1) calibration window and refit cadence (§9b).
- ``data/holdout/`` SHA-256 hash (§4).

The fingerprint protects what is locked *today*. The addendum will add new
fields and bump the fingerprint via the documented update process above.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True)
class LockedSpec:
    # --- §3 Data sources ----------------------------------------------------
    news_window_start: str = "2024-01-01"
    news_window_end: str = "2026-04-30"
    spy_top10_holdings: tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "BRK.B", "AVGO", "JPM",
    )
    news_fields_used: tuple[str, ...] = ("title", "description")
    news_timestamp_field: str = "published_utc"
    options_symbol: str = "SPY"
    options_contract_type: str = "weekly"
    options_strike_window: int = 2
    options_granularity_minutes: int = 1
    options_quote_basis: str = "mid"
    options_max_spread_pct_of_mid: float = 0.10
    underlying_symbol: str = "SPY"
    underlying_granularity_minutes: int = 1
    risk_free_rate_source: str = "FRED:DGS3MO"

    # --- §4 Sample construction --------------------------------------------
    event_window_open_et: str = "09:35:00"
    event_window_close_et: str = "15:30:00"
    forward_horizon_minutes: int = 30
    train_start: str = "2024-01-01"
    train_end: str = "2025-09-30"
    validation_start: str = "2025-10-01"
    validation_end: str = "2026-02-15"
    holdout_start: str = "2026-02-16"
    holdout_end: str = "2026-04-30"

    # --- §5 Feature engineering --------------------------------------------
    finbert_model_id: str = "ProsusAI/finbert"
    finbert_max_tokens: int = 512
    finbert_score_formula: str = "P_positive - P_negative"
    aggregation_lookback_hours: int = 4
    ewma_halflife_minutes: int = 30
    iv_inversion_method: str = "black_scholes"
    iv_quote_basis: str = "call_put_mid"
    expiry_rule: str = "nearest_weekly_friday"
    strike_rule: str = "atm_closest_listed"

    # --- §6 Primary outcome metric -----------------------------------------
    primary_metric: str = "spearman_rank_correlation"

    # --- §7 Baselines ------------------------------------------------------
    baseline_definitions: tuple[tuple[str, str], ...] = (
        ("B1_persistence", "forward_iv_change = prior_30min_iv_change"),
        ("B2_random", "sentiment_scores_shuffled_across_timestamps"),
        ("B3_prior_day_same_time", "yesterday_sentiment_at_same_time_of_day"),
    )
    baseline_margin_multiplier: float = 1.5

    # --- §8 Statistical tests ----------------------------------------------
    min_holdout_events: int = 1000
    bootstrap_resamples: int = 10000
    confidence_level: float = 0.95
    significance_pvalue: float = 0.01
    test_sidedness: str = "two_sided"
    subperiod_split: str = "equal_halves_by_time"

    # --- §9 Primary success criteria (S1-S4) -------------------------------
    s1_min_holdout_ic: float = 0.05
    s2_max_holdout_pvalue: float = 0.01
    s3_baseline_margin_multiplier: float = 1.5
    s4_subperiod_sign_consistency_required: bool = True
    grey_zone_lower_exclusive: float = 0.03
    grey_zone_upper_exclusive: float = 0.05

    # --- §9b Variant B combined-signal spec --------------------------------
    # FinBERT thresholds are on the [-100, +100] rescaled score that matches
    # the production thresholds in optionsacademy.ai. The §5 raw score
    # (P_pos - P_neg in [-1, +1]) is rescaled by ×100 before applying these.
    finbert_up_threshold: float = 50.0
    finbert_down_threshold: float = -50.0
    lstm_up_return_threshold: float = 0.005
    lstm_down_return_threshold: float = -0.005
    garch_pop_gate: float = 0.55
    garch_monte_carlo_paths: int = 10000
    garch_path_model: str = "gbm_risk_neutral_drift_zero"
    entry_slippage_pct: float = 0.02
    exit_slippage_pct: float = 0.02
    max_concurrent_positions: int = 1
    days_to_expiry_assumed: int = 7

    # --- §9b Variant B success criteria (E1-E5) ----------------------------
    e1_min_holdout_sharpe: float = 0.5
    e2_min_holdout_hitrate: float = 0.55
    e3_min_total_pnl_net_of_slippage: float = 0.0
    e4_subperiod_sign_consistency_required: bool = True
    e5_random_shuffle_sharpe_margin_multiplier: float = 1.5

    # --- §9b Robustness variants (exploratory only) ------------------------
    robustness_variants: tuple[tuple[str, str], ...] = (
        ("A", "finbert_only_directional_no_lstm_no_pop_gate"),
        ("C_uni", "finbert_lstm_arima_prophet_unanimous_pop_gate"),
        ("C_super", "finbert_lstm_arima_prophet_supermajority_75pct_pop_gate"),
        ("D", "garch_vol_forecast_accuracy_vs_persistence_rmse_no_trades"),
    )

    # --- §12 Stopping rule -------------------------------------------------
    holdout_executions_permitted: int = 1

    # --- §13 Publication commitment ----------------------------------------
    publication_deadline_days_after_holdout: int = 14

    def fingerprint(self) -> str:
        """Deterministic sha256 over the canonical JSON of all locked fields."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SPEC: Final[LockedSpec] = LockedSpec()
