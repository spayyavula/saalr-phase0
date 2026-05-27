# SAALR — Phase 0 Signal Validation: Pre-Registration

## Filing information

- **Filed:** 2026-05-27
- **Author:** Sreekanth Payyavula
- **Affiliation:** Saalr (Founder & CEO)
- **Code repository:** https://github.com/spayyavula/saalr-phase0
- **Pre-registration platform:** GitHub Gist (snapshot at filing time)

---

## 1. Statement of intent

This document publicly pre-registers an experiment to test whether news-derived sentiment scores (computed with FinBERT) carry statistically significant, economically meaningful predictive information about near-term changes in SPY implied volatility.

The experiment is a backtest on historical data. The discipline is:

1. The hypothesis, data sources, methodology, and success/failure thresholds are fixed **before** any modelling work begins.
2. The holdout data set is sealed (committed by hash) and not examined until the final test in week 7.
3. The results — whether positive, negative, or inconclusive — will be published publicly within 14 days of the final test.
4. If the experiment fails, Saalr publicly retires the claim that its ML stack produces directional predictive signal, and repositions product messaging accordingly.

The purpose of pre-registering is to remove the degrees of freedom that ordinarily lets a researcher rationalize a negative result into a positive one. By publishing the thresholds before seeing the holdout, we commit to honoring whatever the data says.

---

## 2. Hypothesis

### Primary hypothesis (H1)
FinBERT-derived news sentiment scores have statistically significant predictive lift on 30-minute forward implied-volatility changes in SPY ATM weekly options, on out-of-sample data.

### Null hypothesis (H0)
FinBERT sentiment scores have zero predictive lift (Spearman IC = 0) on 30-minute forward SPY ATM IV changes, after accounting for trivial baselines.

### Statistical alternatives
Two-sided test. Either direction of correlation counts as evidence against H0; the sign and stability of the correlation are diagnostic.

---

## 3. Data sources

### News
- **Provider:** Massive News API (or equivalent provider if Massive coverage is inadequate; switch documented if necessary in week 2)
- **Window:** 2024-01-01 → 2026-04-30
- **Filter:** articles tagged with SPY or any of its top-10 holdings by weight (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, BRK.B, AVGO, JPM as of filing)
- **Field used:** `title` + `description`
- **Timestamp used:** `published_utc` field; treated as the moment the news became publicly knowable (acknowledging that scoops may have been known earlier — see §11 limitation)

### Options
- **Provider:** Massive Options Advanced
- **Symbol:** SPY weekly options
- **Strikes:** ATM ±2 strikes only
- **Granularity:** 1-minute bars (mid-quote)
- **Window:** same as news window
- **Filter:** exclude any (strike, expiry) where the 1-min mid-quote bid-ask spread exceeds 10% of the mid

### Underlying
- **Provider:** Massive Stocks Aggregates
- **Symbol:** SPY
- **Granularity:** 1-minute bars
- **Use:** spot price for Black-Scholes IV inversion

### Risk-free rate
- 3-month Treasury bill yield, daily, from FRED (`DGS3MO`)

---

## 4. Sample construction

### Event definition
Each news article with `published_utc` in [09:35:00 ET, 15:30:00 ET) on a US trading day is a candidate event. (Carve-out of first/last 5 minutes excludes opening/closing-auction noise.)

### Matching
For each event, identify the SPY ATM weekly option contract at `published_utc`. Record:
- IV at `published_utc` (computed via Black-Scholes inversion from call+put mid)
- IV at `published_utc + 30 min` (same methodology)
- Forward IV change = IV(t+30) − IV(t)

### Sample size estimate
Approximately 30,000 events after filtering. Will be confirmed and reported in week 3.

### Train / validation / holdout split (time-based, NOT random)
- **Train:** 2024-01-01 → 2025-09-30 (~70%)
- **Validation:** 2025-10-01 → 2026-02-15 (~15%)
- **Holdout:** 2026-02-16 → 2026-04-30 (~15%)

**Holdout file SHA-256 hash:** [to be filled in at the end of Week 3, committed to repo immediately, and published in an addendum to this pre-registration]

---

## 5. Feature engineering (LOCKED)

### Sentiment score per article
- Model: `ProsusAI/finbert` from Hugging Face (specific version pinned in repo `requirements.txt`)
- Input: concatenation of `title` and `description`, truncated to 512 tokens
- Output: probabilities (P_positive, P_negative, P_neutral)
- **Signed score:** S = P_positive − P_negative

### Aggregation per timestamp
- Signal at time t = exponentially-weighted sum of all article scores within previous 4 hours
- Weight function: exp(−(t − published_utc) / λ) where λ = 30 minutes (half-life)
- Articles older than 4 hours: dropped

### Implied volatility
- ATM mid-IV via Black-Scholes inversion using call + put mid-prices
- Maturity: nearest weekly expiry (Friday)
- Strike: ATM = closest listed strike to current spot
- Forward IV change at t: IV(t+30 min) − IV(t)

---

## 6. Primary outcome metric

**Spearman rank correlation** between aggregated FinBERT signal at time t and forward 30-minute IV change at time t, computed on the **holdout** sample.

Choice of Spearman (vs. Pearson) is because IV change distributions are heavy-tailed and the relationship is not assumed linear.

---

## 7. Baselines (must beat at 1.5× margin)

Computed on the training set in week 4 and locked:

| # | Baseline | Definition |
|---|---|---|
| B1 | Persistence | Forward IV change predicted = prior 30-min IV change |
| B2 | Random | Sentiment scores shuffled randomly across timestamps |
| B3 | Prior-day-same-time sentiment | Yesterday's sentiment at the same time-of-day used in place of today's |

**The strongest baseline IC × 1.5** is the threshold the primary model must exceed on the holdout.

---

## 8. Statistical tests

- **Primary test:** Spearman ρ between signal and forward IV change on holdout
- **Sample size requirement:** n ≥ 1,000 events on the holdout
- **Confidence interval:** 95% bootstrap CI (10,000 resamples)
- **Significance threshold:** p < 0.01, two-sided
- **Sub-period stability:** holdout split into two equal halves by time; sign of IC must agree in both halves

---

## 9. Success / failure criteria

Pre-locked. All four success conditions must hold simultaneously for the experiment to be declared successful:

| # | Condition |
|---|---|
| S1 | Holdout Spearman IC ≥ 0.05 |
| S2 | Holdout p-value < 0.01 |
| S3 | Holdout IC ≥ 1.5 × strongest baseline IC |
| S4 | Sign of holdout IC consistent across both sub-periods |

If any condition fails, the experiment is declared a failure. **The grey zone (IC between 0.03 and 0.05) is explicitly a failure, not a success.**

---

## 9b. Secondary economic-lift analysis (pre-registered, locked)

In addition to the primary IC test in §9, we pre-register a single **secondary** analysis that asks the related but distinct question: *if the combined signal had been traded, would it have produced economic lift?*

This is a different hypothesis from the primary IC test. They are evaluated and reported together; **neither validates the other**, and the secondary cannot, on its own, justify a claim of validation success beyond what §9 establishes.

### Locked variant — "Variant B"

**Combined signal:**

1. **FinBERT direction:** UP if score > +50 AND trend = rising; DOWN if score < −50 AND trend = falling; FLAT otherwise. (Matches the production thresholds in `optionsacademy.ai` post the fix/sentiment-descriptive-language PR.)
2. **LSTM direction:** UP if predicted forward return > +0.5%; DOWN if < −0.5%; FLAT otherwise. The LSTM model specification (architecture, training window, hyperparameters) will be locked in a Week 3 addendum to this pre-registration, before the holdout file hash is committed.
3. **Direction-agreement gate:** both readings must be non-FLAT and identical. Otherwise → HOLD.
4. **GARCH POP gate:** GARCH(1,1) 1-step-ahead annualized vol forecast → Monte-Carlo (10,000 paths, GBM, risk-neutral drift = 0) → Probability that ATM weekly settles ITM in the direction of the agreed signal. If POP < 0.55 → HOLD. The GARCH model specification will also be locked in the same Week 3 addendum.
5. Otherwise: simulate BUY_CALL (UP) or BUY_PUT (DOWN) on the ATM weekly contract.

**Execution model:**
- Entry price = mid × (1 + 0.02). Exit price = mid × (1 − 0.02). Symmetric 2% slippage.
- Exit on next signal-change event (or end of holdout window).
- Single-position model: at most one contract held at any time.
- Days to expiry assumed = 7 (nearest weekly Friday expiry).

### Success criteria for Variant B (all five must hold)

| # | Condition |
|---|---|
| E1 | Holdout Sharpe-approx ≥ 0.5 |
| E2 | Holdout hit-rate ≥ 55% |
| E3 | Holdout total cumulative P&L > 0 net of slippage |
| E4 | Sign of total return consistent across both sub-period halves of the holdout |
| E5 | Variant B Sharpe ≥ 1.5 × Sharpe of the random-shuffled-FinBERT baseline |

If any condition fails, the secondary economic-lift hypothesis is declared failed. **The primary IC hypothesis (§9) is evaluated entirely separately under its own criteria.**

### Robustness variants (pre-registered, exploratory, cannot claim success)

The following variants are also run on the same holdout. They are **exploratory robustness checks** — reported in the public writeup, but explicitly **cannot, individually or collectively, justify a success claim** beyond what §9 and §9b above establish.

| ID | Variant |
|---|---|
| A | FinBERT-only directional (no LSTM gate, no POP gate) |
| C-uni | FinBERT + LSTM + ARIMA + Prophet, unanimous agreement, POP gate |
| C-super | Same 4 factors, supermajority (≥ 75% agreement), POP gate |
| D | GARCH 1-step-ahead vol forecast accuracy vs persistence baseline (RMSE; no trades) |

### Correlation caveat (acknowledged in advance, cannot be retro-cited)

LSTM, ARIMA, and Prophet all learn from past underlying returns. Their votes are **not** independent. If Variant C-uni or C-super materially outperforms Variant B on the holdout, the most plausible explanation is overfitting to correlated noise from the three price-history-based models. The writeup will report B, C-uni, and C-super side-by-side and explicitly address this gap.

---

## 10. Multiple-comparisons discipline

- **Two pre-registered hypotheses, evaluated independently:** the primary IC test (§9) and the secondary economic-lift test (§9b, Variant B). Each is a one-shot evaluation on the holdout.
- The four robustness variants enumerated in §9b (A, C-uni, C-super, D) are **exploratory only**. They are reported in the writeup but cannot, individually or collectively, justify a success claim.
- Any analysis on additional underlyings (QQQ, individual stocks), additional time horizons (15-min, 60-min, daily), or alternative sentiment aggregations beyond what is locked above is **exploratory**, labeled as such in the writeup, and explicitly cannot be used to argue success in this pre-registration.
- The holdout is not used to evaluate exploratory analyses other than the four enumerated robustness variants.

---

## 11. Known limitations of this design

These are documented now so they cannot be retroactively cited as reasons to discount a negative result:

1. **Timestamp reliability.** `published_utc` is the article-publication time, not necessarily the moment the news became publicly known. Scoops may have moved markets minutes-to-hours before the article's published timestamp. We accept this as an inherent limitation of cheap news data.
2. **Survivorship.** Massive's news archive may have retroactively removed articles. We do not have a way to detect or correct for this.
3. **FinBERT calibration.** FinBERT was trained on financial news from a window predating our test window. Its calibration may have drifted.
4. **Regime dependence.** The 2024-2026 period includes specific macro regimes (rate cycles, geopolitical events). Findings may not generalize.
5. **Single-underlying focus.** SPY is a market-aggregate index; results may not transfer to single-stock options.

We acknowledge these limitations openly and they do not change the success/failure thresholds.

---

## 12. Stopping rule

The holdout test is executed exactly once, in Week 7 of the timeline. The result of that single execution is final. No additional adjustment to the model, signal, or thresholds is permitted after the holdout test begins, regardless of outcome.

---

## 13. Publication commitment

- Results (success, failure, or inconclusive) will be published publicly within **14 days** of the holdout test.
- Publication will include:
  - Link to this pre-registration
  - Link to the code repository (public on GitHub)
  - All metrics with confidence intervals
  - All hyperparameter choices and decisions
  - The decision log from the experiment
- **If the primary IC hypothesis (§9) fails:** Saalr's product positioning will be updated to remove any claim that FinBERT sentiment carries predictive information about forward IV. The deck's ML-stack slide will be updated to reflect the negative result.
- **If the secondary economic-lift hypothesis (§9b, Variant B) fails:** any product claim that the combined signal produces tradable alpha is retired. The descriptive sentiment surface, vol-surface, Greeks aggregation, and educational content surfaces are unaffected.
- **If both fail:** the product is repositioned entirely around vol surface + Greeks + education, with no predictive or alpha claim of any kind.

---

## 14. Conflicts of interest

The author is the founder and CEO of Saalr, and Saalr's commercial positioning includes an "honest ML" engine that depends in part on this experiment producing positive results. This is the conflict that the pre-registration discipline exists to neutralize: the thresholds are public, the holdout is hashed, and the publication commitment is binding.

---

## 15. Sign-off

I, **Sreekanth Payyavula**, founder of Saalr, commit to executing this experiment according to the methodology above, to publishing the result publicly within 14 days of the holdout test regardless of outcome, and to honoring the retire-on-fail commitments in §13 for both the primary IC hypothesis (§9) and the secondary economic-lift hypothesis (§9b, Variant B).

**Date:** 2026-05-27

**Signed:** Sreekanth Payyavula
