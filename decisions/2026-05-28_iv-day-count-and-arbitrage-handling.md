# 2026-05-28 — IV-inversion day-count convention and arbitrage handling

Resolves two implementation questions in `src/iv_surface.py` that the
pre-registration does not lock:

1. The day-count convention for the time-to-expiry input `T`.
2. The semantics for prices that violate Black-Scholes no-arbitrage
   bounds (sub-intrinsic, supra-underlying, etc.).

## Decision

### Day-count: ACT/365 with expiry at SPEC.event_window_close_et (15:30 ET)

`time_to_expiry_years(sample_time_utc, expiry)` computes:

```
T = (expiry_close_utc - sample_time_utc).total_seconds()
    / (365.0 * 86400.0)
```

where `expiry_close_utc` is the **15:30 America/New_York moment on the
expiry date, converted to UTC via `zoneinfo`** (so DST is handled
correctly year-round). Sub-day resolution; tz-naive samples are treated
as UTC.

### Arbitrage handling: out-of-band prices return `NaN`

A price outside the no-arbitrage band returns `NaN` instead of raising.
For European options the bands are:

| leg | lower | upper |
| --- | --- | --- |
| call | `max(0, S - K·e^{-rT})` | `S` |
| put  | `max(0, K·e^{-rT} - S)` | `K·e^{-rT}` |

`call_put_mid_iv` propagates `NaN`: if either leg is out-of-band, the
event's per-event IV is `NaN` and downstream sample-construction is
expected to drop the row (and account for it in the coverage-failure
ledger).

## Context

`pre-registration.md` §5 locks `iv_inversion_method = "black_scholes"`,
`iv_quote_basis = "call_put_mid"`, `expiry_rule = "nearest_weekly_friday"`,
and `strike_rule = "atm_closest_listed"`. It does **not** lock:

- Whether `T` is calendar-days-to-expiry or trading-days; whether
  expiry is at the open (09:35 ET) or the close (15:30 ET); the
  day-count divisor (252, 360, 365, 365.25).
- What to do when a market mid-quote produces a price below intrinsic
  (which happens on stale quotes, crossed markets, dust prices at deep
  ITM strikes, etc.).

Both choices have to be made before the validation-set IV can be
computed, so we make them now and document them rather than burying
them in a constant.

## Alternatives considered

### Day-count

1. **ACT/252 (trading-days).** Common in equity-vol literature.
   Rejected: SPY weekly options trade and settle on calendar-day
   tenors; using 252 implicitly assumes the weekend has zero variance,
   which biases short-dated IV downward by ~10–15 % vs ACT/365 and
   matters more for 1–3 day-to-expiry contracts (which are common in
   our sample) than for monthlies.
2. **ACT/365 with expiry at 16:00 ET (SPX-style settlement close).**
   SPY options actually cease trading at 16:00 ET, but cash settlement
   is at 15:30 ET market close for AM-settled European indexes and at
   the closing print for American equity options. SPY weeklies are
   American-style on the equity. Using `SPEC.event_window_close_et`
   (15:30 ET) aligns the IV-inversion clock with the locked event
   window, so the same point-in-time anchor is used for both
   "when is this event" and "what is T to expiry."
3. **ACT/365 with expiry at 15:30 ET (chosen).** Matches
   `SPEC.event_window_close_et` so the IV inversion is consistent
   with the event-window definition without introducing a new locked
   constant. Sub-day resolution captures the 1-minute-granularity
   pre-reg locks.
4. **ACT/365.25 (calendar-year average including leap days).**
   Cosmetic difference in 4th decimal place of IV; not worth the
   irregularity.

### Arbitrage handling

1. **Raise on out-of-band price.** Caller-friendly for a one-shot
   script; hostile for batch processing where a few percent of stale
   quotes would crash the whole eval run.
2. **Clamp to band edge and invert.** Returns a finite IV but
   silently corrupts the per-event score with a fabricated number;
   would inflate IC estimates in the wrong direction.
3. **Return NaN; let the caller count and drop (chosen).** Matches
   the Q1 coverage-failure pattern from
   `2026-05-27_q1-strike-window-and-q2-mid-quote.md` — a structurally
   logged drop rather than a silent imputation. Downstream
   sample-construction is expected to maintain a
   `iv_inversion_failures.parquet` side-file with the count and reason
   per event so the validation-set reviewer can see how many events
   were lost to feed quality vs. real market dislocation.

## Rationale

Both decisions push reporting up the pipeline. The IV-inversion
function is now a pure math primitive: given valid inputs it returns a
finite IV to machine precision (45 round-trip tests cover this); given
out-of-band inputs it returns `NaN` with no side effects. Sample
construction owns the counting, the dropping, and the audit trail.

`SPEC` does not need a new field for either decision. Both are
implementation details that don't appear in the locked Spearman rank
correlation — IC ranks are invariant to a monotone transformation of
IV, so the choice of day-count divisor scales every IV uniformly and
leaves the IC unchanged. The arbitrage NaN policy *does* affect the
sample size, but a transparent count is the right place to handle
that, not a tweakable threshold.

## Reversible?

Yes on both pieces. The day-count divisor is one constant
(`_DAYS_PER_YEAR = 365.0`) and the expiry-close clock is one line in
`_expiry_close_utc`. Switching to ACT/252 or to 16:00 ET would not
change any persisted artifact — IVs are derived, not stored as a
locked field. The arbitrage NaN behavior could be swapped for an
exception or a clamp via the function body; the test suite codifies
the current contract so a change is loud.
