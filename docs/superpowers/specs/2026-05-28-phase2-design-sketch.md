# Phase 2 Design Sketch — Cross-Asset, Scheduled-Event Predictive Test

**Date:** 2026-05-28
**Status:** PRELIMINARY SKETCH — not a spec, not pre-registered, not locked.
A parking-lot design to be developed into a real pre-registration **only
after Phase 0 publishes** (its thresholds, universe, and even whether to
run it should depend on what Phase 0's holdout shows).

**Parent:** saalr-phase0 (Phase 0 = SPY-options primary-IC test) and
saalr-forward-test (Phase 1 = frozen out-of-sample replay).

---

## Motivation

The working intuition is **"buy the rumour, sell the news"**: markets
price an event during anticipation, then fade/reverse on confirmation. In
vol terms this is the IV ramp into an event followed by IV crush; in price
terms it's drift into the event followed by reversal.

Two findings from Phase 0 exploratory probes (train-split, §10, diagnostic
only) motivate this:

- Signed sentiment co-moved with the **prior** IV move (ρ≈+0.12, p≈0.005)
  far more than with the locked **forward** IV move (ρ≈+0.001) — i.e. the
  information appears to sit in the anticipation window.
- News **intensity** did *not* cleanly drive vol up; the only durable
  relationship was intense sentiment co-occurring with high IV *level*
  (regime co-movement, not causation).

## The honesty constraint — what is predictively testable

"Buy the rumour, sell the news" has two halves; only one is cleanly
testable with *published-article* data:

- **"Sell the news" (forward / post-signal): testable.** At article time
  t, the signal is knowable; predict the return/IV move over t→t+H.
- **"Buy the rumour" (anticipation): NOT directly testable on generic
  news** — the predictor (the article) does not exist before it
  publishes, so the prior-window co-movement is descriptive, not a
  tradeable predictor.

**Scheduled events resolve this.** Earnings and FOMC have **known dates in
advance**, so the anticipation window is a well-defined run-up to a date
everyone already knows. That makes the "buy the rumour" half predictively
testable — and it is where signal-to-noise is highest. Hence scheduled
events are the centerpiece of Phase 2.

---

## Arm 1 (centerpiece) — scheduled events, predictive anticipation + reversal

Anchored on **known calendars**: per-ticker **earnings** dates and **FOMC**
meeting dates. For each event at date D, with a pre-registered run-up
length k:

| phase | vol target | price target |
| --- | --- | --- |
| **anticipation** ("buy the rumour"), window [D−k, D] | does the pre-event signal predict the IV ramp into D? | does it predict price drift into D? |
| **reversal** ("sell the news"), window [D, D+] | IV crush D→D+ | price reversal D→D+ |

Two sub-arms, kept **separate** because the mechanisms differ:

- **Earnings** — idiosyncratic, per-stock, *many semi-independent* events
  (≈30–50 names × roughly quarterly over the window). The statistically
  strong **workhorse** of Phase 2.
- **FOMC** — market-wide, ~8 meetings/year. Observations across tickers on
  a given FOMC date are **highly correlated**, so effective n is small
  even when crossed with many underlyings. **Underpowered** — reported,
  but flagged as a known limitation; likely *secondary/exploratory* rather
  than confirmatory.

Signal = the **frozen** Phase 0 FinBERT sentiment pipeline, accumulated
over the run-up window.

---

## Arm 2 (broad context) — general-news pooled tests

The cross-asset generalization of Phase 0's forward test:

- **H2a — IV reversal:** signal at t predicts IV(t+H) − IV(t), over the
  **narrow liquid-options subset** (SPY + a few optionable ETFs / mega-caps;
  single-stock weeklies are too illiquid across a wide universe).
- **H2b — directional returns:** signal at t predicts the underlying return
  t→t+H, over the **wide universe** (~30–50 pre-specified liquid stocks +
  major ETFs). Price-only, so it scales cheaply.

Each yields **one pooled-panel confirmatory IC** (see below).

---

## Pooled-panel mechanics (multiplicity control)

Testing "does the signal generalize across a universe" with **one** number,
not N:

- Within each event timestamp, rank signals cross-sectionally (or
  per-name standardize), pool all (event × underlying) rows, compute a
  single Spearman IC per hypothesis/arm.
- Per-name, per-horizon, and per-phase breakdowns are **exploratory only**
  — reported, never success claims.

Phase 2 spans many cells (2 scheduled sub-arms × 2 phases × {vol, price}
plus 2 general hypotheses). The pre-registration MUST designate **one
confirmatory primary**; proposed default:

> **Primary:** earnings × price × anticipation, pooled cross-stock.

Everything else is pre-specified **secondary or exploratory**. Success
criteria mirror Phase 0 §9 (IC threshold, p<0.01 two-sided, beats baselines
×1.5, sub-period sign-stable), applied to the primary pooled IC. Thresholds
set at pre-registration time, informed by Phase 0.

---

## Data + reuse

- **Frozen, reused:** the Phase 0 FinBERT sentiment pipeline (pinned, as in
  the Phase 1 forward repo) and the Black-Scholes IV inversion.
- **New plumbing:** per-ticker **earnings calendar**, **FOMC dates**
  (public; can be enumerated), multi-ticker 1-minute **price** aggregates,
  and a **narrow** options pull for the IV subset, plus the **pooled
  evaluator** and the **run-up windowing** around event dates.
- FinBERT stays **off-the-shelf frozen**; a fine-tuned variant would be its
  own separately pre-registered arm (see the fine-tuning decision note).

---

## Discipline / dependencies

- Separate pre-registration, filed (gist) **before any data look**;
  holdout sealed by hash; **one-shot** evaluation.
- Runs **only after Phase 0 publishes** — no leakage, no preempting the
  Phase 0 or Phase 1 results.
- A Phase 2 result cannot rehabilitate a failed Phase 0; it is reported as
  corroborating/contradicting evidence, scoped to its own hypotheses.

---

## Open questions — resolved at pre-registration time (post-Phase-0)

1. Run-up length k (e.g., 3 / 5 / 10 trading days) and the post-event
   window length.
2. Exact wide-universe list (~30–50 names + which ETFs); the optionable IV
   subset membership.
3. Horizon(s) H for the general-news arm.
4. Confirmatory thresholds (S1–S4 analogues) for the pooled primary.
5. Whether FOMC earns a confirmatory slot or stays exploratory given low
   effective n.
6. Earnings-calendar data source (vendor endpoint vs. a maintained
   calendar) and its survivorship/accuracy properties.
7. Whether a separately-pre-registered **fine-tuned-FinBERT** arm is worth
   including.
8. Whether the (harder) generic-news anticipation question justifies a
   later Phase 3 with pre-publication / story-development data.

---

## Not in scope here

This sketch is **not** an implementation plan and is **not** being built
now. No code is written for Phase 2 until Phase 0 publishes and this sketch
is developed into a locked pre-registration. The next step, when the time
comes, is a fresh brainstorming → spec → plan cycle seeded by this sketch
and by Phase 0's actual results.
