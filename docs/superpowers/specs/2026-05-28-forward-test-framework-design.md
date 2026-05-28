# Forward-Test Framework — Design Spec

**Date:** 2026-05-28
**Status:** Approved (design); pending spec review → implementation plan
**Parent project:** saalr-phase0 (this repo)
**New repo:** `saalr-forward-test`

## Purpose

A separate, pre-registered Phase 1 experiment that answers exactly one
question:

> Does the **locked Phase 0 primary signal** retain its predictive
> Spearman IC on data the model could not have seen
> (2026-05-01 → 2026-07-31)?

The sole motivation is **overfitting confidence**. Phase 0 establishes
whether FinBERT sentiment predicts forward IV change on a fixed
2024-01-01 → 2026-04-30 sample (with a sealed holdout). A forward test
applies that *frozen* methodology to genuinely new data and asks
whether the result persists. Building and pre-registering it **while
still blind to the Phase 0 holdout** is a feature: it removes the
ability to tune the forward test to flatter a backtest we have not yet
seen.

## Scope

**In scope:** the §9 primary IC hypothesis only — FinBERT sentiment →
30-min forward ATM IV change, evaluated with the same S1-S4 criteria.

**Out of scope (deliberately):**
- §9b Variant B trading economics. It depends on the LSTM and GARCH
  specs that are not locked until the Phase 0 Week 3 addendum.
  Forward-testing it now would require inventing those specs, which
  defeats the purpose. Becomes a *second* pre-registered forward test
  after the addendum locks them.
- True-live snapshot capture (`get_snapshot_option` each trading
  minute). Defends only against vendor news-archive survivorship
  (Phase 0 §11 limitation #2); second-order, deferred to a later
  hardening. v1 uses periodic batch pulls.

## Key decisions (from brainstorming)

| # | decision | choice |
| --- | --- | --- |
| 1 | methodology relationship | **Frozen-methodology replay** — verbatim Phase 0 transforms, nothing tunable |
| 2 | hypothesis scope | **Primary IC only** (Variant B deferred) |
| 3 | data collection | **Periodic batch pull** (reuses Phase 0 fetchers; captures already-elapsed May data) |
| 4 | forward window | **3 months: 2026-05-01 → 2026-07-31** (~3,200 events; ~1,600 per S4 sub-period half) |
| 5 | success criteria | **Same S1-S4 absolute thresholds** as §9 (IC≥0.05, p<0.01, 1.5× baseline, sign-consistent) — preserves the blind property |
| 6 | code-freeze mechanism | **Git submodule pinned to the locked Phase 0 commit** |
| 7 | evaluation timing | **One-shot, after Phase 0 publishes** — zero leakage between experiments |

## Methodology freeze (anti-overfit core)

- `saalr-phase0` is a **git submodule pinned to a specific commit** —
  the one carrying the complete locked pipeline with an unchanged
  `SPEC`. (Exact SHA finalized at build time; must be a commit where
  `tests/test_locked_spec.py` passes, i.e. the filing-time fingerprint
  is intact.)
- The forward repo imports the frozen transforms verbatim:
  - `locked_spec.SPEC`
  - `sentiment.score_articles`
  - `options.fetch_contracts_for_month`, `options.fetch_option_mid_quote_at`
  - `match_events.build_events_frame_for_month`
  - `iv_surface.call_put_mid_iv`, `iv_surface.time_to_expiry_years`
  - `evaluate.evaluate_primary`, `baselines.*`
- A **provenance test** asserts:
  1. the submodule HEAD equals the expected pinned SHA, and
  2. `SPEC.fingerprint()` matches the filing-time pin.
  Either drifting fails the suite loudly. This makes "frozen"
  structurally enforced, not a verbal claim.
- The `src.`-prefixed flat layout of saalr-phase0 is bridged with a
  2-line `sys.path` shim in a single `frozen.py` module that re-exports
  the locked symbols; all forward code imports from `frozen`.

## Data flow

```
collect (2026-05-01 → 2026-07-31)          [frozen phase0 fetchers]
  -> forward_storage: data/forward/{source}/YYYY-MM.parquet   [FLAT layout]
  -> score_sentiment    [frozen sentiment.score_articles]
  -> build events       [frozen build_events_frame_for_month]
  -> compute IV         [frozen iv_surface.call_put_mid_iv]
  -> evaluate_forward   [frozen evaluate.evaluate_primary] -> IC + S1-S4 verdict
```

**Critical wrinkle — split guard:** Phase 0's
`storage.split_for_date` *fails closed* for any date after
`SPEC.holdout_end` (2026-04-30). That is correct behavior (it's the
holdout integrity guard) and must not be weakened. Therefore the
forward repo cannot reuse `write_partitioned_parquet`; it needs its
own thin `forward_storage` that writes a **flat** `data/forward/...`
layout with no train/validation/holdout routing. All *other* Phase 0
code (the four fetchers, sentiment, match_events, iv_surface, evaluate,
baselines) is reused unchanged because none of it splits.

The forward sample is a single undivided dataset; the only sub-division
is the S4 two-halves-by-time check, computed inside the frozen
`evaluate_primary`.

## Forward pre-registration

A `pre-registration-forward.md` (published as its own gist, mirroring
Phase 0's discipline) locks **before any forward IC is computed**:

- **Hypothesis:** the locked Phase 0 primary signal retains
  IC ≥ 0.05, p < 0.01 (two-sided), ≥ 1.5× the strongest of the same
  three baselines, and sign-consistent across the two equal-time halves
  of the forward window.
- **Window:** 2026-05-01 → 2026-07-31. Requires n ≥ 1000 events.
- **Stopping rule:** one evaluation, executed once, **after Phase 0
  publishes**.
- **Publication:** result published regardless of outcome, linking the
  Phase 0 pre-registration and result.
- A `forward_spec.py` encodes the forward-specific locked parameters
  (window dates, n threshold, S1-S4 reuse) with its own
  `fingerprint()` and a pinning test, exactly as Phase 0 locks `SPEC`.

## Isolation guarantees

- The forward repo **never reads** `saalr-phase0/data/` — the submodule
  is code-only. The Phase 0 holdout is never touched.
- Forward data is genuinely post-window, hence out-of-sample by
  construction.
- The forward pre-registration is committed (and gisted) before the
  first forward IC is computed (blind).
- The single evaluation runs only after Phase 0 publishes — no leakage
  in either direction.

## Repo structure

```
saalr-forward-test/
  pre-registration-forward.md        # locked forward hypothesis + window + S1-S4 + stopping rule
  saalr-phase0/                      # git submodule @ pinned locked commit (code-only)
  src/
    frozen.py                        # sys.path shim + re-exports of locked SPEC + transforms
    forward_spec.py                  # forward-specific locked params + fingerprint
    forward_storage.py               # flat data/forward/{source}/YYYY-MM.parquet writer
    collect.py                       # batch pull driver for the forward window
    build_forward_events.py          # assemble frames -> frozen build_events_frame_for_month
    compute_forward_iv.py            # frozen iv_surface over the events frame
    evaluate_forward.py              # one-shot frozen evaluate_primary -> verdict
  tests/
    test_frozen_provenance.py        # submodule SHA + SPEC fingerprint pins
    test_forward_spec.py             # forward_spec fingerprint pin
    test_forward_storage.py          # flat-write path
  decisions/                         # decision log (same discipline as phase0)
  journal/                           # daily journal
  data/                              # gitignored; forward parquets
  .gitignore
  requirements.txt                   # mirrors phase0 pins (CPU torch fine; no LSTM here)
  README.md
```

## Testing strategy

- `test_frozen_provenance.py` — the anti-overfit guarantee, as code.
- `test_forward_spec.py` — fingerprint pin for the forward params.
- `test_forward_storage.py` — flat-write path + month partitioning.
- The submodule's own suite remains runnable for regression.
- New orchestration code (`collect`, `build_forward_events`,
  `evaluate_forward`) gets unit tests for assembly/flat-write logic;
  the frozen transforms are already tested in the submodule and are
  not re-tested.

## Open items to finalize at build time

1. **Exact submodule pin SHA** — the saalr-phase0 commit to pin. Must
   have the complete pipeline (sentiment, options, match_events,
   iv_surface, evaluate, baselines) and a passing
   `test_locked_spec.py`. Current HEAD (the match_events commit or
   later) qualifies.
2. **Repo hosting** — confirm `saalr-forward-test` is a new GitHub repo
   under the same owner; whether it is public from the start (Phase 0
   is public).
3. **Forward gist** — the forward pre-registration is filed as its own
   gist before the first IC computation, same as Phase 0.

## Non-goals / explicitly not doing

- Not re-deriving or re-tuning any Phase 0 parameter.
- Not building the Variant B trading harness (deferred to a post-
  Week-3-addendum forward test).
- Not running the evaluation before Phase 0 publishes.
- Not weakening Phase 0's `split_for_date` holdout guard.
