# 2026-05-28 — Pre-registered interpretation lens: "buy the rumour, sell the news"

**Filed before any holdout look.** This note records a prior expectation
about the *shape* of the result, so that a weak or negative forward IC is
read as *consistent with a stated hypothesis* rather than rationalized
post-hoc. It does **not** change any locked parameter of the experiment.

## The observation

Markets often price an event during the anticipation ("rumour") phase and
fade or reverse once the news is confirmed. In **implied-volatility** terms
this is the well-known **IV crush / vol-of-event** pattern: uncertainty
builds *into* an event (IV rises during the rumour phase), then collapses
once the unknown resolves (IV drops after publication).

## Why it matters for this experiment

The locked design measures, per news article:

- event time = `published_utc`
- target = `forward_iv_change` = IV(t+30 min) − IV(t)

If the informative move occurs **before** `published_utc` (the rumour
phase), and the post-publication move is dominated by mechanical vol-crush,
then:

1. The forward IC may be **weak or negative**, even if the sentiment
   content is genuinely informative — because we are measuring the fade,
   not the anticipation.
2. `forward_iv_change` after a news event may be **mechanically negative**
   (vol-crush) somewhat independent of sentiment content, which can
   **confound or mask** the sentiment→IV relationship.

This is the same effect already acknowledged in `pre-registration.md`
§11 limitation #1 ("scoops may have moved markets minutes-to-hours before
the published timestamp"); this note states its expected *direction* and
its IV-specific mechanism explicitly.

## Pre-registered expectation (the lens)

On the buy-rumour-sell-news / IV-crush mechanism, we expect it is
**plausible a priori** that the locked forward IC is weak or negative.
A result of that shape is therefore **consistent with a documented prior**,
not a surprise to be explained away after the fact. Per `pre-registration.md`
§2 the test is two-sided, so a fade-shaped (negative) IC is evidence against
H0, and per §13 a near-zero forward IC is a legitimate publishable result.

## What this does NOT change

Nothing locked moves. Event = `published_utc`, forward horizon = 30 min,
target = IV(t+30) − IV(t), two-sided test, S1–S4 thresholds — all sealed.
The holdout stopping rule (§12) is untouched. This note is interpretive
context, not a methodology change; no SPEC fingerprint change.

## In-bounds exploratory probe (§10 — cannot claim success)

We already compute `prior_iv_change` = IV(t) − IV(t−30) for the B1
baseline. As a **strictly exploratory** analysis (labeled per §10, reported
in the writeup, *cannot* justify a success claim and gets no extra holdout
looks), we may compare:

- `Spearman(signal, forward_iv_change)` — the locked, post-news relationship
- `Spearman(signal, prior_iv_change)` — the anticipation/run-up relationship

If the signal lives in the rumour phase, the *prior* correlation would be
the stronger of the two. Script: `exploratory/rumour_window_check.py`. This
is exploratory only; it is run on the **train/validation** splits, never as
an additional holdout evaluation.

## Phase 2 (separately pre-registered, if pursued)

A proper test of the rumour intuition would pre-register a **pre-news
window** design (e.g., target = IV change over [t−W, t], or a paired
pre/post comparison) as its own experiment, before any data look — the same
way the forward test (Phase 1) is pre-registered. It is explicitly out of
scope for Phase 0.
