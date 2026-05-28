# 2026-05-28 — Quote-fetch resilience; match_events ordering fix

## Decision

Two changes after the orchestrator's `match_events` stage hit an
infinite fail-loop on a single transient vendor error:

1. **`fetch_option_mid_quote_at` is now resilient.** Transient vendor
   failures (502/503/504, connection resets, timeouts, rate-limit
   blips) are retried up to `QUOTE_RETRY_ATTEMPTS` (3) times with a
   `QUOTE_RETRY_SLEEP_SECONDS` (5 s) pause between attempts. On
   persistent failure it returns `None` — the caller treats that as a
   missing quote (an auditable `failure_reason`), not a crash. A
   **non-transient** error (a real programming bug) is re-raised so it
   surfaces rather than being silently masked.

2. **`build_events_frame_for_month` has a per-event guard.** Each
   event is built in its own try/except; an unexpected raise on one
   event records a `failure_reason = "exception:<Type>"` row and the
   loop continues. One bad event can no longer abort a month of
   ~1,500 events.

Also fixed a **stage-ordering bug**: the `STAGES` list had
`compute_iv` before `match_events`, leftover from the original stub
order. Since `match_events` produces the events frame that
`compute_iv` consumes, the order is now `match_events → compute_iv`
(module docstring updated to match).

## Context

The `match_events` stage (commit c87b361) was launched as an
overnight backfill. It failed 16 consecutive times — every failure on
the identical contract `O:SPY240105C00469000` (the $469 call on
2026... actually 2024-01-04) — with:

```
HTTPSConnectionPool(host='api.polygon.io', ... Max retries exceeded ...
Caused by ResponseError('too many 502 error responses')
```

Polygon returned repeated 502s for that one quote query; the polygon
client's internal retries exhausted and **raised**. The raise
propagated out of `build_events_frame_for_month`, the stage marked the
whole month `failed`, and the orchestrator re-attempted the same month
every ~4 minutes — re-hitting the same early-January quote first and
dying there again. Net result: **zero events were ever written**, and
Polygon was hammered with retries for hours.

The reported symptom ("lots of errors in the console log") was 16
identical month-level failures.

## Root cause

One transient vendor error on one quote aborted an entire month, and
the orchestrator's retry-failed-chunk behavior turned that into an
infinite loop. The fragility — not the 502 itself — was the bug. This
is the same class of failure as the 2026-05-27 silent news-pull hang:
*a single transient vendor failure must not abort a whole chunk.*

## Alternatives considered

1. **Just catch and return None (no retry).** Simple, stops the
   crash. But a transient 502 that would have succeeded on retry would
   silently drop the event — if the 502s were load/rate-induced, we'd
   lose many real events to our own concurrency. Rejected alone.
2. **Retry with exponential backoff (0.5s, 1s, 2s).** Standard, but
   the user asked specifically for a 5 s pause, and for a 502-storm a
   longer fixed pause gives the gateway more room to recover than a
   sub-second first retry. Chosen variant: flat 5 s (+ small jitter).
3. **Retry (5 s) then None, re-raise non-transient (chosen).** Rides
   out transient blips, degrades gracefully to a logged missing quote
   on persistent failure, and still surfaces genuine bugs. Paired with
   the per-event guard so even an unforeseen raise can't lose a month.
4. **Lower the 6-way quote parallelism to reduce 502s.** Considered,
   but the smoke runs (4-way then 6-way) showed no rate problems at
   small scale, and the specific contract failed repeatedly across
   hours, which points to a transient server issue on that query more
   than steady-state rate limiting. Left parallelism at 6; the retry
   absorbs occasional blips. Revisit if persistent-failure rates are
   high in the real backfill.

## Rationale

Resilience belongs at the lowest layer that can classify the error.
`fetch_option_mid_quote_at` knows it's doing a single quote pull, so
it owns the retry/None decision; everything above it just sees "quote
or no quote." The per-event guard in `build_events_frame_for_month` is
belt-and-suspenders for the unattended 12 h grind — the dominant
failure mode (the 502) is fixed at the quote layer, but a long
autonomous job shouldn't lose a month to any single unforeseen raise.

The non-transient re-raise is important: silently returning `None` for
a real bug (e.g., a bad ticker format, an attribute error) would mask
defects as "missing quotes" and quietly shrink the sample. Transient
classification is by exception type (urllib3 HTTPError, ConnectionError,
TimeoutError, OSError, polygon BadResponse) with a message-marker
fallback for the wrapped MaxRetryError we actually observed.

## Reversible?

Yes. `QUOTE_RETRY_ATTEMPTS` / `QUOTE_RETRY_SLEEP_SECONDS` are module
constants. The per-event guard and the stage reordering are local. No
persisted artifact changes. The locked `SPEC` is untouched — none of
this is pre-registered methodology, it's acquisition robustness.

## Related

- [2026-05-27_data-acquisition-architecture.md] — the news-pull hang, same fragility class.
- [2026-05-28_match-events-sampling-architecture.md] — the events frame this hardens.
