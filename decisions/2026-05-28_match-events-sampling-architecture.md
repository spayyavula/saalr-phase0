# 2026-05-28 — match_events sampling architecture

Settles the implementation choices for the per-event sample
construction (`src/match_events.py`). These are the unlocked
engineering decisions that sit between the pre-registered
specification and the persisted events frame that `compute_iv` and
`evaluate_validation` consume.

## Decisions

### D1 — Same-contract policy for IV(t) and IV(t+30)

For each event, the ATM-closest-listed strike and nearest-weekly-Friday
expiry are picked **once at the event time** and held constant for
both samples. `forward_iv_change = IV(t+30) - IV(t)` measures the
implied-vol change of **the same option contract**, not the change in
"whatever ATM happens to be 30 minutes later."

### D2 — EWMA aggregation includes the article AT t

The aggregated signal at time t includes the article published exactly
at t with weight 1.0 (i.e., `exp(0) = 1`). The pre-reg says "all
article scores within previous 4 hours" with a half-open look-back
window. We interpret "within" as inclusive on both ends; an article
that arrives at t IS the news that the signal at t is supposed to
reflect.

### D3 — Trading-day calendar derived from on-disk underlying data

We do not add a holiday-calendar dependency. Instead, the date of an
event is a trading day **iff** `data/{split}/underlying/{YYYY-MM}.parquet`
contains at least one bar with an ET date equal to the event date.
The `backfill_underlying` stage is gated upstream of `match_events`,
so this is always known by the time we filter events.

### D4 — Spot at event time uses the bar that *contains* t (close)

`find_spot_at_minute(df, t)` returns the `close` of the latest bar
whose `timestamp` (bar-open) is `<= t`. This is the "as-of price" at
moment t: the price that an observer at exactly t would have most
recently seen printed. Same convention as the Q2 mid-quote helper's
`sip_timestamp <= sample_time`.

### D5 — Risk-free rate forward-fills weekends and holidays

`find_rfr_at_date(df, d)` returns the FRED DGS3MO observation on `d`,
or the most recent observation before `d` if FRED has no obs on `d`
(weekends, federal holidays). FRED's DGS3MO is a daily series whose
intra-week stability over short windows is well within the 4th decimal
place of any IV computation we run, so forward-fill is faithful to the
underlying signal.

### D6 — Event window is half-open [09:35, 15:30) ET

`is_in_event_window` uses `_EVENT_OPEN_ET <= t < _EVENT_CLOSE_ET`.
The pre-reg literally writes `[09:35:00 ET, 15:30:00 ET)`. An article
at exactly 15:30 is excluded; the +30 min forward sample of an
included event at 15:30 - epsilon would otherwise need a 16:00 quote
which is past close.

## Context

`pre-registration.md` §4 / §5 lock:

- Event = news article with `published_utc` in
  `[SPEC.event_window_open_et, SPEC.event_window_close_et)` ET on a
  US trading day.
- Signal at t = EWMA over last `SPEC.aggregation_lookback_hours` of
  article scores with halflife `SPEC.ewma_halflife_minutes`.
- ATM = `SPEC.strike_rule = "atm_closest_listed"` strike.
- Expiry = `SPEC.expiry_rule = "nearest_weekly_friday"`.
- Forward horizon = `SPEC.forward_horizon_minutes = 30`.

The spec does not pin:

- Whether the contract identity (expiry + strike) is held constant
  between t and t+30 or re-picked at t+30.
- Whether the article AT t contributes to the aggregated signal at t.
- The source of the US trading-day calendar.
- The bar-close-vs-bar-open convention for spot lookup.
- Whether the risk-free rate is forward-filled across weekends.
- Whether the event window is open-open, open-closed, or half-open.

All six were settled here.

## Alternatives considered

### D1 — Same-contract policy

1. **Re-pick ATM at t+30.** Faithful to "ATM mid-IV" at both samples.
   But `forward_iv_change` then measures the change in "whatever ATM
   means at this moment," which has spurious variance from spot
   movement crossing strike boundaries. A spot change from $471.40
   to $470.40 with $0.50 strikes would re-pick ATM from 471.50 to
   470.50, and the apparent IV change would include the listed-strike
   smile slope, not just the time decay we want to measure.
2. **Hold (expiry, strike) constant from t (chosen).** The natural
   read of "forward IV change" — same option, two snapshots. The
   spec is silent, but the term "forward change" in options
   literature universally means a single contract observed twice.
3. **Hold strike but re-pick expiry at t+30.** Strange — for events
   late in the week, the t+30 sample might be on next Friday's expiry
   while t is on this Friday's. Asymmetric for no clean reason.

### D2 — EWMA inclusion of the article at t

1. **Exclude the article at t (lagged signal).** Avoids any
   look-ahead concern. But there is no look-ahead — the article's
   `published_utc` is when it became public; the signal at t is
   computed from public information at t. Excluding it would lose
   the most-informative observation at every event.
2. **Include the article at t with weight 1 (chosen).** Each event's
   own news contributes to its own signal value. The pre-reg's wording
   ("all article scores within previous 4 hours") includes the
   boundary points; we choose to include both.

### D3 — Trading-day calendar source

1. **Add `pandas_market_calendars` to requirements.** Standard
   library for US-equity holidays. Heavy dependency for one boolean
   check; we already have all the data we need.
2. **Hardcode the NYSE holiday list 2024-2026.** Brittle; we'd own
   the responsibility for half-day-Christmas-Eve, Presidents Day
   shifts, etc.
3. **Derive from on-disk underlying (chosen).** If
   `data/{split}/underlying/YYYY-MM.parquet` has bars with this ET
   date, it's a trading day. Vendor authoritatively answers the
   question; we don't have to maintain a parallel calendar. The
   `backfill_underlying` dependency makes this always available.

### D4 — Spot lookup convention

1. **Bar `open` of the bar starting at t.** Off by up to 1 minute on
   the wrong side — the open of the bar starting at t hasn't happened
   yet from the perspective of someone at t.
2. **Bar `close` of the latest bar with open `<= t` (chosen).**
   "Last printed price as of t."
3. **VWAP of the containing bar.** Closer to "fair price" but
   introduces another data-quality dependency (`vwap` field), and
   doesn't materially change the IC since spot enters BS only through
   `ln(S/K)` at the picked strike.

### D5 — Risk-free rate

1. **Strict: NaN on missing date.** Drops a few percent of events
   for being on a weekend/holiday FRED gap — but our events are by
   definition on trading days, so they always have a same-day FRED
   obs anyway. Forward-fill is a safety net.
2. **Forward-fill (chosen).** Robust to the rare case where FRED
   publishes late (weekend release, federal holiday). DGS3MO moves
   slowly enough that a day or two of forward-fill is well within
   IV-decimal-place sensitivity.

### D6 — Event window endpoints

1. **Closed-closed `[09:35, 15:30]`.** Last-minute events have their
   t+30 sample at 16:00 which is past market close — quotes either
   don't exist or are illiquid.
2. **Half-open `[09:35, 15:30)` (chosen).** Last includable event
   is at 15:29; its t+30 = 15:59, still within the regular session.
   Matches the pre-reg's literal notation.
3. **Open-open `(09:35, 15:30)`.** Excludes 09:35 events for no good
   reason; the open auction concludes by then.

## Rationale

All six choices are documented because they affect every event
identically. Per the multiple-comparisons discipline in
`pre-registration.md` §10, a *single* unlocked convention applied
consistently is fine — what's not fine is shopping conventions to find
the one that gives the best IC. By writing them down now, before
`compute_iv` runs anything on the validation set, we eliminate the
post-hoc temptation.

These six form one coherent sampling pipeline:

```
event_filter -> trading_day_check -> aggregate_signal
              \              \    \
               \              \    -> spot_at_t -> atm_strike
                \              \                \
                 -> nearest_friday -------------> (expiry, strike) frozen
                                                \
                                                 -> mid_quotes(t, t+30)
                                                                \
                                                                 -> rfr_at_date
```

The pure helpers in `src/match_events.py` implement everything left
of "mid_quotes" — the network-bound mid-quote pulls and the
orchestrator stage land in a follow-on commit once
`backfill_options` has finished writing the per-day contract universe.

## Reversible?

Yes on all six. Each is a single function in `src/match_events.py`
plus a parameter or two. Changing D1 would require rewriting the
events frame (`data/{split}/events/*.parquet`) — feasible without
re-pulling raw vendor data. Changing any of D2-D6 is a no-op for
on-disk vendor data and just re-runs the sample-construction stage.

The pre-registration is unchanged. No SPEC fingerprint bump is
required because none of D1-D6 contradict the locked spec — they're
the choices the spec deliberately left open.
