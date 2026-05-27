# 2026-05-27 — Q1 (ingest strike window) and Q2 (mid-quote source)

Resolves the two questions deferred at the top of `src/options.py` and
called out in
[2026-05-27_data-acquisition-architecture.md](2026-05-27_data-acquisition-architecture.md).
Empirical probe script: `.phase0/probe_q1_q2.py`. Raw probe output:
`.phase0/q1_q2_probe.json` (committed under `.phase0/` only via the
side-artifact path; the numbers below are reproducible against the
locked window).

## Decision

### Q1 — ingest strike window

**Ingest `ATM_open ± 25` strikes** centered on SPY's daily-open spot,
where strikes are integer dollars (the SPY weekly grid is `$1` wide).
Sample construction then post-filters per minute to the **locked
`SPEC.options_strike_window = 2`** band centered on the *event-time*
ATM strike. The post-filter step asserts coverage: if for any sample
minute the required `[ATM_event-2, ATM_event+2]` window is not fully
contained in the ingested set, a row is emitted to a
`coverage_failures.parquet` side file rather than silently dropped.
That file is reviewed before any model fit.

`SPEC` is not touched. The ingest half-width is an implementation
constant in `src/options.py`, not a locked field.

### Q2 — mid-quote source

**Path (b): windowed NBBO tick quotes via `client.list_quotes` with an
"as-of" semantic at each sample time.** For each `(contract, sample_time)`
pair we issue:

```python
client.list_quotes(
    ticker=occ,
    timestamp_gte=sample_time - 30s,
    timestamp_lte=sample_time + 30s,
    limit=500,
    sort="timestamp",
    order="desc",
)
```

and take the **first row with `sip_timestamp <= sample_time`** (i.e.,
the last NBBO quote at or before the sample). `mid = (bid_price +
ask_price) / 2`. A row is dropped if `(ask - bid) / mid >
SPEC.options_max_spread_pct_of_mid (0.10)`.

## Context

`pre-registration.md` §3 locks:

- `options_strike_window = 2` (the *sample* window)
- `options_granularity_minutes = 1`
- `options_quote_basis = "mid"` and `iv_quote_basis = "call_put_mid"`

The two implementation questions left open on filing day were *how wide
to ingest* (so the locked sample window is always covered) and *where
the mid-quote actually comes from* (the vendor's minute aggregates are
OHLC of **trades**, not quotes).

## Alternatives considered

### Q1 candidates

Coverage of the *locked* `ATM_event ± 2` requirement, computed as
`P(intraday_excursion + 2 <= W)` over 602 SPY trading days from
2024-01-01 to 2026-05-27:

| Ingest half-width `W` | Day-coverage of ATM±2 sample | Days uncovered (of 602) |
| --- | --- | --- |
| 10 | ~89.4 % (excursion ≤ 8) | ~64 |
| 15 | ~96.8 % (excursion ≤ 13) | ~20 |
| 20 | ~98.7 % (excursion ≤ 18) | ~8 |
| **25** | **~99.3 %** (excursion ≤ 23) | **~4** |
| 30 | ~99.5 % (excursion ≤ 28) | ~3 |

Excursion distribution (max of `|high - open|`, `|open - low|` in $):

| stat | value |
| --- | --- |
| median | $4.05 |
| p90 | $8.59 |
| p95 | $10.76 |
| p99 | $18.61 |
| max | $55.18 (April 2024 reversal day) |

Considered:

1. **ATM-open ± 10 (the journal's provisional value).** Rejected.
   ~89 % day-coverage of the locked ATM±2 sample is unacceptable when
   on the failing days we'd silently drop *every* event whose minute
   falls outside ingest. The journal's value pre-dated the empirical
   probe.
2. **ATM-open ± 30.** Marginal gain over ± 25 (0.2 %). Doesn't catch
   the genuine tail day at $55, and the cost of going wider isn't
   storage (contract metadata is small) — it's the cognitive load of
   "is wider always better?" without a stopping rule. ± 25 is
   sufficient: the remaining 0.7 % of days are extreme tail moves
   where event coverage failures will be loud and traceable.
3. **Adaptive width per day based on prior-day VIX.** Rejected as
   premature complexity. The base case is fixed; if the Week-3
   sanity check shows a meaningful number of events being dropped on
   high-VIX days, we revisit.
4. **ATM-open ± 25 with explicit coverage assertion (chosen).** The
   ATM±2 post-filter must verify membership, not assume it. A coverage
   failure is logged structurally to `coverage_failures.parquet`. Per
   the existing storage convention these failures are inspected before
   any model fit and decisions about whether to re-pull (a per-event
   strike top-up) are deferred to that point.

### Q2 candidates

Three paths were considered:

(a) **Per-minute *quote* aggregates endpoint.** The empirical probe
    confirms this **does not exist** in the public REST API. The
    `/v2/aggs/ticker/{ticker}/range/.../minute/...` path returns
    trade OHLC. `list_aggs` against the option ticker for one trading
    day returned 366 minute bars in 0.5 s, with `open = 2.18`, but
    that `2.18` matches the *bid* of the first NBBO quote in the same
    second — these are trades crossing at the bid, not the mid.
    Rejected: would violate `SPEC.options_quote_basis = "mid"`.

(b) **NBBO tick quotes via `list_quotes`, bucketed to 1-minute mid.**
    Empirically: a full day of NBBO for one ATM SPY weekly contract
    exceeds **100,000 ticks** in 6 s of pulling (we hit our probe
    cap; real daily count is likely 500 k – 1 M+). First-row sample:
    `bid=2.18, ask=2.22, size=106/106`. A naive whole-day pull
    multiplied by `~5,000 events × ~10 contracts ≈ 50,000 contract-days`
    is impractical. A **windowed pull** at each sample time
    (`sample ± 30 s`, ~50–500 rows per query) keeps the per-event cost
    bounded and is what the production data path uses. Chosen.

(c) **`get_snapshot_option`.** The probe confirms this works
    cleanly — the live snapshot returns
    `last_quote.{bid=3.18, ask=3.20, midpoint=3.19, timeframe="REAL-TIME"}`
    plus `implied_volatility=0.140` and `open_interest=4446`. **But**:
    `timeframe == "REAL-TIME"` is the whole point — there is no
    historical snapshot endpoint. Useless for the
    2024-01-01 → 2026-04-30 backfill. Kept in mind for any
    forward-paper-trading extension after Phase 0.

## Rationale

`SPEC.options_strike_window = 2` is locked; ingest width is not, so we
get to pick the implementation parameter that maximizes coverage of
the locked sample window without re-opening the pre-reg. The 99.3 %
coverage at ± 25 is the right operating point because the cost is in
contract metadata storage (negligible) and the failure mode is
detectable per-event rather than silent.

For Q2 the locked spec leaves only one viable engineering path:
windowed `list_quotes` with as-of semantics. The other two are ruled
out by physics (no minute-quote agg endpoint exists) and by data
availability (no historical snapshot). The 30 s window on either side
of the sample time is conservative — typical ATM SPY weekly NBBO
update rates are sub-second around the event window — but matches the
pre-reg's tolerance for "minute granularity" and protects against
data-feed gaps without bloating per-query payloads.

The orchestrator's `backfill_options` stage (currently a stub per
[2026-05-27_eval-harness-and-orchestrator.md](2026-05-27_eval-harness-and-orchestrator.md))
will land in a follow-on commit that calls the Q1+Q2 paths chosen here.
The coverage-failure parquet file is part of that follow-on, not this
decision.

## Reversible?

Yes on both pieces.

- Q1: the ingest half-width is a constant in `src/options.py`. Widening
  it later just means a re-pull for the new contracts; existing data
  is unaffected. Narrowing it would require a coverage re-audit but
  costs nothing in pre-reg credibility.
- Q2: switching from windowed `list_quotes` to a hypothetical future
  minute-quote agg endpoint, or to (c) for a forward-paper run, is a
  function-body swap behind the same `fetch_option_mid_quote_at(...)`
  signature. The pre-reg pinning is `options_quote_basis = "mid"`,
  which any of the three paths satisfies in principle.
