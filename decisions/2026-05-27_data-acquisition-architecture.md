# 2026-05-27 — Acquisition-layer architecture (§3) and the vendor question

## Decision

The Phase 0 acquisition layer is implemented as four source-specific
fetchers (`src/news.py`, `src/options.py`, `src/underlying.py`,
`src/risk_free.py`) plus a CLI dispatcher (`src/data_acquisition.py`) and
a partitioning + manifest helper (`src/storage.py`). Splits are derived
per-row from the locked windows in `SPEC` and parquet files land under
`data/{split}/{source}/YYYY-MM.parquet`. The data vendor is treated as a
single account: the same API key works against both the "Massive" name
in `pre-registration.md` and the `polygon-api-client` package.

## Context

`pre-registration.md` §3 prescribes the data sources by name ("Massive
News API," "Massive Options Advanced," "Massive Stocks Aggregates," FRED
DGS3MO) and the README points at a single `src/data_acquisition.py` as
the populator. Two non-obvious questions had to be resolved before any
acquisition code could land.

## Alternatives considered

1. **One monolithic `data_acquisition.py`.** Honours the README literally
   but makes each source's quirks (news fan-out + dedupe; options
   contract-listing; FRED CSV vs vendor SDK) collide in one file.
2. **Per-source modules behind a generic `Provider` protocol.** Cleaner
   in the abstract but premature — the four sources share almost no
   surface beyond "return a frame," so the protocol is overhead.
3. **Per-source modules called from a thin CLI dispatcher (chosen).**
   Each fetcher owns its source's quirks; the dispatcher only knows how
   to route subcommands and call `storage.write_partitioned_parquet`.
   Honours the README's "one populator" hint while keeping the modules
   testable in isolation.

## Rationale

The four sources differ enough that hiding them behind a uniform
abstraction would buy nothing. The dispatcher pattern keeps the CLI
thin and the per-source modules narrow. `storage.split_for_date` is the
one tightly-coupled-to-the-spec utility, and it fails closed: a row
whose date is outside every locked window raises, which is what you
want during ingest.

Two acquisition questions are explicitly deferred to a follow-on commit
in Week 2, called out at the top of `src/options.py`:

- **Q1 (strike window).** Spot drifts intraday, so ATM at minute `t` is
  not ATM at minute `t+30`. We will ingest a wider strike window
  (provisionally ATM±10 by daily-open spot) and post-filter to the
  locked ATM±2 per event at sample-construction time. The post-filter
  step will assert coverage so silent drops are loud.
- **Q2 (mid-quote source).** The vendor's minute aggregates are
  trade-OHLC; the pre-reg locks mid-quote. The empirical question is
  whether minute *quote* aggregates exist on this account, or whether
  we need to bucket tick quotes ourselves. `fetch_option_aggregates`
  pulls trade-OHLC today purely to validate the contract-listing path;
  mid-quote arrives in a follow-on commit once Q2 is decided.

On the vendor itself: PyPI verification on 2026-05-27 confirmed the
package name is still `polygon-api-client` and still targets polygon.io.
The user reported their polygon.io key rolled over to Massive and is
treating them as the same vendor. Code therefore uses the official
client with the `MASSIVE_API_KEY` env var; module names stay
vendor-neutral so the public pre-reg's "Massive" language can remain
unchanged without forcing a code rewrite if the rebrand completes
asymmetrically.

End-to-end smoke confirmed against the no-auth FRED path:
`python -m src.risk_free 2026-05-01 2026-05-26` pulled 15 real
observations of DGS3MO (3.65–3.70 %) without touching the vendor.

## Reversible?

Yes. The module boundaries can be re-cut cheaply — no schema
commitments leak into pre-registration.md, and the parquet partition
layout is regenerable from a fresh pull. The two deferred questions
will be answered with a paired decisions entry once empirically settled.
