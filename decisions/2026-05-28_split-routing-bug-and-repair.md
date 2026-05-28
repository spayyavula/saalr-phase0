# 2026-05-28 — Storage split-routing bug; pre-holdout repair

## Decision

`write_partitioned_parquet` now computes the split **per row** and
groups by `(split, year_month)`. The previous implementation grouped
by `year_month` alone and inferred the whole month's split from the
first row's date. That assumption ("all rows in a month share a
split") was false at February 2026 because the locked
`SPEC.validation_end = 2026-02-15 / SPEC.holdout_start = 2026-02-16`
boundary falls mid-month.

Two regression tests in `tests/test_storage_split.py` pin the new
routing: a 6-row Feb 2026 frame must produce exactly two files
(validation Feb 1-15 + holdout Feb 16-29), and a pure-month frame
must still produce one file.

A one-shot repair script
(`.phase0/repair_feb_2026_split.py`) re-routed the four contaminated
`validation/{src}/2026-02.parquet` files through the fixed function.
New row counts in `data/MANIFEST.json` document the repair.

## Context

Discovered today during pre-`backfill_options` design review. The
contamination magnitude on disk before repair:

| source | total rows in `validation/2026-02` | misrouted rows (≥ Feb 16) |
| --- | --- | --- |
| news | 1 114 | 496 (44.5 %) |
| underlying | 17 569 | 8 278 (47.1 %) |
| risk_free | 20 | 10 (50.0 %) |
| sentiment | 1 114 | 496 (44.5 %) |

`holdout/{src}/2026-02.parquet` was missing entirely for all four
sources before the repair.

## Severity

Pre-registration-invalidating if it had reached the §12 holdout
evaluation. The holdout test compares a single locked signal against
a locked holdout sample; a holdout that's missing half of February
and a validation that's contaminated with half of February's
post-Feb-16 rows would have:

- Inflated the validation IC by carrying ~5 000 extra training-window
  events into validation (especially harmful for Variant B because
  many post-Feb-16 events involve different macro state than pre-Feb).
- Deflated the holdout IC (or made it computationally impossible)
  by depriving holdout of half its samples — possibly below
  `SPEC.min_holdout_events = 1000`.

Caught before the holdout test by an audit-during-development. This
is exactly the failure mode pre-registration discipline exists to
prevent.

## Repair audit

After repair (idempotent — re-running on a fixed disk is a no-op):

| file | before rows | after rows | date range |
| --- | --- | --- | --- |
| `validation/news/2026-02.parquet` | 1 114 | 618 | 2026-02-01 → 2026-02-15 |
| `holdout/news/2026-02.parquet` | (missing) | 496 | 2026-02-16 → 2026-02-27 |
| `validation/underlying/2026-02.parquet` | 17 569 | 9 291 | 2026-02-02 → 2026-02-14 |
| `holdout/underlying/2026-02.parquet` | (missing) | 8 278 | 2026-02-17 → 2026-02-28 |
| `validation/risk_free/2026-02.parquet` | 20 | 10 | 2026-02-02 → 2026-02-13 |
| `holdout/risk_free/2026-02.parquet` | (missing) | 10 | 2026-02-17 → 2026-02-27 |
| `validation/sentiment/2026-02.parquet` | 1 114 | 618 | 2026-02-01 → 2026-02-15 |
| `holdout/sentiment/2026-02.parquet` | (missing) | 496 | 2026-02-16 → 2026-02-27 |

(2026-02-15 was Sunday; 2026-02-16 was Presidents Day. Earliest
trading day in holdout is 2026-02-17. The `validation/underlying`
ceiling at 2026-02-14 reflects after-hours bars on 2026-02-13 ET
rolling into 2026-02-14 UTC; same UTC-aware split routing as
documented in [2026-05-27_data-acquisition-architecture.md].)

Row-count arithmetic checks: 618 + 496 = 1 114, 9 291 + 8 278 =
17 569, 10 + 10 = 20.

## Alternatives considered

1. **Lock split boundaries to month-ends.** Would have prevented the
   bug but requires editing the pre-registration's locked window
   fields (`SPEC.validation_end`, `SPEC.holdout_start`), bumping the
   fingerprint, and publishing an addendum to the gist. The locked
   windows were chosen to give ~5.5 months of validation and ~2.5
   months of holdout in calendar terms; rounding to month-ends would
   move the boundary by 14 days in one direction. Rejected because
   the *code* was wrong, not the *spec*.
2. **Add a runtime assertion that a month's rows share a split.**
   Would have caught the bug at write time, but is the same as
   fixing the bug — `groupby(["_split", "_year_month"])` is the
   assertion-as-implementation.
3. **Re-pull all data from the vendor.** Would have worked but is
   ~5 hours of unnecessary API calls. The on-disk Feb 2026 data is
   already correct in its row contents; only the path is wrong.
   Re-partitioning is the right scope.
4. **Per-row split + per-split file grouping (chosen).** Each row's
   destination is now determined by `split_for_date(row.date)`. The
   grouping key is `(split, year_month)`; the output paths follow
   from that, so the split routing and the file layout are one
   operation, not two coordinated ones.

## Rationale

The bug was a load-bearing comment ("All rows in a month share a
split (each split aligns on month boundaries in SPEC — verified by
tests/test_storage_split.py)") that was wrong about what the tests
verified. `test_storage_split.py` checked `split_for_date(boundary)`,
not the partitioning behavior across a mid-month boundary. The two
new tests fill that gap.

Re-partitioning is preferable to re-pulling because:

- The rows on disk are the same vendor-returned rows; only their
  file paths were wrong.
- `data/MANIFEST.json` sha256 entries before the repair correspond
  to the contaminated files; after the repair the manifest has new
  entries for both halves of each source's Feb 2026 file, so a
  future integrity check can see exactly which artifacts changed.
- No additional API quota consumed.

## Reversible?

Yes. The fix is `src/storage.py`. The repair script is in
`.phase0/repair_feb_2026_split.py` and is idempotent. The
contaminated files on disk no longer exist; if the repair were
wrong, the disaster-recovery path is "re-pull Feb 2026 from the
vendor" — ~10 minutes of API calls through
`src/data_acquisition.py {underlying,news,risk_free} --start 2026-02-01
--end 2026-02-28`. The pre-registered spec is unchanged.

## Related

- [2026-05-27_data-acquisition-architecture.md] — original storage layer design (and the now-removed wrong comment about month-aligned splits).
