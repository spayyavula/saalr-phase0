# Data Directory

Raw and processed data lives here. Most of it is **gitignored** to respect
vendor license terms — see the top-level `.gitignore`.

## Subdirectories

- `train/` — training set (2024-01-01 → 2025-09-30)
- `validation/` — validation set (2025-10-01 → 2026-02-15)
- `holdout/` — holdout set (2026-02-16 → 2026-04-30) — SHA-256 hash committed
  to `MANIFEST.json` at end of Week 3, before any model fitting begins.

## Reproducibility

`src/data_acquisition.py` (added in Week 2) will populate these directories
from Massive's API given a valid `MASSIVE_API_KEY` in `.env`.

## MANIFEST.json

At end of Week 3, this file will record row counts and SHA-256 hashes for
each snapshot. The holdout hash is the binding pre-registration commitment
— if `holdout.parquet` is modified after the manifest is committed, the
hash change is detectable.
