# Saalr Phase 0 — Signal Validation Experiment

This repository contains the code, decision log, and journal for **Saalr's Phase 0 signal-validation experiment** — a publicly pre-registered, out-of-sample test of whether FinBERT-derived news sentiment scores carry statistically significant, economically meaningful predictive information about near-term implied-volatility changes in SPY ATM weekly options.

**Public pre-registration:** _Gist URL added immediately after publication on 2026-05-27_

## What's public

- Methodology — see [`pre-registration.md`](./pre-registration.md), mirroring the gist
- All source code (added during the experimental timeline)
- All decision-log entries (`decisions/`)
- All daily-journal entries (`journal/`)
- Final results (Week 8), regardless of outcome
- Baselines and validation metrics

## What's not public

- Raw data snapshots (subject to vendor license terms; reproducible by anyone with the same Massive subscription via `src/data_acquisition.py`)
- API keys (use `.env.example` as a template)

## Repository discipline

- The `data/holdout/` SHA-256 hash is recorded in `data/MANIFEST.json` at end of Week 3, before any model fitting. Any modification of the holdout file after that point is detectable via the recorded hash.
- One primary hypothesis. One holdout test. One shot.
- All non-trivial decisions logged in `decisions/`. All daily notes in `journal/`.
- Results published publicly within 14 days of the holdout test, regardless of outcome.

## Project status

**Pre-registration phase.** Repository scaffold is in place; source code lands starting Week 2 of the experimental timeline (after the post-publication pause).

## Links

- Pre-registration gist: _added immediately after publication_
- Saalr: https://saalr.io

## License

- Code: MIT
- Methodology and pre-registration: CC-BY-4.0

## Contact

Sreekanth Payyavula — sreekanth@saalr.io
