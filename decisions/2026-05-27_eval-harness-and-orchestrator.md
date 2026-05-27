# 2026-05-27 — Eval harness (§§6-9) and a resumable Windows-laptop orchestrator

## Decision

Two artifacts landed together:

1. **Eval harness** — `src/metrics.py` (Spearman IC, bootstrap CI,
   hit-rate, Sharpe-approx), `src/baselines.py` (B1/B2/B3), and
   `src/evaluate.py` (top-level `evaluate_primary` with sub-period
   stability check and explicit pass/fail decisions against the locked
   S1-S4 from §9). All thresholds read from `SPEC`; nothing is
   hardcoded in the harness.

2. **Resumable orchestrator** — `src/orchestrator.py` with a JSON state
   file at `.phase0/state.json`, a registry of nine pipeline stages
   chunked by calendar month, and a `run | status | pause | resume |
   reset | stages` CLI. Two PowerShell scripts
   (`scripts/install-orchestrator.ps1`, `scripts/uninstall-orchestrator.ps1`)
   register the runner as a Windows Scheduled Task that starts at logon
   and restarts on failure.

## Context

User wants Phase 0 to make progress on a laptop while they're busy —
backfill takes hours of rate-limited API calls, sentiment scoring is
compute-heavy, and the experiment timeline runs seven weeks. A
human-driven `python -m src.data_acquisition` invocation per pull is
not the right shape. At the same time, the eval harness is independent
work that must exist before any model does, so the success-criteria
machinery can't be retrofitted to whatever IC the candidate happens to
hit on the holdout.

## Alternatives considered

### For the runtime

1. **systemd/cron equivalent.** Not applicable — Windows laptop.
2. **Prefect / Dagster / Airflow.** Solid workflow engines, but each is
   a multi-hour install with a UI and a database. Overkill for one
   user on one laptop running one experiment for seven weeks.
3. **NSSM (Non-Sucking Service Manager) wrapping a Python loop.** Good
   choice if "always running, even when no one is logged in" mattered;
   here the user IS the operator, so logon-triggered is enough.
4. **Windows Scheduled Task + JSON-state-backed Python loop (chosen).**
   Zero extra deps, transparent state, idempotent across sleep/wake/
   reboot/Ctrl+C. The state file is the source of truth; the loop is
   restartable from scratch at any time.

### For the harness

1. **Build only the metrics needed for the candidate model.** Tempting,
   but the pre-reg locks the baselines and the sub-period check too;
   delaying them risks "we'll add them later" turning into "we
   never quite got around to it."
2. **Inline the thresholds in evaluate.py.** Rejected — would create a
   second source of truth alongside `SPEC` and silently drift.
3. **Top-level `evaluate_primary` that reads all thresholds from SPEC,
   with the grey-zone explicitly classified (chosen).** §9's grey-zone
   ("IC between 0.03 and 0.05 is explicitly a failure, not a success")
   is a notorious place where founders rationalize — encoding it loudly
   in `EvaluationResult.in_grey_zone` removes the temptation.

## Rationale

The orchestrator is a deliberately small state machine: nine stages, a
JSON file, a forever loop. The complexity stays in the stages, not the
runner. Four stages are real today (`backfill_risk_free`,
`backfill_underlying`, `backfill_news`, plus the already-existing
options stub); five stages declare themselves `skipped` with an
explicit reason that names the missing module. As `src/sentiment.py`,
`src/iv_surface.py`, etc. arrive, swapping a `_stub_stage(...)` for a
real one is a one-line change.

The orchestrator deliberately **never touches the holdout**. The
`evaluate_validation` stage is explicit: it runs only on the
validation split. The Week-7 holdout evaluation is a manual,
one-shot operation per §12 of the pre-registration — automating it
would defeat the stopping rule.

The eval harness skips its own tests if pandas isn't installed (rather
than failing), so the locked-spec and storage-split tests keep passing
in environments without the heavy deps. The Polygon-dependent code
paths import at function scope for the same reason: `python -m
src.orchestrator status` should work on a fresh checkout before any
`pip install`.

## Reversible?

Yes on both pieces.

- The orchestrator's state file is regenerable from a fresh
  `python -m src.orchestrator run` — the parquet files on disk are
  the durable artifacts.
- The harness modules can be re-cut without touching the pre-registration.
- The Scheduled Task is removed with one PowerShell invocation
  (`scripts\uninstall-orchestrator.ps1`); `.phase0/` is left in place
  so re-installing the task resumes where it left off.

If the orchestrator approach turns out to be the wrong shape (e.g. we
move to a real cluster), the per-stage functions are pure Python and
port directly into any workflow engine.
