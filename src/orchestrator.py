"""Phase 0 background orchestrator.

A resumable, restartable runner that walks the pipeline in order:

    backfill_risk_free → backfill_underlying → backfill_news
                       → (backfill_options — deferred)
                       → score_sentiment → compute_iv → match_events
                       → fit_baselines → evaluate_validation

Each stage is chunked by calendar month so a single iteration does a
small unit of work and writes progress to ``.phase0/state.json``. Killing
the process and restarting it (Ctrl+C, laptop sleep, Task Scheduler
restart, machine reboot) picks up where it left off — the JSON state is
the only source of truth.

Subcommands:

::

    python -m src.orchestrator run         # main loop (Task Scheduler calls this)
    python -m src.orchestrator status      # print current state as a table
    python -m src.orchestrator pause       # halt at end of current chunk
    python -m src.orchestrator resume      # clear pause flag
    python -m src.orchestrator reset STAGE # wipe one stage's progress
    python -m src.orchestrator stages      # list registered stages

The Windows Task Scheduler installer at
``scripts/install-orchestrator.ps1`` wires this to run at logon and
restart on failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

logger = logging.getLogger("phase0_orchestrator")

STATE_VERSION = 1
RATE_LIMIT_SLEEP_SECONDS = 15
NOTHING_PENDING_SLEEP_SECONDS = 600  # 10 min when every stage is done or stub
PAUSED_POLL_SECONDS = 60


# ---------------------------------------------------------------------------
# repo + state paths
# ---------------------------------------------------------------------------


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward looking for pre-registration.md as the repo marker."""
    cursor = (start or Path.cwd()).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / "pre-registration.md").exists():
            return candidate
    raise RuntimeError(
        "could not find Phase 0 repo root (no pre-registration.md found "
        f"upward from {cursor}); run the orchestrator from inside the repo"
    )


def _state_dir(repo_root: Path) -> Path:
    p = repo_root / ".phase0"
    p.mkdir(exist_ok=True)
    return p


def _state_path(repo_root: Path) -> Path:
    return _state_dir(repo_root) / "state.json"


def _pause_path(repo_root: Path) -> Path:
    return _state_dir(repo_root) / "pause"


def _log_path(repo_root: Path) -> Path:
    return _state_dir(repo_root) / "orchestrator.log"


# ---------------------------------------------------------------------------
# state I/O
# ---------------------------------------------------------------------------


def load_state(repo_root: Path) -> dict:
    path = _state_path(repo_root)
    if not path.exists():
        return {"version": STATE_VERSION, "stages": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"state file version mismatch in {path}: expected {STATE_VERSION}, "
            f"got {data.get('version')}"
        )
    return data


def save_state(repo_root: Path, state: dict) -> None:
    path = _state_path(repo_root)
    payload = {**state, "updated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def is_paused(repo_root: Path) -> bool:
    return _pause_path(repo_root).exists()


def set_paused(repo_root: Path, paused: bool) -> None:
    pause_file = _pause_path(repo_root)
    if paused:
        pause_file.write_text(
            datetime.utcnow().isoformat(timespec="seconds") + "Z\n", encoding="utf-8"
        )
    elif pause_file.exists():
        pause_file.unlink()


# ---------------------------------------------------------------------------
# month chunking + locked-window iteration
# ---------------------------------------------------------------------------


def _iter_months(start_iso: str, end_iso: str) -> list[str]:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    out = []
    cursor = start.replace(day=1)
    while cursor <= end:
        out.append(cursor.strftime("%Y-%m"))
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def _month_range(year_month: str) -> tuple[date, date]:
    year, month = (int(part) for part in year_month.split("-"))
    start = date(year, month, 1)
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    end = next_month - timedelta(days=1)
    return start, end


def _all_locked_months() -> list[str]:
    from src.locked_spec import SPEC

    return _iter_months(SPEC.train_start, SPEC.holdout_end)


# ---------------------------------------------------------------------------
# stage registry
# ---------------------------------------------------------------------------


@dataclass
class StageOutcome:
    """One stage's report after one iteration."""

    did_work: bool
    message: str = ""


StageFn = Callable[[dict, Path], StageOutcome]


def _stage_state(state: dict, name: str) -> dict:
    return state.setdefault("stages", {}).setdefault(
        name, {"status": "pending", "chunks": {}}
    )


def _backfill_stage(
    name: str,
    source: str,
    fetch_one_month: Callable,
) -> StageFn:
    """Build a backfill stage that pulls one pending month per call."""

    def stage(state: dict, repo_root: Path) -> StageOutcome:
        from src import storage  # lazy

        stage_state = _stage_state(state, name)
        chunks = stage_state["chunks"]
        for month in _all_locked_months():
            if chunks.get(month, {}).get("status") == "complete":
                continue
            start, end = _month_range(month)
            try:
                df = fetch_one_month(start, end)
                rows = int(getattr(df, "shape", (0,))[0])
                if rows > 0:
                    storage.write_partitioned_parquet(
                        df, repo_root / "data", source
                    )
                chunks[month] = {
                    "status": "complete",
                    "rows": rows,
                    "completed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
                stage_state["status"] = "in_progress"
                return StageOutcome(did_work=True, message=f"{name} {month} rows={rows}")
            except Exception as exc:  # noqa: BLE001
                logger.error("%s %s failed: %s", name, month, exc)
                logger.debug("%s", traceback.format_exc())
                chunks[month] = {
                    "status": "failed",
                    "error": str(exc),
                    "failed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
                stage_state["status"] = "in_progress"
                return StageOutcome(
                    did_work=True, message=f"{name} {month} FAILED: {exc}"
                )
        stage_state["status"] = "complete"
        return StageOutcome(did_work=False)

    return stage


def _stub_stage(name: str, reason: str) -> StageFn:
    """A not-yet-implemented stage. Marks itself ``skipped`` once per state
    file and never does work."""

    def stage(state: dict, _repo_root: Path) -> StageOutcome:
        stage_state = _stage_state(state, name)
        if stage_state.get("status") != "skipped":
            stage_state["status"] = "skipped"
            stage_state["reason"] = reason
            stage_state["skipped_at_utc"] = (
                datetime.utcnow().isoformat(timespec="seconds") + "Z"
            )
        return StageOutcome(did_work=False)

    return stage


def _stage_backfill_risk_free() -> StageFn:
    def fetch_one_month(start: date, end: date):
        from src.risk_free import fetch_risk_free

        return fetch_risk_free(start, end)

    return _backfill_stage("backfill_risk_free", "risk_free", fetch_one_month)


def _stage_backfill_underlying() -> StageFn:
    def fetch_one_month(start: date, end: date):
        from src.underlying import fetch_underlying

        return fetch_underlying(start, end)

    return _backfill_stage("backfill_underlying", "underlying", fetch_one_month)


def _stage_backfill_news() -> StageFn:
    def fetch_one_month(start: date, end: date):
        from src.news import fetch_news

        return fetch_news(start, end)

    return _backfill_stage("backfill_news", "news", fetch_one_month)


def _stage_backfill_options() -> StageFn:
    return _stub_stage(
        "backfill_options",
        "deferred: see src/options.py module docstring Q1 (strike window) and "
        "Q2 (mid-quote source). Resolve those decisions and replace this stub.",
    )


STAGES: list[tuple[str, StageFn]] = [
    ("backfill_risk_free", _stage_backfill_risk_free()),
    ("backfill_underlying", _stage_backfill_underlying()),
    ("backfill_news", _stage_backfill_news()),
    ("backfill_options", _stage_backfill_options()),
    ("score_sentiment", _stub_stage(
        "score_sentiment",
        "not yet implemented; will live in src/sentiment.py (FinBERT batch scoring)",
    )),
    ("compute_iv", _stub_stage(
        "compute_iv",
        "not yet implemented; will live in src/iv_surface.py (Black-Scholes inversion)",
    )),
    ("match_events", _stub_stage(
        "match_events",
        "not yet implemented; joins news → IV(t) → IV(t+30) into the events frame "
        "that evaluate_primary expects",
    )),
    ("fit_baselines", _stub_stage(
        "fit_baselines",
        "not yet implemented; runs src.baselines.compute_all_baselines on the "
        "training events frame and persists results",
    )),
    ("evaluate_validation", _stub_stage(
        "evaluate_validation",
        "not yet implemented; runs src.evaluate.evaluate_primary on the validation "
        "events frame. NOTE: never touch the holdout from the orchestrator — "
        "the §12 stopping rule allows exactly one holdout evaluation, done by hand.",
    )),
]


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def _configure_logging(repo_root: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(_log_path(repo_root), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_forever(repo_root: Path) -> int:
    _configure_logging(repo_root)
    logger.info("orchestrator starting; repo_root=%s", repo_root)
    try:
        while True:
            if is_paused(repo_root):
                logger.info("paused; sleeping %ds", PAUSED_POLL_SECONDS)
                time.sleep(PAUSED_POLL_SECONDS)
                continue

            state = load_state(repo_root)
            did_any_work = False
            for stage_name, stage_fn in STAGES:
                outcome = stage_fn(state, repo_root)
                if outcome.did_work:
                    if outcome.message:
                        logger.info("%s", outcome.message)
                    save_state(repo_root, state)
                    did_any_work = True
                    break  # one stage per iteration; cycle through fairly

            if not did_any_work:
                logger.info(
                    "nothing pending across all stages; sleeping %ds",
                    NOTHING_PENDING_SLEEP_SECONDS,
                )
                save_state(repo_root, state)
                time.sleep(NOTHING_PENDING_SLEEP_SECONDS)
            else:
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)
    except KeyboardInterrupt:
        logger.info("orchestrator interrupted by user; state already persisted")
        return 0


# ---------------------------------------------------------------------------
# status / pause / resume / reset CLI
# ---------------------------------------------------------------------------


def cmd_status(repo_root: Path) -> int:
    state = load_state(repo_root)
    paused = is_paused(repo_root)
    print(f"orchestrator state @ {_state_path(repo_root)}")
    print(f"  paused: {paused}")
    print(f"  updated_at_utc: {state.get('updated_at_utc', '<never>')}")
    print()
    print(f"{'stage':<24} {'status':<14} {'done/total':<12} {'notes'}")
    print("-" * 80)
    locked_months = _all_locked_months()
    for stage_name, _ in STAGES:
        stage_state = state.get("stages", {}).get(stage_name, {})
        status = stage_state.get("status", "pending")
        chunks = stage_state.get("chunks", {})
        if status == "skipped":
            done_total = "-"
            notes = stage_state.get("reason", "")[:60]
        else:
            complete = sum(1 for c in chunks.values() if c.get("status") == "complete")
            failed = sum(1 for c in chunks.values() if c.get("status") == "failed")
            done_total = f"{complete}/{len(locked_months)}"
            notes = f"{failed} failed" if failed else ""
        print(f"{stage_name:<24} {status:<14} {done_total:<12} {notes}")
    return 0


def cmd_pause(repo_root: Path) -> int:
    set_paused(repo_root, True)
    print(f"paused; orchestrator will halt after current chunk. Resume with: "
          f"python -m src.orchestrator resume")
    return 0


def cmd_resume(repo_root: Path) -> int:
    set_paused(repo_root, False)
    print("resumed")
    return 0


def cmd_reset(repo_root: Path, stage_name: str) -> int:
    state = load_state(repo_root)
    if stage_name not in {name for name, _ in STAGES}:
        print(f"unknown stage {stage_name!r}; known: {[n for n, _ in STAGES]}")
        return 2
    state.setdefault("stages", {}).pop(stage_name, None)
    save_state(repo_root, state)
    print(f"reset {stage_name}")
    return 0


def cmd_stages(_repo_root: Path) -> int:
    for name, _ in STAGES:
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("stages")
    reset = sub.add_parser("reset")
    reset.add_argument("stage")

    args = parser.parse_args(argv)
    repo_root = _find_repo_root()
    if args.command == "run":
        return run_forever(repo_root)
    if args.command == "status":
        return cmd_status(repo_root)
    if args.command == "pause":
        return cmd_pause(repo_root)
    if args.command == "resume":
        return cmd_resume(repo_root)
    if args.command == "stages":
        return cmd_stages(repo_root)
    if args.command == "reset":
        return cmd_reset(repo_root, args.stage)
    return 2


if __name__ == "__main__":
    sys.exit(main())
