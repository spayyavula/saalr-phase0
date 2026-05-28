# Forward-Test Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone repo `saalr-forward-test` that frozen-replays the locked Phase 0 primary-IC methodology on out-of-sample data (2026-05-01 → 2026-07-31) to test whether the signal's predictive IC persists — i.e., that it was not overfit.

**Architecture:** A new git repo that includes `saalr-phase0` as a submodule pinned to a fixed commit, imports its locked transforms verbatim through a thin `frozen` shim, and applies them to newly-pulled forward data written in a flat (non-split) layout. A small forward-specific `forward_spec` locks the window/threshold parameters; a provenance test makes the freeze structurally enforced. One-shot evaluation runs after Phase 0 publishes.

**Tech Stack:** Python 3.11, the pinned `saalr-phase0` modules (pandas, scipy, polygon-api-client, python-dotenv, pyarrow), git submodules.

**Submodule pin:** `saalr-phase0` @ `56c299f` (current HEAD — has the complete pipeline: sentiment, options, match_events, compute_iv with t−30, iv_surface, evaluate, baselines; `test_locked_spec.py` passes). If a later Phase 0 commit is preferred at build time, it MUST still pass `test_locked_spec.py` (filing-time SPEC fingerprint intact).

**Spec:** `docs/superpowers/specs/2026-05-28-forward-test-framework-design.md` (in saalr-phase0).

---

## Task 1: Scaffold the repo + submodule

**Files:**
- Create: `c:/Users/sreek/myprojects/saalr-forward-test/` (new repo, sibling of saalr-phase0)
- Create: `.gitignore`, `README.md`, `requirements.txt`, `.env.example`
- Submodule: `saalr-phase0/` pinned to `56c299f`

- [ ] **Step 1: Create repo + directory skeleton**

```bash
cd /c/Users/sreek/myprojects
mkdir saalr-forward-test && cd saalr-forward-test
git init
mkdir -p src tests decisions journal data
```

- [ ] **Step 2: Add saalr-phase0 as a submodule pinned to the locked commit**

```bash
git submodule add ../saalr-phase0 saalr-phase0
cd saalr-phase0 && git checkout 56c299f && cd ..
git add .gitmodules saalr-phase0
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.env
.env.local
*.env
data/forward/*
!data/MANIFEST.json
__pycache__/
*.pyc
.pytest_cache/
.venv/
.phase0/
```

- [ ] **Step 4: Write `requirements.txt`** (mirror the pinned phase0 deps; CPU torch is fine — no LSTM here)

```text
# Mirror of saalr-phase0 pins needed for the frozen primary-IC pipeline.
torch==2.4.1
transformers==4.45.2
sentencepiece==0.2.0
huggingface-hub==0.25.2
pandas==2.2.3
numpy==1.26.4
scipy==1.14.1
pyarrow==17.0.0
polygon-api-client==1.16.3
requests==2.32.3
python-dotenv==1.0.1
pytest==8.3.3
```

- [ ] **Step 5: Write `.env.example`**

```text
MASSIVE_API_KEY=
DATA_DIR=./data
```

- [ ] **Step 6: Write minimal `README.md`** stating purpose, the pinned commit, and the one-shot-after-Phase-0-publishes rule. (3-5 sentences; reference the spec.)

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt .env.example README.md
git commit -m "Scaffold saalr-forward-test with pinned saalr-phase0 submodule"
```

---

## Task 2: `frozen.py` — the import shim

**Files:**
- Create: `src/frozen.py`
- Test: `tests/test_frozen_imports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frozen_imports.py
def test_frozen_reexports_locked_symbols():
    from src import frozen
    assert frozen.SPEC.options_symbol == "SPY"
    # callables exist
    for name in ("score_articles", "fetch_contracts_for_month",
                 "fetch_option_mid_quote_at", "build_events_frame_for_month",
                 "compute_iv_for_events", "evaluate_primary"):
        assert callable(getattr(frozen, name)), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_frozen_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.frozen'`

- [ ] **Step 3: Write `src/frozen.py`**

```python
"""Single import surface for the FROZEN saalr-phase0 transforms.

The submodule uses a flat `src.` layout, so we put its root on sys.path
and re-export the locked symbols. All forward code imports from here, so
the freeze boundary is one file.
"""
import sys
from pathlib import Path

_PHASE0 = Path(__file__).resolve().parent.parent / "saalr-phase0"
if str(_PHASE0) not in sys.path:
    sys.path.insert(0, str(_PHASE0))

from src.locked_spec import SPEC  # noqa: E402
from src.sentiment import score_articles  # noqa: E402
from src.options import fetch_contracts_for_month, fetch_option_mid_quote_at  # noqa: E402
from src.match_events import build_events_frame_for_month  # noqa: E402
from src.compute_iv import compute_iv_for_events  # noqa: E402
from src.evaluate import evaluate_primary  # noqa: E402
from src.news import fetch_news  # noqa: E402
from src.underlying import fetch_underlying  # noqa: E402
from src.risk_free import fetch_risk_free  # noqa: E402

__all__ = [
    "SPEC", "score_articles", "fetch_contracts_for_month",
    "fetch_option_mid_quote_at", "build_events_frame_for_month",
    "compute_iv_for_events", "evaluate_primary", "fetch_news",
    "fetch_underlying", "fetch_risk_free",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_frozen_imports.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/frozen.py tests/test_frozen_imports.py
git commit -m "Add frozen import shim re-exporting locked phase0 transforms"
```

---

## Task 3: `tests/test_frozen_provenance.py` — enforce the freeze

**Files:**
- Create: `src/provenance.py` (constants), `tests/test_frozen_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frozen_provenance.py
import subprocess
from pathlib import Path

from src.provenance import PINNED_PHASE0_COMMIT, FILING_SPEC_FINGERPRINT
from src.frozen import SPEC

def test_submodule_is_at_pinned_commit():
    root = Path(__file__).resolve().parent.parent
    sha = subprocess.check_output(
        ["git", "-C", str(root / "saalr-phase0"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert sha == PINNED_PHASE0_COMMIT, (
        f"submodule at {sha}, expected {PINNED_PHASE0_COMMIT} — the freeze drifted"
    )

def test_spec_fingerprint_matches_filing():
    assert SPEC.fingerprint() == FILING_SPEC_FINGERPRINT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_frozen_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.provenance'`

- [ ] **Step 3: Capture the real values and write `src/provenance.py`**

First read the true fingerprint:
`PYTHONPATH=. .venv/Scripts/python.exe -c "from src.frozen import SPEC; print(SPEC.fingerprint())"`

Then write (substituting the printed fingerprint):

```python
"""Provenance pins that make the methodology freeze structurally enforced.

PINNED_PHASE0_COMMIT must match the submodule HEAD; FILING_SPEC_FINGERPRINT
must match the locked SPEC. If either drifts, the suite fails loudly.
"""
PINNED_PHASE0_COMMIT = "56c299fb549bba4b80bdbb0115bb01798fb20fe5"
FILING_SPEC_FINGERPRINT = "<paste the printed SPEC.fingerprint() here>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_frozen_provenance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/provenance.py tests/test_frozen_provenance.py
git commit -m "Pin submodule commit + SPEC fingerprint as a provenance test"
```

---

## Task 4: `forward_spec.py` — locked forward parameters

**Files:**
- Create: `src/forward_spec.py`
- Test: `tests/test_forward_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forward_spec.py
from src.forward_spec import FORWARD_SPEC

def test_window_is_three_months_post_phase0():
    assert FORWARD_SPEC.forward_start == "2026-05-01"
    assert FORWARD_SPEC.forward_end == "2026-07-31"

def test_min_events_matches_phase0_floor():
    assert FORWARD_SPEC.min_events == 1000

def test_fingerprint_is_stable():
    assert FORWARD_SPEC.fingerprint() == FORWARD_SPEC.fingerprint()
    assert len(FORWARD_SPEC.fingerprint()) == 64  # sha256 hex
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_forward_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.forward_spec'`

- [ ] **Step 3: Write `src/forward_spec.py`**

```python
"""Forward-test-specific locked parameters. The primary success criteria
(S1-S4) are inherited UNCHANGED from the frozen phase0 SPEC; only the
out-of-sample window and the event-count floor are pinned here."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True)
class ForwardSpec:
    forward_start: str = "2026-05-01"
    forward_end: str = "2026-07-31"
    min_events: int = 1000
    evaluate_only_after_phase0_publishes: bool = True

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


FORWARD_SPEC: Final[ForwardSpec] = ForwardSpec()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_forward_spec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/forward_spec.py tests/test_forward_spec.py
git commit -m "Lock forward window + event floor in forward_spec with fingerprint"
```

---

## Task 5: `forward_storage.py` — flat (non-split) writer

**Files:**
- Create: `src/forward_storage.py`
- Test: `tests/test_forward_storage.py`

**Rationale:** Phase 0's `storage.split_for_date` fails closed past `holdout_end` (2026-04-30) — that holdout guard must NOT be weakened. The forward window is entirely after it, so the forward repo writes a flat `data/forward/{source}/YYYY-MM.parquet` with no split routing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forward_storage.py
import importlib.util
import pandas as pd
import pytest

if importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow required", allow_module_level=True)

from src.forward_storage import write_forward_parquet

def test_writes_one_file_per_month(tmp_path):
    df = pd.DataFrame.from_records([
        {"timestamp": pd.Timestamp("2026-05-04 14:00", tz="UTC"), "v": 1.0},
        {"timestamp": pd.Timestamp("2026-06-04 14:00", tz="UTC"), "v": 2.0},
    ])
    results = write_forward_parquet(df, tmp_path, "underlying")
    paths = sorted(str(r.path.relative_to(tmp_path)).replace("\\", "/") for r in results)
    assert paths == ["forward/underlying/2026-05.parquet",
                     "forward/underlying/2026-06.parquet"]

def test_roundtrip_and_sha_recorded(tmp_path):
    df = pd.DataFrame.from_records([
        {"timestamp": pd.Timestamp("2026-05-04 14:00", tz="UTC"), "v": 1.0},
    ])
    results = write_forward_parquet(df, tmp_path, "events")
    assert results[0].row_count == 1
    assert len(results[0].sha256) == 64
    back = pd.read_parquet(tmp_path / "forward" / "events" / "2026-05.parquet")
    assert len(back) == 1

def test_empty_frame_writes_nothing(tmp_path):
    df = pd.DataFrame(columns=["timestamp", "v"])
    assert write_forward_parquet(df, tmp_path, "events") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_forward_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.forward_storage'`

- [ ] **Step 3: Write `src/forward_storage.py`**

```python
"""Flat parquet writer for the forward window — data/forward/{source}/YYYY-MM.parquet.

No split routing: the forward window is entirely post-holdout, so phase0's
split_for_date would (correctly) raise. We do NOT touch that guard; we just
write flat here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class WriteResult:
    path: Path
    row_count: int
    sha256: str


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_forward_parquet(df, data_root: Path, source: str,
                          timestamp_col: str = "timestamp") -> list[WriteResult]:
    import pandas as pd

    if df.empty:
        return []
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.assign(_ym=ts.dt.strftime("%Y-%m"))
    results: list[WriteResult] = []
    for (ym,), group in df.groupby(["_ym"]):
        out_dir = data_root / "forward" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ym}.parquet"
        clean = group.drop(columns=["_ym"])
        clean.to_parquet(out_path, index=False)
        results.append(WriteResult(out_path, len(clean), _sha256_of(out_path)))
    _update_manifest(data_root, results)
    return results


def _update_manifest(data_root: Path, results: list[WriteResult]) -> None:
    manifest_path = data_root / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {"files": {}}
    )
    for r in results:
        key = str(r.path.relative_to(data_root)).replace("\\", "/")
        manifest["files"][key] = {
            "rows": r.row_count, "sha256": r.sha256,
            "written_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_forward_storage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/forward_storage.py tests/test_forward_storage.py
git commit -m "Add flat forward_storage writer (no split routing)"
```

---

## Task 6: `collect.py` — pull the forward window

**Files:**
- Create: `src/collect.py`

**Note:** This is glue over the frozen, already-tested fetchers; validation is a smoke run, not a unit test (it hits the live API).

- [ ] **Step 1: Write `src/collect.py`**

```python
"""Pull the forward-window data (risk_free, underlying, news, options) using
the FROZEN phase0 fetchers, writing flat via forward_storage. Reuses the
frozen sentiment scoring too. Run per source; resumable by skipping months
whose output already exists."""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src import frozen
from src import forward_storage
from src.forward_spec import FORWARD_SPEC


def _months(start_iso: str, end_iso: str) -> list[str]:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    out, cur = [], start.replace(day=1)
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def _month_range(ym: str) -> tuple[date, date]:
    y, m = (int(p) for p in ym.split("-"))
    start = date(y, m, 1)
    nxt = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start, nxt - timedelta(days=1)


def _data_root() -> Path:
    return Path(os.environ.get("DATA_DIR", "./data"))


def collect_source(source: str) -> None:
    data_root = _data_root()
    for ym in _months(FORWARD_SPEC.forward_start, FORWARD_SPEC.forward_end):
        start, end = _month_range(ym)
        if source == "risk_free":
            df = frozen.fetch_risk_free(start, end)
        elif source == "underlying":
            df = frozen.fetch_underlying(start, end)
        elif source == "news":
            df = frozen.fetch_news(start, end)
        else:
            raise SystemExit(f"unknown source {source!r}")
        results = forward_storage.write_forward_parquet(df, data_root, source)
        for r in results:
            print(f"{r.path} rows={r.row_count} sha256={r.sha256[:12]}")


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["risk_free", "underlying", "news"])
    args = p.parse_args(argv)
    collect_source(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke — pull risk_free for the forward window (no auth, fast)**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m src.collect risk_free`
Expected: writes `data/forward/risk_free/2026-05.parquet` (and 06, 07) with ~20 rows each. Confirm via `python -c "import pandas as pd; print(pd.read_parquet('data/forward/risk_free/2026-05.parquet').shape)"`.

- [ ] **Step 3: Commit**

```bash
git add src/collect.py
git commit -m "Add forward-window collector over frozen phase0 fetchers"
```

---

## Task 7: options collection + sentiment scoring

**Files:**
- Modify: `src/collect.py` (add `options` + `sentiment` handling)

- [ ] **Step 1: Add options + sentiment to `collect_source`**

Options needs the underlying frame for the per-day open spot (frozen `fetch_contracts_for_month` signature: `(year_month, underlying_df)`). Sentiment reads the news parquet and scores it. Add these branches (reading prior-written flat files via pandas):

```python
# inside collect_source, extend the if/elif chain:
        elif source == "options":
            import pandas as pd
            up = data_root / "forward" / "underlying" / f"{ym}.parquet"
            if not up.exists():
                print(f"{ym}: no underlying yet; run `collect underlying` first")
                continue
            underlying_df = pd.read_parquet(up)
            df = frozen.fetch_contracts_for_month(ym, underlying_df)
        elif source == "sentiment":
            import pandas as pd
            np_ = data_root / "forward" / "news" / f"{ym}.parquet"
            if not np_.exists():
                print(f"{ym}: no news yet; run `collect news` first")
                continue
            df = frozen.score_articles(pd.read_parquet(np_))
```

Add `"options"` and `"sentiment"` to the argparse `choices`.

- [ ] **Step 2: Smoke — options for one forward month**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m src.collect underlying` then `... -m src.collect options`
Expected: `data/forward/options/2026-05.parquet` with a few thousand contract rows.

- [ ] **Step 3: Commit**

```bash
git add src/collect.py
git commit -m "Extend collector with options + frozen sentiment scoring"
```

---

## Task 8: `build_forward_events.py` + `compute_forward_iv.py`

**Files:**
- Create: `src/build_forward_events.py`, `src/compute_forward_iv.py`

- [ ] **Step 1: Write `src/build_forward_events.py`**

```python
"""Assemble the per-month forward events frame via frozen
build_events_frame_for_month, write flat. Loads prior-month sentiment too
so the 4h EWMA look-back is complete across month boundaries."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src import frozen, forward_storage
from src.forward_spec import FORWARD_SPEC


def _months(start_iso, end_iso):
    start, end = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    out, cur = [], start.replace(day=1)
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def _prev_month(ym):
    y, m = (int(p) for p in ym.split("-"))
    return f"{y-1:04d}-12" if m == 1 else f"{y:04d}-{m-1:02d}"


def _read(data_root, source, ym):
    p = data_root / "forward" / source / f"{ym}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    data_root = Path(os.environ.get("DATA_DIR", "./data"))
    for ym in _months(FORWARD_SPEC.forward_start, FORWARD_SPEC.forward_end):
        sent_frames = [_read(data_root, "sentiment", m) for m in (_prev_month(ym), ym)]
        sent_frames = [f for f in sent_frames if not f.empty]
        sentiment_df = pd.concat(sent_frames, ignore_index=True) if sent_frames else pd.DataFrame()
        underlying_df = _read(data_root, "underlying", ym)
        options_df = _read(data_root, "options", ym)
        rfr_df = _read(data_root, "risk_free", ym)
        events = frozen.build_events_frame_for_month(
            ym, sentiment_df, underlying_df, options_df, rfr_df
        )
        results = forward_storage.write_forward_parquet(events, data_root, "events")
        for r in results:
            print(f"{r.path} rows={r.row_count} sha256={r.sha256[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `src/compute_forward_iv.py`**

```python
"""Apply the frozen IV inversion to each month's forward events frame and
write data/forward/iv/{ym}.parquet."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src import frozen, forward_storage
from src.forward_spec import FORWARD_SPEC
from src.build_forward_events import _months  # reuse


def main(argv=None) -> int:
    data_root = Path(os.environ.get("DATA_DIR", "./data"))
    for ym in _months(FORWARD_SPEC.forward_start, FORWARD_SPEC.forward_end):
        p = data_root / "forward" / "events" / f"{ym}.parquet"
        if not p.exists():
            continue
        iv = frozen.compute_iv_for_events(pd.read_parquet(p))
        results = forward_storage.write_forward_parquet(iv, data_root, "iv")
        for r in results:
            print(f"{r.path} rows={r.row_count} sha256={r.sha256[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke — build events + IV for one already-collected forward month**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m src.build_forward_events` then `... -m src.compute_forward_iv`
Expected: `data/forward/events/2026-05.parquet` and `data/forward/iv/2026-05.parquet` written; confirm the iv file has `signal`, `forward_iv_change`, `prior_iv_change` columns.

- [ ] **Step 4: Commit**

```bash
git add src/build_forward_events.py src/compute_forward_iv.py
git commit -m "Add forward events assembly + frozen IV inversion stages"
```

---

## Task 9: `evaluate_forward.py` — the one-shot evaluation

**Files:**
- Create: `src/evaluate_forward.py`
- Test: `tests/test_evaluate_forward.py`

- [ ] **Step 1: Write the failing test** (concatenation + n-floor guard logic, no network)

```python
# tests/test_evaluate_forward.py
import importlib.util
import pandas as pd
import pytest

if importlib.util.find_spec("scipy") is None:
    pytest.skip("scipy required", allow_module_level=True)

from src.evaluate_forward import load_forward_iv

def test_load_forward_iv_concatenates_months(tmp_path):
    d = tmp_path / "forward" / "iv"
    d.mkdir(parents=True)
    for ym, n in (("2026-05", 3), ("2026-06", 2)):
        pd.DataFrame({
            "timestamp": pd.to_datetime([f"{ym}-04 14:00"] * n, utc=True),
            "signal": [0.1] * n, "forward_iv_change": [0.01] * n,
            "prior_iv_change": [0.0] * n,
        }).to_parquet(d / f"{ym}.parquet")
    out = load_forward_iv(tmp_path)
    assert len(out) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_evaluate_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluate_forward'`

- [ ] **Step 3: Write `src/evaluate_forward.py`**

```python
"""One-shot forward evaluation: concatenate all forward IV months, run the
FROZEN evaluate_primary against the SAME S1-S4 thresholds, write the verdict.

GATE: this is the single permitted forward evaluation, run only after Phase 0
publishes (FORWARD_SPEC.evaluate_only_after_phase0_publishes). The gate is a
human checkpoint — the function prints a reminder and requires --confirm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src import frozen
from src.forward_spec import FORWARD_SPEC


def load_forward_iv(data_root: Path) -> pd.DataFrame:
    d = data_root / "forward" / "iv"
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    return (
        pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if frames else pd.DataFrame()
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true",
                   help="Required: confirms Phase 0 has published and this is the one-shot.")
    args = p.parse_args(argv)
    if not args.confirm:
        raise SystemExit(
            "Refusing to run: the forward evaluation is one-shot and must run "
            "only AFTER Phase 0 publishes. Re-run with --confirm when that holds."
        )
    data_root = Path(os.environ.get("DATA_DIR", "./data"))
    events = load_forward_iv(data_root)
    usable = events[events.get("forward_iv_change").notna()] if len(events) else events
    print(f"forward events: total={len(events)} usable={len(usable)} "
          f"(floor {FORWARD_SPEC.min_events})")
    result = frozen.evaluate_primary(usable, label="forward_primary")
    out_path = result.write(Path("results"))
    print(f"verdict: passes_all={result.verdict.passes_all} "
          f"IC={result.primary_ic.ic:.4f} p={result.primary_ic.p_value:.4g} "
          f"grey_zone={result.in_grey_zone}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_evaluate_forward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluate_forward.py tests/test_evaluate_forward.py
git commit -m "Add one-shot forward evaluation gated behind --confirm"
```

---

## Task 10: `pre-registration-forward.md` + final test run

**Files:**
- Create: `pre-registration-forward.md`

- [ ] **Step 1: Write `pre-registration-forward.md`**

Mirror the Phase 0 pre-reg structure for the forward test. Lock: hypothesis (the frozen primary signal retains IC ≥ 0.05, p < 0.01, ≥ 1.5× the same baselines, sign-consistent across the two halves of 2026-05-01 → 2026-07-31); n ≥ 1000; one-shot evaluation after Phase 0 publishes; publish regardless of outcome; reference the pinned saalr-phase0 commit + SPEC fingerprint. (Full prose; mirror sections 1-13 of saalr-phase0/pre-registration.md, scoped to the primary IC only.)

- [ ] **Step 2: Run the full suite**

Run: `PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q`
Expected: all PASS (frozen imports, provenance, forward_spec, forward_storage, evaluate_forward).

- [ ] **Step 3: Commit + (manual) file the forward gist**

```bash
git add pre-registration-forward.md
git commit -m "File forward-test pre-registration (primary IC, one-shot, post-publish)"
```

The forward gist is published manually before any forward IC is computed, mirroring Phase 0.

---

## Execution notes

- The forward DATA collection (Tasks 6-8 at full scale over 3 months) reuses the same rate-limited Polygon path as Phase 0 match_events, so it inherits the quote-fetch resilience (retry + skip) committed in saalr-phase0 `56c299f`. Expect a multi-hour collection for the options/events/IV stages, same as Phase 0.
- `evaluate_forward` must NOT be run until Phase 0 has published its holdout result. The `--confirm` flag is the tripwire.
- Do not modify anything under `saalr-phase0/` — it is the frozen submodule. Any methodology change happens in Phase 0, is committed there, and the submodule pin is advanced deliberately (with the provenance test updated).
