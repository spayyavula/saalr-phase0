"""Forcing function for ``split_for_date`` against the locked windows.

These tests are pure Python — they do not import pandas or pyarrow, so they
run before ``pip install -r requirements.txt``.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.locked_spec import SPEC
from src.storage import split_for_date


def test_train_boundaries() -> None:
    assert split_for_date(SPEC.train_start) == "train"
    assert split_for_date(SPEC.train_end) == "train"


def test_validation_boundaries() -> None:
    assert split_for_date(SPEC.validation_start) == "validation"
    assert split_for_date(SPEC.validation_end) == "validation"


def test_holdout_boundaries() -> None:
    assert split_for_date(SPEC.holdout_start) == "holdout"
    assert split_for_date(SPEC.holdout_end) == "holdout"


def test_no_gap_between_splits() -> None:
    """Splits must be contiguous: end-of-train + 1 day == start-of-validation,
    and end-of-validation + 1 day == start-of-holdout. A silent gap would let
    a row fall through to a ValueError instead of being correctly classified."""
    train_end = date.fromisoformat(SPEC.train_end)
    validation_start = date.fromisoformat(SPEC.validation_start)
    validation_end = date.fromisoformat(SPEC.validation_end)
    holdout_start = date.fromisoformat(SPEC.holdout_start)
    assert validation_start - train_end == timedelta(days=1)
    assert holdout_start - validation_end == timedelta(days=1)


def test_outside_all_splits_raises() -> None:
    too_early = date.fromisoformat(SPEC.train_start) - timedelta(days=1)
    too_late = date.fromisoformat(SPEC.holdout_end) + timedelta(days=1)
    with pytest.raises(ValueError):
        split_for_date(too_early)
    with pytest.raises(ValueError):
        split_for_date(too_late)


def test_iso_string_input() -> None:
    assert split_for_date("2024-06-15") == "train"
    assert split_for_date("2026-01-15") == "validation"
    assert split_for_date("2026-03-15") == "holdout"


# -- end-to-end partitioning across a mid-month boundary --------------------
# SPEC's validation_end = 2026-02-15 / holdout_start = 2026-02-16 boundary
# falls in the middle of February. write_partitioned_parquet originally
# inferred the split from the first row of each month group, silently
# routing all of Feb 2026 to validation. These tests pin the per-row
# routing so the bug cannot recur unnoticed.


def _pandas_available() -> bool:
    import importlib.util

    return all(
        importlib.util.find_spec(m) is not None for m in ("pandas", "pyarrow")
    )


@pytest.mark.skipif(
    not _pandas_available(),
    reason="pandas + pyarrow required for write_partitioned_parquet",
)
def test_write_partitioned_parquet_routes_mid_month_boundary(tmp_path):
    import pandas as pd

    from src.storage import write_partitioned_parquet

    # 6 rows: 3 in the validation half of Feb 2026, 3 in the holdout half.
    rows = [
        {"timestamp": pd.Timestamp("2026-02-10 12:00", tz="UTC"), "value": 1.0},
        {"timestamp": pd.Timestamp("2026-02-14 12:00", tz="UTC"), "value": 2.0},
        {"timestamp": pd.Timestamp("2026-02-15 23:59", tz="UTC"), "value": 3.0},
        {"timestamp": pd.Timestamp("2026-02-16 00:00", tz="UTC"), "value": 4.0},
        {"timestamp": pd.Timestamp("2026-02-20 12:00", tz="UTC"), "value": 5.0},
        {"timestamp": pd.Timestamp("2026-02-28 23:59", tz="UTC"), "value": 6.0},
    ]
    df = pd.DataFrame.from_records(rows)
    results = write_partitioned_parquet(df, tmp_path, "smoke")

    # Should produce two files: one validation, one holdout.
    by_split = {str(r.path.relative_to(tmp_path)).replace("\\", "/"): r for r in results}
    val_path = "validation/smoke/2026-02.parquet"
    hold_path = "holdout/smoke/2026-02.parquet"
    assert val_path in by_split, f"validation file missing; got {list(by_split)}"
    assert hold_path in by_split, f"holdout file missing; got {list(by_split)}"
    assert by_split[val_path].row_count == 3
    assert by_split[hold_path].row_count == 3

    val_df = pd.read_parquet(tmp_path / val_path)
    hold_df = pd.read_parquet(tmp_path / hold_path)
    val_dates = pd.to_datetime(val_df["timestamp"], utc=True).dt.date
    hold_dates = pd.to_datetime(hold_df["timestamp"], utc=True).dt.date
    assert val_dates.max() == date(2026, 2, 15)
    assert hold_dates.min() == date(2026, 2, 16)


@pytest.mark.skipif(
    not _pandas_available(),
    reason="pandas + pyarrow required for write_partitioned_parquet",
)
def test_write_partitioned_parquet_pure_month_still_one_file(tmp_path):
    """A month that sits cleanly inside one split still writes one file."""
    import pandas as pd

    from src.storage import write_partitioned_parquet

    df = pd.DataFrame.from_records([
        {"timestamp": pd.Timestamp("2024-06-03 12:00", tz="UTC"), "value": 1.0},
        {"timestamp": pd.Timestamp("2024-06-28 12:00", tz="UTC"), "value": 2.0},
    ])
    results = write_partitioned_parquet(df, tmp_path, "smoke")
    assert len(results) == 1
    assert str(results[0].path.relative_to(tmp_path)).replace("\\", "/") == (
        "train/smoke/2024-06.parquet"
    )
    assert results[0].row_count == 2


@pytest.mark.skipif(
    not _pandas_available(),
    reason="pandas + pyarrow required for read_month_across_splits",
)
def test_read_month_across_splits_unions_boundary_files(tmp_path):
    """A boundary month with rows in two splits must read back as one frame."""
    import pandas as pd

    from src.storage import read_month_across_splits, write_partitioned_parquet

    df = pd.DataFrame.from_records([
        {"timestamp": pd.Timestamp("2026-02-10 12:00", tz="UTC"), "value": 1.0},
        {"timestamp": pd.Timestamp("2026-02-15 23:59", tz="UTC"), "value": 2.0},
        {"timestamp": pd.Timestamp("2026-02-16 00:00", tz="UTC"), "value": 3.0},
        {"timestamp": pd.Timestamp("2026-02-20 12:00", tz="UTC"), "value": 4.0},
    ])
    write_partitioned_parquet(df, tmp_path, "smoke")

    unioned = read_month_across_splits(tmp_path, "smoke", "2026-02")
    assert len(unioned) == 4
    assert list(unioned["value"]) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.skipif(
    not _pandas_available(),
    reason="pandas + pyarrow required for read_month_across_splits",
)
def test_read_month_across_splits_missing_month_returns_empty(tmp_path):
    import pandas as pd

    from src.storage import read_month_across_splits

    result = read_month_across_splits(tmp_path, "smoke", "2024-06")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
