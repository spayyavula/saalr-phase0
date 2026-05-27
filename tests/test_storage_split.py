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
