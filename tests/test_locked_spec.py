"""Forcing function for the locked pre-registered spec.

This test fails if any value in ``src.locked_spec.LockedSpec`` is changed
without also updating the pinned fingerprint below. Updating the
fingerprint is a deliberate act that must be paired with a decision-log
entry under ``decisions/`` per the contract in ``src/locked_spec.py``.

The point: an honest accident (somebody nudges a threshold while
refactoring) becomes a noisy red test, not a silent shift in the
pre-registered methodology.
"""
from src.locked_spec import SPEC

PINNED_FINGERPRINT_2026_05_27 = (
    "2d2020230bfaf3530aff08e36ed969a5b533e5e6df2376ca183bbeb0728a5814"
)


def test_locked_spec_fingerprint_unchanged() -> None:
    actual = SPEC.fingerprint()
    assert actual == PINNED_FINGERPRINT_2026_05_27, (
        "\n\nThe pre-registered locked spec has changed.\n"
        "Expected fingerprint (pinned at filing, 2026-05-27):\n"
        f"  {PINNED_FINGERPRINT_2026_05_27}\n"
        "Actual fingerprint (current src/locked_spec.py):\n"
        f"  {actual}\n\n"
        "If this change is intentional (e.g., the Week 3 addendum locking\n"
        "LSTM/GARCH specs or the holdout SHA-256), follow the process in\n"
        "src/locked_spec.py's module docstring:\n"
        "  1. Update the value.\n"
        "  2. Update PINNED_FINGERPRINT_2026_05_27 above (and rename to the\n"
        "     new filing date).\n"
        "  3. Add a decision-log entry under decisions/ describing the\n"
        "     public addendum that accompanies this change.\n"
        "If this change is NOT intentional, revert it. The pre-registration\n"
        "is a binding public commitment and is the entire point of Phase 0.\n"
    )
