"""Top-level evaluation per §§6–9 (primary IC) and §9b (Variant B economic lift).

This module is the **single shot** the pre-registration permits on the
holdout. It computes IC, bootstrap CI, baseline ICs, sub-period stability,
and decides pass/fail against the locked S1–S4 (primary) and E1–E5
(Variant B) criteria.

Sub-period stability per §8: holdout split into two equal halves **by
time** (calendar-midpoint of the time range), sign of IC must agree in
both halves.

Output is a JSON-serializable ``EvaluationResult`` that lands in
``results/`` for the publication commitment in §13.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import BaselineResults, compute_all_baselines
from src.locked_spec import SPEC
from src.metrics import ICResult, bootstrap_ic_ci, hit_rate, sharpe_approx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubperiodCheck:
    midpoint: str
    first_half_ic: float
    first_half_n: int
    second_half_ic: float
    second_half_n: int
    signs_agree: bool


@dataclass(frozen=True)
class PrimaryVerdict:
    s1_ic_meets_threshold: bool
    s2_pvalue_meets_threshold: bool
    s3_beats_baseline_margin: bool
    s4_subperiod_sign_consistent: bool

    @property
    def grey_zone(self) -> bool:
        """Helper: not used in the locked decision, but flags the explicit
        §9 grey-zone (IC in 0.03-0.05) for the writeup."""
        # Note: grey-zone classification needs the IC value; the harness
        # surfaces it separately in EvaluationResult.in_grey_zone.
        return False

    @property
    def passes_all(self) -> bool:
        return (
            self.s1_ic_meets_threshold
            and self.s2_pvalue_meets_threshold
            and self.s3_beats_baseline_margin
            and self.s4_subperiod_sign_consistent
        )


@dataclass(frozen=True)
class EvaluationResult:
    label: str
    n_events: int
    primary_ic: ICResult
    baselines: BaselineResults
    subperiod: SubperiodCheck
    verdict: PrimaryVerdict
    in_grey_zone: bool
    computed_at_utc: str
    spec_fingerprint: str

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    def write(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{self.label}.json"
        out_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return out_path


def _midpoint(timestamps: pd.Series) -> pd.Timestamp:
    earliest = timestamps.min()
    latest = timestamps.max()
    return earliest + (latest - earliest) / 2


def subperiod_stability_check(events: pd.DataFrame) -> SubperiodCheck:
    ts = pd.to_datetime(events["timestamp"], utc=True)
    midpoint = _midpoint(ts)
    first_mask = ts < midpoint
    second_mask = ~first_mask

    first_ic = float("nan")
    second_ic = float("nan")
    if first_mask.sum() >= 2:
        r = bootstrap_ic_ci(
            events.loc[first_mask, "signal"],
            events.loc[first_mask, "forward_iv_change"],
            resamples=100,  # subperiod CI is informational, not pinned
        )
        first_ic = r.ic
    if second_mask.sum() >= 2:
        r = bootstrap_ic_ci(
            events.loc[second_mask, "signal"],
            events.loc[second_mask, "forward_iv_change"],
            resamples=100,
        )
        second_ic = r.ic

    signs_agree = (
        not np.isnan(first_ic)
        and not np.isnan(second_ic)
        and np.sign(first_ic) == np.sign(second_ic)
        and np.sign(first_ic) != 0
    )
    return SubperiodCheck(
        midpoint=str(midpoint),
        first_half_ic=first_ic,
        first_half_n=int(first_mask.sum()),
        second_half_ic=second_ic,
        second_half_n=int(second_mask.sum()),
        signs_agree=signs_agree,
    )


def evaluate_primary(
    events: pd.DataFrame,
    *,
    label: str,
    seed: int = 42,
) -> EvaluationResult:
    """Run the locked §§6–9 evaluation on a matched-events frame.

    ``events`` columns required:

    - ``timestamp``  — event UTC datetime
    - ``signal``     — candidate signal value at the event
    - ``forward_iv_change`` — realized IV(t+30) − IV(t)
    - ``prior_iv_change``   — realized IV(t) − IV(t−30) (needed by B1)
    """
    n = len(events)
    if n < SPEC.min_holdout_events:
        logger.warning(
            "evaluate_primary called with n=%d events; pre-reg §8 requires n >= %d",
            n,
            SPEC.min_holdout_events,
        )

    primary = bootstrap_ic_ci(events["signal"], events["forward_iv_change"], seed=seed)
    baselines = compute_all_baselines(events)
    subperiod = subperiod_stability_check(events)

    in_grey_zone = (
        SPEC.grey_zone_lower_exclusive
        < abs(primary.ic)
        < SPEC.grey_zone_upper_exclusive
    )

    verdict = PrimaryVerdict(
        s1_ic_meets_threshold=abs(primary.ic) >= SPEC.s1_min_holdout_ic,
        s2_pvalue_meets_threshold=primary.p_value < SPEC.s2_max_holdout_pvalue,
        s3_beats_baseline_margin=(
            abs(primary.ic)
            >= SPEC.s3_baseline_margin_multiplier * baselines.strongest_ic
        ),
        s4_subperiod_sign_consistent=subperiod.signs_agree,
    )

    return EvaluationResult(
        label=label,
        n_events=n,
        primary_ic=primary,
        baselines=baselines,
        subperiod=subperiod,
        verdict=verdict,
        in_grey_zone=in_grey_zone,
        computed_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        spec_fingerprint=SPEC.fingerprint(),
    )
