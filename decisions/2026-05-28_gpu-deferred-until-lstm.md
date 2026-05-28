# 2026-05-28 — GPU usage deferred until the Week 3 LSTM addendum

## Decision

Keep the CPU-only `torch==2.4.1+cpu` wheel installed for the
remainder of Phase 0's current work. Defer installing a CUDA-enabled
torch wheel until the Week 3 pre-registration addendum locks the
LSTM specification (§9b Variant B). Switch to GPU torch at that
point — that is the first stage where the GPU is both a clear win
and free of pre-registration-integrity concerns.

## Context

The laptop has an NVIDIA GeForce RTX 3050 (4 GB VRAM, CUDA-capable
via driver 32.0.15.5599). The current `requirements.txt` pins
`torch==2.4.1` which resolved to the `+cpu` build at install time,
so `torch.cuda.is_available()` is False and FinBERT batch inference
has been running on CPU. This was observed today after
`backfill_options` was wired and the user noticed `WS` and CPU on
the orchestrator process were both modest.

The GPU would meaningfully help exactly two stages of the project:

| stage | helps? | why |
| --- | --- | --- |
| FinBERT sentiment scoring | yes | batch transformer inference, embarrassingly parallel; ~6m45s/month on CPU drops to seconds/month |
| compute_iv (BS inversion) | no | `scipy.optimize.brentq` is scalar; CPU-bound by design |
| match_events mid-quote pulls | no | network-bound on Polygon API; rate-limited per account |
| LSTM (§9b Variant B, deferred to Week 3 addendum) | yes | model training is the canonical GPU workload; the spec is not yet locked, so this is where adding GPU first matters |
| baselines / evaluation | no | tiny numpy/scipy ops |

Of those, only FinBERT sentiment and LSTM are GPU-relevant.

## Alternatives considered

1. **Install GPU torch now and re-score the existing sentiment
   parquets.** Tempting because the speedup is real (~6m45s/month
   → seconds). Rejected. The 28-month sentiment backfill is already
   complete and its sha256s are pinned in `data/MANIFEST.json`. GPU
   and CPU FinBERT inference produce **bit-different** scores in
   the 4th-6th decimal place because fused-kernel reduction order
   differs across devices. Re-scoring would silently change the
   sha256 of every `train/sentiment/*.parquet` file with no
   scientific benefit — the logical signal is the same. Every
   change to a locked artifact in a pre-registered project should
   be auditable and reasoned, and "we wanted the same number faster"
   is not a reason that justifies overwriting an integrity record.

2. **Install GPU torch now but only use it for new work.** Possible
   but awkward — every sentiment-scoring call would need to know
   "this contract was already CPU-scored, don't re-do it on GPU."
   The `score_sentiment` orchestrator stage already has this
   on-disk skip logic, so in practice this is what we'd get
   automatically once Jan 2024 is on disk. But pre-installing GPU
   torch with no immediate consumer just adds 2.5 GB of wheels and
   a future driver/CUDA version-skew risk to the venv for no
   present benefit.

3. **Defer until Week 3 LSTM addendum (chosen).** The Week 3
   pre-reg addendum will lock the LSTM architecture, training
   window, and hyperparameters per the §9b commitment. That is
   the first moment we have a new GPU-relevant workload with no
   pre-existing artifact to overwrite. Install GPU torch then;
   train the LSTM on the GPU; verify reproducibility with a fixed
   seed; record the device choice as part of the addendum.

## Rationale

The pre-registration discipline treats every locked or pinned
artifact as something whose modification has to be justified, not
"because it could be done." Re-scoring sentiment to get the same
logical answer on different hardware would replace a pinned
artifact with one that has a different sha256 — and would have to
be reported in the §13 publication writeup as "we re-ran feature
engineering after seeing partial results." That is the exact shape
of post-hoc tampering the pre-reg is built to prevent, even when
the actual motivation is entirely innocent.

The LSTM is a different situation. Its spec is not yet locked;
the Week 3 addendum will lock it. At that locking moment we can
write into the addendum "trained on NVIDIA RTX 3050 with seed S,
weights sha256 pinned" and it is part of the locked spec from
inception. No revision of any prior artifact.

## What to do when the time comes

Per Week 3 LSTM addendum:

```powershell
# In the .venv:
pip uninstall -y torch
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The RTX 3050 supports CUDA 12.x; the cu121 wheel is the right
choice for the current GeForce driver. If the driver is older or
the wheel is unavailable, fall back to `cu118`. `requirements.txt`
should be updated to a pinned `+cu121` wheel via a separate
`requirements-gpu.txt` so the CPU-only path stays a viable
reproducibility target for anyone without a CUDA GPU.

## Reversible?

Yes. The CPU torch in the current venv is unchanged. Adopting GPU
torch later is a `pip uninstall && pip install` away. No on-disk
artifacts change. If GPU FinBERT inference is ever needed for a
re-pull of a corrupted month, the procedure above will be run and
the resulting sha256 difference will be flagged in a follow-on
decision-log entry.

## Related

- [2026-05-27_data-acquisition-architecture.md] — initial pin of `torch==2.4.1` resolved to the `+cpu` wheel.
- §9b Variant B in [pre-registration.md] — locks LSTM scaffold; spec to land in Week 3 addendum.
