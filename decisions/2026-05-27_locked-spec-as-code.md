# 2026-05-27 — Encode the locked pre-registration spec as code

## Decision

All locked values from `pre-registration.md` §§3–13 are encoded as a frozen
dataclass in `src/locked_spec.py`, and a pytest fingerprint test in
`tests/test_locked_spec.py` fails whenever any value is changed without
explicitly updating the pinned hash.

## Context

The pre-registration was filed publicly today as a GitHub gist and
mirrored to `pre-registration.md` in this repo. The discipline only works
if it's hard to drift from. A markdown file in a repo is easy to edit; a
constant referenced by every downstream module and guarded by a CI-runnable
test is hard to edit by accident.

## Alternatives considered

1. **Leave the spec in markdown only.** Downstream modules would read the
   same constants from comments or hard-code their own copies — drift is
   guaranteed over a 7-week timeline.
2. **One constant module, no fingerprint test.** Better, but a quiet diff
   could still slip through during a late-night refactor or an AI-assisted
   change. The whole point of pre-registration is to make this kind of drift
   impossible to do without acknowledgment.
3. **Encode constants + fingerprint test.** Chosen. Editing any locked
   value now requires (a) changing the constant, (b) updating the pinned
   hash in the test, and (c) adding a decision-log entry — three signals
   that you are knowingly amending the pre-registered methodology.

## Rationale

The fingerprint test reframes "edit a number" as "amend the pre-registration."
Three deliberate edits beats one accidental nudge. This forcing function
costs roughly zero ongoing maintenance: it only fires when the spec
actually changes, and the failure message points the author at the
correct documented update process.

The deferred items in `pre-registration.md` (LSTM spec, GARCH spec, holdout
SHA-256 hash) are intentionally absent from `LockedSpec` today. The Week 3
addendum will add them as new fields, recompute the fingerprint, update
the test pin, and ship a paired decision-log entry. This makes the
"addendum" workflow concrete and repeatable.

## Reversible?

Yes. The locked-spec module and test can be replaced with plainer code
if it becomes friction (it should not). Reverting does not affect the
pre-registration itself — only the in-repo guard against drift.
