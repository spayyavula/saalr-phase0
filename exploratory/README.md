# Exploratory analyses

Everything in this directory is **exploratory** per `pre-registration.md`
§10: reported in the writeup, but **cannot — individually or collectively —
justify a success claim** beyond what §9 (primary IC) and §9b (Variant B)
establish.

These analyses run on the **train** and **validation** splits only. They
never constitute an additional holdout evaluation; the §12 stopping rule
permits exactly one holdout look, done by hand in Week 7.

- `rumour_window_check.py` — compares `Spearman(signal, forward_iv_change)`
  (the locked post-news relationship) against `Spearman(signal,
  prior_iv_change)` (the anticipation/run-up relationship), probing the
  "buy the rumour, sell the news" interpretation logged in
  `decisions/2026-05-28_buy-rumour-sell-news-interpretation.md`.
