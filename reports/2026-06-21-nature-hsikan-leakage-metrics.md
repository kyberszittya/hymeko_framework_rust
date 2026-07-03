# Nature-hsikan: leakage metrics (L, Δ) from the shuffle audit

*2026-06-21 · Aiko (Claude Code) for Dr. Csaba Hajdu*

## Summary

Defined and computed two scalar leakage metrics for the audit paper, from the committed
JSONL (no new experiments): **L = max(0, shuffle_AUROC − 0.5)** (residual leak surviving the
train-label shuffle) and **Δ = real_AUROC − shuffle_AUROC** (audit drop). Per (method,
dataset), 5-seed mean ± 95% CI (Student-t). Tables emitted in markdown + Springer LaTeX.

## Builder

`hymeko_neuro/paperkit/build_leakage_metric_table.py` (matches the figure-builder
conventions; `--self-check`). Reads `no_leak_baselines.jsonl` (strict, 350 rows),
`no_leak_baselines_topo.jsonl` (420→350 after last-wins dedup of the re-run seed 0),
`cycle_reachability_grid.jsonl` (60). Emits to `hymeko_neuro/assets/paper/tables/`:
`leakage_metrics.md` + `leakage_metrics_baselines.tex`.

## Headline numbers

**Baselines (mean over 5 datasets):** L = 0.015–0.045, and **L ≈ L_topo** (e.g. dadsgnn
0.022 vs 0.023) → the topology-shuffle residual equals the strict one ⇒ **no structural
leak**; Δ ≈ 0.33–0.37 (most of the real score is shuffle-destructible = legitimate). sigat
cleanest (L 0.015).

**By dataset:** the residual leak concentrates on **reddit_body** (L 0.06–0.09, flagged for
6/7 methods) and partly sesgformer/bitcoin; epinions & slashdot → L ≈ 0 (clean).

**Cycle/HSiKAN method across the lattice:** strict 0.500/0.500 (degenerate), **topo L = 0**
(real 0.80–0.84, shuffle→chance — clean), **full L = 0.16–0.24** (leaks). The leak is the
label-reachable channel at `full`, not the topology — the metric companion to the figures.

## Test results

- `pytest hymeko_neuro/tests/test_build_leakage_metric_table.py` — **5 passed** (L/Δ
  definitions, seed dedup last-wins, real-data smoke shows the 7 methods + reddit_body leak,
  cycle leaks only at full). `--self-check` OK. `ruff` clean.

## CORE.YAML / dependencies

**None.** `hymeko_neuro` + `paper/tables` (non-core).

## Open / follow-up (the rest of "what's needed" for the paper)

- **Needs a run:** explain the reddit_body residual (degree-preserving rewire / per-node-prior
  control) — the one scientific gap.
- **Needs author:** bib `% VERIFY` items; wire the EPS figures + these LaTeX tables into
  `01_analysis/.../nature_hsikan/`; reproduction-parity caption.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). Data files dated 2026-06-14/15. No new
experiments — pure aggregation.
