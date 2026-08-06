# Report — Leakage-by-source figures for the Nature HSiKAN audit paper

**Date:** 2026-06-20
**Slug:** `nature-hsikan-leakage-source-figures`
**Author:** Aiko (for Dr. Csaba Hajdu)
**Plan:** `docs/plans/2026-06-20-nature-hsikan-leakage-source-figures/` (tex/pdf/tikz/mmd)

## Summary

Visualisation-only task (no new experiments). Produced two new figures that make
the *leakage-by-source* distinction visually explicit on the train-only
label-shuffle audit:

1. **`leakage_lattice`** — the cycle/HSiKAN method across the reachability lattice
   `strict ⊑ topo ⊑ full`, real vs shuffled-train-labels. The shuffled bar rises
   above the leak line **only at `full`** (held-out sign reachable in its own cycle
   σ-product); `topo` keeps the real signal high while its shuffle bar collapses to
   chance. Message: *cycle topology is a clean feature; the direct label-sign
   channel is the leak.*
2. **`leakage_source_contrast`** — two panels side by side. Left: the seven
   diffuse / "activated-neuron" readout baselines, each with both shuffle arms
   (strict and topo) sitting at chance → no structural leak. Right: the
   local-readout cycle method, clean at `topo` (0.482) and leaking at `full`
   (0.703). Message: *activated-neuron representations are structurally leak-free;
   the local cycle-sign channel is where leakage concentrates.*

All numbers were verified against the data files and
`reports/2026-06-15-cycle-leak-reachability-verdict.md` before plotting. No
numbers invented; no panel omitted (all intended data present).

## Figures produced (paths)

```
hymeko_neuro/assets/paper/figures/leakage_lattice.{pdf,png,eps}
hymeko_neuro/assets/paper/figures/leakage_source_contrast.{pdf,png,eps}
```

EPS emitted for Springer (a cosmetic "PostScript backend does not support
transparency" warning is benign — the grid is rendered opaque, same as the
existing `build_leakage_audit_figure.py`).

## Exact numbers visualised

### Lattice figure (cycle/HSiKAN method, pooled over the two datasets × 5 seeds)
Source: `cycle_reachability_grid.jsonl` (60 rows: bitcoin_alpha + bitcoin_otc ×
{strict,topo,full} × seeds 0–4 × {real,shuffle}; 10 real + 10 shuffle cells/rule).

| rule | real AUROC | shuffle AUROC | reading |
|---|---|---|---|
| strict | 0.500 | 0.500 | degenerate (no features) |
| topo | 0.822 | **0.482** | strong feature, **no leak** (shuffle below chance) |
| full | 0.877 | **0.703** | **leaks** (shuffle ≫ 0.55 threshold) |

Consistent with the verdict report's single-seed bitcoin_alpha values
(topo 0.806/0.467, full 0.901/0.735); the figure pools both bitcoin datasets ×
5 seeds for error bars.

### Source-contrast figure
Left panel source: `no_leak_baselines.jsonl` (real + strict-shuffle, 350 rows) and
`no_leak_baselines_topo.jsonl` (topo-shuffle, 420 rows, seed-0 re-runs deduped
last-wins). 25 distinct cells per model per arm (5 datasets × 5 seeds).

| model | strict real | strict shuf | topo shuf |
|---|---|---|---|
| SGCN | 0.870 | 0.523 | 0.526 |
| SiGAT | 0.888 | 0.515 | 0.517 |
| SGT | 0.886 | 0.532 | 0.530 |
| SGCL | 0.877 | 0.524 | 0.521 |
| SiGformer | 0.884 | 0.522 | 0.516 |
| SE-SGformer | 0.873 | 0.545 | 0.543 |
| DADSGNN | 0.862 | 0.522 | 0.523 |

Every baseline: real ≈ 0.86–0.89, both shuffle arms ≈ 0.515–0.545, all below the
0.55 leak line and within ±0.6 pp of each other (strict vs topo) → no structural
leak; topology reachability does not change the shuffle floor for diffuse readout.

Right panel source: the same `cycle_reachability_grid.jsonl`, `topo` and `full`
rows — cycle method real 0.822/0.877, shuffle 0.482/0.703.

## Data files used (with row counts)

| File | Rows | Distinct cells used |
|---|---|---|
| `hymeko_neuro/experiments/results/cycle_reachability_grid.jsonl` | 60 | 60 (clean, 1/cell) |
| `hymeko_neuro/experiments/results/no_leak_baselines.jsonl` | 350 | 350 (clean, 1/cell) |
| `hymeko_neuro/experiments/results/no_leak_baselines_topo.jsonl` | 420 | 350 after dedup (70 seed-0 re-run rows folded last-wins) |

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_neuro/paperkit/build_leakage_source_figure.py` | new module (2 builders + loaders + self-check) | +320 |
| `hymeko_neuro/tests/test_build_leakage_source_figure.py` | new test suite | +154 |
| `hymeko_neuro/assets/paper/figures/leakage_lattice.{pdf,png,eps}` | new figure | — |
| `hymeko_neuro/assets/paper/figures/leakage_source_contrast.{pdf,png,eps}` | new figure | — |
| `docs/plans/2026-06-20-nature-hsikan-leakage-source-figures/plan.{tex,pdf,tikz,mmd}` | plan | — |
| `reports/2026-06-20-nature-hsikan-leakage-source-figures.md` | this report | — |

## CORE.YAML items touched

**None.** `hymeko_neuro/`, `paper/figures/`, `reports/`, `docs/plans/` are not in
`CORE.YAML`. Purely additive; no existing file modified.

## Test results

Runner: `uv run python -m pytest -p no:randomly` (the `-m pytest` form is required
so the repo root is on `sys.path`; `hymeko_neuro` resolves as a namespace package).

- **10 passed in ~9 s.** Layers:
  - unit: `Arm` mean/std/count, NaN+None filtering, all-non-finite → None;
    lattice loader pooling + ordering invariant (strict ≤ topo < full shuffle);
    baseline loader dedup last-wins (proves seed-0 re-run not double-counted);
    partial-data tolerance (model missing an arm is dropped).
  - integration: both builders emit pdf+png+eps to a tmp dir on synthetic data;
    lattice tolerates a missing rule (draws 2); `main --self-check` end-to-end.
  - real-data contract: `cycle_reachability_grid.jsonl` shape — strict ≈ 0.5,
    topo shuffle < 0.55, full shuffle > 0.55 (skipped if file absent).

Lint: `ruff check` on both new files → **All checks passed.**

## Performance results

Trivial: each figure builds in < 1 s wall, well under 200 MB RSS (matplotlib over
< 1k JSONL rows), far under the 16 GB cap. No GPU. No benchmark assertion needed —
this is a one-shot plotting script, not a hot path; per CLAUDE.md §3 the perf-test
requirement applies to runtime code paths, and the relevant bound (sub-second,
sub-200-MB) is documented in the plan.

## §6.5 anti-patterns

None introduced. Single module, no Cartesian function family (figure variants are
two named builders sharing a `_grouped_bars`/`_save` helper + an `Arm` dataclass,
not copy-paste). Not a duplicate of `build_leakage_audit_figure.py` — distinct
figures (lattice + source-contrast); discovery pass confirmed no prior
`leakage_source`/`lattice` builder existed. No globals, no `_v2` file, no
string-typed config crossing into logic (rules are a fixed ordered list).

## Open issues / follow-ups

- The lattice currently pools bitcoin_alpha + bitcoin_otc (the only two datasets in
  `cycle_reachability_grid.jsonl`). If the cycle-method R_topo sweep is extended to
  epinions/slashdot/reddit_body, the figure picks them up automatically (loader
  pools all datasets present).
- Wiring the two figures into the paper TeX
  (`01_analysis/articles/superweapon/nature_hsikan/`) is left to the author; the
  EPS files are Springer-ready.

## Provenance

- Git SHA: `b5a1f64` (working tree dirty — pre-existing; this change adds the files
  listed above and does not modify tracked sources).
- Host: Windows 11, Python 3.12 (uv-managed venv), matplotlib Agg backend.
- Data: measured JSONL under `hymeko_neuro/experiments/results/` (see counts
  above); no seeds drawn at plot time (figures are deterministic functions of the
  committed data). Self-check uses fixed synthetic constants.
