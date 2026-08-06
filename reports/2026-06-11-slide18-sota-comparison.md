# Slide 18 — SOTA accuracy-vs-cost comparison (latency bars + Pareto + table)

**Date:** 2026-06-11 · **Plan:** `docs/plans/2026-06-11-slide18-sota-comparison/`
(plan.tex/pdf/tikz/mmd) · **Builds on:** `reports/2026-06-11-latency-bench-extension.md`

## Summary
Extended the forward-latency bench so slide 18 ("SOTA range — at a fraction of
the cost") can show HSiKAN against the external baselines honestly, on **both**
cost axes:
- **Latency bars** (here-measured forward): added **SiGAT, SGT, MLP-blind** next
  to SGCN + the HSiKAN-lean/joint width variants, via a **baseline registry**
  (Strategy) that replaces the per-model near-copies (§6.5 #1/#9).
- **Accuracy-vs-params Pareto** (committed 5-seed AUC vs parameter count): every
  method in the apples-to-apples `phase8_bitcoin_5seed.json` panel **plus** the
  optuna-tuned HSiKAN point, with HSiKAN shown as **two points** (leanest +
  optuna) so the tuning range is visible.
- **Merged numbers table** (`inference_bench_table_<dataset>.md`): every method's
  AUC (5-seed mean ± pstdev), params, here-measured CPU/CUDA forward, and a
  provenance column; a missing axis is `—`, never faked.

**Honesty design (decided with the user):** the Pareto cost axis is **params**
(config-intrinsic, single provenance from the committed jsons) so the tuned
optuna point and the untuned baselines never share a here-measured latency axis.
Tuning asymmetry (optuna-HSiKAN tuned, baselines untuned) is shown via a star
marker + an explicit caption, per `RESULTS_DISCIPLINE.md`. No literature numbers
on the measured plot.

## Files touched
**New:**
| LOC | File |
|---:|---|
| 142 | `hymeko_neuro/experiments/runs/sota_compare.py` — committed-accuracy loaders (pure) |
| 130 | `hymeko_neuro/tests/test_sota_compare.py` — 13 tests (loaders + Pareto + table) |

**Modified:**
- `hymeko_neuro/experiments/runs/run_inference_bench.py` — baseline registry
  (`BenchUnit`, `build_{sgcn,sigat,sgt,mlp}_unit`, `BASELINES`,
  `_bench_baselines`) replacing the SGCN-specific path; `h_baseline=32` in
  CONFIGS to config-match phase8; summary + honest note updated.
- `hymeko_neuro/experiments/runs/bench_to_png.py` — extended bar palette;
  `pareto_points` / `render_pareto`; `table_rows` / `write_table`; `main_all`
  orchestrator + `--all` CLI.
- `hymeko_neuro/tests/test_inference_bench.py` — baseline-registry build/run
  test; bar-order test relaxed to relative order.

**Regenerated artifacts** (`hymeko_neuro/experiments/results/`): `inference_bench.json`
(24 rows: 2 graphs × 2 devices × {SGCN, SiGAT, SGT, MLP-blind, HSiKAN-lean,
HSiKAN-joint}); `inference_bench_{cpu,cuda}.png`;
`inference_bench_pareto_bitcoin_{alpha,otc}.png`;
`inference_bench_table_bitcoin_{alpha,otc}.md`.

## CORE.YAML items touched
**None.** Changes confined to `experiments/` and `tests/`; the baseline model
classes in `src/baselines/` are constructed + timed, not modified.

## Measured numbers (seed 0; this host, CUDA 13.2)
**Forward latency, median ms (here-measured):**

| dataset | device | SGCN | SiGAT | SGT | MLP-blind | HSiKAN-lean | HSiKAN-joint |
|---|---|---:|---:|---:|---:|---:|---:|
| bitcoin_otc | cpu | 20.4 | 91.8 | 212.4 | 0.6 | 95.5 | 309.7 |
| bitcoin_otc | cuda | 5.1 | 203.0 | 469.2 | 0.9 | 24.6 | 31.8 |

**Accuracy vs cost (committed 5-seed, bitcoin_otc):** HSiKAN-optuna **0.9933**
@ 23.8k params (tuned); SGCN 0.942 @ 203k; SiGAT 0.932 @ 202k; MLP-blind 0.908 @
190k; GCN-blind 0.906 @ 192k; HSiKAN-leanest 0.851 @ 95k (untuned);
SignedKAN-L1 0.802 @ 189k. Full table in `inference_bench_table_bitcoin_otc.md`.

**Honest read for the talk:** the optuna-tuned HSiKAN dominates accuracy-per-
parameter (top-left of the Pareto, ~8.5× fewer params than SGCN at higher AUC),
**but it is tuned and the baselines are not**. The untuned HSiKAN in the same
phase8 panel (leanest, 0.851) is *below* the baselines — shown as the second
HSiKAN point so the claim is not oversold. On absolute forward latency, SGCN and
MLP-blind are lightest; SiGAT and SGT are heavier than HSiKAN-lean.

## Test results
- `pytest -p no:randomly` on the two files: **22 passed**, 17.4 s.
  - `test_sota_compare.py` (13, pure): phase8 mean/std/seed-count, unmapped +
    null-AUC drop, optuna `auc` field + `tuned`, merge precedence;
    `pareto_points` filter/sort, `render_pareto` PNG smoke + empty guard;
    `table_rows` union + missing-axis `None` + params sort; `write_table` smoke.
  - `test_inference_bench.py` (9): baseline registry builds+runs every model;
    existing `_summarize`/bars/HSiKAN tests.
- Production-scale smoke (deliverable run): full extended bench + `--all` render,
  all six artifacts written.

## Performance results (vs plan budget)
- **Peak RSS 1.52 GB** — budget <3.0 GB ✓ (cap 16 GB ✓).
- **Wall 160.5 s** — budget <180 s ✓ (≥5 repeats: 20 timed + 5 warmup/cell).
- Render (`--all`): <10 s, <0.5 GB.

## Static analysis
- `ruff check` (all 5 files): **All checks passed**.
- `mypy --strict` `sota_compare.py` + `bench_to_png.py`: **Success** (the latter
  needs `--explicit-package-bases --namespace-packages` to avoid a dual module
  path; no type errors). `run_inference_bench.py` keeps the existing untyped
  `experiments/runs/` convention (torch/numpy-heavy).

## No §6.5 anti-patterns introduced
- Baseline registry (Strategy) removes the per-model duplication (#1/#9); each
  builder is a `BenchUnit` factory, no Cartesian wrappers. Accuracy/cost loaders
  are pure and reused across Pareto + table (no copy). Provenance is explicit; no
  silent caps (missing axis logged as `—`, SGT's absent accuracy is visible).
  GCN-blind has no on-disk adjacency builder → Pareto/table-only, flagged, not
  faked. No globals (#11).

## Provenance
- Git SHA `af803ee` (dirty — this change + prior seminar items).
- Python 3.12; torch 2.12.0+cu132; matplotlib 3.10.9; CUDA available.
- Deterministic: seed 0; fixed caps; no system entropy.
- Accuracy sources: `phase8_bitcoin_5seed.json` (5-seed, untuned, both graphs);
  `bitcoin_optuna_best_5seed_2026_05_13.jsonl` (5-seed, tuned). Latency measured
  this host.

## Open issues / follow-ups
- Slide 18 should embed `inference_bench_cpu.png` (bars) + the per-dataset
  Pareto PNG; the `.pptx` binary is not in this change.
- Remaining seminar build items: Demo 1 balance, Demo 4 mesh + Sinkhorn,
  Demo 5 bridge.
