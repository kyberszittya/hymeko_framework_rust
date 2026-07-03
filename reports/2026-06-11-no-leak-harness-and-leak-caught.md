# No-leakage benchmark (E1) — harness built, transductive leak caught by the smoke

**Date:** 2026-06-11 · **Plan:** `docs/plans/2026-06-11-no-leakage-structural-benchmark/`

## Summary
Built the E1 driver for the no-leakage structural benchmark and validated the
strict + label-shuffle path end-to-end. The production-scale smoke did its job:
it **caught a transductive leak in 25 s** before any overnight run, forcing a
re-point onto the audit-clean strict harness.

## What the smoke caught (the §11 halt)
First attempt routed E1 through `cell_signed_graph` (`run_final_cell.py`).
Smoke result on bitcoin_alpha, c2-ablated HSiKAN:

| path | real AUC | shuffled AUC | verdict |
|---|---:|---:|---|
| `cell_signed_graph` (HSiKAN) | 0.8900 | **0.7336** | **LEAKS** (gate 0.55) |

Diagnosis (confirmed): `run_final_cell.py:469–471` enumerates the cycle pool
over **`g` (the full graph)** via `cached_construct_k(g, ...)`, so test edges
sit inside the cycles and their σ-products leak the test sign — shuffled labels
still score 0.73, not chance. This matches the audit's *joint_mix* row (0.8902,
"moderate σ-leakage"), **not** the strict Gömb row (0.5402). It is the
transductive path, not the strict one. Halted; did not scale.

## The fix — reuse the existing strict harness
`run_gomb_smoke.py` already implements the strict protocol (line 445: "the
strict protocol (train-only cycle pool)"; default, `--unrestricted-cycles` is
the transductive opt-out) with `--shuffle-train-signs`. Standalone confirmation,
bitcoin_alpha, seed 0, 60 epochs, default Gömb (M_outer=8, d=16/32, n_tiers=3,
193k params):

| path | real AUC | shuffled AUC | verdict |
|---|---:|---:|---|
| `run_gomb_smoke` (Gömb-strict) | **0.8923** | **0.5027** | **CLEAN+SIGNAL** |

Shuffled at chance (Δ = 0.389 learned signal, no leakage), matching the
2026-05-14 audit's 0.54. Real 0.892 already ≈ the tuned audit number (0.897).

The E1 driver was re-pointed to dispatch each model through its **strict**
runner as a subprocess (fresh process per cell for RSS isolation; no training
logic reimplemented). Driver smoke reproduced **CLEAN+SIGNAL** (real 0.8923 /
shuffled 0.5027), peak RSS 524 MB, wall 42 s.

## Files touched
**New:**
| LOC | File |
|---:|---|
| 145 | `hymeko_neuro/experiments/runs/run_no_leak_benchmark.py` — E1 driver: `STRICT_RUNNERS` dispatch, `--smoke`/`--full`, shuffle gate, JSONL |

**Plan updated:** `docs/plans/2026-06-11-no-leakage-structural-benchmark/plan.tex`
— added the "Incorporated findings & pinned components" section (narrow-deep
Pareto, CPML, Clifford-FIR kernel-ON, the ABB/Friedler/cycle-basis enumeration
triad = Lever 1, 2/3 shipped); PDF recompiled.

**Artifacts:** `hymeko_neuro/experiments/results/no_leak_smoke.jsonl`.

## CORE.YAML items touched
**None.** Driver is new under `experiments/runs/`; strict runners reused unmodified.

## Test / gate results
- Driver smoke: **CLEAN+SIGNAL** (gomb bitcoin_alpha, real 0.8923 / shuffled
  0.5027). ruff clean; `audit_gate` unit-checked ((0.91,0.52)→clean+signal;
  (0.99,0.99)→leaks).
- The smoke is the §3 production-scale gate (real dataset, real cap k4=20k, real
  wall) before any multi-seed/overnight scale-up.

## Performance
- Smoke wall 42 s (2 subprocess Gömb runs), peak RSS 524 MB (well under 16 GB).

## Open issues / follow-ups
1. **SGCN-strict baseline cell** — verify `run_sgcn_baseline`'s JSON AUC key and
   `--shuffle-train-signs`/`--n-epochs` args, then add to the smoke.
2. **Narrow-deep config** — `cell`/`run_gomb_smoke` need an `n_layers` (depth)
   knob to run the pinned $h{=}16,L{=}8$ Pareto config; the smoke used default depth.
3. **Scale to full E1** — both Bitcoin graphs × {gomb, SGCN} × {real, shuffle},
   ≥5 seeds, checkpointed; then Epinions/Slashdot.
4. E2 (determinability) and E3 (masked-edge SSL) per the plan.

## Provenance
- Git SHA `af803ee` (dirty). Python 3.12; torch 2.12.0+cu132; CUDA available.
- Deterministic: seed 0; train-only cycle pool; shuffle stream seed+100003.
- Dataset: bitcoin_alpha (|V|=5881-class Bitcoin OTC sibling; alpha |E| split
  19349 train / 4837 val).
