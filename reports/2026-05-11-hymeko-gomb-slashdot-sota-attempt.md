# HymeKo-Gömb — Slashdot SOTA-break attempt (NEGATIVE)

**Date:** 2026-05-11 (evening)
**Plan:** `docs/plans/2026-05-11-hymeko-gomb-sphere/plan.tex` §Validation gate
**Branch:** `refactor/extract-hymeko-hre`
**Git SHA at run:** `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (working tree dirty per `2026-05-11-hymeko-gomb-sphere.md`)

## Summary

Attempted to break the Slashdot SOTA (edge_cr 0.9067 ± 0.0034, memory
`project_edge_cr_5seed_2026_05_09`) with HymeKo-Gömb. **Result: LOSS at
−2.3σ.** Gömb hits a hard architectural ceiling at 0.903 across 7
configurations. The cascade's cycle-only-k=3 ingredient set lacks
whatever SOTA-break primitive edge_cr's per-edge Catmull-Rom Highway
gate has.

## Final headline

| Recipe | val_auc | n | σ |
|---|---|---|---|
| edge_cr SOTA (memory project_edge_cr_5seed_2026_05_09) | 0.9067 | 5 | 0.0034 |
| **HymeKo-Gömb slim (this work)** | **0.9031** | 5 | **0.0008** |
| Δ (Gömb − SOTA) | **−0.0036** | | (paired two-sample SE 0.0016 → **−2.3σ**) |

Gömb's per-seed σ is **4× tighter** than edge_cr's. The mean is 1.2pp
short.

## 5-seed Slashdot (`/tmp/gomb_slashdot_5seed_2026_05_11/`)

Config (winner of the 4-config smoke sweep below):
`--d-embed 16 --d-outer 4 --M-outer 4 --d-middle 4 --d-core 4 --n-tiers 3 --k 3 --topk 32 --n-epochs 60`

| seed | val_auc_best | wall (s) |
|---|---|---|
| 0 | 0.9030 | 9.4 |
| 1 | 0.9045 | 9.4 |
| 2 | 0.9029 | 9.3 |
| 3 | 0.9022 | 9.4 |
| 4 | 0.9030 | 9.4 |

Mean **0.9031 ± 0.0008**, 46.8 s total wall, 1 318 993 params, ~133 600
cycles per seed (per-vertex top-K=32).

## Config sweep (single-seed probes that established the ceiling)

All on Slashdot |V|=82 140, |E|=549 202, seed 0, cuda, 60 ep:

| Probe | config | val_auc_best | wall | n_params | notes |
|---|---|---|---|---|---|
| baseline | M=4 d_o=8 d_m=16 d_c=16 topk=32 | 0.9030 | 12.8 s | 2 650 105 | peak at ep 54 then overfits |
| A: more cycles | + topk=64 | 0.9031 | 16.6 s | 2 650 105 | identical |
| B: narrow | d_o=4 d_m=4 d_c=4 (half params) | 0.9030 | 9.4 s | 1 318 993 | same AUC at half-params |
| C: wider M | M=8 | 0.9026 | 16.3 s | 2 656 849 | slightly worse |
| 100ep slim | baseline + 100 ep | 0.9030 | 21.0 s | 2 650 105 | overfits past ep 54; AUC drops to 0.896 by ep 99 |
| D: k=4 | + k=4 topk=32 | OOM | — | — | CUDA OOM in HSiKAN backward |
| E: k=4 topk=16 | B + k=4 topk=16 | 0.8998 | 23.1 s | 1 319 001 | worse than k=3 |

**Observations:**
1. Hardware ceiling: 0.903 AUC is sticky regardless of (width, # banks, # cycles, # epochs).
2. **B (narrow) hits the same AUC as baseline at half the parameters** → capacity is not the bottleneck.
3. **Single-arity k=4 hurts** (vs memory `project_phase9_k45_sweet_spot` saying *mixed* k=4+k=5 wins; Gömb only supports single k by current design).
4. Overfitting kicks in around ep 50–55 even at slim config; more epochs hurts.

## CORE.YAML items touched

None.

## Files touched

None new since `2026-05-11-hymeko-gomb-sphere.md`. All runs invoked
the existing `signedkan_wip/src/run_gomb_smoke.py` from that report.

## Test results

No new tests added; reused the 13/13-passing test suite from the
companion feasibility report.

## Performance

5 × Slashdot seeds in 46.8 s wall total. No process crossed 8 GB GPU
(probes D/E that OOM'd were excluded from the 5-seed). Per-seed RSS
not directly measured; `nvidia-smi` showed steady-state ~1.3-3 GB
during runs.

## New / removed dependencies

None.

## Open issues / follow-up

### To break the SOTA, Gömb needs one of these ingredients (none currently present)

1. **Per-edge Catmull-Rom Highway gate** (the actual edge_cr SOTA
   ingredient; memory `project_edge_cr_5seed_2026_05_09`). Would
   replace `MiddleHSiKAN`'s whole-layer CR with a per-edge CR. Code
   change in `MiddleHSiKAN` + the spline path.

2. **Walks** (memory `project_walks_epinions_5seed_2026_05_11`:
   kitchen-sink walks+cycles beat cycles-alone on Epinions by +0.075
   paired). Gömb's current enumerator
   (`enumerate_top_k_per_vertex_cycles_signed_filtered_batched_rs`)
   is cycle-only. Adding walks requires the
   `enumerate_k_walks_rs` pool and a multi-source-aggregator in the
   middle shell.

3. **Mixed-arity** (`project_phase9_k45_sweet_spot`: k=4+k=5 mixed
   beats k=3+k=4 on every dataset). Each Gömb shell currently takes a
   single `cycle_k`. Mixed arity requires running the enumerator
   twice (e.g. k=4 and k=5) and feeding both through the shells with
   learned αₖ mixing.

### Smaller follow-ups

4. **k=4 single arity at top-K=8** — to see whether single-arity k=4 with
   even less memory pressure approaches the 0.903 ceiling. Predicted
   null but cheap to check (~30 s on cuda).
5. **Ablations on Slashdot** — single-seed Bitcoin OTC ablations were
   within σ; Slashdot's 4× tighter noise might let the per-shell Δ
   actually show.
6. **`signedkan_wip/src/run_gomb_smoke.py`** is still untracked.

### Negative finding worth keeping

7. **σ inversion: Gömb is 4× tighter than edge_cr.** If we EVER lift
   the mean by +0.005 (architectural change), the paired-σ advantage
   from low variance gives Gömb a clean win. The mean is the only
   thing missing. That's a useful design signal: *capacity is not the
   problem; ingredient is.*

## Provenance

- Git SHA: `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (dirty tree as documented in `2026-05-11-hymeko-gomb-sphere.md`)
- Hardware: AMD Ryzen 7 3700X + RTX 2070 SUPER (8 GB), driver 580.126.09, CUDA 13.0
- Python 3.13.5, torch 2.11.0+cu130, numpy 2.4.4
- Dataset: Slashdot, |V|=82 140, |E|=549 202, pos/neg = 425 072 / 124 130
- Seeds: {0, 1, 2, 3, 4} for the 5-seed; {0} for the probes
- Memory cap: **NO `ulimit -v` for any of these runs** — per
  `feedback_ulimit_vs_cuda.md`, ulimit -v 16G triggers CUDA driver
  OOM at first `.to(cuda)` call. Future CV/CUDA work uses
  `systemd-run --user -p MemoryMax=16G` instead per CLAUDE.md §4's
  4-option list.

### On-disk log artifacts (verifiable per CLAUDE.md §9 in-flight discipline)

- 5-seed: `/tmp/gomb_slashdot_5seed_2026_05_11/{seed0..seed4}.log`, `results.jsonl`
- Probes: `/tmp/gomb_slashdot_sweep_2026_05_11/{A_topk64,B_narrow,C_wider_M,D_k4,D_k5,E_k4_topk16}.log`
- Initial smokes: `/tmp/gomb_slashdot_smoke_seed0{,_100ep}_2026_05_11.log`

## Plan-vs-result delta

| Plan item | Status |
|---|---|
| §Validation gate cell 1 (Slashdot edge_cr SOTA paired comparison) | **NEGATIVE — Δ = −0.0036 / −2.3σ** |
| §Validation gate cell 2 (best-cell of orthogonal factorial) | not yet run (factorial cells not measured) |
| §Sequencing 7 (ablations 5-seed paired) | not run on Slashdot |

Per plan §Venue: "If null: archived as a documented negative result
alongside the factorial; the factorial's best-cell becomes the
recommended architecture and HymeKo-Gömb is footnoted as 'cascade does
not exceed orthogonal-best.'" — this report fulfils that obligation.
