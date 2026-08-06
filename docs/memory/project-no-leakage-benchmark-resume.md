---
name: project-no-leakage-benchmark-resume
description: "No-leakage structural benchmark (Gömb-HSiKAN) — harness built + smoke-clean 2026-06-11; resume at narrow-deep n_layers knob, then full E1"
metadata: 
  node_type: memory
  type: project
  originSessionId: 46abbe99-51fa-4910-8e5d-71bc18b31911
---

The no-leakage structural benchmark (E1) harness is **built and smoke-validated**
as of 2026-06-11. Driver: `signedkan_wip/experiments/runs/run_no_leak_benchmark.py`
(`--smoke`/`--full`, shuffle gate, JSONL). Smoke is CLEAN+SIGNAL on bitcoin_alpha,
seed 0: **Gömb-strict real 0.892 / shuffled 0.503**, **SGCN-strict real 0.876 /
shuffled 0.527** — the apples-to-apples strict comparison already favours the
structural prior, both shuffle-clean.

Dispatch: `gomb` → subprocess `run_gomb_smoke` (strict train-only cycle pool);
`SGCN` → in-process `cell_signed_graph` (strict-by-construction adjacency). The
smoke **caught** that `cell_signed_graph`'s HSiKAN path enumerates over the full
graph (transductive) and leaks (shuffled 0.73) — do NOT route the structural
prior through it. See [[project-sisy2026-control-paper]] for the related papers.

**Done 2026-06-11:** narrow-deep depth/width knob wired + full Bitcoin E1 grid
clean. `Cell` carries `(width, depth)` via `Cell.make`; Gömb runs h=16/L=8
(`width→--d-middle`, `depth→--middle-n-layers`), SGCN panel `hidden=32`. Driver
is resumable: per-`(cell,seed,shuffle)` arm checkpointed by JSONL append, `main`
takes `seeds` (`--seeds`), completed arms skipped on restart. 12 tests pass.
**Full E1 (40 arms, 5 seeds, 28 min, RSS 1.41 GB) — ALL CLEAN+SIGNAL**, every
shuffle ≤ 0.534. Gömb BEATS SGCN strict on both graphs: alpha 0.8900±0.0044 vs
0.8528±0.0142 (+3.72pp, 3.2× tighter variance), otc 0.9139±0.0068 vs
0.8790±0.0064 (+3.49pp). H1 holds, stronger than "competitive." Results:
`signedkan_wip/experiments/results/no_leak_e1.jsonl`. Report:
`reports/2026-06-11-no-leak-narrow-deep-depth-knob.md`.

**Next step (resume here):** (1) Epinions/Slashdot cells (heaviest — size wall
against epinions 131k V/841k E first; joint-slot caps / grad-checkpoint may be
needed) + paired-z significance & win-rate before any headline. Capacity-scaling
knobs (lr=3e-3/√2 at d=32, sign_head_hidden, K_static=6/K_dyn=2 — see
`reports/2026-06-06-ac-hsikan-capacity-scaling`) were left OUT (they target
AC-HSiKAN candidate selection, not the Gömb middle); revisit only if a cell
underperforms. After E1: E2 (structural determinability) and E3 (masked-edge
SSL). Bitcoin is near-ceiling/small — lead conclusions on the big graphs + the
label-efficiency curve, not Bitcoin point estimates.

**Why:** the shuffle gate is the linchpin — every reported cell must show
shuffled ≤ 0.55 or it is leaking, not learning. Narrow-deep + the enumeration
triad (ABB/Friedler/cycle-basis, Lever 1, 2/3 shipped) are pinned to raise
determinability-per-parameter, which is the strict axis that actually matters.

**How to apply:** run `python -m signedkan_wip.experiments.runs.run_no_leak_benchmark
--smoke` to re-confirm the path before scaling. Plan:
`docs/plans/2026-06-11-no-leakage-structural-benchmark/`. Report:
`reports/2026-06-11-no-leak-harness-and-leak-caught.md`. Two sibling plans on
disk: `docs/plans/2026-06-11-conv-as-hypergraph-hymeyolo/` and the levers
(informal).
