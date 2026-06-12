# Report — No-leakage E1: narrow-deep depth/width knob wired into the driver

**Date:** 2026-06-11
**Branch:** `feature/ac-hsikan`
**Plan:** `docs/plans/2026-06-11-no-leakage-structural-benchmark/`
**Predecessor:** `reports/2026-06-11-no-leak-harness-and-leak-caught.md` (harness built + leak caught)
**CORE.YAML items touched:** none.

## Summary

The E1 driver (`run_no_leak_benchmark.py`) ran every Gömb cell through the
runner's *shallow* default middle (`middle_n_layers=1`, `d_middle=32`). The plan
pins the structural prior to the determinability-per-parameter Pareto point
**h=16, L=8** (`reports/2026-05-30-hsikan-depth-narrow-pareto.md`): narrow-deep
beats wide-shallow at ~13 % of the params with ~5× lower seed variance, which is
exactly the strict axis E1 measures. Running the shallow generic would bias H1
against the prior.

This change pins per-model architecture as `(width, depth)` on `Cell` and
threads it through both dispatch paths:

- **Gömb (subprocess):** `width → --d-middle`, `depth → --middle-n-layers`
  (≥2 dispatches `StackedMiddleHSiKAN`). Pinned `(16, 8)`.
- **SGCN (in-process):** `width → hidden=` (phase8 panel `32`); depth is inert
  (SGCN's depth is fixed inside `cell_signed_graph`). The old module-level
  `SGCN_HIDDEN` constant is removed — width now lives on the cell.

`Cell.make(model, dataset, n_epochs)` is the single place the narrow-deep /
panel pin is applied, so smoke and full grids share it (no §6.5 #1 Cartesian
duplication, no magic constant).

A second change makes the grid **resumable and multi-seed** (the plan requires
"checkpoint every cell to JSONL, resumable"). The result row now carries
`seed`/`width`/`depth`; `main` takes `seeds` (default `(0,)` smoke, `(0..4)`
full) and runs each `(cell, seed)` through `_run_cell_seed`, which checkpoints
each `(real|shuffle)` arm by appending to the JSONL the moment it finishes and
skips arms already present on disk. An interrupted grid resumes where it
stopped; a crash loses at most the in-flight arm. The old "write the whole
JSONL once at the end" path is gone.

## Files touched

| File | +/− | Note |
|:--|:--|:--|
| `signedkan_wip/experiments/runs/run_no_leak_benchmark.py` | +95 / −28 | `Cell` gains `width`/`depth` + `make`; both runners thread the arch; `SGCN_HIDDEN` removed; resumable per-arm checkpointing + seed loop; `--seed`→`--seeds` |
| `signedkan_wip/tests/test_no_leak_benchmark.py` | +150 / 0 | new; regression + unit + resume tests |

## Test results

`pytest -p no:randomly signedkan_wip/tests/test_no_leak_benchmark.py` — **12 passed, 9.51 s**.

- **Regression** (`test_subprocess_command_carries_narrow_deep_flags`): the Gömb
  command must carry `--d-middle 16 --middle-n-layers 8`. Fails against the prior
  driver, which emitted neither flag (silently shallow).
- **Unit:** `Cell.make` arch mapping (Gömb narrow-deep, SGCN panel); shuffle-flag
  append; SGCN `hidden=` pass-through; `audit_gate` boundaries (clean/leak/
  no-signal/runner-failure).
- **Resume:** `_result_key` legacy-row tolerance; `_load_done`/`_append_row`
  roundtrip; `_run_cell_seed` skips a completed arm and checkpoints the fresh one.

## Performance / smoke (production scale, §3 contract)

`run_no_leak_benchmark --smoke` on bitcoin_alpha (CUDA, RTX 2070 SUPER):

| Cell | real AUC | shuffled AUC | verdict | n_params |
|:--|--:|--:|:--|--:|
| Gömb narrow-deep | **0.8833** | **0.4888** | CLEAN+SIGNAL | 251 234 |
| SGCN panel | 0.8756 | 0.5266 | CLEAN+SIGNAL | 135 585 |

**Peak RSS 1.41 GB** (cap 16 GB), **wall 89.9 s** (new checkpointing `main`).
The shuffle gate holds on both cells (≤0.55), so the depth change did not
re-introduce leakage. Gömb params moved 193 041 → 251 234: the middle is now an
8-layer stack at width 16; the absolute count is higher than the vision report's
14.5k because this middle is one shell of the full Gömb cascade (outer
Clifford-FIR + core + embeddings dominate), not a standalone HSiKAN. A second
`--smoke` invocation reported `[resume: 4 arms done]` and finished in 0.0 s
(523 MB) — the resume path skips completed arms without retraining.

## Full E1 grid — results

`run_no_leak_benchmark --full`: both Bitcoin graphs × {gomb, SGCN} × seeds 0–4 ×
{real, shuffle} = **40 arms**, `n_epochs=200`, narrow-deep Gömb. **Wall 1677 s
(28 min), peak RSS 1.41 GB.** All 40 arms **CLEAN+SIGNAL** — every shuffled
control ≤ 0.5345, under the 0.55 gate; no cell leaks.

| dataset | model | real (mean±pstdev) | shuffled (mean±pstdev) | paired Δ | max shuf | n_params |
|:--|:--|--:|--:|--:|--:|--:|
| bitcoin_alpha | **Gömb** | **0.8900 ± 0.0044** | 0.4954 ± 0.0086 | **+0.3946** | 0.5038 | 251 234 |
| bitcoin_alpha | SGCN | 0.8528 ± 0.0142 | 0.5231 ± 0.0043 | +0.3297 | 0.5297 | 135 585 |
| bitcoin_otc | **Gömb** | **0.9139 ± 0.0068** | 0.4939 ± 0.0138 | **+0.4199** | 0.5111 | 351 938 |
| bitcoin_otc | SGCN | 0.8790 ± 0.0064 | 0.5129 ± 0.0163 | +0.3661 | 0.5345 | 202 721 |

**H1 holds, and stronger than the plan predicted on Bitcoin.** Under the
*same* strict, shuffle-audited protocol the narrow-deep structural prior
**beats** SGCN on both graphs — alpha **+3.72 pp** (0.8900 vs 0.8528), otc
**+3.49 pp** (0.9139 vs 0.8790) — not merely "competitive." The depth
prediction also reproduces: Gömb's cross-seed pstdev is **3.2× tighter** than
SGCN's on alpha (0.0044 vs 0.0142), consistent with
`2026-05-30-hsikan-depth-narrow-pareto.md`'s "variance shrinks with depth."
Significance/win-rate and the larger graphs (Epinions/Slashdot) remain for the
next step before any headline claim — Bitcoin is near-ceiling and small.

- **Log (on-disk anchor, §3):** `signedkan_wip/experiments/results/no_leak_e1.log`
- **Output JSONL:** `signedkan_wip/experiments/results/no_leak_e1.jsonl` (40 rows)
- **Background task id:** `b13pqpgls` (exit 0)

## Static analysis

- `ruff check` — clean on both files.
- `mypy --strict` (isolated, `run_no_leak_benchmark`) — **Success, no issues**.
  The 5 errors in a whole-tree run are pre-existing (missing sklearn/scipy/triton
  stubs in `run_final_cell.py` / `signedkan.py`, plus the `experiments` dual
  module-path) and untouched by this change.

## §6.5 anti-patterns

None introduced. The arch is parametric (config, not class-per-variant, #8); the
single `Cell.make` avoids per-cell duplication (#1, #3); no new file/v-suffix
(#13); the existing plan covers this step — no new plan dir (#12).

## Open issues / follow-up

1. **Capacity-scaling knobs not yet wired.** The plan also pins `lr=3e-3/√2 at
   d=32`, `sign_head_hidden` scaling, mixed `K_static=6, K_dyn=2`
   (`reports/2026-06-06-ac-hsikan-capacity-scaling`). E1 here uses the runner's
   default `lr=3e-3` at `d_middle=16`. Decide whether those belong on `Cell` as a
   second pinned axis before the full grid, or are out of scope for the Gömb
   middle stack (they target the AC-HSiKAN candidate selection, a different
   shell).
2. **Scale to full E1.** Both Bitcoin graphs × {gomb, SGCN} × {real, shuffle},
   ≥5 seeds, checkpointed (`FULL_CELLS` already narrow-deep). Then
   Epinions/Slashdot (the heaviest cell — size wall against epinions per plan
   §risk).

## Provenance

- **Git SHA:** `af803ee` (dirty — tree carries the seminar/demo WIP listed in
  `git status`; the only files this task touched are the two above).
- **Interpreter:** miniconda3 / torch 2.11.0+cu130. **GPU:** RTX 2070 SUPER 8 GiB.
- **Seed:** 0. **Dataset:** bitcoin_alpha (19 348 train edges after split).
- **Artifact:** `signedkan_wip/experiments/results/no_leak_smoke.jsonl` (4 rows).
