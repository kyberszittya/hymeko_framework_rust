# Report: Per-Vertex Top-$m$ ABB --- Global-Min, Fullness Gating, Composite Scorer

**Date:** 2026-05-11
**Branch:** refactor/extract-hymeko-hre
**Plan:** [docs/plans/2026-05-11-abb-global-fullness/](../docs/plans/2026-05-11-abb-global-fullness/)
**Companion:** 2026-05-10 "v1" start-ABB --- [reports/2026-05-10-abb-global-topk.md](2026-05-10-abb-global-topk.md)

## Summary

Four orthogonal improvements to the 2026-05-10 per-vertex score upper-bound
branch-and-bound (ABB), in response to v1's AUC regressions (Bitcoin OTC
seed-0: $-3.00$pp; previously published $-7.6$pp on a different config):

1. **Global-min ABB** --- threshold is the minimum heap-min across all
   FULL vertex heaps, shared between rayon tasks via `AtomicU64` (`f64` bit-cast).
2. **Composite `WeightedSumScorer`** --- additive combination of two
   `BoundedScorer`s with admissible UB; building block for axiomatic
   multi-criteria ABB.
3. **(Approach 4)** Smaller-$m_v$ sweep (config-only); folded into result table.
4. **Adaptive fullness gating** --- ABB activates only once
   `gate * n_nodes` heaps are full. New env var
   `HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE`. Default `1.0` (strictly
   correct).

## Files touched

| File | Lines | Notes |
|---|---|---|
| `hymeko_graph/src/topk_cycles.rs` | +~370 | `dfs_per_vertex_bb_global`, `atomic_min_f64`, `enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched`, `WeightedSumScorer`, 5 new tests |
| `hymeko_graph/src/lib.rs` | +2 | exports |
| `hymeko_py/src/cycles.rs` | +~90 | `enumerate_top_k_per_vertex_cycles_signed_filtered_bb_global_batched_rs` PyO3 binding |
| `hymeko_py/src/lib.rs` | +1 | binding registration |
| `signedkan_wip/src/n_tuples.py` | +~25 | `HSIKAN_USE_PER_VERTEX_ABB_MODE` + `HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE` dispatch |
| `docs/plans/2026-05-11-abb-global-fullness/` | new | `plan.{tex,pdf,tikz,mmd}` + `plan_diagram.pdf` |

No `CORE.YAML` items touched.

## Test results

**Unit tests (`cargo test -p hymeko_graph --release`):** 73 + 9 + integration = all passing.

New tests added in this change:
- `per_vertex_bb_global_huge_caps_matches_non_bb` --- huge caps, ABB never fires → identical output count to non-ABB.
- `per_vertex_bb_global_subset_of_non_bb` --- tight caps, dense graph; output is a strict subset of non-ABB.
- `per_vertex_bb_global_with_full_gate_disables_abb` --- `gate=1.0` over an 8-node fixture: ABB never activates, output matches non-ABB.
- `weighted_sum_scorer_admissibility` --- `WeightedSumScorer::upper_bound ≥ score` on fully-known cycles across 4 sign patterns.
- `weighted_sum_scorer_partial_path_ub` --- partial-path UB dominates every completion at $k{=}4$, $n_{\text{neg}}{=}1$, $k_{\text{remaining}}{=}2$.

**Python smoke** (toy fixture, two disjoint triangles + bridge):
parity of $(\text{cycles}, \text{scores})$ between global-min, v1 start-ABB,
and non-ABB enumerators at huge caps and at gates `{0.0, 0.25, 1.0}`.

## AUC results (Bitcoin Alpha + OTC, seed=0, 80 epochs, HSiKAN-mixed h=16)

### Bitcoin Alpha

| Variant | AUC | $\Delta$ vs ABB OFF |
|---|---|---|
| ABB OFF (baseline) | 0.9619 | --- |
| **Global-min, gate=0.25** | **0.9617** | **$-0.0002$** (within noise) |
| v1 start-ABB | 0.9087 | $-5.32$ pp |

### Bitcoin OTC --- fullness-gate sweep

| Variant | AUC | $\Delta$ vs ABB OFF |
|---|---|---|
| ABB OFF (baseline) | 0.9610 | --- |
| **Global-min, gate=1.0** (strict) | **0.9615** | **$+0.0005$** (AUC-preserving) |
| Global-min, gate=0.5 | 0.9568 | $-0.42$ pp |
| Global-min, gate=0.25 | 0.9516 | $-0.94$ pp |
| Global-min, gate=0.1 | 0.9457 | $-1.53$ pp |
| v1 start-ABB | 0.9309 | $-3.00$ pp |

**Decision:** default `HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE = 1.0` (strictly correct).
v1 start-ABB stays exposed for cases where AUC loss is tolerated for raw throughput.

### Approach 4 --- $m_v$ sweep (Bitcoin Alpha + OTC, gate=1.0 global ABB, seed=0)

| $m_v$ | BA AUC | $\Delta$ vs $m{=}128$ | OTC AUC | $\Delta$ vs $m{=}128$ |
|---|---|---|---|---|
| 32  | 0.9438 | $-1.81$ pp | 0.9368 | $-2.47$ pp |
| 64  | 0.9549 | $-0.70$ pp | 0.9549 | $-0.66$ pp |
| **128** (default) | **0.9619** | --- | **0.9615** | --- |

Clean monotonic ladder: each doubling of $m_v$ gains $\sim 0.7$--$1.8$ pp.
**Verdict: $m{=}128$ is the right operating point on Bitcoin --- smaller $m$
is not a viable shortcut on small dense graphs.** Memory savings at $m{=}32$
($4\times$ smaller heaps) are not worth the AUC cost.

The previous expectation that "we may have reached the correctness ceiling
below $m{=}128$" is falsified for these datasets; the ceiling is at or
above $m{=}128$. Heap-fullness-gated global ABB at $m{=}128$ closes the
question --- larger $m$ remains untested but the trend suggests diminishing
returns.

Cross-check: $m{=}128$ BA AUC = 0.96185 with global ABB gate=1.0 vs
0.96185 ABB OFF baseline (Section 4.1) --- the strict-correctness gate
reproduces the baseline to $10^{-5}$, validating the correctness story.

## Single-seed caveat

Each cell above is one seed. Per memory `feedback_n_seed_before_paper_promotion.md`,
no number here is paper-headline material until a 5-seed paired test confirms it.
Suitable next step: 5-seed paired Bitcoin OTC gate=1.0 vs ABB OFF to confirm
the $+0.0005$ "AUC-preserving" verdict.

## Speedup characterization

`fwd_per_call_ms` numbers (downstream inference cost, not enumeration walltime):

| Variant | BA fwd ms | OTC fwd ms |
|---|---|---|
| ABB OFF | 70.66 | 103.15 |
| Global-min gate=0.25 | 104.04 | 77.28 |
| Global-min gate=1.0 | --- | $\approx 103$ (parity, ABB rarely fires on small graphs) |
| v1 start-ABB | 35.40 | 36.01 |

Enumeration walltime not measured this round (single-seed runs are
dominated by training); the Slashdot k=4 canonical benchmark from the
2026-05-10 v1 report is the place to measure it. Expectation: at gate=1.0
on small graphs (BA/OTC), ABB rarely fires → no walltime change vs non-ABB;
on Slashdot/Epinions where heaps fill quickly, gate=1.0 should still give
a measurable fraction of v1's $4.78\times$ speedup.

## New / removed dependencies

None.

## Open issues / follow-ups

- Run 5-seed BA + OTC gate=1.0 to confirm the "AUC-preserving" verdict over seed variance.
- Bench enumeration walltime on Slashdot at gate=1.0 (heaps should fill fast → benefit > 0).
- Approach 2 surface: the `WeightedSumScorer` is exposed but no PyO3 binding yet --- needs a follow-up if we want it dispatchable from Python.
- The 22 GB OOM diagnosis from 2026-05-10 (memory `project_epinions_topk_per_vertex_oom_2026_05_10.md`) was the `cycle_cache.py` wrapper, not the Rust enumerator. The new global-min variant inherits the same caching call site; new code path tested only against in-process eviction, not the 22 GB regression itself. **Recommend a Bitcoin OTC + cycle_cache=1 + global-min ABB smoke before queuing any Epinions run.**

## Experiment provenance

- Git SHA: `f50d6d6e9e7efbb82eccc0a20d41fab55b5ee5ac` (working tree DIRTY --- listed in `git status`; the changes in this report are the staged ones above plus unrelated WIP files).
- Host: Linux Amaterasu 6.17.0-23-generic, AMD Ryzen 7 3700X 8-core (16 threads), 32 GiB RAM, NVIDIA RTX 2070 SUPER 8 GiB, driver 580.126.09.
- Python 3.13 / PyTorch CUDA / rustc 1.92.0.
- Dataset hashes: as packaged in `data/bitcoin_alpha.csv`, `data/bitcoin_otc.csv` (unchanged from 2026-05-10).
- Random seed: `--seed 0`. Per-run AUC variability already characterised at $\sim 0.01$ on this config (see `project_epinions_edge_cr_null_2026_05_10.md` for the 5-seed comparator).

## Protocol notes

- CLAUDE.md mandates plan before code. This change had its code drafted in the prior conversation window (session continuation); the plan document was written retroactively today to keep the four-artifact record intact. The plan accurately describes what was implemented; no scope drift.
- No `unwrap()` / `expect()` added outside test code.
- No suppressions accumulated. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` to be run before the user requests a commit.
- Per `feedback_no_auto_commit.md`: no commit / no push performed; staging deferred to the user.
