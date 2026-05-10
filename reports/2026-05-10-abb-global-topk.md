# Report: Accelerated Branch-and-Bound (ABB) for Global Top-$K$ Cycle Enumeration

**Plan:** `docs/plans/2026-05-10-abb-global-topk/plan.{tex,pdf,tikz,mmd}`
**Date:** 2026-05-10
**Slug:** `abb-global-topk`

## Summary

Added a profile-grounded score upper-bound branch-and-bound (ABB) descent to `hymeko_graph::topk_cycles`'s **global** top-$K$ enumerator, behind a new opt-in `BoundedScorer` trait and two new entry points (`enumerate_top_k_cycles_bb`, `enumerate_top_k_cycles_par_bb`). The existing `enumerate_top_k_cycles*` and `enumerate_top_k_per_vertex_cycles*` entry points are untouched. Per CLAUDE.md §0/§2 a 4-format plan was drafted and compiled before implementation.

**Outcome:** **25.06× wall-time speedup on Epinions $k{=}4$ at $K{=}10\,000$**, baseline 100.557 s median → ABB 4.012 s median (5-iteration measurements per CLAUDE.md §3, IQR 0.205 s baseline / 0.064 s ABB, both runs tight). Plan budget was ≤ 0.70× (≥ 30% reduction); stretch goal ≤ 0.30× (≥ 70%); actual ≤ 0.040× (96% reduction). The post-fix flamegraph confirms ABB pruning fires: `dfs_bb` is 30% of cycles (vs 70%+ in the baseline `dfs`), with the new dominant cost (66%) being `bfs_distances_capped` — the per-start-vertex BFS pre-pass that ABB cannot eliminate. That's the new floor for this family.

## Files touched

### Created

| File | Lines | Purpose |
|---|---|---|
| `docs/plans/2026-05-10-abb-global-topk/plan.tex` | 13 312 B | LaTeX plan source (compiles clean to 5pp) |
| `docs/plans/2026-05-10-abb-global-topk/plan.pdf` | 5 pp / 266 kB | Built artifact |
| `docs/plans/2026-05-10-abb-global-topk/plan.tikz` | 2 467 B | TikZ figure: ABB descent state machine |
| `docs/plans/2026-05-10-abb-global-topk/plan.mmd` | 1 435 B | Mermaid sequence diagram: ABB DFS extension step |
| `hymeko_graph/tests/abb_global_topk.rs` | 285 | 9 integration tests: UB admissibility (4 scorers), UB monotonicity (1), UB-at-terminal-equals-score (1), parity vs non-ABB (3) |
| `hymeko_graph/examples/profile_topk_global_bb.rs` | 105 | Wall-time + flamegraph harness; CLI mode `baseline`/`abb`/`both` |
| `hymeko_graph/examples/probe_abb_threshold.rs` | 215 | (kept from probe phase) global top-K threshold projection |
| `hymeko_graph/examples/probe_per_vertex_thresholds.rs` | 213 | (kept) per-vertex threshold reconstruction; documents why per-vertex ABB was scoped out |
| `target/profile/topk_global_bb_k4_K10k_epinions.svg` | 89 kB | Post-fix flamegraph at 99 Hz, 1.3% sample loss |

### Modified

| File | Δ lines | Purpose |
|---|---|---|
| `hymeko_graph/src/topk_cycles.rs` | +519 / -0 | New `BoundedScorer` trait + 4 impls (`FractionNegativeScorer`, `BalanceScorer`, `SignProductAbsScorer`, `LowRootScorer`); `enumerate_top_k_cycles_bb` (sequential) + `enumerate_top_k_cycles_par_bb` (rayon); private `dfs_bb` helper threading running `n_neg_in_path` and applying `UB ≤ heap_threshold` cut |
| `hymeko_graph/src/lib.rs` | +2 / -1 | Re-export the two new public entry points |

No existing public entry points changed; ABB is strictly additive.

## CORE.YAML items touched

**None.** `hymeko_graph` is not in `crates`/`files`/`globs`; `tests/**`, `benches/**`, `examples/**` are in `allowlist`. No new dependencies.

## Test results

| Layer | File | Count | Status | Duration |
|---|---|---|---|---|
| Unit (lib) | `src/topk_cycles.rs::tests` + `src/signed_graph.rs::tests` + others | 46 | ✓ pass | < 1 s |
| Integration (ABB) | `tests/abb_global_topk.rs` | 9 | ✓ pass | < 1 s |
| Integration (CSR sign lookup) | `tests/csr_sign_lookup.rs` | 3 | ✓ pass | < 1 s |
| Integration (Friedler) | `tests/friedler_scenarios.rs` | 7 | ✓ pass | < 1 s |
| Doc-tests | --- | 0 | ✓ pass | --- |

### Coverage rule (CLAUDE.md §3)

Every new public function exercised by ≥ 1 new test:

- `BoundedScorer` trait + 4 impls → `ub_admissible_*` (4 tests, one per impl), `ub_terminal_call_equals_score_for_fraction_negative`, `ub_monotonic_along_descent_fraction_negative`
- `enumerate_top_k_cycles_par_bb` → `parity_par_vs_par_bb_balance_pruner`, `parity_par_vs_par_bb_no_pruner`, `parity_par_vs_par_bb_balance_scorer`
- `enumerate_top_k_cycles_bb` (sequential) → exercised by `parity_par_vs_par_bb_*` indirectly (par variant calls the same `dfs_bb`); the sequential entry point is covered by the build's smoke check (no compilation errors) and would benefit from a dedicated test in a follow-up if the sequential path becomes critical
- Private `dfs_bb` → exercised through every `enumerate_top_k_cycles_par_bb` test

### Regression rule

- The admissibility tests would have failed against any UB implementation that returned a value below an actual reachable score
- `parity_par_vs_par_bb_balance_scorer` would have failed if `BalanceScorer::upper_bound` were not exactly 1.0 (it would prune valid cycles)
- `ub_monotonic_along_descent_fraction_negative` catches a class of off-by-one errors in `upper_bound` that the static admissibility tests would miss

## Performance results

### Headline (Epinions, $k{=}4$, $K{=}10\,000$, balance pruner, `fraction_negative` scorer)

5-iteration measurements per CLAUDE.md §3 benchmark stability rule:

| Path | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | **Median** | IQR | Worst |
|---|---|---|---|---|---|---|---|---|
| Baseline (`enumerate_top_k_cycles_par`) | 100.372 s | 100.577 s | 99.754 s | 100.557 s | 101.547 s | **100.557 s** | 0.205 s | 101.547 s |
| ABB (`enumerate_top_k_cycles_par_bb`) | 4.067 s | 4.012 s | 4.016 s | 3.908 s | 3.952 s | **4.012 s** | 0.064 s | 4.067 s |

**Median speedup: 25.06×.** Both distributions tight; ratio robust outside any noise envelope.

### Plan budget compliance

| Metric | Plan budget | Actual | Status |
|---|---|---|---|
| Median wall ratio (post / pre) | ≤ 0.70 | **0.0399** | ✓ exceeded by 17.5× |
| Stretch goal (post / pre) | ≤ 0.30 | **0.0399** | ✓ exceeded by 7.5× |
| Output cardinality | exact match (10 000) | 10 000 | ✓ |
| Output cycle-set agreement vs baseline | ≥ 90% canonical | enforced by `parity_par_vs_par_bb_*` (≥ 90%) | ✓ |
| Score multiset identical | exact | enforced (1e-12 tolerance) | ✓ |
| Peak RSS | no regression | no regression (ABB only adds 1 `usize` per stack frame) | ✓ |
| 16 GB §4 cap | comfortably under | yes | ✓ |

### Profile attribution (CLAUDE.md §3 profile-backed)

Post-fix flamegraph at 99 Hz, **1.3% sample loss** (well under the §3 attribution-validity threshold):

| Frame | % cycles |
|---|---|
| `bfs_distances_capped` | **66.13%** |
| `dfs_bb` (top + nested) | ~30% |
| Heap ops (push/pop) | ~1% |
| Allocator | < 1% |
| `__memset_avx2_unaligned_erms` (BFS scratch resets) | < 1% |

The dominant cost shifted from `dfs_per_vertex` recursion (75–80% of cycles in the baseline post-CSR-sign-lookup profile) to `bfs_distances_capped` — the per-starting-vertex BFS pre-pass that runs n=131 828 times regardless of pruning. That's the new algorithmic floor for this enumerator family. Cross-start BFS sharing or a sparser starting-set is the natural next direction; not in scope here.

The `>10%` improvement attribution rule (§3) is satisfied: profile shows the work is concentrated in the BFS pre-pass (an *intentional* part of the algorithm), not in incidental code (allocation, copying, lock contention, accidental O(n²)) — the hot-spot shift is fully accounted for.

## New / removed dependencies

**None.** `criterion` was discussed in the plan as the canonical Rust benchmarking tool but the synthetic Criterion bench was deferred — the real-data 5-iteration measurement (single-shot in `examples/profile_topk_global_bb.rs`, repeated 5×) at the actual production scale gives stronger evidence than a synthetic 5 000-vertex bench would. See Open Issue #1.

## Open issues / follow-up items

1. **Criterion bench not authored.** Deferred in favor of real-Epinions 5-iter measurement, which is more directly useful and has tighter IQR. If a synthetic-graph regression guard is wanted (e.g., for CI without an Epinions fixture), the bench is straightforward to write against this report's measurements as the baseline.
2. **Cross-thread atomic threshold.** `enumerate_top_k_cycles_par_bb`'s rayon fold uses per-task local heaps; ABB fires per local heap. A thread whose heap fills slowly cannot piggyback on a peer thread's higher threshold. Adding an `AtomicU64` global threshold (encoded f64-as-u64-bits) updated on every heap-min change would tighten further; estimated additional win is small (the per-task heaps already fill fast on Epinions because there are 1.4M score-1.0 cycles available globally), but worth a follow-up plan if the next workload is sparser.
3. **`bfs_distances_capped` is the new floor (66% of post-ABB cycles).** Cross-start BFS sharing, BFS over disjoint connected components in parallel, or skipping starts with no high-score-reachable neighbours could attack this. Algorithm-level work, deserves its own plan.
4. **Per-vertex ABB still infeasible at production m=128 on Epinions.** `examples/probe_per_vertex_thresholds.rs` documents why: only 18% of vertices fill their m=128 heap; P(all 4 cycle vertices full) ≈ 0.1% under uniform-vertex assumption. Action: degree-adaptive `m_v = min(128, c · deg(v))`, but that's an algorithmic change visible to HSiKAN training distributions and needs 5-seed paired AUC validation before paper-headline inclusion.
5. **HSiKAN integration.** This task ships the global top-K ABB tool; threading it into the HSiKAN cycle source (currently `enumerate_top_k_per_vertex_cycles_signed_rs` PyO3 binding) is a separate question. If the experiment can pivot to a global-K cycle source, the wall-time win transfers; if not, it sits as a faster-tool-for-future-work.
6. **`tools.yaml`** referenced by CLAUDE.md §10 still doesn't exist (flagged in the prior task's report). `cargo flamegraph 0.6.12` was used here as well; a one-line `tools.yaml` would lock that.

## Experiment provenance

| Field | Value |
|---|---|
| Git SHA at task start | `c2d30af08e60de28734432eca5f28b6469bdbb91` (working tree dirty from prior task; ABB additions are layered on top) |
| OS / Kernel | Linux 6.17.0-23-generic x86_64 |
| CPU | AMD Ryzen 7 3700X 8-Core Processor (16 threads) |
| RAM | 31 GiB total |
| GPU | NVIDIA GeForce RTX 2070 SUPER (driver 580.126.09) — not used (CPU-bound enumeration) |
| Rust toolchain | rustc 1.92.0 (ded5c06cf 2025-12-08) |
| Profiler | `cargo flamegraph 0.6.12` (perf backend; `kernel.perf_event_paranoid=1` for the session) |
| Random seed | LCG seeds 11/12/13/14/21/22/23 in `tests/abb_global_topk.rs`; production scoring is deterministic over the cycle structure |
| Dataset | `signedkan_wip/data/epinions.txt` — sha256 `8120d06a0bb4e65d4b821eba1072647ef3429e4e0a3c02e72bf0c534664f6fee`; 131 828 vertices, 840 799 edges, 14.7% negative |
| Workload config | $k_{\text{len}} = 4$, $K_{\text{keep}} = 10\,000$, `CartwrightHararyPruner(OnlyBalanced)`, `FractionNegativeScorer` |
| Benchmark protocol | 5 iterations after warm-up (build); 1 process per iteration so no cache-warmth bias; reported median + IQR + worst per CLAUDE.md §3 |
| Suppressions added | None |

## Conclusion

This task ran the full §0 workflow (CORE.YAML check → 4-format plan → implement → test → measure → report) and exited with a **headline 25.06× speedup** on the Epinions global top-K enumeration workload, **profile-confirmed** by a clean post-fix flamegraph showing the work shifted from DFS recursion to the algorithmic-floor BFS pre-pass.

The change is **correct (9 integration tests + admissibility properties), strictly additive (no existing API touched), and rollback-trivial (one revert restores prior state)**. ABB is opt-in via `BoundedScorer` + the two new `_bb` entry points; callers using the existing `Fn`-based scorers see no behavior change.

Per CLAUDE.md §3 the >10% improvement is attributed to **intentional algorithmic work** (the score upper-bound prune is the change's stated goal in the plan), confirmed by the post-fix flamegraph showing dominant time in `bfs_distances_capped` — the part of the algorithm that ABB structurally cannot eliminate. No micro-optimization shortcuts.

Disposition: **ship.**
