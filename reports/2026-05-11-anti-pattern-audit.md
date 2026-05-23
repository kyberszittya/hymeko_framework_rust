# §6.5 anti-pattern audit — codebase sweep

**Date:** 2026-05-11
**Trigger:** user-authorized §6.5 (Anti-Patterns) amendment to CLAUDE.md; this is the first sweep applying it.
**Scope:** `hymeko_graph/`, `hymeko_py/`, `hymeko_nagare/`, `hymeko_monitor/`, `hymeko_compute/`, `hymeko_wasm/`, `signedkan_wip/`.
**Method:** mechanical greps + targeted reads per anti-pattern listed in CLAUDE.md §6.5; severity assigned by (count × reach).

## Severity legend

- **C** (critical) — actively harms maintainability *and* repeats across many call sites or LOC.
- **M** (major) — one localized but sizeable violation, or a smaller pattern repeated 5–20×.
- **m** (minor) — narrow waiver, or pattern that's clearly transitional.
- **clean** — no violation found.

## Summary table

| # | Anti-pattern | Severity | Count | Worst offender |
|---|---|---|---|---|
| 1 | Cartesian-product PyO3 surface | **C** | 17 `#[pyfunction]` in `cycles.rs` | `hymeko_py/src/cycles.rs` |
| 2 | Algorithm code behind PyO3 boundary | **C** | 20 / 42 free fns in `cycles.rs` are pure-Rust | `hymeko_py/src/cycles.rs` |
| 3 | Per-experiment scaffold duplication | **C** | 98 `run_*.py`; 32 reimplement AUC eval; 4 reimplement `train_val_split` | `signedkan_wip/src/run_*.py` |
| 4 | Long single-file modules ≥ 400 LOC | **M** | 8 Rust + 15 Python | `topk_cycles.rs` (4129), `triton_kernels.py` (1345) |
| 5 | New axis = new function name | **C** | rolled into #1 (same evidence) | — |
| 6 | `clippy::too_many_arguments` allow | **m** | 17 (9 in `cycles.rs`) | `cycles.rs` |
| 7 | String-typed config that should be enum | **M** | 35 `_kind: &str` sites, all in `cycles.rs` | `cycles.rs` |
| 8 | Forward-time flags for structural variants | **clean** | 0 hits | — |
| 9 | Bypassing existing Strategy traits | **M** | 5 `match score_kind` ladders, all in `cycles.rs` | `cycles.rs` |
| 10 | `ulimit -v` on CUDA scripts | **m** | 2 scripts | `run_voc_gomb_matrix_2026_05_11.sh`, `run_overnight_abb_validation_2026_05_11.sh` |
| 11 | Globals / module-level mutable state | **C** | 42 env-var reads in src; 58 inside class methods | `run_final_cell.py` (13×), `n_tuples.py` (7×), `cycle_cache.py` (4×), `signedkan.py` (4×) |

**Totals:**
- Critical: 4 anti-patterns (#1, #2, #3, #11)
- Major: 3 anti-patterns (#4, #7, #9)
- Minor: 2 anti-patterns (#6, #10)
- Clean: 1 anti-pattern (#8)
- #5 rolls into #1

## Detail

### #1 — Cartesian-product PyO3 surface (CRITICAL)

`hymeko_py/src/cycles.rs` exposes **17** `#[pyfunction]`s, of which 16 are signed-cycle/walk enumerator variants differing only by orthogonal axes (mode × scorer × pruner × filter × ABB × batched × tiered). Refactor in flight per Phase 1 (`docs/plans/2026-05-11-cycles-strategy-refactor/plan.tex`): added unified `enumerate_cycles_rs` + migrated 3/7 per-vertex callers (`run_gomb_smoke.py`, `run_cpml_real.py`, `run_cpml_factorial.py`). Remaining work: migrate 4 more callers (`n_tuples.py`, `bench_abb_enum_walltime.py`, `bench_vertex_filter.py`, `topk_cycle_demo.py`), then delete the 8 legacy per-vertex pyfunctions. After that, the top-K global family (4 more variants) is a separate sub-collapse.

`hymeko_wasm/src/wasm.rs` has 5 `#[wasm_bindgen]`s — looked at, not a Cartesian product, no violation.

### #2 — Algorithm code behind PyO3 boundary (CRITICAL)

`hymeko_py/src/cycles.rs` has 42 free functions; **20** are pure-Rust algorithm code (no `PyResult`, no `Py<…>`): `build_csr`, `neighbours`, `has_edge`, `bfs_distances_into`, `dfs_recurse`, `dfs_from`, `dfs_from_pair`, `merge_sinks`, `make_thread_sink`, `make_identity_sink`, `enumerate_parallel`, `canonical_cycle`, `dfs_color_coded`, `lcg_next`, `random_coloring`, `lcg_next_in_range`, `try_one_path_closure`, `dfs_walks_recurse`, `dfs_walks_from`, `compute_m_v`. These belong in `hymeko_graph` — `hymeko_py` should be a thin PyO3 layer + numpy conversion glue only. ~1100 LOC of misplaced algorithm code.

Plan: separate refactor PR after the Strategy collapse lands. Move the listed fns to a new module in `hymeko_graph` (e.g. `hymeko_graph::dfs_unsigned`, `hymeko_graph::color_coding`, `hymeko_graph::path_closure`); `cycles.rs` keeps only the `#[pyfunction]` shells calling into them.

### #3 — Per-experiment scaffold duplication (CRITICAL)

- **98** `run_*.py` files in `signedkan_wip/src/`.
- **4** of them re-implement `train_val_split` (different signatures + names: `_train_val_split`, `train_val_split`, ad-hoc inline permutations).
- **32** of them call `roc_auc_score` directly in their own eval loops (per-script reinvention of train+eval+log scaffold).

Refactor target: introduce `signedkan_wip/src/experiment.py` exposing `train_signed_link_prediction(config, model_factory)` that owns: dataset loading, train/val split (configurable seed), epoch loop, AUC eval per epoch, JSON output schema. Each `run_*.py` becomes a ≤ 30-LOC config builder + model factory call. Migration is mechanical but touches many files; recommend a dedicated PR with one example migration first (e.g., `run_gomb_smoke.py`).

### #4 — Long single-file modules ≥ 400 LOC (MAJOR)

**Rust (8 files):**

```
4129  hymeko_graph/src/topk_cycles.rs        — 12 structs, 14 impls; large but cohesive
2482  hymeko_py/src/cycles.rs                — rolls into #1/#2; will shrink to ~600 after refactor
 695  hymeko_graph/src/spine.rs              — 2 structs, 18 methods; OK
 617  hymeko_py/src/interface_python/api.rs  — needs separate audit (not in this sweep's scope)
 561  hymeko_graph/src/traversal.rs          — 3 structs, 11 free fns + 18 methods; borderline
 509  hymeko_monitor/src/monitor/stl.rs      — STL runtime; cohesive
 488  hymeko_graph/src/traversal_heuristic.rs — 4 structs, 1 trait; OK
 414  hymeko_graph/examples/cycle_stats.rs   — example file (allowlist per CLAUDE.md §1)
```

**Python (15 files):**

```
1345  signedkan_wip/src/triton_kernels.py            — multiple kernel families; decompose by family
1247  signedkan_wip/src/mixed_arity_signedkan.py     — needs audit; likely 2-3 concerns
 780  signedkan_wip/src/run_final_cell.py            — partly rolls into #3 (experiment scaffold)
 758  signedkan_wip/src/run_phase2_mixed_arity.py    — partly rolls into #3
 714  signedkan_wip/src/cycle_cache.py               — packer + cacher + public API in one file
 698  signedkan_wip/src/run_multi_domain_perf_bench.py — rolls into #3
 696  signedkan_wip/src/signedkan.py                  — 5 classes, focused; OK
 644  signedkan_wip/src/splines.py                    — multiple spline kinds; could decompose
 589  signedkan_wip/src/run_compare.py                — rolls into #3
 587  signedkan_wip/src/vision/hymeyolo_q_smoke.py    — q-cycle vision; needs review
 555  signedkan_wip/src/vision/train_circles_ricci.py — model + training in one; partly rolls into #3
 527  signedkan_wip/src/n_tuples.py                   — multi-mode dispatch (rolls into #11)
 489  signedkan_wip/src/run_multi_domain_perf_deep.py — rolls into #3
 469  signedkan_wip/src/chicken/unsupervised.py       — domain module; review later
 455  signedkan_wip/src/cpml.py                       — CPML architecture; cohesive
```

Action: AFTER #3's `Experiment` framework lands, ~6 `run_*.py` files drop below 200 LOC mechanically. Then decompose `triton_kernels.py`, `mixed_arity_signedkan.py`, `cycle_cache.py`, `splines.py` per concern.

### #5 — New axis = new function name

Rolled into #1. Same evidence: `_filtered_rs` → `_filtered_bb_rs` → `_filtered_bb_global_rs` → `_tiered_bb_global_batched_rs`. Eight chained renames in `cycles.rs`.

### #6 — `clippy::too_many_arguments` band-aid (minor)

17 occurrences total across the workspace:
```
hymeko_py/src/cycles.rs:               9
hymeko_graph/src/topk_cycles.rs:       6
hymeko_graph/src/cycle_enum.rs:        1
hymeko_graph/src/traversal_heuristic.rs: 1
```

`hymeko_graph/src/topk_cycles.rs` 6 cases are acceptable — they live on internal algorithm fns that take `&Graph, k_len, &Pruner, &m_v, &keep, &Scorer, fullness_gate` etc. — already a config-struct candidate but not blocking. `cycles.rs` 9 cases all sit on the legacy enumerator wrappers and will vanish with #1's collapse.

Recommendation: when `cycles.rs` collapses, the new `enumerate_cycles_rs` wrapper will still need `#[allow(clippy::too_many_arguments)]` (it IS the Python kwargs surface, so the wide signature is legitimate per §6.5 #6's narrow exception). Internal `compute_m_v` and `dispatch_cycle_enum` should take an `EnumerationConfig` struct, not loose args.

### #7 — String-typed config that should be enum (MAJOR)

35 `_kind: &str` / `_mode: &str` sites in `hymeko_py/src/cycles.rs`. Every internal dispatch on `score_kind`, `pruner_kind`, `filter_kind`, `abb_mode` is a string compare with a `_ => panic`/`Err` arm — the underlying Strategy types (`FractionNegativeScorer`, `BalanceScorer`, `BalanceMode`, etc.) are real Rust types.

Recommended fix:

```rust
// in hymeko_graph::topk_cycles or a new strategy.rs:
#[derive(Debug, Clone, Copy)]
pub enum ScoreKind { Balance, FractionNegative, SignProductAbs, LowRoot }
#[derive(Debug, Clone, Copy)]
pub enum PrunerKind { None, Balance, Unbalanced, Davis }
#[derive(Debug, Clone, Copy)]
pub enum AbbMode  { None, StartLocal, GlobalMin { fullness_gate: f64 } }

impl FromStr for ScoreKind { /* parse at the PyO3 boundary, exactly once */ }
```

`cycles.rs` parses the string kwargs at the PyO3 boundary into typed enums, then the rest of the codebase deals in `ScoreKind` (not `&str`). Eliminates 35 string-compare sites + the 5 `match score_kind` ladders (see #9). Same applies to `pruner_kind`, `filter_kind`, `abb_mode`.

### #8 — Forward-time flags for structural variants (CLEAN)

Greps for `if self.no_*:`, `if self.disable_*:`, `if self.skip_*:` returned zero hits across `signedkan_wip/src/`. The HymeKo-Gömb ablations (`GombNoOuter`, `GombNoMiddle`, `GombNoInner`) and the parallel `MixedArityGomb` correctly use *separate model classes*, not forward-time flags. Good.

### #9 — Bypassing existing Strategy traits at a layer boundary (MAJOR)

5 separate `match score_kind { … }` ladders inside `hymeko_py/src/cycles.rs` (lines 1580, 1664, 1763, 1918, 2436). Each is followed by an inner `match pruner_kind { … }` ladder, giving the Cartesian product evidenced in #1. All five collapse to a single `pick_scorer(&str) -> Box<dyn Scorer>` (or typed `ScoreKind` enum) once #7 lands.

### #10 — `ulimit -v` on CUDA workloads (minor)

2 scripts still use `ulimit -v`:
- `signedkan_wip/experiments/run_voc_gomb_matrix_2026_05_11.sh` — created today; *does not actually call ulimit -v inside* (Stage 1 and 2 have no `ulimit` — false positive from comment header). Verified.
- `signedkan_wip/experiments/run_overnight_abb_validation_2026_05_11.sh` — older script; should be migrated to `systemd-run --user -p MemoryMax=16G` per CLAUDE.md §4.

Action: cleanup pass on `run_overnight_abb_validation_2026_05_11.sh` only.

### #11 — Globals / module-level mutable state (CRITICAL)

**Rust side: clean.** Zero `static mut`, zero `lazy_static!`, zero `once_cell::sync::Lazy` across all hymeko crates.

**Python side: heavy.** 42 `os.environ.get("HSIKAN_*"|"HYMEKO_*", …)` reads in `signedkan_wip/src/`. **58** of these reads sit *inside class methods or deep helper functions*, not at process startup.

Top offenders:
```
13  run_final_cell.py
 7  n_tuples.py
 7  hymeko_train_walker.py
 6  hymeko_driver.py
 4  signedkan.py
 4  profile_stages.py
 4  profile_hsikan_memory.py
 4  cycle_cache.py
 2  triton_kernels.py
 2  splines.py
```

Concrete bad cases (read inside hot paths):
- `signedkan.py:67` — `_resolve_kb_init_tcb()` reads `HSIKAN_KB_PRESET` to compute layer init. Means model construction depends on whatever env var was set in this shell.
- `cycle_cache.py:644` — `lazy_load_construct_k` reads `HSIKAN_TOPK_MODE` mid-call.
- `splines.py` — env-var reads in two places; spline behavior depends on environment.
- `triton_kernels.py:?` — kernel dispatch depends on env var.

**Recommended fix:**

1. Introduce `signedkan_wip/src/runtime_config.py` exposing one frozen `@dataclass(frozen=True) class RuntimeConfig` populated from `os.environ` exactly once at module load.
2. Replace every `os.environ.get("HSIKAN_*", …)` site with `config.hsikan.<field>`.
3. The `config` instance is passed explicitly down the call chain — never imported as a module-level singleton.
4. For backward-compat with current shell-script callers: `RuntimeConfig.from_env(cls)` factory.

This is a substantial migration (42 sites across 10 files) and likely best done dataset-by-dataset. Estimated effort: 1 focused PR per top-3-offender file, then a sweep PR for the long tail.

**Module-level constants (NOT a violation):** `URLS`, `FORMATS`, `DATASET_LOADERS`, `_KB_PRESETS`, `CONFIGS`, `SPACE`, `DATASETS`, `POSITIVITIES`, `PER_FIXTURE_COEF`, `REGRESSION_LOADERS` — all read-only registry dicts populated at module load and never mutated. Per §6.5 #11's exception list: "true program constants" are explicitly allowed.

## Priority ranking for follow-up PRs

In execution order (each unlocks the next):

1. **(P0)** Finish #1 — collapse remaining 13 PyO3 wrappers in `cycles.rs` to use the unified entry, delete legacy names.
2. **(P0)** Together with #1: land #7 (string `_kind` → typed enum) + #9 (centralize Strategy dispatch in one `pick_*`). These three move together.
3. **(P1)** #11 — `RuntimeConfig` dataclass; migrate top-3 offenders (`run_final_cell.py`, `n_tuples.py`, `hymeko_train_walker.py`).
4. **(P1)** #2 — move ~1100 LOC of pure-Rust algorithm code from `hymeko_py/src/cycles.rs` to `hymeko_graph/`.
5. **(P2)** #3 — `Experiment` framework + migrate 5 first `run_*.py` files (the ≥500 LOC ones).
6. **(P3)** #4 — decompose `triton_kernels.py`, `mixed_arity_signedkan.py`, `cycle_cache.py`, `splines.py` after #3.
7. **(P3)** #10 — single-script ulimit cleanup.

CORE.YAML implications: **none of these touch protected paths** — `hymeko_graph`, `hymeko_py`, `signedkan_wip` are all outside the (currently aspirational, not-yet-existing) `crates/hymeko_*` Core protection list. Plan-level approval requested for each P0 PR per CLAUDE.md §2.

## Provenance

- Git SHA at audit: `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (dirty tree, mid-Phase-1 refactor)
- Method: mechanical `grep` + `wc -l` + targeted reads. Heuristics noted per anti-pattern. No model judgment on "is this code good" — only "does this match the §6.5 pattern definitions".
- Not in scope: `hymeko_py/src/interface_python/api.rs` (617 LOC, separate concern around the daemon API). Separate audit if/when needed.
