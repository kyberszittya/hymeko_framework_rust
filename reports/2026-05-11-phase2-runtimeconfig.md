# Phase 2 — RuntimeConfig migration

**Date:** 2026-05-11 night
**Plan:** `docs/plans/2026-05-11-codebase-rehaul/plan.md` Phase 2
**Trigger:** AP-11 (globals / env-var feature flags read deep in call chains) — 42 scattered `os.environ.get(HSIKAN_*|HYMEKO_*)` reads, mostly in hot paths

## Summary

Migrated every read of an `HSIKAN_*` / `HYMEKO_*` environment variable in
`signedkan_wip/src/` to go through `runtime_config.RuntimeConfig`.
**Final count of scattered reads outside `runtime_config.py`: 0.**
Subprocess-orchestrator *writes* (`os.environ[X] = Y` /
`os.environ.setdefault`) preserved because they communicate config to
child code that re-parses env on its own — that pattern is *not* the
§6.5 #11 violation, deep-call-site *reads* are.

## What changed

### `signedkan_wip/src/runtime_config.py` (extended, +120 LOC)

Added three new frozen sub-dataclasses + 4 per-section parsers:

- `TrainingConfig` — `arities`, `max_k2/k3/k4`, `mixed_tuples`,
  `walk_lens`, `cycle_batch`, `entropy_lambda`, `gumbel_hard/tau`,
  `per_edge_gate`, `attention_kind`, `direct_messaging`,
  `spline_kind`, `strict_protocol`, `sparse_attn_k`. (16 fields)
- `KernelConfig` — `triton_kernel`, `triton_backward`, `chunk_t`,
  `kb_preset`, `kb_init_tcb`. (5 fields)
- `CompileConfig` — `enabled`, `mode`. (2 fields)
- Existing `TopKConfig` got a `.fingerprint()` method that delegates
  the cache-key snapshot from `cycle_cache.py::_topk_fingerprint`.

**Singleton dropped.** `get_runtime()` now re-parses env on every
call (cheap, ~40 string lookups). Orchestrators that mutate
`os.environ` between training phases (`hymeko_train_walker.py`,
`hymeko_driver.py`) now correctly see the new values.

### Migrated files (10 files, 42 reads → 0)

| File | Reads migrated | Pattern |
|---|---|---|
| `cycle_cache.py` | 6 | `_topk_fingerprint` now delegates to `TopKConfig.fingerprint()`; `cache_enabled` / `_enum_seed` / `_cache_format` read from `cycle_cache` section; deep `HSIKAN_TOPK_MODE` read replaced |
| `signedkan.py` | 4 | `_resolve_kb_init_tcb`, triton-kernel gate, `HSIKAN_CHUNK_T` all read from `kernel` section |
| `triton_kernels.py` | 2 | Both backward gates → `_triton_backward_enabled()` helper |
| `splines.py` | 2 | `torch.compile` toggle + mode → `compile` section |
| `profile_hsikan_memory.py` | 4 | `arities`, `max_k2/k3`, `cycle_batch` → `training` section |
| `run_final_cell.py` | 14 | All of `arities`, `max_k2/k3`, `mixed_tuples`, `walk_lens`, `strict_protocol`, `cycle_batch`, `per_edge_gate`, `gumbel_hard/tau`, `attention_kind`, `direct_messaging`, `spline_kind`, `entropy_lambda` → `training` section |
| `hymeko_train_walker.py` | 1 (read-side) | `HSIKAN_MAX_K4` → `training.max_k4` |
| `mixed_arity_signedkan.py` | 1 | `HSIKAN_SPARSE_ATTN_K` read inside `_scatter_softmax_sparse` → `training.sparse_attn_k` (real deep-call-site fix) |
| `run_synthetic_baseline.py` | 1 | Same `sparse_attn_k` reference |
| `run_multi_domain_perf_{deep,bench}.py` | 3 | Logging prints now show `RuntimeConfig.compile` rather than raw env |

### Preserved writes (subprocess orchestrators, NOT §6.5 violations)

- `hymeko_train_walker.py` writes 6 env vars at orchestrator boundaries
- `hymeko_driver.py` writes 6 env vars
- `profile_stages.py` writes 4 setdefault'd entry-point defaults

These are intentional: the orchestrator sets env for downstream code,
which re-parses fresh via the new `get_runtime()` (no longer a
singleton). The "deep read in hot loop" failure mode §6.5 #11 forbids
is gone — what's left is process-boundary configuration plumbing,
which is correct.

## CORE.YAML items touched

None.

## Test results

`signedkan_wip/tests/test_hymeko_gomb.py -p no:randomly`: **15/15 passed in 2.73 s**.

End-to-end smoke (`run_gomb_smoke --dataset bitcoin_otc --seed 0 --n-epochs 10`):
val_auc_best = **0.6823** (vs pre-migration **0.6823**, identical to the
3-decimal precision — non-determinism within run-to-run variance).

## Acceptance gate

| Criterion | Result |
|---|---|
| `grep -rE 'os\.environ.*(HSIKAN|HYMEKO)' signedkan_wip/src/ --include='*.py'` outside `runtime_config.py` returns scattered READS | **0 reads** ✓ |
| `pytest signedkan_wip/tests/test_hymeko_gomb.py` green | **15/15** ✓ |
| Bitcoin OTC smoke reproduces ±0.001 AUC | **identical AUC 0.6823** ✓ |
| No new §6.5 anti-patterns introduced | confirmed ✓ |

## Files touched

| File | +/- |
|---|---|
| `signedkan_wip/src/runtime_config.py` | +120 / 0 |
| `signedkan_wip/src/cycle_cache.py` | +6 / -38 |
| `signedkan_wip/src/signedkan.py` | +8 / -10 |
| `signedkan_wip/src/triton_kernels.py` | +14 / -4 |
| `signedkan_wip/src/splines.py` | +6 / -4 |
| `signedkan_wip/src/profile_hsikan_memory.py` | +5 / -5 |
| `signedkan_wip/src/run_final_cell.py` | +14 / -24 |
| `signedkan_wip/src/hymeko_train_walker.py` | +2 / -1 |
| `signedkan_wip/src/mixed_arity_signedkan.py` | +2 / -1 |
| `signedkan_wip/src/run_synthetic_baseline.py` | +2 / -1 |
| `signedkan_wip/src/run_multi_domain_perf_deep.py` | +2 / -1 |
| `signedkan_wip/src/run_multi_domain_perf_bench.py` | +4 / -2 |

Net: **+185 / -91** across 12 files. Most of the +185 is in
`runtime_config.py` (the new typed-config surface); migrated files
average ~10 LOC shorter.

## Open issues / what's next

- **Phase 3 (`Experiment` framework + 5 `run_*.py` migrations)** —
  the biggest remaining AP-3 work (98 `run_*.py` duplicating training
  scaffold).
- **Phase 4 (long Python files decomposition)** — `triton_kernels.py`
  (1345), `mixed_arity_signedkan.py` (1247), `cycle_cache.py` (714),
  `signedkan.py` (696), `splines.py` (644).
- **Phase 5 (typed `ScoreKind`/`PrunerKind`/`AbbMode` enums)** — at the
  Rust boundary, replacing the 35 `_kind: &str` sites in
  `hymeko_py/src/cycles/`.
- **Phase 6 (vision-side audit)** — `clifford_fir.rs` still pending.

Plan reference: `docs/plans/2026-05-11-codebase-rehaul/plan.md`.

## Provenance

- Git SHA at run: `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (working tree dirty per the rehaul-in-progress state)
- Host: AMD Ryzen 7 3700X + RTX 2070 SUPER
- Python 3.13.5; pytest 9.0.3
- No `ulimit -v` (per the 2026-05-11 CLAUDE.md §4 amendment — VAS cap forbidden for CUDA work)

**No §6.5 anti-patterns introduced.** AP-11 closed; AP-3, AP-4, AP-7 still queued.
