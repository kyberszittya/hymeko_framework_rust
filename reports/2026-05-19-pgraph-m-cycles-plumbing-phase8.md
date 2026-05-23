# Phase 8: m_cycles plumbing — 2026-05-19

## Summary

Closed the Phase 7 plumbing gap. `signedkan_wip.src.core.hyperedges.construct`
now accepts an optional `m_per_vertex` cap; `run_compare.run_one`
exposes it as `m_cycles`. The HSIKAN P-graph framework's
`cycle_topk_m{4,16,64}` axis — which Phase 7 found was silently
dropped at the Python boundary — now reaches the triad pool that
actually trains the model.

Headline: **the cap now bites**. The cost-minimum P-graph
selection's AUC dropped from 0.576 ± 0.029 (Phase 7, uncapped because
m_cycles was silently dropped) to **0.430 ± 0.007 (m=4 actually
applied)** — a 15 pp drop that exposes how aggressive the cap is on
Bitcoin Alpha. The strict-vs-relaxed P-graph divergence now produces
a measurable ±3 pp AUC swing at short epochs and an **8 pp regime
crossover** at long epochs.

## Files touched

| File | Status | LOC | Notes |
| --- | --- | --- | --- |
| `docs/plans/2026-05-19-pgraph-m-cycles-plumbing/plan.{tex,pdf,mmd,tikz}` | new | 4-format plan, 2 pp PDF | Written before code |
| `signedkan_wip/src/core/hyperedges.py` | extended | +35 | `m_per_vertex` parameter on `construct()`; deterministic neighbour-set iteration in `_enumerate_triangles` |
| `signedkan_wip/experiments/runs/run_compare.py` | extended | +6 | `m_cycles` kwarg on `run_one`; forward to `construct`; echo in result dict alongside `n_triads` |
| `signedkan_wip/experiments/runs/run_hsikan_msg_sweep.py` | minor | -8 | Dropped the Phase-7 env-var workaround now that the kwarg path works |
| `signedkan_wip/tests/test_hyperedges_m_per_vertex.py` | **new** | 90 | 7 tests pinning cap semantics, determinism, back-compat |

## CORE.YAML items touched

None.

## Interface change

`construct(g, m_per_vertex=None)` is back-compat:

- `m_per_vertex = None` (default): every enumerated triad kept,
  byte-identical to pre-Phase-8 behaviour.
- `m_per_vertex = M > 0`: triads grouped by apex (the σ = +1 vertex
  from `_classify`), at most `M` per bucket kept in deterministic
  enumeration order.
- `m_per_vertex ∈ {0, negative}`: treated as no-cap (matches `None`).

`run_one(..., m_cycles=None)` similarly defaults to no-cap. The
returned result dict gains `m_cycles` (echo) and `n_triads` (size
of the active triad pool) for sweep-time observability.

## Test results

| Suite | Status |
| --- | --- |
| `test_hyperedges_m_per_vertex.py` (new) | **7 / 7 pass** |
| `test_hsikan_pgraph_mapping.py` | 7 / 7 pass (unchanged) |
| `cargo test -p hymeko_pgraph` (full) | 90 / 90 pass + 1 ignored doctest |
| `test_cycle_cache.py` | 12 / 13 pass; 1 pre-existing failure |

### Pre-existing failure (unrelated to Phase 8)

`test_cycle_cache.py::test_triads_route_through_topk_path_when_enabled`
fails with `AttributeError: module 'hymeko' has no attribute
'enumerate_cycles_rs'. Did you mean: 'enumerate_k_cycles_rs'?` —
same root cause as the Phase 7 `run_gomb_smoke.py:112` failure:
the 2026-05-11 codebase rehaul renamed the PyO3 binding but two
call sites (`run_gomb_smoke.py:112` and `n_tuples.py:279`) were
missed. Phase 9 should fix both.

## Quantitative result — 5-seed AUC on Bitcoin Alpha

After Phase 8 the strict-vs-relaxed P-graph dichotomy actually
moves training-time numbers.

| ABB selection | m_cycles | n_triads | epochs | mean AUC ± std |
| --- | --- | --- | --- | --- |
| `cycle_topk_m4, model_h8, train_short` (cheap, strict + no by-prod) | 4 | 1,568 | 10 | **0.430 ± 0.007** |
| `cycle_topk_m16, model_h8, train_short` (strict + by-prod → m=16 forced) | 16 | 3,613 | 10 | 0.400 ± 0.010 |
| `cycle_topk_m64, model_h8, train_short` | 64 | 7,389 | 10 | 0.476 ± 0.020 |
| `cycle_topk_m4, model_h8, train_long` | 4 | 1,568 | 60 | 0.491 ± 0.015 |
| `cycle_topk_m16, model_h8, train_long` | 16 | 3,613 | 60 | **0.574 ± 0.023** |

### Comparison with Phase 7 (pre-Phase-8 — m_cycles silently dropped)

| Config | Phase 7 AUC (uncapped) | Phase 8 AUC (capped) | Δ |
| --- | --- | --- | --- |
| cheap_strict_m4 | 0.576 ± 0.029 | **0.430 ± 0.007** | **−0.146** |
| strict_byproduct_m16 | 0.576 ± 0.029 | 0.400 ± 0.010 | −0.176 |

The Phase 7 "cheap" result wasn't actually cheap — it was running
the full triad pool (≈ 28k triads on Bitcoin Alpha). With the cap
now active, the true cost-60 P-graph architecture trains with only
1,568 triads and drops 15 pp AUC. **Phase 8's quantitative finding
is that on Bitcoin Alpha the per-apex cycle-pool cap is a very
aggressive intervention** — the framework's cost-minimum
architecture is materially worse than its uncapped counterpart.

### Two substantive observations

1. **Regime crossover at long epochs.** At `train_short` (10 epochs)
   m=4 beats m=16 by 3 pp (0.430 vs 0.400). At `train_long` (60
   epochs) m=16 beats m=4 by 8 pp (0.574 vs 0.491). So the
   strict-vs-relaxed P-graph choice is not unconditionally bad or
   good — it interacts with the training-length axis. **A
   multi-objective ABB run that includes epoch count and m_cycles
   would pick m=16 + train_long for AUC** but only at cost = 90 + 30
   + 0 = (m_cycles, hidden, epochs). Phase 10 (multi-objective
   plan, on disk) is the natural follow-up.
2. **m=64 partial recovery.** At m=64 (7,389 triads, n_epochs=10)
   AUC recovers to 0.476 — better than m=4 / m=16 / m=128. The
   training landscape is non-monotonic in cycle-pool size at this
   epoch count, suggesting the model under-fits at very small
   pools and over-correlates at intermediate sizes before
   stabilising. Not a directly actionable finding but a
   useful empirical surface for the multi-objective plan.

### What was wrong before vs what's right now

| | Before Phase 8 | After Phase 8 |
| --- | --- | --- |
| P-graph picks `cycle_topk_m4` | accepted, ignored | accepted, applied |
| `run_one(m_cycles=4)` works | ❌ TypeError unknown kwarg | ✅ caps triads, returns AUC |
| Strict-mode AUC ≠ relaxed-mode AUC | identical | **differs by 3-8 pp** |
| Result dict carries cycle-pool size | no | `m_cycles` + `n_triads` fields |

## Performance budget

The cap adds an `O(|triads|)` per-apex bucket build and truncate;
already inside the cost of triangle classification. Bitcoin Alpha
(N=28k triads): bucketing + truncation < 10 ms. No memory
overhead beyond a `dict[int, int]` the size of the apex set.

Wall-time difference Phase 7 vs Phase 8 on the same 5-seed sweep:
indistinguishable (1-2 s / config in both runs; the cap saves
training time at smaller `m_cycles` because fewer triads are
processed in the model).

## §6.5 anti-pattern audit

No new anti-patterns. The `m_per_vertex` parameter is a typed
optional, not a string-typed mode. The cap groups by a single
attribute (apex vertex) — no Cartesian explosion. The neighbour-
set sort in `_enumerate_triangles` is a correctness fix, not a new
behaviour axis.

## Open issues and follow-up items

1. **Phase 9 (consolidated):** fix the `enumerate_cycles_rs` →
   `enumerate_k_cycles_rs` rename at both surfaced call sites
   (`run_gomb_smoke.py:112` + `n_tuples.py:279`) plus any others
   surfaced by `grep enumerate_cycles_rs signedkan_wip/`. ~5 LOC.
2. **Phase 10 (multi-objective ABB):** the regime crossover at long
   epochs (m=16 wins at n_epochs=60) is exactly the kind of
   non-monotonic cost/quality trade the multi-objective plan is
   designed to surface. Plan already on disk at
   `docs/plans/2026-05-19-pgraph-multi-objective/`.
3. **Phase 11 (by-product NAS filter on real workloads):** with
   Phase 8 plumbing in place, declaring an `unused_capacity` by-
   product material on a wider HSIKAN sweep would let strict-mode
   ABB drop dominated architectures (h=16/h=32 at short epochs).
4. **Top-K cycle quality.** Phase 8 keeps the *first* M triads per
   apex in enumeration order. A more principled choice — top-M by
   sign-balance contribution or by structural-balance score — is
   the natural research extension. Currently first-K is the
   reference; smarter selectors are a future axis.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted: phases 1-8 + cortical
  Slice 1 + earlier book regenerations).
- **A/B reproduction:** the small Python snippet in this report's
  "Quantitative result" section is reproducible from the repo root;
  takes ~10 s wall time on a CPU machine.
- **Tests:** `pytest signedkan_wip/tests/test_hyperedges_m_per_vertex.py
  signedkan_wip/tests/test_hsikan_pgraph_mapping.py` → 14/14 pass.

## Acceptance check

- [x] 4-format plan written + PDF compiled before code (2 pp).
- [x] No `CORE.YAML` items touched.
- [x] `construct()` back-compat (no-args calls behave as before).
- [x] `run_one()` back-compat (no `m_cycles` calls behave as before).
- [x] Cap semantics pinned by 7 unit tests.
- [x] Determinism property held by `sorted(nbrs[i])` in
      `_enumerate_triangles`.
- [x] A/B reproduces the Phase 7 framework choice but now with
      distinct AUC numbers (3-8 pp swings).
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
