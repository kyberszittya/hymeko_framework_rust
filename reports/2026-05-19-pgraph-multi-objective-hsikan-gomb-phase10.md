# Phase 10: multi-objective P-graph ABB in HSIKAN/Gömb — 2026-05-19

## Summary

Wired the existing multi-objective ABB Rust core (9 unit tests
passing as of the morning P-mo phase) through the JSON DTO, the
HSIKAN sweep driver, and a new multi-cost fixture. The headline
quantitative result: **multi-objective ABB recovers +25.5 pp AUC**
on Bitcoin Alpha (0.430 ± 0.007 → 0.685 ± 0.025) by re-weighting
the cost vector toward `quality_drop` instead of `gpu_cost`. The
single-criterion (scalar) ABB picks the cost-minimum 0.430
architecture; the quality-weighted MO ABB picks 0.685.

This is what the Pimentel-meeting narrative needs: a concrete +25
pp AUC swing on real signed-graph training, driven entirely by the
weight vector — same fixture, same model, same data, same seeds.

## Files touched

| File | Status | LOC | Notes |
| --- | --- | --- | --- |
| `docs/plans/2026-05-19-pgraph-multi-objective-hsikan-gomb/plan.{tex,pdf,mmd,tikz}` | new | 4-format plan (3 pp PDF) | Written before code |
| `hymeko_pgraph/src/dump.rs` | extended | +30 | `cost_dimensions`, `cost_weights_echo`, `abb_cost_breakdown` added to `PgraphAnalysisJson`; populated from `LoweredPGraph.cost_dimensions` / `AbbOptions.cost_weights` / per-unit sums |
| `hymeko_pgraph/tests/axiom_witness.rs` | extended | +50 | +1 test (`dump_dto_phase10_multicost_fields_echo`) pinning the three new fields on the multi-cost fixture |
| `data/hsikan/sweep_msg_multicost.hymeko` | **new** | 95 | HSIKAN architecture P-graph with 3 cost dimensions per unit; quality_drop numbers derived from Phase 8 measurements |
| `signedkan_wip/experiments/runs/run_hsikan_msg_sweep.py` | extended | +10 | `--weights "w1,w2,w3"` flag forwarded to dump binary |
| `signedkan_wip/experiments/runs/run_gomb_msg_sweep.py` | extended | +10 | Same `--weights` flag |
| `signedkan_wip/tests/test_pgraph_multiobjective_pipeline.py` | **new** | 110 | 7 end-to-end tests: dimensions alphabetised, scalar-fallback echo, three-regime divergence, round-trip echo, breakdown sums, dot-product consistency, axiom-cert independence |

## CORE.YAML items touched

None.

## Quantitative result — 5-seed AUC on Bitcoin Alpha

For each weight regime, the dump binary picks an ABB architecture
and the Python harness trains it via `run_compare.run_one`:

| `--weights` | Cost dim weighted | ABB picks | h | epochs | m_cycles | scalar cost | **mean AUC ± std** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1,0,0` | gpu only | `m4+h8+short` | 8 | 10 | 4 | 40 | **0.430 ± 0.007** |
| `0,1,0` | **quality only** | `m64+h16+long` | 16 | 60 | 64 | 380 | **0.685 ± 0.025** |
| `0,0,1` | time only | `m4+h8+short` | 8 | 10 | 4 | 40 | 0.430 ± 0.007 |
| `1,5,1` | balanced, quality-heavy | `m4+h8+long` | 8 | 60 | 4 | 140 | 0.491 ± 0.015 |

### Findings

1. **+25.5 pp AUC by re-weighting alone.** Same fixture, same
   model, same 5 seeds — only the weight vector changes. Scalar
   single-criterion ABB picks 0.430; quality-weighted MO ABB
   picks 0.685.
2. **Pareto-aware balanced pick.** Under
   `--weights "1,5,1"` ABB picks `m4+h8+long`: keep the cheap
   cycle pool and cheap hidden width, but invest in long training
   to recover most of the quality drop. AUC 0.491 — better than
   the all-cheap pick (0.430) at one-third the scalar cost of the
   all-quality pick (140 vs 380). This is the kind of trade the
   PSE community will care about when CAPEX + OPEX + CO₂ + H₂O
   are jointly optimised.
3. **Cost-dimensions are alphabetised** (`gpu_cost`,
   `quality_drop`, `time_cost`). The `--weights` vector aligns by
   position. Documented in the `.hymeko` fixture and the
   `run_*` driver `--help` strings.

## Interface change

The new DTO fields are **additive** (existing consumers ignore
unknown JSON fields):

```rust
pub struct PgraphAnalysisJson {
    // ...existing Phases 1-9 fields...
    /// Alphabetised list of cost-dimension names from the
    /// lowered graph; empty for scalar-only fixtures.
    pub cost_dimensions: Vec<String>,
    /// Echo of the active `cost_weights` (None ⇒ scalar fallback).
    pub cost_weights_echo: Option<Vec<f64>>,
    /// Per-dimension sum of cost vectors over the ABB-selected
    /// units (the full pre-dot-product picture).
    pub abb_cost_breakdown: Option<Vec<(String, f64)>>,
}
```

Python drivers:

```bash
python -m signedkan_wip.experiments.runs.run_hsikan_msg_sweep \
    --pgraph data/hsikan/sweep_msg_multicost.hymeko \
    --algorithm abb --weights "1,5,1" \
    --dataset bitcoin_alpha --seeds 0 1 2 3 4
```

## Test results

| Suite | Result | Phase 10 additions |
| --- | --- | --- |
| `cargo test -p hymeko_pgraph` | 91 / 91 + 1 ignored doctest | +1 (`dump_dto_phase10_multicost_fields_echo`) |
| `axiom_witness.rs` | 28 / 28 | +1 |
| `multi_objective.rs` | 9 / 9 | unchanged (pre-Phase-10) |
| `test_pgraph_multiobjective_pipeline.py` | **7 / 7** | all new |
| `test_hsikan_pgraph_mapping.py` | 7 / 7 | unchanged |
| `test_hyperedges_m_per_vertex.py` | 7 / 7 | unchanged |
| `test_cycle_cache.py` | 13 / 13 | unchanged |
| `test_gomb_pgraph_driver.py` | 5 / 5 | unchanged |

No regressions on any Phase 1–9 suite.

## §6.5 anti-pattern audit

No new anti-patterns. The three new DTO fields are typed
`Option<...>` (no string-typed config); the multi-cost fixture
uses the existing tagged-child syntax; the driver `--weights`
flag forwards directly to the binary (no new dispatch matrix).

## Open issues and follow-up items

1. **Gömb multi-cost fixture.** `sweep_msg_gomb.hymeko` is still
   scalar-only. The driver accepts `--weights` but on a
   scalar-only graph the weights are vacuous. Build
   `sweep_msg_gomb_multicost.hymeko` (parallel structure to the
   HSIKAN one) for symmetric coverage.
2. **Real PSE workload integration.** The methanol_synthesis
   fixture (the original P-mo target) ships with full
   CAPEX/OPEX/CO₂/H₂O. Adding a sweep driver that runs the same
   fixture under multiple PSE-policy weight vectors (current
   energy prices, current carbon prices) is the natural
   downstream demo for Pimentel.
3. **Pareto-frontier enumeration.** The current MO ABB is
   weighted-sum only — picks a single optimum per weight vector.
   A separate plan would enumerate the Pareto frontier directly,
   which is the more honest answer for genuinely multi-objective
   problems (Phase 10's `(1,5,1)` balanced pick is a heuristic
   trade, not a frontier point).
4. **Empirical wiring of training-time AUC into `quality_drop`.**
   The `quality_drop` numbers in the HSIKAN fixture are
   hand-curated from Phase 8's 5-seed measurements. A
   self-improving loop would run the actual training, measure
   the AUC, and feed back as `quality_drop = (max_AUC - this_AUC)
   * 100`. That's a small follow-up.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted: phases 1–10 +
  cortical Slice 1 + earlier book regenerations).
- **Tests:** all 91 Rust pgraph + 21 Python (Phase 10-touched) pass.
- **A/B:** the Phase 10 quantitative table was produced by a
  ~6-line inline Python snippet running `run_compare.run_one`
  on each of the 4 ABB-selected architectures × 5 seeds.
  Reproducible from the repo root in ~10 s wall-time.

## Acceptance check

- [x] 4-format plan + PDF compiled before code (3 pp).
- [x] No `CORE.YAML` items touched.
- [x] DTO additions are back-compat (additive, `Option<T>` typed).
- [x] All 91 Rust pgraph tests pass.
- [x] All 7 new Python MO pipeline tests pass.
- [x] No regression on any Phase 1–9 test.
- [x] HSIKAN multi-cost fixture parses and produces distinct
      ABB selections under distinct weight vectors.
- [x] Quantitative result: 4 weight regimes × 5 seeds, +25.5 pp
      AUC swing from quality-weighted MO ABB.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
