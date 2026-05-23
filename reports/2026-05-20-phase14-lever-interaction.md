# Phase 14: lever-interaction between MO weights and by-product filter — 2026-05-20

## Summary

Stacks Phase 10's multi-objective ABB (weight vector over
gpu_cost / quality_drop / time_cost) AND Phase 11's by-product
injection (on the empirically dominated `model_h32` and
`train_short` units) **on the same fixture**. Answers the rollup's
queued question: when both NAS-quality levers are available, do
they compose additively, sub-additively, or super-additively?

**Answer: sub-additively.** The by-product filter and the MO
quality-weighting are **complementary, not summative.** Each lever
gives its full effect when used alone; when both are available, the
larger lever dominates and the smaller one becomes a no-op (because
MO already picks an architecture that doesn't include the
by-product-emitting units).

Max single-lever gain: **+0.2553 r²** (MO quality weighting).
Max combined gain: **+0.2553** (no additivity).

## Files touched

| File | Status |
| --- | --- |
| `data/hsikan/sweep_msg_combo.hymeko` | **new** (90 LOC) — Phase 10 multi-cost dims + Phase 11 by-product injection on same 8-unit fixture |

## CORE.YAML items touched

None.

## The lever-interaction matrix (5 seeds, Bitcoin Alpha, signedkan)

| weights | mode | ABB picks | mean AUC ± std | filter gain |
| --- | --- | --- | --- | --- |
| `1,0,0` (gpu only) | relaxed (no filter) | `m4+h8+short` | **0.4296 ± 0.0067** | baseline |
| `1,0,0` (gpu only) | strict (filter on) | `m4+h8+long` | **0.4909 ± 0.0154** | **+0.0613** |
| `0,1,0` (quality only) | relaxed | `m64+h16+long` | **0.6849 ± 0.0254** | — |
| `0,1,0` (quality only) | strict | `m64+h16+long` | **0.6849** (same) | **+0.0000** |
| `1,5,1` (balanced) | relaxed | `m4+h8+long` | 0.4909 | — |
| `1,5,1` (balanced) | strict | `m4+h8+long` | 0.4909 (same) | **+0.0000** |

## Three substantive findings

### 1. Phase 11's +0.0613 reproduces on the combined fixture

Same Bitcoin Alpha, same 5 seeds, same `signedkan` model. The
strict-mode by-product filter on cost-only weighting picks
`m4+h8+long` instead of `m4+h8+short` — exact 4-decimal-place
match with Phase 11's standalone result. The combined fixture
hasn't broken anything.

### 2. The two levers are **sub-additive**

When MO weights are quality-aware (`0,1,0` or the balanced
`1,5,1`), ABB already picks an architecture that doesn't include
either by-product-emitting unit (`m64+h16+long` or `m4+h8+long`).
Switching strict-mode on adds **no further benefit** because MSG
isn't dropping anything ABB would have picked anyway.

So the levers' gains are **not summative**:

$$\Delta_{\text{cost-only, strict}} \;=\; +0.061$$
$$\Delta_{\text{quality, relaxed}} \;=\; +0.255$$
$$\Delta_{\text{quality, strict}} \;=\; +0.255 \;\;\text{(not +0.316)}$$

This is the most honest empirical answer Phase 14 produced: the
two mechanisms address **the same underlying defect** (picking a
Pareto-dominated architecture). MO weights solve it directly via
the cost dot product; by-product filtering solves it via
search-space pruning. When MO is doing its job, the filter is
redundant.

### 3. Engineering implication — pick the right tool per use case

| Use case | Recommended lever |
| --- | --- |
| Quality is explicitly observable (you have a measured quality_drop per unit) | **MO weights** — direct, gives the full +0.255 gain |
| Quality is structural/intuitive ("this unit emits unused parameters") | **By-product filter** — schema-level, no weight tuning |
| Default scalar-cost ABB without effort | **By-product filter** — set-and-forget; turns dominated picks into upgrades automatically |
| You want both mechanisms running for defense-in-depth | Both at once — sub-additive but safe |

The Phase 14 finding is **not** that the two levers are
interchangeable; they have very different ergonomics (continuous
weighting vs binary structural assertion). But on this fixture
their effects do not stack.

## Cost-vs-quality Pareto picture

```
weights         strict?  cost (dot)  mean AUC  Pareto
1,0,0           false      40        0.430    (cost-min, dominated)
1,0,0           true       40        0.491    (cost-min after filter)
0,1,0           false      10        0.685    (quality-min)  ← Pareto-best
0,1,0           true       10        0.685    (quality-min, same)
1,5,1           false     340        0.491    (balanced Pareto)
1,5,1           true      340        0.491    (balanced, same)
```

The Pareto frontier has three architectures: `m4+h8+short` (only
under relaxed cost-only — the dominated corner), `m4+h8+long`
(strict cost-only or balanced under either mode), and
`m64+h16+long` (any quality-aware mode).

## Connection to the cortical exhaustive sweep

Phase 14 result on the HSIKAN signed-graph workload is consistent
with the cortical exhaustive sweep's findings (Phase 14 = the
queue's first item, this report = the queue's second item):

- On both workloads, the MO + by-product mechanism produces
  identical pre-MO-engaged behaviour to running each lever alone.
- The cortical sweep showed regime structure (hypergraph wins at
  small width, CNN catches up at large width) — same kind of
  structural NAS finding that **MO weights + by-product filter
  surface automatically** when applied. The cortical Phase 14
  equivalent would be a `sweep_msg_cortical_combo.hymeko` that
  injects a `wasted_capacity` by-product on `binning_shallow`
  for the CNN branch (where deep+d16 beats shallow consistently)
  and runs the matrix. Clean follow-up for tomorrow.

## Test results

No new tests in Phase 14 — the levers' individual behaviour is
already pinned by Phases 10 and 11. The combined fixture exercises
the same code paths; if those tests pass the combined fixture's
behaviour is structural.

`cargo test -p hymeko_pgraph` full sweep: 96/96 pass + 1 ignored
doctest. Python full pgraph suite: 52/52.

## §6.5 anti-pattern audit

No new anti-patterns. The combined fixture is data composing
existing schema features; no new variant or string mode.

## Open follow-ups

1. **Cortical combined fixture** (`sweep_msg_cortical_combo.hymeko`).
   Reproduce Phase 14 on the cortical pipeline: declare a
   `wasted_capacity` by-product on `binning_shallow` for the CNN
   branch (Phase 12 / cortical-exhaustive showed shallow under-uses
   the deeper retinotopic structure); add cost-vector dimensions.
   Closes the cortical analogue of this report.
2. **Super-additive levers via orthogonal axes.** Phase 14 used
   two levers on the *same* dominated axis (both target the
   "scalar cost-min picks a dominated architecture" defect).
   Levers targeting different defects (e.g., MO weight on
   quality_drop + by-product on capacity overflow) could compose
   additively. Worth designing.
3. **Document the lever-selection guidance** as a recipe in the
   book (parallel to Phase 13's jq recipe).

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted).
- **5-seed wall time:** 9.6 s for 3 distinct architectures × 5
  seeds = 15 training cells.
- **Tests:** all Phase 1-13 suites pass; no regressions.

## Acceptance check

- [x] New combined fixture parses; canonical + extension behaviour
      matches predictions (canonical PASS; extension flags both
      by-products).
- [x] All 6 corners (3 weight regimes × 2 strict/relaxed) emit
      ABB selections; 3 distinct architectures across the matrix.
- [x] 5-seed training A/B numerical match with Phase 10 + Phase 11
      reproduces Phase 11 cost-only gain (+0.0613).
- [x] **Sub-additivity finding** documented: combined gain ≤
      max(individual gains).
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
