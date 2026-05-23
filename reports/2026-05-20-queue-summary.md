# Overnight queue summary — 2026-05-20

Continuation of `reports/2026-05-20-overnight-summary.md`. The user
said "queue up" pointing at the four open items at the bottom of
that rollup; three are now done, plus one additional counter-finding
that emerged.

## What the queue produced

### 1. Exhaustive cortical sweep ([report](2026-05-20-cortical-exhaustive-sweep.md))

120 training cells (5 seeds × 12 architectures × 2 backbones) in
164.8 s wall.

**New finding: regime crossover at d=8.** Hypergraph wins at d=4
(+0.083 r² over CNN); CNN catches up at d=8; CNN edges by +0.010
at d=16+deep. Hypergraph σ uniformly 3–5× tighter than CNN's
worst-case σ.

**Sub-finding: the PLS axis is a structural no-op at current
fixture scale** — pls_25 and pls_50 give byte-identical mean r²
because `min(n_pls, d_feat-1, n_train-1)` saturates at 15 in both
cases. Actionable: either drop the PLS axis or rescale `n_images`
so pls_50 reaches a higher rank.

### 2. Phase 14 — HSIKAN lever interaction ([report](2026-05-20-phase14-lever-interaction.md))

Built `data/hsikan/sweep_msg_combo.hymeko` stacking Phase 10
multi-cost dimensions and Phase 11 by-product injection on the
same 8-unit HSIKAN schema. 6-corner matrix (3 weight regimes ×
2 strict/relaxed modes) + 5-seed training A/B:

| weights | relaxed | strict | filter gain |
|---|---|---|---|
| cost-only | 0.430 | 0.491 | **+0.061** |
| quality-only | 0.685 | 0.685 | 0.000 |
| balanced (1,5,1) | 0.491 | 0.491 | 0.000 |

**Sub-additive composition.** By-product filter is no-op when MO
weights already pick a non-by-product architecture. MO quality
weighting (+0.255) dominates the filter (+0.061). The two
mechanisms address the same defect (Pareto-dominated picks); MO
solves it via cost dot product, filter via search-space pruning.

### 3. Phase 14 cortical — counter-finding ([report](2026-05-20-phase14-cortical-counter-finding.md))

Built `data/hsikan/sweep_msg_cortical_combo.hymeko` injecting a
by-product on `binning_shallow`. Same 6-corner matrix:

| weights | relaxed | strict | filter gain |
|---|---|---|---|
| cost-only | 0.284 | **0.235** | **−0.050 (REGRESSION)** |
| quality-only | 0.397 | 0.397 | 0.000 |
| balanced | 0.397 | 0.397 | 0.000 |

**Counter-finding: the by-product filter is adversarial on
cortical at cost-only weight.** The exhaustive-sweep data showed
shallow under-uses retinotopic structure at d≥8 but is fine at
d=4. The schema author (me) injected the by-product on a unit
that's only sometimes dominated; the filter regresses the
cost-only pick.

**Refined lever-selection rule (from the cortical counter-finding):**

> By-product injection only works when the targeted unit is
> dominated at every orthogonal-axis point. MO weights are
> failure-tolerant; by-product injection is a sharp knife.

When both are available, prefer MO weights — graceful failure mode
(smaller gain) rather than adversarial (regression).

## What remains in the queue

| item | status | next-session estimate |
| --- | --- | --- |
| RicciStimBackbone per-anchor Python-loop optimisation (Slice 2) | deferred — touches hot inference path, deserves a focused session | 2-4 h |
| Refined cortical combined fixture (no by-product, MO only) | trivial follow-on from counter-finding | 30 min |
| Lever-selection book recipe (parallel to Phase 13's `jq` recipe) | documentation | 30 min |
| Bidirectional check on HSIKAN dominance claims | runtime audit | 1 h |

The Slice 2 backbone optimisation deserves its own plan and a
careful before/after benchmark; it's the highest-impact remaining
item but the riskiest. Defer to a focused session, not overnight.

## Aggregate test state

| Suite | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` | 96 / 96 + 1 ignored doctest |
| `test_cortical_benchmark.py` (Slice 1) | 21 / 21 |
| `test_cortical_pgraph_mapping.py` | 5 / 5 |
| `test_byproduct_filter_e2e.py` | 5 / 5 |
| `test_pgraph_multiobjective_pipeline.py` | 7 / 7 |
| `test_hsikan_pgraph_mapping.py` | 7 / 7 |
| `test_hyperedges_m_per_vertex.py` | 7 / 7 |

Zero regressions across the queue work.

## Files produced overnight (rollup + queue together)

New fixtures (5):
- `data/hsikan/sweep_msg_byproduct_dominated.hymeko`
- `data/hsikan/sweep_msg_cortical.hymeko`
- `data/hsikan/sweep_msg_cortical_gomb.hymeko`
- `data/hsikan/sweep_msg_combo.hymeko`
- `data/hsikan/sweep_msg_cortical_combo.hymeko`

New Python modules (3):
- `signedkan_wip/src/cortical_pgraph_mapping.py`
- `signedkan_wip/experiments/runs/run_cortical_msg_sweep.py`
- `signedkan_wip/experiments/runs/run_cortical_exhaustive_sweep.py`

New tests (2 Rust + 2 Python):
- `hymeko_pgraph/tests/byproduct_filter_phase11.rs`
- `signedkan_wip/tests/test_byproduct_filter_e2e.py`
- `signedkan_wip/tests/test_cortical_pgraph_mapping.py`

New book recipe:
- `docs/book/src/recipes/filter-by-friedler-certificate.md`

Reports (7):
- `reports/2026-05-20-pgraph-nas-byproduct-filter-phase11.md`
- `reports/2026-05-20-pgraph-cortical-sweep-phase12.md`
- `reports/2026-05-20-pgraph-cortical-gomb-cv-phase12_5.md`
- `reports/2026-05-20-overnight-summary.md`
- `reports/2026-05-20-cortical-exhaustive-sweep.md`
- `reports/2026-05-20-phase14-lever-interaction.md`
- `reports/2026-05-20-phase14-cortical-counter-finding.md`
- `reports/2026-05-20-queue-summary.md` (this file)

Plus the JSONL+summary from the cortical exhaustive sweep.

## Most important findings to read first (in order)

1. **Phase 12.5 + cortical exhaustive — hypergraph CV result.** The
   hypergraph backbone wins cost-min by +0.083 r² over the CNN
   with 3-5× tighter variance at ¼ the parameters. Cleanest
   empirical hypergraph-machine CV demonstration.

2. **Phase 11 — +0.061 AUC from strict-mode by-product filter
   alone, predicted to 4 decimal places.** Strongest falsifiability
   check in the audit.

3. **Phase 14 lever-interaction + cortical counter-finding** —
   MO and by-product injection compose sub-additively when the
   injection is correct; *adversarially* when it's not. Both
   findings together refine the engineering guidance from Phase 10/11.

Good morning.
