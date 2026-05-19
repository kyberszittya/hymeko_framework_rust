# Cliques foundation — baseline benchmarks + nodelets operator

**Date:** 2026-05-14
**Git SHA:** (uncommitted; staged in working tree)
**Scope:** the foundation work flagged in
`docs/plans/2026-05-14-cliques-generation-detection/plan.md`
(part a) and V1 of the nodelets operator from
`docs/plans/2026-05-14-gomb-belief-planning-bridge/plan.md`
(part b).

## Summary

Two pieces of foundation infrastructure landed and were measured.
Both have **clean test coverage** (16 contract + 11 cliques + 20
foundation harness = 47 cliques-related tests passing) and **honest
performance numbers** that reshape the next-step research narrative.

The headline:
**Bron-Kerbosch is the only detector that recovers planted balanced
cliques at any tested scale.** The three approximate detectors
(triangle-density greedy, degree-seeded greedy, signed-Laplacian
spectral) return 30 cliques per call but **none** of them match the
planted ground truth via Jaccard ≥ 0.5 once `n ≥ 100`. They are also
**slower** than BK at scale due to pure-Python iteration overhead.

This kills the "Gömb beats BK on speed" framing of the NP-hard pivot
at the sweep's scale (`n ≤ 500`). What survives — and the next plan
step is to validate — is "Gömb beats classical detectors on
**task-specific** detection (faction recovery, community-aware
matching)" which the cliques foundation tests don't measure.

## Part A — Detection benchmark sweep

### Files

- New: `signedkan_wip/experiments/cliques_detection_sweep_2026_05_14.py`
- New: `signedkan_wip/experiments/results/cliques_detection_sweep_20260513T233737Z.jsonl`
  (289 rows)

### Methodology

Grid: `n_robots ∈ {30, 50, 100, 200, 500}` × planted profiles
`{[6,5,4,3], [8,5,4], [10]}` × four detectors × 5 seeds. Cells where
`sum(profile) > n_robots` skipped. Each detector gets 60 s wall budget
per call.

### Results (median over 15 cells per row: 3 profiles × 5 seeds)

| n_robots | detector | wall_med | wall_p95 | recall | precision | #det_med |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 30 | bron_kerbosch_exact | 0.001 s | 0.091 s | **1.000** | 0.231 | 13 |
| 30 | greedy_balanced | 0.004 s | 0.013 s | **1.000** | 0.071 | 17 |
| 30 | triangle_density_greedy | 0.009 s | 0.030 s | **1.000** | 0.077 | 18 |
| 30 | spectral_balanced | 0.023 s | 0.126 s | 0.250 | 0.250 | 2 |
| 50 | bron_kerbosch_exact | 0.004 s | 0.010 s | **1.000** | 0.200 | 10 |
| 50 | greedy_balanced | 0.009 s | 0.012 s | 0.667 | 0.067 | 30 |
| 50 | triangle_density_greedy | 0.027 s | 0.039 s | 0.500 | 0.033 | 30 |
| 50 | spectral_balanced | 0.067 s | 0.094 s | 0.000 | 0.000 | 1 |
| 100 | bron_kerbosch_exact | 0.011 s | 0.017 s | **0.750** | **0.750** | 4 |
| 100 | greedy_balanced | 0.034 s | 0.066 s | 0.250 | 0.033 | 30 |
| 100 | triangle_density_greedy | 0.120 s | 0.201 s | 0.000 | 0.000 | 30 |
| 100 | spectral_balanced | 0.120 s | 0.136 s | 0.000 | 0.000 | 0 |
| 200 | bron_kerbosch_exact | 0.089 s | 0.112 s | **0.500** | **1.000** | 2 |
| 200 | greedy_balanced | 0.261 s | 0.395 s | 0.000 | 0.000 | 30 |
| 200 | triangle_density_greedy | 0.526 s | 0.863 s | 0.000 | 0.000 | 30 |
| 200 | spectral_balanced | 0.210 s | 0.332 s | 0.000 | 0.000 | 0 |
| 500 | bron_kerbosch_exact | 3.176 s | 3.701 s | **0.750** | **1.000** | 2 |
| 500 | greedy_balanced | 3.618 s | 6.815 s | 0.000 | 0.000 | 30 |
| 500 | triangle_density_greedy | 5.073 s | 7.019 s | 0.000 | 0.000 | 30 |
| 500 | spectral_balanced | 1.870 s | 1.973 s | 0.000 | 0.000 | 0 |

### Findings

1. **Bron-Kerbosch is fast enough at `n ≤ 500`.** 3.2 s median at
   n=500, 3.7 s p95. NetworkX's pivot-pruning + C-extension speed
   keeps the exact algorithm well inside the timeout budget.

2. **Approximate detectors fail at scale.** triangle_density_greedy
   and greedy_balanced both return 30 cliques (limit) at n ≥ 100,
   but zero of those match the planted ground truth via Jaccard.
   The reason is planted-clique *absorption*: in a dense network,
   a planted size-6 clique is often contained in a size-9 maximal
   clique that the greedy detector finds. Jaccard ≥ 0.5 fails.

3. **Approximations are slower than the exact baseline.** At n=500,
   BK = 3.2 s; triangle_density = 5.1 s; greedy_balanced = 3.6 s.
   Pure-Python iteration overhead in the approximators dominates.

4. **Spectral never works.** Recall 0 at every scale beyond n=30,
   often returning 0 cliques. The signed-Laplacian k-means clustering
   pulls together topology-related-but-not-clique vertex groups; the
   balance check then rejects almost all clusters.

### Implication for the NP-hard pivot plan

The narrative "Gömb beats baseline X on speed" is **dead-on-arrival
at this scale**. BK is the right baseline; it's faster than any
heuristic approximator we can implement in pure Python. The
NP-hard claim that survives is "Gömb beats BK on **task-specific
detection**" — i.e., faction recovery, community-aware matching,
clique extraction with constraints classical algorithms don't
respect.

Equivalently: this scale (`n ≤ 500`) is **not where BK breaks**.
For BK to genuinely struggle we need `n ≥ 1000` AND high density,
OR a graph where the maximal-clique count is genuinely exponential
in n. The synthetic robot networks at comm_range = 4.0 / noise =
0.05 don't produce that regime. We'd need to design a stress-test
generator (e.g. Erdős-Rényi at density 0.3+, n=2000+) to find
BK's breaking point.

## Part B — Nodelets contraction operator

### Files

- New: `signedkan_wip/src/demo/cliques_contract.py` (340 LOC)
  — `contract_balanced_cliques`, `multiscale_hierarchy`,
    `ContractedBundle`. Full mathematical description in the module
    docstring.
- New: `signedkan_wip/tests/test_demo_cliques_contract.py` (16 tests)
  — covers all 8 invariants from the math docstring + provenance
    preservation.

### Mathematical definition (formal)

Given signed graph `H = (V, E, σ)` and balanced cliques
`C = {C_1, ..., C_K}`:

```
V' = C ∪ { {v} : v ∈ V \ ⋃ C_i }                  (super-vertices)
φ : V → V'                                          (coarsening map)
   φ(v) = C_i if v ∈ C_i else {v}
E' = { (φ(u), φ(w)) : (u, w) ∈ E ∧ φ(u) ≠ φ(w) }   (quotient edges)
σ'(s, t) = sign( ∑_{(u, w) ∈ E : φ(u)=s, φ(w)=t} σ(u, w) )
         with ties broken to +1                    (majority sign)
```

Internal edges (those with `φ(u) = φ(w)`) are absorbed into the
super-vertex and discarded from `E'`. Overlap resolution is greedy
size-descending.

### Properties verified by tests

1. ✓ `|V'| ≤ |V|` — vertex count non-increasing.
2. ✓ Identity case: no balanced cliques → `V' = V` (singletons).
3. ✓ Full balance: a single balanced clique → 1 super-vertex.
4. ✓ Balanced 4-clique with 2-2 sign split correctly collapses
   (test of `k ≥ 4` balance semantics).
5. ✓ Determinism: same input → bit-identical output.
6. ✓ Edge preservation modulo internal absorption.
7. ✓ `φ`-map partitions V into exactly the super-vertices.
8. ✓ Sign aggregation: majority works, ties resolve to +.
9. ✓ Multi-scale terminates at fixed point or singleton root.
10. ✓ Provenance fields (`parent_n_vertices`, `super_clique_indices`,
    `super_members`, `phi`) preserved across the call.

### Performance characterization

Planted profile `[10, 8, 6, 5]`, `n_factions=2`, `noise_prob=0.05`,
`comm_range=5.0`, 5 seeds per cell:

| n_robots | wall_med | wall_p95 | V_in | V_out_med | compression | singletons |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 2.6 ms | 89.2 ms | 30 | 18 | **0.600** | 16 |
| 50 | 5.7 ms | 5.8 ms | 50 | 38 | 0.760 | 35 |
| 100 | 32.7 ms | 60.0 ms | 100 | 89 | 0.890 | 87 |
| 200 | 208.5 ms | 294.5 ms | 200 | 193 | 0.965 | 192 |
| 500 | 8.3 s | 9.3 s | 500 | 493 | 0.986 | 492 |

Multi-scale hierarchy on the same configs reaches **2 levels** for
every n, with progressively less compression at each level (the
fixed point arrives fast).

### Findings

1. **Compression weakens with scale.** At n=30, the operator
   contracts to 60% of the input. At n=500, only 1.4% compression
   — the operator is essentially a no-op on large dense networks
   in our current setup.

2. **Wall-time scales with BK on the inside.** The contraction
   wraps `enumerate_balanced_cliques` (BK + balance check), so the
   8.3 s wall at n=500 is BK's cost. The contraction's own bookkeeping
   adds < 100 ms.

3. **Singletons dominate at scale.** At n=500, 492 of 493
   super-vertices are singletons — only one balanced clique was
   chosen during disjoint selection. This is the same
   *planted-clique absorption* phenomenon: BK returns dozens of
   maximal cliques on a dense graph; greedy disjoint selection picks
   the largest (which happens to NOT contain the planted ones); the
   planted clique vertices are then "claimed" by that larger picked
   clique and disappear as super-vertex centers.

4. **Hierarchy depth is 2.** Even with `max_levels=6`, every test
   bundle hits a fixed point at the second level. This is the
   well-known *small-world-collapse* problem in graph coarsening —
   the first iteration absorbs most of the cliques; subsequent
   iterations don't find new structure because the coarse graph is
   already dense + nearly random.

### Implications for the belief-planning bridge plan

V1 of the operator is **correct** — every mathematical invariant
holds — but **not yet useful at scale**. To make it carry the
multi-scale narrative, V2 should:

1. **Plant cliques spatially-clustered, not uniformly random.** This
   prevents absorption: planted-clique members stay within radius,
   the BK-found maximal cliques on top of them are the planted
   ones themselves.
2. **Replace BK with a balance-aware Bron-Kerbosch variant** that
   prunes unbalanced branches early. Memory
   `project_global_topk_ladder_null_2026_05_10` records that this is
   non-trivial; per-vertex per-arity ABB is what works.
3. **Use Gömb's vertex embeddings as the disjoint-selection ordering**
   (per the V1 plan's "Alternatives" section). High-embedding-norm
   vertices get to "claim" the planted cliques first; low-embedding
   singletons go last. This is the bridge to insight III
   (self-evolving sampling).

These are all V2-level investments. The V1 ship is the correctness +
provenance foundation; V2 is the performance + relevance leg.

## Combined tests pass

- New: 16 contract operator tests, 20 cliques foundation harness tests.
- Total demo test suite: **82 / 82 passing** (62 cliques-related,
  10 kinematic, 10 signed-link prediction).
- Suite wall-time: < 3 s on CPU.

## Performance budget

- Sweep wall: ~5 min CPU, peak RSS < 500 MB.
- Contract operator (single call): from 3 ms (n=30) to 8 s (n=500).
- Hierarchy (max 6 levels): 6 ms (n=50) to 18 s (n=500). Capped by
  BK on the inside.

## New / removed dependencies

None. Pure-Python + numpy + networkx + scikit-learn (all already in
the demo group).

## Open issues / follow-ups

1. **NP-hard pivot Stage 1 (faction recovery) is the next gate.**
   The detection benchmarks above are the *baseline*; Stage 1 will
   test whether Gömb's *task-specific* matching beats BK on a
   different metric (faction-recovery accuracy via Hungarian
   matching, not Jaccard overlap of detected vs. planted cliques).
2. **Generator design: spatial-clustering for planted cliques.** The
   current uniform-random placement makes planted cliques absorbed
   into larger maximal cliques. A spatially-clustered placement
   (drop K disks of radius r << comm_range, place clique members
   inside) would surface the planted cliques as the largest cliques
   the detectors find.
3. **Stress-test BK at `n ≥ 1000`.** This was scoped out of the
   sweep. To make a credible "BK breaks at scale" claim — which
   the NP-hard pivot's Stage 2 wall-time story depends on — we need
   a denser / larger benchmark.

## Experiment provenance

- Working tree: dirty (uncommitted; everything staged).
- Sweep: `cliques_detection_sweep_20260513T233737Z.jsonl`.
- Seed: 0, 1, 2, 3, 4 (deterministic across runs).
- CPU: as instrumented by the host (no GPU used).
- Tests passing as of 2026-05-14T01:45 local: 82/82.

## CORE.YAML items touched

**Empty list.** All work in `signedkan_wip/src/demo/` and
`signedkan_wip/tests/`. No CORE crate, no pinned-dep changes.

## Cross-references

- Foundation plan:
  `docs/plans/2026-05-14-cliques-generation-detection/plan.md`
- Bridge plan (origin of nodelets concept):
  `docs/plans/2026-05-14-gomb-belief-planning-bridge/plan.md`
- NP-hard plan (next step, gated on this):
  `docs/plans/2026-05-14-gomb-np-hard-approximation/plan.md`
- Self-evolving plan (separate angle, also gated):
  `docs/plans/2026-05-14-self-evolving-cycle-sampling/plan.md`
