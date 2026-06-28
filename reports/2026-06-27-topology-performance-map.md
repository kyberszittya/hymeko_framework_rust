# Phase 1 — the topology→performance map (Kato's isomorphic-controllers program)

**When:** 2026-06-27 17:30 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu & Prof. Kato
**Plan:** `docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/` · **Status:** Phase 1 done.

## Summary

Phase 1 of Kato's program: *generate typical hypergraph topologies → instantiate each as an isomorphic
controller → benchmark which topology controls which plant best.* Eight topology families (chain, ring, star,
balanced tree, 2-D grid, small-world, random `G(n,p)`, complete) over matched `N=9`. Each is instantiated as a
**learned controller** — a HSiKAN (`SignedKANBackbone`) over its signed adjacency — and trained to predict each
**plant's** structural-regression target (reusing the `structural_probe` harness). The held-out median MSE
(3 seeds) fills a plant×controller matrix.

## Result (measured)

**The matching controller wins 8/8.** Every plant's target is best predicted by the controller whose
interconnection topology *mirrors the plant* — diagonal MSE 0.001–0.028 vs off-diagonal up to 1.6.

| plant\controller | chain | ring | star | tree | grid | small_w | random | complete |
|---|---|---|---|---|---|---|---|---|
| chain    | **0.004** | 0.086 | 0.241 | 0.127 | 0.110 | 0.084 | 0.070 | 0.193 |
| ring     | 0.111 | **0.009** | 0.337 | 0.230 | 0.211 | 0.112 | 0.210 | 0.244 |
| star     | 0.421 | 0.760 | **0.001** | 0.150 | 1.114 | 0.452 | 0.850 | 0.358 |
| tree     | 0.575 | 1.170 | 0.921 | **0.018** | 1.147 | 1.055 | 0.995 | 1.322 |
| grid     | 0.402 | 0.620 | 1.225 | 0.741 | **0.028** | 0.457 | 0.335 | 1.167 |
| small_w  | 0.456 | 0.386 | 0.646 | 0.460 | 0.545 | **0.013** | 0.616 | 0.543 |
| random   | 1.185 | 0.578 | 1.624 | 1.240 | 0.927 | 0.692 | **0.026** | 1.523 |
| complete | 0.293 | 0.440 | 0.912 | 0.521 | 1.073 | 0.275 | 0.719 | **0.018** |

Edge counts: chain 8, ring 9, star 8, tree 8, grid 12, small_world 12, random 13, complete 36.
Figure: `reports/topology_map/topology_map.png` (heat-map, ★ = best controller per plant).

### The interesting nuance — it's *structural match*, not raw capacity

The **complete** controller (36 edges, by far the densest, most expressive) is **not** a universal winner — it is
best only on the *complete* plant, and is a poor controller for star (0.912) and grid (1.073). More edges ≠ better
control. The signal is **topological match**: a controller that shares the plant's coupling structure represents
its dynamics with near-zero error; a structurally mismatched controller — even a much denser one — does worse.
This is the strong form of Kato's hypothesis (topology, not capacity, is the lever).

## Well-definedness guard (the discriminating test, per the operating contract)

The first guard I wrote was a *statistical* invariance check; it read a 50% relative gap, which looked alarming.
It was **confounded** — it divided two near-zero MSEs and did not permute the input with the graph. I replaced it
with the **exact equivariance check**: with *identical weights*, the controller's pooled output on `(H, x)` must
equal its output on the isomorphic `(π(H), π(x))`. Measured residual: **1e-8** (machine epsilon) across all
topologies → the controller is **exactly permutation-equivariant**. So an isomorphic relabelling is a genuine
no-op, and the map's off-diagonal structure is *real*, not a labelling artefact. (Measured: residual 1e-8.
Inferred: the 50% statistical gap was the near-zero-denominator artefact, now discarded.)

## Honest scope

- This is the **cheap supervised structural proxy** of the plan (rank topologies by how well they represent a
  graph-structured target) — **not** closed-loop control yet. The plan gates the control-task follow-up on this
  rank correlating with closed-loop performance; that correlation is **not yet measured** (next step).
- The plant target is the same `Σ tanh(B²x)` structural family the `structural_probe` uses; a different target
  family could shift the off-diagonal gradient. Phase 3 widens targets and plants.
- Learned controllers only. The control-theory leg (structured `u=-Kx`, gain-sparsity = `H`) is Phase 2.

## Files touched

- `hymeko_rl/topology_zoo.py` — **new**, 119 LOC: 8 topology generators + `permuted()`.
- `hymeko_rl/controller_bench.py` — **new**, ~150 LOC: `run_topology_map`, `equivariance_check`,
  `plot_topology_map`, CLI.
- `hymeko_rl/tests/test_topology_zoo.py` — **new**, 16 tests (topology validity, family shapes, grid guard,
  permutation, diagonal-wins, exact equivariance ×4).
- `reports/topology_map/{topology_map.json, topology_map.png}` — **new** artifacts.

## Tests & provenance

- **16/16 pass** (`pytest -p no:randomly`), ruff clean. Equivariance residual 1e-8 (exact).
- CORE.YAML touched: none. New deps: none (generators hand-rolled).
- Seeds: graph_seed 0; controller init seeds 0–2; data seeds 1000–1002. N=9, hidden 24, 2 layers, 250 epochs.
- §6.5: no anti-patterns introduced (Strategy registry `TOPOLOGIES`; reused `structural_probe` harness +
  `SignedKANBackbone` rather than re-implementing; `_iqr` reused from `reach_arch_compare`).

## Next

1. **Control-task correlation** (the plan's gate): does the supervised rank track closed-loop control on a real
   plant (cart-pole/arm)? — machine-bound, after the coin-toss baseline frees the CPU.
2. **Phase 2:** the structured `u=-Kx` controller leg (the control-theory contribution).
3. **Topology-property correlate:** does algebraic connectivity / diameter / signed balance predict the
   off-diagonal cost? — turns the map into a *law*.
