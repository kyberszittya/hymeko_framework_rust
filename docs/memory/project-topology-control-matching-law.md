---
name: project-topology-control-matching-law
description: "Isomorphic-controllers Phase 4 — the topology→control law is MATCHING (each plant best controlled by its OWN topology, 9/9), NOT a universal tight-frame; coherence does NOT rank control. \"Steiner is best\" was plant-specific"
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-28: extended the topology zoo (Kato's isomorphic-controllers program) with extremal generated
architectures and ran the topology→control map. Built + tested (`hymeko_rl/topology_zoo.py`,
`topology_invariants.py`, tests): **petersen** (SRG(10,3,0,1)), **kneser** K(n,2), **grotzsch** (=Mycielskian of
C5, triangle-free), **expander** (random d-regular), via the existing `_signed_graph` edge-list helper.

**WALK-STRUCTURE FRAMING (Hajdu intuition, holds):** SA-HSiKAN gathers along B^L = signed walk-holonomy, so the
topology GENERATES the walks (the structural prior). The graph invariants are walk-structure measures: frame
coherence = walk overlap, spectral gap = mixing, Z2 frustration = holonomy, girth = return. **Structural fact:**
Petersen = Steiner S(2,3,9) = Fano S(2,3,7) all reach frame coherence = **1/3** SIGN-INDEPENDENTLY (μ=1: every pair
shares exactly one common neighbour/block → equiangular tight frame). Hubs (star/sunflower) = 1.0 (degenerate).

**THE PREDICTION WAS REFUTED (the discriminating test worked).** Tempting guess: tight-frame walks (coherence 1/3)
control best. Matched-N=10 `controller_bench` sweep (learned HSiKAN controller fits each plant's structural target):
**coherence does NOT predict control** — Petersen (coh 1/3) avg-MSE 0.54 = near-WORST (7th/9); chain (coh 0.71) =
BEST (0.39); no monotone relation. **The real, stronger law = MATCHING: best_controller[plant] == plant for all
9/9** — every plant is best controlled by its OWN topology (isomorphic controller wins). So structure is
load-bearing (the isomorphic-controllers thesis holds) but there is NO universal best topology; **the earlier
"Steiner AG(2,3) is the best basis" ([[project-structural-actor-walk-holonomy]]) was PLANT-SPECIFIC, not a
tight-frame law.** Caveat: supervised structural-fit (Phase 1, 2 seeds); closed-loop confirmation is the follow-up.

**ARITY STUDY (2026-06-29, both claims REFUTED).** Added a hypergraph lift of the graph families:
`closed_neighbourhood_blocks` + `graph_to_kuniform` (each vertex's {v}∪N(v) → a star-expanded hyperedge;
k=d+1-uniform for d-regular) in `hypergraph_designs.py`; `HYPER_TOPOLOGIES` (petersen_h/ring_h/expander_h/
complete_h) in `topology_zoo.py`; 4 tests. The lift raises frame coherence (petersen 0.33→0.50, expander 0.67→1.0
— hub-mediated walks add overlap). User claim "hubs make it MORE computationally + accurately performant" tested
+ FALSE on both: (1) COMPUTE — lift doubles N, dense B^L is 4× costlier; factored sparse (torch.sparse.mm) is
SLOWEST (~10³ const overhead swamps O(Nk); crossover only ~N≈7000, irrelevant at control N~10-300). (2) ACCURACY —
hyper controller 18-38× WORSE than graph on a graph plant (petersen 0.064→1.17), below the chain baseline, at
matched AND reduced params. WHY (user pushed "20× doesn't make sense" — he was right; DIAGNOSED not asserted): NOT scale (pre-acts 5.5 vs
6.3 comparable), NOT unnormalized-agg (degree-norm left MSE bit-identical 1.1717), NOT pool-over-hubs (points-only
readout no better, 1.27) — all eliminated. CAUSE = FUNCTION-CLASS MISMATCH: on a graph plant the hyper controller
can't even fit TRAIN (0.30 vs graph 0.0004), the bipartite hub-mediated walk basis doesn't contain the direct-graph
target; anti-generalizes (test 1.17>1.0). PROOF it's matching not "hubs bad" = REVERSE THE PLANT (definitive_arity.py):
2×2 test-MSE matrix graph/hyper plant × graph/hyper controller = [[0.064, 1.17],[0.876, 0.0006]] — DIAGONAL wins
BOTH; on the HYPERGRAPH plant the hyper controller is near-EXACT (0.0006) and the graph controller fails (0.876,
memorizes train blind to hubs). **MATCHING LAW EXTENDS TO ARITY**: hypergraph task→hypergraph controller, graph
task→graph controller. (NB: I first mis-diagnosed on the SA-HSiKAN B^L operator — the test used the message-passing
"hsikan" backbone; per contract, diagnose the ACTUAL failing config. Caught + redone.) Fig arity_matching.png. Report
`reports/2026-06-29-arity-study.pdf`; figs `compute_arity.png`/`accuracy_arity.png`; data `arity_control.json`.
Plan `docs/plans/2026-06-28-extremal-topology-zoo/`. Report (for Kati, corrected) `reports/
2026-06-28-walk-geometry-kati.pdf` + figs `reports/figures/walk_geometry/`. Data `reports/overnight/
topo_control_map.json`. Ties [[project-hymeko-as-control-substrate]] (Kato), [[feedback-user-intuition-is-calibrated]]
(the higher-level intuition held; the specific universal-best guess didn't — matching did).
