---
name: project-gomb-soma-rl-remap
description: 2026-07-02 Hajdu direction — remap the full Gömb-Soma cognitive stack into an RL backbone (Soma spatial-tree + Gömb shells + CPML grade-0 control readout); the Clifford-FIR + rotary-spike pipeline is step-1 (the Gömb membrane)
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-07-02, Hajdu: **remap Gömb-Soma into RL scenarios.** The architecture (per `architecture/cognitive_stack/`):
- **Gömb = 3 nested shells**: **Clifford-FIR** (outer membrane, signed-cycle FIR, `hymeko_graph::spine`, Cl(0,1)≅ℂ) ⊃ **HSiKAN** (middle: mixed-arity signed cycle pool + Catmull-Rom — *already our standalone RL backbone*) ⊃ **CPML** (inner core: grade-preserving polynomial layers + **grade-0 readout ⟨·⟩₀**), in `signedkan_wip/src/hymeko_gomb/`.
- **Soma-Chordex = reflex lane**: **quadtree** (the SPATIAL TREE, `soma/vision/quadtree_rust.py`) + Hodge ∂₂ + stim graphs.

**RL remap**: `scene → Soma quadtree spatial-tree → Clifford-FIR membrane → HSiKAN → CPML core → ⟨·⟩₀ grade-0 → action/value`. The **two missing pieces are exactly CPML + the spatial tree**: the quadtree is the spatial analogue of the kinematic hypergraph / [[project-structural-actor-walk-holonomy]] MultiTreeChannel (maps workspace occupancy / object+target positions / coin→zone field); CPML's grade-0 ⟨·⟩₀ IS the policy/value head. Same "one substrate, many targets" — Soma-vision (done, [[project-soma-vision-readout-bound]]) + control share the stack.

**Step 1 = the Gömb MEMBRANE**: "rotary spikes as mapping + Clifford-FIR" = **rotor** (connection, `hymeko_clifford` Cayley→quat SO(3)) × **spike** (timing, selects non-abelian traversal order — [[project-gauge-holonomy-signed-hsikan]]) → **Clifford-FIR** (signed-cycle filter). Plan (4 artifacts): `docs/plans/2026-07-02-clifford-fir-rotary-spikes/`. **Load-bearing seam**: Clifford-FIR is Cl(0,1)≅ℂ but the rotor is quaternion/SO(3) — the mapping→filter bridge π must be chosen (per-blade / linear read / lift-FIR); the validation TOY resolves it FIRST (ablations: −spike=order-blind, −rotor=abelian, −FIR=no cycles). **Validate the toy before building the whole stack.**

**Novelty (searched 2026-07-02, bounded)**: ingredients established — GCAN (Ruhe & Brandstetter ICML 2023), GA adaptive/FIR filters (2016), spiking phase-encoding (2024), and a close **2026 hybrid geometric-neuromorphic** (Clifford+spiking via "representation-conversion hyperedges") = the must-cite nearest neighbour. NOT found pre-empted: the specific **signed-hypergraph-cycle Clifford-FIR fed by spike-order-selected non-abelian holonomy** over the HyMeKo IR. Cite generics; claim only the cyclic-holonomy composition.
