---
name: project-structural-actor-walk-holonomy
description: "StructuralActor (Hajdu idea, 2026-06-28): replace HSiKAN's iterative message-passing with a static gather along a design's precomputed walks (= the Z2 holonomy, = one matmul B^L). HSiKAN-class accuracy at ~10x speed / 30x fewer params; Steiner S(2,3,9)=AG(2,3) is the BEST structural basis. Closes the loop (controls a linear plant within 6-9% of LQR). Deep equivalences: aromaticity, E/I balance, GRID CELLS=path-integration-holonomy."
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

**SHORTHAND (Hajdu, 2026-06-28): "SA-HSiKAN" = Structural-Actor HSiKAN.** Prose label "SA-HSiKAN" (expand on
first mention); code/config key `sa_hsikan` (e.g. in `exp_pernode_actor_ab.py` `_CONFIGS`). Keeps the family name
intact — HSiKAN stays HSiKAN ([[project-unify-hsikan-core]] never-rename rule); SA- only marks the frozen-structure
holonomy-gather variant.

**THE ARCHITECTURE (`hymeko_rl/structural_actor.py`, design `docs/theory/structural_actor_design.md`,
report+PDF `reports/2026-06-28-structural-actor-steiner.{md,tex,pdf}`).** Hajdu's idea: HSiKAN message-passing is
launch-bound (~2.8ms, spline-per-edge × layers). Over a FIXED topology the structural features are the signed
walks/cycles — enumerate ONCE, then forward = gather features along walks weighted by the SIGN-PRODUCT (the Z2
holonomy/parallel-transport), scatter to start node, readout. NO message-passing. Algebraically it collapses to
ONE precomputed operator **B^L** (signed L-hop adjacency power) — the gather+sign+scatter IS a matmul (oracle test:
bl == B^2 exactly). KEY FIX = holonomy = sign-PRODUCT not sign-SUM (v1 used sum → MSE 0.83; product → 0.0088, ~95×).
Dual of the existing `structural_critic.py` (which gathers BACKBONE features for value; this gathers RAW features
for action, no backbone). Pattern proven by `spike_probe.py` (walk-gather→output generalizes).

**MEASURED (N=9 supervised structural target):** StructuralActor MSE ~HSiKAN, BEATS MLP, at ~10× lower latency
(232-282µs vs 2600-3100) + 30× fewer params (73 vs 2209). B^L collapse 318→~230µs; torch.compile unavailable (no
MSVC cl.exe on this box) so ~230µs is the eager floor (~5 launch-bound ops; hand-fuse didn't help — einsum
decomposes same). **STEINER S(2,3,9)=AG(2,3) IS THE BEST GENERATOR** (actor MSE 0.0008, nearly matching HSiKAN
0.0004, order tighter than k-uniform) — balance ("every pair once") = non-redundant walk basis. Works over ALL
generators (chain=k2/k-unif/Steiner/sunflower) once `keep` covers the walks (keep=64 truncated the denser
star-expanded designs → must scale keep; production = scored top-K). Figures `reports/figures/design_hypergraphs.*`
+ `generator_accuracy.*` (render_designs.py, networkx; graphviz/dot module not installed).

**CLOSED-LOOP DONE (`structural_control_loop.py`):** StructuralActor CONTROLS a networked linear plant (differentiable
rollout, min LQR cost) — ρ=J/J*: chain 1.093, Steiner 1.063 (Steiner controls better, consistent). HONEST: MLP gets
~1.00 (LQR-optimal) on these BENIGN plants → structure not load-bearing for easy control (the session refrain;
structure wins for REPRESENTATION + HARD plants). Real test = the MuJoCo ladder (velocity→2link→SCARA→6dof→quadruped
as .hymeko descriptors), machine-bound, DEFERRED.

**THEORY: it's a principal bundle** — flat bounded finite-affine base AG(2,3) + a connection whose HOLONOMY is the
nonlinearity. Frame theory: Steiner→equiangular tight frames (Fickus-Mixon-Tremain) likely explains "Steiner best"
(testable: does frame coherence rank generators?). A nonlinear basis ON the blocks (block-wise Steiner-KAN) curves
the effective geometry = recovers "nonlinear geometry" without a separate connection. Ties
[[project-gauge-holonomy-signed-hsikan]] [[project-hymeko-as-control-substrate]].

**CHEM/BIO/NEURO EQUIVALENCES (`docs/theory/chem_bio_neuro_equivalences.md`, user asked overnight 2026-06-28).**
DEEP ones (testable, not analogy): (1) **aromaticity = cycle holonomy** — signed graph IS the Hückel Hamiltonian;
Hückel(4n+2)/Möbius(4n) = trivial/flipped ring holonomy (our balance = aromaticity, same theorem). (2) **E/I balance
= signed-graph balance** — excit/inhib = ±, loop sign-product = positive(runaway/seizure)/negative(homeostasis)
feedback. (3) **CROWN JEWEL: grid cells = path integration = holonomy over a hexagonal/affine lattice** — entorhinal
grid cells path-integrate (velocity→position = parallel transport = holonomy) on a triangular lattice, pop activity
on a TORUS (Gardner 2022). The StructuralActor IS a normative model of grid-cell computation. (4) ring/torus
attractors = rotors (head-direction = S¹ rotor); spike-rotor = phase coding (theta-gamma, phase precession). THE
BET: a StructuralActor on a hex lattice trained to path-integrate should develop grid-like/toroidal codes (cf
Cueva-Wei, Banino) → "control in a bounded affine geometry via holonomy" = the grid-cell computation in control
terms. Loose ones (morphogenesis, generic reaction nets) noted, not leaned on.
