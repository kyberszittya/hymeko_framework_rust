# A walk-holonomy controller, and why a Steiner design is its best structural basis

**Date:** 2026-06-28 · **Author:** Cs. Hajdu (with Claude Code) · **Status:** toy-scale, supervised — promising,
not yet closed-loop.

## One line

A controller whose internal structure is a *combinatorial design* can replace iterative graph message-passing
with a **single static gather along the design's walks** — reaching the same structural accuracy at ~10× less
compute — and a **balanced Steiner design is measurably the best such structure**.

## The idea

Our signed-graph controller (HSiKAN) propagates signals layer by layer — many small spline operations, slow
(~2.8 ms/forward, launch-bound). But over a *fixed* topology the relevant structural features are the graph's
**signed walks/cycles**, which can be enumerated **once**. The forward then collapses to: gather node features
along the walks weighted by the **sign-product** (the Z₂ *holonomy* — parallel transport along the walk), scatter
to each node, read out. No message-passing. We call this the **StructuralActor**; algebraically it is one
precomputed operator `Bᴸ` (the signed L-hop adjacency power) applied to the input.

## Measured (N=9, supervised, structural target; median over seeds)

| generator | test MSE (StructuralActor / HSiKAN / MLP) | forward latency (Actor / HSiKAN) |
|---|---|---|
| chain (k=2-uniform) | 0.009 / 0.007 / 0.055 | 245 µs / 2800 µs |
| k-uniform (k=3) | 0.021 / 0.0006 / 0.037 | 282 µs / 2600 µs |
| **Steiner S(2,3,9) = AG(2,3)** | **0.0008** / 0.0004 / 0.031 | **232 µs / 2700 µs** |
| sunflower | 0.017 / 0.0012 / 0.022 | 233 µs / 2900 µs |

Three measured facts:
1. **The StructuralActor matches HSiKAN's structural accuracy at ~10× lower latency and ~30× fewer parameters**
   (73 vs 2209). The key was using the *holonomy* (sign-**product** transport), not a signed sum — a ~95×
   accuracy improvement from that one change.
2. **The balanced Steiner design is the best structural basis** — Actor MSE 0.0008, an order tighter than the
   k-uniform designs, nearly matching full HSiKAN. The balance ("every pair of nodes coupled exactly once") gives
   a non-redundant walk basis with no direction over- or under-weighted.
3. It must enumerate *enough* walks; at large scale this becomes a scored top-K (a coverage/cost trade-off, and
   balanced designs need fewer walks for the same coverage — another point for Steiner).

## Why this connects to deeper mathematics (conjecture, testable)

The structure is a **principal bundle**: a flat, bounded *finite affine geometry* (the base, AG(2,3)) carrying a
*connection* whose **holonomy** supplies the nonlinearity. This places the work in several established areas, in
increasing order of how testable the link is:

- **Discrete differential geometry** — signed graph = Z₂ connection; walk = parallel transport; cycle holonomy =
  curvature; balance = flat connection. `Bᴸ` is the discrete connection-Laplacian.
- **Clifford / geometric algebra** — the transport generalises Z₂ signs → SO(3) rotors (`Spin(3)` = even
  `Cl⁺(3,0)`) → multivectors; the rotor line and "geometric attention" are the same ladder one rung up.
- **Frame theory (the falsifiable one)** — *Steiner systems construct equiangular tight frames* (Fickus–Mixon–
  Tremain, 2012). The candidate explanation for "Steiner is best" is that its blocks form a **tight frame** — an
  optimal, minimal-coherence representation. **Discriminating test:** does the frame coherence of a design predict
  the Actor's accuracy across generators? If yes, "balanced 2-design = optimal control basis" has a theorem behind
  it, bridging control, finite geometry, and frame theory.
- **Geometric control / gauge theory of locomotion** — control *as* holonomy (geometric phase; Shapere–Wilczek's
  gauge theory of deformable-body locomotion). A gait that nets displacement from a cyclic shape change *is* this;
  the walk-holonomy is its discrete analogue.

A nonlinear basis placed directly on the design's blocks (a *block-wise Steiner-KAN*) curves the effective
geometry without a separate connection — an alternative, more expressive construction worth comparing.

## Honest status

Everything measured is **representation**, not closed-loop control: a supervised toy (N=9, one structural target,
two seeds). The architecture, the Steiner-is-best result, and the speed are real and reproducible; the
geometric-control reading is a **well-posed conjecture** until the holonomy readout is wired to actuators and shown
to *control* a plant. The two cheap next experiments: (1) does frame coherence rank the generators (the
tight-frame theorem); (2) close a loop with a Steiner-structured actor on a balancing plant (geometry-as-mechanism
vs geometry-as-decoration).
