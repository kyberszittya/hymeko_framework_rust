# Novelty, contribution, originality, and effect — an assessment

**Date:** 2026-06-29 04:22 (+09:00 JST) · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Type:** assessment (no code/experiment changed; companion to the holonomy-discriminator plan).
**Method:** read the on-disk artifacts (the three theory docs, the structural-accountability
article, the acceptance-novelty execution report, the toy harnesses) + two bounded literature
rounds (WebSearch + HF paper_search). Search was **not exhaustive**; "none found" means
none found in a bounded search, not "proven novel" (CLAUDE.md novelty policy).

## What the body of work actually is

One thesis instantiated three times: **the signed hypergraph is the shared substrate; holonomy
is the common invariant; consistency is machine-checked.** Concretely, three pillars:

1. **HyMeKo — a canonical signed-hypergraph IR with machine-verified cross-view consistency.**
   One `.hymeko` source fans out to many targets (SysML, kinematic, torch `nn.Module`, DOT, HIVE,
   MuJoCo, ROS); the commuting square `Xᶠ(εᶠ(H)) = Q(H)` is verified against the **real** CLI
   emitters (Z3 + sympy) across 16 robotics + 2 systems-engineering fixtures, with a generalised
   faithful-extraction theorem, a global-invariant closure corollary (forest/DAG), and a
   drift-prevention demo that *proves* the pairwise baseline drifts.
2. **Declarative Topological Control (DTC).** A controller *is* a (signed) hypergraph topology,
   declared; what it can represent **and** control is governed by topological **match** to the
   plant. Generatable / measurable / gauge-grounded / declarable. Measured gradient (toy, N=9):
   match buys ~100× in representation, ~3% in easy control, 2–7% in constrained MPC, predicted
   decisive in hard control.
3. **Gauge-holonomy learning + StructuralActor.** Balance = trivial Z₂ holonomy (Zaslavsky);
   the predictive signal in signed-link prediction *is* holonomy/frustration; a rotor is a learned
   connection whose Spin-holonomy refines the parity. StructuralActor = walk/cycle gather = `Bᴸ`
   in one matmul (HSiKAN-accuracy at ~10× speed / ~30× fewer params; Steiner AG(2,3) best basis).
   Cross-disciplinary witnesses: aromaticity = cycle holonomy (Hückel/Möbius, *exact*); E/I
   balance = signed balance; grid cells = path-integration = holonomy.

## Corrections to my own first two passes (recorded for honesty)

- **The gauge-holonomy line is NOT pre-empted by lattice gauge-equivariant CNNs.** Those build
  networks *equivariant to a known gauge symmetry of physics data*. This work claims the *learning
  signal itself* is a holonomy and derives an estimator for it — a categorically different move. I
  keyword-collided; the neighbour does not collide on the claim.
- **The cross-view kernel's real competitor is MDE global model management** (EMF/Eclipse-OCL,
  Epsilon, megamodels, MLIR multi-dialect, CWL) — not RoboChart (which verifies controller
  refinement, a different axis). Those enforce consistency *by convention or against a metamodel*;
  this verifies a commuting diagram against the *real emitters*. That is the delta, and the
  acceptance-novelty report already scopes it correctly.
- **The StructuralActor's honest neighbour is SGC** (Wu et al. 2019, precompute `SᴷX`, drop message
  passing). Delta: *signed* transport over an *enumerated walk/cycle motif basis* (holonomy, not a
  plain adjacency power) on a *designed* combinatorial basis, read out to actuators. Incremental as
  method; the speed/param win is real engineering.

## Per-pillar verdict

| Pillar | Novelty | Evidence | Nearest neighbour (delta) | Disposition |
|---|---|---|---|---|
| 1 — Cross-view verified IR | moderate–high, **certified** | strong (16+2 fixtures, 22 tests, Z3/sympy) | EMF/MLIR/global-model-mgmt — *by convention*, not machine-verified vs real emitters | **Publish now** (T-SMC). The flag. |
| 2 — DTC (controller=topology, match=lever) | moderate (synthesis) | **low** (toy N=9, linear, Phase 3 only predicted) | structured optimal control (Lin–Fardad–Jovanović); network controllability (matching=driver nodes, Liu–Slotine) | Original program; invariant→performance *law* is open. |
| 3 — Gauge holonomy + StructuralActor | framing original; math classical | **low** (HSiKAN ties MLP; C1/C3 untested) | gauge-equiv. CNNs (machinery, *not* the claim); SGC (the actor); comp-neuro path-integration (grid-cell claim) | High ceiling, unproven. Needs one decisive measurement. |

## The four axes, named

- **Novelty:** real and **concentrated**, not distributed. Pillar 1 is novel *and* proven; Pillars
  2–3 are novel *as syntheses* but rest on conjectures. Several flashy sub-claims (HSiKAN-as-signed-KAN,
  reward-as-DSL, balance=holonomy) are classical or occupied and must be cited and *delineated*, not
  asserted as new.
- **Originality (is the idea yours, non-obvious):** high across all three. The unifying position —
  signed hypergraph as substrate, holonomy as invariant, consistency machine-checked — is a
  distinctive school, and the aromaticity identity shows the invariant is not ad hoc.
- **Contribution:** Pillar 1 = a method + a working verified artifact (knowledge added now).
  Pillars 2–3 = a program + falsifiable predictions (promissory until one bet lands).
- **Effect:** Pillar 1 is publishable today (MDE/formal-methods). Pillars 2–3 are the high-ceiling
  bets — if C1 or the grid-cell prediction lands, a much bigger paper (ML / comp-neuro). Effect is
  *conditional* on demonstrating the verifier catches a real cross-view bug, and on one task where a
  flat net provably cannot follow.

## Recommendation (decisive)

- **Plant the flag on Pillar 1** — the verified fan-out. T-SMC now.
- **Convert Pillar 3's central bet cheaply before claiming it.** The single highest-leverage,
  novelty-converting experiment is the **holonomy-discriminator toy** (planned this session,
  `docs/plans/2026-06-29-holonomy-discriminator-toy/`, BACKLOG P1): a classification whose label is
  *provably* a cycle holonomy and nothing else, through StructuralActor / HSiKAN / MLP, plus a
  zero-cost aromaticity certification. Win → "structure is load-bearing" becomes a measurement and
  the HSiKAN-ties record is explained. Tie → the gauge line is honestly down-graded. Either way the
  swing factor is resolved.
- **The honest one-line grade:** *one certified-novel formal-methods kernel; two high-ceiling,
  well-posed, empirically-unproven research programs; unusually coherent originality binding them.*

## Search provenance (bounded, non-exhaustive)

WebSearch + HF `paper_search`, 2026-06-29. Surfaced as nearest neighbours: arXiv 2501.00709
(KAN×signed-GNN), SEMBA, MFNN (matrix-function GNN), Graph Mamba, SCRaWl, RLang, SPECTRL, network
controllability (Wikipedia/Liu–Slotine), MI-HGNN, Hajdu & Hegyi 2025 (MDPI Appl. Mech. 6(4):74 — the
precursor), HyperGraphOS (2412.04923), RoboChart/RoboTool, lattice gauge-equivariant CNNs (PRL
128.032003; arXiv 2604.20797), TorchLean. From established literature (not re-searched this session,
used for delineation): Zaslavsky (signed graphs), SGC (Wu 2019), Lin–Fardad–Jovanović (sparsity-
promoting LQR), Heilbronner (Möbius aromaticity), Sorscher/Gao (path-integration as Lie-group action).

## Files touched
- New: `reports/2026-06-29-novelty-and-contribution-assessment.md` (this file).
- New plan: `docs/plans/2026-06-29-holonomy-discriminator-toy/` (plan.tex/pdf/tikz/plan_tikz.pdf/mmd).
- Edited: `BACKLOG.md` (+1 area, +1 P1 item + 1 stretch sub-line).
- CORE.YAML items touched: **none.**
