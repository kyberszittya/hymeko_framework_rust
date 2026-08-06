# Paper candidate 2 — Gauge holonomy as the learning signal on signed (hyper)graphs

**Working title:** *Balance is Trivial Holonomy: Learned Connections on Signed Hypergraphs and the
Invariant Shared by Aromaticity, Neural Balance, and Path Integration*
**Target venue:** ceiling is a flagship (Nature/Science) *if* one cross-domain prediction lands
empirically; the realistic first paper is NeurIPS/ICLR (theory + discriminator experiment).
**Status:** theory on disk and internally coherent; the single decisive experiment is **planned but
not executed** (`docs/plans/2026-06-29-holonomy-discriminator-toy/` — plan exists, no report).

## Abstract seed

Structural balance on a signed graph is exactly the statement that a Z₂ gauge connection has trivial
holonomy (Zaslavsky). We argue the predictive signal exploited by signed-link predictors *is*
holonomy/frustration, generalize the sign to a learned continuous connection (a rotor) whose
Spin-group holonomy refines the parity, and show the resulting StructuralActor — a static gather
along enumerated walks/cycles, i.e. Bᴸ in one matmul — matches message-passing accuracy at ~10×
speed and ~30× fewer parameters. The invariant is not ad hoc: Hückel/Möbius aromaticity is *exactly*
cycle holonomy, cortical E/I balance is signed balance, and grid-cell path integration is holonomy of
a learned connection.

## Central claims (delineated)

- **C0 (theorem, classical):** balance ⇔ trivial Z₂ holonomy. Cite Zaslavsky; do not claim.
- **C1 (hypothesis):** the learnable signal in signed-link prediction is holonomy/frustration and
  nothing else — a model that can only read holonomy should match one that reads everything.
- **C2 (measured, engineering):** StructuralActor achieves HSiKAN accuracy at ~10× speed / ~30×
  fewer params; Steiner AG(2,3) is the best walk basis (`reports/2026-06-28-structural-actor-steiner.md`).
- **C3 (hypothesis):** a learned continuous (rotor) connection generalizes the discrete balance
  parity and wins where Z₂ underfits.
- **C4 (cross-domain identity, exact in one case):** aromaticity = cycle holonomy (Hückel/Möbius —
  checkable at zero cost against known molecules); grid cells = path integration = holonomy
  (prediction, comp-neuro); E/I balance = signed balance (framing).

## Evidence ledger

**Measured:** C2 (StructuralActor speed/params/accuracy; Steiner basis). NAGARE holonomy toys
(fixed quaternion-holonomy features are linearly separable on toy tasks — see candidate 3) are
*consistent with* but do not test C1.

**Inferred:** the coherence of the three-pillar framing (substrate/invariant/verification) across
the repo's results.

**Still hypothesis:** C1, C3, C4's neuro predictions. On record: HSiKAN *ties* MLP across RL tasks
(wiring-bug hypothesis falsified, 2026-06-26) — the tie is the anomaly C1 must explain: if the
signal is holonomy and the tasks' graphs carry none, ties are predicted; that is a testable account.

## The swing experiment (planned, NOT run)

`docs/plans/2026-06-29-holonomy-discriminator-toy/plan.pdf`: a classification task whose label is
*provably* a cycle holonomy and nothing else, run through StructuralActor / HSiKAN / MLP.
- Win (structured models separate, MLP cannot at matched params) → "structure is load-bearing"
  becomes a measurement; the HSiKAN-ties record is explained; C1 graduates from hypothesis.
- Tie → the gauge line is honestly downgraded; the paper becomes C2's engineering result plus C0's
  framing, venue drops accordingly.
Either outcome resolves the program's swing factor. Cost was estimated small (toy scale, CPU).
**This is the highest-leverage unexecuted experiment in the repository.**

## On-disk artifacts

- Theory: `docs/theory/gauge_holonomy_signed_hsikan.{tex,pdf}`,
  `docs/theory/chem_bio_neuro_equivalences.md`, `docs/theory/structural_actor_design.md`.
- Measurements: `reports/2026-06-28-structural-actor-steiner.md`; rotor-holonomy toy
  `reports/2026-06-26-rotor-holonomy-toy.md`.
- Plan (pending): `docs/plans/2026-06-29-holonomy-discriminator-toy/` (plan.tex/pdf/tikz/mmd).
- Anomaly record C1 must explain: `reports/2026-06-26-hsikan-wiring-audit.md`,
  memory `project-hsikan-loses-possible-bug` (falsified wiring hypothesis).

## Prior art and delineation (from the bounded 2026-06-29 search; non-exhaustive)

- **Lattice gauge-equivariant CNNs** (PRL 128.032003) — build equivariance to a *known* physical
  gauge symmetry; do NOT claim the learning signal itself is holonomy. No collision on C1 (this was
  explicitly corrected in the 06-29 assessment after an initial keyword collision).
- **SGC** (Wu et al. 2019) — the honest neighbor of StructuralActor (precompute SᴷX). Delta:
  *signed* transport over an *enumerated, designed* walk/cycle basis (holonomy, not adjacency
  powers), read out to actuators. C2 alone is incremental-as-method; the win is engineering.
- Zaslavsky (signed graphs), Heilbronner (Möbius aromaticity), Sorscher/Gao (path integration as
  Lie-group action) — the classical spine; cite and build on.
- Search debt: targeted search for "frustration as learning signal", "holonomy features GNN",
  and signed-GNN theory papers post-2025 before drafting.

## Missing work to reach submission

1. **Run the discriminator toy.** Everything else is contingent on its outcome. (Plan on disk;
   BACKLOG P1.)
2. **Aromaticity certification (zero-cost).** Compute cycle holonomy on known aromatic/antiaromatic
   molecules and show the exact identity — a table, not a training run. Turns C4's strongest case
   from prose into a result.
3. **C3 test:** a task where Z₂ parity underfits but a continuous rotor connection separates
   (the rotor-holonomy toy is the seed; needs a designed task where the continuous group is
   provably required).
4. **Explain-the-tie section:** re-read the HSiKAN-ties record through C1 (which benchmark graphs
   carry holonomy signal, measured — e.g. frustration of the task graph vs the tie/win outcome).
5. **Scope decision:** flagship framing (C4 with a neuro collaboration) vs ML-venue framing
   (C0–C3 + aromaticity table). Recommend the latter first; the former needs external data.
6. Multi-seed discipline on every measured claim (§3); graphics per §9.

## Risks / falsifiers

- Discriminator toy ties → C1 dead as stated; salvage is C2 + framing (paper survives, smaller).
- Aromaticity identity is exact but may be judged "known in spirit" by chemists — the contribution
  must be stated as the *unification*, not the chemistry.
- Grid-cell claim without new data is citation-synthesis; do not oversell it as a result.
