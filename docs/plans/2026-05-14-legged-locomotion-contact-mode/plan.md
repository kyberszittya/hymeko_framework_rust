# Legged-locomotion contact-mode reasoning via Gömb

**Date:** 2026-05-14
**Status:** plan v1 — **future work / Niitsuma talk slide**, NOT a
commitment to ship in the near term.
**Estimated effort:** 1 quarter of focused work. Too large for a
single talk-prep cycle; sized for an ICRA / IROS / RA-L paper.

## The claim

> A legged robot's contact graph — which links touch which surfaces
> at each instant — is a time-varying signed graph. Predicting the
> *next* contact mode (which feet land where, which lift off) is a
> combinatorial / NP-hard problem currently solved by hand-tuned MPC
> schedules. Gömb's cycle pool is the right inductive bias for this
> prediction problem because: (a) transient closed kinematic loops
> form precisely during contact phases, (b) the σ-product
> structure encodes force-balance invariants, (c) faction-recovery
> generalises to "which feet are in the same contact group".

This is the strongest applied-control story for Gömb. It is also far
out of scope for the current talk; the goal of this plan is to make
it a *credible* future-work slide, not to ship anything yet.

## Why this matters

The current state of legged-robot optimal control:

- **Quadrupeds (Cheetah, ANYmal, Spot)**: contact-mode sequences are
  hand-designed gait patterns (trot, gallop, bound). MPC optimises
  joint torques *given* the gait pattern; the gait selection is
  combinatorial and largely engineered.
- **Humanoids (Atlas, Digit, HRP)**: contact-mode planning is even
  harder — 12+ contact points, friction cones, balance constraints.
  Currently solved by sample-based planners + MPC tracking.
- **Recent learned-control work (MIT Cheetah, ANYmal-C, MIT Humanoid)**:
  end-to-end RL using MLPs / Transformers on the state vector. Often
  beats classical MPC on robustness, but has no explicit contact-graph
  structure.

The gap Gömb fills: **a learned controller with structural inductive
bias for the contact graph**. Not as an end-to-end replacement; as a
*contact-mode predictor* feeding the existing MPC scheduler. The MPC
still does the torque optimisation; Gömb chooses the mode.

## CORE.YAML items touched

Adding new sim infrastructure (MuJoCo MJX, Brax, or Genesis) is a
new dependency — **treated as a CORE.YAML edit** per CLAUDE.md §1.
Approval required before any code is written. This plan does not
presume approval; it documents the path *if* the user later
authorises the sim addition.

## Three-stage research arc

### Stage 1 — Sim + contact-graph extraction (~2 weeks)

- Pick a baseline simulated quadruped: **MIT Mini-Cheetah** model in
  MuJoCo MJX (cleaner contact model than Brax, differentiable, ~kHz
  on a GPU).
- Build `contact_graph_from_mjx(state) -> SignedGraph` — at each
  timestep extract:
  - Vertices: links + ground patches.
  - Edges: contact pairs with friction-cone-direction-derived sign.
- Validate: visualise the contact graph alongside the simulated robot
  at three gaits (stand, trot, gallop) — graph topology should change
  visibly in sync with foot lift/land.

**Deliverable:** notebook + 30 s video of contact graph evolving
during a trot.

### Stage 2 — Contact-mode prediction (~3 weeks)

- Generate a corpus of locomotion episodes (50 k timesteps across
  random gait commands + terrain perturbations).
- Each timestep: (joint state, base pose, terrain context) → contact
  graph at *t + Δt*.
- Train Gömb (4-class output: stance / trot / gallop / fall) AND an
  MLP baseline on identical inputs.
- Metric: prediction accuracy at Δt = 50 ms, 100 ms, 200 ms.
- **Headline target:** Gömb matches MLP on `Δt = 50 ms` but pulls
  ahead at 200 ms (longer-horizon prediction is where structural
  bias matters).

**Deliverable:** report with prediction curves + sample-efficiency
plot (Gömb should learn faster from less data due to the inductive
bias).

### Stage 3 — Closed-loop control via Gömb-predicted mode (~3 weeks)

- Replace the hand-coded gait scheduler in a standard MIT-Cheetah MPC
  pipeline with Gömb-predicted mode.
- Compare against:
  - Hand-coded scheduler baseline.
  - MLP-predicted mode (Stage 2's baseline).
- Metric: locomotion robustness on uneven terrain (success rate at
  crossing a randomly-perturbed terrain bench).

**Deliverable:** ICRA-shaped paper: "Learned contact-mode prediction
for legged locomotion via signed cycle aggregation".

## What this is NOT (scope guard)

- **Not** end-to-end RL replacing MPC. The MPC stays; Gömb chooses
  modes.
- **Not** model-predictive learning of full dynamics. We predict
  *which contact mode is active*, not joint torques.
- **Not** a Niitsuma-talk deliverable. The talk gets the *idea*
  (one slide), not the experiment.
- **Not** humanoid in v1 — quadruped first because contact-mode
  reasoning is more dramatic and the sim setup is simpler. Humanoid
  follows in a separate plan if Cheetah works.

## Why a quadruped first (Cheetah > humanoid)

- **Cleaner contact structure** — 4 well-defined feet; humanoid has
  toe / heel / arm contact pairs that complicate the graph.
- **Sharper dynamics** — Cheetah at full speed switches contact mode
  every ~100 ms; the prediction problem is genuinely time-critical.
- **Better instrumented sim** — MIT Mini-Cheetah in MuJoCo / MJX is
  the de facto research baseline; many published comparisons.
- **Lower failure cost** — a Cheetah falling is less of a paper
  problem than a humanoid falling (which often correlates with a
  buggy balance term).

Humanoid (Atlas, MIT Humanoid) follows in a v2 plan if Cheetah
demonstrates Stage-2 advantage.

## Prerequisites

1. **Cliques generation/detection foundation**
   (`docs/plans/2026-05-14-cliques-generation-detection/plan.md`)
   landing first — needed to validate that Gömb's mode-prediction is
   driven by structural pattern recognition, not statistical
   regularity.
2. **Faction-recovery demo from the NP-hard plan**
   (`docs/plans/2026-05-14-gomb-np-hard-approximation/plan.md` Stage 1)
   — gives a synthetic proving ground before attempting real contact
   data, where ground truth is much messier.
3. **Sim dependency approval** — MuJoCo MJX or Genesis added under
   CORE.YAML §1 mechanism.

## Risk anticipation

- **Contact precision in MuJoCo MJX.** Differentiable contact has
  known precision issues at high speed. Mitigation: validate the
  contact-graph extractor against MuJoCo's standard pipeline
  (non-differentiable) before training.
- **Mode labels are noisy near switch boundaries.** During the ~10 ms
  transition between gaits, "which mode" is ambiguous. Mitigation:
  define modes as 50-ms windows, label each window by majority foot
  state, exclude transition windows from training.
- **MLP baseline already matches Gömb.** Plausible if the contact
  graph is too simple for cycle features to matter. Mitigation: focus
  on harder terrains (uneven ground, sliding patches) where contact
  modes change faster and structural reasoning genuinely helps.
- **MPC integration is engineering-heavy.** Stage 3 might consume
  more time than budgeted. Cut to "Gömb predicts mode but is not
  integrated with MPC, only evaluated offline" if Stage 2 numbers
  come in late.

## Why no TikZ/PDF/Mermaid plan now

This is *future work for the talk*, not a build commitment. When the
prerequisites (cliques foundation + NP-hard Stage 1) ship and the
user gives a CORE.YAML green-light for sim, upgrade to a full
four-format plan at that time.

## Empty-plan-dir hygiene

If this future-work direction is later abandoned, delete
`docs/plans/2026-05-14-legged-locomotion-contact-mode/`.

## Slide-shaped narrative for the Niitsuma talk

Three sentences for the future-work slide:

> Gömb's σ-product features over kinematic cycles are not just a
> structural classifier — they're an inductive bias for the
> *contact graph* of a legged robot. We hypothesise that Gömb can
> predict the next contact mode of a Cheetah-class quadruped 200 ms
> ahead of time more accurately than an MLP baseline, by exploiting
> the transient closed-loop structure that forms during foot contact.
> This would slot in as the learned mode-selector inside a standard
> MPC pipeline — the existing controllers stay; Gömb chooses the gait.
