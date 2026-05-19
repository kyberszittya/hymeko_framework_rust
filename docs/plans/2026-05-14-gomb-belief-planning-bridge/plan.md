# Gömb as a belief → planning → execution bridge over signed hypergraphs

**Date:** 2026-05-14
**Status:** **research-vision plan**, not a build commitment.
**Effort:** multi-year if pursued fully; near-term validation pieces
sized in months.
**Origin:** user shower-thinking session, 2026-05-14. Two insights —
(I) Gömb as the unifying representation across belief / planning /
execution layers on a hypergraph world model, and (II) a "derivative
of a clique = nodelet" multiscale operator. They compose.

## The claim

> A signed hypergraph is the right shared representation for an
> embodied agent's three classical layers — **belief**, **planning**,
> **execution** — and Gömb is the architecture that makes the bridge
> implementable. The standard symbolic / sub-symbolic divide
> dissolves because (a) HyMeKo's IR already represents environments
> as signed hypergraphs, (b) Gömb's cycle pool computes σ-products
> over those hypergraphs in poly time, and (c) σ-products are the
> *consistency operator* that all three layers need:
>
>  - **belief layer**: σ-product = evidence consistency (do
>    independent observation chains agree?)
>  - **planning layer**: σ-product = plan coherence (does a
>    candidate action coalition stay balanced under projected
>    consequences?)
>  - **execution layer**: σ-product = motor-loop closure (do
>    actuator-feedback cycles return consistent state?)

This unifies what historically required three different
representations (probabilistic factor graphs / STRIPS-style symbolic
planners / motor primitive libraries) under one hypergraph
substrate.

## CORE.YAML items touched

**Empty for the vision plan.** Implementation pieces — when scoped
into separate plans — will likely require:

- New dependency: a planner library (PDDL parser, hierarchical
  task network) or differentiable sim (already covered by the
  locomotion plan's CORE escalation).
- Possible architectural extension to `hymeko_gomb` for multi-scale
  pooling (the "nodelets" operator) — would be a `lockdown:
  implementation` concern; additive public API only.

Each sub-experiment that touches CORE will request approval under
its own plan slug.

## The two insights, explained

### Insight I — Gömb as the bridge

Three classical layers in BDI / robotics architectures:

1. **Belief (B)** — what the agent thinks is true about the world.
   Classical: factor graph, Bayesian network, knowledge graph,
   probabilistic database. Modern: latent state of a recurrent
   network / world model.
2. **Planning (P)** — what action sequence the agent picks given
   beliefs and goals. Classical: STRIPS, PDDL, hierarchical task
   networks. Modern: MCTS, model-predictive control, RL policy.
3. **Execution (E)** — how the chosen action becomes motor output.
   Classical: motor primitive library, finite-state machine,
   feedback control law. Modern: end-to-end learned policy.

The crossing-the-streams problem is well known: B and P are
typically symbolic / structured, E is typically continuous /
numerical. Neuro-symbolic AI is the field that tries to bridge.

**The hypergraph proposal:** all three layers live on the **same
signed hypergraph** `H = (V, E, σ)`, with three projections:

- `H_B` — belief annotation: edges carry epistemic sign (+ = evidence
  for, − = evidence against), σ-product on cycles = belief coherence.
- `H_P` — plan annotation: same vertex set, edges represent action
  preconditions / effects, σ on cycles = plan consistency under
  rollback.
- `H_E` — execution annotation: edges are motor-feedback paths, σ on
  cycles = sensorimotor closure.

These aren't three graphs — they're three *signings* of the **same
topology**. The vertices (entities, objects, agents) and the
underlying connectivity (which entities are related) are shared.
What changes per layer is the *meaning* of the sign.

Gömb's cycle pool computes σ-products natively. The same architecture
that does signed-link prediction on Bitcoin Alpha also does:
- *belief calibration* (predict the sign of an unobserved evidence
  edge),
- *plan-step validity* (predict the sign of a proposed action),
- *motor-loop convergence* (predict the sign of a feedback
  correction).

### Insight II — Cliques to nodelets as a "derivative" operator

A **clique** is a set of mutually-related vertices. A **nodelet** is
the *single object* that represents that clique after contraction —
the local coarsening operator collapses each clique into one
super-vertex with edges to whatever the clique was connected to.

**Why this is a "derivative":** taking the derivative of a graph
loses local detail in exchange for a global shape — much like
differentiating a function loses pointwise values for slopes. A
graph and its clique-contracted form sit in different "scales" of
representation:

  `H` (fine) → `H'` = clique-contracted hypergraph (coarse)

Iterate to get a hierarchy:

  `H → H' → H'' → H'''` (multi-scale)

Each level loses some detail but exposes higher-order structure:
- `H` knows which robots talk to which.
- `H'` knows which **teams** (balanced cliques) talk to which.
- `H''` knows which **coalitions of teams** form.
- ... up to a single root nodelet representing "the entire
  community".

**Formally**, the operator is some flavor of:

  `D(H) = (V', E', σ')`
  where  `V' = {balanced maximal cliques of H}`
         `E' = induced edges between cliques (mediated by shared vertices)`
         `σ' = aggregate sign of the boundary between two cliques`

The "derivative" framing connects to discrete exterior calculus on
hypergraphs (where the boundary operator `∂` is well-defined for
simplicial complexes), to hypergraph sheaf theory (where each
nodelet is a "stalk" carrying the local algebra), and to
renormalization-group methods on graphs (Coen / Lambiotte 2019,
Castellano-Pastor-Satorras).

**Why this matters for the bridge:**

The belief / planning / execution layers need to operate at
**different resolutions**:

- Belief tracking happens at the finest scale (every individual
  observation matters).
- Plan synthesis happens at coarse scales (you plan teams of robots,
  not joint angles).
- Execution dispatches back down to the finest scale (joint torques).

The cliques → nodelets operator is the **multi-scale glue**. A plan
made at scale `H''` (teams of teams) gets dispatched down through
`H'` (team-level coordination) to `H` (per-robot commands).
Belief flows the other direction: per-robot observations get pooled
into team-level state, then coalition-level summaries.

## How the two insights compose

Gömb's σ-product features at scale `H` feed cliques → cliques become
nodelets at scale `H'` → Gömb runs *again* on `H'` → and so on. The
**same architecture** operates at every scale of the multi-scale
hypergraph. This is the hierarchical bridge.

The flow:

```
   observation chain          execution chain
        │                          ▲
        ▼                          │
   ┌────────┐                ┌────────┐
   │  H_B   │ ◄── belief  ── │  H_E   │   ← fine-scale: per-vertex
   └────────┘                └────────┘
        │                          ▲
        │       cliques → nodelets │
        ▼                          │
   ┌────────┐                ┌────────┐
   │  H'_B  │                │  H'_E  │   ← team-scale
   └────────┘                └────────┘
        │                          ▲
        │       cliques → nodelets │
        ▼                          │
        ▼─────────► PLANNER ◄──────▲   ← coalition-scale
                    (operates on
                     coarse layer)
```

At each scale, Gömb is the consistency-check engine. The planner
runs on the coarsest scale; execution dispatches actions back down
through nodelet expansion.

## Concrete validation experiments (small, near-term)

The vision is multi-year. But each step can be validated piece by
piece. Sized roughly:

### V1 — Cliques-to-nodelets operator on signed graphs (~2 weeks)

- Implement `contract_balanced_cliques(H) -> H'` as a new module
  `signedkan_wip/src/demo/cliques_contract.py`, building on the
  cliques generation/detection foundation
  (`docs/plans/2026-05-14-cliques-generation-detection/plan.md`).
- For each detected balanced clique, replace its members with a
  single super-vertex; aggregate boundary edges (sign = majority
  sign of merged edges).
- Verify: applying `contract` twice on a hand-built 3-level
  hierarchy returns the expected root.
- Demo: extend the GUI cliques tab with a "Contract" button that
  shows `H` and `H'` side by side, plus a "Show hierarchy" view that
  iterates contraction.

**Why this is a real step:** Renormalisation-group on signed
hypergraphs is a clean math object, and we'd have it implemented.
Even without Gömb integration, it's a publishable contribution to
graph theory tooling.

### V2 — Train Gömb on contracted hypergraphs (~3 weeks)

- After V1 ships, train two Gömb models: one on `H`, one on `H'`.
- Compare: does Gömb on `H'` capture the *same* downstream signal
  (e.g., edge-sign prediction) at fewer parameters and faster
  inference?
- This is the empirical leg of "Gömb is scale-equivariant" — the
  cycle pool's inductive bias should work at every scale of the
  hypergraph.

**Why this is a real step:** scale equivariance under
clique-contraction would be a defensible architectural claim,
worth a workshop paper.

### V3 — Belief layer prototype on Bitcoin / Slashdot (~3 weeks)

- Repurpose the signed-link prediction setup as a belief-tracking
  demo: each edge sign is "agent u's evidence about agent v's
  trustworthiness".
- Train Gömb online: stream observations one edge at a time, update
  cycle-pool features incrementally.
- Measure: does the model maintain calibrated probabilities for
  unobserved edges as new evidence arrives?
- Connect to standard online-belief-tracking baselines (particle
  filters, factor-graph BP).

**Why this is a real step:** demonstrates the *belief* leg of the
bridge on real data we already have checkpoints for.

### V4 — Planning prototype on robot communication cliques (~4 weeks)

- Extend the cliques demo (Niitsuma narrative) into a planner: given
  a goal "form a team of size ≥ N with property P", search the
  contracted multi-scale hypergraph for satisfying coalitions.
- Use Gömb's vertex embeddings + balanced-clique extraction (from
  the NP-hard plan Stage 2) as the planner's heuristic and operator.
- Compare against classical signed-graph planners (Halpern,
  Halpern-Meliou) and MCTS over the discrete action space.

**Why this is a real step:** demonstrates the *planning* leg, ties
back to the cliques work, and gives the Niitsuma talk a teaser for
the unified vision.

### V5 — Execution layer: motor-loop closure on a simulated robot (~quarter)

- Co-developed with the legged-locomotion plan
  (`docs/plans/2026-05-14-legged-locomotion-contact-mode/plan.md`).
- The contact graph at each timestep is one scale of the hypergraph
  (`H_E`); the action repertoire (gait patterns) is the next
  coarsening (`H'_E`); strategic mode (locomotion regime) is `H''_E`.
- Gömb's σ-product features on `H_E` close the sensorimotor loop —
  detecting when contact is stable (balanced cycle) vs. slipping
  (imbalanced).

**Why this is a real step:** ties the execution leg to a concrete
robot platform; closes the perception-action loop on Gömb's
representation.

## What this is NOT

- **Not** a near-term Niitsuma talk deliverable. The talk should
  focus on cliques v0.5 + the NP-hard pivot Stage 1; this vision is
  the *closing slide*, not the demo.
- **Not** a replacement for existing classical methods (POMDP
  solvers, ROS / MoveIt, factor-graph BP). It's an *integrative*
  layer that consumes and exposes interfaces compatible with each.
- **Not** a build commitment without further plan-doc work. Each V1
  – V5 will get its own scoped plan when it's the active priority.

## Connections to existing work in this repo

- **`hymeko_monitor`** (`project_hymeko_monitor_scaffold`) — already
  scaffolds STL runtime monitors over signed-incidence hypergraphs.
  Belief-layer V3 should integrate with it.
- **HyMeKo IR + URDF parser** — already represents robot kinematics
  as a signed hypergraph. Execution-layer V5 reuses this directly.
- **HymeKo-Gömb three-shell cascade** — `OuterFIRShell` /
  `MiddleHSiKAN` / `InnerCPMLCore` is already a multi-scale
  architecture. Insight II's nodelets operator can be viewed as the
  cascade's geometric foundation made explicit.
- **HymeKo-driven HSiKAN** (`project_hymeko_driven_hsikan_2026_05_05`)
  — already has Gömb training driven by .hymeko config files.
  Planning layer V4 extends this to runtime plan synthesis.
- **`hymeko_pgraph`** — the pgraph crate is the lowering layer for
  HyMeKo IR; multi-scale operators (V1) would be additive there.
- **Self-evolving cycle sampling**
  (`docs/plans/2026-05-14-self-evolving-cycle-sampling/plan.md`) — the
  *learned-sampling* leg pairs with this plan's *learned-coarsening*
  leg. Bridge V2 contracts topology; the sampling plan reweights the
  feature pool. Composed: contracted hypergraph + importance-weighted
  cycles at every scale of `H → H' → H'' → ...`. **Both legs make the
  same architectural argument** — Gömb's own outputs feed back into
  Gömb's inputs — from different angles.

## Risks + open mathematical questions

- **What exactly is the right "derivative" operator?** Many viable
  definitions; pinning down the right one (likely a sheaf-cohomology
  flavor) is hard math. V1 starts with the most operational version
  (balanced-clique contraction) and validates empirically before
  committing to a formal foundation.
- **Does scale-equivariance actually hold for Gömb?** Architectural
  claim, empirically testable in V2. If it fails, the multi-scale
  bridge becomes brittle.
- **Belief calibration is hard.** Online-trained neural networks
  generally aren't calibrated without explicit techniques (Platt
  scaling, conformal prediction). V3 needs to test against calibrated
  baselines.
- **Planning combinatorics.** Even at coarse scales, planning over
  hypergraph action spaces is expensive. V4 needs to show that
  multi-scale dispatch genuinely reduces the planning search space,
  not just defers it to a lower level.

## Why no TikZ/PDF/Mermaid plan (yet)

This is a **research vision document**, not a build artifact. The
4-format plan requirement (CLAUDE.md §2) kicks in when implementation
starts. Each of V1 – V5, when picked up as the active priority, will
get its own 4-format plan with the standard rigor. The vision
document above frames *what the pieces fit into*, not what gets
built next week.

## Empty-plan-dir hygiene

If this direction is permanently abandoned, delete
`docs/plans/2026-05-14-gomb-belief-planning-bridge/`. But this
plan is meant to be **load-bearing for the long-term research
identity**, not disposable. It should still exist 6 months from
now even if no V-step has launched, as the reference point for
whether a given sub-experiment fits the broader arc.

## Slide-shaped narrative for the closing of the Niitsuma talk

Three sentences:

> Gömb's σ-product features over signed hypergraphs are not just a
> link-prediction tool — they are the **consistency operator** that
> a unified belief / planning / execution stack needs. Multi-scale
> reasoning emerges by *contracting balanced cliques into nodelets*
> and running Gömb again at the coarser scale: the same architecture,
> the same inductive bias, every layer of abstraction. This is the
> bridge between sub-symbolic graph learning and classical symbolic
> planning that neuro-symbolic AI has been trying to build for thirty
> years.

## Order of work — none, until the prerequisites land

This plan does not dictate immediate work. It exists so that when
the cliques foundation + NP-hard Stage 1 + Niitsuma talk are done,
we can pick V1 – V5 in priority order rather than re-discovering
the framing each time.
