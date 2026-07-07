---
name: project-gsphf-attractor-planning-integration
description: LONGER GOAL (user 2026-06-20) — extend G-SPHF as the pivotal integrating substrate unifying continuous attractor-field/ODE control + discrete graph planning; both ride on the signed hypergraph via the MDSD meta_/from_hymeko pattern (mostly NON-core)
metadata: 
  node_type: memory
  type: project
  originSessionId: e049ea12-7387-4a59-87f4-051966d7cfcb
---

**User's longer goal (2026-06-20):** "extend G-SPHF here, as pivotal integration of other approaches.
Via description of attractor fields (differential equations) and graph planning."

**REFINED (2026-06-20):** G-SPHF acts as a **MIXIN + ACTOR MIXTURE** so plans are generated
**ALGORITHMICALLY first (for precision)**, then that **plan is LEARNT for the use case**. I.e.
plan-then-learn: a precise model-based/algorithmic actor (graph plan composing attractor controllers)
seeds/warm-starts a learned actor (RL specialises it); G-SPHF composes them as mixins → a mixture of
actors (algorithmic + learned). The model-based actor gives precision + can't do degenerate shortcuts
(knock-out/clump); learning adapts to the specific use case. THIS IS A LONGER PLAN — deferred; near-term
work continues on patching the Galambos RL ([[project-galambos-reward-shaping]]).

**Reading:** G-SPHF (the GGK 4-tuple K=(B,G,μ,r) in `hymeko_core`, CORE `lockdown:full`) becomes the
COMMON substrate that integrates three paradigms, each a *view/use* of one signed-hypergraph description:
1. **Attractor fields / ODEs** (continuous control) — describe the DESIRED closed-loop dynamics as a
   vector field / differential equation attached to hypergraph elements (point attractor, limit cycle,
   DMP-style forcing). The Galambos RL reward (pull-to-zone + center_bonus) is ALREADY an *implicit*
   point attractor at the goal zone — the extension makes it EXPLICIT/declarative.
2. **Graph planning** (discrete) — plan a path/sequence of subgoals over the hypergraph. HALF-BUILT:
   `hymeko_pgraph` (A1–A5, MSG/SSG/ABB, reachability), `data/robotics/planning_examples/
   dual_agent_blocksworld_2x3.hymeko`, the reachability-rules article ([[project-reachability-rules-article]]).
3. **RL** (learn the controller) — already done for Galambos (HSiKAN actor+critic over the hypergraph).

**The integration = sequential composition of funnels** (Burridge–Koditschek precedent): a task is a
GRAPH PLAN (which attractor-basin → which) where each edge is realised by an ATTRACTOR FIELD (continuous
flow to the subgoal). Attractor side ≈ DMPs (Ijspeert) / SEDS (Khansari-Zadeh) / contraction. G-SPHF is
the single formalism describing both the funnels and the plan that composes them. "Pivotal integration"
= one description → RL (learn attractor) + model-based control (specify attractor/ODE) + planning (graph).

**Concrete path = the SAME MDSD move as this whole session (mostly NON-CORE):** new
`meta_attractor.hymeko` (vector-field/ODE per subgoal) + `meta_plan.hymeko` (graph plan / reachability
operators, reuse pgraph) + executors (`AttractorController` flows the field; `Planner` sequences funnels),
read via `from_hymeko` like `EnvSpec`/`StrategySpec`/`RewardSpec`. Rides on **B (incidence)** as typed
annotations/edges — NO change to the 4-tuple → non-core, like meta_env/meta_strategy/meta_reward.
**CORE caveat:** only if the attractor needs a NEW COMPONENT in K=(B,G,μ,r) itself (a 5th field for the
flow) does it touch `hymeko_core` (lockdown:full) → plan + `APPROVED-CORE-EDIT` (§1). Prototype the
profiles+executors first, escalate only if the core tuple genuinely must change.

**First concrete step (when started):** sketch `meta_attractor.hymeko` (point-attractor field for the
Galambos goal zone) + an `AttractorController` that flows the disk to the zone — a model-based controller
to sit BESIDE the learned HSiKAN policy (compare/compose). Then `meta_plan.hymeko` over pgraph for a
multi-subgoal task. Ties [[project-galambos-reward-shaping]], [[project-kato-collaboration-grasping]],
[[project-reachability-rules-article]]. NOT a sprint — multi-paper direction; don't over-promise.