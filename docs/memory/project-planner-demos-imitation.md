---
name: project-planner-demos-imitation
description: "Dr. Hajdu 2026-06-25 (toward the 'former vision'): replace the scripted demonstrator with an EXACT planner (A*/RRT) as the BC demo source; RL mimics + amortises. Plan-then-amortise = planner is the verifiable deliberation, RL the fast reflex. 4-artifact plan on disk."
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**Idea (Dr. Hajdu, 2026-06-25), toward his 'former vision'.** The delivery line is capped by DEMO QUALITY — the
hand-coded `GalambosDemonstrator` is brittle/task-specific, BC can't exceed it (~0.17). Replace the demo SOURCE
with an **exact motion planner** (A* optimal on a discretised config/workspace; RRT(*) complete/asymptotically-
optimal in continuous high-DOF). RL then **mimics + amortises** it: ms inference (planner is slow per query) + the
contact-rich dynamics the kinematic planner can't model. **Plan-then-amortise = the planner is the slow VERIFIABLE
deliberation, the RL policy the fast reflex** — the rate-asymmetric loop [[project-kato-dual-discriminator-plan]]
but with an EXACT algorithm as the deliberation. Realises the G-SPHF attractor/graph-planning vision
[[project-gsphf-attractor-planning-integration]] + `docs/plans/2026-05-14-gomb-as-planner-heuristic` (Gömb as the
A* heuristic = phase-2 bridge); A* IS reachability search over the config graph [[project-reachability-rules-article]].

**Why it works:** from-scratch RL can't DISCOVER the approach (the wall [[project-galambos-hsikan-tie-rootcause]]);
a COMPLETE planner always finds it. Planner breaks reach/transport; RL learns the rest.

**HONEST LIMIT (central):** A*/RRT are COLLISION-AVOIDANCE (kinematic) planners — they do NOT plan CONTACT. They
give reach/transport, NOT the grasp closure (nor the non-prehensile coin PUSH, which needs the coin's dynamics).
Grasp-closure stays a scripted primitive or RL refine. Still a big win — reach is exactly where from-scratch dies.

**Plan: `docs/plans/2026-06-25-planner-demos-imitation/` (4 artifacts compile/validate).** Reuse (NO rebuild,
discovery-confirmed): `hymeko_graph::{traversal_heuristic, bfs, traversal}` already have A*/heuristic search; the
`collect_galambos_demos`/`collect_demos` → (obs,acts) seam is the drop-in for `collect_planner_demos`; BC→refine
pipeline + `campaign_viz` GIF/plot unchanged. NEW = `hymeko_rl/planner.py` (astar_config + rrt_config ~150 LOC +
collect_planner_demos) + a `demo_source ∈ {scripted,astar,rrt}` switch in galambos_bc/gripper_pick_bc + the A/B
exp (scripted vs A* vs RRT vs from-scratch × HSiKAN/MLP, delivery + §9 graphics). Non-core. Headline test: BC on
planner demos ≥ scripted-demo delivery, planner demo success-rate ≫ scripted. MVP = A* on the planar arm (Galambos
reach); arc = RRT for FANUC 6-DOF + Gömb-heuristic + planner as the deliberative loop of the rate-asymmetric ctrl.
**How to apply:** when picked up, plan-first done; build planner.py, wire demo_source, run the A/B with GIF+plot.
