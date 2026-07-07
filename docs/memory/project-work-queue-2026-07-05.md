---
name: project-work-queue-2026-07-05
description: "STANDING ROBOTICS QUEUE (Hajdu 2026-07-05, corrected): (1) coin-toss; (2) extend coin-toss to k arms; (3) FANUC pick-and-place; (4) quadruped standing/walking; (5) humanoid standing/walking; (6) Niitsuma rapport through robot-robot interaction. Clean OO/dataflow code throughout."
metadata: 
  node_type: memory
  type: project
  originSessionId: d2ccb45c-9c6f-4422-a725-08dd14fe9109
---

Corrected directive (2026-07-05 07:36): convey the robotics plan in this order: coin-toss; extension to k-arm
coin-toss; FANUC pick-and-place; quadruped standing/walking; humanoid standing/walking; finally Niitsuma
rapport through robot interaction.

**Queue and state:**
1. **Coin-toss delivery maximization** (current: push-controller teacher useful, BC step-0 floor protected,
   TD3+BC/SAC-style refinement degrades). Immediate lever is **Q-term/critic diagnosis** using frozen-clone
   probes; do not run more refine grids or redesign the scenario until the implementation failure is isolated.
   After diagnosis, use only bounded, cached cells. "Without regression" = every change carries a one-seed
   identity check against the cached number before adoption.
2. **Extend coin-toss to k arms** — the controller is k-general (`fan_offsets`/`push_slots`/permutation assignment, k≤6
   tested at the geometry level); the ENV is not (2 hand-authored arms; `make_planar_arms_mjcf`, collab actor
   2-channel). Scope first: k-arm scene emission + k-channel actor + k in the task .hymeko.
3. **FANUC pick-and-place** — plan compiled (`docs/plans/2026-07-05-fanuc-pick-place-controller/`), discovery
   done ([[project-fanuc-pick-place-push-next]]): port = declare `expert_action`'s stages as a controller
   profile + 3D PickObs/guards/laws over `DampedPoseIK`; success only via divergence-guarded `LiftPlaceMetric`.
4. **Quadruped standing/walking** — standing exists but pure TD3 diverged/underperformed; treat standing as the
   first stable postural gate, then walking/gait as the cyclic control target. Do not skip the standing gate.
5. **Humanoid standing/walking** — after quadruped gates, move to humanoid posture and gait. Standing first,
   then walking; use the same dataflow/FSM/monitor discipline and no unverified success claims.
6. **Niitsuma rapport through robot interaction** — final collaboration-facing goal: robot-robot interaction
   scenarios that demonstrate rapport, coordination, and interaction hypergraphs. This is not merely a report;
   it should be a demonstrable interaction substrate.

**Operating constraints (from tonight, all in CLAUDE.md/memory):** oracle-certify before queue; measurements
are cached facts (no re-measuring); consult before naming; declared configs/profiles over kwargs; batch work
into few turns; monitors report verdicts only.
