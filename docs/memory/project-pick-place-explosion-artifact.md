---
name: project-pick-place-explosion-artifact
description: "2026-06-30 MAJOR CORRECTION: the FANUC pick-place \"BC-anchor win\" (HSiKAN 0.875 lift / 0.5 place) was an EXPLOSION ARTIFACT, not a result. The gripper detonated on contact (coarse 1e-3 sub-step), ejecting the box upward; eval_success trusted info[\"lifted\"] post-blow-up and counted it as a lift. Real number ≈ 0.125 lift / 0.0 place. Hajdu caught it by noticing the gripper exploded."
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

**The finding.** The headline pick-place result — HSiKAN+BC-anchor **0.875 lift / 0.5 place** vs MLP collapse —
was an **explosion artifact**. Decisive test: same checkpoint, same eval seeds, only the MuJoCo sub-step differs:
`timestep 1e-3 (explodes) -> lift 0.875 / place 0.5`; `timestep 5e-4 (stable) -> lift 0.125 / place 0.0`. The
gripper detonated on a table↔object contact (`|qacc| ~1e4`, box pinched), the box was ejected upward
(`lifted ≫ 0.035 threshold`) and sometimes flung over the target (`reached`), so `eval_success` **manufactured**
lifts and places. The policy's real behaviour: barely lifts, never places. **The pick-place task is largely
unsolved**, not won. (Honest re-run on the fixed env was in flight 2026-06-30; existing checkpoint already evals
0.125/0.0 stable.)

**Root cause (two bugs).** (1) `PickPlaceEnv` (and `PlanarGraspEnv`) ran a coarse ~1–2e-3 s sub-step → the
position servo over-corrected on contact penetration → solver blow-up. (2) `eval_success` read `info["lifted"]`
after the physics had diverged, with no guard — a crash that *inflates* the metric, the worst kind.

**Fixes shipped (all non-CORE, 2026-06-30).**
- `PickPlaceEnv` + `PlanarGraspEnv`: shrink the sub-step to ≤5e-4 and raise the substep count so
  `control_dt = frame_skip*timestep` is **preserved exactly** → trained weights transfer with no re-train; no more
  blow-up (`qacc 1e4 -> ~1e2`).
- `eval_success`: blow-up guard (`_DIVERGE_QACC`; a diverged episode is a FAILURE, never a counted lift).
- `PlanarGraspEnv._clear_of_arms`: the coin no longer spawns inside an arm link (the other [[project-fast-rl-sanity-suite]] demo artifact Hajdu flagged — coin spawned on the arm, knocked to the zone = false delivery).
- New `hymeko_rl/rollout_monitor.py` (`RolloutMonitor`/`RolloutHealth`): watches **physics health** (qacc
  divergence + actuated-joint stall) and aborts dead rollouts early — complements the HTL task-logic monitor
  (`htl_reward`). Hajdu asked for this to "save time"; it would have caught this on day one.

**Re-check of the other MuJoCo results.** **Quad: CLEAN** (stable under random actions; the −84 vs −33 return is
real — see [[project-quadruped-collaboration-derisk]]). **Coin: same bug** (planar diverged at step 0 under random
actions) — fixed; the 0.25 delivery is suspect and was being re-evaluated. **Holonomy/rotor: unaffected** (pure
tensor task, no MuJoCo) — [[project-fast-rl-sanity-suite]] stands.

**The lesson (load-bearing).** A physics blow-up does not only crash — it can **silently inflate a success metric**
when the success flag is read from post-divergence state. NEVER trust a MuJoCo success number without a divergence
guard. Run `RolloutMonitor` on any new env/policy before believing a metric. Withdraw the pick-place claim from the
showcase until the fixed-env re-run lands. Credit: Hajdu caught it from the gripper exploding in the gif — the
demo-watch (§9 animated output) is what surfaced it.
