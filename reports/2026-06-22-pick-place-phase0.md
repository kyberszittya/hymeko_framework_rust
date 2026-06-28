# Pick-and-place grasping — Phase 0 (HyMeKo gripper + liftable object + contact)

**Date:** 2026-06-22 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu · Kato collaboration
**Plan:** `docs/plans/2026-06-22-pick-and-place-grasping/` (Ph0)

## Summary
The first manipulation scene with an **actuated gripper that closes** on a **3-D object that lifts** — the #1
gap (the prior envs only *pull* a planar coin). **Properly decomposed**: the environment reuses the *same*
`data/robotics/anthropomorphic_arm.hymeko` as the reach scenario (6-DOF arm), attaches a two-finger gripper at
its `tool` body, and adds a table + freejoint box. So the scene is **a robotic arm + a gripper + an object +
a plane**, and the policy reads per-vertex features on the **arm+gripper** kinematic hypergraph
(`HypergraphState.from_mjcf`, 9 vertices) — same contract as every other scenario.

## What was built
- **`data/robotics/arm_gripper_import.hymeko` — arm AND gripper as ONE HyMeKo source, NO duplication.** It
  ``@``-imports `anthropomorphic_arm.hymeko`, does `using robot as arm`, and attaches the gripper fingers +
  prismatic `grip_l`/`grip_r` joints to the **imported** `arm.tool` — i.e. **HyMeKo cross-model kinematic
  attachment works** (this updates the earlier note that it "awaited" support — it is supported). `hymeko emit`
  produces the full arm+gripper MJCF (`nbody=10`, joints `j0..jtool, grip_l, grip_r`, 9-vtx hypergraph; fingers
  verified children of `tool`), validated in MuJoCo. `PickPlaceEnv` emits *this* — the arm kinematics are not
  restated and the gripper is not spliced; fully HyMeKo, no duplication.
- `data/robotics/arm_gripper.hymeko` — the self-contained equivalent (restates the arm), kept as a fallback.
- `data/robotics/gripper.hymeko` — the standalone gripper (used by the `GripperPickEnv` testbed).
- `hymeko_rl/env/gripper_world.py` — `compose_pick_place_scene` injects table + box + target; `attach_gripper`
  (MJCF-splice) + `make_gripper_mjcf` remain as alternatives.
- `hymeko_rl/env/pick_place_env.py` — `PickPlaceEnv`: reuses the 6-DOF arm + gripper; node-feature obs
  `(9, 8)` on the arm+gripper hypergraph; action `[6 arm joint targets, grip]` (grip mapped to the two fingers
  oppositely); dense approach→grasp→lift→place reward; body-based finger-object contact.
- `hymeko_rl/tests/test_pick_place_env.py` — 4 tests (arm-emit + 9-vtx hypergraph + shapes, step contract,
  **graspable: box between closed fingers → both contact**, arg validation). All pass; ruff + mypy clean.

## Two real bugs found and fixed (worth recording)
1. **Emitted geoms are not named after the links** — a geom-name lookup (`mj_name2id(GEOM, "finger_l")`)
   returned −1, so (a) contact detection compared against the wrong geom and (b) the gantry-collision-disable
   clobbered geom −1 = the *object*, turning off its collision. Fix: everything is **body-based**
   (`geom_bodyid`), as `compute_planar_metrics` already does.
2. **Gantry self-collision** — the structural carriage/palm boxes overlap and spuriously self-collide,
   jamming `slide_y`/`slide_z`. Fix: disable collision on the carriage/palm bodies (fingers/object/floor
   still collide).

## Acceptance + honest scope
Phase-0 acceptance is that the env is **graspable** — at a closed-around-the-box pose, both fingers contact
the object (geometry + detection correct). Achieving that pose via **control** (open-loop or learned) is
**Phase 1 (BC)**: an isolation test showed the position-servo gantry tracks targets only slowly/partially in
the step budget, which is exactly the covariate-shift/control problem BC + curriculum address — not a Phase-0
concern. So Phase 0 delivers the *correct, HyMeKo-described, graspable* env; control is next.

## Files touched
NEW: `data/robotics/gripper.hymeko`, `hymeko_rl/env/gripper_world.py`, `hymeko_rl/env/pick_place_env.py`,
`hymeko_rl/tests/test_pick_place_env.py`, `scripts/dev/_cr_kernel_bench.py` (earlier). CORE.YAML: none.

## Phase 1 — expert → BC → learned grasp (path B: floating-gripper de-risk)
The 6-DOF **arm** grasp expert proved hard (top-down pose-IK drifts to the wrong side / degenerate poses, and
the arm base at table level fights top-down reach) — a real IK+geometry effort, deferred. Per the chosen path
B, the **expert→BC→learned-grasp loop was proven first on a floating gripper** (the gantry *is* the palm
position → direct Cartesian control, no IK), to be ported onto the arm once its IK/pedestal is sorted.
- `hymeko_rl/env/gripper_pick_env.py` — `GripperPickEnv`: direct-Cartesian floating gripper (same
  `gripper.hymeko` / contact / obs machinery), action `[x, y, z, grip]`, plus a trivial staged
  `expert_action` (above→descend→close→lift→carry→place).
- **Expert works:** grasps 6/6, lifts+places 5/6 seeds.
- `hymeko_rl/gripper_pick_bc.py` — collect expert demos, **reuse `bc.behaviour_clone`**, eval grasp success.
  **BC works:** HSiKAN goes **0% → 75% lift+place** (untrained floor 0).
- `hymeko_rl/tests/test_gripper_pick.py` — 3 tests (env+shapes, expert lifts ≥3/4, **BC ≥25% beating the 0
  floor**). All pass; ruff + mypy clean.

So the imitation pipeline is validated end-to-end.

## Arm grasp diagnostic — is the robot/axes wrong? (falsification, 2026-06-23)
The arm's scripted grasp is hard, so three "the robot is wrong" hypotheses were tested by Monte-Carlo FK
workspace probes (20k–30k random configs), **all refuted**:
- **Axis configuration wrong** — built an axis variant (shoulder/elbow/wrist pitching about Y instead of the
  `X`+90°-twist): top-down reach **1.3%→1.5%**, and *worse* in the object-radius zone (0.39%→0.21%). Not it.
- **Mount too low (needs pedestal)** — top-down reach at the table, object-radius, vs pedestal height
  0.0–0.6 m: **flat at ~0.1–0.23%**. Raising the base does not help. Not it.
- **Object too close to the base** — top-down reach binned by radius **peaks at r 0.15–0.25 m (2.44%)**, which
  *overlaps* the object spawn band (r 0.10–0.18). The object is at a good radius, not a dead zone. Not it.

**Conclusion:** top-down poses are only ~2% of config space at *every* radius, and the tool *does* reach down
(tip-z to −0.67 at the object radius); a scripted search *found* a valid down-pose at the object (r=0.14,
z=0.069, tool pointing down). Pointing a 6-DOF tool straight down is an inherently narrow orientation
constraint for **any** arm — that is what IK is for. So the blocker is purely the **IK controller** (+
grasp-height tuning), **not** the robot geometry or axes. (Measured; refuted artifact `arm_v2.hymeko` removed.)

## Path A — proper IK + control, and a deeper structural finding (2026-06-23)
Built the real control stack and isolated the true blocker — which **vindicates the "robot is wrong" instinct**,
just one layer deeper than raw kinematics:
- **`hymeko_rl/env/ik.py` — `DampedPoseIK`**, a reusable *iterative* DLS pose-IK solver (position + optional
  tool→down orientation) that converges on a scratch `MjData` (the prior inline single-step DLS in
  `pick_place_env`/`arm_reach_env` drifted). Tested: hits an FK-image target to <1 cm; `down=True` orients the
  tool downward. Plus `solve_collision_free` (multi-start + caller collision predicate).
- **Control fixes in `PickPlaceEnv`:** the emitter's kp=40 servo can't hold the heavy arm (sags); brute
  stiffness destabilises the integrator (NaN QACC). Fix = **gravity compensation** (`body_gravcomp`) + moderate
  gain — benign configs now track to ~0°. Added a **pedestal mount** (`mount_height=0.45`).
- **The wall (measured, structural):** even with IK + tracking + pedestal, the arm **cannot reach a
  collision-free top-down grasp at the object**. At r=0.25 a *kinematically perfect* down-pose exists (tool
  z=0.15, down-ness 1.00) but **self-collides** (`link_0↔link_3`, `link_3↔tool`); at the spawn radius r=0.14,
  **no collision-free pose reaches the position at all** (120 multi-start seeds). The fat-cylinder links
  (radius 7.5 cm) must fold onto themselves to point the tool straight down — a structural property of this
  arm's link geometry + Z-X-Z wrist, not a tuning gap. So the earlier FK-only probes (axes/pedestal/radius
  "refuted") were *incomplete*: with **self-collision** accounted for, the robot genuinely is ill-suited to
  compact top-down grasping. Aiko's correction to her own earlier conclusion.

**Realistic next options (none is a quick tune):** (a) **side/tilted grasp** strategy (don't force tool-down);
(b) **thinner collision geometry** (capsules) so non-adjacent links don't false-collide when folded;
(c) keep the **floating gripper** as the working pick demo (it sidesteps exactly this). Then Phase 2 (obs/reward
as `.hymeko` + staged FSM) and Phase 3 (HSiKAN-vs-MLP on the contact-cycle topology) ride on whichever grasps.
