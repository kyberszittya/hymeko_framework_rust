# FANUC LR Mate-config arm — solving the top-down grasp (Phase 1, path A)

**Date:** 2026-06-23 · for Dr. Csaba Hajdu · Kato collaboration
**Follows:** `reports/2026-06-22-pick-place-phase0.md` (which proved the anthropomorphic arm *structurally*
could not grasp top-down: it self-collided to point the tool down, at every radius).

## Summary
The earlier diagnosis was that the anthropomorphic arm's link geometry + Z-X-Z wrist made a collision-free
top-down grasp impossible (kinematically valid down-poses always self-collided — `link_0↔link_3`,
`link_3↔tool`). Per the user's direction, this work (a) re-authored the arm with the **FANUC LR Mate 200iD
joint-rotation configuration** and (b) slimmed the collision geometry. The structural wall is gone: the FANUC
arm reaches **collision-free top-down grasps at r ∈ [0.20, 0.40] (down-ness 1.00), with no pedestal**, and the
scripted expert **grasps 8/8 and lifts the box** (seed 2: 25 cm — see `reports/gifs/fanuc_pick.gif`).

## What was built
- **`data/robotics/fanuc_lrmate.hymeko`** — 6R arm, FANUC LR Mate axis sequence **j0=Z, j1=Y, j2=Y, j3=Z,
  j4=Y, jtool=Z**: base yaw + shoulder/elbow pitch about a common horizontal axis + a **Z-Y-Z spherical wrist**
  (roll-pitch-roll, near-intersecting axes). LR Mate 200iD-scale (reach ~0.7 m). Collision geometry = **slim
  cylinders** (r ≈ 0.03–0.045 vs the old 0.075) — the capsule equivalent (the meta layer has no capsule), so
  non-adjacent links no longer false-collide when the wrist bends.
- **`data/robotics/arm_gripper_fanuc_import.hymeko`** — imports `fanuc_lrmate.hymeko` and attaches the gripper
  fingers to `arm.tool` (the same cross-model composition as `arm_gripper_import.hymeko`). Joint/tool names
  match the env, so `PickPlaceEnv(robot=...)` reuses everything.
- **`hymeko_rl/env/ik.py` — `DampedPoseIK`** (built path-A): iterative DLS pose-IK (position + tool→down),
  converges on a scratch `MjData`; `solve_collision_free` is a multi-start variant with a caller collision
  predicate (used for the structural probe; the FANUC reaches collision-free on a single warm solve, so the
  closed-loop expert uses the smooth single solve to stay on one IK branch).
- **`hymeko_rl/env/pick_place_env.py`** — now parameterised: `mount_height`, `obj_radius`, `arm_home`
  (a bent non-singular ready posture — the zero/vertical home is an IK singularity). Control stack to make the
  arm trackable + grasp: **gravity compensation** (`body_gravcomp`) + moderate gain (kp=45 arm); firm grip
  (kp=250) + finger friction; a **grip-settle dwell** before lifting; **straight-up lift then transport**
  (no shear); a phase-dependent rate limit (fast approach 0.4, gentle carry 0.18). `_ik_pose_valid` rejects
  self/floor collisions.
- **`hymeko_rl/render_pick_place.py`** — runnable demo: `python -m hymeko_rl.render_pick_place --seed 2`
  renders the FANUC pick to a GIF. `fanuc_pick_env(**overrides)` is the shared, reusable config factory.

## Key measurements
- **Collision-free top-down reachability (FANUC, no pedestal):** r=0.20/0.30/0.40 → YES (tool z hit, down-ness
  1.00, collision-free); r=0.50 → no (reach limit). Contrast: anthropomorphic arm = **no** clean top-down pose
  at any radius.
- **Scripted expert (8 seeds, r∈[0.28,0.40]):** grasp **8/8**; lift (>6 cm) **2/8** (seed 2: 25 cm, seed 6:
  15 cm). The box rises with the tool on the successful seeds (held, contact maintained).

## Reliability pass (2026-06-23, follow-up) — expert now does the FULL pick-and-place
The first cut was ~2/8 on a full lift. Diagnosed and fixed each failure mode (each isolated with a trace):
- **Box knocked on descent** → open the fingers PAST rest during approach/descent (`_OPEN = -0.014`, the grip
  joints are unlimited) for wide clearance over the box.
- **Lift drifted sideways, no rise** → the straight-up lift was chasing the *current* (drifting) tool xy;
  capture the grasp xy ONCE (`_lift_xy`) and lift from that fixed point.
- **Grip slip-equilibrium under fast lift** → committed-phase rate 0.28 (not 0.4) gives a uniform, holdable
  lift; heavier box (`box_mass=0.15`) resists knock and is well within grip capacity.

**Result over 10 seeds: grasp 10/10, lift 9/10, place 9/10** — the arm grasps the box top-down, lifts it, and
transports it into the target zone, reliably. New env params: `box_mass`, `obj_angle`, plus the wide-open /
fixed-lift-xy control. Test `test_fanuc_expert_grasps_lifts_and_places` asserts the full pick-and-place.

## BC port (path A) — fits the demos, but the rollout hits the compounding wall
Ported the floating-gripper BC machinery to the arm (generalised `gripper_pick_bc` helpers to a `PickEnv`
protocol + `only_success` demo filter; new `hymeko_rl/pick_place_bc.py`; runnable `python -m
hymeko_rl.pick_place_bc`). **The clone FITS the demos near-perfectly** (BC loss 3.6e-4; its t=0 action matches
the expert to <1°) **but picks 0%** — *and 0% even in-distribution*. So this is **not** a fit failure: it is
**BC compounding / covariate shift** over the 620-step horizon (tiny per-step errors accumulate, the arm
drifts off the demo states, the policy cascades), the same wall the earlier 6-DOF arm BC hit
([[project-kato-collaboration-grasping]]). The expert's hidden phase state (grip-settle counter, captured
grasp xy) — not in the obs — compounds it. **Known fix: DAgger or PPO (on-policy), or a Markovian expert +
task-phase obs features — not more BC.** The floating-gripper BC worked precisely because it was short-horizon
+ direct-Cartesian.

## PPO (path A, learned policy) — BC warm-start + on-policy; partial
On-policy PPO trains on the policy's *own* states, so it directly attacks the BC compounding. From-scratch PPO
on a 7-DOF arm cannot stumble on a grasp, so PPO is **warm-started from the BC clone** (`hymeko_rl/pick_place_ppo.py`,
runnable; reuses the generic `train_ppo`). 70 iters / ~143k env steps, HSiKAN, BC warm-start (16 demos),
value-warmup 5, ent 2e-3; ~32 min wall; checkpoint `checkpoints/fanuc_pick_ppo_hsikan.pt`; curve
`reports/figures/fanuc_ppo_return.png`.
- **Return climbed −32 → +83** (init 29 from the BC warm-start), noisy but clearly upward.
- **Greedy eval: the policy LEARNED to approach + make finger-box contact (7/8), min-approach ~0.12 m** — a
  real gain over BC (which rolled out to nothing). **But it only nudges the box (max_lift ~1.2 cm); full
  grasp-and-lift did NOT converge** at this budget. The grasp-lift is the hard-exploration step (contact earns
  a small reward; the firm-grip-then-lift sequence is rarely discovered).
- So PPO is the right tool and shows clear learning on the HyMeKo-derived structure, but a *reliably picking*
  learned policy needs more — more steps, grasp/lift reward shaping (or a grasp-success curriculum), and the
  HSiKAN-vs-MLP ablation once a policy actually picks.

## Curriculum PPO (2026-06-23 follow-up) — return climbs, greedy place still 0
Added a runtime `set_difficulty` knob + curriculum hook (ramp difficulty over PPO iters) + the "evil" env
generator (`hymeko_rl/evil_pick.py`). Curriculum PPO (CPU-only, BC-warm-started, 70 iters): **return −321 →
+93 at difficulty 0.87, final −89** — the curriculum mechanism works (return rises as the object is pushed to
the reach edge). **But the greedy policy places 0% at every difficulty (0.0–1.0)** — same wall as the plain
PPO: it learns approach+contact (dense reward up) but the full grasp-lift-place does not converge on-policy.
Honest conclusion: on-policy PPO (even curriculum + warm-start) does not crack the full long-horizon
manipulation at this budget → the learned-control comparison needs **off-policy (DDPG/SAC, ~250× sample-eff)**,
per the ablation plan `docs/plans/2026-06-23-hsikan-ablations/`. Checkpoint `checkpoints/fanuc_pick_curriculum_hsikan.pt`.
The scripted expert remains the reliable pick (10/10 grasp, 9–10/10 place).

## Still open
- Reward + observation are still **Python**, not `.hymeko` (Phase 2 = `pick_place_task.hymeko`).
- A *reliably picking* learned policy (more PPO budget + grasp-lift reward shaping / curriculum), then the
  HSiKAN-vs-MLP backbone ablation. The scripted expert (10/10 grasp, 9/10 place) remains the reliable pick.

## Tests (all pass)
- `hymeko_rl/tests/test_fanuc_pick.py` (3): env builds + (9,8) contract; **collision-free top-down pose exists**
  (the regression the old arm failed); expert grasps + lifts >8 cm on seed 2.
- `hymeko_rl/tests/test_ik.py` (4): solver position convergence, down-orientation, joint limits, multi-start.
- `test_pick_place_env.py` (4) still green (additive params, defaults unchanged). 15/15 across the suite.
- ruff + mypy --strict clean on all changed modules.

## Files touched
NEW: `data/robotics/fanuc_lrmate.hymeko`, `data/robotics/arm_gripper_fanuc_import.hymeko`,
`hymeko_rl/env/ik.py`, `hymeko_rl/render_pick_place.py`, `hymeko_rl/tests/test_fanuc_pick.py`,
`hymeko_rl/tests/test_ik.py`, `reports/gifs/fanuc_pick.gif`.
MODIFIED: `hymeko_rl/env/pick_place_env.py` (IK, gravity comp, pedestal/obj_radius/arm_home params, grasp
control). CORE.YAML: none. No §6.5 anti-patterns (the IK was *de*-duplicated into `ik.py`; the FANUC config
reuses the env + gripper-import unchanged).

## Next
- Get a *learned* FANUC pick: **DAgger** (relabel the policy's own rollout states with the IK expert) or
  **PPO** on this env — BC alone is compounding-bound. Then the HSiKAN-vs-MLP backbone ablation has a real
  learned policy to compare.
- Phase 2: reward + obs as `pick_place_task.hymeko` (the declarative MDP) — also gives the policy task-phase
  features that would reduce the expert's non-Markovian gap.

## Files touched (follow-up)
NEW: `hymeko_rl/pick_place_bc.py`. MODIFIED: `hymeko_rl/env/pick_place_env.py` (box_mass/obj_angle/n_actions
params, wide-open descent `_OPEN`, fixed `_lift_xy`, committed rate 0.28), `hymeko_rl/render_pick_place.py`
(reliable config), `hymeko_rl/gripper_pick_bc.py` (generalised to `PickEnv` protocol + `only_success`),
`hymeko_rl/tests/test_fanuc_pick.py`. CORE.YAML: none. No §6.5 anti-patterns (BC helpers de-duplicated into the
shared protocol, not copied).
