# Status + next steps — proper robot generation & the Galambos planar grasp

**Date:** 2026-06-20 · **Branch:** soma-vision · **For:** Dr. Csaba Hajdu (end-of-session save)

## What landed this session (robot-generation focus)
1. **Emitter fix — HyMeKo can now describe *proper* robots** (`hymeko_formats/src/transforms.rs`,
   non-core). The MJCF emitter now emits the link's geometry `origin` as **both** the geom `pos`
   **and** the `<inertial>` COM. Before, every geom sat at the body origin (the joint) and the COM
   at `0 0 0` — so links were stubs at the joint (gaps) and mass was off-centre. Now a link describes
   a **rod spanning from its joint to the next** (connected arm), mass centred on the rod. Strictly a
   geometry improvement; the 212-test `hymeko_query` generation suite stays green.
2. **`galambos_planar.hymeko` — a proper top-down planar 2-arm robot, described in HyMeKo.** Z-axis
   hinges (the chain sweeps in the XY table plane), box-rod links with `origin` mid-points so they
   connect joint-to-joint, bases at x=±0.14 reaching forward; the workspace (coin + zone) is well
   inside reach.
3. **`hymeko_rl/env/verify_arm.py` — generate-and-verify proper arms.** `verify_arm(mjcf, ee_body)`
   → `ArmReport(loads, articulates, connected, reach_radius, max_link_gap)`. Articulation checks
   position **and orientation** (catches roll joints); connectivity flags floating/stub links; reach
   flags degenerate workspaces. **All three arms verify [OK]:** galambos (reach 0.46 m), reach_arm
   (0.88 m), anthropomorphic 6-DOF (1.77 m).
4. **`PlanarGraspEnv` rebuilt to the corrected task** (top-down table): the coin is **placed at a
   random reachable spot, not dropped**; arms pull it to a centre zone. Uses the **HyMeKo-emitted**
   arm by default (`robot=None` → hand-authored baseline). Obs = per-vertex hypergraph features +
   each link's **vector to the coin** + the coin→zone goal.
5. **Eval harness + scoreboards** (`hymeko_rl/evaluate.py`): goal/death/timeout tally + points,
   outcome plots, rendered GIF. Reach-arm scoreboard is meaningful (expert 17 goals / 3 deaths / 85%
   vs random all timeouts).

**107 hymeko_rl tests green; ruff clean; mypy only the `mujoco` baseline; no core edits.**

## Honest status of the Galambos policy
The env is correct (connected, reachable, contact verified) but **PPO does not yet solve the pull**:
100–150 iters beats random on points (≈ −15 vs −31) but scores **0 goals** — it doesn't complete
catch-and-pull. Contact-rich, underactuated; needs much more training + shaping. Not claimed as solved.

## NEXT STEPS (priority order)
1. **Train Galambos to actual goals.** Longer PPO via `python -m hymeko_rl.train_planar_grasp`
   (+ a curriculum: spawn the coin near the zone first, then widen; tune the `both_contact` /
   `in_zone` weights in `galambos_task.hymeko`). Report the return curve + scoreboard honestly.
2. **URDF/SDF emitters: apply the same `origin`→geom-pos/COM fix** (`hymeko_formats/src/{urdf,sdf}.rs`).
   Currently only MJCF is fixed; URDF/SDF still emit geoms at the body origin → the same stub/gap
   defect for those formats. Do this so *all* outputs describe proper robots.
3. **6-DOF reach safety scoreboard is all-deaths** — the expert self-collides during the *reaching
   motion* (not just home). Slim the 6-DOF collision geoms further / add a contact-penetration
   threshold to `compute_safety`, so that scoreboard becomes meaningful.
4. **Generalise `bc._make_policy`** to size the policy from `env.observation_space` (it hardcodes
   `env.obs_spec`); then the planar env uses it directly (the runner currently calls `build_policy`).
5. **Surface verify_arm** as a `hymeko` CLI verb (e.g. `hymeko verify <robot.hymeko> --ee <body>`) or
   a generation-suite gate, so "generate + verify proper robot" is one command.
6. **Commit the rest of the session** — the viewer, reach-safety penalties, j3 axis fix, and scene
   beautifier are tested but uncommitted (see their reports). The cross-profile feature is committed
   (`73ee5a6`); this robot-generation unit is being committed now.

## Key files
- `hymeko_formats/src/transforms.rs` (emitter fix), `data/robotics/galambos_planar.hymeko`,
  `data/robotics/galambos_task.hymeko`, `hymeko_rl/env/{planar_grasp_env,verify_arm,evaluate}.py`,
  `hymeko_rl/train_planar_grasp.py`, `hymeko_rl/tests/test_{planar_grasp_env,verify_arm,agent}.py`.
- Reports: this file, `2026-06-20-galambos-topdown-correction.md`, `2026-06-20-simulation-scoreboards.md`.

## Provenance
- Pre-commit SHA `73ee5a6`; MuJoCo 3.9.0, torch per CORE pins, matplotlib 3.11. Windows 11, CPU.
