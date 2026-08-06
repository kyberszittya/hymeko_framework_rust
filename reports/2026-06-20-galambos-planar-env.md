# Report — Galambos planar two-finger grasping env

**Date:** 2026-06-20 · **Plan:** `docs/plans/2026-06-19-galambos-planar-grasping/` (4 artifacts)
**Status:** ✅ **Env built, tested, PPO-trainable.** Full training (to a learned pull) is the next step.

## Summary
Galambos-sensei's planar "hello world": two planar 2-link arms (the `galambos_planar.hymeko`
hypergraph) pull a disk into a target zone. Built the MuJoCo scene, the task metrics, the
declarative reward, and a Gymnasium env whose policy reads the **two-arm kinematic hypergraph**
(6 vertices) — the distinctive HyMeKo claim. A 2-iter PPO smoke trains without NaN (6.7 s).

## What was built
- **`hymeko_rl/env/planar_grasp_env.py`** (NEW):
  - `compose_planar_scene` — injects a **planar** disk (slide-x / slide-z / hinge-y, so it stays in
    the XZ plane; verified ~0 out-of-plane drift), a non-colliding target-zone `site`, and a ground
    floor into the emitted two-arm MJCF. Appended after the arms so arm body indices stay aligned
    with the arm-only hypergraph (the disk is **not** a hypergraph vertex).
  - `compute_planar_metrics` / `PlanarGraspMetrics` — disk position, disk-to-zone, per-finger
    contact (scans disk contacts by left/right body), `in_zone` (within zone radius **and** rested
    on the floor).
  - `PlanarGraspEnv` — obs = per-vertex `(6, 6)` on the two-arm hypergraph (`[qpos, qvel, disk_x,
    disk_z, disk_x−zone_x, disk_to_zone]`); action = 4 arm joint targets (disk unactuated); reward
    from the declarative spec; disk **spawned outside the zone** (so the arms must pull it in, not a
    trivial drop-in); success-termination when the disk holds in-zone.
- **`data/robotics/galambos_task.hymeko`** (NEW) — the reward spec: dense pull (reuses
  `reach_distance` with `disk_to_zone` as the distance) + `both_contact` (0.5) + `in_zone` (+10
  sparse success) + `action_cost`.
- **`meta_reward.hymeko`** — `+ @both_contact`, `+ @in_zone` term kinds.
- **`reward.py`** — `both_contact` / `in_zone` extractors (read `env._planar_metrics`, 0 on a
  non-grasp env); `RewardSpec.evaluate` env type loosened to `Any` (terms are duck-typed, now serve
  both `ArmReachEnv` and `PlanarGraspEnv`).
- `galambos_planar.hymeko` already existed (the de-risk artifact).

## Tests
- **`tests/test_planar_grasp_env.py`** (5): reward kinds registered; `galambos_task` parses (4 terms);
  scene is planar (`nu=4`, `nq=7`, one floor, zone site, ~0 out-of-plane drift); env shapes `(6,6)`
  + disk spawns outside the zone over 8 seeds; a disk planted at the zone centre settles in-zone and
  the episode terminates as success.
- **Full suite:** `pytest hymeko_rl/tests/` **102 passed**. `ruff` clean; `mypy --strict` only the
  `mujoco` baseline.
- **PPO smoke (§3):** `train_ppo` on the env (HSiKAN policy over the hypergraph, 2 iters, 256 steps)
  → finite returns `[-14.2, -14.4]`, 6.7 s, no NaN. Confirms the env is trainable by the in-repo PPO.

## CORE.YAML / dependencies
**None.** All in `hymeko_rl/` + `data/robotics/` (non-core); no new dependency.

## §6.5 anti-patterns
None. The dense pull **reuses** `reach_distance` (no new "distance" term); metrics/reward are
Strategy-style; discovery confirmed no planar/two-finger env existed before. Reward is declarative
(`meta_reward` kinds + a task profile), consistent with the reach line.

## Open / follow-up
1. **Committed PPO runner + full training.** `bc.py::_make_policy` sizes from `env.obs_spec`
   (reach-specific); a planar runner needs it generalised to size from `observation_space` (the
   smoke sized the policy directly via `build_policy`). Then a real multi-hundred-iter PPO run to a
   learned pull, reported with the return curve (no over-claiming — measured, per §3).
2. **Render** the planar grasp (the viewer/`render_reach` are scene-agnostic; point them at the
   composed scene).
3. **Reward tuning** — spawn range vs zone size, contact-bonus shaping, success dwell — once training
   runs.

## Provenance
- Git SHA `73ee5a6` (working tree dirty; uncommitted increment). MuJoCo 3.9.0, torch per CORE pins.
  Windows 11, CPU. Seeds 0–7 (tests), 0 (PPO smoke).
