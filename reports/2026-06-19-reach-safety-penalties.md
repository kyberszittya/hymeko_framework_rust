# Report — safety + configuration penalties for the reaching MDP (declared in the HyMeKo model)

**Date:** 2026-06-19 · **Plan:** `docs/plans/2026-06-19-reach-safety-penalties/` (4 artifacts, compiles)
**Author:** Aiko (agent), for Dr.\ Csaba Hajdu
**Status:** ✅ **Built, gated, tested** (95 `hymeko_rl` tests green). The initial spurious-self-collision
caveat on the 6-DOF arm is **resolved** in the same session (slim collision + parent/child exclusion
+ compile-time floor seating), and **termination is now model-declared** — see "Follow-ups completed".

## What was built
- **`hymeko_rl/env/safety.py`** (NEW) — `SafetyState(ground_contact, self_collision, joint_margin,
  below_ground)` + `compute_safety(model, data, floor_geom)` (pure, unit-testable). Relies on
  MuJoCo's default `filterparent`, so any non-floor contact is a genuine non-adjacent self-collision.
- **`arm_reach_env.py`** — opt-in `enable_safety` flag (default **off**): when on, injects a ground
  plane, computes `SafetyState` each step, and **terminates on ground-contact ∨ self-collision**
  (death). `reset` now reject-samples the target `≥ reach_min_radius` from the base axis (target
  **outside the robot**); `info` carries `safety`/`death`.
- **`reward.py`** — 4 new term kinds + extractors reading the env's `SafetyState`:
  `ground_penalty`, `self_collision_penalty` (death indicators), `joint_limit_penalty`
  (`-(1-margin)²`), `below_ground_penalty`. Inert (0) on a clean/safety-off env.
- **`meta_reward.hymeko`** — the 4 new `reward_term` kinds declared (default weight 0).
- **`arm_reach_safe_task.hymeko`** (NEW) — the reach reward + penalties with **bounded** weights
  (ground/self-collision 5.0, joint-limit 0.5, below-ground 1.0). `arm_reach_task.hymeko` (plain
  reach, used by the BC comparison) left untouched.
- **`arm_world.py`** — `with_collision_floor(mjcf)` (idempotent; never adds a second floor).
- **`_profile.py`** — **bug fix:** read `.hymeko` as UTF-8, not the OS locale (cp1250 on Windows
  choked on a non-ASCII byte). Benefits every profile read.
- **`render_reach.py`** — `build_render_env` now sets `reach_min_radius=0.2`, so the viewer/render
  show a target **outside the robot** (independent of the death machinery — keeps the window watchable).

## Design decisions honored (your calls, 2026-06-19)
- **Death = ground contact AND self-collision** → terminate + penalty.
- **Soft penalties** = joint-limit proximity + below-ground.
- **Bounded penalty** (weight 5, not `−∞`): an unbounded terminal spike corrupted the shared HSiKAN
  critic in the Phase 2 PPO debug; bounded + termination is stable. The magnitude is a `.hymeko`
  weight, tunable without code.
- **In the model:** rewards are declarative now; **termination-as-`constraint`-hyperedge** (the
  `<isa> constraint` vocab exists) is staged as the next step for "the whole MDP, including failure,
  from one `.hymeko`".

## ⚠️ Caveat — spurious self-collision on the 6-DOF emitted arm
The plan flagged this risk; it is **real**. The emitted `anthropomorphic_arm`'s links are fat
cylinders (radius 0.075) that overlap on **non-adjacent** links in ordinary poses, so
`self_collision` fires immediately → with `enable_safety=True` the arm "dies" on step 1. The
hand-authored 4-DOF `arm_world` arm (thin links) is clean (a unit test confirms a clean home pose).
**Consequence:** the safe reward is usable on `arm_world` now, but **not** on the 6-DOF arm until
its collision geometry is slimmed (or `compute_safety` gains a contact-penetration threshold /
explicit exclude pairs). Documented, not hidden — this is a geometry fix, a follow-up, not a bug in
the machinery. The viewer/render therefore keep `enable_safety=False` (target-outside still applies).

## Tests
- `pytest -p no:randomly hymeko_rl/tests/` — **91 passed**, 116 s. New `test_arm_reach_safety.py`
  (9): safe task parses 5 weighted terms; plain task unchanged (regression); exactly one floor;
  clean home pose; joint_margin drops at a limit; reward terms map a planted `SafetyState` to the
  right signs (and are inert when clean); **safety off by default → no death**; driving max torque
  into the floor **terminates as death with a bounded penalty**; target sampled outside the robot.
- **Regressions fixed:** `test_emitted_arm_expert_reaches` (was dying on spurious self-collision —
  fixed by the opt-in gate) and `test_emitted_mjcf_is_articulated` (asserted the old `0 1 0` axis —
  a test **missed in the earlier j3 axis fix**; corrected to `0 0 1`).
- **Static:** `ruff` clean; `mypy --strict` on the changed env files — only the `mujoco`
  import-untyped baseline, no new errors.

## CORE.YAML / dependencies
**None.** `hymeko_rl/` + `data/robotics/` are non-core; no new dependency. (The `meta_reward.hymeko`
additions are additive term-kind declarations, not a schema change.)

## §6.5 anti-patterns
None. Reward terms are a Strategy registry (`_REWARD_TERMS`), not per-penalty wrappers; `SafetyState`
is a config struct; `enable_safety` is a parametric flag (the penalties are parametric, not a
structural variant); discovery confirmed no contact/safety code existed before. No `v2` files.

## Performance
Contact scan is O(ncon)/step; negligible. Not benchmarked (machinery, not a hot path). The PPO
retrain under the safe reward is a separate, separately-budgeted follow-up (and is blocked on the
6-DOF geometry caveat above, or runs on `arm_world` now).

## Follow-ups completed (same session)
1. **6-DOF self-collision resolved (was the blocker).** Three fixes: (a) `slim_arm_collision`
   shrinks the arm's cylinder radii (×0.4) when safety is on; (b) `compute_safety` now **excludes
   parent/child and same-body geom pairs** explicitly (the emitted scene does *not* reliably honour
   `filterparent`, so adjacent links touching at their joint were counted as self-collision — the
   real cause); (c) the ground plane is **seated at compile time** just below the arm's home extent
   via `_arm_home_floor_z` (the emitted lower link clips `z=0` at its mount; runtime `geom_pos`
   does **not** move a plane's collision surface — verified). Result: the 6-DOF arm's home pose is
   contact-free and zero instant deaths over reset seeds (test: `…_safe_home_is_clean`).
2. **Termination-as-constraint in the model.** New `meta_task.hymeko` `termination` vocab +
   `termination_spec` bundle; `hymeko_rl/env/termination.py::TerminationSpec` (kind→predicate
   Strategy, `from_hymeko`); `arm_reach_safe_task.hymeko` declares `@dies_when: termination_spec
   (+ on_ground, + on_selfcol)`; `ArmReachEnv` reads it (`from_hymeko`), falling back to
   `DEATH_ON_CONTACT`. So the death *predicate* now lives in the `.hymeko`, beside the reward —
   "the whole MDP, including failure, from one source."

## Follow-up (still open)
- **PPO retrain** under `arm_reach_safe_task` (now viable on the 6-DOF arm; `arm_world` also fine).

## Additional files (follow-up work)
- `hymeko_rl/env/termination.py` (NEW), `meta_task.hymeko` (+`termination` vocab),
  `arm_reach_safe_task.hymeko` (+`termination_spec`), `arm_world.py` (`slim_arm_collision`,
  `with_collision_floor(z=)`), `arm_reach_env.py` (`_arm_home_floor_z`, `termination_spec` wiring,
  `enable_safety` seats+slims), `safety.py` (parent/child exclusion). Tests: 95 green (4 new).

## Provenance
- Git SHA `7d16ad0` (working tree dirty; `hymeko_rl` uncommitted increment). MuJoCo 3.9.0,
  Python 3.12, Windows 11. Seeds: tests 0/1; reject-sampling 0–9.
