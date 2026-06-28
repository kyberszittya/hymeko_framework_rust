# Stage 1 — HyMeKo as the base of RL scenario description

2026-06-23 · hymeko_rl · plan `docs/plans/2026-06-23-rl-scenario-ablation-entropy/` (Stage 1)

## Summary
A whole RL problem is now described by one `.hymeko` *scenario* — scene geometry + references (by `@"…"`
import) to the robot and the reward — and the environment is built from it instead of a hand-written Python
`cfg` dict. This generalizes the parity-tested declarative reward (`RewardSpec.from_hymeko`) from *reward* to
the *whole scenario*. The gate (behavioral parity with the former `fanuc_pick_env` Python config) passes.

## Files touched (all non-core)
- **New** `data/robotics/meta_scenario.hymeko` (13 lines) — scenario vocabulary: a `scenario` namespace with a
  `@scene` kind (scene-parameter bundle).
- **New** `data/robotics/pick_place_scenario.hymeko` (28 lines) — the FANUC pick-and-place as one scenario:
  scene scalars/vectors + `@"arm_gripper_fanuc_import.hymeko"` (robot) + `@"pick_place_task.hymeko"` (reward).
- **New** `hymeko_rl/env/scenario.py` (+118) — `ScenarioSpec` (frozen dataclass) with `from_hymeko` (parse +
  validate + classify robot/reward imports by content) and `build(**overrides) → PickPlaceEnv`.
- **Edit** `hymeko_rl/env/_profile.py` (+33) — `read_scene_fields` (scalar/vector field parse) and
  `read_imports`, extending the existing `.hymeko` bridge-reader module.
- **New** `hymeko_rl/tests/test_scenario.py` (+52) — field/import parse + step-for-step env parity.

## CORE.YAML items touched
None. `hymeko_rl`, `data/` are non-core. The engine (`hymeko_core`/`parser`) was not modified.

## Test results
- `test_scenario.py`: 2/2 pass — (1) the scenario parses into the verified FANUC config (robot, reward,
  `obj_radius=(0.28,0.40)`, `target_xy=(0.34,0)`, `arm_home`, `max_steps=620`, …); (2) **the scenario-built env
  reproduces `fanuc_pick_env` step-for-step** on seed 3 + a fixed 60-action sequence (`np.allclose` obs every
  step, reward `|Δ|<1e-5`, identical termination). Reward parity is transitive via `test_pick_place_task`
  (declarative == procedural).
- Full pick/scenario/evil/ik suite: **18/18 pass** (~9.6 s, CPU-only). `ruff` clean; `mypy --strict` clean on
  the two changed source files.

## Performance
N/A (construction + a 60-step rollout; well under any budget). CPU-only.

## Known limitation (measured, honest)
The scenario `.hymeko` is **not yet `hymeko inspect`-clean**, by a *pre-existing engine behavior*, not a defect
in this code:
- **Measured:** `meta_scenario.hymeko`, `arm_gripper_fanuc_import.hymeko`, and `pick_place_task.hymeko` each
  `inspect` cleanly *on their own*.
- **Measured:** inspecting `pick_place_scenario.hymeko` fails with
  `UnresolvedRef { from "…arm_gripper", target "meta_kinematics.kinematics.elements" }`.
- **Inferred:** the engine resolves **one level** of `@"…"` imports. The scenario → robot composite →
  `meta_kinematics` chain is two levels, so the composite's own imports are not followed when it is itself
  imported. The Python bridge reader (regex, `_profile`-style) does follow transitively, which is why parity
  passes today.
- **Not done (correctly):** fixing this is **transitive-import support in the engine** — a `hymeko_core`/parser
  change, approval-gated (CORE.YAML §1). Not attempted. Documented as the Stage-1 follow-up.

This is the honest split: HyMeKo *is* the base of the scenario description and drives the env (parity proven);
full engine-side validation of the multi-reference instance awaits transitive imports.

## Anti-pattern / discovery notes
- No §6.5 anti-patterns introduced. The `.hymeko` bridge reader was *extended* (`read_scene_fields`,
  `read_imports` added to `_profile.py`), not duplicated; `ScenarioSpec` reuses the `RewardSpec.from_hymeko`
  idiom rather than a parallel parser.
- Robot/reward imports are classified by **content** (a reward profile declares `reward_spec`; a robot composite
  declares `kinematics`) rather than by filename — mirrors the compiler, avoids brittle name coupling.

## Follow-up
1. Engine: transitive (`>1`-level) import resolution → the scenario instance becomes `inspect`-clean (core,
   approval-gated). Until then, the bridge reader is the path.
2. Extend the scenario vocabulary with task-automaton + HTL-spec references (the FSM-structured-RL plan,
   `docs/plans/2026-06-23-fsm-structured-rl/`), so one scenario carries structure + obs + reward + spec.
3. Stage 2 (next): wire PPO into `compare_offpolicy`; HSiKAN/MLP × {PPO,DDPG,TD3,SAC} on arm/Galambos.
