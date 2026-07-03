# hymeko_rl — isolated package scaffold + the actor-critic unification (Phase 0, increment 1)

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-mujoco-rl-grasping](../docs/plans/2026-06-18-mujoco-rl-grasping/) (4 artifacts, PDF compiles).
**Status:** ✅ isolated package created, approved deps installed + verified, dep-free
core (the actor-critic unification) implemented + tested, MuJoCo stack smoke-clean. The
mujoco/gymnasium-dependent build (env, PPO, HSiKAN backbone) is the next increment.

## Summary
Honoured the user's directive to isolate the RL work in "another crate or folder": since
the policy/critic are HSiKAN/torch and the loop is Python, the idiomatic home is a
**folder** (a top-level Python package, importable from repo root like `hymeko_neuro`),
not a Rust crate. Created `hymeko_rl/` with the unification expressed *in code*: one
`ActorCritic` backbone feeds **both** the actor and critic heads, so the planned
architecture ablation (HSiKAN-on-hypergraph vs MLP) is a backbone swap under one shared
PPO. Installed the approved deps and proved the MuJoCo stack works on the existing 4-DOF
arm.

## The unification (four roles, one substrate)
| role | artifact |
|---|---|
| geometric description (kinematic hypergraph) | `env/arm_world.py` (`ARM_MJCF`, `load_arm`) |
| actor (policy head) | `policy.py` `ActorCritic.actor_mean` on the shared backbone |
| critic (value head) | `policy.py` `ActorCritic.critic` on the **same** backbone |
| agent description (the MDP as data) | `agent.py` `AgentSpec` (`.hymeko` loader pending) |

## Files touched
**New package `hymeko_rl/` (7 files, ~330 LOC):**
- `__init__.py` — package docstring (the four-role unification) + re-exports.
- `policy.py` (+130) — `ActorCritic` (shared-backbone diagonal-Gaussian actor-critic;
  `act` no-grad rollout, `evaluate` grad-enabled PPO update), `mlp_backbone`, and the
  `build_policy` Strategy with a backbone registry (`hsikan` registered, raises
  `NotImplementedError` until the bridge increment — a typo fails as `ValueError`, a
  pending feature distinguishably).
- `agent.py` (+45) — `AgentSpec` frozen dataclass (validated MDP description).
- `env/arm_world.py` (+75) — canonical 4-DOF arm MJCF + `load_arm()`.
- `env/__init__.py`, `tests/__init__.py`, `README.md`.
- `tests/test_policy.py` (+75), `tests/test_arm_world.py` (+30).

**CORE.YAML items touched:** none. **New dependencies:** `mujoco==3.9.0`,
`gymnasium==1.3.0` (+ transitive `glfw`, `pyopengl`, `farama-notifications`), installed
via `uv pip install` into `.venv` (manifest/lock untouched), under
`APPROVED-CORE-EDIT: mujoco-gymnasium-robot-rl`.

## Test results
- `hymeko_rl/tests/` **12 passed** (5.7 s, `pytest -p no:randomly`): actor-critic
  shapes + shared-backbone + grad/no-grad split; logprob consistency; `build_policy`
  errors (NotImplemented vs ValueError); backbone/dim validation; `AgentSpec`
  validation (4 invalid cases + 1 valid); **MuJoCo arm loads (nq==nu==4) and steps
  (responds to a setpoint, qpos finite)**.
- `ruff check`: clean. `mypy --strict` (policy.py, agent.py): clean (the distribution
  is typed `Any` — torch.distributions is only partially stubbed; one comment, no
  per-line ignores).

## §6.5 anti-patterns
None. The ablation is a backbone Strategy + registry (§6.5 #1/#9), not per-kind wrapper
functions; one `ActorCritic` serves actor+critic (no duplication); no globals (the
registry is an immutable module constant of pure callables — the narrow allowed
exception); errors are specific (`ValueError`/`NotImplementedError`, §6.4). The arm MJCF
is duplicated from `sim_mujoco_scenario.py` *transitionally* — `arm_world.py` is now the
canonical home and a follow-up should make the figure script import `ARM_MJCF` (noted in
the file).

## Performance
Trivial at this stage: 12 tests in 5.7 s; the arm smoke is 500 physics steps (~1 s sim)
on CPU. Peak RSS ≪ 16 GB. No benchmark claim (no training yet).

## Provenance
Git SHA `7d16ad0` (tree dirty). Device CPU (MuJoCo physics) + CUDA available for torch.
`mujoco 3.9.0`, `gymnasium 1.3.0`, torch 2.12.0+cu132. Seeds fixed in tests (0, 1).

## Open / next (the dependent build)
1. **Phase 0 bridge** — robot `.hymeko` → star-expansion → torch state tensor (reuse
   `demos/hero` / `demo_web/export_star_expansion.py`); implement `hsikan_backbone` and
   register it so `build_policy("hsikan", …)` works.
2. **Phase 1** — REACHING via behaviour cloning (MVP; no reward design).
3. **Phase 2** — GRASPING: gripper+object MJCF + Gymnasium `ArmEnv` + in-repo PPO; run
   the HSiKAN-vs-MLP ablation (≥3 seeds, random-policy floor, sample-efficiency +
   success + params).
4. **Phase 3** — Nagare/embedded deploy of the tiny policy; warm-start from the
   kinematics-from-topology prior.
