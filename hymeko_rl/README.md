# hymeko_rl — a learned control head on the tensorised HyMeKo hypergraph

Isolated package for the Kato-collaboration grasping POC (plan:
[`docs/plans/2026-06-18-mujoco-rl-grasping/`](../docs/plans/2026-06-18-mujoco-rl-grasping/)).
Deps `mujoco` + `gymnasium` added under `APPROVED-CORE-EDIT: mujoco-gymnasium-robot-rl`.

## The unification (why this package exists)

One HyMeKo hypergraph substrate carries **four roles**, processed by the *same*
HSiKAN/Gömb machinery:

| role | what it is | where |
|---|---|---|
| **geometric description** | the robot's kinematic hypergraph (the scene) | `env/arm_world.py` |
| **actor** | policy head on a HSiKAN/Gömb backbone reading that hypergraph | `policy.py` (`ActorCritic`) |
| **critic** | value head on the **same shared backbone** | `policy.py` (`ActorCritic`) |
| **agent description** | the MDP (obs/action/reward) as a `.hymeko` profile | `agent.py` (`AgentSpec`) |

`ActorCritic` is literally one backbone with two heads, so the architecture ablation —
**HSiKAN-on-hypergraph vs an MLP baseline** — is a *backbone swap* under one shared PPO:
**fix the algorithm, ablate the architecture** (the codebase's house style, cf. the
rotor-vs-MLP signed-link ablation).

## Status (2026-06-18, increment 1 of Phase 0)

Done — the isolated package, the deps, the dep-free core, and the stack smoke:
- `policy.py` — `ActorCritic` (shared-backbone Gaussian actor-critic) + `mlp_backbone`
  + `build_policy` Strategy (`hsikan` registered, raises until wired).
- `agent.py` — `AgentSpec` (the MDP-as-data; `.hymeko` loader pending).
- `env/arm_world.py` — the canonical 4-DOF arm MJCF + `load_arm()`.
- `tests/` — 12 tests (actor-critic contract, agent-spec validation, MuJoCo
  arm-loads-and-steps smoke). `mujoco 3.9.0`, `gymnasium 1.3.0`.

Next (staged in the plan):
1. **Phase 0 bridge** — robot `.hymeko` → star-expansion → torch state tensor
   (reuse `demos/hero` / `demo_web/export_star_expansion.py`); wire the HSiKAN/Gömb
   backbone (`build_policy("hsikan", …)`).
2. **Phase 1** — REACHING via behaviour cloning (MVP, no reward).
3. **Phase 2** — GRASPING: gripper+object MJCF + Gymnasium env + the in-repo PPO,
   running the **HSiKAN-vs-MLP architecture ablation** (≥3 seeds, random-policy floor).
4. **Phase 3** — deploy the tiny policy via the Nagare/embedded line; warm-start the
   encoder from the kinematics-from-topology prior.

## Run

```
python -m pytest -p no:randomly hymeko_rl/tests/
```
