# Tutorial — one HyMeKo model → structure + state + action, learned with HSiKAN

A runnable walkthrough of the Kato-collaboration grasping line. The thesis in one sentence:

> **A single HyMeKo model is compiled into the robot's kinematic structure, the hypergraph the policy
> observes as *state*, and the actuator interface it *acts* through — and the policy (HSiKAN) reads that
> same compiled structure. Structure, perception, and control share one declarative source.**

Everything below is reproducible from the repo root. Commands assume the `hymeko` CLI is built
(`cargo build -p hymeko_cli`) and the Python env has `mujoco` + `gymnasium`.

---

## 0. The picture

```
                         ┌───────────────────────── one .hymeko model ─────────────────────────┐
 data/robotics/*.hymeko ─┤  links + joints (kinematics)   +   [obs profile]   +   [reward]       │
                         └───────┬───────────────────────────┬────────────────────────┬─────────┘
                                 │ emit                       │ read                    │ read
                                 ▼                            ▼                         ▼
                        MJCF + kinematic            per-vertex STATE on        scalar REWARD
                        hypergraph (structure)      the hypergraph             (declarative)
                                 │                            │                         │
                                 └──────────────► HSiKAN actor-critic ◄────────────────┘
                                       (message-passes over the SAME hypergraph)
                                                      │  action = joint targets + grip
                                                      ▼
                                                   MuJoCo
```

The same hypergraph is the robot's body, the policy's observation domain, and (via its joints) the action
interface. The novelty is not "an RL grasp" — it is that **all three come from one declarative model**, and
the network is shaped by that structure rather than reading a flat vector.

---

## 1. One model → three outputs

The robot is `data/robotics/arm_gripper_fanuc_import.hymeko`: it `@`-imports a 6-DOF arm with the **FANUC LR
Mate joint-rotation configuration** (`fanuc_lrmate.hymeko`, axes `Z Y Y Z Y Z` = base yaw + shoulder/elbow
pitch + a `Z-Y-Z` spherical wrist) and attaches a parallel-jaw gripper to the imported `arm.tool` link via
HyMeKo **cross-model kinematic composition**.

Emit the MJCF (the *kinematic structure*) straight from the model:

```bash
target/release/hymeko emit -f mjcf data/robotics/arm_gripper_fanuc_import.hymeko -n fanuc
```

…or, in Python, get all three outputs from the one source:

```python
import mujoco
from hymeko_rl.env.arm_world import emit_arm_mjcf
from hymeko_rl.hypergraph_state import HypergraphState

mjcf = emit_arm_mjcf("data/robotics/arm_gripper_fanuc_import.hymeko", name="fanuc", control_mode="position")
model = mujoco.MjModel.from_xml_string(mjcf)         # (1) kinematic structure (MJCF)
hg    = HypergraphState.from_mjcf(mjcf, is_path=False)  # (2) the kinematic hypergraph (state domain)
print(hg.n_vertices, model.nu)                        # 9 link-vertices, 8 actuators (6 arm + 2 grip)
#                                                       (3) the action interface = the emitted joints/actuators
```

The 9 vertices are `base_link, link_0..4, tool, finger_l, finger_r`; the incidences are the joints. This one
graph is reused everywhere below.

> **Why the FANUC config matters.** The earlier anthropomorphic arm (axes `Z X X Z X Z`, fat links) could only
> point its tool straight down by folding onto itself → self-collision, so no collision-free top-down grasp
> existed at any radius (`reports/2026-06-22-pick-place-phase0.md`). The spherical `Z-Y-Z` wrist + slim links
> point the tool down *without* folding → collision-free top-down grasps at r ∈ [0.20, 0.40], no pedestal.

---

## 2. The environment — state and action on the structure

`hymeko_rl/env/pick_place_env.py::PickPlaceEnv` wraps the emitted robot into a scene (table + a free-joint box
+ a target zone) and exposes the MDP:

- **State** — `node_features()` returns `(9, 8)`: for each hypergraph vertex, `[qpos, qvel, x, y, z,
  (object − vertex)(3)]`. The observation is *per-vertex on the hypergraph*, not a flat vector.
- **Action** — `[6 arm joint targets, grip]`; the dimension and bounds come from the emitted joints.
- **Reward** — dense approach → grasp → lift → place shaping (Python here; declarative `.hymeko` for the reach
  task — see §6).

```python
from hymeko_rl.render_pick_place import fanuc_pick_env
env = fanuc_pick_env()
obs, info = env.reset(seed=5)
print(obs.shape, env.action_space.shape)   # (9, 8)  (7,)
```

---

## 3. The policy — HSiKAN reads the same hypergraph

`hymeko_rl/policy.py::build_policy` builds an `ActorCritic` whose backbone is **HSiKAN** — signed-hypergraph
message passing `CR(W_self·h + W₊·A⁺h + W₋·A⁻h)` over the robot's hypergraph (`CR` = the learnable Catmull-Rom
KAN activation). Actor and critic are two heads; swap the backbone to a params-matched MLP for the ablation
(*fix the algorithm, ablate the architecture*).

```python
from hymeko_rl.gripper_pick_bc import build
ac = build("hsikan", env, hidden=64)   # or build("mlp", ...) for the baseline
```

The point: the policy is *structured by the same graph* the robot and state came from.

---

## 4. A scripted expert picks (the loop closes)

`PickPlaceEnv.expert_action` is a closed-loop demonstrator: a converged damped-least-squares pose-IK
(`hymeko_rl/env/ik.py::DampedPoseIK`) drives the tool top-down through reach → descend → grasp → lift →
transport → place. Render it:

```bash
python -m hymeko_rl.render_pick_place --seed 5     # → reports/gifs/fanuc_pick.gif
```

Reliability over the random workspace: **grasp 10/10, lift 9/10, place 9/10**. This is the "a controller on the
HyMeKo-derived structure performs the task" evidence (`reports/2026-06-23-fanuc-lrmate-top-down-grasp.md`).

---

## 5. Learning a policy — BC, then PPO (honest)

**Behaviour cloning** (`python -m hymeko_rl.pick_place_bc --kind hsikan`) fits the demos almost perfectly
(loss ≈ 3.6e-4) **but rolls out to 0%** — classic *BC compounding*: tiny per-step errors accumulate over the
~600-step horizon and the arm drifts off the demo states. (The floating-gripper BC *did* work — short horizon,
direct-Cartesian — see `hymeko_rl/gripper_pick_bc.py`.)

**PPO is on-policy**, so it trains on the policy's *own* states and attacks that compounding. From-scratch PPO
on a 7-DOF arm cannot stumble on a grasp, so it is **warm-started from the BC clone**:

```bash
python -m hymeko_rl.pick_place_ppo --kind hsikan --iters 70   # ~30 min; checkpoint + return curve
```

Result so far: return **−32 → +83**, and the greedy policy **learns to approach + contact the box (7/8)** — a
real gain over BC. **Full grasp-and-lift has not yet converged** at 143k steps (the firm-grip-then-lift is the
hard-exploration step). So: clear learning on the structure; a *reliably picking learned* policy needs more
budget + grasp-lift reward shaping. Curve: `reports/figures/fanuc_ppo_return.png`.

> **Scope, honestly:** this is a POC — *structurally-derived robot + imitation/RL control*, not "solved
> manipulation." The reliable pick today is the scripted expert; the learned policy is converging.

---

## 6. The fully-declarative MDP (proof on reaching)

On the **reach** scenario the *observation and reward are themselves HyMeKo*, not Python:

```python
from hymeko_rl.env.arm_reach_env import ArmReachEnv
env = ArmReachEnv.from_hymeko(
    "data/robotics/anthropomorphic_arm.hymeko",
    obs_profile="data/robotics/arm_reach_observation.hymeko",
    task_profile="data/robotics/arm_reach_task.hymeko")   # robot + obs + reward, all from .hymeko
```

This is the end-to-end form of the thesis — robot, state, *and* reward from one declarative source. Moving the
pick-and-place obs/reward onto the same footing (`pick_place_task.hymeko`) is the next step.

---

## 7. Reproduce everything

| What | Command | Artifact |
|---|---|---|
| Emit the arm+gripper | `hymeko emit -f mjcf data/robotics/arm_gripper_fanuc_import.hymeko -n fanuc` | MJCF on stdout |
| Scripted pick GIF | `python -m hymeko_rl.render_pick_place --seed 5` | `reports/gifs/fanuc_pick.gif` |
| Behaviour cloning | `python -m hymeko_rl.pick_place_bc --kind hsikan` | JSON (lift/place rates) |
| PPO (BC warm-start) | `python -m hymeko_rl.pick_place_ppo --kind hsikan --iters 70` | `checkpoints/fanuc_pick_ppo_hsikan.pt`, `reports/figures/fanuc_ppo_return.png` |
| Tests | `pytest -p no:randomly hymeko_rl/tests/test_fanuc_pick.py hymeko_rl/tests/test_ik.py` | 7 pass |

**Key files:** `data/robotics/{fanuc_lrmate,arm_gripper_fanuc_import}.hymeko` ·
`hymeko_rl/env/{pick_place_env,ik,arm_world}.py` · `hymeko_rl/{policy,pick_place_bc,pick_place_ppo}.py` ·
`hymeko_rl/render_pick_place.py`. Reports: `reports/2026-06-23-fanuc-lrmate-top-down-grasp.md`,
`reports/2026-06-22-pick-place-phase0.md`.
