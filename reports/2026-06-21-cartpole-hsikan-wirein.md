# HyMeKo cart-pole — HSiKAN actor-critic wire-in (inverted pendulum)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** [docs/plans/2026-06-21-cartpole-hsikan-wirein/](../docs/plans/2026-06-21-cartpole-hsikan-wirein/)

## Summary
Wired the existing HSiKAN actor-critic (`build_policy("hsikan", …)`) to the canonical **inverted
pendulum** as a clean control benchmark that isolates the architecture from the hard Galambos grasp task.
Per the user's decision, the kinematic hypergraph (`hg_state`) comes from a **HyMeKo description** — one
source yields the MuJoCo scene *and* the 2-vertex (cart, pole) signed graph the HSiKAN backbone
message-passes over.

The HSiKAN policy **learns to balance**: a 120-iteration PPO smoke went from `init_return` 25.8 to
`final_return` 152.3, and mean upright-steps rose from an untrained floor of **27.8** to **144.0** (of
`max_steps` 200). The task discriminates (floor ≪ trained), so it is a real test of the actor-critic, not a
trivially-survivable one.

## What was built
- **`data/robotics/inverted_pendulum.hymeko`** (new) — world –slide(X)→ cart –hinge(Y)→ pole, on
  `meta_kinematics`. The same `.hymeko` is emitted to MJCF *and* read by `HypergraphState.from_mjcf`.
- **`hymeko_rl/env/arm_world.py`** — `strip_actuators(mjcf, joint_names)` added beside
  `with_collision_floor`/`slim_arm_collision` (same MJCF-transform family; no new module).
- **`hymeko_rl/env/inverted_pendulum_env.py`** (new) — `InvertedPendulumEnv` (a *class*, not a flag on
  `ArmReachEnv`; §6.5 #8): +1 alive reward, terminate on `|θ|>angle_limit` or `|x|>cart_limit`, per-vertex
  obs `(2, 2) = [qpos, qvel]`. Flattened, that is the classic `[x, ẋ, θ, θ̇]` — so the MLP baseline and
  HSiKAN read the same signal (a pure backbone swap).
- **`hymeko_rl/ppo.py`** — introduced a `RolloutEnv` **Protocol** and retyped `_collect`,
  `_warmup_critic`, `train_ppo` to it (annotation-only; §7 Strategy/contract made explicit). `ArmReachEnv`
  and `InvertedPendulumEnv` both satisfy it structurally — the *same* PPO loop trains either.
- **`hymeko_rl/train_inverted_pendulum.py`** (new) — `run_balance(...)` + CLI + `eval_balance`.

### Two emitter quirks, both confined to the env layer (not the `.hymeko`)
1. The emitter actuates **every** non-fixed joint; the canonical inverted pendulum is under-actuated, so
   `strip_actuators(mjcf, ["hinge"])` removes the pole motor → `nu = 1` (cart force only).
2. The MJCF emitter applies a deg→rad factor to *slide* joint ranges
   ([transforms.rs:619](../hymeko_formats/src/transforms.rs#L619)) — wrong for a translational cart. In
   practice the emitter wrote **no** slide range here (nothing to strip on the rail); the cart bound is
   enforced as a termination instead. Documented so a future limited-rail cart-pole does not get bitten.

`hymeko validate` emits one benign warning ("Joint 'rail' references unknown parent link 'world'") — `world`
is the frame root the emitter maps to the MuJoCo `<worldbody>`; the emitted MJCF is correct.

## Files touched
| File | Δ |
|---|---|
| `data/robotics/inverted_pendulum.hymeko` | new (+58) |
| `hymeko_rl/env/inverted_pendulum_env.py` | new (+126) |
| `hymeko_rl/train_inverted_pendulum.py` | new (+108) |
| `hymeko_rl/tests/test_inverted_pendulum_env.py` | new (+170) |
| `hymeko_rl/env/arm_world.py` | +24 (`strip_actuators` + `Iterable` import) |
| `hymeko_rl/ppo.py` | +18/−3 (`RolloutEnv` Protocol + 3 retypes) |

**CORE.YAML items touched:** none. **New/removed deps:** none.

## Test results
- **New suite** `test_inverted_pendulum_env.py` — **15 tests** (14 passed, 1 skipped: the RSS-budget branch,
  psutil unavailable — see Performance). Unit: `strip_actuators`, env shapes/`nu`/reset-uprightness/invalid
  params/pole-fall + cart-bound termination/alive-reward/task-non-triviality. Integration: both backbones
  forward + `train_ppo` runs end-to-end (parametrized hsikan/mlp). Perf: wall budget.
- **Regression** — `test_ppo.py`, `test_policy.py` (the `ppo.py` retype): **pass**. `test_arm_world.py`,
  `test_agent.py` (the `strip_actuators` addition): **8 passed**.
- **Coverage:** every new function/method is driven by a test; the `ppo.py` retype is guarded by the new
  `train_ppo`-on-`InvertedPendulumEnv` integration test (the regression that proves the contract is
  structural, not nominal).
- **Static:** `ruff check` clean; `mypy --strict` clean on the new modules and the retyped `ppo.py` apart
  from the **endemic** `mujoco` import-untyped (identical to `arm_reach_env.py`/`hypergraph_state.py` — not
  introduced here, and not suppressed to stay consistent with those neighbors).

## Performance results
- **Production-scale smoke (§3):** HSiKAN, 1 seed (0), 120 PPO iters × 1024 steps, real env.
  - `init_return` 25.8 → `final_return` 152.3; untrained upright-steps 27.8 → trained **144.0** / 200.
  - `n_params` 26 243; wall **452.7 s** (~3.8 s/iter, CPU).
- **Peak RSS:** **660 MB** (Win32 `GetProcessMemoryInfo` peak working set; dominated by the torch+mujoco
  import base) — far under the 16 GB cap.
- **Perf test (median-of-5, fresh policy/env per round):** one 512-step `train_ppo` iter, median < 5 s
  budget (measured ~2 s/iter); asserted as a regression alarm. The dominant cost is the single-environment
  rollout (batch-1 policy forwards), consistent with the Task-2 policy-efficiency analysis.
- **Measurement note:** `psutil` is not installed and `memray` (the pinned mem profiler) is Linux/macOS-only,
  so RSS was read via stdlib `ctypes` + the Win32 API (no dependency added; adding one is a §1 core change
  not taken). The pytest RSS assertion skips gracefully when `psutil` is absent.

## §6.5 anti-patterns
None introduced. `strip_actuators` joins the existing MJCF-transform family (no new module); the env is a
class for a structural variant (not a forward-time flag); the PPO loop's env dependency was lifted to a
Protocol (Strategy made explicit) rather than branching per env type.

## Experiment provenance
- Git SHA `292388b` (working tree dirty — the 10 files above).
- Env: Windows 11; Python 3.13 (`.venv`); torch 2.12.0+cu132 (CPU used), mujoco 3.9.0, gymnasium 1.3.0
  (CORE.YAML-pinned torch/numpy).
- Seeds: training seed 0; eval seeds 10 000 (floor) / 20 000 (trained).
- Source: `data/robotics/inverted_pendulum.hymeko` @ this SHA (dirty); emitted via `target/debug/hymeko.exe`.
- Smoke log: the run's stdout JSON (`init_return`/`final_return`/`upright_steps`/`wall_s`) captured in the
  session task output; no GPU used.

## Multi-seed ablation result (5 seeds, vectorized, 2026-06-21)
Artifact: `reports/2026-06-21-cartpole-multiseed.jsonl` (5 seeds × {HSiKAN, MLP}, vec N=16, 120 iters).

| policy | upright-steps mean ± sd | per-seed | learn rate |
|---|---|---|---|
| HSiKAN | **192.0 ± 15.3** | 161, 200, 199, 200, 200 | **5/5** |
| MLP | 98.6 ± **74.7** | 39, 38, 182, 36, 198 | **2/5** |

First read (WRONG, kept for honesty): HSiKAN converges every seed (sd 15) while the MLP is bimodal (2/5
learn) — looked like a structural-robustness win.

**Control overturned it — capacity, not structure** (`reports/2026-06-21-cartpole-controls.jsonl`):

| net | params | upright (5-seed) | learn |
|---|---|---|---|
| HSiKAN | 26 243 | 192.0 ± 15.3 | 5/5 |
| MLP (orig) | 9 091 | 98.6 ± 74.7 | 2/5 |
| **MLP matched** | **26 659** | **195.2 ± 8.4** | **5/5** |
| MLP over-param | 134 659 | 200.0 ± 0.0 | 5/5 |

A params-matched MLP ties HSiKAN (5/5, lower variance). The MLP "failure mode" was **under-parameterization**
(hidden=64), not absence of structure. So on the 2-vertex cart-pole the kinematic-hypergraph structure is
**not load-bearing** — capacity is. (Same verdict/trap as the 2026-06-18 rotor-vs-MLP-embed ablation: a
matched-capacity control closes the gap.) Caveat: a 2-vertex graph has no topology to exploit, so cart-pole
cannot test the structure hypothesis — a fair test needs the 6-DOF arm-reach or Galambos. The PPO *algorithm*
baseline (HSiKAN 192±15) stands; the *architecture* claim does not.

## Open issues / follow-ups
1. **Reward stays in code, not `.hymeko`.** The *structure* is from HyMeKo as asked; the upright/alive reward
   is canonical-IP code. Lifting it to `meta_reward` (an `upright`/`alive` term + a balancing `env_spec`) is
   the natural next unification step — deferred, not in scope.
2. **HSiKAN-vs-MLP ablation at parity.** A multi-seed `run_balance` comparison (the MLP baseline reached
   similar upright-steps in the 3-iter smoke) would quantify whether the structure-reading backbone helps on
   so small a graph (2 vertices) — likely not, and that is itself an honest finding worth recording.
3. **Editor view for the pendulum.** `inverted_pendulum.hymeko` renders in Hypergraph 3D (2 vertices, 2
   joint hyperedges) and in the kinematic/URDF view; not yet added to the editor's project gallery.
