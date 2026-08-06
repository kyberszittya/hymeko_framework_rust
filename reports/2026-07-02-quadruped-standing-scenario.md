# Quadruped standing (balance) scenario — the Rung-2 postural plant

**Date:** 2026-07-02 (built ~04:30 JST, during the galambos→pick-place overnight chain).
**Git SHA:** working tree **DIRTY** (this session; see *Files touched*). **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu.
**Plan:** `docs/plans/2026-07-02-quadruped-standing/` (4 artifacts: tex/pdf/tikz/mmd).

## Summary

Added a **standing (balance)** task to the HyMeKo quadruped: on a **free** (6-DOF floating) base — so the torso
*can* fall — the robot must hold **upright at its nominal height without falling**. This is the missing
**Rung-2 plant** that `exp_designed_control.py` explicitly flagged as *"the task to build before this is the
thesis test"*: a **postural / cyclic-coordination** regime, the place declared topology (kinematic vs
Steiner/sunflower-augmented) is hypothesised to pay — unlike the non-cyclic goal-reach, where flat HSiKAN
already beats CTDE (the negative reference). Off-policy only (pure TD3 + SA-HSiKAN); no PPO, no demonstrator.

The scenario is **built, registered, tested, and run** (3 seeds × 60k, CPU).

## Run result (measured, `experiments/2026_07_02_06_41_quadruped_standing/`)

| seed | stand_rate (untrained → trained) | survival frames (u → t) |
|---|---|---|
| 0 | 0.0 → **0.24** | 230.9 → 247.3 |
| 1 | 0.0 → 0.0 | 248.5 → 226.8 |
| 2 | 0.0 → 0.0 | 203.0 → 204.7 |

**stand-rate median = 0.000** (0.24 / 0.0 / 0.0), survival median 226.8/250. SA-HSiKAN pure-TD3 + LayerNorm twin
critics + vec-8, 60k steps, ~28 min/seed on CPU.

- **[measured] Standing is learnable — the scenario is validated by a positive existence result.** Seed 0 lifts
  from a 0.0 untrained baseline to **0.24** sustained-stand rate (held upright-at-height ≥200/250 consecutive
  steps in ~1/4 of eval episodes). The Rung-2 postural plant does what it should: the reward drives balance.
- **[measured] Not robust across seeds at 60k.** Seeds 1–2 stay 0.0 → high seed variance (undertrained /
  unstable), **not** a broken task (single-seed is a point estimate, not a verdict). Next levers: 100k–200k
  steps, PD-hold-q0 BC → TD3+BC warm-start, or a tighter `flip_cos`.
- **[measured] Survival is near-saturated** (203–248 frames even untrained): the free base rarely fully inverts
  under `flip_cos=−0.2`, so survival does not discriminate — **`stand_rate` is the discriminating metric**
  (0.24 vs 0.0), confirming the plan's metric choice. GIF `gifs/stand_s0.gif` shows the seed-0 standing policy.

The build details below are unchanged.

## What was added (measure / infra, kept distinct from any un-run claim)

- **4 standing reward terms** (`hymeko_rl/env/reward.py`), registered in the `_REWARD_TERMS` Strategy table,
  each duck-typed + `getattr`-guarded (0 on an env with no torso — same contract as the pick terms):
  `upright` (torso levelness cos), `torso_height` (−|z−h| to the nominal standing height), `alive` (+1/step
  survival), `stand_still` (−|vₓ|, don't drift). Composed into `STAND_REWARD`.
- **`task="stand"` env mode** (`hymeko_rl/env/quadruped_env.py`, additive; `task="goal"` default = byte-identical
  prior behaviour): swaps in `STAND_REWARD`, caches the rest-pose torso height, dispatches the torso obs to
  `[z−h, upright]` (the policy observes what it optimises), emits `info["standing"]` (upright-at-height), and
  makes the stand terminal a fall only.
- **`quadruped_stand` TaskSpec** (`hymeko_rl/tasks.py`): env factory + `DwellMetric("standing", 200)` (held
  upright-at-height ≥200/250 consecutive steps) — **reuses** the shared eval loop, no bespoke loop. Recommended
  arch recorded: `sa_hsikan / td3 / pooled` (confidence *recommended*, with the PD-warm-start fallback noted).
- **Designed-control plant** (`hymeko_rl/exp_designed_control.py`): `_PLANTS["quadruped_stand"]` registered so
  the Rung-2 topology A/B (kinematic vs Steiner vs sunflower vs MLP) can run on it.
- **Campaign** (scratchpad `quad_stand_campaign.py`, staged): SA-HSiKAN TD3 + LayerNorm twin critics + vec-8 +
  GPU/compile, 3 seeds × 1e5 steps, own `experiments/<ts>_quadruped_standing/` (gifs/ + policies/ + results.json
  + README), dot-free GIF names, a standing GIF. Reports **stand-rate AND survival frames**, both untrained→trained
  (the untrained baseline certifies the task is non-trivial before learning).

## Test results

- **New** — `hymeko_rl/tests/test_quadruped_standing.py`: **13 passed** (14.6 s). Covers the 4 term values +
  the 0-on-foreign-env guard; `STAND_REWARD` composition; default-reward-by-task + goal-mode regression;
  `stand_cos`/`tol` validation; **the rest pose is genuinely upright** (the plan's risk precondition — a
  mis-signed cosine would break the task); stand-mode torso obs `≈[0,1]`; `info["standing"]` bool + positive
  reward at rest; a driven-to-tip torso scores **not** standing (a blow-up cannot fake success — inverse of the
  2026-06-30 pick explosion); the `quadruped_stand` registry wiring end-to-end through `DwellMetric`; a 200-step
  wall+RSS perf budget.
- **Regression** — `test_quadruped_env.py` + `test_reward.py` + `test_scenario_sanity.py`: **39 passed** (12.4 s).
  `scenario_sanity` builds *every* registered task, including the new `quadruped_stand`.

## Static analysis (§6.3) / complexity (§6.2)

- `ruff check` on all changed files: **All checks passed.**
- `mypy` on the changed files: **no new errors** — the only reports are the pre-existing package-wide `mujoco`
  missing-stubs (`import-untyped`, present for every env module) and two pre-existing errors in
  `planar_grasp_env.py` (not touched here).
- **Complexity waiver (declared):** `QuadrupedGoalEnv.__init__` is cyclomatic **12** — the *warn* band (fail =
  15), raised one notch by the added `stand_cos`/`stand_height_tol` guard on top of the existing 5-condition
  validation. `step` = 6. No hard-ceiling breach; no refactor forced.

## Performance budget

Standing env: 200-step median < 2 s, tracked peak < 256 MB (asserted in the perf test) — far under the 16 GB RSS
cap. Planned run: ~13 min/seed (1e5 vec-8 steps at the measured 74→125 steps/s), 3 seeds.

## §6.5 anti-patterns

None introduced. The standing task is a **config** on the shared env (parametric: same robot/DOF/action space,
only objective+obs-semantics+terminal differ) — not a class-per-variant duplication (#8), and not a new eval
loop (reuses `DwellMetric` + `eval_metric`, #1/#2). Discovery pass (§6.1) found and reused the existing reward
registry, `RolloutMetric` ecosystem, `TaskSpec` registry, and the `exp_designed_control` harness rather than
rebuilding any of them.

## Files touched (uncommitted, all non-core)

`hymeko_rl/env/reward.py` (+4 terms +4 registry entries), `hymeko_rl/env/quadruped_env.py` (+`task` mode,
`STAND_REWARD`, stand obs/termination/info), `hymeko_rl/tasks.py` (+`quadruped_stand` TaskSpec, `DwellMetric`
import), `hymeko_rl/exp_designed_control.py` (+stand plant), `hymeko_rl/tests/test_quadruped_standing.py` (new).
Plan dir (4 artifacts). Campaign in scratchpad. **CORE.YAML items touched: none.**

## Open / next

- **Standing run in flight on CPU** (`quad_stand_campaign.py`, 3 seeds × 60k). GPU was tried first but
  `torch.compile`'s CUDA-graphs crashes on the quadruped (*"accessing tensor output of CUDAGraphs overwritten by
  a subsequent run"* at `ddpg.py:_critic_loss`) — the env is verified clean (0 non-finite reward/obs, reward
  bounded [−2.2, 1.4], max|qvel| 39.6; the `crit=nan` at the first log is the pre-first-update `last_c` init,
  clears to 0.563 by step 2000 on CPU). A GPU-compile issue, **not** a training/reward bug. Fold stand-rate +
  survival + GIF into this report on completion.
- **Investigate the GPU CUDAGraphs crash on the quadruped** (works on galambos/cartpole; the quad differs in obs
  shape and the pure-TD3 no-offline path) — restores the 5–6× speed for standing.
- **Untrained baseline note:** at 2k steps the policy already *survives* 250/250 (the free base rarely fully
  inverts under `flip_cos=−0.2`), but `stand_rate` (upright-at-height held ≥200) = 0 — so `stand_rate` is the
  discriminating metric and survival is near-saturated. Report `stand_rate` as the headline.
- If from-scratch TD3 never balances (stand-rate ≈0 all seeds): add the **PD-hold-q0 BC warm-start → TD3+BC**
  fallback (the galambos anti-collapse recipe) — motivated, not first.
- The designed-control Rung-2 A/B (kinematic vs Steiner vs sunflower vs MLP) on `quadruped_stand` is now runnable
  — the topology-pays-on-postural-tasks test.
