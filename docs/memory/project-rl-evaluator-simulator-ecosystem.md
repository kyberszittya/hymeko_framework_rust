---
name: project-rl-evaluator-simulator-ecosystem
description: "2026-07-01: built the unified RL evaluator + simulator ecosystem (consolidating 5 duplicate eval loops + scattered env factories) AND registered the best-known architecture per scenario. FUTURE WORK USES THESE — register a TaskSpec, eval via evaluate_task, read best_arch(name); do NOT rebuild eval loops / env helpers / eval_* copies."
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

After the RL push spawned a lot of near-duplicate drivers, a redundancy audit found **5 copies of the same
"reset → roll → count" eval loop**, 3 BC-collection scaffolds, 3 `_env()` copies. Consolidated 2026-07-01:

**The evaluator ecosystem — `hymeko_rl/evaluate.py`:** one `eval_metric(env, action_fn, metric, *, n_episodes,
seed0)` rollout + a `RolloutMetric` Protocol (`reset`/`on_step`→stop-bool/`finalize`) + `greedy_action_fn(ac)`.
Metric strategies: `FlagSeenMetric(key)`, `FinalScalarMetric(key)`, `StepCountMetric`, `ReturnMetric`,
`LiftPlaceMetric` (carries the 2026-06-30 divergence guard). All four `eval_*` (`eval_delivery`/`eval_success`/
`eval_reach`/`eval_balance`) are now 2-line WRAPPERS over `eval_metric` — behaviour-preserving (20 tests).

**The simulator ecosystem — `hymeko_rl/tasks.py`:** a `TaskSpec` registry. Each scenario declares
`make_env` + `metric` + `recommended` (an `ArchRecommendation`) + `default_steps`. Helpers: `get_task(name)`,
`evaluate_task(name, ac, n_episodes=)` (one-line eval through the declared metric), `best_arch(name)`,
`task_names()`, `register_task(spec)`.

**Registered best architecture per scenario** (`ArchRecommendation`: backbone, algorithm, head, confidence, basis
— honest provenance, CLAUDE.md):
- **cartpole** → `hsikan + ddpg + pooled` [measured] — learns upright; DDPG ~250× sample-eff vs PPO.
- **galambos** → `hsikan + bc_ppo + pooled` [measured] — BC teaches grasp, PPO refines; single ≥ CTDE (no collab win).
- **pick_place** → `hsikan + td3_bc + per_node` [**recommended**] — BC works 0.75–0.875; PPO COLLAPSES it (known
  offline→online problem, [[project-pick-place-explosion-artifact]]); TD3+BC holds at smoke; per_node head is the
  un-validated lever (3723× on structured targets) — NOT yet validated on this task, so confidence = recommended.
- **arm_reach** → `hsikan + bc + pooled` [measured] — HSiKAN ≈ MLP (serial arm, small gap); BC suffices.
- **quadruped** → `hsikan + ppo + pooled` [measured] — FLAT HSiKAN > 4-leg CTDE on goal-reach (−33 vs −84);
  gait (cyclic=holonomy) is the untested place CTDE+rotor might win ([[project-quadruped-collaboration-derisk]]).
- **quadruped_stand** → `sa_hsikan + td3 + pooled` [**recommended**] — NEW 2026-07-02: the Rung-2 **postural**
  plant (balance on a free base = it can fall). `task="stand"` on `QuadrupedGoalEnv` (STAND_REWARD =
  upright+torso_height+alive+stand_still; torso obs→[z−h, upright]; `info["standing"]`). Metric =
  `DwellMetric("standing", 200)` (upright-at-height held ≥200/250). This is where declared topology is
  hypothesised to PAY (postural/cyclic), unlike the non-cyclic goal-reach where flat HSiKAN wins — the balance
  test `exp_designed_control.py` flagged as "the task to build before this is the thesis test"; the stand plant
  is now registered there too ([[project-topology-control-matching-law]], [[project-structural-actor-walk-holonomy]]).
  Pure TD3 from the dense reward (no demonstrator); PD-hold-q0 BC→TD3+BC is the fallback lever. Built+tested
  2026-07-02 (13 tests). **RAN 3 seeds × 60k (CPU): stand-rate median 0.0 (seed0 0.0→0.24, seeds1-2 0.0) —
  standing IS learnable (seed0 existence result) but NOT robust at 60k (high seed variance = undertrained,
  not broken). survival near-saturated (203-248 even untrained; free base rarely inverts under flip_cos=-0.2) →
  stand_rate is THE discriminating metric.** Levers: 100k-200k steps / PD-hold-q0 BC warm-start / tighter
  flip_cos. GOTCHA: GPU `torch.compile` CUDAGraphs-crashes on the quadruped (`_critic_loss`, "tensor overwritten
  by subsequent run") — env verified clean, a GPU-compile bug not a training bug; runs on CPU (~28min/seed).
  `experiments/2026_07_02_06_41_quadruped_standing/`, report `reports/2026-07-02-quadruped-standing-scenario.md`.

**Graph-planning pre-trainer — `hymeko_rl/graph_planner.py`:** `astar(start, is_goal, neighbours, heuristic)`
(self-contained A*; the Rust `hymeko_graph::traversal_heuristic::astar` is not Python-bound) + `GridAStarPlanner`
(8-connected workspace grid, routes around obstacles, env-agnostic via `goal_xy`/`pos_xy`/`to_action`/`blocked`
adapters) + `pretrain_from_planner(policy, env, planner, ...)` (roll the explicit plan → demos → BC-clone). The
plan-then-amortise warm-start: clone the kinematic plan, then TD3+BC refines the contact (off-policy holds where
PPO collapsed). Honest limit: planner is kinematic (reach/transport), not grasp. 5 tests.

**Metric correction (2026-07-01):** `DwellMetric(key, min_consecutive)` — a delivery/success counts only if the
flag is HELD for N consecutive steps (mirrors the env's `success_steps` rule), not an instantaneous touch.
`eval_delivery` now uses it (was over-counting grazes). 2nd measurement artifact caught after the explosion-lift.

**Overnight optimization results (2026-07-01, corrected metric):** collaborative galambos 5-seed —
**single ≥ collab** (single median 0.167 stable; collab median 0.042, all-or-nothing variance) — NO collab win;
variance (not coordination) is the diagnosis → cross-agent propagation is the lever. Pick SA-HSiKAN
cross-propagation A/B: NEGATIVE (both 0.0; SA-HSiKAN backbone capacity is the bottleneck, not trunk-sharing →
per-node head is the lever). Quad off-policy TD3 from scratch: NEGATIVE (degrades −24.8→−86.3, Q-overestimation,
no demos → graph-planner demos+TD3+BC is the lever). Report `reports/2026-07-01-overnight-optimization.md`.

**RULE for future RL work (why this exists):** to add a robot/task, **register one `TaskSpec`** — do NOT write a
new `_env()`, a new `eval_*`, or copy a driver. To evaluate, `evaluate_task(name, ac)`. To pick an architecture,
read `best_arch(name)` first. The off-policy algorithms live as config presets in `ddpg.py`
(`td3_bc_config`/`adaptive_bc_config`, + `n_critics`=k-critic, `kind="mixture"`). Optimisation levers (survey):
SA-HSiKAN (fast, weak BC capacity here), `PerNodeActionHead` (the un-wired structured-readout lever),
VectorizedEnv (17× B=1). Still TODO: BC harness (`collect_demonstrations` unifying the 3 collect scaffolds) +
wire the `exp_*` drivers + `offpolicy_eval` through `tasks.py`.
