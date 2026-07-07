# Overnight optimization — collaborative, pick-place, quadruped (+ a metric correction)

**Date:** 2026-07-01 (overnight, unattended). **Context:** "grind and optimize"; priority order k-coin-toss
collab → pick-place → quadruped → humanoid → Niitsuma. Existing scenarios optimized autonomously; the new ones
(k-arm / humanoid / Niitsuma) planned for a supervised build (new envs are not safe to author unattended).

## Headline: a measurement correction + three honest results

### 0. Metric fix (the important one) — `DwellMetric`
The collaborative delivery metric counted an **instantaneous** `in_zone` touch as a delivery — a coin that grazed
the zone and rolled out still scored. Corrected: `eval_delivery` now uses `DwellMetric("in_zone", success_steps)`
— a delivery requires the coin to be **held** in the zone for the env's `success_steps` (mirroring the env's own
sustained-success rule). The loose-metric run was discarded; everything below is the corrected metric. (Tested;
this is the 2nd measurement artifact caught this stretch, after the explosion-inflated lift.)

### 1. Collaborative galambos — single ≥ collab (corrected, 5 seeds)
| | per-seed delivery | median | IQR |
|---|---|---|---|
| single HSiKAN (30 025 p) | 0.208, 0.125, 0.167, 0.167, 0.375 | **0.167** | [0.167, 0.208] |
| collab CTDE 2-arm (20 489 p) | 0.167, 0.25, 0.0, 0.0, 0.042 | **0.042** | [0.0, 0.167] |

**Verdict: no collaborative win.** And the real finding is **variance**: single delivers *reliably* (~0.17, tight
IQR); the 2-arm CTDE is **all-or-nothing** (one 0.25 win, two outright 0.0s). The CTDE doesn't *coordinate
reliably* — it occasionally lucks into a delivery. Plot: `reports/figures/collab_grind.png` (per-seed dots + IQR).
This is the case *for* cross-agent structural propagation (coordination as a shared B^L walk), the un-built lever.

### 2. Pick-place — SA-HSiKAN cross-propagation A/B: negative
SA-HSiKAN separate-trunk vs shared-cross-propagation trunk, TD3+BC, 30k steps: **both 0.0 / 0.0**, post-BC and
post-RL. SA-HSiKAN's BC is weak/variable (0.0–0.25) — **the bottleneck is backbone capacity, not trunk-sharing**;
the B^L collapse can't represent the pick task, and the value-gradient cross-propagation can't fix a backbone that
can't express it. (`eval_success` verified correct against `hsikan_s0` = 0.125/0.0 — the 0.0 is real, not a
migration bug.) The capable backbone (vanilla HSiKAN, BC 0.75) is too slow off-policy; **the un-pulled pick lever
is the per-node action head** (3723× on structured targets), which needs the `act_vertices` mapping — a supervised
morning task.

### 3. Quadruped — off-policy TD3 from scratch: negative
Return *degraded* over training: −24.8 (untrained) → −86.3 (final), curve `[-41, -68, -44, -72, -86]` — a textbook
off-policy Q-overestimation collapse with **no demos to anchor**. PPO (the registered best) stands. **Next lever:
graph-planner demos → TD3+BC** (the `graph_planner.pretrain_from_planner` built this stretch can generate the
kinematic quad demos the off-policy needs).

## Built this stretch (the infrastructure the grinds ran on)
- **Eval framework + simulator ecosystem** (`evaluate.eval_metric` + `RolloutMetric` strategies incl. the new
  `DwellMetric`; `tasks.py` registry with best-arch recommendations) — 5 eval-loop copies → 1.
- **Graph-planning pre-trainer** (`graph_planner.py`: A* + `GridAStarPlanner` + `pretrain_from_planner`).
- **Structural cross-propagation** (`build_offpolicy(shared_trunk=)` + DrQ-style split optimisers) — actor↔critic
  shared B^L trunk, tested (8 tests). The *cross-agent* version (CTDE joint-hypergraph trunk) is the next build.

## Honest scorecard + the next levers
Two negative optimizations + one sharpened prior — not the wins hoped for, but **real and pointed**:
- collab: variance is the problem → **cross-agent structural propagation**.
- pick: capacity is the problem → **per-node action head**.
- quad: no-demo collapse → **graph-planner demos + TD3+BC**.

## For the supervised morning (the new scenarios, in priority order)
1. **k-coin-toss collab** — k-arm planar env (the `CollaborativeGalambos` CTDE already handles k agents via
   `arm_action_partition`; the gap is a k-arm MJCF). + cross-agent propagation.
2. **humanoid** — backlog-gated; a new env.
3. **Niitsuma dual-UR5** — two 6-DOF arms (the ROS2 demo has the scenario; the RL env is new).
Each needs a new MuJoCo env → a smoke before any long run (why they waited for supervision).
