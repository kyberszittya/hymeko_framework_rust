# R11.6A — Reward-Driven Delivery RL (v1): TD3 destabilizes the warm-start

**Date:** 2026-08-04 (Mac)
**Verdict:** `R11_6A_RL_UNSTABLE` (corrected classifier) — TD3 **destabilizes** the warm-start (drift + critic collapse).
The run *emitted* `R11_6A_ACTION_COORDINATE_INSUFFICIENT` because the original classifier had no instability category; the
fix distinguishes "degraded the warm-start" (RL_UNSTABLE) from "couldn't beat it" (ACTION_COORDINATE). See Diagnosis.
**Robot-positive:** the BC warm-start — a learned, amortized, **CEM-free** policy — delivers **93% of certified grasps
(train)** at strict K6.

## Result (44 train / 7 dev, 3 TD3 seeds, expl 0.1, teacher-BC warm-start)
| | train K6 | dev K6 | safe |
|---|---|---|---|
| box-center θ | — | 0.00 | 1.0 |
| **BC warm-start** (256-wide actor, fit 44 teacher θ) | **0.932** | 0.286 | 1.0 |
| seed 0 (best_val) | 0.932 | 0.286 | 1.0 |
| seed 1 (best_val) | 0.136 | 0.429 | 1.0 |
| seed 2 (best_val) | 0.295 | 0.571 | 1.0 |
| **mean (RL)** | **0.454** | 0.429 | 1.0 |

Gate: mean train 0.454 < 0.60 (fails); mean dev 0.429 (below 0.50 but > box-center 0 and > the R11.4B BC 0.25);
0 safety regression throughout.

## Diagnosis — TD3 drifts off the warm-start (training curves)
The instant updates begin (after warmup 40), train K6 **collapses** and the critic value crashes:
- **seed 0:** warm-start 0.932 → it120 **0.30** → it240 **0.025** → it480 **0.0** (Q ≈ −13 … −10), partial recovery to ~0.26.
  `best_val` kept **update0** (the warm-start, dev 0.286) because RL never beat it on dev.
- **seeds 1–2:** `best_val` selected **drifted** checkpoints — dev 0.429 / 0.571 but train **collapsed to 0.14 / 0.30**.
  The dev "gains" are **anti-correlated with train** → artifacts of selecting on only 7 dev scenarios, not learning.

**Root cause:** no immutable positive anchor. The replay buffer fills with warmup-random + drifted on-policy θ (mostly
K6-misses) → the critic values the whole θ region negatively (Q ≈ −13) → the actor is pushed *off* the memorized
warm-start → train craters. This is precisely the no-anchor failure the RL-infra map flagged (and the R10.2 lineage: a
critic that doesn't rank the good region highest). The apparent dev improvement across seeds is selection noise on 7
scenarios, correlated with — not independent of — train degradation.

**Verdict-label caveat:** the classifier fired `ACTION_COORDINATE_INSUFFICIENT` (mean train 0.454 < 0.60 ≤ warm-start
0.932). That category means "the coordinate can't beat the teacher"; here TD3 *destabilized* an already-good warm-start.
v2 adds an explicit `RL_UNSTABLE` sub-classification (train collapses below warm-start with a Q-value crash).

## Correction to R11.4B (honest)
R11.4B (`BC_REPRESENTATION_INSUFFICIENT`) reported smooth regressors fitting only 0.386 on train with a 64-wide MLP. The
256-wide RL actor here BC-fits train at **0.932** — so R11.4B's "can't even fit train" was **capacity-limited**, not a hard
representation wall. However, held-out generalization is still ~0.29 (≈ R11.4B's 0.25), so R11.4B's *core* finding holds:
the descriptor→θ map is hard to **generalize** from 44 demos (train is memorizable; held-out is not). The two are consistent
once capacity is separated from coverage.

## What this means for the robot (eyes on the robot)
A learned, amortized policy delivers **93% of certified-grasp scenarios (train) at strict K6 with no per-instance CEM and
no oracle** — the first deployable no-search delivery policy on the training distribution. It does not yet generalize to
unseen coin/target (dev 0.29 — that is R11.6B). RL (v1) did not improve on the BC policy; it destabilized it.

## Next — v2, not per-step RL, not more demos
1. **Immutable positive replay (the fix):** seed the buffer with the 44 teacher (state, z, reward) transitions, never
   evicted, mixed ~25% into every minibatch, so the critic keeps the known-good region high-valued and TD3 improves
   *from* the warm-start instead of forgetting it. Reduce/skip warmup random. (Reuse the `torque_path_td3` positive-replay
   pattern WITHOUT its ranking gate.)
2. **Robust selection:** select on train+dev (or a perturbation-robust dev metric), not the noisy 7-scenario dev alone.
3. If v2 still cannot beat the BC warm-start on the training distribution, the honest read is that RL adds nothing over
   amortized BC *here*, and the frontier is generalization (R11.6B) / coverage — still not per-step torque RL.

## Files touched (all non-core)
- `hymeko_rl/coin_delivery/theta_option/delivery_theta_env.py` (`CoinDeliveryThetaOptionEnv` + `DeliveryReward` +
  box↔θ), `hymeko_rl/experiments/r11_6a_delivery_rl.py` (bank + BC-warmstart + TD3 + dev-select + gate) — committed
  `5cc4154a`. `hymeko_rl/tests/test_r11_6a_delivery_rl.py` (7 tests). Report + `reports/2026-08-04-r11-6a-delivery-rl/`.
- **Read-only:** `option_rl/*`, `forward_displacement`, `solver`, `delivery_bc/*`, the demonstration bank.

## CORE.YAML items touched
**None.** New Python under `hymeko_rl/`; no dependency added.

## Test results
7 tests pass (`test_r11_6a_delivery_rl.py`): reward safety-barrier + shaping, box↔θ round-trip/clip, env terminal step,
gate PASS + 3 negative classifications. ruff / mypy --strict / radon (no C+) clean.

## Performance / provenance
Wall ~1 h 50 m (≈36 min reconstructing 51 handoffs + 3 × ~25 min TD3). Per-process RSS ≪ 16 GB. Mac, 48 GB, Apple Silicon,
`OMP_NUM_THREADS=2`, deterministic (fixed seeds; fresh-rig reconstruction). Teacher θ from the R11.4B dataset
(`reports/2026-08-03-r11-4b-bc/dataset/`, bank md5 `473244de4795254f5de99f4ca7732714`). §2 plan
`docs/plans/2026-08-04-r11-6a-delivery-rl/` (4-format). Energy diagnostic-only (frozen R11 contract). No deps.

## Guards
Safety held (0 unsafe rollouts, all seeds). Energy stayed out of the objective (R11.8 contract). Full structured θ (not a
bounded residual — avoids R8/R9). Multi-scenario shaped reward (not a σ-ball — avoids R10.2). 56 demos across 51 scenarios
(not 2 cradles — removes the R7 coverage gate). The v1 negative is **instability**, cleanly attributable to the missing
positive-replay anchor — a mechanism fix, not an escalation to per-step RL.
