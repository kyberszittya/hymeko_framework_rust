---
title: REALISTIC_MOTION_CONTRACT_V1 — audit, contract, governor, and the DEFINITIVE governed-coin verdict (outcome 3)
date: 2026-07-25
branch: feat/architectural-assimilation-v1
status: CONTRACT GATED; DYNAMICS CONTRACT V2 FROZEN (G0); G1 VERDICT = LEGACY_COIN_SOLUTION_DEPENDED_ON_UNREALISTIC_DYNAMICS
contract: REALISTIC_MOTION_CONTRACT_V1
---

## DEFINITIVE RESULT (G0 frozen contract + G1 re-measurement)
**G0 — `COIN_DYNAMICS_CONTRACT_V2` calibrated on PHYSICS ONLY (not delivery) and FROZEN** (commit `b5e9c7b9`): qdot_soft
1.5, qdot_hard 3.0, armature 0.4, damping 15.0, friction 0.1, torque-rate 25.0, control_dt 0.01, 20 substeps. Passes all
five physical criteria (free-space speed cap, velocity reversal/braking, sudden release decay, 12 rad/s contact-impulse
recovery within cycles via inertia+damping — no artificial clamp — normal-range undistorted). Lighter configs failed the
free-space cap; calibrating on physics rather than delivery is what forced the stronger inertia.

**G1 — re-measure balltip deploy + expert under the FROZEN V2, expert RE-SEARCHED under the new dynamics, no retraining**
(commit `a5539756`, 16 states, `coin_governed_remeasure.json`):

| arm | K6 (governed V2) | legacy fast | mean peak joint vel |
|---|---|---|---|
| deploy (update-0 + b8) | **0.125** (2/16) | ~10/10 on delivering states | ~2.2 rad/s |
| structured expert (192-shot) | **0.312** (5/16) | ~10/10 | ~2.2 rad/s |

Velocities are now realistic (mostly ≤ 3.2 rad/s). Both deploy AND the 192-shot expert collapse. **Verdict:
`LEGACY_COIN_SOLUTION_DEPENDED_ON_UNREALISTIC_DYNAMICS` (outcome 3).** The legacy coin success was substantially a
dynamics exploit (27 rad/s "throws"), not a strategy robust to realistic motion. This is the separation the contract was
built to force — and it landed on the uncomfortable-but-important outcome.

**Implication (per the pre-registered G2):** do NOT retrain the same option language on V2. The push→brake→release macro
with ±3.0 impulses is what the fast dynamics rewarded; under governed motion it needs a STRATEGIC redesign — longer/slower
push, more continuous contact retention, lower-impulse acceleration, earlier braking, velocity-dependent release, longer
settling, more closed-loop replanning. O3 (triangle contact) MUST be evaluated only on the frozen V2 contract; it stays
paused until the redesign + the 6D re-run land.


# REALISTIC_MOTION_CONTRACT_V1

Triggered by the user's observation that the arm "teleports" in the coin videos. Physical limits first, reward second,
audit before implementing.

## Audit (discriminating result)
- **Video is 3.33× slow-motion**, not time-compressed (`sim_dt·frame_skip = 10 ms/step` shown over `33 ms` at 30 fps).
  So the teleport is REAL motion, not a playback artifact.
- **6D-1 arm is realistic**: peak joint vel 0.6 rad/s, EE 0.14 m/s.
- **Coin arm is the offender**: peak joint vel **27.2 rad/s** (real robots 1–3), coin slides at **1.54 m/s**. Root cause
  in the actuator config: torque actuators with **armature = 0** and low damping. Armature/damping ALONE are insufficient
  (0.05 armature still leaves ~20 rad/s) — the velocity/slew LIMIT must be the primary layer, exactly as directed.

## Contract (`hymeko_rl/env/motion_contract.py`, task-independent, 11 tests)
- `MotionLimits`, `MotionMetrics` (peak/mean joint vel+acc, EE speed, jerk, Δu, integrated effort, terminal velocity).
- `slew_limited_position` — the anti-teleport velocity limiter for POSITION control (6D-0/6D-1/pick-place/AIBO).
- `TorqueGovernorConfig` + `govern_torque` — for TORQUE arms (the coin): suppress ONLY the accelerating torque above
  `qdot_soft` (zeroed at `qdot_hard`); **braking torque is never touched**; magnitude never increased, sign preserved.
  NOT plain torque scaling. armature/damping are the secondary stabilisation layer.
- `motion_penalties` — thresholded `ReLU(x/safe−1)²` (normal speed unpenalised), the reward's second layer.
- `terminal_velocity_certified` — no throw-through: success requires a near-stationary terminal state.
- `assert_realistic_motion` — the HARD gate, on **velocity + EE speed** (instantaneous `qacc` is servo/contact-transient —
  a fine 1.9 rad/s arm still spikes ~1600 rad/s² — so acceleration is reported + soft-penalised, not hard-gated).

## Integration
- Position control: the slew limiter is wired into `ArmReachEnv` (opt-in; 22 reach/SE3 tests unchanged). 6D-1 under a
  2.0 rad/s limit → peak 1.94, EE 0.87, terminal 0.007 → **gate PASS**. The coin (unlimited) → **gate FAIL** (vel 27),
  so an unrealistic run can no longer silently ship.
- Coin (torque): the governor is applied per sub-step via MuJoCo's `mjcb_control` **around the run** (set-then-reset, no
  persistent global), on model COPIES — the frozen baseline files are never mutated (path (a), additive).

## Preliminary governed-coin finding (5 states, 96-shot expert, quick tuning — NOT the final measurement)
| dynamics | peak joint vel | mean-peak | expert K6 |
|---|---|---|---|
| legacy fast (COIN_LEGACY_FAST_V1) | 27.2 | — | high (≈10/10 on delivering states) |
| governor + armature 0.15 / damping 6 | 7.1 | 4.4 | 2/5 |
| governor + armature 0.3 / damping 10 | 4.1 | 2.5 (realistic) | 1/5 |

**The more realistic the motion, the more delivery collapses — even for the structured expert.** This points at the
user's **outcome 3** (`LEGACY_SOLUTION_DEPENDED_ON_UNREALISTIC_DYNAMICS`): the legacy coin solution leaned substantially on
the fast dynamics; under realistic limits, even the expert search struggles. Two caveats before this is a verdict: (i) the
governor does not yet cleanly cap contact-imparted velocity spikes (7 rad/s at the moderate setting — contact impulses are
not torque, so the armature/damping layer must dissipate them, needs tuning); (ii) 5 states / 96 shots is a pilot.

## Status labels (per the user's versioning)
- **COIN_LEGACY_FAST_V1** — the frozen coin baseline: historically reproducible, algorithmically informative, **not
  physical-transfer evidence**. Preserved, not mutated. Legacy videos → appendix ("why the motion contract was introduced").
- **COIN_GOVERNED_MOTION_V2** — the governed dynamics-version (torque governor + torque-rate limit + armature/damping +
  terminal-velocity certificate). Every governed manifest must carry `dynamics_contract`: motion_limit_version, qdot_soft/
  qdot_hard, torque limits, torque-rate limit, armature, damping, control_dt.

## Next (the user's decision list — additive, no retraining yet)
- [ ] Tune the governor so peak velocity is cleanly ≤ hard (stronger armature/damping to dissipate contact spikes, or a
  velocity-aware terminal clamp), report the `dynamics_contract`.
- [ ] Proper COIN_GOVERNED_MOTION_V2 re-measurement: frozen deploy (update-0 + search), structured expert, clamp, balltip
  on the SAME panels/seeds/budgets, K6 + peak/mean vel + EE speed + terminal vel + option duration + time-to-delivery +
  effort + contact retention + failure decomposition → the 3-outcome verdict.
- [ ] Re-render coin videos ONLY from governed scored rollouts (legacy → appendix).
- [ ] In parallel: 6D-0 + 6D-1 re-run with PHYSICALLY-DERIVED horizons (travel_time ≈ path/EE_speed + accel/settle margin,
  in seconds / control_dt steps), same seeds, new execution-based eligibility, timeout as a separate failure category,
  new hierarchical bootstrap → the robot-realistic claim `MULTIMODAL_POLICY_SEARCH_VALIDATED_UNDER_REALISTIC_MOTION_LIMITS`.
- [ ] O3 stays PAUSED until both motion regressions land.
