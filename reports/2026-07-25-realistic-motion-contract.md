---
title: REALISTIC_MOTION_CONTRACT_V1 — audit, contract, torque governor, and the preliminary governed-coin finding
date: 2026-07-25
branch: feat/architectural-assimilation-v1
status: CONTRACT BUILT + GATED; COIN GOVERNED RE-MEASUREMENT PRELIMINARY (points at outcome-3: legacy leaned on fast dynamics)
contract: REALISTIC_MOTION_CONTRACT_V1
---

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
