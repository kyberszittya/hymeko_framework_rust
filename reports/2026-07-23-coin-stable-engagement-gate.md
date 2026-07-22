---
title: PHASE_GATE_STABLE_ENGAGEMENT_PASS — hybrid bilateral + unilateral-co-motion gate
date: 2026-07-23
slug: coin-stable-engagement-gate
task: coin_v3 delivery — PHASE_GATED_LEARNED_RESIDUAL_CONTROLLER (§1-§8 hybrid gate)
verdict: PHASE_GATE_STABLE_ENGAGEMENT_PASS
supersedes_gate: coin_phase_gate.PhaseGate (d739e8af, rejected — premature unilateral activation)
---

# STABLE_OBJECT_ENGAGEMENT_V1 — hybrid deployable gate

**Created-at:** 2026-07-23 03:05 CEST
**Accepted going in:** `PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION` (commit `50a6296`) rejected the `left OR right`
predicate. This stage builds the directed hybrid gate and audits it. **The rejected V1 gate is preserved in
`coin_phase_gate.py` (SHA `d739e8af`), not modified in place** (§10).

## Design (new module `coin_stable_engagement.py`)

Two arm paths, shared hysteresis; the gate returns a multiplier `g_t ∈ {0,1}` and generates no actions.

- **BILATERAL fast path** — arm on `left AND right` for `bilateral_arm_after=3` consecutive steps.
- **UNILATERAL co-motion slow path** — arm on the **same** contacting side for `uni_arm_after=6` consecutive steps
  **AND** the coin co-moving with that fingertip over a trailing `kin_window=4` window.
- **Disarm** — complete contact loss `disarm_after=2` steps → `REACQUIRE`; **co-motion is required only to ARM,
  never to stay armed** (§5 — settling must not disarm when the coin correctly stops).

### Co-motion test (§4), thresholds geometry-derived (NOT rollout-tuned)

Over the trailing window: `Δcoin`, `Δtip`, `slip = Δcoin − Δtip`. Qualify iff

- `|Δcoin| ≥ coin_motion_floor = disk_r·0.025 = 5e-4 m` (coin moves beyond jitter; sim jitter ≈ 0),
- `cos(Δcoin, Δtip) ≥ 0.707 = cos 45°` (directional agreement),
- `|slip| ≤ slip_bound = disk_r/2 = 0.01 m` (bounded relative slip).

Coin position from the canonical observation; fingertip position from FK (MuJoCo sites `tip_left`/`tip_right` =
`inner._tip_sites`); finite differences are causal. **No** `disk_to_zone`, target, success, seed, trajectory id,
planner state, future obs, or hidden state.

### Bug caught (contract discipline)

The first co-motion probe read `|Δtip| = 0` everywhere → would have falsely suggested BLOCKED. Cause:
`np.asarray(data.site_xpos[i][:2])` returns a **live MuJoCo view**, so every stored history row aliased the same
buffer. Fixed with a forced copy (`stable_engagement_signals` copies all three positions). A buggy probe is not a
negative result — the real signal shows co-motion `dot ~0.95–1.00` for controlled contact, `−0.08`/`0.44` for
uncontrolled (1358/1202).

## §6 controller state — `PHASE_GATE_CONTROLLER_STATE_V2`

`EngagementState` = {mode, bilateral_counter, uni_counter, uni_side, loss_counter, coin_hist, ltip_hist, rtip_hist,
last_arm_mechanism, comotion_ok}. `state_v2()`/`load_state_v2()` round-trip; contract SHA prefix `7633dd3c`. Test
`test_controller_state_v2_resume_reproduces_transition` proves a mid-episode serialize→resume reproduces the next
transition and gate bit-for-bit.

## §8 frozen audit — PASS

14 trajectories (9 π₀ headline + 5 certified), hybrid gate on deployable signals:

| check | result |
|-------|--------|
| zero activation during approach | **0/14** (rejected OR gate: 11/14 premature) |
| zero activation on acquisition brushes | **0/14** |
| bilateral grasp-style transport covered | all grasp deliveries armed; BILATERAL_FAST fires (1164, 6000/6001/6003) |
| unilateral push route arms on genuine push | yes (1202, 1278, cert 6002/6005) |
| **seed 1447 push detected without target** | **UNILATERAL_COMOTION @ t=22** (never forms bilateral) |
| all headline deliveries armed | 3/3 (1011, 1447, 1568) |
| alternating/transient cannot accumulate | unit test + brush_act 0/14 |
| settling/dwell no spurious disarm | unit test `test_settling_does_not_disarm` |
| all signals deployable/causal | contract_v2 |

**`PHASE_GATE_STABLE_ENGAGEMENT_PASS`.** Every activation is in TRANSPORT phase. Seed 1358 (sustained but coin ⊥ tip,
dot −0.08) is **correctly NOT armed** — co-motion rejects uncontrolled contact, which duration alone would not.

Note: grasp deliveries (1011 bilat@60, 1568 bilat@49) arm via UNILATERAL_COMOTION *first* (the sustained co-moving
push precedes bilateral closure) — legitimate; the directive requires bilateral transport be *covered*, not that it
arm via bilateral first. My audit's initial check wrongly demanded BILATERAL_FAST-first and self-reported BLOCKED;
corrected to the directive's coverage criterion.

## Tests

`hymeko_rl/tests/test_coin_stable_engagement.py` — **14/14 pass** (0.74 s): bilateral-arms-at-3, unilateral-short-
never-arms, unilateral-comotion-arms-at-6, stationary/opposite/high-slip rejected, alternating-no-accumulation,
disarm+reacquire, settling-no-disarm, terminal, reset, V2-resume, contract-SHA, invalid-config.

## Files touched

- `hymeko_rl/coin_delivery/coin_stable_engagement.py` (new, ~200 L) — hybrid gate, V2 state, co-motion, signals.
- `hymeko_rl/tests/test_coin_stable_engagement.py` (new, 14 tests).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_stable_engagement_audit.py` + `stable_audit.json`
  + `coin_comotion_probe.py` + `plot_stable_engagement.py`.
- `reports/figures/coin_stable_engagement.png`.

**CORE.YAML:** none. **Rejected V1 gate `d739e8af` preserved unmodified.** SAC quarantined. Mac; kato14 clean.

## Next (§9, after this PASS)

`PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED` → `EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS` →
`PHASE_GATE_REPLAY_STATE_CONTRACT_PASS` (V2 state in replay/target) → `PHASE_GATED_RESIDUAL_CRITIC_PASS` →
`LATE_PHASE_RESIDUAL_GRADIENT_CONTRACT_PASS` → guarded residual TD3 micro-smoke. Update 0 must still reproduce
3/9 · 2/30 · 9/9 · {1011,1447,1568}.
