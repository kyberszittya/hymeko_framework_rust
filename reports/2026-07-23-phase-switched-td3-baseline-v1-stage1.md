---
title: Phase-switched TD3 baseline V1 — Stage 1
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT
tags: [coin, phase-switched, td3, late-controller, baseline, negative]
---

# PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT

Stage 1 of the first phase-switched full-action late-controller campaign (Arm A, pi_late = exact pi_0 copy). **The
contracts all functioned; the baseline does not pass** — an unanchored TD3 actor diverges from the frozen pi_0 basin the
moment it is trained. No neutral-reset composed eval (gated), no Stage 2 (gated), no Arm B, no SAC, final-test bank
unopened, reward/gate/pi_0 unchanged.

**Provenance.** config_sha `d4cf53ae`, banks `late_train 422eb63f` / `late_dev da7a29fb`, pi_0 `1902454c`, seeds
{torch 0, collect 0, replay 1}, wall 55.9 s. Launch manifest saved (`td3_baseline_v1_launch.json`).

## Contracts that worked (measured)
- **update-0 identity:** ckpt 0 — Δ(all dev metrics) = 0, actor drift = 0 (pi_late ≡ pi_0).
- **critic warm-up held the actor frozen:** through ckpt 2000 (= `critic_warmup_steps`), actor drift stayed **0.000** —
  no actor updates during warm-up, exactly as specified.
- Phase-correct n-step target (gate-on ⇒ smoothed pi_late 0.10/0.25; gate-off ⇒ frozen pi_0), terminated/truncated
  masked separately, deterministic replay-to-handoff reconstruction — all exercised.

## Result — the actor diverges post-warm-up (dev vs frozen pi_0 continuation)

| update | Δstrict | Δmax_dwell | Δtarget_exit | Δcontact | actor drift | Q1 mean |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | 0.0 |
| 2000 (warm-up end) | 0.000 | 0.00 | 0.000 | 0.000 | 0.000 | −34.2 |
| 4000 | −0.334 | −1.33 | +0.500 | −0.438 | **8.000** | −70.8 |
| 6000 | −0.417 | −2.08 | +0.583 | −0.439 | **8.000** | −139.8 |
| 8000 | −0.167 | −0.58 | +0.333 | −0.426 | **8.000** | −195.7 |

The moment actor updates begin (step 2000), the actor **saturates to the action bound** (drift 8.0 = the full ±4
range → a bang-bang policy) and every dev metric degrades: strict-K6 success drops, max-dwell drops, target-exit rises,
contact-retention drops. The critic Q diverges monotonically (−34 → −196) and the critic loss grows (70 → 893).

## Mechanism (measured vs inferred)
- **Measured:** with critic warm-up and delayed updates, the actor is stable until unfrozen; once trained it collapses
  to the action bound and coupled actor/critic divergence follows.
- **Inferred:** the baseline has **no anchor to the pi_0 basin** (no BC/trust-region term, actor_lr 3e-4, full TD3 actor
  freedom). On this fragile contact task the actor follows the (diverging) critic gradient straight to the bound, exactly
  the "immediately destroy the fragile handoff basin" failure the campaign warned about — here for TD3. Consistent with
  the whole coin arc: local RL from a supervised init degrades without a trust region.
- **Not** a target-smoothing artifact: smoothing was the amended small 0.10/0.25 (local), not the 0.8/2.0 control — the
  divergence is the unanchored actor, not broad smoothing.

## Gated stops (respected)
Stage 2 **not entered** (§11: only after Stage 1 passes). No neutral-reset composed eval (§12). No Arm B, no SAC,
final-test bank (8000–8049) unopened, reward/gate/pi_0 untouched.

## Claims / non-claims
**Claims:** (1) All Stage-1 contracts function (update-0 identity, warm-up freezes the actor, phase-correct targets,
deterministic reconstruction). (2) The unanchored phase-switched TD3 baseline (Arm A) does not improve over frozen pi_0;
the actor diverges to the action bound post-warm-up and degrades every dev metric; the critic Q diverges.
**Non-claims:** NOT "TD3 cannot work here" — this is one single-seed baseline with **no basin anchor**; the divergence is
robust *within* this run (monotone, drift-saturated, all families) but the fix direction is specific and untested. NOT a
neutral-reset or sealed result. Dev `target_entry` coverage is thin (1 start) — the negative rests on braking +
settling_dwell too.

## Next narrow experiment (Stage-1b, stabilized)
Re-run Stage 1 with a **pi_0 basin anchor**: add a BC/trust-region term keeping pi_late near pi_0 on gate-on states
(or a proximal actor-update cap ≤ a small per-step action-delta, mirroring the residual trust region), and/or a lower
actor_lr, and/or a longer critic warm-up. Gate on the same two-consecutive dev criterion. If an anchored actor still
fails to beat pi_0, the late-phase basin is supervised-ceiling-bound (as the residual line was); if it improves, proceed
to Stage 2 (horizon 60, +transport,+overshoot).

## Files
- impl (commits `074dd13` amended prereg, `dd2bf6d` trainer): `coin_td3_trainer.py`, `coin_td3_baseline_v1.py`,
  `coin_td3_contracts.py`, `coin_late_start.py`, `coin_phase_switched_late.py`, tests.
- results (this commit): `td3_baseline_v1_results.json`, `td3_baseline_v1_launch.json`, this report,
  `reports/figures/2026-07-23-td3-baseline-v1-stage1.png`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; Mac, torch 2.12.0, mujoco 3.10.0. RL is not bit-reproducible under
threaded BLAS; the divergence claim rests on the monotone within-run trend across 5 checkpoints, not one number.
