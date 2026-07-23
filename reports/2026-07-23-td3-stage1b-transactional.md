---
title: Transactional TD3 actor update V1 — Stage 1b
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PHASE_SWITCHED_TD3_STAGE1B_NO_IMPROVEMENT
tags: [coin, phase-switched, td3, transactional, trust-region, basin-preserving, negative-for-improvement]
---

# PHASE_SWITCHED_TD3_STAGE1B_NO_IMPROVEMENT — divergence solved, no local improvement

One bounded Stage-1b stabilization campaign (`TRANSACTIONAL_TD3_ACTOR_UPDATE_V1`, Arm A). Banks, reward, phase switch,
horizon (30), n-step (4), target smoothing (0.10/0.25), eval protocol, dev bank — **all unchanged**; only the actor
update is transactional behind a critic-authorization gate. config_sha `d4cf53ae`, banks `422eb63f`/`da7a29fb`, wall
55.7 s. No Stage 2, no neutral-reset composition, no Arm B, no SAC, no dev-bank widening, final-test bank unopened.

## The mechanism works (the load-bearing result)
The V1 baseline diverged the instant the actor was unfrozen (drift 0 → **8.0**, the full ±4 range). The transactional
update eliminates that:

| | V1 (unrestricted) | Stage-1b (transactional) |
|---|---|---|
| actor drift from update-0 (final) | **8.000** (full range) | **0.095** |
| dev Δstrict / Δdwell / Δexit | −0.17 / −0.58 / +0.33 | **0 / 0 / 0** |
| dev Δcontact | −0.43 | −0.094 |
| critic Q1 (final) | −196 | −71 |

- **Critic-authorization gate fired correctly:** at ckpt 0 the untrained critic (twin disagreement 1.97 > 0.5) was
  **not authorized** — no actor updates. After warm-up (ckpt 2000) it authorized (twin 0.02) and updates began.
- **Trust region + backtracking held the basin:** 18 updates accepted (scales {1.0:17, 0.5:1}), then the cumulative
  anchor drift reached the 0.060 cap and **every subsequent update was rejected** (1482 rejections). The actor froze at
  drift 0.095 (probe) / ≤0.060 (anchor) — an ~84× reduction vs V1. No divergence.

## But there is no local improvement
Stage-1b **matched** pi_0 on the target metrics (Δstrict 0, Δdwell 0, Δexit 0) but **failed** the pass gate on one
condition: contact retention degraded −0.094 (< the −0.05 threshold). Even the tiny bounded actor movement (≤0.06 on the
anchor bank) found no improving direction — it left strict/dwell unchanged and slightly reduced contact retention.

- **Measured:** within a trust region that provably prevents divergence, TD3 finds no direction that improves late-phase
  delivery over the frozen pi_0; the only measurable effect of the bounded movement is a small contact-retention drop.
- **Inferred:** the fragile handoff basin is **locally supervised-ceiling-bound** — pi_0 sits at (or very near) a local
  optimum for the late phase, so any admissible local step is neutral-to-slightly-harmful. This is the same wall the
  residual-critic line hit (`RESIDUAL_CRITIC_ROUTE_BLOCKED`, no useful local authorization) and consistent with the coin
  arc: local policy-improvement caps at the supervised ceiling; gains came only from nonlocal search.

## Verdict logic
- critic authorized at some checkpoint ⇒ not `TD3_CRITIC_NOT_AUTHORIZED_FOR_ACTOR_UPDATE`.
- two-consecutive beat-or-match gate fails only on Δcontact (−0.094) ⇒ `PHASE_SWITCHED_TD3_STAGE1B_NO_IMPROVEMENT`.

## Claims / non-claims
**Claims:** (1) The transactional update (snapshot → trust region → backtracking → reject, behind a critic-authorization
gate) **eliminates the V1 divergence**: drift 8.0 → 0.095, no bound saturation, no Q blow-up. (2) The auth gate correctly
withholds updates from an untrained critic. (3) Within the trust region, TD3 yields **no local improvement** over pi_0
(Δstrict/dwell/exit = 0; Δcontact −0.094).
**Non-claims:** NOT "TD3 can never improve the late phase" — one single-seed run, one trust-region setting, ~18 accepted
updates before the cumulative cap; more training, a stronger critic, or a wider cumulative cap are untested. NOT a
neutral-reset or sealed result. Dev `target_entry` coverage is thin (1 start).

## Next narrow experiment
The transactional harness now safely permits larger local exploration without divergence. Two bounded options (either
needs your go): (a) **relax the cumulative cap** (e.g. cum_max 0.060 → 0.15) and re-run — does a slightly larger but
still-bounded basin admit an improving direction, or does contact keep degrading? (b) **improve the critic first**
(longer warm-up / n-step {4,8} sweep, more collection) and re-run the same trust region — the current critic Q drifts
to −71; a better-calibrated critic may point somewhere useful. If both stay neutral-to-harmful, the late-phase basin is
confirmed supervised-ceiling-bound and improvement must be sought nonlocally (as the coin arc found), not via a local
actor step.

## Files
- impl (commit `91e1031`): `coin_td3_transactional.py`, `coin_td3_stage1b.py`, tests.
- results (this commit): `td3_stage1b_results.json`, `td3_stage1b_launch.json`, this report,
  `reports/figures/2026-07-23-td3-stage1b-transactional.png`.
- upstream: V1 baseline `78d6859` (PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; Mac, torch 2.12.0, mujoco 3.10.0; seeds {torch 0, collect 0, replay 1}.
RL not bit-reproducible under threaded BLAS; claims rest on the within-run trend across 5 checkpoints (divergence
elimination is unambiguous; the no-improvement is single-seed).
