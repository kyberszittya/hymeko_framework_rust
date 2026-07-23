---
title: Transport-to-Dwell TD3 baseline V1
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: TRANSPORT_DWELL_TD3_NO_IMPROVEMENT
tags: [coin, td3, transport-dwell, phase-conditioning, negative-for-improvement]
---

# TRANSPORT_DWELL_TD3_NO_IMPROVEMENT

One bounded Transport-to-Dwell campaign (Arm A) on the **correctly re-scoped** ontology. config_sha `3ec6dbeb`, wall
73.9 s. No old target_entry curriculum, no Stage 2, no Arm B, no SAC, no neutral-reset, no final-test.

## Re-scoped ontology (contract verified before the run)
Control modes = the phases that actually persist: **{transport, braking, settling_dwell}**; `target_entry` demoted to
**event features** `[inside_target_zone, just_entered, just_exited, distance_to_target, radial_velocity]`; contact an
orthogonal flag. Actor/critic conditioning = onehot3(mode) ++ onehot2(contact) ++ event5 = 10 (obs_48 ++ 10 = 58),
zero-init ⇒ update-0 == pi_0 (tested ∀ conditioning). Persistent banks clear the §6 thresholds: train transport 30 /
braking 20 / settling_dwell 9 (min 20/12/8), dev 15/10/6, seed-disjoint. Horizon **60**, 50/30/20 dynamically-balanced
sampling. Everything else (pi_0, reward, gate, 4-step, smoothing 0.10/0.25, exploration, transactional caps, term/trunc,
eval) unchanged.

## Result — the correct ontology still does not improve over pi_0
Dev vs frozen pi_0, trained checkpoints 4000/6000/8000 (`accepted=13`):

| metric | Δ vs pi_0 |
|---|---|
| strict K6 success | 0.000 |
| max dwell | **−0.29** |
| target-entry rate | −0.032 |
| braking effectiveness | **+0.0012** |
| transport progress | **+0.0041** |
| contact retention | **−0.068** |
| target exit | 0.000 |

The only positives are marginal (braking, transport progress); dwell and contact **degrade**. The §13 gate fails on
contact retention (−0.068 < −0.05). ⇒ `TRANSPORT_DWELL_TD3_NO_IMPROVEMENT`.

**First-pass trap avoided.** The 400-update smoke showed Δmax_dwell **+0.26** and Δstrict **+0.097** — an early-training
transient that **reversed** at 8k (Δdwell −0.29, Δstrict 0). Reporting the smoke as a result would have been a false
positive; only the converged campaign counts.

## Drift (constrained metric)
anchor-L2 cumulative max **0.044** (cap 0.060, ~73% used — more than Stage-1c's 0.031); per-step max ≈0.010 (still the
binding cap); probe-L∞ 0.051 (diagnostic only). 13 accepted / 1487 rejected.

## Interpretation across the arc (measured vs inferred)
- **Measured:** every correctly-built local-improvement attempt over the frozen pi_0 late controller — residual critic,
  unanchored TD3, transactional TD3, dynamic-phase TD3, and now the correctly-scoped transport-to-dwell TD3 — yields no
  net dev improvement; the transactional ones stay in-basin and mildly degrade contact.
- **Inferred (increasingly supported, NOT closed):** the frozen pi_0 late controller sits at or very near a local
  optimum that bounded local RL does not beat within the trust region. This is now consistent across five distinct
  formulations, each of which fixed a real defect the prior one had (ontology, phase-blindness, drift metric, gate
  false-PASS). I am **not** stamping a supervised-ceiling verdict from a single seed — but the direction is strong.

## Claims / non-claims
**Claims:** (1) The re-scoped ontology + persistent banks clear the §6 thresholds and the update-0 identity holds
(tested). (2) The correctly-scoped transport-to-dwell TD3 does not improve over pi_0 (Δstrict 0, Δdwell −0.29,
Δcontact −0.068); marginal braking/progress gains do not offset the degradation, and the gate fails on contact.
(3) The smoke's dwell/strict gains were an early-training transient that reversed by 8k.
**Non-claims:** NOT a supervised-ceiling verdict (single seed; per-step cap 0.010 still binding and untested; critic
quality untested). NOT a neutral-reset/sealed result.

## Next narrow experiment (needs your go)
Two levers remain, both isolatable within this now-correct harness: (a) the **per-step trust-region cap (0.010)** binds
while cumulative is ~73% used — a modest per-step relaxation is the one untested degree of freedom; (b) **critic quality**
(longer warm-up / n-step {4,8} sweep / more collection). If a modestly larger per-step step *still* only trades marginal
braking for dwell/contact loss, the local-improvement route is done and improvement must be nonlocal (as the coin arc
found earlier).

## Files
- impl: `coin_transport_dwell.py`, `coin_phase_conditioning.py` (n_cond), `coin_transport_dwell_campaign.py`,
  `freeze_transport_dwell_config.py`, tests.
- results: `transport_dwell_results.json`, `transport_dwell_launch.json`, `transport_dwell_config.json`, this report,
  `reports/figures/2026-07-23-transport-dwell-td3.png`.
- upstream: ontology `18cf036`, underpowered `d2bc1a3`, Stage-1c `4b4a0d3`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; Mac, torch 2.12.0, mujoco 3.10.0; seeds {torch 0, collect 0,
replay 1}. RL not bit-reproducible under threaded BLAS; the no-improvement rests on the converged-checkpoint trend,
single seed.
