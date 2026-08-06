---
title: Stage-1c dynamic phase-conditioned transactional TD3
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PHASE_SWITCHED_TD3_STAGE1C_NO_IMPROVEMENT
tags: [coin, td3, phase-conditioning, dynamic-phase, transactional, negative-for-improvement]
---

# PHASE_SWITCHED_TD3_STAGE1C_NO_IMPROVEMENT

One Stage-1c campaign with **DYNAMIC per-transition phase conditioning** (the audit's `PHASE_CONDITIONING_MISSING` fix),
everything else identical to Stage-1b (banks, reward, horizon 30, n-step 4, smoothing 0.10/0.25, exploration, and the
transactional trust-region thresholds — **not relaxed**, to isolate the phase effect). config_sha `d4cf53ae`, wall
59.1 s. No Stage 2, no Arm B, no SAC, no neutral-reset, no final-test seeds.

## Contract verified before the run (DYNAMIC_PHASE_TRANSITION_CONTRACT_V1)
`phase_t = PhaseDetector(current state)` (not `LateStart.family`); stored `phase_t`/`phase_tp1`; actor `obs++onehot(phase_t)`,
critic `obs++onehot(phase_t)++action`, bootstrap `obs_boot++onehot(phase_boot)`; actor mask on `gate_t`, target on
`gate_tp1`+`phase_tp1`. 7 tests pass (distinct entry/braking/settling; phase changes output only after training;
update-0 == pi_0 ∀ phase; phase_t in actor loss; phase_boot in TD target; replay preserves the phase sequence
`phase_tp1(k)==phase_t(k+1)`; `phase_t != static family`).

## §10/§11 phase-transition report — the static labels mislabel the dynamics
On the Stage-1 banks, the **dynamic** phase distribution is dominated by phases *outside* the nominal Stage-1 set:
`transport` 260 steps, **`contact_retention` 321**, `braking` 54, `settling_dwell` 48, **`target_entry` only 1 step**.
The nominal transitions barely occur: `target_entry→braking` 0, `braking→settling_dwell` 1, `settling_dwell→braking` 0;
the real dominant transitions are `transport↔contact_retention` (70) and `transport↔braking` (33). So an actor
conditioned on the static `LateStart.family` would be conditioning on a label the episode leaves almost immediately —
dynamic conditioning is the correct construction, and this run genuinely differs from the phase-blind Stage-1b.

## Result — dynamic phase conditioning did NOT unlock local improvement
Dev vs frozen pi_0 (trained checkpoints 4000/6000/8000, `accepted=21`):
Δstrict 0, Δmax_dwell 0, **Δtarget_exit +0.083**, **Δcontact −0.069**. The trained phase-conditioned actor **degrades**
contact retention and raises target exit — like Stage-1b (Δcontact −0.094), no better on strict/dwell.

**Correct-gate note (a false PASS caught and fixed).** The gate first reported PASS — but only because the two
*pre-training* checkpoints (update-0 and end-of-warm-up) are exactly pi_0 and therefore trivially "match" pi_0. The gate
now additionally requires `accepted > 0` (the actor was actually updated); under it Stage-1c is `NO_IMPROVEMENT`
(trained checkpoints fail on Δcontact and Δexit) and Stage-1b remains `NO_IMPROVEMENT`.

## Drift in the constrained metric (audit fix applied)
Reported now in the metric the cap actually constrains: **anchor-L2 cumulative** median/p95/max = 0.024/0.030/**0.031**
(cap 0.060 — only half used); **per-step over accepted** max ≈ 0.010 (the per-step cap 0.010 is the **binding**
constraint, exactly as the audit predicted); probe-L∞ 0.089 is reported **separately as a diagnostic**, never as the
budget. 21 accepted / 1479 rejected — rejections are on the per-step cap, not the cumulative.

## Interpretation (measured vs inferred)
- **Measured:** with the correct dynamic phase conditioning and the same trust region, the trained actor still finds no
  improving local direction — it mildly degrades contact/exit. The cumulative budget is half-used; the per-step cap
  (0.010) binds.
- **Inferred:** the phase-blindness was a real defect but **not** the reason for no-improvement; per-phase specialization
  did not open a beneficial local direction within this trust region. The two live levers are now isolated: (1) the
  per-step cap (0.010) is binding — a larger per-step step (still bounded) is untested; (2) the critic (Q drifts, warm-up
  short) may not point anywhere useful. The "supervised ceiling" question remains **open** — this run narrows it but does
  not settle it.

## Claims / non-claims
**Claims:** (1) DYNAMIC_PHASE_TRANSITION_CONTRACT_V1 holds (7 tests). (2) On the Stage-1 banks the dynamic phase is
dominated by transport/contact_retention; target_entry is essentially absent (1 step). (3) Dynamic phase conditioning,
same trust region, does not improve over pi_0 (Δstrict/dwell 0; Δcontact −0.069; Δexit +0.083). (4) A gate false-PASS
(pre-training identity) was found and fixed (`accepted > 0`).
**Non-claims:** NOT a supervised-ceiling claim (still open). NOT a neutral-reset/sealed result. Single-seed; the per-step
cap and critic quality are untested confounds. Dev `target_entry` coverage is ~absent dynamically.

## Next narrow experiment (needs your go)
Isolate the binding constraint: re-run Stage-1c changing ONLY the per-step trust-region cap (e.g. 0.010 → 0.02–0.03,
cumulative still 0.060), since the cumulative budget was only half used — does a slightly larger bounded step find an
improving direction, or keep degrading? If it keeps degrading with more room, the critic is the lever (longer warm-up /
n-step sweep), and only if both fail is the late basin supervised-ceiling-bound.

## Files
- impl: `coin_td3_phase_stage1c.py`, `coin_phase_conditioning.py` (PhaseDetector), gate fix in `coin_td3_transactional.py`,
  `coin_stage1c.py`, tests; `phase_transition_report_v1.py`.
- results: `td3_stage1c_results.json`, `td3_stage1c_launch.json`, `phase_transition_report_v1.json`, this report,
  `reports/figures/2026-07-23-td3-stage1c-dynamic-phase.png`.
- upstream: contract `a4b963d`, Stage-1b `b53abe6`, audit `a9098a2`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; Mac, torch 2.12.0, mujoco 3.10.0; seeds {torch 0, collect 0, replay 1}.
RL not bit-reproducible under threaded BLAS; the no-improvement rests on the trained-checkpoint trend, single seed.
