---
title: Actor-objective & routing audit V1
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PHASE_CONDITIONING_MISSING + TRUST_REGION_METRIC_MISMATCH
tags: [coin, td3, audit, phase-conditioning, trust-region, no-training]
---

# Verdict: PHASE_CONDITIONING_MISSING + TRUST_REGION_METRIC_MISMATCH

Narrow actor-objective and routing audit of the Stage-1/1b implementation. **No training.** The Stage-1b
"no improvement" result is **not sufficient** to claim a supervised ceiling / local optimum — two implementation facts
undercut that inference. Everything else is correct.

## What is correct (no defect)
- **§1 actor loss** — hardened to the exact required form `masked_actor_loss = −Σ(gate_t·Q1(s,π_late(s)))/max(Σgate_t,1)`
  and wired into both trainers. Tests prove: gate_t==0 rows give **exactly zero** π_late gradient; changing only
  gate-off observations cannot change the update; all-gate-off ⇒ zero gradient; and the masked form equals the mean over
  the gate-on subset (the equivalent sampling form the completed runs used — so those runs were not defective on this).
- **§2 gate masking** — actor masks on `gate_on` (= current **gate_t**, stored pre-step); target action uses
  `gate_on_next` (= **gate_tp1**). Correct split.
- **§3 exploration routing** — gate-off executed action equals `clip(pi_0(obs))` **bit-identically**: leakage
  `max_inf = 0.0` over 993 gate-off transitions. Gate-on = `clip(pi_late+coherent_noise)`.
- **§4 replay executed-action identity** — the critic receives the **final executed clipped** action (max|a| = 4.0,
  gate-on 100% perturbed, gate-off = pi_0); no requested/pre-clip action is stored.
- **§6 minibatch composition** — gate-on pool 255/1248 (20.4%), family fractions target_entry 0.36 / braking 0.42 /
  settling_dwell 0.22 (balanced, not degenerate); terminated 0.0, truncated 0.024.
- **§8 target init** — `online π_late = target π_late = pi_0` (fp `3a4b5d09b44f`); `critic = critic_target`
  (fp `4e0f69c24106`); no random target actor.

## Finding 1 — PHASE_CONDITIONING_MISSING
`pi_late` and both critics received **`obs_48` only** — no `phase_id`/`phase_onehot`. The completed Stage-1/1b was a
**binary early/late controller** (a single `pi_late` for target_entry, braking, and settling_dwell alike), **not** a
phase-conditioned multi-state baseline. A single actor forced to serve three physically different late phases cannot
specialize; "no local improvement" from it does not imply the per-phase basins are individually at a supervised ceiling.
- **Added (not trained):** `coin_phase_conditioning.make_phase_actor_from_pi0` / `make_phase_critic` —
  `actor_input = obs_48 ++ phase_onehot`, `critic_state = obs_48 ++ phase_onehot`, every new phase-input weight
  **zero-initialized** so update-0 reproduces `pi_0` exactly for every phase (tested, tol 0.0).

## Finding 2 — TRUST_REGION_METRIC_MISMATCH
The report's "actor drift 0.095" and the trust-region cap "0.060" are **different metrics on different observation sets**:
- reported drift = `max |π_late(probe) − pi_0(probe)|` = **L∞ over a 64 random-obs probe batch**;
- the cap constrains `‖π_prop(s) − π_ref(s)‖₂` **per anchor (gate-active) state** — per-step {median 0.0025, p95 0.005,
  max 0.010}, cumulative {p95 0.030, max 0.060}.

The cap was in fact **respected**: Stage-1b's `anchor_cum_max` (the L2 metric the cap actually bounds) was **0.032 ≤
0.060** — only *half* the cumulative budget. So the actor was **not** cornered by the cumulative cap; the 18-then-reject
pattern was driven by the **per-step** bound (max 0.010), and the "0.095" probe-L∞ was never the constrained quantity. An
illustration on a uniform 0.02-bias actor confirms same-actor anchor-L2 (0.04) ≠ probe-L∞ (0.02).

## Consequence for the Stage-1b interpretation
Neither finding touches the *positive* Stage-1b result — the transactional update still eliminated the V1 divergence
(that used the same probe metric consistently for V1 vs 1b, so the 8.0 → ~0.1 comparison stands). But the **negative**
inference ("supervised ceiling / local optimum") is **withdrawn**: it rested on a phase-blind actor and a mislabeled
drift budget that was only half-used. The late-phase local-improvability question is **still open**.

## Verdict
`PHASE_CONDITIONING_MISSING`, `TRUST_REGION_METRIC_MISMATCH`. Not `ACTOR_GATE_MASK_MISSING`, not
`TRAINING_EXPLORATION_GATE_LEAKAGE`, not `REPLAY_EXECUTED_ACTION_MISMATCH` (all verified correct).

## Non-claims
NOT a supervised-ceiling claim (withdrawn). NOT a training result. The phase conditioning is added + tested but **not
trained**. No Stage 1c, no trust-region widening, no reward change, no Arm B, no SAC, no final-test seeds.

## Next narrow experiment (needs your go)
Re-run the transactional Stage-1 with (a) **phase conditioning** (the zero-init actor/critic added here) and (b) a
**drift budget reported in the metric the cap constrains** (anchor-L2 per-step + cumulative, printed as
median/p95/max), so "budget used vs available" is legible. Since the cumulative cap was only half-used, the
per-step cap (0.010) is the current binding constraint — consider whether to relax it before concluding anything about
local improvability.

## Files
- impl (this change): `coin_phase_conditioning.py`, `coin_td3_trainer.py` (masked_actor_loss + sample_actor_batch),
  `coin_td3_transactional.py` (masked form), `hymeko_rl/tests/test_coin_actor_audit.py`, updated
  `test_coin_td3_transactional.py`; audit `experiments/…/audit_actor_routing_v1.py`.
- results: `experiments/…/actor_routing_audit_v1.json`, this report.
- upstream: Stage-1b `b53abe6`, transactional impl `91e1031`.
