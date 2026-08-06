---
title: Receding-horizon action-chunk TD3 V1 — contracts + supervised baseline
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: CHUNK_WARMSTART_AND_EXECUTION_CONTRACTS_PASS
tags: [coin, td3, action-chunk, receding-horizon, semi-mdp, supervised-baseline, no-td3]
---

# CHUNK_WARMSTART_AND_EXECUTION_CONTRACTS_PASS — TD3 not started

Contracts A–E for `RECEDING_HORIZON_ACTION_CHUNK_TD3_V1`. **No TD3 fine-tuning** (deferred per instruction until the
chunk warm-start + execution contracts pass — they do). The pivot: optimize a temporally-coherent **action chunk**, not
an isolated action perturbation.

## A. Chunk actor + twin critic (tested)
`pi_chunk(state 62) → chunk (K=8, 4)` clip-squashed to [-4,4]; each chunk step **independently predicted** (NOT a
residual hold). Twin `Q_i(state 62, chunk 32)`. State = obs_48 ++ onehot3(control_mode) ++ onehot2(contact) ++ event5
++ prev_action4 (causal history). M=2.

## B. Semi-MDP M-step target (tested)
`y = Σ_{j<M'} γ^j r_{t+j} + γ^{M'}·mask·min(Q1_t,Q2_t)(state_{t+M}, chunk_target)`; M' = actual executed prefix;
**terminated ⇒ no bootstrap, truncated ⇒ bootstrap** (verified on all three cases).

## C. Receding-horizon execution (tested)
Execute only the first M=2 planned actions with gate routing: gate-on ⇒ chunk step; **gate-off ⇒ frozen pi_0
bit-identical** (verified against a forced gate-off); then replan. `execute_chunk` returns the requested chunk, the
executed prefix, per-step records (action/reward/term/trunc/gate_on), and the next observation.

## D. Warm-start dataset (manifests)
The actor learns the **first K actions of the feedback rollout from each state** (not a time-indexed trajectory).
Sources: **A** frozen pi_0 rollouts (pi_0 rehearsal), **B** the receding-horizon **planner CEM** (`plan_chunk`,
WARM_START_ONLY — no planner runs at eval), **C** certified open-loop continuations (declared REPLAY_ONLY, not
integrated this turn). Dataset from the frozen transport-dwell banks: **59 examples** (20 planner + 39 pi_0),
sha `87b1aa95`.

## E. Supervised chunk baseline (measured — a cautionary starting point)
Supervised regression converges tightly: **sequence MSE 0.0003, first-action MSE 4e-05, two-step-prefix MSE 4e-05**
(per-index MSE ≤ 0.001). But executed **receding-horizon** on 31 disjoint dev states it is **worse than pi_0**:

| metric | chunk | pi_0 | Δ |
|---|---|---|---|
| strict K6 | 0.129 | 0.194 | −0.065 |
| max dwell | 0.97 | 1.48 | −0.52 |
| target-entry rate | 0.39 | 0.52 | −0.129 |
| **contact retention** | **0.10** | **0.60** | **−0.496** |
| target exit | 0.23 | 0.03 | +0.194 |

The actor imitates the warm-start almost perfectly, yet open-loop-M execution **collapses contact** (0.10 vs 0.60). The
20 **planner CEM** targets optimize strict/dwell/entry by *pushing* the coin — trading away the bilateral contact that
pi_0's per-step feedback preserves — and the mixed warm-start inherits that. This is the honest baseline TD3 starts from,
and it flags a **warm-start-quality lever**: the planner-sequence share must be filtered to contact-preserving maneuvers
(or down-weighted) or the critic must penalize contact loss hard.

## Verdict
Contracts A–E pass; the supervised baseline is measured (underperforms pi_0, contact-collapsing). Per instruction, TD3
fine-tuning is **not** started.

## Claims / non-claims
**Claims:** (1) Chunk actor/critic, semi-MDP M-step target (term/trunc correct), and receding-horizon execution
(gate-off = pi_0 bit-identical) contracts hold (8 tests). (2) Supervised regression on the pi_0+planner warm-start
converges (seq MSE 3e-4). (3) The receding-horizon supervised baseline underperforms pi_0, collapsing contact
(0.10 vs 0.60) — the planner CEM warm-start trades contact for progress.
**Non-claims:** NOT a TD3 result (deferred). NOT that chunk control cannot beat pi_0 — the supervised baseline is the
*start*, and the contact collapse is a fixable warm-start/critic issue. No neutral-reset/final-test; no planner at eval.

## Next narrow experiment (needs your go)
Before TD3, fix the warm-start: (a) **filter planner sequences to contact-preserving** ones (require contact retained
over the K-step CEM rollout) and/or raise the pi_0-rehearsal share; then re-evaluate the supervised baseline. Only once
the supervised chunk baseline is not contact-collapsing does TD3 fine-tuning (semi-MDP twin critics + sequence-space
trust region on the executed prefix) become worth running.

## Files
- impl (this commit): `hymeko_rl/coin_delivery/coin_chunk_td3.py`, `hymeko_rl/tests/test_coin_chunk_td3.py`,
  `experiments/…/coin_chunk_warmstart_v1.py`; generalized `coin_phase_conditioning` (n_cond, prior commit).
- results: `experiments/…/chunk_warmstart_v1.json`, this report.
- upstream: transport-dwell `c1da225`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Mac, torch 2.12.0, mujoco 3.10.0. Planner CEM
deterministic per seed; supervised seed 0. No planner during evaluation.
