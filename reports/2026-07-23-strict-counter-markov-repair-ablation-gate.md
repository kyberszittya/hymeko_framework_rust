---
title: Strict-counter Markov repair ablation — pre-training gate
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: MARKOV_REPAIR_GATE_PASS — matched TD3 campaign authorized (integration next)
tags: [coin, markov, strict-counter, ablation, td3, gate, reward-variant]
---

# STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — pre-training gate PASSES

The 5 tests required *before* training are implemented and pass. Arm A/B primitives are built; the canonical v3 reward
file is **not** overwritten (Arm B composes a variant spec). pi_0, physics, certifier, and all non-terminal reward terms
are unchanged.

## Arms
- **Arm A** — expose the exact strict counter to actor/critic/replay/targets; canonical reward + termination preserved
  bit-exact (still fires the +30 at the K5 off-by-one step).
- **Arm B** — same exact-counter exposure; the graded +30 delivery bonus moves onto the **K6 terminal transition**, paid
  once (latched), so it cannot be farmed by reaching K5, breaking dwell, and rebuilding.

## Primitives (`hymeko_rl/coin_delivery/coin_strict_markov_ablation.py`)
- `strict_onehot(strict, k=6)` — exact one-hot of the dwell counter (7 dims), distinct for every value 0..6.
- `augment_with_strict(state, rl)` — appends the counter one-hot to the actor/critic/replay/target state (obs48 → 55).
- `arm_b_terminal_bonus(strict, touched, bonus_paid, grade)` — pure: `+30·grade` once, only at K6 (strict≥6 ∧ touched).
- `CoinRL4DofAblation(arm)` — Arm A = canonical bit-exact; Arm B = variant spec (v3 minus `terminal_deliver_graded`) +
  latched K6 bonus.

## Gate tests (all pass — `test_coin_strict_markov_ablation.py`)
1. **strict 0..5 produce distinct state inputs** — the 6 augmented states are distinct (55-dim, exact-counter appended).
2. **identical physical state + distinct strict not aliased** — for a fixed obs, strict 4 vs 5 give different inputs; the
   physical part is unchanged.
3. **reward + termination exactly reproduced (Arm A)** — Arm A rollout is bit-identical to canonical `CoinRL4Dof`
   (reward, termination, strict) from the same reconstructed handoff.
4. **Arm B bonus once, at K6, terminal** — 0 at K5, +30·grade at K6 (latched, not paid twice, requires robot attribution).
5. **no K5 farming** — the exploit sequence (dwell→K5, break, rebuild→K5, then K6) yields exactly one +30, only at K6.

## Matched campaign design (both arms, next step)
- initialise from the same pi_0; the augmented actor is pi_0 with the 7 strict-input weights zero-initialised
  (update-0 ≡ pi_0); same transactional-TD3 config, banks, seeds, caps; one bounded matched run.
- report **CONTINUATION_STRICT** (inherit `handoff_strict`) and **RESET_AT_HANDOFF_STRICT** (strict=0 at the boundary)
  separately for each arm, never mixed.
- no SAC / chunk / planner / primitive / final-test seeds; no other reward term changed.

## Integration note (honest)
The transactional-TD3 loop threads the raw 48-dim obs through `make_late_actor_from_pi0`, `LateTwinCritic`,
`LateReplayBuffer`, `collect_late_episode`, and `eval_late_controller`; the 55-dim augmentation + the Arm A/B env must be
wired through all of them, and a matched two-arm multi-seed run is long compute. The gate above is the verified
precondition; the training integration + bounded run is the next action (not rushed into an unreliable single-pass).

## Files
- impl: `hymeko_rl/coin_delivery/coin_strict_markov_ablation.py`, `hymeko_rl/tests/test_coin_strict_markov_ablation.py`.
- commit: gate `35c1dc52`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; canonical v3 reward NOT modified. No CORE.YAML items.
