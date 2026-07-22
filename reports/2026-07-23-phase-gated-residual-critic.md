---
title: PHASE_GATED_RESIDUAL_CRITIC — CRITIC_NO_USEFUL_LOCAL_RANKING (actor update NOT authorized)
date: 2026-07-23
slug: phase-gated-residual-critic
task: coin_v3 delivery — phase-gated learned-residual TD3 (§6)
verdict: CRITIC_NO_USEFUL_LOCAL_RANKING
authorizes_actor_update: false
---

# Composite-action critic authorization — measured negative

**Created-at:** 2026-07-23 04:55 CEST
**No actor training was run.** This stage validates critic semantics and asks whether the critic can locally rank
composite actions well enough to authorize a residual actor update. **It cannot, at this configuration** — the gate
does **not** pass and the actor update is **not** authorized.

## Infrastructure (validated, correct)

- **Encoder** `PHASE_GATE_CONTROLLER_STATE_ENCODER_V1` (fingerprint `d6301d06`): deterministic [gate, mode one-hot,
  normalized counters, side one-hot], only stored causal fields, no strings/target/success/future (8/8 critic tests
  incl. `test_encoder_only_stored_fields`, `test_distinct_modes_distinct_encoding`).
- **Critic** `CompositeTwinCritic` (`423d7699`): twin Q(obs 48, **composite** action 4, encoded state 11), independent
  params; consumes the deployed clipped composite action (never residual-only).
- **Target**: built from stored `gate_tp1` via the `9fa35a4` contract (residual target = 0 ⇒ target action = base);
  no fresh FSM; base grads `None`; frozen base absent from optimizers.
- **Panels**: train `6000–6059`, auth `7000–7039`, **final `7060–7075` SEALED** — pairwise disjoint by seed.
- **Ground truth**: counterfactual rollouts (restore state → apply candidate → continue frozen π₀ → realized
  discounted v3 return). Verified deterministic (r1==r2 exactly) and candidate-sensitive.

## Authorization result — ranking ≈ chance in every family

| family | n | mean pairwise rank acc | harmful-reject |
|--------|---|------------------------|----------------|
| transport | 8 | **0.500** | 1.00 |
| entry | 6 | **0.536** | 0.50 |
| settling | 3 | **0.543** | 1.00 |
| contact_retention | 8 | **0.571** | 0.875 |

No family reaches a reliable-ranking threshold (0.6); the best (contact_retention) is 0.571. **Boundary diagnostics
are clean**: `boundary_pref_rate 0.0`, |residual-norm|-vs-Q correlation 0.05 — the critic has **no** OOD preference
for residual magnitude or action saturation (rules out `CRITIC_BOUNDARY_PREFERENCE`). Encoder distinguishes modes
(rules out `CRITIC_PHASE_ALIASING`); panels disjoint (rules out `CRITIC_PANEL_LEAKAGE`); transport does not pass so
it is not `CRITIC_CONTACT_BLIND`.

## Mechanism (the discriminating test was run)

The realized-return **signal exists** — candidate spreads reach **85** at contact_retention (a 0.25 residual *can*
break a fragile grip), 6–47 at transport/entry. But the critic's **action-induced Q-spread is ~2**, at or below its
own **value-estimation noise floor**:

| run | LR | noise | steps | twin disagreement | ranking |
|-----|----|----|------|-------------------|---------|
| 1 | 3e-4 | 0.15 | 12k | 1.95 | transport 0.36 (panel underpopulated) |
| 2 | 3e-4 | 0.35 | 20k | 5.07 | ~chance |
| 3 | 3e-4 | 0.35 | 40k/512 | **15.2 (diverging)** | ~chance |
| 4 | **1e-4 + grad-clip** | 0.20 | 30k | **4.81 (stable)** | ~chance |

Run 3 showed the twin disagreement **diverging** with more training (value inflation); run 4's stabilization
(lower LR + gradient clipping + action scale matched nearer the residual bound) brought it back to 4.8 **but the
ranking stayed at chance**. So the chance-level ranking is **not** mere under-convergence — a *stable* critic still
cannot resolve the value difference of a within-bound (0.25) residual over a competent base, because the state value
dominates and the action's marginal contribution sits below the critic's own Q-noise.

## Decision — `CRITIC_NO_USEFUL_LOCAL_RANKING`

The critic does not reliably rank helpful composite actions above harmful ones in the residual-enabled phases. **No
residual actor update is authorized.** A TD3 actor step against this critic would follow a ~chance action-gradient.

## Limitations / non-claims (honest scope)

This is a negative for the **standard TD3 composite-action critic at the 0.25 residual scale over a competent base**.
It is **not** a proof that residual improvement is impossible: I did **not** test alternative critic targets
(advantage/A2C-style that isolates the action-marginal, n-step returns, or a residual-marginal critic that removes
the dominant state value). The infrastructure (encoder, twin critic, stored-gate target, counterfactual labeling) is
correct and reusable; the blocker is the critic *target formulation*, which the next stage should redesign rather
than proceeding to an actor update. This is consistent with the earlier coin finding that only **nonlocal
exact-rollout search** — not local policy-improvement — exceeded the supervised ceiling ([[project-coin-toss-genuine-rl-accelerated]]).

## §6.15 regression — unchanged

Update-0: HL **3/9**, VAL **2/30**, grasp **9/9**, delivered **{1011,1447,1568}**, composite−base maxdiff **0.0**,
π₀ hash unchanged. All tests green: **52/52** (44 prior + 8 critic). Ruff crit clean.

## Files touched

- `hymeko_rl/coin_delivery/coin_residual_critic.py` (new, ~90 L) — encoder + composite twin critic.
- `hymeko_rl/tests/test_coin_residual_critic.py` (new, 8 tests).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_residual_critic_authorize.py` + `plot_critic_auth.py`
  + `critic_auth.json`.
- `reports/figures/coin_residual_critic_auth.png`.

**CORE.YAML:** none. Frozen π₀ (`1902454c`) / reward / gamma / bundle / obs / gate thresholds / residual range /
action bounds unchanged. **No actor optimization performed.** SAC quarantined. Final-test bank untouched; final
panel `7060–7075` sealed. Mac; kato14 clean.

## Exact next experiment (NOT an actor update)

Redesign the critic target to isolate the action-marginal (advantage/n-step/residual-marginal critic) and re-run
this same authorization panel + counterfactual labels. Only a critic that ranks reliably (≥0.6) in transport +
contact-retention (or a clearly delimited subset for a scoped pass) may then authorize a first, tiny, fixed-budget
controlled residual-only TD3 update.
