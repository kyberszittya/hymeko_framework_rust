---
title: Divergent-K6 prefix-value audit — the critic's VALUE ranks divergent-K6 consequences above chance
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: POST_PREFIX_TERMINAL_VALUE_ORDERING_CONFIRMED + ONE_STEP_VALUE_TO_ACTION_CREDIT_UNRESOLVED (review-refined from CRITIC_VALUE_RANKS_DIVERGENT_K6_CONSEQUENCE) — bootstrap-Q above chance on both arms (value recognises the better post-prefix consequence, not just reward), but first-action-Q ≈ chance (that knowledge is not reliably credited back to the starting action)
tags: [coin, markov, critic-value, divergent-k6, prefix-value, per-state, hierarchical-bootstrap, decomposition]
---

# DIVERGENT_K6_PREFIX_VALUE_AUDIT_V1 — does the critic's value rank a physically-divergent consequence?

V2 showed a single trust-scale action never moves K6/dwell, so the fidelity question was inconclusive on what matters.
This test manufactures the identifiability by using a short, fixed, critic-INDEPENDENT action prefix to create genuinely
divergent K6/dwell outcomes, then asks whether the critic's value ranks the better one higher — with the reframe and every
statistical safeguard the review required.

## Reframe (why it is an *audit*, not a one-step ranking test)
The score `G = Σγ^t r_t + γ^K Q(s_K, π_target(s_K))` includes the observed prefix reward, so a high G can be reward-driven,
not the critic's merit. And `Q(s0,a0)` assumes the critic's *own* continuation policy follows `a0` — but here K−1 artificial
offset actions follow, so a first-action-Q failure is **continuation mismatch, not a critic defect**. So this audits whether
the critic's **value** (Bellman-target bootstrap) + reward can value a physically-produced prefix; first-action-Q is a
diagnostic only.

## Method (all safeguards)
- **Critic-independent generator**: fixed K-step matched-norm actuator-offset prefixes, then frozen pi_0.
- **Physics first → matched, robust, primary-certifier divergent pairs** (K6 | Δdwell | containment, margin ≥2 to be
  jitter-robust; matched on effective post-clamp norm) formed BEFORE any critic is consulted.
- **Decomposed score** — R_prefix (reward-only, critic-independent), **bootstrap-Q** (target critic + target actor at s_K —
  the trainer's Bellman contract), G (n-step), empirical full-return, first-action-Q (online critic, diagnostic).
- **Per-state statistical unit** — pairwise accuracy per physical state, then **hierarchical bootstrap over
  critic-seed × state** (pairs within a state are correlated; never flattened).
- **DISCOVERY vs EVAL states**; the (K,mag) chosen as the SMALLEST meeting a pre-registered robust-pair yield (→ K=2,
  mag=0.08; effective cumulative offset ≈0.15 — a deliberate, reported probe larger than the trust scale).
- **Separate Arm-A/Arm-B verdicts + cross-arm consistency**; CI-vs-chance with an equivalence band (a CI that merely
  contains 0.5 is INCONCLUSIVE, never "defective"). Underpower gate: <8 robust-paired states ⇒ verdict UNDERPOWERED.

## Result (held-out boundary panel n=144; 13/120 EVAL states robust-paired; 119 pairs; 3 seeds; hierarchical CI)
| metric | critic A | critic B | reading |
|---|---|---|---|
| **bootstrap-Q** (critic value @ s_K) | **0.730 CI[0.616, 0.832]** ABOVE | **0.778 CI[0.660, 0.875]** ABOVE | the critic's value ranks the better consequence above chance on BOTH arms |
| R_prefix (reward-only) | 0.788 ABOVE | 0.635 ABOVE | reward also ranks — so G is partly reward-driven |
| G (n-step) | 0.817 CI[0.751, 0.883] | 0.844 CI[0.771, 0.913] | reward + value combined |
| first-action-Q (diagnostic) | 0.395 | 0.435 | near/below chance — **continuation mismatch, not a critic verdict** |

**Verdict: `CRITIC_VALUE_RANKS_DIVERGENT_K6_CONSEQUENCE`.** The critic's Bellman-target bootstrap value ranks the
physically-better consequence above chance on both arms (CI-lower 0.62 / 0.66 > 0.5). This is not merely the reward: the
pure bootstrap-Q clears chance, and on **Arm B the critic's marginal is visible** — the transient Arm-B reward under-ranks
(R_prefix 0.635) while the value corrects it (bootstrap-Q 0.778, G 0.844).

## What this establishes (and does NOT)
- **DOES:** the critic's *value function* understands terminally-relevant alternatives — when a short prefix genuinely
  diverges on K6/dwell, the critic values the better resulting state higher. This confirms MARKOV_CRITIC_REPAIR
  (value ≈ terminal proximity) on held-out, robustly-divergent, per-state, powered data.
- **DOES NOT:** show the critic's *one-step action-value* ranks the first action — that test is confounded by continuation
  mismatch (first-action-Q ≈ chance is expected and non-diagnostic of fidelity).
- Coheres with the whole arc: value knows which consequences are better (MARKOV_CRITIC_REPAIR ✓) but a **one-step actor
  gradient cannot convert it** (V2: one nudge doesn't move K6; here: first-action-Q ≈ chance) — the historical result that
  only **nonlocal exact-rollout search** exceeded the ceiling (ledger R59) is the same story from the value side.

## Honest limitations
- **Low robust-pair yield** (13/120 states): small in-support prefixes rarely produce robust K6 divergence; the chosen
  mag=0.08 (eff. cumulative ≈0.15) is a deliberate, larger-than-trust-scale probe — so this is "when a moderately-OOD short
  prefix creates a robust K6 divergence," with the offset magnitude reported. 13 states × 3 seeds is powered enough for
  CI-lower > 0.5 but modest.
- G's accuracy is partly reward (R_prefix ABOVE); the critic's honest add is the bootstrap-Q (clears chance) and the Arm-B
  gap. Robust-divergence uses an integer margin ≥2 (jitter-robust proxy), not a re-roll jitter test.

## Next lever (needs your go)
Per the decision tree: since value-ranking exists when the consequence is identifiable, the remaining problem is
**converting value into policy** — not value fidelity. Options: (1) a **nonlocal / multi-step improvement operator** or
**sequence/chunk critic** that exploits the value (the one-step actor gradient provably cannot, per V2 + first-action-Q);
(2) if a paired-difference ranking critic is still wanted, it should target the *action-chunk* value, not the one-step Q.
Keep the trust region. Only after a converter demonstrably beats pi_0 on a graded held-out panel does TD3/SAC follow.

## Files
- lib: `coin_action_perturbation.py` (+`lex_key/lex_better/lex_ranks`, `primary_divergence`, `nstep_return`,
  `classify_vs_chance`, `hierarchical_bootstrap_ci`); `coin_markov_ablation_train.py` (`prefix_candidate_rollout` with
  both-arm rewards + effective-norm + full-return; `train_arm` now returns `critic_target`); `coin_late_start.py`
  (`build_boundary_panel` shared; `HandoffRecord` +strict/dtz/speed).
- entry: `experiments/…/rl_entry/coin_divergent_k6_pairs.py`; results `…/divergent_k6_prefix_value_audit_v1.{json,svg,png}`.
- tests: `test_coin_action_perturbation.py` (nstep_return, primary_divergence, lex certificate, classify_vs_chance,
  prefix_candidate_rollout both-arm/bootstrap). Full suite passes; ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; critics Arm A+B 4000 updates, seeds {0,1,2},
single-thread. Panel 144 held-out `settling_dwell` strict{3:53,4:47,5:44}, seeds ≥6200. Bootstrap = target critic + target
actor. No new campaign, no reward/task change, no CORE.YAML items.
