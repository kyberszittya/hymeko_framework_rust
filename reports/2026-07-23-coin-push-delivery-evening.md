---
title: Coin push-delivery evening — RESIDUAL_CRITIC_DEVELOPMENT_HARNESS_INVALIDATED
date: 2026-07-23
slug: coin-push-delivery-evening
task: coin_v3 delivery — PHASE_GATED_LEARNED_RESIDUAL_CONTROLLER integration
verdict: RESIDUAL_CRITIC_DEVELOPMENT_HARNESS_INVALIDATED
sealed_panels_opened: none
policy_final_test_bank: untouched
---

# Evening critic-development harness — INVALIDATED (does not test the declared controller)

**Created-at:** 2026-07-23 04:20 CEST. The first-pass critic-development harness does **not** exercise the declared
`PHASE_GATED_LEARNED_RESIDUAL_CONTROLLER` or the preregistered advantage fallback. Its `RESIDUAL_CRITIC_ROUTE_BLOCKED`
conclusion is **withdrawn** and reclassified `RESIDUAL_CRITIC_DEVELOPMENT_HARNESS_INVALIDATED`. Scripts, JSON and logs
are preserved **only as diagnostic artifacts**. No actor update; every sealed panel closed; policy final-test
`8000–8049` untouched.

## Blocking harness mismatches (recorded)

- **A. Behavior collector off-contract.** It executed `action = clip(pi_0(obs) + N(0,0.20), −4, 4)` at *every* phase
  — a full-action Gaussian perturbation, not the gated residual `action = clip(pi_0 + gate·clip(δ, ±0.25), ±4)`. It
  perturbs approach/acquisition, ignores the gate, does not clamp to the residual bound, changes the 9/9 grasp-
  producing early policy, and collects actions outside the deployable residual-controller distribution.
- **B. `truncated` dropped.** The transition tuple and Bellman mask used only `terminated`, so it may bootstrap across
  time-limit truncations.
- **C. Cross-state ranking.** The advantage ranking loss compared independently sampled examples that may come from
  different restored physical states; paired advantages must be compared **within the same state**.
- **D. Instantaneous critic state.** The critic saw only obs_48 + gate encoding; `node_features` lacks coin velocity,
  which transport/contact/settling need — so state-sufficiency was never actually controlled.
- **E. Mechanism-balanced datasets not integrated.** The manifest listed the permitted REPLAY_ONLY corpora but the
  training used only 60 randomly-perturbed π₀ episodes.
- **F. Ad hoc development gate.** The implemented `ck_pass` is a reduced gate, not the frozen evening metric suite.

**Additional discipline gap.** The run was on the Mac, though the evening mission named kato14 as primary; a parity
certificate (or a kato14 run) is required before any corrected scientific verdict.

## What the diagnostic artifacts still (weakly) suggest — NOT a verdict

Under the invalid off-contract harness: the advantage critic fit the one-step marginal in-sample (+0.994) but did
not generalize from obs_48 (held-out +0.06, unchanged at 6× data). This only *motivates* the causal-state contract
and the controlled state-sufficiency ablation below — it does **not** establish `RESIDUAL_CRITIC_ROUTE_BLOCKED`,
because the collector, mask, ranking, state, dataset and gate were all off-contract.

## Corrected plan (this and following turns)

1. `RESIDUAL_CRITIC_STATE_V2` = `FULL_ACTION_OBS_HISTORY_V1` (152) + encoded `PHASE_GATE_CONTROLLER_STATE_V2` — for
   the critic only; π₀ keeps its 48-dim obs → `RESIDUAL_CRITIC_CAUSAL_STATE_CONTRACT_PASS`.
2. Gated residual behavior collector (bit-identical to π₀ when gate=0 / residual off) →
   `PHASE_GATED_RESIDUAL_BEHAVIOR_CONTRACT_PASS`.
3. Mechanism-balanced V2 datasets (rescored v3 reward, provenance) + grouped counterfactual candidates
   (`state_group_id`, fixed residual scales 0/0.01/0.025/0.05/0.10/0.25) + within-state twin advantage loss.
4. Controlled state-sufficiency ablation (instantaneous vs causal history) →
   `RESIDUAL_CRITIC_STATE_ALIASING_CONFIRMED` / `RESIDUAL_CRITIC_CAUSAL_STATE_NO_GAIN`.
5. Frozen V2 development gates; kato14 run or parity certificate.

The counterfactual label (one residual action then π₀ continuation) is a `ONE_STEP_RESIDUAL_IMPROVEMENT_CRITIC` —
it may authorize only the first bounded residual step; relabel under the updated continuation before multi-step TD3.

**CORE.YAML:** none. Frozen artifacts unchanged. No actor update; sealed/final panels + policy final-test never
opened; SAC quarantined. Mac (kato14 unused — parity required for the corrected run).
