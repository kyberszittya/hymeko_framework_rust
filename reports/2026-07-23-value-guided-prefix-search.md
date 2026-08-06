---
title: Value-guided prefix search — pi_0 is near-optimal on boundary states, so there is almost nothing to convert
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: NO_SEARCH_CONVERTS_PI0_NEAR_OPTIMAL_ON_BOUNDARY — value PRESERVES pi_0 (net ΔK6 0, low drift); reward WANDERS and slightly hurts; no headroom to demonstrate conversion
tags: [coin, markov, value-guided-search, receding-horizon, mpc, prefix, three-scorer-ablation, per-state]
---

# VALUE_GUIDED_PREFIX_SEARCH_V1 — can the validated value be converted by a bounded critic-guided search?

Following POST_PREFIX_TERMINAL_VALUE_ORDERING_CONFIRMED + ONE_STEP_VALUE_TO_ACTION_CREDIT_UNRESOLVED, we test whether a
short (K=2), receding-horizon search — candidates ranked ONLY by the learned signal — converts the value into a real K6
gain over pi_0.

## Method (safeguards)
- **Critic-independent generator**: {pi_0} ∪ {pi_0 ± ε·e_i} (ε=0.08, support-bounded, reported). Three receding-horizon
  controllers, executing only the first action then replanning, ranking the SAME candidates by three scorers:
  **REWARD_ONLY** (Σγ^t r_t), **BOOTSTRAP_VALUE_ONLY** (γ^K Q_target(s_K, π_target(s_K))), **REWARD_PLUS_VALUE**.
- The exact simulator is used ONLY for the bounded K=2 lookahead; the terminal value is the LEARNED Q_target (trainer
  Bellman contract: target critic + target actor) — never a full exact-rollout-to-episode-end oracle.
- **Physics-primary** (K1/K3/K5/K6, dwell, containment exit, speed, effective drift vs pi_0, replay-support distance).
- **Per-STATE unit + hierarchical bootstrap over seed×state**; matched states/seeds; ΔK6 classified vs 0 with an
  equivalence band (a CI merely containing 0 is INCONCLUSIVE, never "defective"); underpower gate; **both Arm-A and
  Arm-B** critics; 3 seeds. Held-out robust-pair boundary states (a chunk reaches K6, another clearly fails).

## Result (30 held-out robust-pair states, 3 seeds, both arms; hierarchical CI over seed×state)
**pi_0 K6 = 0.967** — pi_0 already delivers on 29/30 states. Only 1 state is pi_0-fail.

| scorer | K6 rate | net ΔK6 vs pi_0 | class | non-pi_0 | cum drift | reading |
|---|---|---|---|---|---|---|
| REWARD_ONLY | 0.933 | **−0.033** CI[−0.10, +0.03] | INCONCLUSIVE | 0.64 | 0.195 | wanders off pi_0, slightly loses |
| BOOTSTRAP_VALUE_ONLY | 0.967 | **0.000** CI[0, 0] | EQUIVALENT | 0.21–0.25 | 0.04 | **preserves** pi_0 (picks pi_0 ~79%) |
| REWARD_PLUS_VALUE | 0.967 | 0.000 CI[0, 0] | EQUIVALENT | 0.75 | 0.13 | preserves K6 despite exploring choices |

(Both arms identical to two decimals — Arm-A/Arm-B agree.) pi_0-fail subset (n=1, not powered): reward converts it,
value/reward_value do not.

**Verdict: `NO_SEARCH_CONVERTS_PI0_NEAR_OPTIMAL_ON_BOUNDARY`.** No scorer nets a K6 gain, and none clearly hurts. The
decisive fact is the test bed: **pi_0 is near-optimal at these boundary states (K6 0.967)** — there is almost no headroom
to convert value into K6 gain. (Pure pi_0-fail states are ~0.8% of boundary — too rare to power on their own; this is why
the improvement lens has n=1.)

## What the scorers reveal (the ablation earns its keep)
- **BOOTSTRAP_VALUE_ONLY preserves** — it stays close to pi_0 (drift 0.04, non-pi_0 0.21), keeping K6 at 0.967 = pi_0.
  The value does not *destroy* pi_0; used conservatively it correctly ranks pi_0 as best where pi_0 is best. But it is
  also too conservative to rescue the 1 pi_0-fail state.
- **REWARD_ONLY wanders** — dense-reward-guided search drifts far off pi_0 (drift 0.195, non-pi_0 0.64), rescuing the 1
  fail state but slightly losing elsewhere (net −0.033). So naive reward-guided search can *hurt* a near-optimal baseline.
- **REWARD_PLUS_VALUE** ≈ value (preserves K6) — the value keeps the combined score from wandering as far as reward alone.

## Honest reading (not over-claimed)
This is **not** a clean "value converts / does not convert" answer, because the boundary test bed has essentially no
improvement headroom — pi_0 already solves it. The result that *is* clean and powered: on held-out robust-pair boundary
states, **a bounded value-guided prefix search does not beat pi_0 (net ΔK6 ≈ 0), the value-only scorer preserves pi_0
safely, and reward-only wanders and slightly hurts.** The conversion question cannot be decided here because there is
little to convert; it needs a test bed with genuine pi_0 headroom.

This coheres with the arc: LOCAL operations around pi_0 cap at the supervised ceiling, and pi_0 IS near the ceiling at
these boundary states (only NONLOCAL search exceeded it, ledger R59). The value's post-prefix consequence recognition
(POST_PREFIX_TERMINAL_VALUE_ORDERING_CONFIRMED) is real but has no local headroom to act on here.

## Next lever (needs your go)
The bounded-search causal test is not yet decisive on conversion — because of the test bed, not the method. Options:
1. **A headroom test bed** — earlier states (approach / braking, strict 1–2) or a harder start distribution where pi_0
   genuinely underperforms, so a value-guided search has room to improve; re-run this exact 3-scorer receding search.
2. **If headroom is found and value still can't convert** → the chunk critic learning `Q_K(s, a_0:K-1)` directly becomes
   justified (as a compression of a search that would then be proven to work, per your logic).
Trust region / certificate gate stay. NOT full TD3/SAC.

## Files
- lib: `coin_markov_ablation_train.py` (`receding_horizon_rollout` — arm-independent certifier physics; per-step
  drift/state capture); reuses `prefix_candidate_rollout`, `train_arm(critic_target)`, `classify_vs_chance`,
  `hierarchical_bootstrap_ci`, `build_boundary_panel`.
- entry: `experiments/…/rl_entry/coin_value_guided_prefix_search.py`; results
  `…/value_guided_prefix_search_v1.{json,svg,png}`.
- tests: `test_receding_horizon_rollout_pi0_select_and_info` (pi_0-select ≡ frozen pi_0, info enrichment, one-call-per-step).
  24 tests pass, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; critics Arm A+B 4000 updates, seeds {0,1,2},
single-thread; bootstrap = target critic + target actor. Panel = 30 held-out robust-pair boundary states (scanned 254,
seeds ≥6200; build_boundary_panel exhausted the range and fell back to include strict-2). K=2, mag=0.08, H=20. No new
campaign, no reward/task change, no CORE.YAML items.
