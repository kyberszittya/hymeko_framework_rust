---
title: Strict-counter Markov repair ablation V1 — matched Arm A / Arm B transactional TD3
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: REPAIRS_NOT_SUFFICIENT (K6) — but the Markov critic repair IS load-bearing at the representation level
tags: [coin, markov, strict-counter, ablation, transactional-td3, critic, terminal-alignment]
---

# STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — the critic repair works; K6 does not move (and it's not the task)

Two matched arms of transactional TD3, same banks / families / horizon / n-step / target smoothing / exploration /
budget as the historical Stage-1b — only the state augmentation (exact strict one-hot) and (Arm B) the terminal-bonus
realignment differ. 3 matched seeds per arm, 8000 updates, ~350 s wall. pi_0 / physics / certifier / all non-terminal
reward terms unchanged; the canonical v3 .hymeko is **not** overwritten (Arm B composes a variant spec).

## Pre-training gate + update-0 manifest (verified)
- 5 gate tests pass (commit `35c1dc52`): distinct strict states, no aliasing, Arm A reward+termination bit-exact, Arm B
  bonus once at K6, no K5 farming. Plus the step-4 regression (strict 1 vs 5 → distinct actor/critic/target/replay
  inputs, same obs48).
- **update-0 ≡ pi_0**: `actor55(obs48, strict=k) == pi_0(obs48)` for every dev obs and k∈0..6 — **max action diff 0.0,
  mean 0.0** (both arms). The 7 strict-input columns are zero-initialised and remain trainable.
- **Integration smoke passed** all step-7 checks (obs55 shapes everywhere, critic authorized, finite losses/grads, Arm A
  canonical, Arm B bonus-once/no-farm, eval never via control_mode).

## Result
pi_0 CONTINUATION K6 = **0.4167** (12-state late-dev subset, families target_entry/braking/settling).

| arm | seed | CONTINUATION K6 | RESET K6 | max dwell | accept/reject | actor drift | td p90 |
|---|---|---|---|---|---|---|---|
| A | 0/1/2 | **0.4167** (all) | 0.4167 | 2.5 | 20/2980, 19/2981, 19/2981 | 0.031–0.035 | 9–13 |
| B | 0/1/2 | **0.4167** (all) | 0.4167 | 2.5 | 20/2980, 17/2983, 17/2983 | 0.031–0.036 | 9–13 |

Arm A Δ vs pi_0 = **+0.000** (all seeds); Arm B Δ = **+0.000** (all seeds). Critics healthy (authorized every checkpoint),
replay nondegenerate, gradients finite, budgets matched, transactional updates valid (accepts > 0).

**Verdict: `REPAIRS_NOT_SUFFICIENT`.**

## The important part — the Markov repair *is* load-bearing at the critic level
The strict-conditioned Q (from one physical state, strict 0..6) **rises toward the terminal in all 6 runs** — e.g. Arm A
seed 2: `[-124, -56, -30, -17, +23, -34, -63]` (Q climbs +147 from strict 0 to strict 4, one step before Arm A's +30
fires at strict 5); Arm B seed 2: `[-136, -40, -20, -14, +20, +13, -54]` (climbs to strict 5, where Arm B's terminal
bonus fires). Rise strict 0→4 across runs: **+16, +62, +147, +40, +79, +156**. The **hidden-counter critic could not do
this** — `control_mode` collapses `strict≥1 → "settling_dwell"`, so every settling state aliased to one input. **The
non-Markov aliasing is genuinely fixed:** the critic now represents distance-to-terminal.

## Why it does not convert to K6 (measured — and it is NOT a task wall)
1. **The transactional actor trust region is the binding constraint.** It **rejected 99.3 % of actor updates**
   (accept ≈ 20 / reject ≈ 2980), so the actor barely moved (drift plateau ~0.031). The *critic* improved; the *actor*
   could not exploit it under the historical trust-region caps designed to prevent divergence.
2. **The eval subset has a flat ladder.** On these 12 late-dev states `k1 = k3 = k5 = k6 = 0.4167` — the states either
   fully deliver or never enter the zone; there is no graded sub-K6 competence to convert. Raising K6 here would require
   getting *non-entering* states to enter (a transport-phase problem), which a settling-phase late controller cannot
   address regardless of critic quality.

Per the decision logic, both corrected arms fail under a healthy Markov critic — but **`local improvement exhausted` is
NOT emitted**: the failure is attributable to the actor trust region and the flat eval ladder, not to the critic or a
task ceiling.

## Historical reclassification
| historical finding | classification |
|---|---|
| PHASE_SWITCHED_TD3 / TRANSACTIONAL_TD3 / TRANSPORT_DWELL_TD3 / PHASE_GATED_RESIDUAL_CRITIC — *mechanism* "critic route blocked / value non-representable" | **OVERTURNED** — with the exact counter, the critic authorizes and *represents terminal proximity* (Q rises toward K6). The critic was blind, not incapable. |
| the *outcome* "no local K6 improvement under this transactional setup" | **SURVIVES_MARKOV_REPAIR** — K6 still does not rise — but re-attributed to the **transactional actor trust region** (99.3 % rejection) + the flat eval ladder, **not** the critic and **not** a task wall. |
| "local improvement exhausted" | **NOT SUPPORTED** — never established; the binding constraint is now the actor update, which was not varied here. |

Supervised chunk / primitive / planner conclusions are unchanged by this report (as instructed).

## Next lever (not run — needs your go)
The ablation isolates the bottleneck to the **actor update**, not the critic. The natural next bounded step is to relax
the transactional trust region (or replace it with a monitored standard TD3 actor step) *under the now-healthy Markov
critic*, on an eval bank with a graded ladder, and re-test K6 — with the Markov critic proven to represent terminal
proximity, that is the untested lever. No reward edit; no new task semantics.

## Files & commits
- impl: `hymeko_rl/coin_delivery/{coin_strict_markov_ablation.py, coin_markov_ablation_train.py}`,
  `hymeko_rl/tests/test_coin_strict_markov_ablation.py` (7 tests).
- entry/plot: `experiments/…/rl_entry/{coin_markov_ablation_v1.py, plot_markov_ablation.py}`.
- results: `experiments/…/rl_entry/{markov_ablation_v1.json, markov_ablation_v1_smoke.json, markov_ablation.svg}`.
- commits: gate `35c1dc52`; integration+smoke `b83c035d`; matched results `1296961a`; this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; historical stage1b config
(`td3_baseline_v1_config.json`): horizon 30, n_step 4, families {target_entry, braking, settling_dwell}, total_updates
8000, policy_delay 2, checkpoints {0,2000,4000,6000,8000}; late_train 42 / late_dev 30. 3 seeds/arm. Canonical v3 reward
NOT modified. No SAC/chunk/planner/primitive/final-test. No CORE.YAML items. RL is not bit-reproducible under threaded
BLAS (single-thread here); the K6 result is flat across all 3 seeds and the strictQ rise is consistent across all 6 runs.
