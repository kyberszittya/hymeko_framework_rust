---
title: Local action-ranking fidelity V1 — corrected one-step, state-wise, ε-stratified, terminal-aligned
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: STRONG_LOCAL_RANKING_FIDELITY_NOT_DEMONSTRATED + LOCAL_RANKING_SIGNAL_WEAK_AND_UNDERPOWERED (DEV, ε=0.005) — softened from NEAR_ZERO per review: ρ<0.5 excludes only the strong bar, not all useful positive ranking
tags: [coin, markov, critic-fidelity, action-ranking, one-step, state-wise, lexicographic, in-distribution]
---

# LOCAL_ACTION_RANKING_FIDELITY_V1 — does the Markov critic rank a single local action change faithfully?

The prior 2×2 measured critic action-fidelity on an every-step perturbed policy over a partly-OOD panel, graded by the
offset Arm-A return. Review flagged that as **not a local-ranking test**. This experiment is the corrected measurement,
with every safeguard the review required.

## Method (safeguards)
- **One-step candidate** — the perturbed action is applied ONLY at t=0; the frozen pi_0 baseline runs t=1…H. The
  critic-gradient direction is computed ONCE at s0. Verified by a deepcopy-fidelity test (a deepcopy'd handoff reproduces
  a fresh reconstruct bit-for-bit in obs/action/strict/dtz/speed/termination **and final simulator state**; clones are
  order-independent).
- **Arm-A and Arm-B critic-gradient candidates get their own physical rollouts** (they are critic-specific); only the
  identical actuator-basis candidates share physics (the certifier variables are arm-independent — proven: `step_ablation`
  physics/strict/termination do not depend on arm, only the reward does).
- **State-wise** — per-state Spearman(ΔQ, physical), top-1 physical win, +ΔQ sign-agreement; aggregated by **bootstrap CI
  over states**; **stratified by ε** (0.005 / 0.01 / 0.02 / 0.04 reported separately — local vs larger-local are different
  questions, never merged).
- **Pre-registered physical target** — a fixed **lexicographic terminal certificate** `max_dwell ≻ fewer target-exits ≻
  closer containment ≻ slower settle` (no post-hoc weighted scalar). Raw K6/dwell/exit/dtz/speed retained; ρ also reported
  under two predeclared sensitivity scalarizations (pure max_dwell, pure K6).
- **TRAIN-ID vs DEV-ID split** — the verdict uses **DEV-ID only** (12 held-out states); TRAIN-ID reported separately. DEV
  `target_entry` has n=1 → no family-level claim.
- Both **Arm-A** and **Arm-B (terminal-aligned)** Markov critics; 2 critic seeds each.

## Result (DEV-ID, median per-state ρ(ΔQ, lex certificate), bootstrap CI over states)
| critic | ε=0.005 | ε=0.01 | ε=0.02 | ε=0.04 |
|---|---|---|---|---|
| A seed0 | −0.05 CI[−0.28, 0.36] | −0.18 | 0.20 | 0.15 |
| A seed1 | −0.08 | 0.04 | 0.12 | −0.02 |
| B seed0 | 0.04 CI[−0.44, 0.30] | 0.13 | 0.31 CI[−0.23, 0.63] | 0.16 |
| B seed1 | −0.04 | −0.16 | −0.01 | −0.01 |

- **Local scale (ε=0.005):** median per-state ρ ≈ 0 for both critics and both seeds; the 95% CI **upper bound is 0.36 < 0.5**,
  so it *excludes* the faithfulness bar (ρ≥0.5), but it also *includes 0* and positive values — the ranking is **near-zero
  informative, not strongly anti-correlated**.
- **top-1 physical win** at ε=0.005 = 0.5 / 0.25 (A), 0.58 / 0.33 (B); **+ΔQ sign-agreement** = 0.47–0.5 (A), 0.45–0.5 (B)
  — i.e. the critic's most-preferred local candidate beats doing nothing at ~**chance**, and +ΔQ candidates improve the
  certificate at ~chance.
- **Larger ε (0.02–0.04):** occasionally positive (up to 0.31) but every CI spans 0; no consistent faithfulness.
- **TRAIN-ID reference** (not in verdict): Arm-A ε=0.005 ρ_lex 0.22 — modestly higher on states the critic trained on, still
  below the bar.

## Verdict — `STRONG_LOCAL_RANKING_FIDELITY_NOT_DEMONSTRATED` + `LOCAL_RANKING_SIGNAL_WEAK_AND_UNDERPOWERED` (DEV, ε=0.005)
Properly measured — one-step, in-distribution, state-wise, terminal-aligned — the Markov critic's **local** action-ranking
does **not demonstrate strong fidelity**: median per-state ρ ≈ 0 with the 95% CI upper bound 0.36 < 0.5, so it fails to
reach the pre-chosen ρ≥0.5 bar. But the honest scope is narrower than "unfaithful": ρ<0.5 excludes only the *strong* bar,
not all useful positive ranking; the CI includes 0 and positives, so the signal is **weak and underpowered**, not a
demonstrated inversion. Consistent across 2 seeds and both the Arm-A and Arm-B critics. (Earlier phrasing
`LOCAL_RANKING_NOT_FAITHFUL_NEAR_ZERO` softened per review — it overstated what the data holds.)

**Honest limitations (do not over-read):**
1. **Underpowered** — DEV n=12 states, 2 seeds; every CI spans 0. This excludes "faithful", but cannot resolve "near-zero"
   vs "mildly positive".
2. **Weak physical discrimination at the local scale** — at ε=0.005 the coarse certifier variables barely move (per-state
   ρ_dwell and ρ_k6 are **undefined** — one nudge rarely changes max_dwell or K6). The lexicographic ρ is therefore driven
   by the fine exit/dtz/speed tiebreakers. A single-action change is often physically inconsequential at this horizon,
   which itself limits how much a local-ranking test can conclude.

## What this refines in the arc
The 2×2's `CRITIC_ACTION_RANKING_NOT_PHYSICALLY_FAITHFUL` was measured on an every-step, partly-OOD, return-graded policy.
Corrected, in-distribution and local, the picture is milder and better-scoped: **the critic's single-action ranking is not
faithful (near-zero, below the bar), not strongly inverted.** So the binding issue is critic *informativeness*, and a fair
critic-ranking test needs more physical discrimination than one local nudge provides.

## Next lever (needs your go)
Not TD3/SAC yet (local ranking does not clear the bar). Two coupled directions:
1. **Make the critic's ranking informative** — train the value/critic to rank actions by physical terminal outcome
   directly (paired-difference / ranking-loss), or add on-distribution critic data / pessimism; re-run this exact test.
2. **Give the local test physical discrimination** — a multi-step candidate window (K>1 candidate steps, still bounded) or a
   more discriminating terminal target, so dwell/K6 can actually move and the ranking question is answerable at power.
Keep the trust region as a safety wall throughout.

## Files
- lib: `hymeko_rl/coin_delivery/coin_action_perturbation.py` (PerturbedActor, actuator/critic-grad deltas, spearman→None,
  bootstrap_ci); `coin_markov_ablation_train.py` (`one_step_candidate_outcome` with capture; additive dtz/speed eval keys).
- entry: `experiments/…/rl_entry/coin_local_ranking_fidelity.py`; results `…/local_ranking_fidelity_v1.{json,svg,png}`.
- tests: `hymeko_rl/tests/test_coin_action_perturbation.py` (8 gate + geometry + deepcopy-fidelity + order-independence).

## Test / lint
14 tests pass (`test_coin_action_perturbation.py` 8 + `test_coin_strict_markov_ablation.py` 6). Ruff F-clean on all touched
files (house style uses `;`/lambda; only F-rules enforced — no config in tree).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; critics = Arm A + Arm B, 4000 updates, seeds {0,1},
single-thread (RL not bit-reproducible under threaded BLAS). Panel = in-distribution {target_entry, braking,
settling_dwell}, train-ID 18 / dev-ID 12. Rollout horizon 30. No new campaign, no reward/task change, no CORE.YAML items.
