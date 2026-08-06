---
title: Local action-ranking fidelity V2 — powered, held-out, boundary; the primary certifier is unmoved by a single local action
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: PRIMARY_CERTIFIER_UNMOVED_LOCAL_TEST_INCONCLUSIVE_ON_K6 — one trust-scale action moves K6/dwell in 0/216 evals
tags: [coin, markov, critic-fidelity, action-ranking, one-step, powered, held-out, boundary, hierarchical-bootstrap]
---

# LOCAL_ACTION_RANKING_FIDELITY_V2 — powering the test reveals the probe cannot test what matters

V1 measured the local one-step ranking correctly but was underpowered. This V2 added every power upgrade the review
asked for — and the result is a clean, important **negative about the probe, not the critic**.

## Method (power upgrades over V1)
- **Held-out boundary panel** — 72 states from seeds ≥6200, **disjoint from critic-train (6000–6088) and old dev
  (6100–6148)** (asserted). Biased to **strict∈{3,4,5}** (mid-dwell, at the CENTER_TOL/SETTLE_VEL edge) so a single
  action *could* flip the certifier. (These boundary states are all `settling_dwell` by nature — strict 3/4/5 *is* the
  dwell phase; target_entry/braking live at strict 0–2.)
- **ε from history, not arbitrary** — the empirical accepted transactional actor-update action-drift: p50/p90/p99 of the
  accepted per-step p95 anchor drift + the trust cap. Measured: **ε = {0.0019, 0.002, 0.01}** (n=17 accepted steps; the
  accepted drifts cluster at ~0.002, cap 0.01) — the trust-region-relevant scale, ~20× smaller than V1's 0.04.
- **Reordered certificate** — K6 ≻ max_dwell ≻ **true full-containment exit (dtz≤CENTER_TOL, fixed from the old
  ENTRY_TOL bug)** ≻ speed ≻ dtz (dtz only the last tiebreaker). Sensitivity under pure K6 / pure max_dwell.
- Single candidate action at t=0 then frozen pi_0 (deepcopy-verified bit-for-bit incl. final sim state); state-wise;
  **≥3 seeds; hierarchical bootstrap over seeds AND states**; both Arm-A and Arm-B critics.

## The decisive measurement
**discrimination @ε=0.0019 (n = 216 state-evaluations = 72 states × 3 seeds):**
`any-change 0.944 | dwell-moves 0.000 | K6-moves 0.000`

A single accepted-drift-scale action change, followed by 29 steps of frozen pi_0, **never moves K6 or max_dwell** — not
on one of 216 boundary-state evaluations. The lexicographic certificate discriminates (94%) **only through the fine
speed/dtz/exit tiebreakers**, which do **not** change the terminal outcome. (`ρ_dwell` / `ρ_k6` are undefined for the same
reason.) The boundary panel + trust-scale ε did *not* achieve their goal of making the primary certifier move.

## Fine-tiebreaker correlations (hierarchical bootstrap over seeds × states)
| critic | ε=0.0019 | ε=0.002 | ε=0.01 |
|---|---|---|---|
| A | −0.263 CI[−0.43, −0.17] | −0.294 CI[−0.43, −0.13] | −0.313 CI[−0.40, −0.19] |
| B | −0.322 CI[−0.42, 0.02] | −0.317 CI[−0.43, −0.05] | −0.300 CI[−0.37, −0.10] |

top-1 beats-baseline 0.255, +ΔQ sign-agreement 0.391 (both below chance). So on the fine tiebreakers the critic is
**mildly negatively** correlated. **But those tiebreakers are terminal-irrelevant** (they never change K6/dwell), so a
K6-faithful critic has no obligation to rank them — a slight negative there is not a K6-fidelity defect.

## Verdict — `PRIMARY_CERTIFIER_UNMOVED_LOCAL_TEST_INCONCLUSIVE_ON_K6`
The powered, held-out, boundary, correct-ε, single-action test establishes a robust fact: **a single trust-scale action
change is physically inconsequential for the primary certifier (K6/dwell) — 0/216.** Therefore this class of probe
**cannot answer** whether the critic ranks K6-relevant actions faithfully. The near-zero-to-mildly-negative ρ is measured
entirely on terminal-irrelevant fine tiebreakers, where correct indifference (ρ≈0) is expected of a K6-optimal critic.

**This blocks the earlier plan branch.** Neither "critic faithful" nor "critic unfaithful" on K6 is established — so the
decision-tree jump to `PAIRED_LOCAL_ACTION_RANKING_CRITIC_V1` is **not yet justified**: we have not shown the critic
mis-ranks K6-relevant actions; we have shown one nudge doesn't move K6. Building a ranking critic on terminal-irrelevant
tiebreakers would optimize the wrong thing.

## What this means / next lever (needs your go)
To test the critic's ranking on **what matters (K6/dwell)**, the candidate actions must actually produce divergent
K6/dwell outcomes. At the single-action, trust-scale limit they provably don't. The minimal honest options:
1. **Bounded multi-step candidate window** — apply the candidate for a small K>1 steps (still short, still local-ish) so
   the certifier can diverge, and re-run this exact state-wise / ε / hierarchical machinery. This *is* a different test
   than "single local action" (the review earlier flagged the every-step confound) — but the 0/216 result now *justifies*
   the minimal step up as the only way to get K6 to respond. Report it as a distinct probe, not conflated with V2.
2. **Action-pair selection for divergent K6** — construct candidate pairs pre-screened (by short rollout) to yield
   different K6 outcomes, then ask whether ΔQ ranks them; this directly tests K6-ranking without a policy over many steps.
Only after a probe where K6 actually moves can fidelity be concluded or a paired-ranking critic be justified. The trust
region stays throughout (still the only thing that caught the earlier critic-exploit).

## Files
- lib: `coin_action_perturbation.py` (+`eps_from_drifts`, `hierarchical_bootstrap_ci`); `coin_markov_ablation_train.py`
  (`one_step_candidate_outcome` now reports true full-containment exit; `train_arm(accepted_sink=…)`);
  `coin_late_start.py` (`HandoffRecord` +strict/dtz/speed for boundary selection).
- entry: `experiments/…/rl_entry/coin_local_ranking_fidelity.py`; results `…/local_ranking_fidelity_v2.{json,svg,png}`.
- tests: `test_coin_action_perturbation.py` — 12 (geometry, deepcopy-fidelity, hierarchical bootstrap, ε-selection,
  HandoffRecord fields, held-out-panel disjointness). 18 pass total, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; critics Arm A+B 4000 updates, seeds {0,1,2},
single-thread. Panel 72 held-out `settling_dwell` strict{3:28,4:23,5:21}, seeds ≥6200. ε empirical-accepted-drift. No new
campaign, no reward/task change, no CORE.YAML items.
