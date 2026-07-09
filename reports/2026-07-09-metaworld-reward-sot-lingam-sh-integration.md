# HyMeKo reward SoT → LiNGAM-SH mechanism integration

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** done. The HyMeKo-declared MetaWorld reward drives a LiNGAM-SH mechanism proposal, scored against the
coffee-push multi-seed evidence and cross-view-verified. No training. **No causal-truth claim from grouping.**

---

## Summary

Wired `data/robotics/metaworld_reward.hymeko` (the reward source of truth) into the LiNGAM-SH pipeline: its declared
terms become a `{reward terms} → {total_reward}` mechanism proposal, which is factorized against the observed
pairwise `B` (coffee-push multi-seed) alongside three baselines and cross-view-verified. **The HyMeKo reward
mechanism explains the evidence better than the common-child grouping at equal parameter count** — because the
declaration correctly identifies `progress_score` (in_place) as a reward driver, where common-child (reading B's
pairwise parents) mis-groups `action_noise`.

## Changed / new files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/cip/metaworld_reward.py` | + `_TERM_TO_CIP_VARIABLE`, `hymeko_reward_terms`, `reward_mechanism_proposal` (SoT → mechanism adapter) |
| `hymeko_rl/eval/cip/reward_mechanism_integration.py` | **new** — `compare_reward_mechanisms` (none / raw-pairwise / common-child / hymeko-reward) + cross-view |
| `hymeko_rl/eval/cip/__init__.py` | exports |
| `hymeko_rl/tests/test_reward_mechanism_integration.py` | **new** — 6 tests |

Reuses `read_reward_terms`, `propose_reward_terms`, `fit_sigma_least_squares`, `propose_common_child`,
`proposals_to_causal_hypergraph`, `cross_view_verify` — no re-implementation. `CORE.YAML` / `pyproject.toml` /
`meta_reward.hymeko` untouched; no dependency added.

## Reward SoT path

`data/robotics/metaworld_reward.hymeko` → `read_reward_terms` → term kinds + weights → `_TERM_TO_CIP_VARIABLE`
mapping → filter to the frame's variables → `propose_reward_terms(...)` → `MechanismProposal` →
`proposals_to_causal_hypergraph` → star-expand → `.hymeko` → engine cross-view.

## Extracted reward terms

`read_reward_terms('data/robotics/metaworld_reward.hymeko')` →
`mw_in_place (8.0), mw_grasp (1.2), mw_near (1.0), mw_dist (10.0)`.
Mapping to CIP frame variables: `mw_near→near_fraction`, `mw_grasp→grasp_fraction`, `mw_in_place→progress_score`,
`mw_dist→obj_to_target_delta`.

## Proposed reward mechanism

Filtered to the coffee-push multi-seed frame (`near_fraction, progress_score, total_reward, action_noise`), only
`mw_near`/`mw_in_place` map to present variables, so:

> **`{progress_score, near_fraction} → {total_reward}`** (source `reward_terms`, from the HyMeKo declaration).

`mw_grasp`/`mw_dist` are dropped here — coffee-push's *monitor* frame has no grasp / distance-delta variable
(coffee-push doesn't grasp).

## Factorization / scoring vs baselines (evidence: coffee-push multi-seed B, stable edges)

`B` from the multi-seed aggregate (`near→total_reward +0.973`, `near→progress +0.870`, `noise→total_reward −0.112`,
`near→noise +0.081`); least-squares Σ per candidate:

| Candidate | mechanism | explained energy | params | fro_error |
|---|---|---|---|---|
| none | — | 0.000 | 0 | 1.312 |
| **hymeko_reward** | `{progress, near} → total_reward` | **+0.275** | **1** | 1.117 |
| common_child | `{near, action_noise} → total_reward` | +0.215 | 1 | 1.162 |
| raw_pairwise | 4 degenerate edges | +1.000 | 4 | 0.000 |

**Key result — HyMeKo reward mechanism vs common-child (both 1 parameter):** the HyMeKo mechanism explains **0.275
vs 0.215** — a better single-parameter summary, because it groups the *reward-declared* drivers (`progress_score`,
`near_fraction`) rather than B's noisiest pairwise parents (`action_noise`). Both beat **none** (0.0). **raw_pairwise**
reconstructs exactly (1.0) but with 4 parameters (no grouping) — the un-grouped view.

**Honest limitation:** 0.275 is modest. A single-strength mechanism assigns *one* weight to all tail→head entries,
but the reward's dependence is asymmetric (`near→reward ≈0.97` ≫ `progress→reward` weak in the collinearity-collapsed
B). Capturing per-term reward weights needs **per-tail weights in `A_in`** (not the 0/1 indicator) — a Step-4
refinement. The declared multi-input dependence is nonetheless real (reward fidelity below), which is exactly the
LiNGAM-SH thesis: pairwise `B` under-represents multi-input mechanisms via mediation.

## Cross-view verification

**Passes.** `{progress_score, near_fraction} → {total_reward}` → `CausalHypergraph` → `check_acyclicity` = **True**
→ star-expand → native `hymeko` engine → **`cross_view.agree = True`** (declared ≡ tensor, Blake3 hash).

## Oracle / certification status

The certification path is `evaluate_reward_fidelity` (`metaworld_reward.py`) — it verifies the HyMeKo terms
reconstruct MetaWorld's Python reward. For **coffee-push**: **R²_fitted = 0.743** (declared-seed 0.223,
rel-error 0.397). So the HyMeKo terms **do** correspond to MetaWorld's reward decomposition, but **not bit-exact**.
Exact missing mapping:
1. **2 of 4 declared terms have no coffee-push-monitor variable** — `mw_grasp` (coffee-push never grasps) and
   `mw_dist` (the monitor frame has `progress_score`, not a raw distance-delta); they enter the *generic* info-frame
   but not the monitor frame.
2. **Un-exposed reach-tolerance nonlinearity** — MetaWorld computes its reward with a `tolerance`-shaped reach term
   that is not in the exposed `info` components (the ~26% residual).
(For the **coin** scenario the reward *is* the `.hymeko` spec, so there certification is exact.)

## What remains before the reward ablation

1. **Per-tail weights** (Step 4) — let a mechanism carry per-term weights (`A_in` real-valued, not 0/1) so the
   reward mechanism captures the asymmetric near-vs-progress dependence and its explained energy rises.
2. **Run the ablation** — with the reward now a HyMeKo `Σ weight·term`, `ablate_reward(drop=['mw_grasp'])` recomputes
   it; re-fit the CIP DAG on a *grasp* task (pick-place, where `mw_grasp` is in-frame) and test whether the
   `grasp → total_reward` edge collapses — the coin Stage-A intervention on MetaWorld.
3. **Generic-frame integration** — run this comparison on the generic-sweep B (which *has* `grasp_fraction`,
   `obj_to_target_delta`), so all four reward terms map and the mechanism is the full `{near, grasp, progress,
   dist} → reward`.

## Constraints honored

No training · no causal-truth claim from grouping (`_disclaimer` in the output) · FANUC v2 / coin-collab v2b /
`CORE.YAML` / `meta_reward.hymeko` untouched · `pyproject.toml` not edited · no existing report/artifact overwritten.
