# MetaWorld reward-ablation Stage A — `mw_grasp` on pick-place

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** done. The coin Stage-A intervention transferred to MetaWorld/pick-place: ablate a declared HyMeKo reward
term, recompute offline on fixed rollouts, re-fit CIP/LiNGAM-SH. **No training.** **Verdict: NOT SUPPORTED** at the
reward-computation level (with a positive control validating the pipeline).

---

## Summary

Because the MetaWorld reward is now a HyMeKo `Σ weight·term` (`data/robotics/metaworld_reward.hymeko`), ablating a
term is a deterministic offline reweighting — no env re-stepping, no training. On **pick-place** (grasp is
in-frame), dropping `mw_grasp` and re-fitting the CIP reward mechanism **does not** collapse the
`grasp_fraction → total_reward` loading, because `mw_grasp` is a **minor** reward term (fitted ≈ 1.0 vs the dominant
`mw_in_place` ≈ 8.5). Ablating it moves the reward only ~8–13 %. A positive control — ablating the dominant
`mw_in_place` — moves the reward ~100 %, confirming the pipeline detects a real term when there is one. So the
grasp reward-farming hypothesis is **rejected at the reward-computation level**; the reward is progress/in_place-
dominated (consistent with every prior finding: pick-place reward tracks proximity/progress, not grasp).

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/cip/reward_ablation_metaworld.py` | **new** — `ablate_reward_spec`, `AblatedRewardSpec`, `run_reward_ablation_stage_a` (offline recompute + CIP re-fit + cross-view) |
| `hymeko_rl/eval/cip/__init__.py` | exports |
| `hymeko_rl/tests/test_reward_ablation_metaworld.py` | **new** — 6 tests (spec A–G + real-env run) |

Reuses `read_reward_terms`, `hymeko_reward`/`ablate_reward`, `reward_mechanism_proposal`,
`fit_loadings_least_squares`, `cross_view_verify`. `CORE.YAML` / `pyproject.toml` / `metaworld_reward.hymeko`
untouched; no dependency added.

## Task / rollout source · reward SoT · ablated terms

- **Task:** MetaWorld `pick-place-v3` (grasp fires, so `mw_grasp`↔`grasp_fraction` is in-frame; coffee-push never
  grasps, so it was **not** used for this ablation).
- **Rollouts:** cached scripted `SawyerPickPlaceV3Policy` + per-episode action noise, N=60, offline recompute.
- **Reward SoT:** `data/robotics/metaworld_reward.hymeko` (terms `mw_in_place, mw_grasp, mw_near, mw_dist`).
- **Ablated:** `drop=["mw_grasp"]` (its weight → 0; the file is not mutated).

## Original vs ablated (representative run `…19_35_cip_reward_ablation_stageA/`)

| Quantity | Original | `mw_grasp` off |
|---|---|---|
| Reward fidelity R² (Σ weight·term vs env reward) | **0.915** | — |
| Fitted weights | `in_place 8.54, grasp 1.05, near −0.33, dist 4.86` | grasp → 0 |
| Reward change ‖Δreward‖/‖reward‖ | — | **0.129** (≈13 %) |
| **Positive control** — dominant `mw_in_place` off | — | **1.005** (≈100 %) |
| `grasp_fraction → total_reward` loading | −38.5 | −49.5 (does **not** collapse; noisy) |
| Reward reconstruction R² (tail → reward) | 0.998 | 0.998 |
| Reward↔monitor disagreement | 0.067 | 0.073 |
| Cross-view verify (original / ablated / re-parented) | ✅ | ✅ / ✅ |

## Decision rule

Grasp reward-farming is SUPPORTED (reward-computation level) iff the grasp loading collapses **and** the reward
moves materially **and** cross-view passes. Here:
- **Loading collapse: NO** — the loading is a noisy collinearity artifact (it swings −38 to −73 across runs, sign
  and magnitude unstable), and it does not sharply decrease under ablation. It is *not* the robust signal.
- **Reward change: small** — ablating `mw_grasp` moves the reward only ~8–13 % (vs ~100 % for `mw_in_place`).
- **Cross-view: yes** — original, ablated, and re-parented mechanism graphs all cross-view-verify.
- **Re-parenting: yes** — the ablated-spec mechanism drops `grasp_fraction` from the tail and re-parents onto
  `{progress_score, near_fraction, obj_to_target_delta}`.

**⇒ NOT SUPPORTED.** `mw_grasp` is a minor reward term on pick-place; ablating it barely perturbs a reward
dominated by `mw_in_place`. This is a genuine negative result for the grasp-farming hypothesis, and it is
consistent with the multi-seed / generic-sweep findings that pick-place's reward tracks proximity/progress.

## Does the intervention support the hypothesis? — No (pipeline validated)

The **intervention infrastructure works** end-to-end: spec-level ablation (deterministic, non-mutating), offline
reward recomputation (fidelity R²=0.92), CIP/LiNGAM-SH re-fit, and cross-view verification all pass. The
**positive control** (ablating the dominant `mw_in_place`) produces a ~100 % reward change, proving the pipeline
detects a term that *is* a driver. The specific `mw_grasp`-farming hypothesis is simply not true on pick-place.

## Honest caveats

- **MetaWorld env randomisation is seed-uncontrolled** → run-to-run variation (grasp fitted weight 0.5–1.05,
  reward change 8–13 %, loading −38…−73). The *conclusion* (grasp minor, no collapse, NOT SUPPORTED) is stable;
  a multi-seed pass would tighten the numbers.
- **The loading value is scale/collinearity-noisy** (total_reward is a per-episode sum; `grasp_fraction` is
  collinear with `near`/`progress`). The **reward-change fraction** is the robust, interpretable signal.
- **Reward-computation level only** — the policy is fixed. No policy-learning claim.

## Tests

6 `test_reward_ablation_metaworld.py` tests (spec A–G + the real-env run). Full CIP suite green. ruff / radon (no
block ≥ C, after extracting `_verdict`) / mypy `--strict` clean. The real-env test skips if `metaworld` absent.

## What remains for Stage B

- **Retrain under the ablated reward** (`galambos_task`-style, but MetaWorld) — the *policy-learning* test the coin
  Stage-B describes: does a policy trained without the grasp term behave differently? Deliberately **not run**
  (training).
- **Multi-seed the ablation** — turn the point estimates into median/IQR (as for the coffee-push multi-seed).
- **Ablate a term that IS a driver where it matters** — e.g. `mw_in_place` on a task, or `mw_dist` — to show a
  *supported* collapse end-to-end (the positive control already shows the reward-change side of this).

## Constraints honored

No training · no production ablation · FANUC v2 / coin-collab v2b / `CORE.YAML` / `metaworld_reward.hymeko` /
`pyproject.toml` untouched · no existing report/artifact overwritten · no causal-truth claim beyond
reward-computation-level support.
