---
title: Pick-place missing assimilation — exact gaps between verified findings and the executable path
date: 2026-07-16
scope: what strong pick-place components exist only in reports/isolated experiments and are NOT in the current default code path
status: gap list (drives the phase-7 integration)
---

# Missing assimilation — exact files & symbols

Ordered by impact. "Executable gap" = a verified result that the canonical loader / GUI / evaluator cannot currently reach.

## G1 — the canonical loader cannot load a TD3+BC (off-policy) policy  [EXECUTABLE GAP]
- **Verified component:** TD3+BC actor = `hymeko_rl/train/ddpg.py::DeterministicActor` (`μ = action_scale·tanh(head(backbone))`), built by `build_offpolicy` (`ddpg.py:133`). Checkpoints exist: `experiments/2026_07_13_02_55_fanuc_pick_td3bc_hsikan/policies/*.pt`, `experiments/2026_07_06_18_0{1,6}_*`.
- **Gap:** `hymeko_rl/experiments/gripper_pick_bc.py::load_pick_policy` (`:95-126`) handles only `recurrent_clone`, `residual_clone`, raw FF `ActorCritic`. A `DeterministicActor` state_dict has keys `backbone.*/head.*` → falls through to `load_pick_actor` (`:126`) → `ActorCritic.load_state_dict` mismatch → `PickPolicyIncompatible`. Confirmed by the routing map. **No silent fallback** (good), but the learned-RL frontier is unreachable through the canonical path.
- **Also:** `hymeko_rl/gui/pick_place_scene.py::list_checkpoints` (`:129-156`) never enumerates a TD3+BC checkpoint; an unlabeled `.pt` in `experiments/pick_place_gui/` is mislabeled `recurrent` and then fails.
- **Fix:** add a `td3bc`/`deterministic` kind to `load_pick_policy` (build via `build_offpolicy`, `action_scale = max|action_high|`, wrap in `greedy_action_fn`), fail-loud on shape mismatch; extend `list_checkpoints` to detect it.

## G2 — the evaluator has no far-spawn split and reports one overloaded `success`  [EXECUTABLE GAP]
- **Verified finding:** F-PP-013 (advisory) — `placed_stable` carries a ~0.458 idle floor (42% spawn-at-target); the skill-isolating read is `placed_stable ∧ ever-grasped` on the far-spawn subset.
- **Gap:** `hymeko_rl/eval/evaluate.py::LiftPlaceMetric` (`:289-309`) counts only `info["reached"]` on the **full** distribution; `eval_success` (`gripper_pick_bc.py:132`) returns `(lift, place)` only. There is **no** far-spawn split and **no** grasp∧place metric in shared code — it lives only in scratchpad probes. The whole discrepancy comes from this gap.
- **Fix:** a canonical evaluator that, per policy/seed, returns `{reached_full, reached_far, grasped_far, placed_real_far (grasp∧place), safety, ep_time}` on a fixed far/near split.

## G3 — the "structured residual holds 0.875" (residual PPO) checkpoint is not a saved canonical artifact
- **Verified component:** residual PPO (mode-gated place+release, δ0.05, zero-anchored) = `hymeko_rl/experiments/pick_place_residual_rl.py`; result `experiments/residual_rl/residual_hsikan_d0.05_settle_place-release.json` [0.875,0.875,0.875].
- **Gap:** that run reports JSON but (per the inventory) did not persist a `residual_clone` `.pt` under `experiments/pick_place_gui/` that the GUI loads; `save_residual_policy` (`pick_place_residual_rl.py:49`) exists but was not invoked for this cell. So the strongest learned-RL-*holds* policy is reproducible-but-not-loadable.
- **Fix:** re-emit the residual_clone checkpoint via `--save-policy` (no retrain — the residual is zero-anchored = base; a fresh zero-residual reproduces it), OR register the base itself as the residual-hold artifact.

## G4 — no canonical pick-place artifact registry / benchmark manifest
- **Gap:** there is a `reports/framework/canonical_components.json` + `canonical_findings.json`, and `reports/canonical_integration/fanuc-pick-place/canonical_manifest.json` (older), but **no single manifest** listing every deployable pick-place artifact (scripted expert, BC, DAgger base, residual PPO, TD3+BC, SAC negative control) with metadata {method, arch, action-abstraction, obs-schema, env version, source experiment, source commit, seed, selection rule, fallback}.
- **Fix:** `reports/canonical_integration/pick_place/canonical_manifest.json` + a `hymeko_rl` registry the canonical evaluator/GUI read.

## G5 — no single canonical entry point; the current experiment stack is fragmented
- **Gap:** pick-place is spread over `pick_place_dagger.py`, `pick_place_hybrid_dagger.py`, `pick_place_residual_rl.py`, `pick_place_td3bc.py`, `pick_place_recurrent_clone.py`, `pick_place_rl_sidebyside.py`, `pick_place_stable_campaign.py` — each with its own eval. There is a `canonical_coin_toss.py` pattern but **no `canonical_pick_place.py`**.
- **Fix:** one `hymeko_rl/experiments/canonical_pick_place.py` that imports the registry (G4) + the canonical evaluator (G2) + the extended loader (G1), evaluates every artifact under one protocol. Compose existing modules; do not fork a new stack.

## G6 — regression guard against silently reverting to an older abstraction
- **Gap:** `test_residual_abstraction.py` guards the residual invariant, but nothing guards that (a) the canonical loader still loads a TD3+BC actor, (b) the evaluator still reports the far/near split, (c) the GUI selector lists all artifact kinds.
- **Fix:** extend the test suite (phase 10) so a future edit that drops TD3+BC loading or the far-spawn split fails CI.

## Non-gaps (verified already-assimilated)
- `SACConfig.stable()` (F-PP-017, this session) — the SAC correction is single-source.
- No scripted-expert fallback anywhere (routing-map verified) — so no fallback-leakage guard needed, but the *manifest* must record `fallback: none` per artifact.
