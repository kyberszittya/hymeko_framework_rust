# Generic MetaWorld CIP sweep (info-signals, 5 task families)

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** done. Generic task-agnostic CIP runner + a 5-task sweep, read-only scripted rollouts, **no training.**

---

## What this adds

The coffee-push real-env path hard-codes an obs→`CoffeePushMonitor` mapping (one task). This new runner
(`hymeko_rl/eval/cip/metaworld_generic_cip.py`) is **task-agnostic**: every MetaWorld V3 env emits the same `info`
dict (`success`, `near_object`, `grasp_success`, `in_place_reward`, `obj_to_target`, `unscaled_reward`), so those
signals are a ready-made CIP variable set for **all 48 MT50 tasks**. Roll a scripted policy + per-episode action
noise → `RolloutFrame` over the info signals → DirectLiNGAM → declare + cross-view-verify the DAG. One representative
task per family was swept.

**Continuous variables:** `action_noise` (observed exogenous input), `near_fraction`, `grasp_fraction`,
`obj_to_target_delta`, `total_reward` (the sink under test); `reward_progress_disagreement` (reward↔`in_place`) is a
prioritizer signal. **Categorical:** `task`, `mw_success`.

## Changed files

| File | Change |
| --- | --- |
| `hymeko_rl/eval/cip/metaworld_generic_cip.py` | **new** — `run_generic_cip`, `run_generic_sweep`, `GENERIC_TASKS` |
| `hymeko_rl/eval/cip/__init__.py` | exports |
| `hymeko_rl/tests/test_metaworld_generic_cip.py` | **new** — 4 tests (pure + tiny real-env) |

Reuses the coffee-push `_fit_declare_render` (DAG + `.hymeko` + cross-view) and `RolloutFrame` / `CausalDiagnosis`
unchanged. `CORE.YAML` / `pyproject.toml` untouched; no dependency added.

## Sweep results (N=80/task, seed 0, `reports/figures/2026_07_09_17_35_cip_metaworld_generic/`)

**Every discovered DAG cross-view-verified (`cross_view_all_pass = True`, 5/5).**

| Task (family) | success | reward↔task disagree | `total_reward` strongest parent | note |
|---|---|---|---|---|
| **push** (push) | 0.61 | 0.205 | `near_fraction → total_reward` **+0.78** | proximity-shaped |
| **pick-place** (grasp) | 0.89 | 0.244 | `near_fraction → total_reward` **+0.75** | proximity-shaped *even though it grasps* |
| **door-open** (articulated) | 0.94 | **0.004** | (none — reward well-aligned) | `obj_to_target_delta` constant → dropped |
| **button-press** (contact) | 1.00 | **1.065** | (none — reward disconnected) | `near`/`grasp` constant → dropped |
| **reach** (approach) | 1.00 | 0.435 | follows `obj_to_target_delta` | no contact terms |

### Findings (measured, PROPOSED)

- **The coffee-push proximity-reward signature replicates.** In both **push** and **pick-place**, the strongest
  parent of `total_reward` is `near_fraction` (~0.75–0.78) — the dense reward is proximity-shaped. Notably
  **pick-place grasps** (grasp_fraction varies 0–0.8) yet the reward still tracks *proximity*, not grasp — a
  concrete cross-task result the single coffee-push run could not show (coffee-push never grasps).
- **CIP separates aligned from misaligned rewards across tasks.** Reward↔task disagreement ranges from **0.004
  (door-open — reward tracks the task cleanly)** to **1.065 (button-press — reward strongly rank-disagrees with
  `in_place` progress)**. This is the diagnostic doing its job across families, not just on coin/coffee-push.
- **Per-task variable structure is honest.** Info signals that don't vary for a task are dropped as constant
  (door-open drops `obj_to_target_delta`; button-press / reach drop `near`/`grasp`) — so each task's DAG is over the
  signals that actually move for it.

## Honest caveats

- **Single run = point estimate.** MetaWorld's env randomisation is seed-uncontrolled (established for coffee-push),
  so per-task orders/edges shift run-to-run; a **multi-seed** pass is required before any ranking claim. Only the
  strongest, high-|w| edges (e.g. `near→reward` ~0.75) should be read here.
- **DirectLiNGAM direction instability.** `action_noise` is exogenous by construction, but LiNGAM sometimes places
  it most-endogenous (e.g. pick-place's `total_reward →−0.89 action_noise` edge is a direction artifact). The
  *coupling* is real; the *direction* on the noise edge is not trusted.
- **Task-truth is MetaWorld-native here, not a HyMeKo monitor.** The generic runner uses MetaWorld's `info` signals
  (the trade for covering all 48 tasks); the HyMeKo-monitor path exists only for coffee-push / dial-turn. The
  discovered DAG is still declared in `.hymeko` and engine-verified — HyMeKo remains the source of truth for the
  causal model's *representation*, not (here) the task variables.

## Tests + static

- **4** `test_metaworld_generic_cip.py` tests (reward-parent filter, disagreement bounds, task registry, tiny
  real-env run + cross-view). Full CIP/causal suite green (regression check: 72 passed on the touched subset). ruff
  / radon (no block ≥ C) / mypy `--strict` clean on the new module. The real-env test skips if `metaworld` absent.

## What remains

1. **Multi-seed the sweep** (reuse `_aggregate_batches`) — turn the per-task point estimates into stable
   present/median/IQR verdicts, as done for coffee-push.
2. **Per-task LiNGAM-SH** — propose + factorize a `{near, grasp, progress} → total_reward` mechanism per task
   (steps 2/3B) and compare explained-energy across families.
3. **HyMeKo-declared MetaWorld reward** — the SoT gap: declaring the reward in `.hymeko` unlocks the real-env reward
   ablation (transfer the coin Stage-A method) and puts the reward under HyMeKo for MetaWorld too.

## Constraints honored

No training · read-only scripted rollouts · FANUC v2 / coin-collab v2b / `CORE.YAML` untouched · `pyproject.toml`
not edited · no existing report/artifact overwritten.
