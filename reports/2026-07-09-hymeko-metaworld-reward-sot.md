# HyMeKo as the MetaWorld reward source of truth — declare + reconstruct

**Date:** 2026-07-09 · Aiko · branch `hymeko-neuro-migration`
**Status:** done. HyMeKo now **declares** the MetaWorld dense reward as a `.hymeko` `Σ weight·term` spec and
reconstructs it at **R² 0.85–1.00**. No training; read-only scripted rollouts. This closes the SoT gap flagged in
`2026-07-08-cip-status.md` (§reward) and makes the reward ablatable (the coin Stage-A intervention, transferred).

---

## Why this matters (the selling point)

Before this, HyMeKo was the source of truth for the task's *meaning* (the monitor) and the causal model's
*representation* (the verified `.hymeko` DAG) — but for MetaWorld the **reward** was MetaWorld's own dense reward,
outside HyMeKo. Now the MetaWorld reward is **declared in HyMeKo** (`data/robotics/metaworld_reward.hymeko`) as a
weighted bundle of named terms — the same `Σ weight·term` grammar the coin reward uses — so:

1. **HyMeKo owns the reward's declared form** (source of truth for structure + weights), not just physics.
2. Because it is `Σ weight·term`, the reward is **directly ablatable** — drop / downweight a term and recompute,
   exactly the coin Stage-A intervention, now on a reward HyMeKo *reverse-engineered* rather than authored.

## Changed / new files

| File | Change |
| --- | --- |
| `data/robotics/metaworld_reward.hymeko` | **new** — self-contained HyMeKo reward spec (terms `mw_in_place`, `mw_grasp`, `mw_near`, `mw_dist` + weights); parses via the existing `read_reward_terms` |
| `hymeko_rl/eval/cip/metaworld_reward.py` | **new** — record components, `hymeko_reward` (Σ weight·term), `fit_reward_weights` (deterministic lstsq), `ablate_reward` (drop/scale), `evaluate_reward_fidelity`, `run_reward_fidelity_sweep` |
| `hymeko_rl/eval/cip/__init__.py` | exports |
| `hymeko_rl/tests/test_metaworld_reward.py` | **new** — 5 tests (parse+map, weighted-sum, fit recovery, ablation, real-env fidelity) |

Reuses `read_reward_terms` (no re-parse). `CORE.YAML` / `pyproject.toml` untouched; no dependency added.
`meta_reward.hymeko` **not** modified (the new spec is self-contained).

## The `.hymeko` reward declaration

```
metaworld_reward: reward
{
    @inplace: reward.terms.mw_in_place {}     // MetaWorld in_place_reward (progress toward target)
    @grasp:   reward.terms.mw_grasp {}        //           grasp_reward   (grasp/hold shaping)
    @near:    reward.terms.mw_near {}          //           near_object    (proximity)
    @dist:    reward.terms.mw_dist {}          //          -obj_to_target  (dense negative distance)
    @mw_reward: reward.reward_spec { (+ inplace 8.0, + grasp 1.2, + near 1.0, + dist 10.0); }
}
```
Each term kind maps to a recorded MetaWorld info component (`_TERM_TO_COMPONENT`); the reward is
`Σ weight·component` per step. The bundle weights are **seeds** — the fit reports the least-squares values per task.

## Fidelity (Σ weight·term vs MetaWorld's `unscaled_reward`; 8 eps/task, `reports/figures/2026_07_09_18_04_cip_metaworld_reward/`)

| Task | R² (declared seed) | **R² (fitted)** | fitted weights (in_place, grasp, near, dist) |
|---|---|---|---|
| push | 0.766 | **0.971** | 6.02, 2.13, 2.59, 3.90 |
| pick-place | 0.738 | **0.851** | 6.62, 1.66, 0.18, 4.57 |
| door-open | 0.977 | **0.993** | 8.33, 0.09, 1.70, 0.00 |
| button-press | −225 | **0.957** | −0.39, 1.88, 0.00, −0.56 |
| reach | 0.133 | **1.000** | 10.0, 0.0, 0.0, 0.0 |

**HyMeKo's `Σ weight·term` reconstructs MetaWorld's dense reward at R² 0.85–1.00 (fitted) across all five task
families.** Notes:
- **reach is R²=1.0** — its reward is *purely* `in_place` (weight 10, all else 0); a single HyMeKo term is the
  whole reward.
- **door-open** already fits the seed weights well (declared R²=0.98).
- **button-press** exposes the seed weights as wrong (declared R²=−225) but the fit recovers R²=0.957 — the fit is
  what makes the declaration faithful per task.
- The declared **structure** (4 terms) is shared; the **weights** are fit (reverse-engineered) per task.

## Ablation-ready (the payoff)

The reward is now `Σ weight·term`, so `ablate_reward(components, terms, weights, drop=[...], scale={...})`
recomputes it with a term removed / downweighted — the coin Stage-A method, offline, on MetaWorld. Tested
(`test_ablate_drops_and_scales_terms`). The full CIP ablation (does dropping `mw_grasp` collapse a
`grasp → total_reward` edge on a grasp task like pick-place?) is the **next step**.

## Tests + static

**5** `test_metaworld_reward.py` tests pass (parse+map, weighted-sum, fit-recovers-known-weights, ablation
drop/scale, real-env fidelity R²>0.7). Full CIP suite green. ruff / radon (no block ≥ C) / mypy `--strict` clean on
the new module. The real-env test skips if `metaworld` absent.

## Honest scope

- **Faithful decomposition, not bit-exact.** R² < 1 for most tasks (0.85 pick-place) — MetaWorld computes the
  reward inside the env with an un-exposed reach-tolerance nonlinearity; HyMeKo declares its dominant `Σ weight·term`
  structure. (For the **coin** scenario the reward *is* the `.hymeko` spec, so there it is exact.)
- **Weights are fit per task.** The `.hymeko` seed weights are a starting point; the deterministic lstsq refit gives
  the per-task values (reported). HyMeKo owns the structure; the fit determines the weights.
- **MetaWorld still computes the reward at run time.** HyMeKo declares + reconstructs + ablates it; it does not (and
  need not) replace MetaWorld's internal computation.
- Single noise level / 8 eps — a fidelity estimate, not a multi-seed claim.

## What remains

1. **Run the CIP reward ablation on MetaWorld** — drop `mw_grasp` / `mw_near` and re-fit the CIP DAG, testing
   whether the reward's causal parent (proximity/grasp) collapses, à la coin Stage-A.
2. **Per-task declared specs** — write the fitted weights back into per-task `.hymeko` files (or one parameterized
   spec) so the declaration is task-faithful out of the box.
3. **Certify** — run the fitted HyMeKo reward through `reward_oracle` where a task delivery notion exists.

## Constraints honored

No training · read-only scripted rollouts · `meta_reward.hymeko` / FANUC v2 / coin-collab v2b / `CORE.YAML`
untouched · `pyproject.toml` not edited · no existing report/artifact overwritten.
