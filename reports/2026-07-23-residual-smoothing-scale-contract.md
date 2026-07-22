---
title: RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS — scale-correct residual smoothing
date: 2026-07-23
slug: residual-smoothing-scale-contract
task: coin_v3 delivery — phase-gated residual critic (§1-§3 smoothing scale contract)
verdict: RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS
---

# Scale-correct residual target-policy smoothing

**Created-at:** 2026-07-23 21:05 CEST
**Context.** The prior critic was trained with target smoothing disabled (zero noise) → `TARGET_SMOOTHING_CONTRACT_
MISMATCH`. The naïve fix would have re-enabled the **absolute** TD3 defaults (std=0.2, clip=0.5), which are meant for
a ~[−1,1] action space and are wrong in the residual's ±0.25 units. This stage freezes a **scale-relative** contract.

## §1 scale-correct contract — `RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_V1` (SHA `d558443d`)

```
residual_bound       = 0.25
smoothing_std_ratio  = 0.20  ⇒  smoothing_std  = 0.20 × 0.25 = 0.05
smoothing_clip_ratio = 0.50  ⇒  smoothing_clip = 0.50 × 0.25 = 0.125

eps            = clamp(N(0, 0.05), -0.125, +0.125)
target_residual= clamp(target_residual_actor(o_next) + eps, -0.25, +0.25)
target_action  = clamp(pi_0(o_next) + gate_next · target_residual, -4, +4)
```

`pi_0` never receives smoothing noise. The library defaults (`bounded_smoothed_residual` / `residual_target_action`)
are updated to `SMOOTHING_STD=0.05`, `SMOOTHING_CLIP=0.125`.

## §2 distribution audit (analytic + Monte Carlo, 4e5 samples) at zero target residual

| regime | std | clip | per-dim bound-hit (MC / analytic) | any-of-4 bound-hit | residual-norm mean / max |
|--------|-----|------|-----------------------------------|--------------------|--------------------------|
| A disabled | 0.0 | 0.0 | 0.000 / 0.000 | 0.000 | 0.000 / 0.000 |
| B unscaled (abs) | 0.2 | 0.5 | **0.211 / 0.2113** | **0.613** | 0.316 / 0.500 |
| **C scale-correct** | **0.05** | **0.125** | **0.000 / 0.000** | **0.000** | 0.093 / 0.246 |

- **B saturates a residual boundary on 61% of targets** — the smoothing scale is comparable to the whole residual
  range; its norm piles toward the bound (max 0.5). MC matches the analytic Gaussian tail (0.211 vs 0.2113).
- **C never saturates the residual bound** (clip 0.125 < bound 0.25 ⇒ the final clamp never engages); norm stays well
  inside (peak ~0.09, max 0.246). This is the scientific regime.

Figure `reports/figures/coin_smoothing_scale_contract.png`. **`RESIDUAL_TARGET_SMOOTHING_SCALE_CONTRACT_PASS`.**

## §3 batch-independence + reproducibility (tests)

`test_coin_residual_replay.py` (13/13): noise shape derives from the target-residual tensor (batch **1/7/256/512**),
no hardcoded batch dim; `generator`-seeded noise is reproducible (same seed → identical, different seed → different);
explicit zero-noise diagnostic mode; active smoothing is the default; final residual ∈ [−0.25, 0.25]; composite ∈
[−4, 4]; `pi_0` unsmoothed; `SMOOTHING_STD==0.05`, `SMOOTHING_CLIP==0.125`; contract SHA deterministic.

## Regression — unchanged

Update-0: HL 3/9, VAL 2/30, grasp 9/9, delivered {1011,1447,1568}, composite−base maxdiff 0.0, π₀ hash unchanged.
All tests green: **57/57**. Ruff crit clean.

## Files touched

- `hymeko_rl/coin_delivery/coin_residual_replay.py` (+contract, scale-correct defaults, `generator`).
- `hymeko_rl/tests/test_coin_residual_replay.py` (+scale/reproducibility tests; 13/13).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_smoothing_scale_audit.py` + `plot_smoothing_scale.py`
  + `smoothing_audit.json`.
- `reports/figures/coin_smoothing_scale_contract.png`.

**CORE.YAML:** none. Frozen π₀ (`1902454c`) / reward / γ / bundle / obs / gate thresholds / residual range / action
bounds unchanged. No critic retrain yet, no actor update, sealed final panel unopened, SAC quarantined. Mac; kato14
clean. §13 commit 1.

## Next (still no actor update, sealed panel closed)

§4 freeze the corrected critic configuration (arch/encoder/optimizer/LR/batch/schedule/grad-clip/replay/this
smoothing/candidate distribution/all three panel manifests/metric-suite) with SHAs → §5 retrain the critic with
scale-correct smoothing, checkpoints {0,1k,3k,6k,10k,20k,40k} → §6-§10 development authorization with **Q1/Q2/min-Q
separate**, **centered** ΔQ vs empirical ΔG, margin-aware metrics, and the empirical **+gradQ1 vs −gradQ1** test →
`PHASE_GATED_RESIDUAL_CRITIC_DEVELOPMENT_PASS`/`_FAILURE`. Only a double development pass opens the sealed final panel.
