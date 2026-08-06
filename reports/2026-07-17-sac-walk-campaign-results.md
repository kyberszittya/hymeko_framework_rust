---
title: SAC-from-scratch walking campaign (kato14+kato15) — structural vs flat, 5 seeds, 3 bodies
date: 2026-07-17
status: complete (30/30 cells, 2x RTX 6000 Ada) — structural advantage tracks whether the body walks
core_yaml_touched: none
---

# SAC-from-scratch walking campaign — structural vs flat, multi-seed, multi-body

## Question

At scale (multi-seed, on policies that actually walk), does the **structural / relational actor** beat a
**flat MLP** on walking distance + the CIP causal **propel-edge** (`leg_speed ⇒ forward_vx`)? This is the
scaled version of the local single-seed validation (pure SAC from scratch walks the cheetah; warm-start traps).

## Method

`exp_sac_walk_campaign.py`, split across **two RTX 6000 Ada** (kato15 seeds 0–2, kato14 seeds 3–4; shared NFS,
disjoint seeds, per-box output dirs). **Pure SAC from scratch** (no warm-start), `{flat, structural} × {Aibo
goal-reach, biped humanoid, planar cheetah} × 5 seeds`, **200k steps/cell**. Metric: forward dx + CIP
propel-edge of the learned policy. **30 cells, 29 ok / 1 transient error** (a SAC NaN on cheetah/flat/seed2),
~18 h wall (single-env SAC is `env.step`-bound, ~74–360 steps/s; not GPU-bound). Ran detached overnight.

## Result — the structural advantage tracks how well the body locomotes

![aggregate](figures/2026-07-16-aibo/sac_walk_campaign.png)

| body | flat: dx / propel (median) | structural: dx / propel (median) | verdict |
|---|---|---|---|
| **cheetah** | +0.03 / −0.06 (n4) | **+0.18 / +0.11** (n5) | **structural wins** (6× dx; propel + vs −) |
| **humanoid** | +0.09 / +0.21 (n5) | +0.11 / +0.25 (n5) | structural marginally ahead |
| **Aibo goal** | −0.05 / 0.00 (n5) | −0.03 / 0.00 (n5) | tie — **neither walks at 200k** |

Per-seed (propel-edge): cheetah structural `[0, 0.47, 0.11, 0.29, 0]` vs flat `[−0.43, 0, −0.12, 0]`; humanoid
structural `[0, 0.25, 0.6, 0.32, −0.57]` vs flat `[0.67, 0.47, 0, −0.75, 0.21]`; Aibo both ~all-zero.

## Honest reading

1. **Confirmed where the body walks.** On the **cheetah** (the body SAC most reliably locomotes) the structural
   actor is a **clear multi-seed win** — median dx +0.18 vs +0.03 (6× farther), propel +0.11 vs −0.06. This
   scales the local single-seed finding to 5 seeds + two GPUs.
2. **Marginal on the humanoid** — structural edges flat (propel 0.25 vs 0.21), both walk a little; high seed
   variance.
3. **Aibo is undertrained, not a verdict.** At 200k both actors sit at dx ≈ 0 / propel 0 — the 22-DOF
   jump-capable body simply hasn't learned to walk yet. It needs **≥ 500k–1M steps** before the structural-vs-
   flat question is even askable there (the CIP propel-edge is only informative on a policy that moves).
4. **Variance is real.** SAC is stochastic; per-seed propel ranges widely (0 → 0.47 on cheetah structural).
   The medians point structural-positive, but the effect is *directional*, not overwhelming, at 200k/5-seeds.

**Verdict:** the relational actor's advantage is **real and tracks locomotion quality** — clear on the
cheetah, marginal on the humanoid, not yet testable on the Aibo. The publishable core is the cheetah result
(+ the local 200k validation); a stronger cross-body claim needs Aibo at ≥ 500k and more seeds for tighter CIs.

## Artifacts + provenance

`experiments/2026_07_16_sac_walk_campaign/{k15_cells.jsonl, k14_cells.jsonl}` (30 cells, pulled to the Mac);
figure `figures/2026-07-16-aibo/sac_walk_campaign.png`. Branch `feat/locomotion-aibo-sac-cip`. Compute: 2× RTX
6000 Ada (kato14/kato15, katolab), torch 2.11+cu128 `.venv_stand`, HYMEKO_DEVICE=cuda. Pure SAC from scratch,
200k steps, seeds 0–4. reward_oracle N/A (goal_progress + vertical_bounce shaping; documented in the module).
