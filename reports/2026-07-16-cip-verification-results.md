---
title: CIP-verification campaign — structural-vs-flat propel-edge across bodies (honest, qualified)
date: 2026-07-16
status: complete (48/48 cells, 6 seeds) — finding CONFIRMED on cheetah, DEGENERATE on Aibo/humanoid
core_yaml_touched: none
---

# CIP-verification campaign — does the structural actor's propel-edge advantage generalise?

## Claim under test

From the Aibo cheap loop: a **structural (relational) actor** improves the CIP causal **propel-edge**
(`leg_speed ⇒ forward_vx`) where a **flat MLP degrades it**, even in a collapse-prone off-policy smoke.
This campaign verifies whether that **generalises across bodies + scenarios**, multi-seed.

## Method

`experiments/exp_cip_verification_campaign.py`: **4 scenarios** (Aibo goal-reach @3 m, @5 m; biped humanoid;
planar cheetah) **× {flat, structural} × 6 seeds** = 48 cells. Each cell: CIP-diagnose the scripted
demonstrator → train under the CIP-informed reward (`vertical_bounce`) + asymmetric CTDE critic + demonstrator
BC anchor (TD3+BC, 6 k steps) → CIP **re-diagnose** the learned policy. Claim rests on the **median** of the
per-actor propel-edge Δ, over 6 seeds. All 48 cells ran, **0 errors**, 120 min, CPU, RSS < 0.5 GB.

## Result — honest and qualified

![per-scenario](figures/2026-07-16-aibo/cip_verification.png)

| scenario | flat Δ (median) | structural Δ (median) | structural advantage |
|---|---|---|---|
| **cheetah_run** | **−0.26** | **+0.28** | **+0.54** ✅ |
| aibo_goal_3m | −0.13 | −0.13 | 0.00 (degenerate) |
| aibo_goal_5m | −0.13 | −0.13 | 0.00 (degenerate) |
| humanoid_walk | 0.00 | 0.00 | 0.00 (degenerate) |

**Confirmed on the cheetah** — a clean multi-seed win: structural **+0.60 / +0.55 / +0.86** on 3 of 6 seeds
(0 on the rest); flat **−0.62 / −0.53 / −0.54**. Where the demonstrator actually locomotes, the relational
actor routes leg energy into forward motion and the flat MLP does not — the finding holds.

**Degenerate on Aibo + humanoid — and the reason is a metric-integrity insight, not equality:**
- Aibo: the recurring **−0.131** is `learned − demo` when the learned policy has **zero propel-edge** — the
  short **TD3+BC collapses to no forward motion**, so DirectLiNGAM finds no `leg_speed ⇒ forward_vx` signal.
  Both actors collapse similarly → no discrimination. (A few seeds break out: flat +0.22, structural +0.10.)
- Humanoid: propel-edge ≈ **0 throughout, including the demonstrator** — the scripted CpgGait barely moves
  the biped, so there is no forward-motion causal signal to improve on either side.

**Verdict:** the propel-edge advantage is **real but scenario-dependent** — it manifests only on a body/gait
that genuinely walks (cheetah). The **CIP propel-edge is only informative on a locomoting policy**; on a
collapsed (Aibo) or barely-moving (humanoid) policy it saturates to 0 and cannot discriminate representations.
So the strong "generalises across bodies" claim is **not supported at this scale** — it is confounded by (a)
TD3+BC collapse and (b) weak scripted demonstrators, exactly the two things the arc already flagged.

## What this implies (the honest next step)

The verification did its job — it **refuted the naive generalisation** and localised *why*. To give the Aibo
and humanoid a fair test:
1. **DAgger, not TD3+BC** (the established non-collapsing lever, per Aibo *standing* 0.958) — so the learned
   policies actually move and the propel-edge has signal.
2. **A stronger humanoid demonstrator** (the CpgGait barely locomotes) — or measure on the cheetah/Aibo where
   a forward gait exists.
3. Then re-run this exact grid; the cheetah result predicts the effect should reappear wherever the policy
   walks.

The cheetah result alone is a legitimate, reproducible, multi-seed finding; the cross-body claim needs the
DAgger campaign before it is publishable as *general*.

## Provenance

Git branch `integration/fanuc-pick-place-canonical` (working tree dirty). 48 cells, seeds 0–5, TD3+BC 6 k
steps/cell, horizon 300. Artifacts: `experiments/2026_07_16_cip_verification/{cells.jsonl, summary.json,
run.log}`. Env: Python 3.11 `.venv`, mujoco 3.10.0, macOS Apple-Silicon CPU. reward_oracle N/A (goal_progress
+ shaping; documented in the campaign module).
