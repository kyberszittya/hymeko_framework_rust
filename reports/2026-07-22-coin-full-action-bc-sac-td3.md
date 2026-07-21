---
title: Standalone full-action BC → SAC/TD3 under corrected physics
date: 2026-07-22
branch: exp/coin-full-action-bc-sac-td3
baseline: bde81ba (residual campaign, preserved)
verdict: SAC FULL_ACTION_RL_REGRESSION / TD3 FULL_ACTION_RL_REGRESSION
---

# Standalone full-action BC → SAC/TD3 (2026-07-22)

## The question this answers (that the residual campaign did not)

The residual campaign (`bde81ba`) trained a residual policy on top of an **always-active scripted base**
(`u_exec = clip(grasp_carry + delta·tanh(policy))`). That measured whether a residual helps a controller that is
already delivering — it did **not** test whether a **standalone BC clone** of the scripted expert can be improved by
RL. This experiment does: **scripted expert → full-action BC clone → standalone SAC/TD3**, with the scripted base
DISABLED during the policy rollout (`u_exec = clip(policy(observation))`).

## Headline — standalone RL REGRESSES the BC (primary baseline)

Headline eval on the frozen panel + 50 held-out POINT transport states (59 states, disjoint from the VAL selection
set), matched horizon 160, one consistent env-native + strict metric:

| source | native center | strict /59 | success-curve AUC | TTS median |
|---|---|---|---|---|
| scripted expert | 0.85 | 41 | 0.348 | 66 |
| **standalone BC (primary baseline)** | **0.71** | **34** | **0.288** | **66** |
| SAC (median / 5 seeds) | 0.53 | 17 | 0.160 | — |
| TD3 (median / 5 seeds) | 0.59 | 24 | 0.236 | — |
| zero-action control | 0.02 | 0 | 0.0 | — |

**Verdict: SAC `FULL_ACTION_RL_REGRESSION`, TD3 `FULL_ACTION_RL_REGRESSION`.** Every one of the 10 RL checkpoints
delivers fewer strict than the BC (best RL = TD3-s1 at 31 < BC 34; SAC seeds 13–26; TD3 seeds 10–31). No temporal
positive either — the faster seeds have *worse* final strict, so there is no "same success, faster" case. With the
scripted base removed, off-policy RL drifts off the competent BC manifold (SAC critic loss climbs to ~140) and
degrades it. See `full_action_regression.png`.

This is **stronger** than the residual result: the residual was `NO_EFFECT` because the scripted base held it at the
ceiling; remove the base and standalone RL falls *below* the BC ceiling.

## Steps (bounded order, all executed)

1. **Evaluator horizon fix** (`9cc0505`) — removed the hard-coded `max_steps=60` from the shared `evaluate()`; it now
   rolls the declared `env.cfg.horizon`. 4 contract tests (default == declared horizon; horizon ≠ 60; a short probe
   never reports more success than the full horizon; success-by-time monotone). A shorter probe is an explicit
   time-to-success diagnostic only.
2. **Full-action env + expert dataset** (`0ca6853`) — `FullActionDeliveryEnv`: `u_exec = clip(policy(obs))`, no
   scripted base (`_base()` never called), no prefix during rollout, center-terminal removed so the 6-step strict
   dwell can accumulate. Verified: zero-action drifts the coin away (base disabled); scripted grasp_carry as the FULL
   action delivers 9/9 center on the panel; expert dataset 131 successful trajectories / 20 960 transitions.
3-4. **Standalone BC + DAgger + gate** (`0ca6853`) — BC target `u_target = u_expert_executed` (full action, not zero
   residual). Plain BC suffered compounding-error covariate shift (panel 3/9 center, 0 strict despite 0.009 per-step
   MSE). DAgger (3 rounds, aggregate 20 960 → 59 360) established it: BC held-out strict 31/50 ≈ expert 33/50 (94 %),
   panel 6/9 center / 4/9 strict. Gate G1–G6 pass → `FULL_ACTION_BC_ESTABLISHED` (imit MSE 0.018, zero-action strict 0,
   action-source delta 0). `BC_FULL_ACTION_PHYSICAL_V1.pt` sha256 a6c1b84d9d66.
5. **SAC/TD3 from the same BC** (`90e323c`) — identical init verified max|Δ| = 0 for SAC-vs-BC, TD3-vs-BC, SAC-vs-TD3.
   Scripted base disabled; same reward v2b, demos, distributions, 5 seeds, 100k, checkpoint selection on native.
6. **Final + temporal + generalization eval** (`90e323c`) — the table above; verdicts against the standalone BC.
7. **Corrected-physics bridge (separate)** — retrained a **fresh** handoff BC on 58 corrected-physics carry demos
   (E-approach → grasp → scripted carry, corrected physics), no scripted carry in the final rollout. The fresh bridge
   delivers **3/9** on the panel (grasp 5/9) — recovering from the frozen filtered-physics policy's 1/9 and matching
   the historical *filtered* bridge's 3/9. The bridge **method** is robust to the correction; only the frozen
   **policy** failed to transfer. `HANDOFF_CORRECTED_V1.pt`.

## Corrected scientific scope (amends the bde81ba report)

The `bde81ba` campaign tested a residual policy over an always-active scripted base; it did **not** test standalone
BC → standalone RL. The broad claim *"local off-policy RL caps at the supervised ceiling"* is **not retained** as
stated. The standalone full-action experiment shows the stronger, more specific result:

- **Residual on scripted base:** `NO_EFFECT` — the base holds the residual at the scripted ceiling.
- **Standalone from a competent BC clone:** `REGRESSION` — off-policy RL falls *below* the BC ceiling on every seed.

So under corrected physics, on this task, the productive direction is imitation (BC + DAgger reached 94 % of the
expert); off-policy RL (SAC/TD3) does not improve a competent standalone clone and degrades it.

## Files / tests / provenance

- New: `hymeko_rl/train/coin_full_action.py`, `hymeko_rl/experiments/coin_full_action_{bc,rl,eval}.py`,
  `hymeko_rl/tests/test_{eval_horizon_contract,coin_full_action}.py`. Modified: `hymeko_rl/experiments/coin_two_arm_sac.py` (horizon).
- Tests: eval-horizon contract 4/4, full-action env 4/4, BC gate G1–G6 pass; ruff clean. CORE.YAML untouched.
- Host: Apple-Silicon Mac, CPU. torch/mujoco per `.venv`. Seeds: train 1100-1300, VAL 1300-1330 (selection),
  headline panel 1011.. + held-out 1000-1099 (disjoint). RL seeds 0-4. RL stochastic (BLAS) → 5-seed medians.
- `BC_FULL_ACTION_PHYSICAL_V1.pt` a6c1b84d9d66; identical-init Δ=0; zero-action strict 0 (base disabled proof).

## Open issues

- SAC critic divergence (loss → ~140) under the standalone contract; the `.stable()` stack bounds but does not cure
  it. Consistent with the regression — the critic overestimation is what pushes the policy off the BC manifold.
- Other `max_steps=60` call sites remain in preserved sibling experiments (`coin_bridge_*`, `coin_nstep_exp`,
  `coin_generator_exp`); left untouched to preserve their committed results. Fix them if those lines are revisited.
