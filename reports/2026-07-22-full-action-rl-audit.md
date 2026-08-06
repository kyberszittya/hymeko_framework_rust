---
title: Full-action RL regression — runtime-identity audit (diagnosis: REWARD_IDENTITY_MISMATCH)
date: 2026-07-22
branch: exp/coin-full-action-bc-sac-td3
status: UNVERIFIED_FULL_ACTION_RL_REGRESSION — runs invalidated, NO new campaign
quarantined: 0ca6853, 90e323c, 0a46d5e (not amended/deleted)
---

# Full-action RL "regression" — runtime-identity audit

The numerical observation stands (all 5 SAC and all 5 TD3 checkpoints below the BC on strict). The **causal
interpretation does not**. The bounded pre-campaign audit (diagnostics 1-3) found the failure mechanism, so no new
five-seed campaign was launched.

## Diagnosis: `REWARD_IDENTITY_MISMATCH` (+ reward↔eval misalignment)

### Step 1 — live runtime reward identity (read from the actual env instance, not the oracle)

| | value |
|---|---|
| live reward implementation | `hymeko_rl.train.coin_delivery_rl.delivery_reward` (Python fn on `DeliveryRLConfig`) |
| live reward terms | `w_progress 10 · (prev−dtz)` + `w_zone 1` (entry) + `w_center 3` (one-shot center) − `w_stall`/`w_drop` |
| live reward impl hash | `b5d4d1ff28577674` |
| declared spec | `data/robotics/galambos_task_deliver_v2b.hymeko` sha256 `98cd3ad6af02a1bf…` |
| v2b terms | `approach 4` + `bothapproach 4` + `oob 2` + `terminal_deliver_graded 30` + `zoneprog 10` + `body_progress_penalty 5` |
| **live env loads v2b?** | **NO** — the full-action path never reads the `.hymeko`; `env.step` calls `delivery_reward()` |
| **live == v2b?** | **FALSE** — `delivery_reward` has no contact-quality gate, no graded terminal, no body-progress penalty, no approach terms |

The full-action RL driver also ran **no reward-identity gate** — the "reward oracle certified" claim (from the
residual campaign's `certify_or_abort`) certifies v2b, which the env does **not** emit. This is exactly the
runtime-identity class of defect the 2026-07 reward audit found (v2b certified while the live env optimized the
default farmable reward).

### Step 2 — reward term trace + the load-bearing check

Basic ordering holds: returns toward `+138.4` > zero `−2.2` > away `−3.2` (the reward does credit progress). But the
decisive check: **`delivery_reward` gives `0.0` for HOLDING at center once the one-shot `w_center` is consumed** —
there is no term that rewards staying. The **strict eval requires a 6-step HELD dwell**. So:

> The training reward rewards *reaching* center; the eval metric grades *holding* center.

SAC/TD3 optimize reach-not-hold and are then graded on hold. The standalone BC holds (it clones the scripted
expert's continuous push). Removing the scripted base (the full-action contract) removed the thing that was doing the
holding — and `delivery_reward` provides no gradient to replace it. The sub-BC strict scores are therefore a
**reward↔eval confound**, not evidence that RL cannot improve a competent BC. (This is why the *residual* campaign
was only NO_EFFECT: the always-active base kept holding, masking the missing hold incentive.)

### Step 3 — observation / action contract (ruled OUT as the cause)

`max |SAC action − BC action| = max |TD3 − BC| = max |SAC − TD3| = 0.00e+00` on demo, BC-rollout, AND held-out obs
distributions; action bounds `[-1,1]`, tanh-squash, scale 1.0 for both archs; no active running obs-normalization
(the MetaWorld "covariate-shift was a norm bug" precedent does **not** apply here). The defect is the reward, not the
representation.

### Steps 4-5 — deferred (justified)

Critic calibration and the first-update microscope are only meaningful **after** the reward is corrected — a critic
calibrated on the wrong reward measures the wrong target. They are not run until step 1's mismatch is fixed.

## Consequence

- `FULL_ACTION_RL_REGRESSION` → **`UNVERIFIED_FULL_ACTION_RL_REGRESSION`**; the `90e323c` SAC/TD3 runs are
  **invalidated** for causal interpretation (kept on disk, not deleted).
- The scope-correction claim in the bde81ba report (standalone RL regresses) is **also unverified** and must not be
  treated as established until re-run under a reward whose optimum coincides with the strict-held-delivery metric.

## Required fix before any re-run

Align the training reward with the strict eval, one of:
1. add a **hold/dwell term** to `delivery_reward` (reward sustained in-zone/at-center residence, matching the 6-step
   strict dwell), or
2. wire the env to the **declared `galambos_task_deliver_v2b.hymeko`** (contact-quality-gated) via the reward loader
   and add a hard `live_reward_hash == declared_hash` launch assertion, or
3. grade on the reward's own optimum (center-reach) — but that abandons the held-delivery definition, so (1)/(2)
   are preferred.

Then re-run diagnostics 4-5 (critic calibration on the corrected reward, first-update microscope) and only then a
campaign, per the bounded order.

## Preserved (unchanged by this audit)

Historical residual SAC native (contact-prepared distribution), learned neutral bridge, corrected-physics bridge
retraining 3/9, and the full-action regression **as UNVERIFIED**. No earlier result rewritten.
