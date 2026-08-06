---
title: PHASE_GATED_RESIDUAL_CRITIC — AUTHORIZATION_BLOCKED; TARGET_SMOOTHING_CONTRACT_MISMATCH found
date: 2026-07-23
slug: phase-gated-residual-critic
task: coin_v3 delivery — phase-gated learned-residual TD3 (§6 critic authorization)
verdict: TARGET_SMOOTHING_CONTRACT_MISMATCH
prior_result_reclassified_as: PHASE_GATED_RESIDUAL_CRITIC_AUTHORIZATION_BLOCKED
authorizes_actor_update: false
---

# Composite-action critic — authorization blocked; smoothing mismatch found

**Created-at:** 2026-07-23 20:35 CEST
**Correction of the prior claim.** An earlier version of this report registered `CRITIC_NO_USEFUL_LOCAL_RANKING` as
a mechanism finding. That was an overreach and is **retracted**:

1. The authorization panel (`7000–7039`) was used **adaptively** to tune LR, batch, update count, gradient clipping,
   perturbation scale, and settling/contact capture → it is a **development panel**, not an untouched final audit.
2. The "signal below the value-estimation noise floor" statement compared within-state Q-spread against **absolute**
   twin disagreement `|Q1−Q2|` — an invalid comparison (absolute twin offset is a calibration offset, not a local
   action-noise floor).

**No residual actor update is authorized.** The prior authorization outcome is reclassified as
`PHASE_GATED_RESIDUAL_CRITIC_AUTHORIZATION_BLOCKED`, and this turn's finding is a plumbing defect that blocks trust.

## §3 target-smoothing audit → TARGET_SMOOTHING_CONTRACT_MISMATCH

The batch-size crash exposed a hardcoded `torch.zeros(256, 4)` in the critic-training target path. The audit:

- **Library** `residual_target_action` / `bounded_smoothed_residual` — batch-independent; default is **active
  stochastic smoothing** (`randn_like·0.2`, clipped to ±0.5, re-bounded to the residual range ±0.25); `pi_0` never
  smoothed; composite target ∈ [−4, 4]. **Correct** (3 new reference tests at batch **1, 7, 256, 512** pass).
- **Frozen critic config** (`train_critic`) — passed **explicit zeros** as the target noise ⇒ **target policy
  smoothing was DISABLED** during critic training. This is the §3 "accidentally replaced by zero noise" case: an
  **undeclared deviation from the declared TD3 target-smoothing contract**.

⇒ **`TARGET_SMOOTHING_CONTRACT_MISMATCH`.** The critic that produced the blocked authorization was trained under a
mis-implemented (smoothing-off) target. Its ranking result cannot be trusted as scientific evidence. **Fixed**:
`train_critic` now uses `noise=None` (the declared active smoothing); batch-independent construction verified.

## §2 panel separation (manifests, `critic_panel_manifests.json`)

| panel | seeds | n | disjoint |
|-------|-------|---|----------|
| critic_train | 6000–6059 | 60 | — |
| development_auth (repeatedly inspected) | 7000–7039 | 40 | ✓ vs train |
| **sealed_final (never inspected)** | 7060–7075 | 16 | ✓ vs train & dev |

Seed-disjoint; rollouts are deterministic per seed ⇒ state/trajectory-disjoint. The settling-capture repair changed
only development bucketing and added **no** cross-panel seeds. Rules out `CRITIC_PANEL_LEAKAGE_DETECTED`.

## §9 corrected interpretation — what the experiments DO and DO NOT support

**Supported:**
- realized returns differ materially among residual candidates (counterfactual-rollout spread up to 85 at
  fragile-contact states; labels deterministic r1==r2);
- the smoothing-disabled critic did **not** pass the repeatedly-inspected development authorization criteria;
- actor learning remains blocked.

**NOT supported (retracted):**
- "the residual value signal is fundamentally below the critic noise floor";
- "no critic can learn the local residual geometry";
- "the phase-gated residual route is exhausted."

## §10 figure relabeled

`coin_residual_critic_auth.png` is relabeled a **development diagnostic** (smoothing-disabled critic; superseded);
the min-Q panel is marked CENTERED and the "value-estimation noise floor" claim is removed.

## Registry

`F-SAC-13` (the premature mechanism negative) is **removed** from `canonical_findings.json` (57 findings). It will be
reconsidered only after the corrected critic is evaluated on the sealed final panel with the proper metrics.

## Validated infrastructure (unchanged, correct)

Encoder `d6301d06` (deterministic, distinct modes/sides), `CompositeTwinCritic` `423d7699` (independent Q1/Q2,
composite-action input), stored-`gate_tp1` target (`9fa35a4`), counterfactual-rollout labeler (deterministic).
Tests: critic 8/8, replay **11/11** (incl. 3 smoothing-audit tests), overall **55/55**.

## §6.15 regression — unchanged

Update-0: HL 3/9, VAL 2/30, grasp 9/9, delivered {1011,1447,1568}, composite−base maxdiff 0.0, π₀ hash unchanged.

## Next turn (the corrected re-audit)

1. Retrain the critic with the **declared active smoothing** (config now corrected), freeze it exactly (§1) with
   full manifests.
2. Evaluate **Q1, Q2, and min-Q separately** (Q1 is the actor-driving critic) (§4).
3. Report **centered** local disagreement `ΔQ_i(a)=Q_i(s,a)−Q_i(s,a0)`: mean |ΔQ1−ΔQ2|, correlation, order/top
   agreement, action-gradient cosine + sign agreement — NOT absolute |Q1−Q2| (§5).
4. **Margin-aware** ranking (|gap| ≥ 1/5/10, top-vs-bottom quartile, best-vs-worst; top-1 regret; best-action recall;
   does the actor-gradient direction improve empirical short-horizon return) — full fixed set declared before the
   audit (§6).
5. **Bootstrap** intervals per family; settling n=3 flagged **underpowered** (not silently expanded) (§7).
6. Run the **sealed final panel `7060–7075` exactly once** on the frozen corrected critic (§8) →
   `PHASE_GATED_RESIDUAL_CRITIC_FINAL_AUDIT_PASS` / `RESIDUAL_CRITIC_LOCAL_RANKING_FAILURE` /
   `RESIDUAL_CRITIC_FINAL_AUDIT_UNDERPOWERED`.

**CORE.YAML:** none. Frozen π₀ (`1902454c`) / reward / γ / bundle / obs / gate thresholds / residual range / action
bounds unchanged. No actor optimization. SAC quarantined. Final panel sealed. Mac; kato14 clean.
