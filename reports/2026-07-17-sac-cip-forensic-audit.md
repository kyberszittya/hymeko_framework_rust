---
title: "Forensic audit — is the Coffee-Push 'plain SAC' canonical? (SB3 reference gate)"
date: 2026-07-17
branch: audit/sac-cip-forensic
core_yaml_touched: none
classification: REFERENCE_PASS_OUR_PASS
verdict: "plain SAC IS canonical + correct; Coffee-Push 0% is exploration, not an implementation bug"
---

# Forensic audit: Coffee-Push SAC / CIP

**Aiko · 2026-07-17 · branch `audit/sac-cip-forensic` (NOT merged) · kato14/kato15 jobs untouched**

## TL;DR

- **Was the old "plain SAC" actually canonical SAC?** **Yes.** Static trace + repo-wide grep + an empirical SB3
  reference gate all agree. It is not an ablated CIP/ACE agent.
- **Classification: `REFERENCE_PASS_OUR_PASS`.** SB3 SAC and our `train_sac` both learn Pendulum-v1 to the same
  level (ours marginally better). The SAC core is functional; the Coffee-Push 0% is **environment/exploration-
  specific**, not a SAC bug.
- **Bugs in the SAC core: none.** No SAC code was changed (nothing to fix). Two *operational* issues found earlier
  this session are already fixed (dedicated eval-env for MetaWorld's reset contract; shared-NFS log-dir split).

## 1. Hypothesis under test

> the current "plain SAC" baseline may be an ablated CIP/ACE agent (reverse policy, empowerment, causal weights).

**Refuted.** Repo-wide grep for `policy_reverse|reverse_policy|empowerment|causal_empower|causal_weight|inverse_model`
across `hymeko_rl/` returns **exactly one hit**: a *docstring note* in `eval/cip/cip_augment.py` stating the
empowerment term is **deferred / not implemented**. There is no reverse policy, no empowerment loss, no causal
weighting, and no in-loop causal model anywhere in the codebase, let alone the training path.

## 2. Static trace of the exact training path

`exp_metaworld_cip_baseline.run_cip_seed` → `build_sac("mlp", …)` (clean actor/critics, no causal refs) →
`train_sac(…, augmentor=None for plain / CipReplayAugmentor for CIP)`. Findings, per the audit checklist:

| Concern | Finding | Canonical? |
|---|---|---|
| Actor objective | `train/sac.py:365` `(alpha * logp - q_pi).mean()`, `q_pi = min_i Q_i(s,a)` | ✅ `E[α·logπ(a|s) − min(Q1,Q2)]` |
| Critic target | `y = r + γ(1−d)(min_i Q̄_i(s',a') − α·logπ(a'))`; `r` optionally RMS-normalised | ✅ (reward-norm is a documented deviation) |
| Entropy temperature | `−(log_alpha·(logp+target_entropy).detach()).mean()`, target `−dim(A)` | ✅ standard auto-α |
| Replay insert/sample | one `ReplayBuffer`; critic samples the **same** buffer the augmentor writes to | ✅ |
| Counterfactual aug | only when `augmentor` set; plain path never augments | ✅ gated |
| Causal weights | used **only** to pick the CDS swap dim; **never** enter any loss | ✅ (CDS-only) |
| Action scaling/clip | `a = action_scale·tanh(pre)`, `action_scale=1` (MetaWorld) / `2` (Pendulum) ⇒ in-bounds; warm-up uniform in `[−scale,scale]` | ✅ |
| terminated vs truncated | `buf.add(…, done=bool(terminated))`; reset on `terminated or truncated` ⇒ **truncation bootstraps** | ✅ correct |
| Target networks | `_polyak(target, src, tau=0.005)` each update; separate actor/critic optimisers | ✅ |
| CIP-only config defaults in plain | `bc_coef=0`, `rollout_anchor_coef=0`, `dagger_teacher=None`, `compile=False`, `augmentor=None` | ✅ all inert |

Actor/critic loss formulas **before == after** (no change was needed):
`L_actor = E_s[ α·logπ(a|s) − min_i Q_i(s,a) ]`, `a~π(·|s)`;
`L_critic = Σ_i E[ (Q_i(s,a) − y)² ]`, `y = r + γ(1−d)(min_i Q̄_i(s',a') − α·logπ(a'|s'))`.

## 3. Empirical gate — SB3 reference vs ours on Pendulum-v1

Isolated venv (`experiments/sac_smoke_test/.venv_sb3`): **stable-baselines3 2.9.0**, **torch 2.13.0**, gymnasium
(classic-control). Our impl runs in the project `.venv` (torch 2.12.0, gymnasium 1.3.0). Seed 42, 50k steps, γ 0.99,
batch 256, lr 3e-4, auto-entropy; deterministic eval 10 before / 20 after.

| impl | config | before | after | improvement | action σ | PASS |
|---|---|---|---|---|---|---|
| SB3 SAC | default | −1341.4 | −154.8 | **1186.6** | 0.855 | ✅ |
| our `train_sac` | matched (init_α 1.0, reward_norm **off**) | −1438.8 | **−147.1** | **1291.7** | >0 | ✅ |
| our `train_sac` | **coffee-push cfg** (`--stable` init_α 0.2, reward_norm **on**) | −1438.8 | **−145.2** | **1293.6** | >0 | ✅ |

Pass bar (improvement ≥ 500, final > −500, no NaN, action σ > 0) met by all three. Our greedy eval reaches ≈ −135
by step 5000 and holds; critic loss small/stable, actor loss positive and decreasing, α auto-anneals — clean SAC
dynamics, no NaN. **The coffee-push config itself learns Pendulum**, so `reward_norm` and `--stable` are not the
cause of the Coffee-Push 0%. Figure: `reports/figures/… → experiments/sac_smoke_test/sac_correctness_comparison.png`;
curves in `sb3_curve.csv`, results in `sb3_result.json` / `our_sac_result.json`.

## 4. Deviations from textbook SAC (documented, all benign)

1. **`reward_norm=True`** (running-RMS reward scaling) — anti-divergence; not in Haarnoja et al. Verified benign:
   learns Pendulum with it **on**.
2. **`max_grad_norm=10`** gradient-norm clip — anti-divergence; not textbook. Benign.
3. `SACConfig` carries dormant coin-toss/pick-place seams (`bc_coef`, `rollout_anchor_coef`, `dagger_teacher`,
   `greedy_rollout`, `alpha_mode`, `compile`) — all default OFF; the plain path exercises none. They are *surface
   area*, not behaviour. (A genuinely minimal SAC would drop them; see §6.)

## 5. Are the running kato14/kato15 results still interpretable?

**Yes.** The plain SAC is correct, so **coffee-push plain = 0% is a valid measurement of the from-scratch
exploration wall**, not a broken baseline. The CIP arm (kato14) is a valid test of whether CDS augmentation breaks
that wall. The one honest caveat is a *task* property, not an implementation defect: from-scratch SAC on coffee-push
may be exploration-limited such that **both** plain and CIP stay near 0 (the documented Stage-B wall) — in which
case the result is "inconclusive at this budget," not "buggy." Nothing in this audit invalidates the running jobs.

## 6. Bugs found / files changed

- **SAC-core bugs: none.** No SAC/CIP source changed on this branch.
- Files added (smoke-test only): `experiments/sac_smoke_test/{sb3_reference.py, our_sac.py, sb3_result.json,
  sb3_curve.csv, our_sac_result.json, sac_correctness_comparison.png}` + this report.
- Prior-session operational fixes (already committed on `integration/hymeko-main`, not part of this branch):
  dedicated eval-env for MetaWorld's reset-after-truncation contract; per-machine output dirs for the shared NFS.

## 7. Commands

```
# SB3 reference (isolated venv)
uv venv experiments/sac_smoke_test/.venv_sb3 --python 3.11
uv pip install --python experiments/sac_smoke_test/.venv_sb3/bin/python stable-baselines3 "gymnasium[classic-control]"
experiments/sac_smoke_test/.venv_sb3/bin/python experiments/sac_smoke_test/sb3_reference.py
# ours (project venv)
PYTHONPATH=$PWD .venv/bin/python experiments/sac_smoke_test/our_sac.py
```

## 8. On the requested 4-mode harness (`sac` / `sac_cda` / `sac_empowerment` / `cip_full`)

The SB3 gate removes the urgency (no SAC bug to fix). The harness remains worthwhile as a **clean separation +
extension**, but note two facts it must honour: (a) a genuinely canonical `sac` mode should be a *separate minimal
trainer* (dropping the dormant coin-toss seams), not `train_sac` with flags off; (b) `sac_empowerment` and
`cip_full` require **implementing the empowerment term** (reverse/source policy `q(a|s,s')` + causal-weighted
intrinsic reward) — which **does not exist yet** (only CDS does). That is net-new work, not a refactor. Recommended
as a follow-up, gated on whether the empowerment modes are actually needed for the Ito/Kato story.
