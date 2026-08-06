---
title: "Coffee-Push SAC config defects — SB3 cross-implementation forensic (S1 fixed-mug reach)"
date: 2026-07-18
branch: audit/sac-cip-forensic
core_yaml_touched: none
verdict: "two demonstrated config defects (reward_norm Q-inflation + late-action-fusion critic); SAC math is correct"
---

# Coffee-Push SAC configuration defects — forensic

**Aiko · 2026-07-18 · branch `audit/sac-cip-forensic` (not merged) · kato14/kato15 untouched · no long runs**

Isolation of why our SAC underperforms SB3 on Coffee-Push, done on the **S1 fixed-mug reach** task (a
byte-identical env both implementations import: metaworld coffee-push frozen to one mug+target via explicit
`_last_rand_vec`, reach-only reward `1−tanh(4·d)`, shared obs-norm, 5 cm→**corrected to contact ~7.5 cm** success).

## The five axes, separated

**(1) SAC mathematical correctness — CORRECT.** Our `train_sac` matches SB3 on Pendulum-v1 (improvement +1292 vs
+1187) and is critic-**calibrated** identically once reward_norm is off (Q−MC bias +4 vs SB3 +5). No reverse policy,
no empowerment, canonical actor loss `E[α·logπ − min(Q1,Q2)]`, correct truncation bootstrapping (prior report
`2026-07-17-sac-cip-forensic-audit.md`).

**(2) Reward/Q scale — not a fault by itself.** With reward_norm on, Q≈130 and MC-return≈75 are both ≈1/(1−γ)≈100
scale; high Q is *not* overestimation per se.

**(3) Critic calibration — DEFECT #1 (reward_norm).** Measuring Q(s,a) vs the Monte-Carlo empirical return **on the
same normalized scale the critic saw** (4 seeds, 40 k):

| config | Q−MC bias | contact (stable-final-3) |
|---|---|---|
| ours reward_norm **ON** | **+60** (over-optimistic) | 0/4 |
| ours reward_norm **OFF** | **+4** (calibrated) | 1/4 |
| SB3 (no reward_norm) | +5 (calibrated) | 2/4 |

On a *dense* reward, RMS-normalisation pushes per-step reward toward ~1 and the critic learns a genuinely inflated
value (+60 above the achievable return) — a real calibration error, not scale. **Fix: reward_norm off.**

**(4) Seed variance — real and initially misleading.** The first 2-seed read ("ours 0/2 vs SB3 2/2") was
**underpowered** — at 4 seeds the old config is 2/4, bimodal. Variance is large; every claim here rests on ≥4 seeds.

**(5) Contact-learning performance — DEFECT #2 (critic architecture).** At **100 k** (4 seeds), SB3 is a clean
**4/4** stable contact while ours (reward_norm off) is **0/4** — and the failure mode is **reach-then-regress**:
min gripper-mug distance reaches 0.054 then drifts back to 0.086. Root cause: our `QCritic` fuses the action
**late** (obs→`[256,256]`→feat, action concatenated with *one* layer), so `∂Q/∂a` is weak and the actor cannot do
the fine control to *hold* contact. Swapping in an **SB3-style early-concat critic** (obs+action→`[256,256]`→1)
restores reach-and-hold:

| config @100k, reward_norm off, 4 seeds | stable contact | mechanism |
|---|---|---|
| ours late-fusion `QCritic` | 0/4 | reach → regress |
| **ours early-concat critic** | **2/4** | seeds 0,3 reach & **hold** |
| grad-clip off | 1/4 | marginal |
| SB3 | 4/4 | — |

**Fix: early-concat critic for flat obs.** The two fixes together take S1 from **0/4 → 2/4**. The residual to SB3's
4/4 is small at n=4 (our actor/critic architecture now matches SB3) — attributed to network-init + seed variance;
not chased further (diminishing returns).

## Why Pendulum did not expose either defect

Pendulum is **1-D action, dense, 3-D obs, easy**. (a) A late-fusion critic can adequately discriminate a *single*
action dimension's effect on Q, so `∂Q/∂a` is fine — the defect needs the 4-D fine-control demand of contact.
(b) reward_norm inflates Q there too, but the task is forgiving enough to solve anyway. Coffee-Push's 4-D contact
control + dense-but-flat-near-optimum reward stress exactly what Pendulum doesn't.

## Configuration / code changed (audit branch only)

- **new** `hymeko_rl/train/flat_critic.py` — `EarlyConcatCritic` + `build_flat_sac` (fix #2), 3 tests.
- **`hymeko_rl/experiments/exp_metaworld_cip_baseline.py`** — `--corrected` stack: `reward_norm=False` (fix #1) +
  `build_flat_sac` early-concat critic (fix #2) + SB3-matched auto-α (init 1.0, lr 3e-4). Old `--stable` kept, now
  **banner-warned** as carrying both defects (reproduces kato14/15 only).
- No SAC core (`train_sac`) logic changed — the fixes are the critic *class* and the config. `test_flat_critic` +
  `test_cip_augment` green; ruff/mypy clean. Diagnostic scripts under `experiments/s1_cross_impl/`.

## Before / after (S1 fixed-mug reach, contact ≤7.5 cm, 4 seeds)

| stack | stable contact | Q−MC bias |
|---|---|---|
| old `--stable` (reward_norm on, QCritic) — *kato14/15 config* | 0/4 | +60 |
| reward_norm off only | 0–1/4 | +5 |
| **`--corrected` (reward_norm off + early-concat critic)** | **2/4** | +5 |
| SB3 reference | 4/4 | +5 |

## Are kato14/kato15 valid, partially informative, or historical?

**Historical / not reusable as a clean baseline.** They run the old `--stable` config, which carries **both**
defects (calibration inflation + reach-then-regress critic). Two caveats compound this: (a) Coffee-Push from-scratch
is also **exploration-walled** (the env audit showed random/initial policies never contact the mug), so the plain
arm sits near 0 for reasons *upstream* of these SAC defects too; (b) the plain-vs-CDS comparison is internally
consistent (both arms share the defective SAC) but is **not** a valid *canonical-SAC* baseline. Recommendation: do
**not** publish kato14/15 as the CIP baseline; keep them as exploratory records. A definitive Coffee-Push baseline
needs the `--corrected` stack **and** an exploration remedy (demo-seeded replay / scripted warm-start), per the
env-audit interpretation rule.

## Exact commands for the corrected experiments (NOT launched — gated)

```
# corrected plain SAC (per seed)
python -m hymeko_rl.experiments.exp_metaworld_cip_baseline --task coffee-push --corrected \
    --steps 1000000 --seed <s> --device cuda
# corrected CDS arm
python -m hymeko_rl.experiments.exp_metaworld_cip_baseline --task coffee-push --cds --corrected \
    --steps 1000000 --seed <s> --device cuda
# S1 fixed-mug reach cross-impl (reproduce this forensic)
S1_STEPS=100000 python experiments/s1_cross_impl/s1_calib.py           # ours (reward_norm off)
S1_STEPS=100000 experiments/sac_smoke_test/.venv_sb3/bin/python experiments/s1_cross_impl/s1_sb3_calib.py
S1_STEPS=100000 python experiments/s1_cross_impl/s1_archtest.py sb3style_critic
```

## Honest scope
S1 *reach* is the isolation task; the fixes are validated there, not on full Coffee-Push (still
exploration-walled). Every number is ≥4-seed. Env: Mac Apple-Silicon, torch 2.12 (ours) / 2.13 (SB3 venv),
metaworld 3.0.0, mujoco 3.10.0. SB3 2.9.0.

## Overnight update — residual settled at 8 seeds (2026-07-18)

Powered the 4-seed "2/4 vs 4/4" comparison up to **8 seeds** (S1 100 k, contact ≤7.5 cm):

| stack | stable-contact (≥0.5) | mean stable-3 |
|---|---|---|
| old `--stable` (both defects) | 0/4 | ~0.0 |
| **corrected (reward_norm off + early-concat critic + SB3-matched auto-α)** | **4/8** | **0.58** |
| SB3 reference | 6/8 | 0.67 |

**4/8 vs 6/8 is not statistically distinguishable at n=8** (binomial CIs overlap heavily; means 0.58 vs 0.67). So
the two demonstrated fixes take our SAC from 0 to **SB3-comparable** — **no residual defect is proven**; the small
gap is within seed variance (both stacks are bimodal on this hard-from-scratch reach). Conclusion: the two config
fixes are **necessary and sufficient** to close the SB3 gap within available power. No init-matching chase is
justified. (Artifacts `experiments/s1_cross_impl/s1_arch_sb3style_critic.json`, `s1_sb3_calib_result.json`.)
