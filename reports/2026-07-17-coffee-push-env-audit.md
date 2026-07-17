---
title: "Coffee-Push environment sanity audit (before calling 0% an exploration wall)"
date: 2026-07-17
branch: audit/sac-cip-forensic
core_yaml_touched: none
verdict: "no env/wrapper/metric defect; 0% is a genuine, mechanistically-characterized exploration barrier"
---

# Coffee-Push environment sanity audit

**Aiko · 2026-07-17 · branch `audit/sac-cip-forensic` (not merged) · kato14/kato15 untouched · no long runs**

Rollout-only audit through the **exact training wrapper** (`_ObsNorm(_make_native_env("coffee-push-v3-goal-
observable"))`), before interpreting the from-scratch 0% as a fundamental wall. Script
`experiments/coffee_push_env_audit/audit.py`; data `env_audit.json`. The wrapper is a pure observation transform,
so the scripted expert is driven with de-normalized obs (`raw = norm*max(std,0.05) + mean`, exact inverse).

## Results (7 checks)

| # | check | result | defect? |
|---|---|---|---|
| 1 | scripted expert **through the wrapper** | **success 20/20 (1.0)** | none — task solvable, wrapper preserves it |
| 2 | eval success metric | `info["success"]` **fires** on the success trajectory | none — metric correct |
| 3 | raw + normalized reward | raw sum 2559, per-step **[0.038, 10.0]** (dense); RMS-normalized **[0.014, 6.13]** | none — dense, non-distorted |
| 4 | action bounds + per-dim effect | `[−1,1]`; dx→eef_x (+0.106), dy→eef_y (+0.113), dz→eef_z (+0.110), **gripper→gripper (−0.72)** | none — every dim correct |
| 5 | initial-policy saturation | action σ ≈ [0.63, 0.63, 0.63, 0.71]; **only 2.3 % of actions saturate `>0.99`**; mean logπ −2.5 | none — healthy stochasticity |
| 6 | random & initial-SAC behaviour | **contact_fraction = 0.0**; object displacement ≈ 2.5 mm; best obj→target ≈ 0.08–0.14 m; **100 % episodes get nonzero reward** | none — but reveals the barrier |
| 7 | resets / distribution | task **randomized every reset** (seed does *not* fix it); object/goal spread > 0; **train wrapper ≡ eval wrapper** | none — no silent shift |

## Verdict — no defect; the 0% is a real, characterized exploration barrier

Every component is sound: the task is solvable (scripted expert **100 %** through the exact wrapper), the success
metric fires, the reward is **dense and non-distorted** (nonzero every step; RMS-normalized to O(1–6)), all four
action dimensions — **including the gripper** — have the correct physical effect and do not saturate under the
initial policy, and there is **no train/eval distribution shift** (identical wrapper; task randomized per reset on
both sides).

**The mechanism of the 0%** is crisp: under random *and* initial-SAC policies the arm **never contacts or moves the
mug** (contact = 0, displacement ≈ 2.5 mm), so **no success signal is ever generated to learn from** — over a
goal-randomized task — even though the dense reward is always nonzero. The barrier is discovering the
reach→contact→push sequence from scratch, not any implementation error.

## Implication for the running experiment

- This does **not** invalidate the running kato14/kato15 jobs (plain SAC is canonical — see
  `2026-07-17-sac-cip-forensic-audit.md`).
- It **predicts** the likely outcome under the interpretation rule: since CDS augments the *irrelevant* dims of
  *already-visited* transitions and adds no contact/exploration, it is unlikely to manufacture the missing success
  signal on its own → **plain and CDS both near 0 = budget-limited both-fail**, pointing to the exploration
  controls (demo-seeded replay, scripted warm-start transitions, curriculum) rather than any SAC/CDS change.
- If instead **CDS rises while plain stays 0**, that would be a genuine (and surprising) signal that CDS's data
  diversification alone crosses the barrier — to be replicated with seeds + minimal ablations before claiming it.

## Naming correction (committed alongside)

The augmented arm is **CDS only** (SAC + counterfactual data augmentation), **not full CIP** (no empowerment /
reverse-policy / causal-weighted intrinsic reward). Renamed accordingly: `CipReplayAugmentor` → `CdsReplayAugmentor`
(back-compat alias kept); runner arm/flag/tag `cip` → `cds` (`--cip` kept as a deprecated alias); docstrings state
"CDS, not full CIP". The **running jobs' filenames use the old `cip_` tag** — when aggregated, those are the **CDS**
arm; do not report them as "CIP".

## Provenance
No SAC/CIP core logic changed (the rename is labels + one class name + alias). Env: Mac Apple-Silicon `.venv`
(torch 2.12, metaworld 3.0.0, mujoco 3.10). Scripted policy `SawyerCoffeePushV3Policy`. 11 `test_cip_augment` tests
green post-rename; ruff clean.
