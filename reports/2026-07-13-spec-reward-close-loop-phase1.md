---
title: "Closing the spec_bench → CIP loop — Phase 1: the arbitrated HTL spec as a reward (de-risk)"
date: 2026-07-13
author: Aiko (Opus 4.8)
branch: hsikan-mlp-hybrid-audit
stage: phase-1-derisk
status: COMPLETE — gates Phase-2 RL
tags: [spec_bench, cip, metaworld, htl, reward, close-the-loop]
---

# Phase 1 — the arbitrated HTL success spec becomes a per-step reward (offline de-risk)

**[2026-07-13 18:46 JST]**

## Summary

The `spec_bench` arc produced an *arbitrated* HTL success spec (LLM structure + HyMeKo threshold-calibration and
`hymeko_pgraph` conjunct-pruning) that **grades** a MetaWorld task (F1 vs native success). This phase builds the
seam that lets that spec **drive** a run, and de-risks it **offline before any RL**:

- the spec's quantitative semantics — the robustness `ρ` (a signed geometric margin) — is the per-step reward,
  and `sign ρ` is simultaneously the monitor verdict (one artifact, two uses);
- the reward bridge **reuses the existing `hymeko_rl/control/htl_reward.py::HtlRewardSpec`** adapter (no new
  logic engine) and **mirrors the `MonitorAlignedEnv` / `HymekoRewardMetaWorld` reward-override pattern** (§6.1);
- the de-risk asks the CLAUDE.md §3 question **before** launching RL: *does this reward rank native success
  above failure, and by how much?* A reward that does not separate the classes cannot drive learning.

**Result (the thesis, visible before training):** on the 200 real coffee-push rollouts, the **arbitrated** spec's
per-episode return separates success from failure at **AUC 0.973 / point-biserial +0.80**; the **raw**
over-constrained spec is **AUC 0.668 / +0.33** — its `min`-robustness is pinned to a constant offset by *dead*
`grasp_success`/`near_object` conjuncts (coffee-push is a push task, no grasp), so its learnable gradient is
≈ 0.3 % of the reward scale. Arbitration is what makes the spec *drivable*.

### Environment correction (measured, supersedes the handoff)

The handoff states "MetaWorld is not on the Mac → all real rollouts are a kato15 job." **This is now false.** On
the Mac `.venv`: `torch 2.12.0` (MPS available), `mujoco 3.10.0`, `metaworld 3.0.0` — coffee-push constructs,
steps, and reaches success. **The entire loop, including the Phase-2 RL drive, runs locally** within the 16 GB
cap; no kato15 handoff is required. (Verified: `test_spec_reward_env_live_metaworld_finite_and_certifies` runs
the live env.)

## Reward-quality de-risk (numbers)

Per-episode spec-return `Σ_t ρ(φ, s_t)`, scored against native success. `separation` = mean(success) −
mean(failure); `auc` = P(return_success > return_failure); `pb` = point-biserial corr(return, success).

| dataset | spec | AUC | pb | separation | mean ret (succ / fail) |
|---|---|---:|---:|---:|---|
| real coffee-push | **arbitrated** `F(obj_to_target ≤ 0.071)` | **0.973** | +0.799 | +12.54 | −4.80 / −17.34 |
| real coffee-push | arbitrated (weak-model gated) `F(in_place ≥ 0.6 ∧ ott ≤ 0.071)` | 0.993 | +0.852 | +46.03 | −34.09 / −80.12 |
| real coffee-push | **raw** (weak-model) `F(near≥.5 ∧ grasp≥.8 ∧ in_place≥.9 ∧ ott≤.1)` | **0.668** | +0.334 | +0.42 | −144.02 / −144.44 |
| synthetic (n=200) | target `F(in_place ≥ 0.9)` | 0.990 | +0.830 | +3.94 | −8.22 / −12.15 |
| synthetic | over-constrained raw | 0.952 | +0.764 | +2.78 | −9.77 / −12.56 |
| synthetic | distractor `F(grasp_success ≥ 0.5)` | 0.539 | +0.064 | +0.27 | +2.13 / +1.86 |

Figure: `reports/figures/2026_07_13_18_46_spec_reward_derisk/spec_reward_derisk.png` (AUC + point-biserial per
spec per dataset; chance line at 0.5 / 0.0).

**Honest reading.** (1) The raw spec is *not literally flat* — AUC 0.668 shows a weak ranking survives; the point
is the **effect size**: its separation is 0.42 on a −144 reward scale, so the drivable gradient is a rounding
error next to the dead-conjunct offset. (2) On *synthetic* data the "over-constrained raw" still scores 0.95,
because there the extra conjuncts are only mildly correlated distractors, not the *dead constants* they are on
coffee-push — this is why the discriminating case is the **real** task. (3) The arbitrated spec's reward reduces
to clean distance-shaping (`0.071 − obj_to_target`); its value is that it was **derived declaratively from a
success spec** and carries its own monitor verdict, not that it out-designs a hand-tuned dense reward. The
discriminating comparison is **arbitrated vs raw**, not arbitrated vs a strong hand baseline.

## Files touched

| file | LOC | note |
|---|---:|---|
| `hymeko_rl/eval/spec_bench/spec_reward.py` | +186 | NEW — `signals_from_metaworld_info`, `SpecRewardEnv`, `spec_reward_separation`, `SpecRewardQuality`, `_auc` |
| `hymeko_rl/experiments/exp_metaworld_spec_reward_ab.py` | +130 | NEW — `--offline` de-risk (this phase); `--rl` gated stub (Phase 2) |
| `hymeko_rl/tests/test_spec_reward.py` | +180 | NEW — 9 tests (unit + live-metaworld integration + thesis regression) |
| `docs/plans/2026-07-13-spec-reward-close-loop/{plan.tex,pdf,tikz,mmd}` | — | plan bundle (all four formats; `plan.pdf` built with tectonic) |

**No existing module was edited.** `HtlRewardSpec` is reused as-is (a MetaWorld signals fn is injected, matching
the `signals_from_planar(env)` convention). No `_v2`/duplicate files (§6.5 #13); no new function-per-axis (§6.5
#1); no globals (§6.5 #11); no §6.5 anti-patterns introduced.

## CORE.YAML items touched

**None.** `hymeko_core`, `parser`, and all pinned deps untouched. No new dependency (torch/mujoco/metaworld
already installed; `tectonic` is a doc-build tool, not a runtime dep).

## Test results

| layer | tests | result | wall |
|---|---:|---|---:|
| unit (extractor, `ρ`=margin, env reward=`ρ`, potential telescope, `_auc` extremes, episode-return) | 6 | pass | — |
| integration (live coffee-push MetaWorld env → finite dense reward, info passthrough) | 1 | pass | — |
| regression / thesis (arbitrated separation ≫ raw on real rollouts; synthetic target ≫ distractor) | 2 | pass | — |
| **total (`test_spec_reward.py`)** | **9** | **pass** | 0.97 s |
| neighbour regression (`test_htl.py`) | 18 | pass | — |

`ruff check`: clean on both new files. `mypy --strict`: **`spec_reward.py` and the experiment file report zero
errors** (the 13 errors mypy prints are all pre-existing baseline in transitively-imported files — `mujoco`/
`hymeko` missing stubs, and files already dirty on this branch that were not touched here). `radon cc -a -nc`:
no function at rank C or worse (all A/B) — §6.2 gate passes.

## Performance results

| metric | measured | budget (plan) |
|---|---:|---:|
| `ρ`/step, arbitrated (1 pred) | median 0.31 µs (IQR 0.01, worst 0.32; n=36 000) | < 200 µs |
| `ρ`/step, raw (4-pred AND) | median 1.00 µs (IQR 0.10, worst 1.11) | < 200 µs |
| offline de-risk wall (200 real + 200 synth × up to 6 specs) | 0.39 s | < 5 s |
| peak RSS (offline) | < 0.5 GB (numpy + pure-python HTL) | < 3 GB |

The load-bearing RSS/wall measurement is Phase 2 (SAC + replay buffer); Phase 1's reward-side overhead is
negligible.

## Experiment provenance

- Git SHA `01462a0` (working tree dirty — branch `hsikan-mlp-hybrid-audit` carries prior uncommitted work; the
  three files above are the additions of this task).
- Host: Apple-Silicon Mac (Darwin 25.5.0), `.venv` uv cpython-3.11, torch 2.12.0 (CPU/MPS), mujoco 3.10.0,
  metaworld 3.0.0, numpy 2.x, matplotlib 3.11.0.
- Seed 0. Real rollouts `coffee_push_rollouts.json` sha256 `1cd21fd95439a147…` (200 episodes, 142 success / 58
  failure, 180-step horizon, noise levels {0.0, 0.4, 0.8, 1.2}).

## Open issues / follow-ups (Phase 2 — the drive, gated)

The de-risk **passes**: the arbitrated spec's reward cleanly separates success from failure. Phase 2 confirms it
*drives* learning:

1. Wire `--rl`: bounded SAC A/B over reward-override arms **{native, spec_arbitrated, spec_raw,
   monitor_aligned}** on coffee-push, same env/budget, BC-warm-started (reuse Stage-B `_bc_base_policy`),
   evaluated on **native success**, oracle-certified per arm, live-`[sac]`-logged, multi-seed median/IQR,
   3-form output (JSON + plot + GIF). Runs locally (MetaWorld confirmed on the Mac).
2. One coupling to resolve first: the Stage-B BC anchor resolves its scripted policy via
   `GENERIC_TASKS[cfg.task]`, which has no `coffee-push` entry. Least-invasive fix: make the task→policy lookup
   injectable (additive param) rather than hard-coded — the coffee-push scripted policy already exists in
   `metaworld_rollouts` (`SawyerCoffeePushV3Policy`).
3. **Prediction (to be tested, not asserted):** `spec_arbitrated` reaches native/monitor-aligned success and
   trains stably; `spec_raw` trains poorly (flat gradient) and, being non-discriminating, only runs under an
   explicit `allow_uncertified` waiver — that failure *is* the thesis result.

**Decision gate:** Phase 2 is a real (if bounded) RL campaign. Awaiting go-ahead before launching it.
