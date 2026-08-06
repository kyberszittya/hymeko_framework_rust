---
title: Coin physical-contact SAC-vs-TD3 rerun — corrected collision model, matched-horizon verdict
date: 2026-07-22
branch: exp/coin-physical-contact-sac-td3-rerun
verdict: SAC NO_EFFECT / TD3 NO_EFFECT (matched 120-step horizon)
---

# Coin physical-contact SAC-vs-TD3 rerun (2026-07-22)

## Summary

Reran the Coin delivery RL comparison under a **corrected physical-contact model** — coin↔arm-link collision
*enabled* (`ARM_LEGALITY` 1/3), collision-filtered pass-through *removed*, deep initial interpenetration forbidden
(0.5 mm tolerance) — with SAC and TD3 improving the **same BC-initialized residual policy**
`u_exec = clip(grasp_carry + delta·tanh(policy))`. Five stages of contract-corrected foundation were built, gated,
and committed before any training. The 5-seed × {SAC, TD3} × 100k campaign ran locally.

**Headline (matched 120-step horizon, env-native + strict certificate):**

| | scripted base | SAC (median/5) | TD3 (median/5) |
|---|---|---|---|
| VAL native (14) | 0.714 | 0.643 | 0.643 |
| VAL strict | 8 | 8 | 7 |
| panel native (9) | 9/9 | 9/9 | 9/9 |
| panel strict | 6/9 | 6/9 | 6/9 |

**Verdict: SAC `NO_EFFECT`, TD3 `NO_EFFECT`.** Under corrected physics, neither residual RL exceeds the scripted
grasp_carry base at the matched horizon (VAL native 0.71→0.64 is a single-seed flip out of 14; strict equal/±1;
panel identical). Local policy-improvement caps at the supervised ceiling — the standing finding, now confirmed
with the collision model corrected.

## The methodological catch — a horizon-truncation artifact (§3)

The in-training `evaluate()` rolls `max_steps=60`, but the training/delivery horizon is **120**. The scripted
grasp_carry base needs ~120 steps to deliver, so at 60 steps it scores low (VAL native 0.286) and the RL policies —
which learned to deliver *faster* within 60 steps — looked strongly positive (SAC 0.50, TD3 0.571 →
`PHYSICS_FIXED_POSITIVE`). **That verdict was an artifact.** At the matched 120-step horizon the scripted base
already delivers (VAL 0.714, panel 9/9) and the residual adds nothing. Caught by re-evaluating every checkpoint at
the training horizon with the same env-native metric used for the ceiling. See `horizon_artifact.png`.

This is the §3 "horizon-match every probe to the production env" rule doing exactly what it is for: a probe on a
truncated horizon measures a different task.

## Stages (branch `exp/coin-physical-contact-sac-td3-rerun`)

| stage | commit | content |
|---|---|---|
| 1 | c7a39c9 | coin↔arm-link collision (ARM_LEGALITY 1/3) + 8 `PHYSICAL_CONTACT_CONTRACT` tests |
| 2 | 0f20588 | strict monitor: arm-link contact legal, `body_shove` redefined, raw-MuJoCo oracle + 4 tests |
| 3 | 2c1e87c | point-to-capsule-segment sampler + 5 state banks with full provenance |
| 4 | 2f981f6 | scripted-base ceiling + BC_PHYSICAL_CONTACT_V1 (identical SAC/TD3 init, max Δ=0) |
| 5 | defd822 | SAC/TD3-from-BC driver + evaluator + import-cycle fix + 3 unit tests |
| 6 | (this) | campaign results + matched-horizon eval + corrected verdict + plot + videos + report |

### §1-3 Physics + monitor
Applied the codebase's existing `ARM_LEGALITY` (1/3) channel to the structural arm geoms so the coin physically
collides with every arm link (verified: coin↔arm-link mask = 2 on all links; forced-overlap detected; RING≡POINT).
Redefined `clean`: arm-link contact is legal (informational), only an excessive body-only-driven sweep with no
grasp is unclean. Added an independent raw-MuJoCo strict oracle (`raw_strict_oracle.py`) that agrees with the
production certifier on a rollout (dz < 1e-3, v < 0.02, verdict equal) — the `STRICT_MONITOR_CONTRACT` gate.

### §4 Sampler + banks
The old centroid clearance was geometrically wrong for capsule links (a coin near a capsule *end* clears the
centroid but penetrates — the seed-1011 −13.1 mm defect). Replaced with exact **point-to-capsule-segment** distance
(`_rest_arm_segs`, clear = disk_radius + 3 mm). **200/200 seeds now arm-clear** (worst +3.3 mm). Regenerated 5 banks
with per-state provenance (seed, qpos/qvel, coin/arm pose, min coin↔arm distance, raw contacts, hashes): N0/N1
neutral 59/59, D1 E-handoff ring 59/59, **D2 E-handoff point 50/59** (honest finding: the frozen E-approach squeezes
the coin ≤1.68 mm into an arm capsule at grasp on 9/59 POINT states under corrected physics — recorded, flagged, not
kept). Deterministic seed replacement recorded (1011: −0.159,0.122 → −0.032,0.227, +77.6 mm).

### §4 (reassessment) Ceiling
Before committing compute: the scripted base under corrected physics is **not broken** — native panel 9/9
(grasp_no_delivery 0), held-out 0.70, strict panel 6/9. Pipeline well-posed → GO. All stage-0 gates pass (g1
zero-residual == scripted, reward oracle certified). BC_PHYSICAL_CONTACT_V1 (zero-residual head) reproduces 9/9
native; SAC (`mu`) and TD3 (`head`) init identically (max|SAC−TD3 Δ| = 0.00e+00 on a 256-sample probe).

### §5 Driver + import-cycle fix
Thin composition root reusing `coin_two_arm_sac` (env/demos/replay/eval/splits) + `train.sac`/`train.ddpg`; no new
trainer. Fixed a **pre-existing import cycle** (`coin_delivery_actor ↔ coin_grip_control`) that made the whole
trainer unimportable — deferred `coin_grip_control.normal_contact_forces` into `rollout()` (its only call site).

## §9 Campaign

5 seeds × {SAC, TD3} × 100k steps, local Mac (no katolab, per user), 4-wide batches. Production smoke (100k SAC):
clean, **227 s wall, peak RSS 0.36 GB**, no NaN. Full campaign: 10 runs, all completed, all under RSS cap.

Checkpoint selection was on the **native** metric only (§8); strict recorded separately. Training-time (60-step)
best: SAC native 0.50/strict 0 (5/5); TD3 native 0.571/strict {0,1,1,0,0}. These are the truncated-horizon numbers
— see the catch above.

## §12 Neutral bridge re-eval

The frozen neutral bridge (E-approach → handoff → transport; historically 3/9 composed on the panel, with
1045/1278/1447 delivering 10/10 under *filtered* physics) **degrades under corrected physics: 1/9 delivered, grasp
5/9.** The bridge was partly relying on the collision-filtered pass-through the correction removed — consistent with
the D2 bank finding. An honest regression of a frozen deploy artifact when the physics is made correct.

## §13 Videos + plots

- `horizon_artifact.png` — VAL native at 60-step (artifact, RL looks positive) vs 120-step (truth, NO_EFFECT).
- `videos/{scripted_base,sac_s0,td3_s0}_seed1011.gif` + `videos/compare_seed1011.gif` (1920×480) — all three deliver
  identically at the matched horizon; labels derived from the live trace (dz, delivered flag), not hard-coded.

## Files touched

- `hymeko_rl/env/planar_grasp_env.py` (§1 collision + §4 sampler, +25/−5)
- `hymeko_rl/coin_delivery/delivery_certificate.py` (§3 clean redefinition)
- `hymeko_rl/coin_delivery/raw_strict_oracle.py` (new, §3)
- `hymeko_rl/coin_delivery/physical_bank_generator.py` (new, §4)
- `hymeko_rl/experiments/coin_physical_contact_rerun.py` (new driver)
- `hymeko_rl/experiments/coin_physical_contact_eval.py` (new evaluator)
- `hymeko_rl/train/coin_delivery_actor.py` (import-cycle fix, +4/−1)
- `hymeko_rl/tests/{test_physical_contact_contract,test_strict_monitor_contract,test_physical_contact_rerun}.py`

## Tests / gates

- `PHYSICAL_CONTACT_CONTRACT` 8/8, `STRICT_MONITOR_CONTRACT` 4/4, rerun-driver units 3/3 — all pass; ruff clean.
- Stage-0 gates g1–g7 pass under corrected physics; reward oracle certified.

## CORE.YAML

None touched (CORE protects Rust crates + spec + rtl; all work is in `hymeko_rl` Python).

## Provenance

- Host: Apple Silicon Mac, CPU (MuJoCo CPU-bound). torch + mujoco per `.venv`. OMP threads 2 (campaign) / 4 (eval).
- Seeds: train 64000-64055, val 64100-64113, demo 64200-64203 (disjoint); RL seeds 0-4; eval panel 1011.., held-out
  1000-1074.
- BC_PHYSICAL_CONTACT_V1.pt sha256[:12] = c03ad6175d31.
- RL runs are stochastic (BLAS multi-threaded) — verdict rests on 5-seed medians (§3 carve-out), not bit-repro.

## Open issues / follow-up

- `evaluate()`'s hard-coded `max_steps=60` is a latent horizon bug in the shared harness — it half-truncates the
  120-step task. It drove the checkpoint selection here (native-at-60). A follow-up should make the eval horizon
  track the env horizon so selection and reporting agree. (The verdict above uses the corrected matched horizon.)
- The frozen neutral bridge needs re-derivation under corrected physics if it is to be a deploy artifact (1/9).
- No residual RL improvement over the scripted base was found under corrected physics — as under filtered physics,
  the productive direction is imitation/search (rollout-DAgger, exact-rollout search), not local off-policy RL.

## Scope correction (appended 2026-07-22, after the standalone full-action experiment)

This campaign tested **residual policies on top of an always-active scripted base** (`u_exec = clip(grasp_carry +
delta·tanh(policy))`). It is valid only as `RESIDUAL_OVER_SCRIPT_NO_FINAL_SUCCESS_EFFECT`. It did **not** test whether
a standalone BC clone of the scripted expert can be improved by RL. The follow-up experiment
(`exp/coin-full-action-bc-sac-td3`, report `2026-07-22-coin-full-action-bc-sac-td3.md`) ran that: scripted expert →
full-action BC clone (script disabled) → standalone SAC/TD3. Result: standalone SAC and TD3 **regress** the BC on
every seed (BC strict 34/59 → SAC 17, TD3 24). Therefore:

- The broad claim *"local off-policy RL caps at the supervised ceiling"* is **withdrawn as stated**. The precise
  findings are: **residual-over-base = NO_EFFECT** (the base holds it at the ceiling); **standalone-from-BC =
  REGRESSION** (RL falls below the BC ceiling). The residual/standalone distinction is load-bearing.
- A secondary correction: the 60-vs-120 discrepancy above conflated two harnesses (`evaluate` loose/momentary vs
  `eval_delivery` env-native) as well as the horizon; the horizon fix (commit `9cc0505`) and a single consistent
  metric are used in the standalone experiment.
