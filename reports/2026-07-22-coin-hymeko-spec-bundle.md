---
title: Coin HyMeKo specification-bundle audit — which specs are load-bearing?
date: 2026-07-22
branch: exp/coin-full-action-bc-sac-td3
verdict: MULTIPLE_HYMEKO_SPECS_IGNORED
scope: bounded closure from the canonical Coin entry points; sentinel-proven; no code change, no committed sentinel
---

# VERDICT: `MULTIPLE_HYMEKO_SPECS_IGNORED`

The canonical Coin RL runtime `make_coin_env → PlanarGraspEnv(robot=None, env=DEFAULT_ENV) → make_coin_contact_env →
ContactFormationEnv → CoinDeliveryTrainEnv` is **Python-authoritative** for geometry, scene, and reward. The
`.hymeko` bundle is loaded (v2b onto `env.reward_spec`) and is load-bearing at the `PlanarGraspEnv` reward layer, but
the Coin entry point bypasses it: robot via `make_planar_arms_mjcf` (`robot=None`), scene via `DEFAULT_ENV = EnvSpec()`
(Python dataclass defaults), reward via `delivery_reward` (`CoinDeliveryTrainEnv` discards `env.reward_spec`). So it
is **not** a reward-only divergence — three spec categories are bypassed.

## 1. Bounded dependency closure (8 files) + hashes

Root entry points: `env_factory.make_coin_env`, `make_coin_contact_env`, `PlanarGraspEnv` (constructor),
`delivery_certificate` (strict certificate), `coin_neutral_start.neutral_env`, `coin_delivery_e0_campaign.direct_e0_env`.

| file | sha256[:16] | purpose | referenced by | includes |
|---|---|---|---|---|
| `galambos_planar.hymeko` | `8027224d550d6af1` | robot (2×2-link planar arm) | `planar_grasp_env._PLANAR_ARM`, `kinematic_variant` | `meta_kinematics` |
| `galambos_env.hymeko` | `8be8246bb2af1a3c` | scene (zone/spawn/bounds/success/disk) | `planar_grasp_env._PLANAR_ENV` | `meta_env` |
| `galambos_task.hymeko` | `e6ea57b37e228eaf` | default PlanarGraspEnv reward | `planar_grasp_env._PLANAR_TASK` | `meta_kinematics`,`meta_reward` |
| `galambos_task_deliver_v2b.hymeko` | `98cd3ad6af02a1bf` | v2b contact-quality-gated deliver reward | `env_factory.DELIVER_V2B` | `meta_kinematics`,`meta_reward` |
| `galambos_task_coord.hymeko` | `45708dcac7a41297` | coordination reward | `env_factory.COORD_HYMEKO` | `meta_kinematics`,`meta_reward` |
| `meta_kinematics.hymeko` | `43c48781c06d2d22` | kinematics term library | planar/task/v2b/coord | — |
| `meta_reward.hymeko` | `96de12009d2cbedc` | reward term library | task/v2b/coord | — |
| `meta_env.hymeko` | `32e2b08dc547ba27` | env term library | galambos_env | — |

Dependency graph:
```
galambos_planar ──▶ meta_kinematics
galambos_env    ──▶ meta_env
galambos_task   ──▶ meta_kinematics, meta_reward
galambos_task_deliver_v2b ──▶ meta_kinematics, meta_reward
galambos_task_coord       ──▶ meta_kinematics, meta_reward
```
Combined bundle hash (sha256 of the sorted per-file hashes): recorded in
`experiments/2026_07_23_coin_hymeko_recovery/` during recovery (§10).

## 2. Runtime consumer of each spec (traced, not assumed)

| file | parser | runtime object | Coin-runtime consumer | changes at runtime? |
|---|---|---|---|---|
| galambos_planar | `emit_arm_mjcf` (only via `robot!=None`) | MJCF arm | **NONE** — `make_coin_env` passes `robot=None` → `make_planar_arms_mjcf()` (Python) | no |
| galambos_env | `EnvSpec.from_hymeko` (only via `PlanarGraspEnv.from_hymeko`) | `EnvSpec` | **NONE** — `PlanarGraspEnv.__init__` uses `env=DEFAULT_ENV = EnvSpec()` (Python) | no |
| galambos_task | `RewardSpec.from_hymeko` | `RewardSpec` | PlanarGraspEnv default reward, **overridden by v2b** then **discarded by CoinDeliveryTrainEnv** | no (RL) |
| galambos_task_deliver_v2b | `RewardSpec.from_hymeko` | `RewardSpec` on `env.reward_spec` | PlanarGraspEnv reward (LOAD-BEARING there), **discarded by CoinDeliveryTrainEnv.step (L218)** | **PlanarGraspEnv: YES; RL: NO** |
| galambos_task_coord | `RewardSpec.from_hymeko` | `RewardSpec` | only if `coord=True` (default False) | no |
| meta_kinematics/meta_reward | include | term tables | feed the compiled RewardSpec (load-bearing at PlanarGraspEnv, discarded at RL) | PlanarGraspEnv only |
| meta_env | include | term tables | feed `EnvSpec.from_hymeko` (unused; DEFAULT_ENV is Python) | no |

## 3. Sentinel results (definitive — change the spec, rebuild via the entry point, fresh process)

| sentinel | RL runtime observable | PlanarGraspEnv observable | conclusion |
|---|---|---|---|
| `galambos_env` disk `radius 0.02→0.05` | `disk_r` = 0.0200 (unchanged) | 0.0200 (unchanged) | scene comes from Python `DEFAULT_ENV = EnvSpec()`, NOT the `.hymeko` |
| `galambos_env` zone `half 0.04→0.09` | `zone_half` = 0.0400 (unchanged) | unchanged | same — scene ignored |
| `v2b` `zoneprog 10→200` (per-step) | `rl_reward3` = −1.0396 (unchanged) | `planar_native_reward3` = −4.03→**−3.72** (changed) | v2b LOAD-BEARING at PlanarGraspEnv, IGNORED by the RL wrapper |
| `v2b` `terminalgraded 30→999` (terminal) | unchanged | unchanged at step 0 (terminal-only term) | (weaker sentinel; superseded by the per-step one above) |
| `galambos_planar` `robot=None` | N/A (Python arm) | N/A | robot geometry is Python `make_planar_arms_mjcf` (deliberate: emitter cannot express connected planar rods, `planar_grasp_env.py:9`) |

Sentinels were applied in place and **reverted from git** (working tree clean; none committed).

## 4. Split-authority table (declared HyMeKo vs active Python)

| task aspect | declared HyMeKo | active Python authority | classification |
|---|---|---|---|
| reset / init-state | (scene spawn in galambos_env) | c1 bank + Python sampler (corrected) | PYTHON_DUPLICATE_DIVERGED |
| observation | none | `ACTOR_FIELDS` (Python) | (Python-only; no HyMeKo obs spec) |
| action dims/scaling | none | 6-DoF cooperative, `[-1,1]` (Python) | (Python-only) |
| full-action vs residual | none | `CoinDeliveryTrainEnv` (residual) vs `FullActionDeliveryEnv` (Python) | LEGITIMATE_EXPERIMENT_ADAPTER |
| scripted expert | none | `p_grasp_carry` (Python) | (Python-only) |
| collision/contact legality | (not in galambos_planar path) | `make_planar_arms_mjcf` ARM_LEGALITY (Python) | PYTHON_DUPLICATE_DIVERGED (robot spec bypassed) |
| handoff readiness | none | Python monitor | (Python-only) |
| **reward** | **v2b (load-bearing at PlanarGraspEnv)** | **`delivery_reward` (RL)** | **HYMEKO_LOADED_BUT_IGNORED + PYTHON_DUPLICATE_DIVERGED** |
| termination | (success in galambos_env) | Python (`center∨safety` / `safety`) | HYMEKO_RUNTIME_MISSING |
| zone containment | galambos_env `zone half` | `DEFAULT_ENV.zone_half` (Python) | HYMEKO_RUNTIME_MISSING |
| velocity/settling | none | Python certificate | (Python-only) |
| dwell | none | `delivery_certificate` 6-step (Python) | (Python-only) |
| strict delivery | none | `delivery_certificate` + `raw_strict_oracle` (Python) | (Python-only) |
| evaluation horizon | none | `env.cfg.horizon` (Python; fixed 60→horizon in `9cc0505`) | PYTHON_ADAPTER_EQUIVALENT |

## 5. Bundle gate `HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS`: **FAIL**

Fails on: (a) v2b `.hymeko` is loaded but its runtime effect is discarded by `CoinDeliveryTrainEnv`; (b) the Python
`delivery_reward` silently overrides it (and diverged — no held-dwell, no contact-quality gate); (c) `galambos_env`
scene + `galambos_planar` robot are not consumed by the Coin entry point (Python `DEFAULT_ENV` / `make_planar_arms_mjcf`
are authoritative); (d) an undeclared fallback exists (`robot=None`, `env=DEFAULT_ENV`). A reward-only repair would
**not** make the gate pass.

## 6. Minimal repair list (for §10, NOT implemented here)

1. **Reward (the load-bearing defect):** wire `env.reward_spec` (the loaded v2b RewardSpec) into
   `CoinDeliveryTrainEnv._transition`, or add `DeliveryRLConfig.reward_source ∈ {hymeko_spec, python_delivery}` with
   `hymeko_spec` the default. Keep `delivery_reward` as an explicit **legacy/ablation** mode (no implicit fallback).
   Add a held-dwell term to v2b (or verify v2b already grades held delivery via `terminal_deliver_graded`) so the
   reward optimum coincides with the strict 6-step dwell. Add a launch assertion `sha256(active_reward) ==
   sha256(declared)` read from the live env.
2. **Scene:** either build `DEFAULT_ENV` via `EnvSpec.from_hymeko(galambos_env)` (make it load-bearing) OR mark
   `galambos_env.hymeko` explicitly as `documentation-only` and stop citing it as executable provenance. Currently
   the Python `EnvSpec()` values equal the `.hymeko` values, so this is `DUPLICATE_EQUIVALENT` — choose one authority.
3. **Robot:** `galambos_planar.hymeko` is superseded by design (`make_planar_arms_mjcf`, documented). Classify it
   `documentation-only`/`legacy` for the Coin path — do not claim it is the runtime geometry.
4. Correct `reward_oracle.py:72` ("the SAME RewardSpec that scores the live env") — false for `CoinDeliveryTrainEnv`.

## 7. Executable-spec status summary

| spec | status |
|---|---|
| galambos_task_deliver_v2b | `HYMEKO_LOADED_BUT_IGNORED` (RL) / load-bearing (PlanarGraspEnv) |
| delivery_reward (Python) | `DUPLICATE_DIVERGED` and active |
| galambos_env | `MISSING_RUNTIME_INTEGRATION` (DEFAULT_ENV Python; DUPLICATE_EQUIVALENT values) |
| galambos_planar | `DOCUMENTATION_ONLY` for Coin (superseded by `make_planar_arms_mjcf`, by design) |
| galambos_task | `LOADED_BUT_IGNORED` (default reward, overridden + discarded) |
| galambos_task_coord | `DEAD_METADATA` for delivery (coord off) |
| meta_kinematics / meta_reward | load-bearing for the compiled RewardSpec (PlanarGraspEnv), discarded at RL |
| meta_env | `DEAD_METADATA` for Coin (feeds unused `from_hymeko` scene path) |

## Provenance

Verification only — no code change, no training. Sentinels applied in place and reverted from git (working tree
clean). Evidence: source lines cited inline (`env_factory.py:9` `robot=None`; `planar_grasp_env.py:481-482`,`526-527`;
`env_spec.py:93` `DEFAULT_ENV = EnvSpec()`; `coin_delivery_rl.py:218` `# env reward UNUSED`,`:240` delivery_reward),
plus fresh-process sentinel runs. HEAD `a7b5035`, branch `exp/coin-full-action-bc-sac-td3`.
