---
title: Coin canonical task, solution lineage, and runtime divergence — bounded reconstruction
date: 2026-07-22
branch: exp/coin-full-action-bc-sac-td3
verdict: REWARD_RUNTIME_ONLY_DIVERGED
scope: verification only — no training, no fix, no new env, no merge/delete
---

# VERDICT: `REWARD_RUNTIME_ONLY_DIVERGED`

The Coin RL **runtime is not duplicated**. Every coin RL environment funnels through one canonical constructor
(`CoinDeliveryTrainEnv`, which wraps `make_coin_contact_env → ContactFormationEnv → make_coin_env → PlanarGraspEnv`);
`NeutralCoinDeliveryEnv` and `FullActionDeliveryEnv` are **subclass adapters** of it, not copies. The single
divergence is the **reward**: `CoinDeliveryTrainEnv.step` explicitly discards the env reward and emits the Python
`delivery_reward`, while the declared `galambos_task_deliver_v2b.hymeko` is loaded onto `env.reward_spec` but never
consumed by the training layer.

- **Exact divergence point:** `c0f62ab` "recovery: preserve uncommitted coin delivery snapshot" — the commit that
  first introduced the active Python `delivery_reward` in `hymeko_rl/train/coin_delivery_rl.py` and the
  `# env reward UNUSED` discard. This is **upstream of every listed canonical commit**; the full-action env
  (`0ca6853`) merely inherited it by subclassing.
- **Diverged component:** the reward only. `delivery_reward` (Python) = `DUPLICATE_DIVERGED` and active; the v2b
  `.hymeko` RewardSpec = `MISSING_RUNTIME_INTEGRATION` (loaded onto `inner.reward_file =
  galambos_task_deliver_v2b.hymeko` but discarded by the training wrapper).
- No source evidence contradicts the supplied ground-truth ledger.

---

## 0. Original task contract (verified intent)

TWO-ARM COIN DELIVERY: neutral or explicitly-declared contact-prepared start → approach → grasp/contact → transport →
enter target → **settle and remain for the strict dwell**. Learning pipeline: scripted expert → BC/DAgger →
BC-initialized SAC/TD3/PPO → reward-driven improvement. Deployment: the learned full-action policy executes; the
scripted expert is not silently active; residual (`scripted base + learned residual`) and standalone policies are
never conflated. Reward authority: `galambos_task_deliver_v2b.hymeko`, intended load-bearing (editing the `.hymeko`
should change the live training reward with no Python reward edit). Primary question: **can reward-driven SAC/TD3/PPO
improve a competent BC/DAgger policy?** — still **UNANSWERED** (see §7.8).

## 1. Runtime comparison table (direct source references)

| # | file · constructor | commit | init state | obs | action equation | online scripted? | reward impl | `.hymeko` loaded | reward hash at runtime | horizon | termination | strict impl | evaluator | checkpoint | valid claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `coin_delivery_rl.py` · `CoinDeliveryTrainEnv`/`make_delivery_rl_env` (L280) | reward from `c0f62ab`; contract in `ac6fd82`/physical-contact stages | contact-prepared (c1 bank via `make_coin_contact_env`) + scripted acquisition prefix (L196) | 41-d ACTOR_FIELDS | `clip(grasp_carry + δ·tanh(raw))` (L216 `residual_action`) | **YES** (base every step) | `delivery_reward` (L94,240); env reward **discarded** (L218) | v2b loaded on `inner.reward_spec` | not asserted | 120 | center∨safety (L249) | `delivery_certificate`/`raw_strict_oracle` | `eval_delivery` (native) | SAC/TD3 residual actors | RESIDUAL over scripted base only |
| 2 | `coin_neutral_start.py` · `NeutralCoinDeliveryEnv(CoinDeliveryTrainEnv)` (L24) / `neutral_env` (L114) | `5026152`,`c38676b` | **true neutral** (arm zeros, `PlanarGraspEnv.reset`, L65); `make_coin_env` (L125) | 41-d | inherited; composed chain drives learned E→handoff | no (learned E + learned handoff; scripted carry only in DEMOS) | inherited `delivery_reward` | v2b (via `make_coin_env`) | not asserted | inherited | inherited | same certificate | `eval_composed` (L261) | E_valselect_v2 + `handoff_best.pt` (8955e8db) | NEUTRAL_DELIVERY (learned chain) |
| 3 | `coin_delivery_e0_campaign.py` · `direct_e0_env` (L119) | `ac6fd82` era | contact-prepared bank | 41-d | inherited residual | YES (bank+base) | inherited `delivery_reward` | v2b | not asserted | inherited | inherited | certificate | `evaluate_policy` | transport BC/SAC | contact-prepared transport |
| 4 | `coin_two_arm_sac.py` · `direct_env` (wraps #1) + `evaluate` (L134) | pre-listed; horizon fixed `9cc0505` | contact-prepared | 41-d | inherited residual | YES | inherited `delivery_reward`; **`certify_or_abort` loads v2b RewardSpec and certifies it (L49) — but the env never emits it** | v2b (in the GATE only) | gate certifies v2b, not the runtime | was hard-coded 60 → `env.cfg.horizon` (`9cc0505`) | inherited | `policy_strict` | `evaluate` | SAC | residual SAC |
| 5 | physical-contact rerun · `coin_physical_contact_rerun.py` + corrected `planar_grasp_env`/`delivery_certificate`/`raw_strict_oracle` | `c7a39c9`,`0f20588`,`2c1e87c`,`2f981f6`,`defd822`,`bde81ba` | contact-prepared | 41-d | `clip(grasp_carry + δ·tanh(policy))` (residual) | **YES** | inherited `delivery_reward` | v2b loaded, discarded | not asserted | 120 | center∨safety | corrected certificate + raw oracle | matched-horizon `eval_delivery` | SAC/TD3 residual | RESIDUAL corrected-physics |
| 6 | full-action · `coin_full_action.py` · `FullActionDeliveryEnv(CoinDeliveryTrainEnv)` / `make_full_action_env` | `0ca6853` | contact-prepared START, **no prefix** (override) | 41-d | **`clip(policy(obs))`** — `_base()` never called | **NO** (base disabled; zero-action → strict 0) | inherited `delivery_reward` (**diverged from v2b + no held-dwell term**) | v2b loaded on env, **discarded** | **not asserted** | 160 | **safety only** (center-terminal removed so strict dwell can accumulate) | `delivery_certificate`+`raw_strict_oracle` | `eval_full_action` (native+strict+temporal) | standalone BC/SAC/TD3 | **UNVERIFIED** (reward mismatch) |

## 2. Git / runtime divergence answers (§5)

1. **First introduced active Python `delivery_reward`:** `c0f62ab` (recovery snapshot), predating all listed
   commits — `git log -S "def delivery_reward" --reverse` → `c0f62ab`.
2. **Full-action env created:** `0ca6853` (`FullActionDeliveryEnv` in `hymeko_rl/train/coin_full_action.py`).
3. **Reuse vs duplicate:** **REUSE.** `FullActionDeliveryEnv` *subclasses* `CoinDeliveryTrainEnv`, overriding only
   `reset` (drop prefix) and `step` (drop base, remove center-terminal). It shares the env, 6→4 action mapping,
   `_transition`, reward, obs, and certificate. It did **not** duplicate task logic.
4. **Which commit should have wired the HyMeKo reward but did not:** none in the full-action lineage — the miss is
   upstream in `coin_delivery_rl` (`c0f62ab`), which chose the Python `delivery_reward` and the `# env reward UNUSED`
   discard. The full-action env inherited it; the residual physical-contact stages inherited it; only the
   `certify_or_abort` gate (`coin_two_arm_sac`) created the *false impression* v2b was the runtime reward.
5. **Canonical `.hymeko` reward loader exists elsewhere:** **YES** — `env_factory.make_coin_env`
   (`env.reward_spec = RewardSpec.from_hymeko(DELIVER_V2B)`, L51-54) + `PlanarGraspEnv.reward_spec`
   (L526-527) + `RewardSpec.from_hymeko` (`hymeko_rl/env/reward.py`). It is used by the PlanarGraspEnv-native reward,
   but bypassed by `CoinDeliveryTrainEnv`.
6. **Did the redesign copy task semantics into Python:** for the reward, **yes** — `delivery_reward` re-implements a
   delivery reward in Python (progress + one-shot center) instead of scoring with the loaded RewardSpec, and it
   diverged from v2b (no contact-quality gate, no held-dwell). For env/action/rollout, **no** — those are shared.
7. **Duplicates vs adapters:** duplicate/diverged = `delivery_reward` (reward). Adapters = `NeutralCoinDeliveryEnv`,
   `FullActionDeliveryEnv`, `direct_env`, `direct_e0_env` (all legitimate subclass/wrapper adapters).

## 3. Duplication classification (§6)

| component | label | evidence |
|---|---|---|
| task specification | CANONICAL_SHARED | one `PlanarGraspEnv`/`make_coin_env` scene + `galambos_*.hymeko` |
| environment constructor | CANONICAL_SHARED | `CoinDeliveryTrainEnv` + subclass adapters; no rival constructor |
| **reward (Python `delivery_reward`)** | **DUPLICATE_DIVERGED (active)** | `coin_delivery_rl.py:94,240`; diverged from v2b (no graded terminal / body penalty / held-dwell) |
| **reward (v2b `.hymeko` RewardSpec)** | **MISSING_RUNTIME_INTEGRATION** | loaded (`inner.reward_file = …v2b.hymeko`) but discarded (`# env reward UNUSED`, L218) |
| reset / state bank | CANONICAL_SHARED (+ adapters) | c1 bank + `NeutralCoinDeliveryEnv` neutral reset override |
| action composition | LEGITIMATE_EXPERIMENT_ADAPTER | residual (`+base`) vs standalone (`clip(policy)`) are the two declared contracts |
| rollout | CANONICAL_SHARED | `coin_delivery_actor.rollout` |
| horizon | CANONICAL_SHARED (post-`9cc0505`) | `evaluate` hard-coded 60 was DUPLICATE_DIVERGED; fixed to `env.cfg.horizon` |
| strict certificate | CANONICAL_SHARED | `delivery_certificate` + independent `raw_strict_oracle` |
| evaluator | LEGITIMATE_EXPERIMENT_ADAPTER | `eval_delivery`/`eval_composed`/`eval_full_action` per contract |
| checkpoint loader | CANONICAL_SHARED | `build_sac`/`build_offpolicy` + `state_dict` |
| run manifest / reward certification | DEAD_METADATA (partly) | `certify_or_abort` certifies v2b that the runtime does not emit — a gate over a non-emitted reward |

## 4. Corrected result ledger (each result under its exact historical contract)

1. **`RELAY_HANDOFF_POSITIVE`** — `6292431`. transport-alone 4/24, fixed relay 9/24, 4 handoffs, 4/4 post-handoff
   strict completions. The handoff mechanism existed and worked.
2. **`NEUTRAL_DELIVERY_POSITIVE`** — `c38676b`. learned E-approach/grasp → learned handoff BC transport → strict
   delivery; 3/9 aggregate ring panel, 10/10 on winner states, zero-action 0, E+frozen-transport 0, no scripted carry
   in the rollout. `handoff_best.pt` (8955e8db). Scripted carry used only to generate handoff-matched demos.
3. **`POINT_ZERO_SHOT_POSITIVE`** — same learned chain on the POINT embodiment (one spherical fingertip/arm), no
   retraining, ≈4/9 aggregate with deterministic winners.
4. **`CORRECTED_PHYSICS_BRIDGE_POSITIVE`** — `0a46d5e`. fresh handoff BC trained from corrected-physics demos → 3/9
   neutral (grasp 5/9), `HANDOFF_CORRECTED_V1.pt`. The frozen filtered-physics policy did not transfer unchanged; the
   learned bridge **method** survived when retrained. The bridge did not depend on collision filtering.
5. **`RESIDUAL_SAC_NATIVE_TRANSPORT_POSITIVE`** — `ae0881f`,`bb861e2`. contact-prepared ring bank, contract
   `u_exec = scripted base + learned residual`: BC residual 2/9, SAC residual up to 9/9. Not a standalone result, not
   true-neutral.
6. **corrected-physics residual (Python reward)** — `bde81ba`. `u_exec = grasp_carry + residual`; no final-horizon
   improvement over the always-active base. Valid scope only: `RESIDUAL_OVER_SCRIPT_PYTHON_REWARD_RESULT`. Does not
   answer standalone BC→RL.
7. **competent full-action BC/DAgger** — `0ca6853`. `u_exec = policy(obs)`, base disabled; DAgger BC reached
   held-out strict 31/50 ≈ expert 33/50 (94%). `BC_FULL_ACTION_PHYSICAL_V1.pt` a6c1b84d9d66. VALID.
8. **`UNVERIFIED_FULL_ACTION_RL_RESULT_DUE_TO_REWARD_IDENTITY_MISMATCH`** — `90e323c` + audit `2d58516`. SAC/TD3
   finished below the BC on strict, but the live env optimized `delivery_reward` (reach + one-shot center, no
   held-dwell), not v2b, so the training optimum ≠ eval optimum. The result is invalidated for causal interpretation.

No later invalidated experiment overwrites an earlier valid result.

## 5. Proposed follow-up patch plan (NOT implemented here)

1. **Wire the loaded RewardSpec into `CoinDeliveryTrainEnv._transition`** (or add a config switch
   `reward_source ∈ {python_delivery, hymeko_spec}`) so the training reward is scored by `inner.reward_spec` (the
   loaded v2b), making the `.hymeko` load-bearing. Keep `delivery_reward` available as an explicitly-labelled
   legacy/ablation reward.
2. **Add a hard launch assertion** in every coin RL entrypoint: `sha256(active_reward_source) ==
   sha256(declared_reward_file)` (read from the live env instance, not the oracle) — abort otherwise.
3. **Add a held-dwell term** to whichever reward is authoritative, so the reward optimum coincides with the strict
   6-step dwell metric (or grade on the reward's own optimum — but that abandons held-delivery, so prefer the term).
4. **Correct `reward_oracle.py:72`'s claim** ("the SAME RewardSpec that scores the live env") — either make it true
   (per patch 1) or annotate that it certifies the *declared* spec, which the current runtime does not emit.
5. Only after 1-4: re-run diagnostics 4-5 (critic calibration on the corrected reward, first-update microscope),
   then a matched campaign to finally answer the primary question.

## 6. Files to preserve / later deprecate

**Preserve (valid under their contracts):** `coin_neutral_start.py` (+ `handoff_best.pt` 8955e8db), the relay/bridge
artifacts (`6292431`,`c38676b`), `HANDOFF_CORRECTED_V1.pt` (`0a46d5e`), the physical-contact corrected stages
(`c7a39c9`…`bde81ba`), `BC_FULL_ACTION_PHYSICAL_V1.pt` (competent BC), and all quarantined commits
(`0ca6853`,`90e323c`,`0a46d5e`,`2d58516`) unamended.

**Later convert to explicit adapters / deprecate (do NOT do now):** `delivery_reward` → demote to a labelled legacy
reward once the RewardSpec is wired (patch 1); `certify_or_abort` → make it certify the *actually-emitted* reward;
`evaluate` other `max_steps=60` siblings (`coin_bridge_*`, `coin_nstep_exp`, `coin_generator_exp`) → align to
`env.cfg.horizon` when revisited. No env constructor needs deprecation (single canonical constructor + adapters).

## Provenance

Verification only — no training run, no reward edit, no env created, no merge/delete. Evidence: `git log -S`,
`git log -1 <commit>` for all 18 listed commits (all present locally; `2d58516` is local-only), source lines cited
inline, and one live-env introspection (`inner.reward_file = data/robotics/galambos_task_deliver_v2b.hymeko`;
`CoinDeliveryTrainEnv.step` discards it).
