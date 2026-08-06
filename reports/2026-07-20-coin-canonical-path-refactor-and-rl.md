---
campaign: COIN-DELIVERY canonical-path refactor (Phase A) + two-arm BC->SAC RL (Phase B)
title: Canonical-path repair of Coin Delivery, then learned two-arm delivery on the clean path
date: 2026-07-20
baseline_commit: 1c704f72977ede79e828d6f29852617a90edda22
repair_branch: repair/coin-delivery-canonical-path
verdict: PARTIAL — learned repeatable bilateral zone entry; certified strict delivery reproducible on a minority of states (2/18), not generalized. Blocker = CONTACT_STRATEGY (fingertip attribution + mechanism), NOT stabilization.
---

# Canonical-path refactor + learned two-arm coin delivery

**Created-at:** 2026-07-20 19:30 JST.

Two phases. **Phase A** made the Coin Delivery code obey the existing CLAUDE.md architecture (four bounded structural
defects, one commit each, behaviour preserved). **Phase B** fixed the concrete `FrozenInstanceError`, integrated the
frozen scientific findings into the shared corrected-SAC path, ran the BC->SAC smoke through the canonical rollout, and
launched the monitored 100k run.

## The behavioural invariance oracle (used after every step)

`artifacts/coin_recovery_baseline/golden_behavior_before.json` — the full `DeliveryResult` fingerprint of all 6 scripted
actors × 8 fixed seeds (48 records) through the real `rollout_delivery`, captured at `1c704f7`. **sha `1fe468b770594fc2`,
strict=1, loose=5.** After every one of the four defect commits AND the Phase-B trace extension, re-capturing this
fingerprint gave **0 mismatches / identical sha** — i.e. the four canonical actuators, delivery-v2b-aligned reward, strict
predicate, explicit-state restore, A1/A4 behaviour, and BC/SAC math are bit-preserved. This, not "tests pass", is the
proof of preserved behaviour.

## Phase A — four defects, four commits

| # | commit | defect | what changed | proof |
|---|--------|--------|--------------|-------|
| 1 | `0d39b3f` | duplicated rollout loops | ONE canonical `rollout()` + public `RolloutTrace`/`RolloutStep` in `coin_delivery_actor`; `PlanarGraspEnv.planar_metrics`/`.arm_body_steps` public accessors; `rollout_delivery` (scripted eval) + `roll_delivery` (RL/SAC eval + replay) both funnel through it | golden bit-identical |
| 2 | `7d5656f` | experiment->library inversion | `_dir_to_zone` -> `PlanarGraspEnv.direction_to_zone()` + pure `coin_zone_direction()`; `p_grasp_carry` -> `train.coin_delivery_rl`; 7 `train/coin_*` callers de-inverted | golden bit-identical |
| 3 | `ac616bf` | magic positional access | `obs[[20,21]]` -> `field_indices('mid_to_coin_x','mid_to_coin_y')` (new named accessor on the `team_tensor` schema); `np.insert(...,[2,4])` -> `_pad_insert_indices()` derived from compiled `pad_hinge_{side}` addresses; both fail loudly on a bad layout | derived indices == old constants ([20,21], [2,4]); golden bit-identical |
| 4 | `ca9c305` | bypassed parallel layer | `K1DistalOrientation.build_env()` now RAISES (was silently returning K0); dead shadowed `provenance.py` deleted, `git_commit` moved into the package (restores the runners' import) | golden bit-identical; new K1-raises regression test |

**Remaining manual `env.step` loops** (grasp_cert / compliant_pad / grip_control / transport) are contact-mechanics
**diagnostic** probes, not scripted/BC/SAC/replay delivery-eval — justified as diagnostic-internal, not experiment-level
rollout duplication.

**One documented, bounded exception (defect 2):** the only remaining coin production->experiments import is
`pedc_selection._env`, the shared eval-corpus **fixture** (arcrl c1 bank + contract). Fully relocating it would drag the
whole arcrl corpus ecosystem into the library — out of the four-defect scope; classified as a fixture, not a reusable
helper.

### Golden before/after (deterministic, exact-equality)

| field (per actor×seed) | before | after (all 4 commits) |
|---|---|---|
| strict_delivery / loose_in_zone / initial_success | fingerprint | identical |
| progress / min_dtz / dwell / settle_vel | fingerprint | identical |
| attribution (L/R/body/free) / fingertip_fraction / body_shove / mechanism | fingerprint | identical |
| **sha256 of the 48-record fingerprint** | `1fe468b77...` | **`1fe468b77...`** |

## Phase B — FrozenInstanceError fix + corrected-SAC scientific findings (`e8d0b734`)

- **FrozenInstanceError** (the driver mutated a frozen `SACConfig.log_every`): fixed by passing `log_every`/`eval_every`
  through `SACConfig.stable(...)` at **construction** (`frozen=True` kept, NOT dropped). Regression test
  `test_sac_competence_gate.py`.
- **§1 competence-gated BC anchor:** `train_sac` gained an optional `bc_coef_fn(step)` hook; the TD3+BC anchor uses it
  instead of the constant `cfg.bc_coef` (milestones 1.0->0.3->0.1->0.05, NOT step-decay). Backward compatible; regression
  test asserts the hook is honoured.
- **§2 phase-stratified demo seed:** A1/A4 demo transitions labelled APPROACH/CONTACT/TARGET_PROGRESS/HOLD, rare phases
  oversampled when seeding the replay.
- **§4 relational observation:** the canonical 41-d `ACTOR_FIELDS` obs (coin->target dir/dist, coin velocity, L/R contacts,
  phase flags) — reused, not duplicated. "signed progress" / "contact attribution" are derivable, not added as new fields
  (adding to the shared schema would change the obs contract; deferred with justification).
- **§5 task reward:** `galambos_task_deliver_v2b.hymeko` **certified delivers=True** (optimal_return 25.40) as the §3
  anti-farming gate; the env trains on its delivery-aligned potential-based reward.
- **§6 checkpoint ranking:** best by (strict count, zone rate, mean progress, two-arm participation).
- **Canonical rollout carries the transition:** `RolloutStep.obs` + `RolloutTrace.final_obs` so demo collection AND
  deterministic eval run through the ONE `rollout()` (defect 1) — no bespoke loop. Golden still bit-identical.

### BC->SAC 5k smoke — every §7 checklist item

| check | result |
|---|---|
| active reward = v2b | delivers=True, optimal_return 25.40 |
| explicit disjoint splits | TRAIN 56 / VAL 14 / DEMO 4 (disjoint seed pools) |
| demo dataset nonempty | 616 transitions (APPROACH 257, CONTACT 136, HOLD 98, PROGRESS 125) |
| demo transitions enter replay | 1600 phase-stratified seeded |
| SAC actor loads BC anchor + steps advance | step counter 500->5000 |
| actor/critic losses finite | finite from step 1500 (steps <=1000 are the pre-update `nan` placeholder — `start_steps=1000`, not divergence) |
| actions nonconstant, both L/R channels | eval L/R contact 0.50/0.37; attribution L/R 0.11/0.25 |
| eval uses the canonical rollout | yes (`rollout()` + policy strict predicate) |
| bc_coef logged each eval | yes (1.0 -> 0.1 after first strict) |

## The 100k monitored run — COMPLETED

- **Command:** `PYTHONUNBUFFERED=1 python -m hymeko_rl.experiments.coin_two_arm_sac --steps 100000 --eval-every 5000 --seed 0`
- **Provenance:** `experiments/2026_07_20_coin_two_arm_sac_100k/` — `train.log`, `train.pid` (PID 19268),
  `launch_manifest.json` (commit 6), `sac_actor_best.pt` + `sac_actor_final.pt` + `run.json` + `eval_curve.png`.
- **Device** CPU; ~416-437 steps/s; **100k steps in ~4 min**; peak RSS well under the 16 GB cap. Losses finite
  throughout (post the step<=1000 pre-update placeholder); no divergence.
- **20 evals through the canonical rollout** (`eval_curve.png`, numerical `run.json`): loose zone-entry stabilised at
  **0.43** with **both arms contacting every eval** (L 0.16-0.43, R 0.10-0.25); mean target progress climbed 0.007 ->
  0.014 m; **strict delivery 0** across all 20 (one fluke strict at eval#1, then 0). Best checkpoint = eval#1 by the
  strict-ranked selector.

**Corrected read (post-run strict-gap diagnostic, `strict_gap_results.json`):** the earlier "strict=0 / contact-mechanics
wall" headline was **over-pessimistic and is retracted**. The correct statement: *the current BC-anchored one-step SAC
configuration plateaued at loose competence and did not achieve **repeatable across the distribution** certified delivery
within 100,000 environment steps.* A mechanical limitation is at most a hypothesis — and the diagnostic argues against it,
because certified delivery **was** achieved (below). The clean canonical framework trained the learned two-arm policy
end-to-end (the task's completion criterion) and produced genuine certified deliveries on a subset of states.

## Post-run strict-delivery gap diagnostic (read-only; predicate/actor/env unchanged)

### §2 eval#1 strict event = REPRODUCIBLE_STRICT
The eval#1 strict was VAL state **64102** under `sac_actor_best.pt`. Re-running that exact checkpoint+state **10×**
through the canonical `rollout()` gave **10/10 strict**, single state hash → **REPRODUCIBLE_STRICT**. It is a genuine,
deterministic certified delivery, not a fluke / monitor-boundary / logging artifact. Preserved as a video.

### §3 checkpoint sweep (deduped to the 2 saved checkpoints) on DEMO(4)+VAL(14)=18 states

| checkpoint | loose-success | strict | zone rate | pass-rates GIVEN zone entry (dwell / settle / attribution / body / mechanism) |
|---|---|---|---|---|
| `sac_actor_best.pt` (eval#1) | 4 | **2** | 0.39 | 1.00 / 1.00 / **0.50** / 1.00 / **0.50** |
| `sac_actor_final.pt` (100k) | 4 | 0 | 0.39 | 1.00 / 0.75 / **0.25** / 1.00 / 0.75 |

("best by strict / loose / progress" all resolve to `sac_actor_best.pt` — the only non-final saved checkpoint; deduped.)

### §4 strict-predicate decomposition (loose-entry trajectories, best checkpoint)
- **Joint failures:** `NONE (all pass)` × 2 (the 2 certified deliveries) · `ATTRIBUTION + MECHANISM` × 2. There are **no
  dwell-only, settle-only or body failures** — the two conditions carrying every miss are fingertip-attribution and the
  clean-mechanism (one-finger) check, and they fail *together*.
- **Component margins (best checkpoint, median over loose entries):** dwell **+3.67** (passes with room), settle **+0.75**
  (passes with room), body **+1.0** (passes), attribution **−0.10** (the one real deficit: 0.47–0.50 vs the 0.60 bar).

### §5 blocker classification = **CONTACT_STRATEGY**
Stabilization (dwell, settle) passes 100 % of zone entries with large margins; every miss is in the contact-quality group
(attribution + one-finger mechanism), `contact_group_pass = 0.50` vs `stabilization_group_pass = 1.00`. The gap from loose
to certified is **how the fingers contact the coin** (a clean bilateral push vs a one-finger bulldoze), **not** holding it
still after entry. Not a "wall".

### §6 presentation videos (`experiments/2026_07_20_coin_two_arm_sac_100k/videos/`, honest labels, HUD-overlaid)

| label | state | strict | attribution | dwell | file (sha256 in `video_manifest.json`) |
|---|---|---|---|---|---|
| Learned **certified** delivery (reproducible 10/10) | 64102 | YES | 0.62 | 30 | `learned_certified_delivery_64102.gif` |
| Learned bilateral zone entry (also certified) | 64201 | YES | 0.64 | 26 | `learned_bilateral_zone_entry_64201.gif` |
| **Near**-certified delivery (attribution 0.47<0.60) | 64111 | no | 0.47 | 26 | `near_certified_delivery_64111.gif` |

HUD overlays: target zone, coin, in-zone state, consecutive-dwell counter, coin settle velocity, L/R contact, fingertip
attribution, body-shove, clean-mechanism flag, loose+strict result. A loose entry is never labelled a strict delivery.

### §7 selected next RL intervention (CONTACT_STRATEGY branch) — NOT implemented here
Specification recorded in `next_rl_intervention.md`: **no hold shaping** (stabilization is already solved); **stratify the
demo replay toward high-fingertip-attribution, low-body-shove A1/A4 contact sequences**, and **implement the explicit
demo/online sampling ratio in the existing `ReplayBuffer` sampler** (the §2/§3 ratio not yet wired); **retain the BC
anchor**. Actor, reward, env, strict predicate unchanged.

## Files touched (Phase A + B)

Production: `env/planar_grasp_env.py`, `eval/team_tensor.py`, `train/{coin_delivery_actor,coin_delivery_rl,sac,
pad_aware_control,coin_grip_control,coin_transport,coin_grasp_cert,coin_compliant_pad,coin_delivery_primitives}.py`,
`experiments/coin_delivery1.py`, `coin_delivery/scenarios/kinematic_variant.py`, `coin_delivery/provenance/__init__.py`
(+ deleted `coin_delivery/provenance.py`). New: `experiments/coin_two_arm_sac.py`. Tests:
`tests/{test_coin_delivery_actor,test_coin_delivery1,test_coin_delivery_primitives,test_coin_delivery_scenarios,
test_sac_competence_gate}.py`.

## Test results

330 coin/pad/grip/transport/scenario/golden tests + 2 competence-gate regression tests pass; ruff clean on all
changed/new files (pre-existing E702 semicolon debt in untouched coin lines left as-is: count 26->22, no new lint).

## Open items

- The competence gate dropped `bc_coef` to 0.1 on a single fluke strict (eval#1); per the exact §1 spec (first strict ->
  0.1) but fragile — a "reliable strict" gate (e.g. >=2/14) would be sturdier.
- v2b is applied as the certified anti-farming *spec*; the env's step-reward is the delivery-aligned potential form.
  Binding v2b's `.hymeko` bytecode as the literal per-step reward is a deeper env change, not done here.
- n_step=3 returns and the exact 50/50->25/75 demo/online batch ratio (§2/§3) are not wired into the shared
  `ReplayBuffer` sampler — they require a sampler change with its own regression test; deferred, flagged (not claimed).
  The strict-gap diagnostic now assigns the demo/online-ratio + attribution-weighted seeding as the *next* intervention
  (`next_rl_intervention.md`); n_step is NOT selected (stabilization is already solved).
- The competence gate dropped `bc_coef` to 0.1 on the eval#1 strict — which the diagnostic confirms was a genuine
  REPRODUCIBLE_STRICT, so the gate fired on a real (if non-general) milestone, not a fluke.
- Graphical outputs delivered per §9: `eval_curve.png` (numerical+plotted) and three HUD-overlaid GIFs under `videos/`.
