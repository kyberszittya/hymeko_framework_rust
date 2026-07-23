---
title: Dynamic phase curriculum scope V1
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: STAGE1_DYNAMIC_PHASE_BANK_UNDERPOWERED
tags: [coin, td3, phase-curriculum, contact-flag, underpowered, no-training]
---

# STAGE1_DYNAMIC_PHASE_BANK_UNDERPOWERED — Stage-1d not run

`DYNAMIC_PHASE_CURRICULUM_SCOPE_V1`, measurement only (no training). The contract's own gate (§6) blocks the campaign:
`target_entry` has **zero** persistent dynamic starts, and the Stage-1 phase set is **3%** of the actual gate-on late
dynamics. Stage-1d is **not** run. No Stage 2 / Arm B / SAC / neutral-reset / final-test.

## §3/§4 detector rework — contact is an ORTHOGONAL flag (option B)
The completed Stage-1c detector made `contact_retention` (unilateral contact) a mutually-exclusive control phase that
fired 321× and **erased** the task-progress phases it can coexist with. Reworked to the authoritative representation:
- `control_phase ∈ {transport, target_entry, braking, settling_dwell}` — task-progress only (contact-independent);
- `contact_flag ∈ {contact_present, contact_lost}` — orthogonal;
- `state_onehot = onehot4(control) ++ onehot2(contact)` (dim 6; actor/critic input unchanged at obs_48 ++ 6).
Precedence: settling_dwell > target_entry > braking > transport. Overlaps are rare (7 multi-predicate states:
braking+target_entry 4, braking+settling_dwell 3). Tested: control phase does not collapse to contact; excluded phases
give zero actor gradient; update-0 == pi_0.

## §1/§2/§9 measurement — the Stage-1 curriculum has no basis in the dynamics
Persistent starts by **live** control-phase (rebuilt from PhaseDetector, not `LateStart.family`):

| control phase | persistent starts (min 2) | (min 4) | gate-on occupancy | eligible? |
|---|---|---|---|---|
| **target_entry** | **0** | **0** | 40 | ❌ underpowered |
| braking | 40 | 19 | 287 | ✅ |
| settling_dwell | 18 | 13 | 36 | ✅ |
| transport | — (excluded) | — | **11706** | (dominant, excluded) |

- **`target_entry` never persists ≥2 gate-active steps** → 0 eligible starts. The transition matrix shows why: it is a
  1-step boundary crossing — `transport→target_entry` 38, `target_entry→transport` 33, `target_entry→braking` only 7.
- The gate-on late phase is **97% `transport`** (11706 vs 363 for all Stage-1 phases combined); fraction in the Stage-1
  set = **0.03**. So the declared Stage-1 curriculum {target_entry, braking, settling_dwell} does not describe the actual
  late dynamics — `transport` does.

## §5 Stage-1 actor mask (implemented + tested; moot under underpowered)
`actor_trainable = (gate_t == 1) ∧ (control_phase ∈ {target_entry, braking, settling_dwell})`, applied as the
masked-actor-loss weight. Test proves excluded-phase (transport) transitions contribute **exactly zero** actor gradient
and that changing only their observations cannot change the Stage-1 update. §6 balanced sampling / §7 episode scope /
§8 separate critic occupancy are training-time mechanisms and are moot while the bank is underpowered.

## Decision
Per §6, `target_entry` support (0 < 6) ⇒ `STAGE1_DYNAMIC_PHASE_BANK_UNDERPOWERED` ⇒ **do not train Stage-1d.** This is
the honest terminal of the curriculum-scope contract: the Stage-1 phase curriculum cannot be trained on the actual
dynamics because its keystone phase (`target_entry`) has no persistent support and the set is 3% of the late phase.

## Claims / non-claims
**Claims:** (1) Reworked detector: contact is an orthogonal flag; control_phase is 4-way and contact-independent (tested).
(2) `target_entry` has 0 persistent dynamic starts (min 2 and 4); braking 40, settling_dwell 18. (3) Gate-on late phase
is 97% transport; Stage-1 set = 3%. (4) Stage-1 actor mask gives excluded phases zero gradient (tested). ⇒ underpowered,
no training.
**Non-claims:** NOT that a late controller is impossible — it says the {target_entry, braking, settling_dwell} scope is
wrong for these dynamics; a `transport`-scoped (or braking+settling-only) curriculum is untested. NOT a training result.

## Next narrow experiment (needs your go — this reframes the curriculum)
The dynamics say the late phase IS transport. Two options: (a) **re-scope Stage-1 to the phases that actually persist**
— `braking` (40 starts) + `settling_dwell` (18), dropping the non-existent `target_entry`; or (b) **scope to transport**
(the 97% phase) as the primary late controller. Either keeps the same reward/pi_0/actor/critic/horizon/n-step/smoothing/
transactional caps; only the eligible-phase set + persistent banks change. Do not train until the re-scoped bank clears
the `MIN_STARTS` gate.

## Files
- impl: `hymeko_rl/coin_delivery/coin_dynamic_phase_scope.py`, `hymeko_rl/tests/test_coin_dynamic_phase_scope.py`,
  `experiments/…/dynamic_phase_scope_report_v1.py`.
- results: `experiments/…/dynamic_phase_scope_report_v1.json`, this report.
- upstream: Stage-1c `4b4a0d3`, contract `a4b963d`.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Mac, torch 2.12.0, mujoco 3.10.0. Deterministic
(frozen pi_0 replay); rebuild over seeds 6000–6299, occupancy over 6000–6159.
