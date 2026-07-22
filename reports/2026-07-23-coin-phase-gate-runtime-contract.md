---
title: PHASE_GATE_RUNTIME_CONTRACT — deployable contact-hysteresis gate for learned-residual TD3
date: 2026-07-23
slug: coin-phase-gate-runtime-contract
task: coin_v3 delivery — PHASE_GATED_LEARNED_RESIDUAL_TD3 (§3–§4)
verdict: PHASE_GATE_RUNTIME_CONTRACT_PASS
---

# Deployable phase gate — §3 contract + §4 runtime validation

**Created-at:** 2026-07-23 02:10 CEST
**Objective:** `PHASE_GATED_LEARNED_RESIDUAL_TD3_SMOKE_PASS` (this stage: the first gate, §4).
**Prior accepted findings gating this design:** `VALID_BUT_CHAOS_SENSITIVE_UPDATE`,
`PROXIMAL_ANCHOR_LIMITS_ACCUMULATED_DRIFT` (F-SAC-11), `PROXIMAL_ANCHOR_FAILS_FIRST_STEP_BASIN_PRESERVATION`
(F-SAC-12, registry `0464d37`). F-SAC-12 is why we protect the early policy **structurally** (a gate that zeroes
the residual) instead of globally regularizing shared parameters.

## The observability question (§4 gate)

`PHASE_GATE_RUNTIME_CONTRACT_PASS` vs `PHASE_GATE_OBSERVABILITY_BLOCKED` turns on one question: **can the
deployable runtime contract distinguish "stable robot-attributed grasp held for N steps" from "transient contact"
without privileged simulator state?** Answer from the code:

- The canonical delivery certificate (`delivery_certificate.CertStep`), `CoinRL4Dof` (`_touched`), and
  `eval_bc_delivery` all read **robot-attributed fingertip contact** = `inner._planar_metrics.left_contact or
  right_contact`. Physically this is a **gripper tactile sensor** — available on a real robot.
- "Stable grasp" vs "transient contact" is a **consecutive-step count** (transient < `arm_after`; stable ≥
  `arm_after`), needing **no** seed, trajectory id, future info, planner state, or `disk_to_zone`.

⇒ The gate is deployable. **`PHASE_GATE_RUNTIME_CONTRACT_PASS`.** (Had contact required privileged coin-vs-zone
pose to disambiguate, we would have stopped with `PHASE_GATE_OBSERVABILITY_BLOCKED`; it does not.)

## §3 gate contract — `coin_phase_gate.PhaseGate`

Deterministic 4-state FSM with explicit hysteresis; **generates no actions** — returns a multiplier `g_t ∈ {0, 1}`
that scales the residual, so `g_t = 0 ⟹ composite = pi_0` exactly.

| state | meaning | g |
|-------|---------|---|
| `EARLY_CONTROL` | approach / acquisition / initial grasp | 0 |
| `LATE_CONTROL_ARMED` | robot-attributed contact held `arm_after=3` consecutive steps | 1 |
| `REACQUIRE` | armed then contact lost `disarm_after=2` consecutive steps; base recovers | 0 |
| `TERMINAL` | strict K=6 / episode end (absorbing) | 0 |

Reused the canonical contact **signal** (not the F21 `ReadinessDetector`, which is a nearest-ready-*distance*
detector with momentary readiness — recorded NO_EFFECT; this is a distinct, simpler contact-count hysteresis the
directive specifies). Gate-contract SHA-256 prefix `d739e8af`; forbidden inputs enumerated in `contract()`.

## §2 frozen base — `pi_0` file-SHA `1902454c`

`pi_0` is loaded from the immutable persisted checkpoint `frozen/pi0_shared_clip_actor.pt` (file-SHA
`1902454ca7a74c27…`), via the new reusable `rl_clip_actor.load_frozen_clip_actor`. Reconciliation note: an earlier
raw-param hash gave a different prefix; the canonical identity is the **saved file** because `torch.save` is not
byte-reproducible across independent saves. The loaded params are **byte-identical (maxdiff 0.0)** to
`build_shared_sac_td3(bc_handoff_only_best)`, and the actor reproduces 3/9 headline · 2/30 val · 9/9 grasp ·
delivered {1011, 1447, 1568}. Every parameter `requires_grad=False`.

## §4 runtime validation — 14 trajectories, 3 classes

Rolled from the canonical **neutral reset**; the gate saw only deployable contact. `dtz`-derived phase labels are
**diagnostic only** (never seen by the gate).

| class | seeds | first-contact → ARM | activation phase | false-early | reacq (total) | hysteresis |
|-------|-------|--------------------|--------------------|-------------|---------------|------------|
| success_pi0_K6 | 1011,1447,1568 | c+2 (3rd contact step) | TRANSPORT (all) | 0 | 0/5/3 | honored |
| fail_pi0 | 6 headline fails | c+2 | TRANSPORT (all) | 0 | 2–5 | honored |
| certified_delivery | 6000–6005 | c+2 | TRANSPORT (all) | 0 | 0–4 | honored |

**All §4 required behaviours hold:**

- residual **never active during APPROACH** — `false_early_activations = 0` on all 14 (gate needs contact to arm);
- residual **never active before stable contact** — first activation = first_contact + 2 everywhere (arm on the 3rd
  consecutive contact step), deterministic;
- residual **active through TRANSPORT / TARGET_ENTRY / SETTLING** where contact is valid — confirmed on all 3 K6
  deliveries (active_by_phase covers transport→dwell);
- **contact-loss behaviour deterministic** — 34 total reacquisitions across the set are **legitimate** REACQUIRE
  events on `pi_0`'s bouncy transport grasp (seed 1447: 5 disarm/5 re-arm), each honoring the full hysteresis window
  (`hysteresis_honored = True` on all 14, no sub-window chatter). Max toggle count 12 (seed 1447) is real contact
  bounce, not gate thrash;
- **no future / privileged information** — gate reads only `left_contact or right_contact`.

Figure `reports/figures/coin_phase_gate_runtime.png`: gate OFF in approach, ARM at first stable transport contact,
green REACQUIRE gaps on contact loss, delivery as `dtz → 0`.

## Tests

- `hymeko_rl/tests/test_coin_phase_gate.py` — **12/12 pass** (0.68 s): never-arms-without-contact, transient-never-
  arms, arms-after-3, single-dropout-holds, disarm-after-2, reacquire-rearms, terminal-absorbing, chatter-suppressed,
  reset-clears, gate-property, SHA-deterministic/config-sensitive, invalid-config-rejected.

## Files touched

- `hymeko_rl/coin_delivery/coin_phase_gate.py` (new, 145 L) — the deployable FSM gate + `robot_attributed_contact`.
- `hymeko_rl/coin_delivery/rl_clip_actor.py` (+27 L) — `load_frozen_clip_actor` (reusable frozen-pi_0 loader).
- `hymeko_rl/tests/test_coin_phase_gate.py` (new, 12 tests).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_phase_gate_validation.py` (new) + `phase_gate_val.json`.
- `experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_manifest.json` (SHA manifest; the `.pt` is host-local
  per §19, byte-regenerable via `build_shared_sac_td3`).
- `reports/figures/coin_phase_gate_runtime.png`.

**CORE.YAML items touched:** none (`hymeko_rl/` is non-core; `on_unknown_path: treat_as_non_core`).
**Static gate:** ruff crit (E9/F) clean on new files; E702 semicolon waiver on the compact validation harness only.

## Scientific classification (§1, §17)

This is **not** a monolithic full-action TD3. Name preserved: `PHASE_GATED_LEARNED_RESIDUAL_TD3`. The base is a
**frozen learned policy** (`pi_0`), not a scripted transport base — distinct from historical residual-on-scripted
experiments. This stage validates only the deployable gate; no residual is trained yet.

## Next gates (subsequent turns)

`PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED` (§2, zero-residual = pi_0) → `EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS`
(§8) → `PHASE_GATED_RESIDUAL_CRITIC_PASS` (§11) → `LATE_PHASE_RESIDUAL_GRADIENT_CONTRACT_PASS` (§12) → guarded
micro-smoke (§13–15).
