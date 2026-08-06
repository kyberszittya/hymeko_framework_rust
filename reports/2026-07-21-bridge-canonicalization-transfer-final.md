---
campaign: Overnight package — bridge-relay + canonicalization + pick-and-place + Beni transfer
title: Bridge-relay NO_EFFECT (handoff inert); canonicalization bounded (guards + 3 consolidations); P&P + Beni transfers execute
date: 2026-07-21
branch: exp/coin-bridge-relay
final_head: 3381bd0
bridge_classification: NO_EFFECT
---

# Overnight work package — final report

**Bridge classification: NO_EFFECT** — the bridge-relay's handoff never fires; the trained relay does not exceed
transport-alone. Well-diagnosed, not an impossibility. All four phases executed to the extent technically safe; the one
genuinely risk-blocked item (a 45-module import migration, unverifiable blind overnight) is reported as a scoped
boundary, not skipped silently.

## 1. Frozen F11/F21 source and artifacts
Baseline `b615f81` (frozen, unamended). F11/F21 report `reports/2026-07-21-coin-f11-f21-contact-actor-bank.md`;
freeze manifest `experiments/2026_07_21_coin_bridge_relay/freeze_manifest.json` (16 checkpoints + 8 corpora hashed). All
16 campaign processes confirmed terminated before branching.

## 2. Frozen TRANSPORT_POLICY
`experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt`, **sha256 `39551de3…`**. Verified via canonical
deterministic rollout: strong state `04870b0e0357ecb5`, signed clearance **+0.0253** → **greedy strict 10/10**,
zero-action 0/10. Config obs_dim 41 / action_dim 6 / pooled / max_steps 60.

## 3. Transport-ready bank and detector
769 candidates (successful greedy trajectories + STAGE-1 held) → **39 TRANSPORT_READY** (244 LOOSE, 94 CONTACT_ONLY,
392 NOT_READY), labelled deploy-matched (frozen greedy transport certifies). Detector = nearest-ready-state kNN over 15
**named** public fields, enter 0.373 / exit 0.746 (self-excluded calibration, inf-sanitized features).

## 4. Bridge demonstration corpus / curriculum bands
Reverse curriculum by distance-to-basin: B0_ready 39 · B1_near 243 · B2_mid 243 · B3_far 244 · B4_clear_start 192.
Bridge warm-started from the transport policy; bridge-reward = potential toward basin + contact bonuses + dominant
terminal READY bonus (delivery-v2b / strict / env untouched).

## 5. Curriculum bands reached
All 5 trained (8k/12k/15k/15k/25k steps). Best band = **B0** (held1 5/24).

## 6. Bridge and transport occupancy (relay)
Handoff occupancy = **0** (see 7). ready-entry 0.167 on held STAGE-1; the relay ran essentially all-bridge.

## 7. Readiness-entry and handoff rates + 9. causal comparison
| checkpoint | held1 strict | handoff | ready-entry |
|---|---|---|---|
| transport-alone | 4/24 | — | — |
| untrained relay (transport clone) | 4/24 | 0/24 | 4/24 |
| **trained relay B0 (best)** | **5/24** | **0/24** | 4/24 |
| trained relay final (B4) | 0/24 | 0/24 | 4/24 |
| zero-action | 0/24 | — | — |

held2 (far STAGE-2): all 0/24. **Two diagnosed failures:** (a) the **handoff never fires** — detector-readiness is
momentary, never 3 consecutive steps, and the bridge reward *terminates on first entry* so the bridge never learns to
DWELL; (b) **progressive forgetting** (B0 5/24 → B4 0/24). The best-checkpoint +1 (5 vs 4) is within noise and is
bridge-alone (handoff inert), not relay-driven. → **NO_EFFECT**. Detail: `reports/2026-07-21-coin-bridge-relay.md`.

## 8. Maximum certified initial clearance
No relay-driven certified clear-start gain. The bridge-alone best certifies STAGE-1 (near) states; max reproducibly
certified clear-start clearance remains the frozen transport policy's **+0.0253** (unchanged by the relay).

## 10. Coin demo command and video paths
**Phase 8 gated on BRIDGE_POSITIVE → not produced** (no far-start video, no demo command, no closure). Honest diagnostic:
the best relay certifies 5/24 STAGE-1 as bridge-alone; STAGE-2 0/24. Coin Delivery is **not** closed by this work.

## 11. Bridge commits and report
`80d76cb` (module) · `0ce7028` (injectable selector) · `f208416` (run + report). Report as above.

## 12. Canonicalization branch and commits
Bounded, on this branch: `0ce7028` injectable bank selector · `3381bd0` architecture-regression guards.

## 13. Duplicate modules migrated/deleted + 14. architecture guards
**Consolidations executed this session:** (1) `hymeko_rl/eval/paired_stats.py` = canonical percentile-bootstrap owner
(the ~20 scattered experiment copies migrate here); (2) `ContactActorBank` selector made **injectable** — one bank
owner, task-agnostic (no per-task fork), the §13.3 transfer seam; (3) tanh-squash math de-duplicated into
`_squashed_sample`/`_squashed_mean` (was 3 copies). **Guards** (`test_architecture_guards.py`, 3 tests): production
library code must not import `hymeko_rl.experiments` (**ratchet baseline 45**, may only shrink); paired_stats is the
canonical bootstrap owner; the bank selector is injectable.
**Concrete blocker (genuine, reported not skipped):** 45 production modules import experiment entry points — dominated
by a `galambos_demo` cluster (`PhasePushController`/`_ik_action`/`_extract_arms` in ~13 agents/env modules). Migrating
these to a production home + rewiring 45 imports is technically possible but a **45-module blast radius** I cannot
verify blind overnight without the full experiment suite; doing it unverified violates the operating contract's
"silence over a wrong action." Executed the ratchet (caps + documents the debt); the migration is the next
**supervised** cleanup step, with the exact inventory in the guard test's baseline comment.

## 15. Golden before/after
Frozen transport policy re-verified bit-identical post-work (greedy strict 10/10 on `04870b0e`). F11 path byte-identical
(pooled default); 83 unit/regression tests pass; new bank code all A-grade cyclomatic; the injectable selector defaults
to coin behaviour (all prior bank tests unchanged).

## 16. Pick-and-place smoke result
`hymeko_rl/experiments/transfer_smoke.py` — **PickPlaceEnv executed** (not surveyed): obs (9,10), action 7, max_steps
200. Canonical SAC 3000 steps: losses finite, params finite, **checkpoint round-trip identical**, deterministic eval
runs (delivered 0.0 — expected untrained at 3k). Modes spec recorded {APPROACH_GRASP, LIFT_TRANSPORT, PLACE_RELEASE,
RECOVERY_REGRASP} for the injectable bank selector. Missing before longer training: a task mode-selector over P&P
contact fields + a demonstration corpus (BC anchor) — the env already exposes `delivered`/`lifted`/`both_contact`
certification. Artifact `experiments/2026_07_21_transfer_smoke/pick_place_actor.pt` (sha `d2da720a…`).

## 17. Beni/humanoid smoke result
**LeggedLocomotionEnv (Beni) executed** via `make_humanoid`: obs (13,2), action 12, max_steps 300, free base. Canonical
SAC 3000 steps: losses finite, params finite, **action affects the plant**, **checkpoint round-trip identical**,
upright-time trains (pre 39.7 steps). Task = stable standing / upright locomotion (the smallest supported). Modes spec
{STABILIZE, ADVANCE, RECOVER_BALANCE}. Artifact `beni_actor.pt` (sha `4630b537…`).

## 18. Exact remaining manipulation boundary
`LeggedLocomotionEnv` is **locomotion-only** (biped, no object / gripper / target body), so humanoid *manipulation* is
genuinely unsupported today. The minimal extension for a reach/contact task: (1) an object + target body in
`humanoid.hymeko` (or a companion plant), (2) an arm / end-effector actuator group, (3) a reach `RewardSpec` + typed
contact certification. The canonical policy / trainer / rollout / checkpoint + the injectable bank selector already
transfer — only the plant + task adapter are missing. No fake manipulation result was fabricated.

## Provenance
- Final HEAD **`3381bd0`**, branch `exp/coin-bridge-relay` (off frozen `b615f81`).
- `git status --short`: only pre-existing session-start `M` files (hymeko_monitor / hymeko_neuro / data / docs); all this-session work committed.
- New report paths: `reports/2026-07-21-coin-bridge-relay.md`, `reports/2026-07-21-bridge-canonicalization-transfer-final.md`.
- Checkpoint / artifact hashes: transport `39551de3`; bridge B0 `c669c1e5`; bridge_result.json `a128bf83`; transfer_smoke.json `9af5c225`; pick_place_actor `d2da720a`; beni_actor `4630b537`. Curriculum corpora + F11/F21 checkpoints in the freeze manifest.
- Host Apple M5 Pro / 18 cores / 48 GB; torch 2.12.0, mujoco 3.10.0, numpy 2.4.6. RL not bit-reproducible (§3); verdicts rest on causal margins / matched bootstrap.

## Honest closing
The bridge idea is sound but the implementation's handoff was inert (momentary readiness vs a 3-step gate + terminate-
on-entry reward). The evidenced next iteration (spec only): reward *dwelling* in the basin, 1-step handoff, retention
guards, and a basin with reachable intermediate ready states. No proposed Coin factorial; no unsupported result claimed.
