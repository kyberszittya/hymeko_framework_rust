---
title: PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION — the OR predicate arms on acquisition brushes
date: 2026-07-23
slug: coin-gate-unilateral-stress
task: coin_v3 delivery — PHASE_GATED_LEARNED_RESIDUAL_CONTROLLER (§7 stress panel)
verdict: PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION
blocks: critic fitting / residual learning until the arm predicate is decided
---

# §7 unilateral stress panel — the deployed OR predicate fires too early

**Created-at:** 2026-07-23 02:35 CEST
**Accepted going in:** `PHASE_GATE_RUNTIME_CONTRACT_PASS` (commit `9778f17`, gate contract SHA `d739e8af`,
predicate `left_contact OR right_contact`). **This stress panel invalidates that predicate for the residual's
purpose** and returns the §7 halt verdict before any critic/residual learning.

## The question (§7)

Can **three consecutive UNILATERAL (single-finger) contact steps** arm the residual during genuine
`CONTACT_ACQUISITION` — i.e. before a stable bilateral grasp has ever formed? The §4 validation labelled every
activation "TRANSPORT", but that label used `touched = left OR right`, so it could not distinguish a single-finger
acquisition brush from stable grasp-transport. §7 records left/right **separately**.

## Result — material premature activation

Synthetic named cases (arm_after=3):

| case | arms at | window unilateral? |
|------|---------|--------------------|
| unilateral_transient_2 | never | — |
| **unilateral_3steps** | **t=3** | **True** |
| **alternating_LR** | **t=2** | **True** |
| bilateral_3steps | t=3 | False |
| contact_bounce | never | — |
| **acquire_unilateral_then_bilateral** | **t=3 (before bilateral at t=4)** | **True** |

Real acquisition dynamics (π₀ on 9 headline seeds + 5 certified deliveries, contact recorded per-finger):

- **11/14 trajectories (material rate 0.786)** arm the residual on a unilateral window during acquisition.
- On **all 9 π₀ seeds the FIRST arm is on a non-bilateral (unilateral) window** (`first_arm_bilateral = False`).
- 39 total arm events, 31 on unilateral windows.

⇒ Under the deployed `OR` predicate the residual would be **active during contact acquisition on ~79% of
trajectories**, exactly the phase where it could perturb the grasp formation π₀ does at 9/9. This is the risk §7
guards. **`PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION`.**

## Proposed deployable refinement (measured, NOT auto-adopted)

Per §7 the refinement must be deployable and must **not** use target distance or offline phase labels. I measured
the first suggested option — **bilateral contact** (arm on `left AND right`, disarm on complete loss `not(L or R)`) —
offline on the same recorded traces (the deployed gate was not changed):

| predicate | traj with premature acq | total premature | still arms in transport |
|-----------|-------------------------|-----------------|--------------------------|
| `left OR right` (deployed) | **11/14** | 25 | yes (but too early) |
| `left AND right` (candidate) | **0/14** | **0** | yes on grasp-style deliveries |

Bilateral-arm **eliminates premature acquisition arming (0/14)** and still engages through transport on deliveries
that form a bilateral grasp (π₀ 1011: 128 active transport steps; certified 6000/6002/6003: ~70–89).

**But a genuine coverage gap** (figure `reports/figures/coin_gate_unilateral_stress.png`): the task is **push-valid**
(force closure not required — `delivery_certificate.COIN_DELIVERY_STRICT`), and seeds **1447 (a delivery)**, 1202,
1278, 1358, **6005** form **no bilateral contact ever** (unilateral push/coast). A pure bilateral predicate would
**never arm the residual** on those — scoping the residual to grasp-style transport only, and abandoning 1/3 of
π₀'s headline deliveries (1447) to no late-phase help.

## Decision required (halt)

Changing the arm predicate changes the gate contract SHA (currently `d739e8af`) on which
`PHASE_GATE_RUNTIME_CONTRACT_PASS` was accepted, and defines `PHASE_GATE_CONTROLLER_STATE_V1` (§3). Per §7 this is a
proposal, not an auto-change. Two deployable options:

- **(A) Bilateral arm** (`left AND right`) — minimal, fully deployable, eliminates premature arming; **accepts** that
  unilateral push-deliveries (1447-style) get no residual (the residual becomes a grasp-transport helper only).
- **(B) Unilateral + a canonical stability signal** — arm on sustained `left OR right` **plus** a deployable
  stability signal (e.g. coin co-motion with the gripper / sustained contact beyond an acquisition-brush horizon),
  to cover push-deliveries too. Heavier: needs a coin-kinematics signal (deployable via perception) and its own
  false-positive audit.

**Recommendation:** (A) if the residual is intended only for grasp-style transport/settling; (B) if push-delivery
coverage (incl. seed 1447) is required. I did not implement either into the gate — awaiting the decision, since it
re-derives the accepted gate contract and everything downstream (controller-state schema, critic conditioning).

## Not built this turn (deliberately halted)

§1 update-0 reproduction, §2 structural preservation, §3 `PHASE_GATE_CONTROLLER_STATE_V1`, §4 critic conditioning,
§5 replay/target-action contract — all depend on the finalized gate (its arm predicate defines `arm_counter`, and
§3's schema SHA). Building them on the premature `OR` gate would bake in the defect. §1/§2 are predicate-independent
(residual=0 ⟹ composite=π₀; gate=0 ⟹ composite=base) and can execute immediately once the predicate is fixed.

## Files touched

- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_gate_unilateral_stress.py` (new) + `unilat_stress.json`.
- `reports/figures/coin_gate_unilateral_stress.png` (new).

**CORE.YAML items touched:** none. **Deployed gate unchanged** (still `d739e8af`). SAC quarantined. Mac; kato14 clean.
