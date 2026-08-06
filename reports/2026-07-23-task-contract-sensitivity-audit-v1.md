---
title: Coin task-contract sensitivity audit V1 (measurement-only reset)
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: MULTIPLE_CONTRACT_MISMATCHES — STRICT_RULE_BRITTLE, V3_REWARD_CERTIFIER_MISALIGNED, LOCAL_GATES_OVERCONSTRAINED
tags: [coin, audit, task-contract, certifier-sensitivity, reward-decomposition, measurement-only, no-training]
---

# COIN_TASK_CONTRACT_SENSITIVITY_AUDIT_V1 — the arc's negatives are partly artifacts of an over-constrained contract

Measurement-only. No controller, reward, certifier, physics, or frozen artifact was modified. Four patched checks were
verified before any verdict (post-step-terminal detection, exact reward decomposition, measured per-rollout clearance,
target-directed braking denominator). Trajectories for pi_0, the canonical H=30 planner, and the repaired H=30 planner
were captured on the 31 dev handoffs as post-step certificate streams and re-certified/graded offline.

## 0. Verified patched checks (gate)
| check | result |
|---|---|
| post-step alignment, terminal K6 detected on the final action | **True** |
| exact v3 reward decomposition (`Σ components == scalar`) | **True, max err 0.0** |
| initial clearance measured per rollout | True |
| initial clearance invariant | **False — range [0.0143, 0.0809]** (zone/coin randomises; hardcoded 0.0225 was wrong; all > 0 so footprints-disjoint holds) |

## 1. Physical units (thresholds are not "strict/lenient" until expressed geometrically)
- sim timestep 0.5 ms; frame_skip 20 ⇒ **control dt = 10 ms (100 Hz)**.
- **K=6 dwell = 0.06 s** — a 60-millisecond hold (temporally *short*).
- coin radius 0.02 m, zone radius 0.04 m ⇒ **`dtz ≤ 0.02` = coin fully contained** (margin `zone_half − coin_r = 0.02`); geometrically *strict*. `ENTRY_TOL 0.05 > zone_half 0.04`, so "entered" is 1 cm *beyond* the zone edge (loose).
- **`settle < 0.06 m/s` = 0.6 mm/control-step** (~3 % of coin radius/step).

So the certifier is **geometrically strict** (full containment) but **temporally short** (60 ms) and **permissive on causal attribution** (touched-ever, see §6).

## 2. Four separated contracts, and requirements in B/C/D not in A
- **A (physical task):** move the coin from outside into the target zone and leave it stably at rest; push-and-coast is valid, **force closure not required** (from `delivery_certificate.py`).
- **B (certifier):** `centered(≤0.02) ∧ settled(<0.06) ∧ clean` for K=6 **∧ robot_touched(ever) ∧ footprints_disjoint**.
- **C (v3 reward):** `terminalgraded 30 · zoneprog 10 · approach 4 · bothapproach 4 · bodyprogpen 5 · oob 2`.
- **D (curriculum gates):** the qualification clauses — required-contact-retention hard gate, exit-before-K6, strict/dwell advantage; late-start families.

**In B/C/D but not required by A:** the full-containment `≤0.02` (A only needs the coin *in* the zone, ≤ `zone_half` 0.04); the exact 60 ms K=6 hold; **grasp/approach reward terms (C)** — A allows push, but C shapes toward a *grasp* pose; the **contact-retention hard gate (D)** — A permits release after placement.

## 3–4. Certifier sensitivity surface + ranking stability → **STRICT_RULE_BRITTLE**
Canonical (0.02/0.06/K6) success: pi_0 **0.032**, h30 **0.452**, repaired **0.355** → ranking `h30 > repaired > pi_0`.
Across the 4×4×5 grid the **top controller flips 36 times**, and the flips are meaningful (not just ties):

| cell | pi_0 | h30 | repaired |
|---|---|---|---|
| canonical 0.02/0.06/K6 | 0.032 | 0.452 | 0.355 |
| **stricter settle 0.02/0.03/K6** | **top** (pi_0 ≥ planners) | — | — |
| looser dwell 0.02/0.06/**K1** | 0.226 | 0.452 | 0.355 |
| loosest 0.04/0.12/K1 | **0.419** | 0.452 | 0.355 |

The "planners dramatically beat pi_0" conclusion holds **only near the canonical cell**: under a 2× *stricter* settle velocity the gentle pi_0 **out-delivers** the aggressive planners (they overshoot and never settle), and under looser dwell the 14× gap collapses to near-parity (0.42 vs 0.45). Single-axis flips: **settle_vel and dwell_K** (not center_tol). ⇒ **STRICT_RULE_BRITTLE**.

## 5. Success ladder → **LOCAL_GATES_OVERCONSTRAINED**
| controller | entry | in-zone | **K3 (0.03 s)** | **K6 (0.06 s)** | K10 (0.10 s) |
|---|---|---|---|---|---|
| pi_0 | 0.516 | — | **0.258** | **0.032** | 0.0 |
| h30 | 0.710 | — | 0.645 | 0.516 | 0.0 |
| repaired | 0.677 | — | 0.516 | 0.387 | 0.0 |

The dwell-survival curve **falls off a cliff between K6 and K10 (K10 = 0 for everyone)**, and **K3 ≫ K6** (pi_0 holds 0.03 s in 26 % of states but 0.06 s in only 3 % — an 8× drop). Binary strict-K6 sits on the steepest part of the curve and **discards the graded sub-K6 competence** a local/curriculum objective could use. ⇒ **LOCAL_GATES_OVERCONSTRAINED**.

## 6. Touched-ever audit — strict on settling, permissive on attribution
Among certified deliveries, the fraction with **no current robot contact through the dwell**: h30 **14/14**, repaired **11/11**, pi_0 0/1. So **100 % of planner deliveries certify while the robot is not touching the coin** (pure push-and-coast; legal under A). The certifier is simultaneously **strict on settling geometry** and **permissive on causal attribution** (touched-*ever*, not current).

## 7. v3 reward decomposition → **V3_REWARD_CERTIFIER_MISALIGNED**
Exact per-step decomposition (`Σ = scalar`, err 0.0). Mean contribution **per step**:

| term (weight) | pi_0 | h30 |
|---|---|---|
| grasp_approach (4) | **−0.39** | **−0.68** |
| both_approach (4) | **−0.46** | **−0.82** |
| zone_progress (10) | +0.009 | +0.009 |
| terminal_deliver_graded (30) | +0.15 | +0.57 |
| body_progress_penalty (5) | 0.0 | 0.0 |

Two measured misalignments with the physical task A: (1) the **dense signal is dominated by *negative* grasp-pose terms** (−0.8 to −1.5/step combined) that penalise the arms for not being in a *grasp* pose — but A allows **push**-delivery, so the reward fights a valid strategy (and penalises the aggressive planners more, −1.5 vs −0.85); (2) **zone_progress ≈ 0/step** (PBRS telescopes to ~0) — there is **no dense progress signal** driving the coin to the zone. The only task-aligned signal is the **sparse binary K6 terminal** (return Δ delivered−not: pi_0 26, h30 58, repaired 75 — dominated by the +30). RL under this reward chases a knife-edge K6 terminal against a grasp-pose penalty baseline. ⇒ **V3_REWARD_CERTIFIER_MISALIGNED**.

## 8. Braking-eligibility recalculation (corrected, target-directed denominator)
Part-A divided by all 272 braking-labeled states, including **48–70 retreating (target-away)** and **8–24 already-slow** states that need no braking. With `target_directed_radial_velocity > v_excess` (signed, not `abs`):

| v_excess | target-directed | target-away | support/target-directed | support/all |
|---|---|---|---|---|
| 0.03 | 84 | 70 | 0.286 | 0.206 |
| 0.05 | 76 | 48 | **0.290** | 0.206 |
| 0.07 | 64 | 39 | 0.328 | 0.206 |
| 0.09 | 52 | 33 | 0.327 | 0.206 |

The denominator **was partly invalid** — correcting it raises support 0.21 → **0.29–0.33** (and reveals 48 retreating + up-to-24 slow states wrongly counted). But even corrected, support stays **below the 0.5 bar**, so the Part-A "insufficient" conclusion **survives** — with a narrowed margin. Not enough to fire `BRAKING_SUPPORT_DENOMINATOR_INVALID`, but the denominator error is real and noted.

## 9–10. Reclassification of prior negative conclusions (frozen reports unmodified)
| prior conclusion | status under this audit |
|---|---|
| supervised ceiling | **depends_on_strict_K6** — the "ceiling" is the binary K6 terminal; the ladder shows K3 competence it hides |
| no beneficial support / local improvement exhausted | **depends_on_strict_K6 local denominator** — local labels required a *direct* strict gain (§9 of the V2 spec) |
| braking primitive insufficient | **mostly robust** — survives the denominator fix (0.29 < 0.5), but the old `abs`-denominator inflated the failure |
| primitive loses required contact | **depends_on_contact-retention hard gate + strict-K6** — the deficit is braking-phase, coupled to the K6 push |
| H30 teacher unqualified | **split** — the *delivery advantage over pi_0* is **threshold-brittle** (flips at stricter settle); the *contact/exit deficit* is threshold-independent (measured directly), so "drops contact vs pi_0" is robust |

## 11. Verdict
**`MULTIPLE_CONTRACT_MISMATCHES`** — independently supported: **`STRICT_RULE_BRITTLE`** (ranking flips across settle_vel/dwell_K; pi_0 tops at stricter settle), **`V3_REWARD_CERTIFIER_MISALIGNED`** (dense reward penalises valid push-delivery, no dense progress term, only a sparse K6 terminal), **`LOCAL_GATES_OVERCONSTRAINED`** (K3 competence hidden by binary K6 on a steep dwell cliff). `TASK_CONTRACT_ALIGNED` is refuted; `BRAKING_SUPPORT_DENOMINATOR_INVALID` did not fire (corrected support still < 0.5).

## 12. Exactly one recommended next experimental baseline
**`GRADED_PUSH_DELIVERY_OBJECTIVE_V1` (measurement-only re-scoring, no new controller, no training).** Re-score the already-captured trajectories (pi_0, H30, repaired, primitive) under a **task-A-aligned graded objective** that: (a) rewards the **centered-settled dwell integral** and the K3/K6 ladder rather than the binary K6 knife-edge; (b) uses a **dense zone-progress** signal (non-telescoping) as the primary driver; (c) **drops the grasp-pose penalty** for the push-delivery task (force closure not required by A). Then re-test whether "local improvement exhausted", "supervised ceiling", and "primitive underperforms" survive when the metric matches the graded ladder and the reward stops penalising push. Only if the negatives survive the aligned objective is a reward/curriculum change (a CORE/task edit, requiring approval) or a new controller warranted. This is the smallest step that discriminates "the methods failed" from "the contract mis-scored them".

## Files
- impl: `hymeko_rl/coin_delivery/coin_contract_audit.py`, `hymeko_rl/tests/test_coin_contract_audit.py` (9 tests).
- entries/plot: `experiments/…/rl_entry/{coin_contract_audit_capture.py,coin_contract_audit_analyze.py,plot_contract_audit.py}`.
- results: `…/audit_trace_{pi0,h30,repaired}.json` (post-step streams + measured clearance + exact reward decomposition),
  `…/task_contract_audit_v1.json`, `…/task_contract_audit.svg`, this report.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c` (immutable); reward v3 `galambos_task_deliver_v3.hymeko`;
strict-K6 certifier; frozen 31-state dev bank (`config_sha 3ec6dbeb`). Measurement-only, deterministic. All four patched
checks passed before the verdict. No CORE.YAML / reward / certifier / controller changes.
