# Coin launch + brake micro-tests — the two remaining walls, isolated

**Date:** 2026-07-25
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + frozen `V4_INTERMITTENT_CONTACT`. No retraining, no proposal, no RL.
**One-line outcome:** with the material model corrected and the controlled impulse proven, the two remaining sub-problems
were isolated: **braking is solved** (a predicted passive barrier lands the coin in the zone 100% of overshoot cells), and
the **one open wall is the target-directed launch** (the impulse exists but drifts laterally, not toward the zone).

---

## Status ledger

| capability | status |
|---|---|
| mis-calibrated table physics | CORRECTED (V2, Coulomb μ≈0.15) |
| under-driven contact impulse | CORRECTED (accel_gain 0.25) |
| single-tip physical authority | PASS on a subset (0.6 m/s, within the motion contract) |
| **target-directed** transport | **PARTIAL — impulse not yet target-directed (launch wall)** |
| **predictive braking / zone landing** | **PASS — passive barrier, 100% of overshoot cells** |

## Micro-test 1 — LAUNCH-ONLY (`coin_launch_benchmark.py`)

Bounded preload → target-directed impulse → release (predictive braking off). Controller-independent diagnostic categories
over 8 states:

```
TARGET_DIRECTED_IMPULSE_CAPABLE  []        (NONE)
IMPULSE_CAPABLE                  [1,6,7]   (launch-scale speed, but NOT toward the zone)
CONTACT_CAPABLE                  [0,2,3,4,5]
→ IMPULSE_EXISTS_BUT_NOT_TARGET_DIRECTED
```

`peak_coin_velocity` masked the problem — the enriched metrics expose it: s1 peak *target-projected* velocity 0.478 but
*cross-track* 0.546 (more sideways than forward); s7 target 0.099 vs cross 0.448; s2/s3/s4 **negative** signed
displacement (coin driven *away* from the zone). The impulse magnitude is there; its **direction** is not. This is a
force-direction / contact-geometry control problem, distinct from (and upstream of) braking.

## Micro-test 2 — BRAKE-ONLY (`coin_brake_benchmark.py`, `brake_control.py`)

Launch removed: the coin is placed at a known remaining distance (4/5/6 cm) with a known velocity toward the zone
(0.5/0.6 m/s); the arm positions the fingertip as a **barrier on the coin's approach line at the zone edge**, driven by
the coast-model prediction. Progressive arms B0–B5; explicit event chain logged.

```
zone entry by mode (overshoot cells):
  B0_coast            0.0    ← free coast never lands (overshoots / arm-knock)
  B1_passive_landing  1.0    ← the predicted PASSIVE BARRIER lands the coin in the zone, every cell
  B2_recontact        0.3
  B3_counter_impulse  0.3    ← ACTIVE re-contact braking does WORSE than the passive barrier
  B4_terminal_corr.   0.3
  B5_settle           0.3
trigger crossed 1.0 | re-contact 0.8
→ PREDICTIVE_PASSIVE_BARRIER_BRAKING_CAPABILITY_ESTABLISHED
```

The coast model (accurate to ~1 cm) predicts *where* the overshooting coin will pass; a passive barrier there catches it in
the zone (stop error 0.7–1.9 cm, held with dwell). The **active** counter-impulse / re-contact chase (B2–B5) is
**over-engineered** — it disturbs the coin and lands only 30% of the time. The right terminal primitive is the simple
predicted barrier, not an active brake.

Honest caveats: the B0 coast reference is contaminated (the arm knocks the coin while retracting home), so B1's *absolute*
100% (not the B1−B0 delta) is the clean signal; the grid is 12 overshoot cells on 2 states (the FD barrier setup makes a
larger grid impractically slow); state 14250 lands less reliably (re-contact fires but stops ~2–4 cm short) — the barrier
placement is state-dependent.

## Main finding

> Removing the mis-calibrated material model did not by itself restore delivery, but it improved transport. An explicit,
> governor-bounded contact impulse then restored the required object velocity (~0.6 m/s) and >15 cm of transport at
> realistic robot speed. Isolating the terminal problem, a coast-model-predicted **passive barrier** stops the moving coin
> inside the zone reliably, so braking/terminal control is **not** the wall. The one remaining limitation is producing a
> **target-directed** launch impulse — the object velocity exists but is not yet aimed cleanly at the zone (lateral drift),
> a force-direction / contact-geometry control problem.

## Decision-tree position + exact next gate

- Braking: `PREDICTIVE_PASSIVE_BARRIER_BRAKING_CAPABILITY_ESTABLISHED` (use the passive predicted barrier; drop the active
  counter-impulse).
- Launch: the open wall — **target-directed impulse**. Next: control the launch *direction* (align the contact-Jacobian
  push with the zone axis; regulate cross-track velocity to ~0; possibly a two-point / edge-aware contact to steer the
  coin), evaluated with the launch benchmark's target-projected vs cross-track metrics.
- Then **compose** the proven pieces: target-directed launch → predictable coast → predicted passive barrier. Only if the
  composed launch→coast→brake lands delivery is a proposal / bounded search / RL layer justified. **O3 stays paused.**

---

### Commits
- `3ecd2c63` — launch-only benchmark + enriched force_slip_carry (target-projected / cross-track / phase log).
- (brake-only controller + benchmark + this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, all prior results.
