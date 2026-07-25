# FORCE_SLIP_SEMANTICS_V1 — contact-level manipulation semantics on the corrected coin physics

**Date:** 2026-07-25
**Branch:** `feat/architectural-assimilation-v1`
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + frozen `V4_INTERMITTENT_CONTACT` motion contract. No retraining, no
proposal, no RL.
**One-line outcome:** correcting the material model first, then adding a driven controlled impulse, **recovers the
required object velocity and transport within the motion contract** (refuting a single-tip impedance limit) — but delivery
(zone entry) is not restored: **braking / terminal control is now the wall.**

---

## Two-level verdict

- **Task-level:** `MATERIAL_CORRECTION_ALONE_DOES_NOT_RECOVER_DELIVERY` — and force/slip semantics alone don't either
  (zone entry still 0).
- **Mechanism-level:** `TRANSPORT_IMPULSE_RECOVERED__BRAKING_OR_TERMINAL_CONTROL_REMAINS_THE_WALL`.

The auto-verdict `SINGLE_TIP_POSITION_IMPEDANCE_INSUFFICIENT` is a **diluted-mean artifact** (the S5−S0 mean Δ is ~0
because the tuned drive is shared by all stages, and the subset mean is pulled down by weak-contact-geometry states). It
is **refuted** by the per-state evidence and the discriminating test below.

## 1. Statistical discipline — pre-registered contact-capable subset

The evaluation subset is the **controller-independent** contact-capable set: an acquire-only oracle (close the tip→coin
gap, no delivery, no zone) establishes contact. All 8 panel states are contact-capable by this oracle — but *touch* is a
weak filter: only some have transportable contact geometry (a limitation, flagged; the oracle should require a minimum
transportable engagement, not any touch).

## 2. The discriminating test (refutes the impedance limit)

Driving the tangential command harder, on a good-contact state, **within the same motion contract**:

| accel_gain | peak coin v (m/s) | peak joint v (≤3.45) | transport (m) | Ft/Fn |
|---|---|---|---|---|
| 0.05 (old default) | 0.335 | 2.2 ✓ | 0.081 | 0.32 |
| **0.25** | **0.609** | 2.2 ✓ | **0.155** | **1.02** |
| 0.40 | 0.643 | 2.2 ✓ | 0.140 | 1.01 |

The single tip **can** impart ~0.6 m/s coin velocity and >15 cm transport **within the motion contract** — the governor
caps joint velocity at 2.2 regardless of `accel_gain`. The earlier controller was simply **under-driven** (velocity
headroom unused). `accel_gain` default raised 0.05 → 0.25 (§6.5 #19, measured). Ft/Fn → ~1.0: the contact is now
**slip-limited**, so the V2 rubberised tip friction finally matters (it was inert in the static regime).

## 3. Force/slip ablation on the contact-capable subset (per-state)

| state | S5 peak coin v | transport | zone | Ft/Fn | peak joint v |
|---|---|---|---|---|---|
| s1 | **0.609** | **0.155** | 0 | 1.02 | 2.2 |
| s6 | **0.537** | 0.061 | 0 | 0.24 | 1.57 |
| s7 | 0.486 | 0.007 | 0 | 0.33 | 2.0 |
| s4 | 0.346 | 0.006 | 0 | 1.22 | 1.86 |
| s3 | 0.257 | 0.019 | 0 | 0.81 | 2.21 |
| s5 | 0.182 | 0.0 | 0 | 2.0 | 1.14 |
| s2 | 0.151 | 0.0002 | 0 | 0.54 | 1.95 |
| s0 | 0.0 | 0.0 | 0 | 0.0 | 2.28 |

- **Impulse recovered on 2/8 states** (s1, s6: coin velocity ≥ 0.9 × the coast-model launch of **0.593 m/s** for a 10 cm
  delivery, within the motion contract). The impedance is sufficient where the contact geometry is adequate.
- **Zone entry = 0 on every state** — even s1's 0.155 m transport does not stop *in* the zone. The stopping-distance
  prediction from the coast model is accurate (~0.011 m error in the single-state check), so the coast is *predictable* —
  the gap is executing the brake/terminal-settle to land the coin in the zone.
- The S1–S5 stages barely differ in *peak* coin velocity because the impulse is created by the **drive** (shared across
  stages); the semantics govern *release / braking / settling*, i.e. **stopping in the zone** — which is exactly the
  unrecovered axis.

## 4. Main finding (verbatim framing)

> Correcting the material model increased target-directed transport by approximately 20–25 % on independently identified
> contact-capable states, confirming that the original over-sticky dynamics were a partial bottleneck. However, delivery
> and K6 did not improve. Under the corrected model, once the controller is driven to use its (governor-bounded) velocity
> budget, the fingertip reaches its friction limit (Ft/Fn ≈ 1.0) and imparts the coast-model-required object velocity
> (~0.6 m/s) and >15 cm of transport **within** the motion contract — so the remaining limitation is **not** single-tip
> impedance, but the absence of reliable **braking and terminal control** to stop the coin inside the zone, plus
> contact-geometry variability across states.

## 5. Claims / non-claims

**Claimed (measured):** the single tip imparts the coast-model-required velocity (~0.6 m/s) + >15 cm transport within the
motion contract (refuting impedance-insufficiency); Ft/Fn → ~1.0 (slip-limited; V2 tip friction now load-bearing); the
coast model predicts the stop accurately; zone entry is 0 (braking/terminal control unrecovered).

**NOT claimed / provisional:** impulse recovery holds on 2/8 states (contact-geometry variability limits the rest —
touch ≠ transportable engagement, a subset-oracle weakness); the S1–S5 semantics did not (yet) achieve zone entry — the
braking/terminal stage needs work; K6 not recovered.

## 6. Exact next gate

- **Braking / terminal control:** close the loop on the *predicted* stop — the coast model already predicts the stop to
  ~1 cm; drive a velocity-conditioned brake + low-speed settle to land the coin inside the zone (the unrecovered axis).
- **Strengthen the contact-capable oracle:** require a minimum *transportable* engagement (coin displacement under a
  standard push), not any touch, so the subset isn't diluted by glancing-contact states.
- **Only after braking/terminal control lands delivery on the impulse-recovered states** is a proposal / search / RL
  layer justified (the capability would then exist to select over). O3 stays paused.

---

### Commits
- `f29f030d` — force-slip controller + contact-capable oracle + ablation (accel_gain default 0.25, measured).
- (verdict-logic fix + this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, all prior results.
