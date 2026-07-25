# RUBBER_TIP_LOW_DRAG_COIN_V2 — material freeze + V1→V2 controller re-run

**Date:** 2026-07-25
**Branch:** `feat/architectural-assimilation-v1`
**Direction (user, 8 points):** preserve V1; build V2 as a new scenario; calibrate coast at multiple velocities; don't treat
viscous ≡ Coulomb; freeze the material model without inspecting delivery; re-run frozen controllers without retraining;
only then test force/slip-aware semantics; keep O3 paused.
**One-line outcome:** the mis-calibrated contact physics was a **partial** bottleneck — on contacting states the calibrated
V2 material improves transport ~40–50 %, but delivery is **not** restored: the remaining limit is the controller imparting
too little coin velocity (impulse) at realistic arm speed. `PHYSICAL_MODEL_BOTTLENECK_PARTIALLY_REMOVED +
OPTION_SEMANTICS_REMAINING_GAP` — which now justifies force/slip-aware semantics.

---

## 1. Multi-velocity coast calibration — viscous vs Coulomb (not interchangeable)

μ_eff from the early deceleration over an unobstructed window, at v0 ∈ {0.2, 0.5, 1.0, 1.5}:

| drag model | μ_eff @ 0.2 → 1.5 m/s | signature |
|---|---|---|
| **as-loaded viscous 2.5** | **0.39 → 1.72** | VISCOUS (μ∝v0), ~15× too sticky |
| pure low viscous 0.1 | 0.03 → 0.29 | still VISCOUS (μ∝v0) |
| **dof_damping 0.02 + Coulomb frictionloss 0.074** | **0.154 → 0.209** | Coulomb-dominant, ~flat, **in band** |

Viscous damping (force ∝ v) gives a *speed-dependent* μ_eff; Coulomb `dof_frictionloss` (≈ constant force) gives a *flat*
μ_eff — a hard coin on a smooth table is Coulomb with μ ≈ 0.1–0.2. Low viscous damping alone does **not** fix the
speed-dependence; the Coulomb term does.

## 2. V2 freeze (delivery-blind)

```
RUBBER_TIP_LOW_DRAG_COIN_V2_FROZEN
```
- **tip↔coin friction 2.0** (rubberised finger). **Effective friction VERIFIED = 2.0** by reading `d.contact.friction`
  after the priority combination — the requested separate check that `geom_priority` produces the intended contact
  friction (not the max-combination, not an explicit `<pair>` which exploded contact stiffness).
- **coin↔floor drag**: `dof_damping 0.02` (numerical residual) + **Coulomb `dof_frictionloss 0.074`** (μ_eff ≈ 0.15, flat).
- **mass FIXED**; **V4_INTERMITTENT_CONTACT motion contract unchanged**. K6 / zone never inspected during the freeze.
- `SINGLE_TIP_LOW_FRICTION_COIN_V1` (as-loaded) untouched.

## 3. V1 → V2 re-run (same controllers, NO retraining, 8 states)

| controller | transport V1→V2 | zone V1→V2 | K6 V1→V2 | coin_pk V1→V2 |
|---|---|---|---|---|
| searched legacy expert | 0.027 → 0.029 | 0.375 → 0.375 | 0.375 → 0.375 | — |
| C1 closed-loop | 0.020 → 0.024 | 0.0 → 0.0 | 0.0 → 0.0 | 0.185 → 0.224 |
| C2 intermittent | 0.020 → 0.025 | 0.0 → 0.0 | 0.0 → 0.0 | 0.176 → 0.217 |

Panel-mean deltas are small **because most states are no-contact / hard** (they dilute the mean). On the **contacting**
states the improvement is real: e.g. state s1 — C1 transport 0.058 → 0.082 (**+41 %**), C2 0.058 → 0.086 (**+50 %**),
searched-legacy 0.079 → 0.091. So the calibrated physics **does** help transport where contact occurs.

**But delivery is not restored:** zone entry and K6 are unchanged for every controller (searched-legacy delivers on the
same 0.375 of states in V1 and V2; C1/C2 reach the zone on none).

## 4. The deeper limit (now visible)

- Coin peak speed stays **~0.2 m/s** and **Ft/Fn ~0.5** even with verified tip μ = 2.0 — the tip↔coin contact is **not
  slip-limited** (static regime): the position-controlled tip pushes the coin *kinematically*, transferring force below
  the friction cone. High tip friction therefore does not help, and the coin never gets a real velocity.
- With coin speed ~0.2 m/s, even the realistic low drag (μ ≈ 0.15) only adds ~1–2 cm of coast (v²/2μg). Transport is
  **push-limited, not coast-limited**.
- This matches the legacy-impact reconstruction: the legacy "success" was a high-speed **impulse** (27 rad/s arm → coin
  ~1.5 m/s → long throw), which the realistic-speed controllers do not produce.

```
VERDICT: PHYSICAL_MODEL_BOTTLENECK_PARTIALLY_REMOVED  +  OPTION_SEMANTICS_REMAINING_GAP
```
(The auto-verdict `MATERIAL_CORRECTION_INSUFFICIENT__DEEPER_LIMIT` is the *panel-mean* reading; the per-state contacting
evidence upgrades it to "partially removed" — the honest, per-state framing.)

## 5. Claims / non-claims

**Claimed (measured):** the as-loaded coin drag was viscous and ~15× too sticky; the calibrated Coulomb-dominant V2 model
is realistic and flat across speed; effective tip friction is verified = set; V2 improves transport on contacting states
(~40–50 %) with no retraining; delivery (zone/K6) is unchanged.

**NOT claimed / provisional:** the panel-mean transport delta is small (diluted by no-contact states); the "controller
imparts too little coin velocity" limit is **inferred** from coin_pk ~0.2 m/s + Ft/Fn ~0.5 + unchanged zone entry, not
from a single isolating experiment; the searched-legacy lacks contact-force decomposition (Ft/Fn shown 0 — not measured).

## 6. Exact next gate — force/slip-aware option semantics (now justified)

The physics is now realistic *and* it is no longer the (sole) wall — so the option semantics the user named are the right
next step, on frozen V2, no retraining:
- **bounded normal preload** → **target-directed tangential force** (drive the contact toward the friction cone, i.e. make
  the tip↔coin contact actually *slip* to transfer force) → **slip detection** → **controlled impulse** (impart real coin
  velocity) → **coast estimation** → **re-contact braking** → **terminal settling**.
- If that restores delivery on V2 → the coin task is solvable once physics + force/slip semantics are both right.
- If it does not → the single-tip embodiment limit, and only then is a geometry change (grasp benchmark) warranted.
- **O3 stays paused** until the corrected-physics coin has a capability verdict.

---

### Commits
- `55c5e348` — multi-velocity coast calibration.
- `8abefde0` — V2 freeze (delivery-blind, effective-friction verified) + re-run harness.
- (re-run hook fix + result) + this report — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, V2/V3/V4 dynamics contracts, `COIN_LEGACY_FAST_V1`, all prior results.
