# v2 HOME_RETRACT_OR_PRESHAPE + sag-sensitive phase fix (2026-07-07)

## Summary

Added the `HOME_RETRACT_OR_PRESHAPE` phase (joint-space move from the over-extended `arm_home` into the
probe-found multi-start `cf_hover` seed) and fixed the sag-sensitive phase logic. **Both are logic-only fixes; no
gains, no scene, no `arm_home` change.** The logic fixes **work as designed** — the arm now reaches over the
object and drags the table half as much — but the v2 smoke **still fails** the clearance gate, and the residual
penetration is the ~2 cm wrist **sag** (the deferred gains issue) applied to a grazing preshape transition.

## What changed (controller logic vs safety, kept separate)

- **`HOME_RETRACT_OR_PRESHAPE`**: `_v2_preshape_target` computes (once/episode, cached) a multi-start
  `solve_collision_free` config **at the object hover** (`cf_hover`), falling back to a retracted mid-radius pose
  (`cf_mid_retract`) if `cf_hover` fails to reach. `_v2_preshape_step` moves the **commanded** config there in
  joint space (sag-independent) and latches `_v2_preshaped` on arrival.
- **Sag-sensitive phase fix**: reach-phase progression no longer reads the sagged physical z. Descent is gated on
  the **physical tool XY** being over the object (`_V2_OVER_HORIZ`, sag-independent) — so the physical transit
  **holds the +6 cm clearance hover** instead of descending to grasp_z (only +1.5 cm) while the command races
  ahead. Descend-vs-close keys off the **commanded** z.
- **Separation preserved**: commanded/seed/waypoint state drives *phase progression*; the physical clearance/contact
  harness (`hymeko_rl/eval/pick_clearance.py`) remains the **sole safety authority** — the controller never
  self-certifies clearance.

## Results (v2 smoke, 4 ep, seeds 50000–50003, horizon 620)

| metric | v1 (ref) | v2 (pre-preshape) | **v2 + preshape + hold-hover** |
|---|---|---|---|
| first-over-object | ~150–225 | **None (never)** | **106–175 (reaches)** |
| transit finger↔table rate | 0.45 | 0.94 | **0.52** |
| min clearance (m) | −0.026 | −0.024 | **−0.017 mean / −0.020 min** |
| first strike step | ~51 | ~37 | ~40 |
| lift / place | 1.0 / 0.75 | 0.0 / 0.0 | **0.0 / 0.0** |
| gate | FAIL (dirty) | FAIL | **FAIL** |

Direction is right on every clearance axis, but not passing.

## Residual diagnosis (traced) — sag on a grazing preshape transition

A commanded-vs-physical trace (seed 50000): the **commanded** finger clearance stays **positive (+0.015 m)**
throughout; only the **physical** one penetrates. The physical wrist holds a ~**2 cm steady-state sag** below the
command (physical z 0.234 vs commanded 0.255) even when the command is *held* — a steady-state servo offset =
gains. `first_fingtab≈40` (during the preshape move) < `first_over≈120`, so the **preshape joint-space transition
itself dips**: the probe measured that path's kinematic clearance at ~0 (grazing), and the ~2 cm sag turns the
graze into ~−1.7 cm penetration. **The residual is sag-dominated — the deferred gains issue — not a phase bug.**

## Answers to the reported checklist

- **Changed files**: `hymeko_rl/env/pick_place_env.py` (preshape phase + helpers, sag-sensitive phase logic, 2
  fields in `__init__`/`reset`). No other source touched.
- **Was `ik.py` touched?** **No** (this task reused the existing `fk_tool` + `solve_collision_free`).
- **Did v1 behaviour change?** **No** — v1 re-smoke byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577);
  `test_v2_preserves_v1_default_and_dispatch` green.
- **Preshape seed strategy**: multi-start `cf_hover` (reached the hover for all 4 seeds; `cf_mid_retract` fallback
  not needed).
- **v2 lift / place**: **0.0 / 0.0**.
- **forbidden-pre-object rate**: **1.0**.
- **transit finger/table contact rate**: **0.52** (was 0.94).
- **min physical clearance**: **−0.020 m min / −0.017 m mean** (was −0.024).
- **first-strike vs first-over-object**: strike ~step 40 vs over-object ~step 106–175 — the strike is in the
  **preshape transition**, before the arm is over the object.
- **Is the 32-episode gate justified?** **No** — min physical clearance is still negative and lift/place are 0. Not
  justified until the residual is closed.

## Options to close the residual (your call — none applied)

1. **The deferred gain / tracking fix** for the ~2 cm sag (stronger arm/wrist position gain or wrist
   gravity-comp) — you deferred this; it is the direct cause.
2. **Higher-margin preshape path (logic only)**: route the preshape through `cf_mid_retract` (retract high, then
   descend over the object) so the transition keeps ≥ +6 cm kinematic clearance and tolerates the 2 cm sag. The
   probe measured `home→cf_mid` at +12 cm on the far seeds; may not fully close it alone.

## Gates

ruff clean · pick/ik/v2 tests 15/15 · v1 re-smoke byte-identical. Stopped after the v2 smoke per instruction; the
32-ep gate was **not** run.
