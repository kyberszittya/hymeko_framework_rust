# Humanoid walking feasibility — rigorous scoping of the stepping wall

**Date:** 2026-07-29
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Question (user):** the humanoid balances + is certified — will it *walk* (reach a goal, like the AIBO)?
**Answer (measured):** not with the current model + quasi-static control. Walking needs a genuine dynamic
(DCM / capture-point) gait controller with a *committed single-support ballistic phase*. This report is
the measurement trail that scopes exactly why and what it takes. **No walking claim is made.**

## Method

All probes drive the emitted humanoid (nq/nv/nu = 23/22/16) at production scale via `HumanoidBalanceEnv`,
computing torques directly (as the PMP/energy-shaping controllers do) and measuring per-foot contact load
(`mj_contactForce`), swing-foot clearance, CoM, ZMP (`hymeko_control.stability`), and uprightness.
A "clean step" requires the swing foot to **unload** (load → ~0 N) AND **clear** (≥ ~0.03 m) while
**upright**, repeatably.

## Findings (8 probes, convergent)

1. **Naive flex (no weight shift):** swing-foot rise +0.000 m (prior result, reconfirmed) — flexing a
   loaded leg just squats the pelvis.
2. **Open-loop weight-shift then lift** (default action, delta_scale 0.4): best clearance +0.005–0.007 m,
   foot stays loaded (300–650 N). Insufficient.
3. **Larger action range** (delta_scale 0.8 / 1.2): foot-*z* rises 0.033 / 0.069 m but load does **not**
   drop (779 / 1244 N) and motion destabilises — the rise is body tilt, not a clean unloaded lift.
4. **Closed-loop CoM-force transfer + lift:** DID unload the swing foot (2 N) and clear 0.054 m — but the
   time trace shows the CoM drifting off the stance foot (ZMP margin stays negative): a **gentle slow tip**,
   not stable single support.
5. **Static lateral lean** (symmetric hip-abduction + ankle-roll, direct joint targets): CoM-*y* moves only
   +0.007 m (need +0.09 to reach over a foot); foot never unloads. **Static transfer is kinematically
   resisted by the double-support closed chain.**
6. **CoM-*y* Jacobian per joint:** `hip_l_ab`/`hip_r_ab` carry the lateral authority (∂com_y/∂q = +0.085,
   same sign); ankle-roll ≈ 0.0007 (negligible). But the closed double-support chain attenuates the
   realised shift to ~⅓ of the free-body Jacobian → static reach over a foot is not attainable.
7. **Dynamic marching** (sinusoidal hip-abduction rock + phased lift): stays upright but **0 clean lifts**
   — sinusoidal rocking alone never unloads a foot. Narrowing the stance (HIP_Y 0.09 → 0.05, a one-line
   change thanks to the parametric model) did **not** unlock it either.
8. **Alternating closed-loop weight-shift march:** feet *do* transiently unload (0 N) but clearance stays
   small (~0.01–0.04 m) and uprightness **degrades across cycles** (1.00 → 0.68) — not a stable limit cycle.

## Diagnosis

The binding limit is **not** stance width, action range, or the certificate — it is that every controller
tried keeps **bilateral quasi-static support** (gravity compensation + stiff PD-hold keep *both* feet
loaded). A real step must **commit to the single-support ballistic phase** — exactly the user's earlier
insight: *"a step is temporarily unstable but stable within a Lyapunov region."* Probe 4 shows that when a
foot does unload, the CoM enters a gentle, slow, recoverable tip (com_y drifts ~0.014 m over ~2 s, upright
0.99) — that **is** the ballistic phase. The missing ingredient is **catching** it: placing the swing foot
at the capture point ξ (which `hymeko_control.stability.capture_point` already computes) before the tip
leaves the recoverable region, then transferring support and repeating. Hand-tuned quasi-static rocking
does not converge to this; it needs the principled DCM/capture-point pattern with timed foot placement.

## What a walking controller requires (concrete plan)

1. **Frontal-plane DCM rock** — drive the lateral CoM as an inverted-pendulum limit cycle; lift the swing
   foot during its unloaded half-cycle (commit to single support).
2. **Capture-point foot placement** — place the swing foot at ξ (+margin) to catch the tip and re-establish
   support — the certificate already provides the target and the hold/step/fall decision boundary.
3. **Sagittal stepping** — add a forward swing component to translate; goal-steer the footstep sequence
   (the AIBO goal-reaching analogue).
4. Likely an **action-space change** (a stepping-oriented interface / re-tuned authority beyond the
   balance-residual servo) — a first-class model change (CLAUDE.md §11) to be escalated before committing.

This is a genuine multi-stage bipedal-gait build, not a tuning tweak. The foundation (certified balance,
Vukobratović ZMP, capture point + capturability, full anthropomorphic DOF, parametric model) is in place;
the gait is the next focused project.

## BREAKTHROUGH (action-space change approved) — resonant rocking unlocks weight transfer

The earlier "dynamic marching failed" (probe 7) used the **wrong frequency**: T = 100–180 steps ≈ 0.1–0.18 s
(~6 Hz), while the lateral LIPM pendulum period is 2π√(z/g) = 2π√(0.645/9.81) ≈ **1.6 s** — I was rocking
~10× too fast, so no sway amplitude built up. Rocking at the correct timescale changes everything:

- **Resonant lateral rocking** (hip-abduction sinusoid, T ≈ 400–2000 steps): the CoM sway builds to
  **±0.04–0.067 m** and **each foot FULLY unloads (0 N)** at its extreme, while staying upright. The
  weight-transfer bootstrap that quasi-static control could not do (probes 1–12) **works dynamically** at
  the pendulum timescale. This is a solid, reproducible primitive.
- **+ phased lift** of the unloaded foot → **clean alternating stepping**: unloaded (< 25 N) foot lifts
  with ≥ 0.03 m clearance, 4–9 clean steps.
- **Remaining gap — limit-cycle stability:** the march is not yet a stable indefinite cycle; it falls after
  ~4–9 steps. Fall-mode diagnosis: **sagittal (pitch)**, not lateral — the lateral rock stays stable
  (roll ±3–8°), but the hip-flexion lift reacts against the pelvis and the sagittal CoM drifts
  (com_x +0.14 → −0.33 m, pitch → 57°). Knee-dominant lift + an ankle-strategy sagittal balancer extend it
  (to ~9 steps) but do not fully stabilise it — additive task-space forces have weak authority under the
  double-support/contact constraints and fight each other.

**Status: clean stepping achieved; a stable indefinite gait needs a proper whole-body controller** (QP task
prioritisation + contact-consistent dynamics + capture-point foot placement to arrest the sagittal drift).
That WBC is the next focused build. No unstable controller is shipped (no broken feature per CLAUDE.md).

## Provenance

- 8 measurement probes, seed 0, deterministic; env rebuilds from `data/robotics/humanoid.hymeko` (validated,
  HIP_Y reverted to 0.09 after probe 7). No committed code/model changed by this investigation (measurement
  only). Shared venv `hymeko_framework_rust/.venv`, macOS 25.5.0.
