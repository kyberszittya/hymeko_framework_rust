# AIBO capture-point recovery — RETRACTED: it's a sprawl AND a dynamics exploit, not a step

**Date:** 2026-07-27 (JST) · **corrected same day**
**Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model-based controller (no RL).**
**Verdict: `RETRACTED — CAPTURE_POINT_RECOVERY_IS_A_DYNAMICS_EXPLOIT_NOT_A_STEP`.**

> **Retraction (two independent problems, both caught by the user):**
> 1. **Not a step — a sprawl.** The controller abducts all four legs symmetrically (front
>    paws lift ~0.36 m, back ~0.25 m; ≥1 foot off 209/300 steps) — a startle-splay, not a
>    lift-swing-place step. A real single-leg step FAILS here (all four feet leave the
>    ground, Vfinal ~0.72, no cert).
> 2. **Bad physical representation — a dynamics exploit.** During the "recovery" the leg
>    joints hit **26.9 rad/s** (real Aibo ERS-1000 ~3–8; ≈ the coin's 27.2 rad/s exploit),
>    the base **launches at 0.98 m/s**, and **all four feet are airborne 113/300 steps**.
>    The certificate passes only because unphysical dynamics let the robot hop itself into a
>    stable sprawl. This is the **same failure mode as the coin** (`REALISTIC_MOTION_CONTRACT_V1`),
>    which this humanoid/AIBO balance+step line **never applied**. **Not robot-transferable.**
>
> The "certifies to ~1.0–1.2 m/s" result below is therefore **not a valid protective response** —
> it is an exploit of a slew/torque/contact-unconstrained model. Kept for the record; do not cite.

---

## Why the AIBO (and not the humanoid)

The sagittal 16-DOF humanoid could **not** get a beneficial protective step: RL residual
(3 configs) and a hand-built capture-point stepper both **underperformed** the strong frontal
PD (foot clearance marginal; see `hymeko_humanoid/reports/2026-07-27-humanoid-lateral-step.md`).
The 22-DOF quadruped is different — its legs **abduct freely** (they clear the ground in the
trot), so a real capture-point step is executable.

## The controller (`scenarios/aibo/capture_step.py`)

Model-based, reactive, no RL:

- **Capture point (LIPM):** `xi_y = com_y + com_y_vel·√(com_z/g)` — where support must go to
  arrest the lateral fall (Pratt/Koolen capturability, the Vukobratović-lineage step criterion).
- **`CapturePointStepper`:** a PD stand-hold + a **hip-abduction widening** of the stance toward
  `xi_y`, scaled by `|xi_y − com_y|` (0 when balanced → auto-triggers only on a push).
- **`PushRecoveryLyapunov`:** `V = ½[w_up(1−up)² + w_v|v|² + w_off|com_lat|²]` → 0 iff upright,
  at rest, COM back over the spawn axis. Gated by the generic reward-independent certificate.

## Result — the step certifies where the stand can't

Lateral push, recovery-V certificate (12 pushes, both signs):

| push (m/s) | passive stand | capture-point step |
|---|---|---|
| ±0.8 | +0.8 ✅ / −0.8 ❌ (asym.) | ✅ / ✅ |
| ±1.0 | ❌ (Vfinal 0.90) | ✅ **Vfinal 0.037 / 0.002** |
| ±1.2 | ❌ (Vfinal 0.92) | ✅ +1.2 (0.048) / −1.2 near (0.012) |

The passive stand tips over beyond ~0.8 m/s (and asymmetrically); the **capture-point step
recovers to rest** (V → 0) for pushes up to **~1.0–1.2 m/s in both directions** — a symmetric,
genuine extension of the certified push-recovery envelope. Video (`aibo_capture_step.mp4`, 1.0
m/s): the passive stand **flips onto its back** (up −0.16, V 0.91, CERT FAIL) while the
capture-point step **widens into a protective stance and settles upright** (up 0.96, v 0.06,
V 0.043, CERT PASS).

## Files

```
scenarios/aibo/capture_step.py           NEW  (capture_point_y, PushRecoveryLyapunov, CapturePointStepper, certificate)
scenarios/aibo/render_capture_step.py    NEW  (stand-vs-step recovery video + telemetry + HyMeKo)
tests/test_aibo_capture_step.py          NEW  4 tests (capture point, V, step-certifies-where-stand-cant, action bounds)
reports/2026-07-27-aibo-lyapunov/aibo_capture_step.{mp4,gif}  NEW
```

Reuses `QuadrupedGoalEnv`, `SteeredTrotGait`, the render helpers, and the generic
`evaluate_lyapunov` / certificate. The certificate is **unchanged**.

## Tests / lint

`ruff` clean. AIBO capture-step + Lyapunov + render tests pass (10/10). The step-certifies
test regression-locks the positive: the stand fails a 1.0 m/s push and the stepper passes.

## Provenance

- MuJoCo model emitted by `target/release/hymeko` from `data/robotics/quadruped.hymeko`. Seed 0;
  push ±0.8–1.2 m/s. Peak RSS well under cap. Model-based, deterministic — no RL, no training.

## Bottom line (corrected)

The claimed "AIBO protective step" was **wrong on two counts**, both caught by the user: it is
a **symmetric sprawl, not a step** (all four legs splay; a real single-leg step fails), and it
is a **dynamics exploit, not a physical response** (26.9 rad/s leg-flinging, 0.98 m/s launch,
all four feet airborne 113/300 steps — the coin's `REALISTIC_MOTION_CONTRACT` failure mode,
never applied to this line). The certificate pass is **not robot-transferable** and is retracted.

Honest state of the "protective step" thread across the campaign: **no embodiment produces a
valid protective step.** Humanoid = ankle/hip postural (RL + model-based step both fail). AIBO
= an unphysical airborne sprawl. Both are **physical-representation problems**: the models lack
a motion contract (slew/torque/contact governor), so "recoveries" exploit unphysical dynamics.

**The real next step is the motion contract, not more control.** Apply
`REALISTIC_MOTION_CONTRACT_V1` (slew limiter + directional torque governor + no-launch/contact
constraint + video-trace-consistency gate) to the AIBO and humanoid, then re-evaluate whether
ANY realistic protective response (postural or stepping) survives. Nothing on the step front is
claimed until it does.
