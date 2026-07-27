# AIBO model-based capture-point protective step (lateral push recovery) — POSITIVE

**Date:** 2026-07-27 (JST)
**Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model-based controller (no RL).** · **Verdict: `CAPTURE_POINT_STEP_RECOVERS_LATERAL_PUSH` — certifies to ~1.0–1.2 m/s where the stand fails at ~0.8.**

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

## Bottom line

The capture-point protective step that was **not** achievable on the sagittal humanoid **is**
achievable on the 22-DOF AIBO: a model-based hip-abduction widening toward the LIPM capture
point recovers the quadruped from lateral pushes (certified to ~1.0–1.2 m/s, both signs) that
tip the passive stand. This is the honest resolution of the "protective step" thread — the
step lives where the kinematics support it (the quadruped), and the humanoid negative stands.
A residual RL layer over this certified scaffold (coin-R8) is the natural next step.
