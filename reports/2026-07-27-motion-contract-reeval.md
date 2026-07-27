# Realistic-motion contract — re-evaluation of the balance/step line (no 27 rad/s motors)

**Date:** 2026-07-27 (JST)
**Branches:** `research/aibo-lyapunov-ph` (AIBO) · humanoid measured in `hymeko_humanoid`
**SIMULATION.** · **Verdict: humanoid balance is REALISTIC; the AIBO recovery was the exploit — tamed by the contract, still a sprawl.**

---

## Why

The user caught that the AIBO "capture-point step" flung its legs at **26.9 rad/s** (no such
motor exists; ≈ the coin's 27.2 rad/s exploit) and went airborne — a bad physical
representation. The whole humanoid+AIBO balance/step line was built without the coin's
`REALISTIC_MOTION_CONTRACT`. So: apply a motion contract (cap joint speeds realistically) and
re-check every "certified" result.

## The contract (`scenarios/aibo/motion_contract.py`)

`JointVelocityGovernor` — a **directional velocity governor** (the coin principle, rebuilt
scenario-side since the coin module is in the off-limits main tree): zero the *accelerating*
action of any joint at `|v| ≥ v_max`, **preserve braking**. `v_max = 8 rad/s` (realistic
servo ceiling). Wraps any controller's action before stepping.

## Result 1 — the humanoid certified balance is REALISTIC (not an exploit)

Measured joint speeds during the committed humanoid balance results:

| controller | max joint speed | verdict |
|---|---|---|
| PD-hold nominal balance (a=0) | **1.5 rad/s** | realistic ✅ |
| sagittal residual (the 12/12 positive) | **1.8 rad/s** | realistic ✅ |

The humanoid's **position-servo** action space (kp=60, kv=10, τ_max=150) inherently limits
velocity — it was *inadvertently a motion contract*. So the sagittal certified balance +
residual-extends-envelope result (12/12) **survives the realism check** and stands. Only the
AIBO's direct-torque gait actuation (no velocity limit) exploited.

## Result 2 — the AIBO recovery was the exploit; the contract tames it (but it's still a sprawl)

AIBO capture-widening under a 1.0 m/s push:

| motion contract | CERT | Vfinal | max joint speed | all-4 airborne |
|---|---|---|---|---|
| none | ✅ | 0.037 | **26.9 rad/s** | 113/300 |
| v_max = 8 | ✅ | 0.046 | 11.5 rad/s | 43/300 |
| v_max = 6 | ✅ | 0.033 | 9.7 rad/s | 24/300 |

The governor cuts the actuator-driven exploit (26.9 → ~10 rad/s, airborne 113 → 24). The
residual ~10 rad/s peak is from **landing impacts** (the widening still slightly launches —
contact forces the governor can't cap). Two honest facts remain:

- The recovery **still certifies under the contract** — so it is *not purely* an exploit; a
  velocity-constrained widening does recover the COM. But
- it is **still a sprawl, not a step**, and **still slightly airborne** — not a valid protective
  step. Full realism needs also a lower torque authority / no-launch constraint.

## Files

```
scenarios/aibo/motion_contract.py        NEW  (JointVelocityGovernor — directional, braking-preserving)
scenarios/aibo/render_capture_step.py    +governor applied; telemetry shows max_jointspd; honest title
tests/test_aibo_motion_contract.py       NEW  (governor cuts the exploit; preserves braking)
reports/2026-07-27-aibo-lyapunov/aibo_capture_step.{mp4,gif}  re-rendered under the contract
```

## Tests / lint

`ruff` clean. AIBO motion-contract + capture-step tests pass (6/6). The governor test locks
that the ungoverned widening exceeds 18 rad/s (exploit) and the governed one is far lower.

## Bottom line

Applying the motion contract splits the line cleanly: the **humanoid certified balance is
physically real** (1.5–1.8 rad/s — position-servo = inherent contract) and stands; the **AIBO
capture-widening was a dynamics exploit** (26.9 rad/s, airborne), which the velocity governor
tames to ~10 rad/s while the recovery still certifies — but it remains a **sprawl, not a step**.
No protective *step* is claimed for either embodiment. The realistic-motion contract is now a
first-class citizen of the AIBO scenario; any future step/recovery claim goes through it.
