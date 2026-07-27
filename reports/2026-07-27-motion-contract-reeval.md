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

## Result 3 — the AIBO actually WALKS forward to the goal (the real point, and it's realistic)

The push-recovery sprawl was a distraction. The **actual task** — walk forward to the goal —
the `SteeredTrotGait` already does, and does **realistically**:

| forward | max joint speed | foot lift | airborne | reaches goal |
|---|---|---|---|---|
| 0.71 m | **5.4 rad/s** (<< the 27 exploit; governor barely triggers) | 2–3 cm (low-clearance trot) | never launches | **yes, dist → 0.29 m** |

`aibo_walk_to_goal.mp4`: the AIBO trots forward to the goal marker with a **foot-contact gait
diagram** showing the diagonal stepping (the `fr` row's swing bars interrupt stance), telemetry
showing distance/forward-progress/joint-speed/feet-down, all under the motion contract. It is a
**real walk** — genuine diagonal-trot stepping, realistic joint speeds, feet lifting 2–3 cm, no
launch — not the sprawl and not an exploit. (A slight bounce exists — all feet briefly ~1.5 cm
up — but the feet never clear >3 cm and the robot never launches; regression-tested
`max_lift < 6 cm`, `joint speed < 10 rad/s`, forward > 0.3 m.)

## Result 4 — both follow-ups hit walls (honest, closes the step thread)

- **(a) Refine the walk:** the baseline trot (hip 0.7 / knee 0.3 / freq 1.2) is already
  near-optimal. Raising amplitude/frequency *slows* forward progress (0.58 → 0.39 m) and pushes
  joint speed toward the cap, while foot clearance barely moves (3 → 4.5 cm, limited by the short
  legs). No parameter "refinement" wins — more would need a different gait or learned locomotion.
- **(b) Real single-leg recovery step, under the contract:** still **FAILS** — CERT False
  (Vfinal 0.79–2.05), all four feet still go airborne (max_feet_off 4), joint speed 16 rad/s
  from **landing impacts** (the velocity governor caps actuator torque, not contact forces).
  Lifting a leg during a fast lateral fall destabilizes the support legs — the **same wall as
  the humanoid**. Reactive stepping does not capture the push; only the (retracted) exploit-sprawl
  "recovered." A real protective step needs **whole-body MPC with contact scheduling + a no-launch
  constraint** (or a learned recovery gait) — a focused controls project, unaddressed here.

## Files

```
scenarios/aibo/motion_contract.py        NEW  (JointVelocityGovernor — directional, braking-preserving)
scenarios/aibo/render_capture_step.py    +governor applied; telemetry shows max_jointspd; honest title
tests/test_aibo_motion_contract.py       NEW  (governor cuts exploit; preserves braking; forward-walk realistic)
scenarios/aibo/render_walk_to_goal.py    NEW  (AIBO walks to goal + foot-contact gait diagram, under contract)
reports/2026-07-27-aibo-lyapunov/aibo_capture_step.{mp4,gif}  re-rendered under the contract
reports/2026-07-27-aibo-lyapunov/aibo_walk_to_goal.{mp4,gif}  NEW (the real deliverable: forward walk)
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
