# Simulating running — a periodic hopping gait (centroidal MPC limit cycle)

**Date:** 2026-07-28 (JST)
**Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model-based trajectory optimisation (no RL).** · **Verdict: `RUNNING_LIMIT_CYCLE_WITH_BOUNDED_CAPTURABILITY`.**

---

## The idea

Running = a cyclic `[stance push → ballistic FLIGHT → stance …]` advancing at steady speed. It
extends the single hop to a **periodic limit cycle**: one stride is planned (single-shooting
trajectory optimisation with a **periodicity constraint** on `[vx, z, vz]`) and repeated to
simulate a continuous run. The vertical state bounces periodically, the horizontal advances at the
target speed, and the **capturability Lyapunov stays bounded every stride** — the orbital stability
of running. Same centroidal model / friction-feasible forces as the hop; embodiment-agnostic.

## Result (`scenarios/aibo/running_mpc.py`)

5 strides at a 0.6 m/s target:

| | forward | steady speed | flight fraction | z bounce | V_cap max | peak Fz | friction |
|---|---|---|---|---|---|---|---|
| **AIBO** (m=2, z0=0.23) | **1.56 m** | 0.60 m/s | **0.54** (airborne 54 %) | [0.111, 0.289] | **0.030** (< 0.1) | 43 N | ✅ |
| **human** (m=15, z0=0.645) | 1.56 m | 0.60 m/s | 0.54 | [0.484, 0.663] | 0.046 | 319 N | ✅ |

`running_mpc.png`: the COM height is a **periodic bounce** with the flight phases shaded; the
forward position is a **straight steady climb**; the capturability Lyapunov is a **bounded sawtooth**
(rises in stance, drops in flight, stays under the 0.1 recoverable line every stride). A real run —
54 % flight phase, periodic vertical bounce, steady forward speed — with the loss of static stability
**planned and bounded** each stride. Identical kinematics for the AIBO and the human (mass only
scales the force: 43 N vs 319 N).

## Files

```
scenarios/aibo/running_mpc.py       NEW  (RunningGaitMPC: periodic stride plan + N-stride simulate + capture_lyapunov)
scenarios/aibo/render_running.py    NEW  (matplotlib: COM bounce + forward + V_cap, AIBO & human)
tests/test_aibo_running_mpc.py      NEW  6 tests (steady speed, real flight phase, periodic bounce, V_cap bounded, friction, human)
reports/2026-07-27-aibo-hop/running_mpc.png  NEW
```

## Tests / lint

`ruff` clean. 6/6 running tests pass, locking: steady forward speed ≈ target, a real flight phase
(> 30 % airborne), a periodic vertical bounce (rises above + crouches below standing), the
capturability Lyapunov bounded (< 0.1) every stride, friction-feasible unilateral forces, and the
human embodiment runs too.

## Full-body realisation — attempted, and it hits the model's physical limit (honest)

Mapped the planned ground reaction force to the AIBO's legs via each foot's contact Jacobian
(`τ = Σ J_footᵀ·F/4`, applied during stance, under the motion contract, planned for the real 5.71 kg
mass). Result: the torso rises only **+0.019 m** (planned +0.65 m) and **never leaves the ground**
(0/60 airborne) — at **realistic joint speeds (3.2 rad/s, not an exploit)**. The legs simply cannot
extend fast/far enough to launch the body: the ground-reaction *impulse* is available in the
centroidal model, but the **short-legged full-body model cannot transmit it into a flight** (the same
short-leg wall that blocks the protective step). So the **execution layer is physically insufficient
on this model, not exploity** — a real jump needs longer / springier legs or a higher-fidelity model.

## Bottom line

Running is simulated as a **certified centroidal limit cycle** — a periodic hopping gait with a
ballistic flight per stride, steady forward speed, capturability bounded every stride, for both the
AIBO and the human. That **planning layer is clean and certified** and realises the user's insight (a
*planned*, bounded loss of static stability). The **full-body execution**, however, **hits the
simplified model's physical wall** — the AIBO's short legs launch the body only +2 cm (never
airborne), realistically but insufficiently. Honest split: *planning* works; *execution* needs a
model that can actually jump. The Hamiltonian/optimal-control arc (energy-shaping → IDA-PBC → PMP →
capturability MPC) is complete at the level the models support.
