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

## Bottom line

Running is simulated as a **centroidal limit cycle** — a periodic hopping gait with a ballistic
flight per stride, advancing at steady speed, the capturability Lyapunov bounded every stride. This
is the honest realisation of the user's insight (a *planned* loss of static stability inside the
Lyapunov region), now as a continuous run rather than a single hop, for both the AIBO and the human.
The next layer is the **full-body realisation** — mapping the planned ground reaction force to joint
torques through the contact Jacobian, under the motion contract, to execute the run on the actual
22-DOF / 16-DOF models (a marginal ask on the short-legged models; centroidal planning is the part
that is clean and certified here).
