# Contact-scheduled hop MPC — a PLANNED flight inside the Lyapunov region (AIBO + human)

**Date:** 2026-07-28 (JST)
**Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model-based trajectory optimisation (no RL).** · **Verdict: `PLANNED_BALLISTIC_HOP_STAYS_IN_CAPTURABILITY_REGION`.**

---

## The idea (the user's, and it's the right one)

The retracted capture-widening went airborne as an **exploit** — uncontrolled, 27 rad/s, the
certificate passing by accident of unphysical dynamics. A **planned** flight phase is the
opposite: contact-scheduled MPC *deliberately* leaves the ground for a ballistic flight, yet keeps
the centroidal state **within a recoverable (capturability) Lyapunov region**. The momentary loss
of *static* stability is bounded and recovered by a controlled landing — exactly how real jumping/
running robots (and people) move. Same reduced model for the AIBO and the human — they differ only
by mass / stand height / force limits.

## What we built (`scenarios/aibo/hop_mpc.py`)

Centroidal model: state `x = [px, vx, pz, vz]`, control = ground reaction force `F = [Fx, Fz]`
during **stance**, `F = 0` during **flight**. Dynamics `v̇x = Fx/m`, `v̇z = Fz/m − g`. A forward
hop (crouch → launch → FLIGHT → land) is planned by single-shooting trajectory optimisation:

    min  Σ dt·‖F‖²  +  w·‖x_N − x_goal‖²
    s.t. F = 0 during the (scheduled) flight;  Fz ∈ [0, F_max];  |Fx| ≤ μ·Fz (friction cone);  pz ≥ 0

and the **capturability Lyapunov** `V_cap = ½‖ξ − p_foot‖²`, `ξ = px + vx·√(pz/g)`, is verified to
stay bounded — the flight is a controlled excursion, not a fall.

## Result (both embodiments)

| | reaches target | at rest | flight | apex liftoff | V_cap (max → final) | friction |
|---|---|---|---|---|---|---|
| **AIBO** (m=2, z0=0.23) | x = 0.250 ✅ | vx≈0, vz≈0 ✅ | **F = 0 (ballistic)** | +0.65 m | **0.031 → 0.000** | ✅ |
| **human** (m=15, z0=0.645) | x = 0.240 ✅ | ✅ | F = 0 | +0.65 m | 0.031 → 0.000 | ✅ |

The COM **leaves the ground** on a ballistic arc (`hop_mpc.png`, left panel — the parabolic flight),
`Fz` drops to **exactly zero during the flight** (middle panel), and the **capturability Lyapunov
stays far below the recoverable-region threshold (0.1) and decays to 0** (right panel). The
trajectory is mass-independent given the schedule; the force scales with mass (AIBO ~32 N, human
~240 N) — physically correct. Friction-cone- and force-limit-feasible throughout.

## Files

```
scenarios/aibo/hop_mpc.py       NEW  (HopParams, CentroidalHopMPC: schedule + rollout + capture_lyapunov + plan)
scenarios/aibo/render_hop.py    NEW  (matplotlib plot: COM arc + Fz + V_cap, AIBO & human)
tests/test_aibo_hop_mpc.py      NEW  6 tests (reaches target at rest, ballistic flight, leaves ground, V_cap bounded, friction, human)
reports/2026-07-27-aibo-hop/hop_mpc.png  NEW
```

New dependency used: `scipy.optimize` (already present), `matplotlib` (already present).

## Tests / lint

`ruff` clean. 6/6 hop-MPC tests pass, locking: the hop reaches the target at rest, the flight is
ballistic (`F = 0`), the COM leaves the ground, the capturability Lyapunov stays bounded (< 0.1)
and recovers (< 0.02), forces are friction-feasible, and the human embodiment plans the same hop.

## Bottom line

This resolves the airborne question the honest way: a **planned, contact-scheduled ballistic hop**
that leaves the ground yet stays inside the **capturability Lyapunov region** — the deliberate,
bounded loss of *static* stability that dynamic locomotion needs, not the retracted exploit. It is
the centroidal core of whole-body MPC, demonstrated identically for the AIBO and the human (mass
only scales the force). The next layer is the **full-body realisation** — map the planned ground
reaction force to joint torques through the contact Jacobian, under the motion contract — turning
the plan into an executed jump on the 22-DOF / 16-DOF models.
