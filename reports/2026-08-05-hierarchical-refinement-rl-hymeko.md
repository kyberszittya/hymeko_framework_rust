# Hierarchical model refinement with RL, as a HyMeKo model hierarchy

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `ee847a2c`)
**The thesis, made concrete:** *from a coarse abstraction, decompose — through a hierarchy — into a more
optimal, detailed model, with RL carrying the policy across the levels; expressed as hierarchical HyMeKo models.*

---

## Summary

This builds the RL thesis the humanoid arc has been circling: start from a **coarse** abstraction (the centroidal
run-stop with a single abstract angular-momentum port, controllable only in stance) and **refine** it into a
**detailed** model (that port split into a stance **foot port** *and* a flight-capable **reaction-wheel arm
port**), carrying the policy **through the hierarchy** by warm-starting the detailed policy from the coarse one.
The two levels are two **HyMeKo models**, the refinement an `<isa>` chain.

### The RL side — `hierarchical_runstop.py`
- Coarse (foot-only) policy → **`warm_start`** lifts it into the detailed parameter space (shared foot/L weights
  copied, the new arm channel zero-initialised) → CEM refines with the arm.
- **Exact transfer, verified:** the lifted detailed policy is **bit-identical** to the coarse one at lift time
  (arm channel inert), so refinement begins exactly where the coarse level ended and **never regresses** it.

### The HyMeKo side — three validated models (`data/robotics/runstop_*.hymeko`)
- `runstop_ports.hymeko`: the port-refinement hierarchy — `angular_momentum_port` (abstract) refined into
  `foot_port` and `arm_reaction_wheel_port`, each `<isa>` the abstract port. **The `<isa>` chain IS the model
  hierarchy.**
- `runstop_coarse.hymeko` / `runstop_refined.hymeko`: the coarse model uses the abstract port; the refined model
  uses the foot + arm ports. All three **validate** (`hymeko validate` ✅).

## Results — honest, and task-dependent

| level | held-out stop-success (flight-heavy run-stop) |
|---|---|
| coarse (foot-only) | 0.49 |
| refined + warm-start (arm) | 0.50 |
| refined from scratch (arm) | 0.50 |

- **The framework works:** the warm-start transfer is exact and the refined model never regresses the coarse one.
- **The arm's gain is task-dependent — honestly reported:**
  - In the **pure balance task** (`reaction_wheel_arms.py`, `ee847a2c`) the flight-phase arm was **decisive**:
    recoverable basin 0.37 → 0.66 (**+0.30**).
  - In the **run-stop** it is **modest** (+0.01 here): the reaction-wheel arm's momentum capacity
    (`I_arm·ω_max ≈ 2.4`) is smaller than the braking-induced angular momentum `L`, so it can only partly offset
    it. Warm-start ≈ from-scratch at convergence; the transfer's value is the *guaranteed non-regression* and a
    warm start, not a larger final ceiling here.
- **The durable contribution is the framework + the HyMeKo hierarchy** (exact policy transfer across an `<isa>`
  port refinement), not the specific arm's magnitude — which the reduced model tells us is bounded by the arm's
  momentum vs the disturbance.

## Files touched

| File | notes |
|---|---|
| `scenarios/humanoid/hierarchical_runstop.py` | +160 — coarse↔refined dynamics, `warm_start`, CEM through the hierarchy |
| `tests/test_hierarchical_runstop.py` | +55 — exact transfer; arm channel inert at lift; non-regression; bounded/deterministic |
| `data/robotics/runstop_ports.hymeko` | port-refinement hierarchy (`<isa>` chain), validated |
| `data/robotics/runstop_coarse.hymeko` / `runstop_refined.hymeko` | coarse vs refined models, validated |
| `reports/2026-08-05-hierarchical-refinement-rl-hymeko.md` | this report |

## CORE.YAML items touched
None. numpy-only RL; the `.hymeko` files are non-core data modules validated by the existing CLI. No dependency change.

## Test results
- `pytest tests/test_hierarchical_runstop.py -p no:randomly` → **4 passed in 35 s**; `ruff check` → clean.
- `hymeko validate` on all three new `.hymeko` models → ✅.

## Open issues / follow-up
- **Deeper hierarchy:** refine further (two arms, shoulder+elbow; feet→ankle+hip) — each a new `<isa>` level with
  a warm-started policy. The framework generalises; this report is the two-level base case.
- **A task where the refined arm is decisive in run-stop:** longer flight, or a torque-coupled arm (as in the
  pure balance task) rather than momentum-capacity-limited — then the hierarchy's *magnitude* gain would show,
  not just the guaranteed non-regression.

## Provenance
Git SHA at start `ee847a2c`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2) + the built `target/release/hymeko` CLI,
macOS (darwin 25.5). Deterministic: seeded CEM, pinned `dt = 4 ms`. No GPU, no dataset.
