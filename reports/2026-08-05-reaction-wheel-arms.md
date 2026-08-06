# Arms as a reaction wheel — the flight-phase balance port (answering "can the hands help?")

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `23b5d4bc`)
**In response to:** *how much does the rest of the body move, and can the arms be used for the "perfect energy"?*

---

## The question, answered

Our centroidal model is a **lump**: it tracks only the total angular momentum `L` (`L̇` = the external contact
moment — **zero in flight**) and the torso pitch; it does not resolve *where* the momentum sits. So it never
credited the **arms** as their own control. Physically they are one: carrying angular momentum `I_arm·ω_arm`, the
arms act as a **reaction wheel** — accelerating them torques the torso oppositely **without any ground contact**,
so they stabilise the pitch **even in flight**, where the foot (stance-only) has no authority. This is the
diver/gymnast/cat strategy, and it is exactly what the "arm swing = L-port" in the visualization abstracted.

## Measured

An inverted torso (`I_t·pitcḧ = mgl·sin(pitch) + τ_foot(stance) + τ_arm`), foot torque bounded and stance-only,
arm a reaction wheel **hard-limited** to `±arm_range`:

| control | recoverable basin | arm swing used |
|---|---|---|
| foot only (stance) | 0.367 | — |
| **foot + reaction-wheel arms** | **0.663** | median **1.60 rad**, max 1.60 rad (= the ±1.6 rad ≈ ±92° limit) |

- **The arms extend the recoverable balance basin by +0.30** (0.37 → 0.66) — they **save ~30% of the perturbation
  grid** the foot loses, and the states they save are the **flight-phase** ones (no foot contact → the only way
  to stabilise the pitch is to move momentum into the arms).
- **How much do the arms move?** A lot: the recovering swings run to the **full ±1.6 rad (~±92°)** range — a big
  windmill — and (with the range strictly enforced) never past it. A real arm's finite range is the binding limit;
  once the arms are wound out they give no further outward torque and the foot must take over in the next stance.

So: **yes — the hands/arms are usable, and decisively so.** They are the flight-phase angular-momentum port the
stance-only model was missing; "perfect energy" balancing wants them because they buy control authority exactly
when the feet have none.

## Files touched

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/reaction_wheel_arms.py` | +90 (new) | inverted-torso balance + reaction-wheel arm (hard range) + stance foot; basin measurement |
| `tests/test_reaction_wheel_arms.py` | +30 (new) | 3 tests (arms extend the basin; swing within the mechanical limit; determinism) |
| `reports/2026-08-05-reaction-wheel-arms.md` | new | this report |

## CORE.YAML items touched
None. numpy-only, new module. No dependency change.

## Test results
- `pytest tests/test_reaction_wheel_arms.py -p no:randomly` → **3 passed in 0.13 s**. `ruff check` → clean.

## Honest scope
This is a **reduced, single-arm-DOF reaction-wheel** abstraction — it captures the *mechanism* (internal
momentum redistribution stabilises the torso without contact) and quantifies it, but it is not the full
multi-body arm (two arms, shoulder + elbow, coupling to the CoM). The total angular momentum is still only changed
by contacts; the arms redistribute it (correctly modelled here as a torso reaction torque). A full-body
multi-segment check (and the arm port added to the run-stop model, where flight is currently uncontrolled) is the
natural next step.

## Open issues / follow-up
- **Add the arm port to run-stop / centroidal:** flight was the uncontrolled phase there; a reaction-wheel arm
  would give flight-phase braking-recovery — likely a further gain over the stance-only policies.
- **Two-arm, multi-segment model** and a MuJoCo cross-check to validate the reduced reaction-wheel numbers.

## Provenance
Git SHA at start `23b5d4bc`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2), macOS (darwin 25.5). Deterministic
(no RNG; fixed grid, pinned `dt = 4 ms`). No GPU, no dataset.
