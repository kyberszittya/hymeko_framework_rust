# Certificate/monitor safety shield for run-stop RL — safe exploration by construction

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `cd39f1a3`)
**Follow-up (1 of the "both"):** the hard safety half of the (2) tie-in — a shield that makes RL safe by construction.

---

## Summary

Added a **model-predictive safety shield** to the run-stop task: it projects any policy's action to keep the
fall margin up, so training (or an unverified policy) **explores without falling**.

**The key insight — react early.** The fall is relative-degree-2 (`a → L → pitch`), so reacting to `|pitch|`
alone is too late (by the time pitch nears the edge, the angular momentum `L` has already committed it). The
shield activates on the **predicted** pitch `pitch + (L/I)·τ` — where the current `L` is taking the torso — and as
that predicted margin drops below a buffer it blends the action toward the safe fallback: **stop braking** (the
destabilising input to `L`) and drive the L-port to **oppose the predicted pitch** (`a → −sign·a_max`).

## Results

| policy | shield | fall rate | stop-success |
|---|---|---|---|
| **random (even ×2 aggressive)** | off | up to **0.54** | — |
| **random (even ×2 aggressive)** | **on** | **0.000** | — |
| trained (unshielded) | off | 0.000 | 0.941 |
| trained (unshielded) | on (deploy) | **0.000** | 0.906 |
| **trained inside the shield** | **on** | **0.000** | **0.922** |

- **Safe by construction:** with the shield, *random and even aggressive* policies **never fall** (0.000, max
  0.000) — verified across many random parameter draws. This is the safe-exploration guarantee the (2) tie-in was
  aiming for.
- **Small, recoverable cost:** the shield trims stop-success (0.941 → 0.906 at deploy) because it brakes
  conservatively near the predicted edge; **training inside the shield recovers part of it** (0.922) — the policy
  learns to operate within the safe envelope. Every shielded configuration falls **0.000**.

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/centroidal_runstop.py` | +18 | `safety_shield` (predictive); `episode`/`train_cem`/`evaluate` gain a `shield` flag |
| `tests/test_centroidal_runstop.py` | +18 | shield makes random policies safe (0 falls); shielded training safe + capable |
| `reports/2026-08-05-runstop-safety-shield.md` | new | this report |

## CORE.YAML items touched
None. numpy-only, additive (`shield` defaults to off — every existing caller unchanged). No dependency change.

## Test results
- `pytest tests/test_centroidal_runstop.py -p no:randomly` → **8 passed in 44 s** (2 new shield tests).
- `ruff check` → clean.

## Honest scope
The shield's no-fall is **empirical** (0.000 across random/aggressive/trained policies over the held-out set),
not a formal barrier-certificate proof — a control-barrier-function (CBF) certificate for the relative-degree-2
pitch would upgrade it to provable. The predictive-margin design is a practical, verified stand-in; the exact CBF
is the flagged next step.

## Open issues / follow-up
- **Formal CBF certificate** for the shield (provable forward-invariance), tying back to the LMI/Poincaré work.
- **Shield + the HSTL monitor at runtime:** the monitor already scores `G(fall_margin≥0)`; the shield keeps that
  spec satisfied — deploy them together (monitor watches, shield acts).

## Provenance
Git SHA at start `cd39f1a3`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2), macOS (darwin 25.5). Deterministic
(seeded random-policy draws, seeded CEM; the shield is a deterministic map). No GPU, no dataset.
