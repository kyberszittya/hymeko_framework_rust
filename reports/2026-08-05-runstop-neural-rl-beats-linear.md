# Run-and-stop: neural RL beats tuned-linear on the hard task, with a HSTL-monitor-robustness reward

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `73161ec4`)
**Follow-ups (1)+(2):** the harder task where a neural policy can win, tied to the verification arc.

---

## Summary

The pure L-regulation task was linearly solvable (a neural policy only matched tuned-linear). **Running then
STOPPING is genuinely hard** — and here a neural policy **beats** the best tuned linear controller.

**Why run-stop is hard (and nonlinear):** braking is a foot force that decelerates ``vx`` *and*, sitting below the
CoM, induces an angular momentum ``L`` (a forward pitch torque) that must be regulated at the same time; control
acts **only in stance** (no foot force in flight, where the pitch grows ballistically), under **bounded**
``|fx|,|a|``. Across a range of stopping speeds no single linear gain is optimal — stopping needs
gain-scheduling + saturation-aware timing. Verified headroom: the **best single linear gain reaches only 0.75**
stop-success on the mixed held-out set.

**Result (held-out, unseen initial states):**

| | stop-success | fall rate |
|---|---|---|
| **neural policy (CEM)** | **0.910** | **0.000** |
| best single linear gain | 0.750 | — |

**Δ = +0.16 held-out, with zero falls.** The neural policy genuinely beats the best tuned-linear controller —
the honest positive that the pure-regulation task could not produce. It learned to schedule braking + L-regulation
by speed and gait phase, which a fixed linear gain cannot.

## (2) The reward IS the HSTL monitor robustness

The training reward is the **HSTL robustness of the safety spec** ``G(fall_margin ≥ 0)`` — the worst-case upright
margin over the episode. Robust-STL Globally makes this exactly ``min_t (fall_pitch − |pitch|)``, which the
episode computes vectorised. A test **verifies** it equals the actual monitor: feeding the per-step ``fall_margin``
to ``make_monitor("G(fall_margin ≥ 0)")`` gives a robustness bit-identical to the episode's reward. So the RL is
driven by the same HSTL monitor built in the verification arc — the (2) tie-in, made concrete (and the Rust
backend can serve it in real time).

## Files touched

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/centroidal_runstop.py` | +150 (new) | run-stop dynamics (braking→pitch coupling, stance-gated, bounded); MLP policy + CEM; HSTL-robustness reward |
| `tests/test_centroidal_runstop.py` | +95 (new) | 6 tests (bounded actions, headroom, **reward=monitor-robustness**, beats-linear held-out, determinism, perf) |
| `reports/2026-08-05-runstop-neural-rl-beats-linear.md` | new | this report |

## CORE.YAML items touched
None. numpy-only (CEM + numpy MLP + the pure-Python HTL monitor for the reward-parity test); no dependency change.

## Test results
- `pytest tests/test_centroidal_runstop.py -p no:randomly` → **6 passed in 24.9 s** (the CEM training fixture is the bulk).
- `ruff check` → clean. No other suite touched (new module).

## Performance
- CEM training (30 iters, pop 56, mixed set): ~32 s. `evaluate` (one held-out sweep): < 1 s. numpy-only, RSS
  negligible.

## §6.5 anti-patterns
None. One `runstop_step` integrator; `RunStopConfig`/`PolicyConfig` typed configs; the MLP policy is one function
family; no globals. The reward reuses the HSTL monitor semantics (verified equal), not a re-implementation.

## Interpretation (honest)

This closes the loop opened by the pure-regulation RL: **neural RL adds nothing on a linearly-solvable task, but
genuinely wins on the hard, nonlinear, phase-gated run-stop** — the user's point that running *and stopping* is
hard enough is borne out. The win is on held-out states (not overfitting) and with zero falls.

## Open issues / follow-up
- **Certificate-shielded exploration:** the reward already uses the monitor robustness; adding a hard shield (the
  policy constrained to keep the monitor satisfied, or to the certified set) would make training safe by
  construction — the remaining half of (2).
- **TD3 / policy-gradient** on the same interface (the memory's scaling method); CEM was the robust first learner.
- A **full-state (5-D, with z-bounce) run-stop** and a MuJoCo cross-check are the path from the reduced model back
  to the embodied humanoid.

## Provenance
Git SHA at start `73161ec4`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2; `hymeko_neuro.eval.htl` for the reward
test), macOS (darwin 25.5). Deterministic: seeded CEM, pinned `dt = 4 ms`, deterministic grids. No GPU, no dataset.
