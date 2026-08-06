# Learned viability boundary + stability-constrained reward gate — mechanism works, one-shot gating did not (co-adaptation needed)

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `6d86e166`.

---

## Summary

Toward *sustained* walking: a **stability-constrained reward** gated by a **learned viability boundary**
(the honest fix the healthy hinge only approximated). We **learn the Lyapunov/viability boundary** from
the walker's own rollouts — a state is viable iff the policy does not fall within `horizon` steps — fit a
logistic `P(viable | reduced-state)`, and **gate the forward reward** by it: the policy is paid for speed
only while provably recoverable, so it should walk *within* the certified region.

**The mechanism works; the one-shot gated training did not improve the walk — it made it worse.**

| stage | result |
|---|---|
| learned boundary | **91 % train accuracy**; physically sensible weights (uprightness +4.05 → viable, CoM-forward-offset −2.61 → about to fall) |
| env reward gate | loads the `.npz`, gates the forward reward by `P(viable) ∈ [0,1]` (tested) |
| viability-gated SAC (220 k) | **mean 0.10 m, falls at 0.46 s** — *worse* than plain torque (0.71 m, 2.0 s) and periodic (0.47 m, 0.72 s) |

## Why one-shot gating failed (honest)

The boundary was learned from the **old** (periodic) policy, which falls at ~0.7 s. The **new** gated
policy explores **different** states, where the fixed boundary **extrapolates off-distribution** and
mis-labels viability — the classic distribution-shift failure of a learned safety critic. The gate then
removes forward reward in the wrong places, and SAC (already juggling 5 reward terms + a gate + 18 DOF +
the phase clock in 220 k steps) converges to a conservative-yet-unstable gait that walks little and falls
early. **This needs iterated co-adaptation** — learn the boundary from the *current* policy, retrain, and
repeat (DAgger-style for the certificate) — not a single fixed boundary.

## What was built (default-off — no regression; obs 43 unchanged)

- `viability_gate.py`: `LearnedViabilityBoundary` (logistic fit/predict/save, same form as
  `viability.LearnedBoundary`) + `collect_labelled` (horizon fall-labelling) + `learn_from_policy`.
- `balance_env`: `viability_state()` (5-D: uprightness, forward tilt, pitch-rate, forward velocity,
  CoM-forward offset), `viability_boundary` config (loads the `.npz`), and the forward-reward **gate**
  `+= w_velocity · fwd_v · P(viable)`.
- `run_humanoid_walk_sac.py`: `--viability_boundary`.

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/viability_gate.py` | +91 / new | learned viability boundary + labelling |
| `scenarios/humanoid/balance_env.py` | +26 / −1 | `viability_state` + boundary load + forward-reward gate |
| `scenarios/humanoid/run_humanoid_walk_sac.py` | +2 / 0 | `--viability_boundary` |
| `tests/test_viability_gate.py` | +60 / new | boundary fit/predict/save + env-gate (mujoco) |
| `reports/2026-08-06-humanoid-viability-gate.md` | new | this report |

## CORE.YAML / dependencies

None. `viability_boundary` defaults off (reward unchanged). No new dependency (numpy logistic; reuses
`hymeko_rl.train.sac`).

## Test / gate results

- `pytest tests/test_viability_gate.py tests/test_footstep_planner.py tests/test_stepping_stone_demo.py`
  → **13 passed, 2 skipped** (boundary fit/predict/save + the mujoco env-gate test; default obs 43
  unchanged). `ruff check` → clean.
- **Production-scale**: boundary fit from 20 seeds of the periodic walker (91 % acc); viability-gated SAC
  220 k steps; best re-measured over 8000-step rollouts across 5 seeds.

## Honest scope / negatives (guard)

- **The gated walk is worse, not better** — do not claim the viability gate improved walking. The learned
  boundary (91 %) and the gate mechanism are the deliverable; the *one-shot* application regressed the gait.
- The failure is **distribution shift** (boundary from the old policy, evaluated on the new one), the
  expected failure mode; it does **not** invalidate the approach — it argues for **co-adaptation**.

## Follow-up (the real path)

- **Co-adaptation loop**: (1) learn the boundary from the current policy, (2) retrain the gated policy,
  (3) re-learn the boundary, … until survival stops improving — the certificate tracks the policy's
  state distribution instead of lagging it.
- **Neural viability certificate**: swap the logistic for the `NeuralLyapunovCertificate` (already in the
  repo) for a nonlinear boundary; keep the horizon-fall labelling.
- **Simplify the combined objective**: gate the *plain* torque walk (which is the most sustained baseline)
  rather than the 5-term periodic+toe stack, so the gate's effect is not drowned out.

## Provenance

Git SHA `6d86e166` at start. Python: master venv (CPython 3.11, mujoco 3.10.0, torch 2.12.0 CPU, numpy 2).
SAC via `hymeko_rl.train.sac`; boundary = numpy logistic. Host macOS (darwin 25.5), Apple Silicon.
Deterministic per seed. timestep 0.001 s.
