# Neural RL for the centroidal L-regulation — a controllability bug caught, then an honest held-out result

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `d01647db`)
**Next arc:** learning the angular-momentum port as a policy, using the certificate/monitor world built above.

---

## Summary

A learned **residual L-port** policy (`a = a_max·π_θ(s)`, on top of the scripted regulator via
`centroidal_step(l_residual=…)`), trained by **CEM** (evolutionary policy search — robust, deterministic, no
autograd tuning to confound the result), evaluated the only way that counts: **held-out** vs the scaffold.

## A real bug, caught by the user (recorded honestly)

The first run showed the RL policy **tying** the scaffold (held-out recover 0.857 = 0.857) with **zero delta as
`a_max` grew to 20** — the user flagged this: *"if there's no delta across `a_max`, the control or the kinematics
is wrong."* Diagnosis (direct, not assumed):

- **No implementation bug:** `rollout(None)` is **bit-identical** to the bare `centroidal_step` loop
  (`array_equal` True) — the residual is a true additive port.
- **The kinematics were wrong:** at `pitch_gain=2.5` a direct pitch hold dominates, so **L cannot drive the fall**
  — a trajectory from `(L=6, pitch=0.3)` reaches only `max|pitch| = 0.608` (never the 1.25 threshold). The falls
  were pure `|pitch₀|` kinematics — nothing for the L port to control → Δ=0 for any `a_max`. My earlier "L
  matters" premise was wrong.

**Fix:** `regulation_task_config` sets `pitch_gain = 0` — the torso pitch is then controllable **only through the
L port** (a genuine angular-momentum-regulation task). Controllability verified: at `l_damp=2`, a corrective port
cuts the fall rate **0.331 → 0.064**; stronger L regulation always recovers more.

## Result (corrected, honest)

Trained on the controllable task, held-out (unseen initial states):

| | held-out recover |
|---|---|
| **RL residual policy** | **0.751** |
| scripted weak scaffold (`l_damp=2`) | 0.549 |
| well-tuned linear regulator (`l_damp=6`) | 0.744 |

- **RL beats the weak scaffold by +0.20 on held-out** (0.751 vs 0.549) — the policy genuinely learns to regulate
  `L` and prevent falls; not overfitting (held-out, disjoint grid).
- **RL only MATCHES the well-tuned linear regulator** (Δ = +0.007, within noise): the neural policy rediscovers
  "damp `L` more" and does **not** beat a properly-tuned linear controller. This is exactly the repo's recurring
  lesson — RL matches but does not clearly beat a well-tuned scaffold
  ([[feedback-heldout-panel-is-single-use]], the coin/AIBO precedent). The honest verdict: **neural RL solves the
  task but adds nothing over tuned-linear here.**

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/centroidal.py` | +3 | optional additive `l_residual` port on `centroidal_step` (default 0 — every caller unchanged) |
| `scenarios/humanoid/centroidal_rl.py` | +130 (new) | `regulation_task_config`, numpy MLP residual policy, CEM trainer, held-out `evaluate` |
| `tests/test_centroidal_rl.py` | +90 (new) | 7 tests (bounded action, **rollout-parity**, controllability, held-out beats-weak, matches-tuned, determinism, perf) |
| `reports/2026-08-05-centroidal-neural-rl.md` | new | this report |

## CORE.YAML items touched
None. numpy-only (no torch needed for CEM + a numpy MLP). `centroidal_step`'s new arg is additive/back-compatible
(M2 + limit-cycle suites unchanged: 16/16 pass). No dependency change.

## Test results
- `pytest tests/test_centroidal_rl.py -p no:randomly` → **7 passed in 8.0 s**.
- Regression: `test_centroidal_certificate.py` + `test_limit_cycle.py` → **16 passed** (the `l_residual` default is a
  no-op for existing callers).
- `ruff check` → clean.

## §6.5 anti-patterns
None. The residual reuses the single `centroidal_step` (no duplicated integrator — the reason `l_residual` is a
parameter, not a forked step); `RLConfig`/`regulation_task_config` are typed configs; no globals; the numpy MLP
policy is one function family, not a per-arch dump.

## Open issues / follow-up
- **Beat tuned-linear, or conclude it can't be beaten:** the task as posed is (near-)linearly solvable, so a
  neural policy has no edge. A harder task — nonlinear dynamics, delays, partial observation, or a
  disturbance-rejection objective — is where a learned policy could genuinely win. Otherwise the honest result
  stands: tuned-linear suffices.
- **Certificate-shielded RL:** constrain the policy to the certified set (M1/Poincaré) or reward the HSTL monitor
  robustness — the natural tie-in to the verification work, and the guard against unsafe exploration.
- **TD3 / policy-gradient** on the same residual interface (the memory's preferred scaling method) — CEM was the
  robust first step to get a trustworthy held-out verdict without RL-tuning confounds.

## Provenance
Git SHA at start `d01647db`. Env: HyMeKo `.venv` (Python 3.11, NumPy 2), macOS (darwin 25.5). Deterministic:
seeded CEM (`RandomState`), pinned `dt = 4 ms`, deterministic grids. No GPU, no dataset.
