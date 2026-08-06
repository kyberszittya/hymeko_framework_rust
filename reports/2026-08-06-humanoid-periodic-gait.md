# Periodic-gait prior — the fastest gait yet (0.83 m/s), but the speed↔stability trade-off holds

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `f7faf362`.

---

## Summary

Added a **periodic-gait prior** to break the one-shot-lunge failure mode toward *sustained* walking: a
**phase clock** in the observation (`sin φ, cos φ`, advancing `2π/gait_period` per step) plus a **cyclic
reward** that pays L/R stride alternation in sync with the phase (`+ w_gait · sin φ · (x_L − x_R)`), so
the policy is pushed to learn a **repeating** gait, not a single dynamic lunge.

**Result:** the periodic prior produced the **fastest, most athletic gait yet** — up to **0.83 m/s**
(+0.77 m in one seed), a clear dynamic run/walk with a swing–stance cycle (rendered
`humanoid_periodic_walk.mp4`) — **but it falls sooner**, not later. The speed↔stability trade-off from the
earlier refinements not only held, it *intensified*.

## The walking arc, measured (mean over seeds)

| controller / prior | speed | distance | falls at | note |
|---|---|---|---|---|
| position-servo (SAC/CEM) | — | ~0.08 m | lunge-and-settle | ceiling (q0 anchoring) |
| **torque** (`cbb5721a`) | 0.36 m/s | **+0.71 m** | **~2.0 s** | **most sustained** |
| torque + toe + healthy (`f7faf362`) | 0.48 m/s | +0.42 m | ~0.94 s | faster, less stable |
| **torque + toe + healthy + periodic** | **~0.6 m/s (peak 0.83)** | +0.47 m | ~0.72 s | **fastest, least stable** |

The pattern is consistent: **each layer of gait machinery (toe DOF, periodic prior) buys speed and costs
survival.** Faster gaits are dynamically harder to keep upright in this model + torque setup; SAC, given a
dominant forward reward, trades stability for speed regardless of the prior. **Indefinite walking was not
achieved** — the periodic prior gives a *faster* repeating gait, not a *slower, stable* one.

## Honest reading

- **The periodic prior is the right shape** (a phase clock + cyclic reference is the standard route to a
  repeating gait) and it *did* produce a genuine cyclic swing–stance gait — but paired with the dominant
  forward reward it converged on a fast, short-lived gait, not a sustained one.
- **Best sustained walk stays the plain torque** (0.71 m, 2 s); **best speed is the periodic gait**
  (0.83 m/s peak). The choice is a dial (forward weight, gait period, healthy weight), not a solved walk.
- To get *sustained* fast walking likely needs a **stability-constrained** objective (reward the certified
  viable set, or cap the forward reward until survival is long), an **episode-length curriculum**, and more
  training than 220 k SAC steps for this 5-term, 18-DOF, phase-augmented problem.

## What was built (default-off — no regression)

- `balance_env`: `periodic_gait` (phase clock in the obs) + `gait_period` + `w_gait` (cyclic
  foot-alternation reward). Obs gains 2 dims (`sin φ, cos φ`) only when on; default obs unchanged (43).
- `run_humanoid_walk_sac.py`: `--periodic`, `--gait_period`, `--w_gait`.

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/balance_env.py` | +16 / −2 | `periodic_gait` phase clock (obs) + cyclic reward + phase advance |
| `scenarios/humanoid/run_humanoid_walk_sac.py` | +6 / −1 | `--periodic`/`--gait_period`/`--w_gait` args |
| `reports/2026-08-06-humanoid-periodic-gait.md` | new | this report |

## CORE.YAML / dependencies

None. `periodic_gait` defaults off (obs + reward unchanged). No new dependency.

## Test / gate results

- No regression: `pytest tests/test_footstep_planner.py tests/test_stepping_stone_demo.py` → **9 passed,
  2 skipped**; default obs dim unchanged (43). `ruff check` → clean.
- **Production-scale**: SAC 220 k steps (toe model, phase-augmented obs, 1200-step episodes), checkpointed,
  held-out forward-distance selection; best re-measured over 6000-step rollouts across 5 seeds
  (mean 0.47 m, 0.72 s; peak 0.83 m/s).

## Follow-up (to actually sustain)

- **Stability-constrained reward**: gate/scale the forward reward by the M0–M2 viability certificate (only
  pay speed while provably recoverable) — the honest fix the healthy hinge approximated.
- **Episode-length curriculum**: grow `max_steps` as survival grows, so the policy is not rewarded for a
  fast lunge that ends the episode.
- **Slower target cadence**: a longer `gait_period` + a *tracked* nominal joint reference (DeepMimic-style)
  rather than a directional bonus, for a stable rather than maximal gait.

## Provenance

Git SHA `f7faf362` at start. Python: master venv (CPython 3.11, mujoco 3.10.0, torch 2.12.0 CPU, numpy 2).
SAC via `hymeko_rl.train.sac`, checkpointed. Host macOS (darwin 25.5), Apple Silicon. Deterministic per
seed. timestep 0.001 s.
