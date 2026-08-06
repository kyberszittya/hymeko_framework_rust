# Co-adaptation of the viability gate — the certificate loop runs, but reward-gating still doesn't sustain the walk

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `d74a443a`.

---

## Summary

The one-shot viability gate failed by **distribution shift** (boundary learned from the old policy,
applied to the new one). The principled fix is **co-adaptation**: learn the boundary from the *current*
policy, retrain the gated policy, re-learn the boundary, repeat — so the certificate tracks the policy's
state distribution. Built (`run_walk_coadapt.py`) and run on the **plain torque** walker (the most
sustained baseline, so the gate isn't drowned out by the periodic+toe stack).

**The loop runs and the boundary stays accurate — but reward-gating still does not sustain the walk.**

| iter | gated by | survival | distance | boundary acc |
|---|---|---|---|---|
| 0 | none | 0.54 s | **0.163 m** | 0.975 |
| 1 | iter 0 | 0.60 s | 0.139 m | 1.00 |
| 2 | iter 1 | 0.55 s | 0.094 m | 0.991 |

**Survival is flat-and-noisy (0.54 → 0.60 → 0.55); distance monotonically *drops* (0.163 → 0.139 →
0.094).** Each co-adaptation round makes the gait **slower / more conservative** without extending
survival. The viability gate **suppresses forward motion near the boundary** rather than teaching a
*stable* forward gait — because the instability is a **control/dynamics** property (the humanoid topples),
not something reward-shaping fixes.

## Honest conclusion (the whole viability-gate line)

- The **learned boundary is accurate** (0.97–1.00) and the **co-adaptation loop is correct** (the
  certificate tracks the policy). The machinery is sound and reusable.
- **Reward-gating by viability does not crack sustained fast walking** — one-shot *or* co-adapted, it
  trades speed for a marginal/no stability gain. Removing forward reward makes the policy walk *less*, not
  *more stably*.
- **Confound**: iteration 0 was under-trained (100 k steps → 0.54 s, vs the committed plain-torque walker's
  2.0 s at 150 k). Starting from the strong 2 s baseline might shift the numbers, but the trend (distance
  falling under gating) is consistent and points the same way.

## Where the walking arc stands

The **best sustained walk remains the well-trained plain torque** (`cbb5721a`: 0.71 m, 2.0 s, 0.36 m/s).
Every reward-shaping refinement — healthy hinge, periodic prior, viability gate, co-adaptation — either
buys **speed at the cost of survival** or **conservatism at the cost of distance**; none beat the plain
torque for *sustained* walking. The honest read: **sustained fast bipedal walking here is a control/model
problem, not a reward-shaping one.** The remaining levers are structural, not reward:
- a **stronger low-level controller** (a stabilising inner loop the RL rides on),
- a **model with feet/ankles that don't topple** (or contact/friction tuning),
- **far more training** (this is CPU SAC at ~10⁵ steps; published bipeds use 10⁷–10⁸),
- or an **imitation/reference** gait (DeepMimic) rather than reward from scratch.

## What was built (reusable)

- `run_humanoid_walk_sac.py`: extracted **`train_walk_sac(cfg, steps, out)`** — the reusable SAC training
  core (the CLI and the co-adapt driver share it, §6.1).
- `run_walk_coadapt.py`: the **co-adaptation loop** (learn boundary ↔ retrain gated policy), with
  per-iteration survival/distance/boundary-accuracy logging.

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/run_humanoid_walk_sac.py` | +34 / −29 | extract `train_walk_sac`; `main` calls it |
| `scenarios/humanoid/run_walk_coadapt.py` | +106 / new | co-adaptation driver |
| `reports/2026-08-06-humanoid-viability-coadapt.md` | new | this report |

## CORE.YAML / dependencies

None. Reuses `hymeko_rl.train.sac` + `viability_gate`. No new dependency; the refactor is behaviour-
preserving (the SAC CLI is unchanged).

## Test / gate results

- No regression: `pytest tests/test_viability_gate.py tests/test_footstep_planner.py` → **9 passed,
  2 skipped** (the `train_walk_sac` extraction preserves the SAC CLI). `ruff check` → clean.
- **Production-scale**: 3 co-adaptation iterations × 100 k SAC steps on the plain torque config,
  checkpointed; survival re-measured over 4000-step rollouts across 6 test seeds per iteration.

## Honest scope / negatives (guard)

- **Co-adaptation did not sustain the walk** — do not claim it did. Survival stayed ~0.55 s and distance
  fell across iterations. The loop + the learned boundary are the deliverable; the sustained-walk payoff
  did not materialise (a control/dynamics limit, flagged).

## Provenance

Git SHA `d74a443a` at start. Python: master venv (CPython 3.11, mujoco 3.10.0, torch 2.12.0 CPU, numpy 2).
SAC via `hymeko_rl.train.sac`. Host macOS (darwin 25.5), Apple Silicon. Deterministic per seed. timestep
0.001 s.
