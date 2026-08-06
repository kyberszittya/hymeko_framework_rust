# Refining the torque walk — (a) viability-hinge healthy shaping + (c) toe push-off: a speed↔stability trade-off

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `cbb5721a`.

---

## Summary

Two refinements over the torque walk (`cbb5721a`, +0.71 m at 0.36 m/s, falls ~2 s):

- **(a) Viability-band healthy shaping** — a **hinge** penalty applied only when the torso leans *past* a
  narrow uprightness band near the fall (`upright_safe`), i.e. penalising **exiting the hysteresis
  region**, not the walking lean (per your guidance: not a continuous lean penalty, only at the stability
  boundary). Tuning is load-bearing: `upright_safe = 0.75` was **too high** (it fired during the normal
  walking lean and hurt the gait); `0.68` (a narrow band just above the `0.6` fall threshold) is correct.
- **(c) Toe/arm push-off** — training the torque SAC on the **articulated-toe model** (`humanoid_toe`),
  whose 18-DOF torque action includes the toe + arm joints, so the policy can push off and swing.

**Result — a speed↔stability trade-off, not a strict win:**

| walker | speed | distance | falls at | gait |
|---|---|---|---|---|
| plain torque (`cbb5721a`) | 0.36 m/s | **+0.71 m** | **~2.0 s** | forward-leaning |
| **(a)+(c) toe + healthy** | **~0.48 m/s** | +0.42 m | ~1.0 s | **faster, wide-stride, visible toe push-off** |

The toe + push-off gait is **faster and more dynamic** (wider stride, toe visible — rendered
`humanoid_ac_walk.mp4`), but **less stable** (falls at ~1 s vs ~2 s). The viability-hinge healthy term
(even correctly tuned) **did not extend survival** for the fast gait — the fast dynamic walk topples
regardless; the instability is dynamic, not a slow lean the hinge can catch.

## Honest reading

- **(a) works as designed but is not sufficient here.** The narrow-hysteresis hinge is the right shape
  (penalise leaving the viability band, not the lean) and it *is* better than the mis-tuned 0.75, but it
  does not, on its own, stop the fast gait from toppling in ~1 s. A viability *certificate*-gated term or a
  periodic-gait prior would likely be needed for indefinite walking.
- **(c) buys speed at the cost of stability.** The toe model's extra DOF give a faster, more natural
  stride but a harder-to-stabilise 18-DOF problem; SAC finds a fast gait that falls sooner.
- **Best distance/survival remains the plain torque walk** (0.71 m, 2 s); **best speed is the toe+healthy
  gait** (0.48 m/s). Neither is indefinite — flagged.

## What was built (reward knobs default-off — no regression)

- `balance_env`: `w_stability` (viability-band hinge) + `upright_safe` (narrow hysteresis threshold, default
  `0.68`). The hinge is `− w_stability · max(0, upright_safe − uprightness)` — zero within the safe band.
- `run_humanoid_walk_sac.py`: `--w_stability` + `--model_src` (e.g. `humanoid_toe.hymeko`).

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/balance_env.py` | +7 / 0 | `w_stability` + `upright_safe` config + viability-hinge reward term |
| `scenarios/humanoid/run_humanoid_walk_sac.py` | +4 / −1 | `--w_stability` + `--model_src` args |
| `reports/2026-08-06-humanoid-walk-refine.md` | new | this report |

## CORE.YAML / dependencies

None. `w_stability` defaults 0 (reward unchanged when off). No new dependency.

## Test / gate results

- No regression: `pytest tests/test_footstep_planner.py tests/test_stepping_stone_demo.py` → **9 passed,
  2 skipped**. `ruff check` → clean.
- **Production-scale**: two 200 k-step SAC runs (toe model, 600- and 1000-step episodes), checkpointed,
  held-out forward-distance selection; best re-measured over 5000-step rollouts across 4 seeds
  (mean 0.42 m, 0.94 s to fall).

## Follow-up (toward indefinite walking)

- **Viability-*certificate*-gated term** (not just a hinge): reward the certified viable set from the M0–M2
  ladder, so the policy is pushed to stay provably recoverable.
- **Episode-length curriculum** + longer SAC: grow `max_steps` as the policy survives longer.
- **Periodic-gait prior** (a phase variable + a nominal cyclic reference) so the torque policy learns a
  *repeating* gait rather than a one-shot dynamic lunge.

## Provenance

Git SHA `cbb5721a` at start. Python: master venv (CPython 3.11, mujoco 3.10.0, torch 2.12.0 CPU, numpy 2).
SAC via `hymeko_rl.train.sac`, checkpointed. Host macOS (darwin 25.5), Apple Silicon. Deterministic per
seed. timestep 0.001 s.
