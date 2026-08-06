# Stepping-stone plan→execute: the pipeline is built; the executor hits a dynamics wall (honest, 4× measured)

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `bfe0ad34`.
**Plan:** `docs/plans/2026-08-06-stepping-stone-plan-execute/` (tex/pdf/tikz/mmd, gitignored).

---

## Summary

The goal was a *real* plan→execute demo: a trained walker executes a shared-A\*-planned foothold path
across a stepping-stone corridor. The **planner side is built and works**; the **executor side is blocked
by the humanoid model + WBC dynamics** — a wall confirmed **four independent ways**. This report documents
the built infrastructure and the measured negative honestly rather than shipping a demo that does not
execute.

## What was built (and works)

- **Shared-A\* corridor planner** (`stepping_stone_demo.plan_stones` + `footstep_planner.solve_astar`):
  plans which forward stones to land on (variable stride, cost = stride² so single steps are preferred and
  a longer stride is spent only to clear a gap); runs on the **same `akoire::astar` engine** (via the
  `hymeko.astar_plan` binding or its fallback). **Tested** (skips gaps, unreachable when a gap exceeds the
  max stride, prefers natural single strides).
- **`footstep_env` extensions** (additive, default-off): a per-step `plan_forward_x` foothold hook, and a
  `target_conditioned` mode that appends the commanded forward target to the obs + a foot-to-target reward.
- **`execute_plan`** (drives the env over a plan) and **`train_target_footstep`** (CEM over randomised
  forward targets).
- Two trained policies (scratchpad, not committed): a forward walker (scaffold −0.50 → **+0.25 m** fwd, 60
  steps upright) and a target-conditioned policy.

## The measured wall — commanded foothold stepping is not realised (4×)

| # | setup | measurement | verdict |
|---|---|---|---|
| 1 | scaffold WBC, **lateral** foothold command 0→0.12 m | swing foot lands at nominal (~−0.085) **regardless**; error grows linearly with the command | lateral command **inert** |
| 2 | scaffold WBC, **forward** march (action 0) | falls after 2 steps, net −0.70 m (backward) | forward scaffold **unstable** |
| 3 | **fixed-gait trained walker**, `plan_forward_x` override to the planned stone | landed feet go **backward** (−0.002…−0.021), reached_x **−0.032**, on-stone 0.12 | override **breaks the learned gait** |
| 4 | **target-conditioned** policy, commanded offset 0.008→0.030 | mean landed advance ≈ **0** (+0.0005…−0.0009) regardless of the command; policy learned to **stand still** | target-conditioning **does not track** |

**Mechanism.** The humanoid model + WBC swing tracking make a *commanded* forward step destabilising: the
fixed forward-stride walker is stable only around its training nominal (≈0.06 m) and advances only ~0.01 m
per step (the residual counteracts the nominal); overriding the nominal breaks it. The target-conditioned
policy, penalised for falling more than for missing the target, learns to **not step** (advance ≈ 0) —
because stepping to the commanded foothold falls. This is a **dynamics limit, not a reward-tuning issue**:
raising the target weight makes it step *and* fall; lowering it makes it stand. No RL on this env realises
commanded-foothold placement.

## Honest conclusion

The **plan→execute pipeline and the shared-A\* planner are built and verified**; the **executor (this WBC +
humanoid model) cannot realise a commanded foothold**, so the walker cannot follow the plan. Demonstrating
plan-following footstep locomotion needs a fundamentally more capable low-level controller (better swing
tracking / a whole-body MPC with a real capture-point step), or a different/actuation-richer model — not
more RL on this env. The frontier is closed at the dynamics level, consistent with the earlier
stepping-frontier report and the R11 "target-conditioned representation is hard" note.

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/footstep_planner.py` | +24 / −16 | extract `solve_astar` (shared A\* dispatch); `plan_footsteps` reuses it (§6.1) |
| `scenarios/humanoid/footstep_env.py` | +15 / −1 | `plan_forward_x` hook + `target_conditioned` obs/reward (default-off, obs unchanged when off) |
| `scenarios/humanoid/stepping_stone_demo.py` | +112 / new | `Corridor` + `plan_stones` (shared A\*) + `execute_plan` |
| `scenarios/humanoid/train_target_footstep.py` | +116 / new | CEM target-conditioned trainer |
| `tests/test_stepping_stone_demo.py` | +67 / new | planner unit tests + a mechanical `execute_plan` exercise (no false-positive following claim) |

## CORE.YAML / dependencies

None. Reuses the already-approved `hymeko.astar_plan` binding, `footstep_env` WBC, and
`train_footstep_walk`. No new dependency.

## Test / gate results

- `pytest tests/test_footstep_planner.py tests/test_stepping_stone_demo.py` → **12 passed, 2 skipped**
  (planner units + the mechanical execute_plan exercise; the rust-backend footstep-planner tests skip
  where `hymeko.astar_plan` is absent). No regression in the existing footstep suite (default obs unchanged
  — dim 10 with target off, 11 with it on).
- `ruff check` → clean on all touched files.

## Honest scope / negatives (guard)

- **No working plan-following demo** — do not claim one. The trained policies are not committed (they do
  not achieve foothold-following); the numbers above are the deliverable.
- `execute_plan` and `train_target_footstep` are exercised mechanically (they run + report); the *following*
  they were built for is the documented negative.

## Provenance

Git SHA `bfe0ad34` at start. Python: master worktree venv (CPython 3.11, mujoco 3.10.0, numpy 2, gymnasium)
for the mujoco runs; training via CEM, 8 workers, ~3–4 min each, checkpointed. Host: macOS (darwin 25.5),
Apple Silicon (18 cores). Deterministic per seed; the A\* is seed-free.
