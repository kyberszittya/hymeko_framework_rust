# Yes, the humanoid moves in MuJoCo — two gaits, and an uprightness fix

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `0f636a74`.

---

## Summary

The humanoid **does move forward in MuJoCo**. The original footstep walker achieved this by leaning far
forward (a precarious creep). Two reward changes give cleaner motion:

1. **`footstep_env` + `w_upright`** — a reward term penalising torso lean. The retrained footstep gait is
   **noticeably more erect** while still walking forward (**+0.179 m over 60 footsteps**, upright).
2. **`balance_env` + `w_velocity`** (`train_balance_walk`) — a *different* approach: direct locomotion on
   the position-servo joint action, rewarding forward base velocity. It is the **most upright / most
   stable** (survives all **300 steps**) but barely advances (**+0.065 m**) — stable standing with a slow
   forward drift.

| variant | forward | survived | posture (rendered) |
|---|---|---|---|
| original footstep walker | +0.10 m / 26 steps | ✓ | strong forward lean |
| **(1) footstep + `w_upright=3`** | **+0.18 m / 60 steps** | ✓ | more erect torso |
| **(3) direct locomotion (`w_velocity=8`)** | +0.065 m / 300 steps | ✓ (300) | most erect, minimal advance |

Videos + keyframes rendered to the scratchpad (`humanoid_walk.mp4`, `humanoid_fwalk_upright.mp4`,
`humanoid_bwalk.mp4`).

## Honest characterisation

The humanoid **moves and stays upright** — but a *fast, clean* upright walk remains out of reach on this
model + controllers. The same tension recurs: forward speed comes partly from leaning, and penalising the
lean (or maximising stability) trades speed for posture. `w_upright` is a real improvement (erecter gait,
still forward); direct locomotion maximises stability at the cost of progress. The forward-speed ceiling
(~a few mm/step) is a dynamics/model property, consistent with the earlier stepping-frontier findings.

## What was built (all reward knobs default-off — no regression)

- `footstep_env`: `FootstepConfig.w_upright` + a `- w_upright·(1 − uprightness)` reward term.
- `balance_env`: `BalanceConfig.w_velocity` + `vel_cap` + a capped `+ w_velocity·forward_base_velocity`
  reward term (turns the balance task into direct locomotion).
- `train_footstep_walk`: a `--w_upright` argument.
- `train_balance_walk.py` — **new**: CEM over the position-servo action for direct forward locomotion
  (reuses the linear `policy`/`_dim`, §6.1).

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/footstep_env.py` | +4 / −1 | `w_upright` config + reward term |
| `scenarios/humanoid/balance_env.py` | +7 / 0 | `w_velocity` + `vel_cap` config + reward term |
| `scenarios/humanoid/train_footstep_walk.py` | +6 / −4 | `--w_upright` arg (+ `_cfg` param) |
| `scenarios/humanoid/train_balance_walk.py` | +116 / new | direct-locomotion CEM trainer |
| `reports/2026-08-06-humanoid-locomotion.md` | new | this report |

## CORE.YAML / dependencies

None. All reward knobs default to 0 (behaviour unchanged when off). No new dependency.

## Test / gate results

- No regression: `pytest tests/test_footstep_planner.py tests/test_stepping_stone_demo.py` →
  **9 passed, 2 skipped** (the default-off knobs leave the env obs/reward unchanged). `ruff check` → clean.
- **Production-scale**: each policy trained at real scale (footstep 60-step episodes; balance 300-step
  episodes), CEM 8 workers, checkpointed; both survive their full horizon at eval.

## Honest scope / negatives (guard)

- The trained policies (scratchpad, not committed) move *slowly*; this is not a fast bipedal walk. Do not
  overclaim — the deliverable is "moves + upright, with a measured posture/speed trade-off", plus the
  reusable reward knobs and the direct-locomotion trainer.
- Direct locomotion (Approach 3) at `w_velocity=8` mostly stands; higher weights trade stability for a
  lunge. A faster clean walk likely needs a stronger controller/learner (PPO/SAC, longer training) — noted.

## Provenance

Git SHA `0f636a74` at start. Python: master worktree venv (CPython 3.11, mujoco 3.10.0, numpy 2). CEM,
8 workers, ~4–10 min per policy, checkpointed. Host macOS (darwin 25.5), Apple Silicon (18 cores).
Deterministic per seed. Renders via `mujoco.Renderer` (per-tick / per-step, camera tracking the pelvis).
