# Not 5 cm — 0.7 m: torque control breaks the forward-speed ceiling (a real walk)

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov` · **Git SHA at start:** `fc59e6ca`.

---

## Summary

"Let's not settle for 5 cm." We didn't. The ~5–8 cm forward "ceiling" was **not** a learner or reward
limit — it was the **position-servo control scheme** (the action is a bounded offset around the standing
pose `q0`, so the servo always pulls back to standing and the humanoid can only *lunge* forward once and
settle). Switching to **direct gravity-compensated torque** (no `q0` anchoring) and training SAC with a
dominant forward reward yields a **sustained forward-leaning dynamic walk**:

| controller / learner | forward | over | speed | behaviour |
|---|---|---|---|---|
| footstep WBC + CEM (w_upright) | +0.18 m | 25 s | 0.007 m/s | sustained but a crawl |
| **position-servo SAC** (w_vel 50, δ 0.7) | +0.08 m | 0.3 s → settles | — | **lunge-and-settle** (~8 cm, then q0 pulls back) |
| **torque SAC** (w_vel 50, δ 0.7) | **+0.71 m** | ~2.0 s | **0.36 m/s** | **sustained walk** (distance grows: 0.08→0.15→0.20→…→0.48→0.71), falls at ~2 s |

The torque walker covers **0.71 m at 0.36 m/s** — a real (if eventually-falling) walking gait, ~9× the
position-servo lunge and ~50× the footstep crawl's speed. Rendered: `humanoid_torque_walk.mp4` +
keyframes (legs alternating in stride, a forward-leaning dynamic gait).

## The diagnosis (why the ceiling was NOT the learner)

The position-servo SAC was pushed hard — **dominant** forward reward (`w_velocity=50`), higher action
authority (`delta_scale=0.7`), higher velocity cap, 120 k SAC steps — and still plateaued at ~8 cm and,
over a long rollout, *settled* (+0.076 m at 0.3 s but only +0.067 m at 1.2 s: it lunges then stops). CEM
and SAC hit the *same* ceiling ⇒ not the learner. Root cause: `tau = kp·(q0 + a·δ − q) − kv·q̇ + bias`
anchors the body to the standing pose, so a sustained gait (repeatedly leaving `q0` to swing the legs) is
fought by the servo. **Torque control removes the anchor** and the same SAC learns to walk.

## What was built

- `balance_env`: **`torque_action`** config + step branch — `tau = a·tau_max + gravity_bias` (direct,
  gravity-compensated, no `q0` anchoring). Default `False` ⇒ the position-servo path is unchanged.
- `run_humanoid_walk_sac.py`: direct-locomotion SAC over `balance_env` (reuses `hymeko_rl.train.sac`),
  checkpoint selected by held-out **forward distance**; args `--torque`, `--w_velocity`, `--vel_cap`,
  `--delta_scale`, `--steps`.

(Earlier in the session, committed at `fc59e6ca`: `w_upright` (footstep), `w_velocity`/`vel_cap`
(balance), `train_balance_walk`, and the footstep-uprightness result.)

## Files touched

| File | +/− | notes |
|---|---|---|
| `scenarios/humanoid/balance_env.py` | +11 / −4 | `torque_action` config + step branch (default-off) |
| `scenarios/humanoid/run_humanoid_walk_sac.py` | +100 / new | direct-locomotion SAC (`--torque`/`--w_velocity`/`--vel_cap`/`--delta_scale`) |
| `reports/2026-08-06-humanoid-torque-walk.md` | new | this report |

## CORE.YAML / dependencies

None. Reuses `hymeko_rl.train.sac` (already present; torch pinned). `torque_action` defaults off (no
regression). No new dependency.

## Test / gate results

- No regression: `pytest tests/test_footstep_planner.py tests/test_stepping_stone_demo.py` → **9 passed,
  2 skipped** (torque_action off ⇒ position-servo unchanged). `ruff check` → clean.
- **Production-scale**: SAC trained at real scale (300–400-step episodes, 120–150 k steps), checkpointed,
  held-out forward-distance selection; torque best re-measured over a 2500-step rollout (the sustained-walk
  evidence above).

## Honest scope / negatives (guard)

- The torque walk **falls at ~2 s** — it is a real forward gait, *not* indefinite walking, and it leans
  forward (a dynamic, not statically-erect, gait). Do not overclaim a robust biped.
- **b+c (toe-off push-off + curriculum) was a NEGATIVE**: on the articulated-toe model the CEM fell
  backward (`best_fwd ≈ −0.67`); the 3-dim toe action + toe dynamics are harder for CEM in this budget.
  The win came from the *control scheme* (torque), not the toe model.
- Trained policies live in the scratchpad (not committed); the numbers + the reusable `torque_action` mode
  + the SAC runner are the deliverable.

## Follow-up

- **Indefinite walking**: add a healthy/upright shaping term (or a periodic-gait prior) so the torque gait
  does not fall at ~2 s; longer SAC / a curriculum on episode length.
- **Erect gait**: penalise the forward lean (as `w_upright` did for the footstep gait) within the torque
  reward for a more upright walk.
- **Toe/arm push-off** on the *torque* controller (where it can actually help), not the servo.

## Provenance

Git SHA `fc59e6ca` at start. Python: master venv (CPython 3.11, mujoco 3.10.0, torch 2.12.0 CPU, numpy 2).
SAC via `hymeko_rl.train.sac`, ~250 steps/s CPU, checkpointed. Host macOS (darwin 25.5), Apple Silicon.
Deterministic per seed (torch.manual_seed(0)). timestep 0.001 s.
