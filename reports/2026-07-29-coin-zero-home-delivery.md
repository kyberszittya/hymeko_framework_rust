# R11.0–R11.1 — exact-zero HOME → strict-K6 baseline + robust RRT-Connect reach across the coin grid

**2026-07-29 · branch `recovery/coin-r9-causal-residual-delivery` · parent `fb681825` · downstream / capture-CEM / physics reused · no snapshot teleport · first frame exactly `q=[0,0,0,0]`**

## Summary

Two results, cleanly separated. **(R11.0)** an honest *positive control*: starting from the exact zero reset `q=[0,0,0,0]`, a Cartesian fingertip reach establishes a valid straddle READY, a READY-specific CEM capture is re-solved, and the frozen downstream delivers strict K6 — no staging pose, no injected snapshot, first frame literally zero. **(R11.1)** a *robust reach planner*: a 4-DOF joint-space bidirectional RRT-Connect that replaces the hand-tuned 3-shell topology and finds collision-free reaches from the exact zero home across the admissible coin grid.

**Delivery remains canonical-only and is not addressed by this milestone** (the learned R2 residual + the handoff orbit are coin-specific). **CEM is still a teacher/oracle for the capture — not teacher-free deployment.**

## R11.0 — exact-zero positive control

Verified continuous chain:

```
EXACT_ZERO_HOME [0,0,0,0], qdot=0
  → Cartesian task-space reach
  → READY (precontact straddle)
  → READY-specific CEM capture (re-solved per READY)
  → frozen downstream (HANDOFF_RESET → R2 → coast)
  → strict K6
```

Measured nominal result (canonical coin):

| quantity | value |
|---|---|
| first frame | exactly `q=[0,0,0,0]`, `qdot=0` |
| staging reset / snapshot teleport | **none** |
| pre-capture coin displacement | **0.0 mm** |
| premature contacts | **0** |
| minimum fingertip clearance | **47.9 mm** |
| strict K6 | **true** |
| min_dtz | **11.54 mm** |
| safe | **true** |

Video: `reports/2026-07-29-coin-zero-home-delivery/video/zero_home_to_k6.{mp4,gif}` + `manifest.json` (first frame `[0,0,0,0]`; the codebase's `assert_trace_render_consistency` gate passes: coin dtz spans ~74 mm, frames vary).

**Honesty:** the capture is solved by CEM search (an oracle/teacher), not by a deployed policy. The learned/RL component in this chain is the frozen **R2** residual in the downstream — load-bearing for K6.

## R11.1 — robust RRT-Connect reach

Planner: 4-DOF joint-space **bidirectional RRT-Connect**; exact start `[0,0,0,0]`; a **goal SET** generated from multiple collision-free precontact/straddle configurations (sampled assigned-side angles × shell clearances × both exact planar-2R IK branches); **inflated-coin** link + fingertip collision checking; **inter-arm** collision checking; shortcut smoothing + densification; governed continuous execution. (`continuous IK branch not required — q itself is the search state`.)

Reach grid sweep (planning; execution clearance in the governed rollout):

| coin offset | RRT | execution |
|---|---|---|
| canonical | 44 goals, 25 wp, 0.5 s | clr 43 mm, coin +2.3 mm, 0 contact |
| left −3 / −5 cm | 24 / 25 goals | clr 46 / 39 mm, coin 0 / 4 mm, 0 |
| right +3 cm | 37 goals | clr 44 mm, coin 0 mm, 0 |
| **down −2 / −4 cm** | 33 / 19 goals | **clr 46 / 49 mm, coin 0 mm, 0** (the fixed 3-shell planner failed here) |
| up +2 / +3 cm | 36 / 38 goals | clr 42 / 43 mm, coin 0 mm, 0 |
| diagonal +3,+3 cm | 44 goals | clr 56 mm, coin 0 mm, 0 |
| right +5 cm | — | **start-in-collision** (see below) |

- **9/10 tested coin placements planned successfully**, planning ≈ **0.5 s** each.
- The **down-shifted cases that defeated the fixed three-shell planner are solved.**
- Most successful executions move the coin **0 mm** before capture; a few nudge it **~2–4 mm** (servo overshoot dipping to ~39–43 mm vs the 40 mm contact distance) — a remaining execution-clearance issue, **not yet a formal zero-motion pass across the whole grid**.
- **right +5 cm** is rejected because the exact zero HOME itself starts in collision with the coin (the coin is placed inside the fully-extended arm). This is an **invalid initial condition** under the current task contract, **not** an RRT failure.

**Verdict: `ZERO_HOME_RRT_REACH_GRID_PASS_WITH_CLEARANCE_TIGHTENING_PENDING`.**

> The hand-written three-shell topology is not required for general reach. Configuration-space RRT-Connect robustly finds collision-free paths from the exact zero reset across the admissible tested coin grid.

> Delivery remains canonical-only and is not addressed by this commit.

## Files

| file | role |
|---|---|
| `hymeko_rl/experiments/coin_zero_home_reach.py` | exact-zero Cartesian 3-shell reach + CEM capture (R11.0 positive control); `do_reach`/`reach_and_deliver`/`solve_capture` |
| `hymeko_rl/experiments/coin_zero_home_rrt.py` | 4-DOF RRT-Connect reach (R11.1): collision checker, goal set, `rrt_connect`, shortcut/densify, `reach_rrt` |
| `hymeko_rl/experiments/video_coin_zero_home.py` | honest zero-home delivery video (reach → capture → downstream → K6) with the trace-consistency gate |
| `hymeko_rl/coin_delivery/theta_option/torque_path_option.py` | **+6/−1**: a read-only `frame_hook` on `TorquePathCaptureRoll.rollout` (identity-preserving; needed by the video) |
| `hymeko_rl/tests/test_coin_zero_home.py` | exact-reset gate, seg-seg distance, collision checker, goal set, RRT-path, and the strict-K6 positive control |
| `reports/2026-07-29-coin-zero-home-delivery/…` | this report + the video artifacts + manifest |

## Tests & checks

- **`ruff check`: clean** (all new files + the modified module). **`radon cc -a -nc`: no C+ block.**
- **Fast tests (5) pass**: exact-reset (`q=[0,0,0,0]`), segment-segment distance, inflated-coin collision checker (start clear / tip-on-coin rejected), non-empty valid goal set, RRT-Connect returns a collision-free path from exact zero to a goal.
- **Physics positive control**: `reach_and_deliver` → first frame `[0,0,0,0]`, 0 premature contacts, pre-capture coin < 5 mm, **strict K6, min_dtz < 20 mm, safe**.
- **Video consistency gate**: `assert_trace_render_consistency` passes in the render (manifest).
- **Regression**: the torque_path identity gate (`TORQUE_PATH_ZERO_DELTA_IDENTITY` / `BIT_EXACT_SCAFFOLD` / `OPTION_ZERO_POLICY_K6`) still **PASS** after the read-only `frame_hook` addition.

## Known limitations

- **~2–4 mm pre-contact coin nudge** in a few RRT executions (servo overshoot near the coin); a few mm more obstacle inflation / gentler final approach is the fix. Not yet a formal zero-motion pass across the whole grid.
- **right +5 cm** invalid initial condition (zero home collides with the coin) — a task-contract boundary, not a planner failure.
- **Capture / delivery is still canonical-specific** (learned R2 + handoff orbit). Arbitrary-workspace *delivery* is **not** solved here; only *reach* generalizes.

## Provenance

- Parent `fb681825`; this milestone committed on top. CORE.YAML items touched: none. Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). Deterministic (fixed seeds; RRT seed 0). Coin canonical `[0.07578, 0.14279]`; grid = canonical ± {2,3,4,5} cm in x/y and a diagonal.
