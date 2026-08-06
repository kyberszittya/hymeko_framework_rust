---
title: Fast-Dynamics Control Demonstration — Foundation (two-scale autonomous vehicles + envs + proven RL pipeline)
date: 2026-07-15
status: milestone (envs+plants+experts+tests done for 5 substrates incl. 2 vehicle scales; RL pipeline de-risked; campaign next)
plan: docs/plans/2026-07-15-fast-control-transfer-demo/
core_yaml_touched: none
---

> **Update (vehicle-first, deep).** Per the user's steer, the vehicle is the centerpiece, and the user
> specified **two scales**: a full-scale car at **~200 km/h** and a **Tamiya F1/10 (F1TENTH)**. Both are
> built and tested — see "Two-scale autonomous vehicles" below. 37 tests pass.

# Fast-Dynamics Control Demonstration — Foundation milestone

**Scope of the overall task (user request):** a control demonstration built from the project's own
RL (SAC / TD3 / DAgger) + Nagare + CIP + HyMeKo machinery on a *very fast vehicle* and a *humanoid* (plus
cheetah), benchmark-grade, **prepared to run on kato15 (RTX6000)**. After a scripted-expert finding
(below), the user chose **vehicle-first, deep**: the diff-drive vehicle is the polished centerpiece;
cheetah/humanoid are SAC/TD3-from-scratch secondary.

This report covers the **foundation** that is built, tested, and de-risked. The training campaign,
HyMeKo FSM/monitor wiring, transfer adapters, and Nagare/CIP harnesses are the next milestones (§Open).

## Summary

- **§2 plan bundle** produced in all four formats (`plan.tex` → `plan.pdf` via tectonic — no `pdflatex` on
  this host; `plan.tikz` compiles standalone; `plan.mmd` validates as a Mermaid sequence diagram).
- **Three plants, mostly reuse**: vehicle = `robot_4wh.hymeko` (diff-drive, reused); humanoid =
  `humanoid.hymeko` (13-DOF biped, reused); **one new** `half_cheetah.hymeko` (planar 6-DOF), authored on the
  exact `humanoid.hymeko`/`quadruped.hymeko` pattern. All compile to MuJoCo and settle stably.
- **Env layer** (`hymeko_rl/env/locomotion_env.py`): a *factored* `LocomotionEnv` base (no triplication,
  §6.5 #3) + `LeggedLocomotionEnv` + `WheeledVehicleEnv`, speaking the repo's `(N,2)` per-vertex hypergraph
  obs + declarative `RewardSpec` + trainer 5-tuple contract, reusing `arm_world`/`quadruped_env` helpers.
- **Scripted experts** (`hymeko_rl/env/locomotion_experts.py`): a CPG gait (legged) + pure-pursuit (vehicle).
- **Vehicle centerpiece works**: the pursuit controller completes a **full 15/15-waypoint circular lap** at
  **2.85 m/s mean (3.0 peak)**, dead upright. Numbers + trajectory plot + speed plot + a **960→640×480 GIF**
  under `reports/figures/2026-07-15-fast-control/`.
- **RL pipeline proven end-to-end**: `train_offpolicy` (TD3) trains directly on the vehicle env, flattening
  `(N,2)`→12, emitting the mandated live line `step | crit/act loss | steps/s | ETA | buf` and passing the
  tensor-contract schema check. This is the §3 "smoke a new code path before scaling" gate for the trainers.

## Files touched

New (all non-core):

| File | Lines | Purpose |
|---|---|---|
| `docs/plans/2026-07-15-fast-control-transfer-demo/plan.{tex,pdf,tikz,mmd}` | 185 (tex) | §2 plan bundle |
| `data/robotics/half_cheetah.hymeko` | 57 | new planar cheetah plant |
| `hymeko_rl/env/locomotion_env.py` | 401 | `LocomotionEnv` base + 2 substrates + factories |
| `hymeko_rl/env/locomotion_experts.py` | 105 | CPG gait + pure-pursuit scripted experts |
| `hymeko_rl/tests/test_locomotion_env.py` | 168 | 24 contract/regression tests |
| `reports/figures/2026-07-15-fast-control/*.{png,gif}` | — | vehicle trajectory, speed, lap GIF |

Reused, unchanged: `data/robotics/{robot_4wh,humanoid}.hymeko`; `hymeko_rl/{train,agents,eval,env/arm_world,
env/quadruped_env,env/reward}`.

## CORE.YAML items touched

**None.** Core is `hymeko_core / hymeko_query / hymeko_client / hymeko_daemon / parser` + spec/RTL files;
nothing here touches them.

## Test results

- **Unit + integration + performance** (`pytest -p no:randomly hymeko_rl/tests/test_locomotion_env.py`):
  **24 passed in ~1.0 s.** Covers: construction/space dims (cheetah 7×2→6, humanoid 13×2→12, vehicle 6×2→4),
  seed-determinism, 5-tuple contract, expert-action validity, legged fall-termination, vehicle upright
  regression, vehicle waypoint progress, privileged-state, CPG determinism, pursuit turn-direction
  regression, and a **step-latency budget** gate (median < 5 ms).
- **Static analysis**: `ruff check` clean on all three new Python files (§6.3).

## Performance results

- **Env step latency (median)**: cheetah 0.07 ms, vehicle 0.04 ms, humanoid 0.18 ms — all far under the 5 ms
  budget. Worst case (humanoid, 348-equiv DOF) has ample headroom.
- **TD3 training throughput** (vehicle, CPU smoke): ~900–1100 steps/s after warmup (26 k/s during the
  random start phase). Peak RSS well under the 16 GB cap (small MLP + one MuJoCo model).
- **Vehicle scripted ceiling** (the metric to rank policies against, per §3): full lap, 2.85 m/s mean.

## Key finding (shaped the plan)

The scripted experts are **weak** as BC anchors: the open-loop CPG legged gait **falls in ~40 steps** with
near-zero net progress; the *initial* pure-pursuit vehicle controller drove upright but tracked only gentle
curvature. This is measured, not assumed. Consequences, adopted:

1. The **legged** demonstration leads with **SAC/TD3 from scratch** (the canonical way HalfCheetah/Humanoid
   are solved), with the CPG as an honestly-labeled weak reference — *not* a primary BC anchor (a weak anchor
   traps TD3+BC, per the coin-toss/pick-place record).
2. The **vehicle** became the deep centerpiece; investing in its controller was then justified, and it now
   completes clean fast laps → a genuine scripted ceiling and a usable DAgger teacher.

## Bugs found and fixed (with regression tests)

- **Runaway torque wheels → chassis flip.** The emitter produced torque `<motor>` wheels though the
  `.hymeko` declared *velocity* control; unbounded torque spun the light 2 kg wheels to ~200 rad/s (slipping,
  no traction) and the reaction torque flipped the 25 kg chassis (upright 1.0→0.16 by step 37). **Fix:**
  convert wheel actuators to `<velocity>` servos in the vehicle MJCF prep — the honest realization of the
  declared diff-drive velocity interface. Regression: `test_vehicle_stays_upright_under_pursuit`.
- **Wheel-side steering sign error.** The vehicle turned to −y for a +y waypoint (wheel groups assigned by
  name, not the `robot_4wh` geometry where +y is the nominal "right" side). **Fix:** the CCW-turn-fast group
  is the −y-side wheels {fl,rl}. Regression: `test_diffdrive_pursuit_steers_toward_waypoint`.

## Two-scale autonomous vehicles (added on the vehicle-first steer)

The user asked for two vehicle tests at two scales. The original diff-drive `robot_4wh` **cannot** do high
speed (measured: it wheelies and backflips above ~5 m/s regardless of ramp/friction — a short wheelbase +
high CG is a wheelie machine), so each scale got a purpose-built HyMeKo plant + env:

- **`race_car.hymeko` (full-scale, ~200 km/h).** Long wheelbase (1.8 m), low CG, big wheels (0.33 m),
  300 kg — built for high-speed stability. Verified: **accelerates to ~209–260 km/h, dead upright, no flip**,
  over 2.5 km. Straight drag course; velocity-servo wheels with a throttle ramp. Artifacts:
  `racecar_acceleration.png` (the acceleration-to-200-km/h curve — the "test" result), `racecar_sprint.gif`.
- **`f1tenth.hymeko` (1/10 Tamiya F1/10 / F1TENTH).** The realistic autonomous-racing platform: **Ackermann
  steering** (front wheels steer via nested steer→roll knuckle joints, rear-wheel drive), wheelbase 0.32 m,
  wheel radius 0.05 m, ~3.4 kg, ~2.5 m/s. Control is the F1TENTH **2-D `[throttle, steer]`** interface
  (`AckermannCarEnv` overrides `_action_dim`/`_apply_control`; the env rewrites emitted motors → rear
  velocity + front position servos and strips the free-roll motors). It tracks a gentle Ackermann lap
  upright. Artifacts: `f1tenth_lap_trajectory.png`, `f1tenth_lap.gif`.

Two physics bugs found + fixed here (both regression-tested): the F1TENTH's light 0.2 kg wheels blew up
numerically under a stiff servo (QACC NaN, 390 m/s) — fixed with armature on **every** wheel dof (incl. the
free front wheels the base skipped) + a low gain; and a standing-start wheelie — fixed with a drive-torque
limit (`forcerange`, i.e. a realistic motor torque cap). This is the same "scale-changes-the-physics" lesson
as the leg experts: measured, not assumed.

Substrate roster (all trainer-ready `make_*` factories in `SUBSTRATES`): `cheetah`, `humanoid`, `vehicle`
(diff-drive), `racecar` (200 km/h), `f1tenth` (Ackermann). **37 tests pass**, ruff clean.

**Rendering + live sim.** First-pass GIFs were near-invisible (a vehicle on a blank floor reads as frozen)
and short. Fixed with two pieces:

- `hymeko_rl/viz/locomotion_render.py` — HD renderer over a **checker floor** (the scrolling ground is the
  motion cue) with the **track drawn** (waypoint pylons + racing line), writing compact **MP4** (via
  imageio-ffmpeg) or GIF. Reuses `scene_style.beautify_mjcf` + `eval.evaluate._write_gif` (no render loop
  reimplemented); drops the env's blank collision plane from the render model and pins the checker floor to
  its z. Artifacts (960×720, 250–460 KB each): `racecar_sprint.mp4`, `f1tenth_lap.mp4` (overhead of the whole
  loop), `vehicle_pursuit_lap.mp4`.
- `hymeko_rl/gui/vehicle_sim.py` — a **runnable interactive** MuJoCo viewer (orbit/pan/zoom) with the track
  pylons drawn live. Run on macOS via the bundled mjpython:
  `.venv/bin/mjpython -m hymeko_rl.gui.vehicle_sim --vehicle {racecar|f1tenth|vehicle|cheetah|humanoid}`.
- `hymeko_rl/gui/vehicle_qt.py` — a **native PySide6 desktop app** (runs on normal python; offscreen
  `Renderer`→`QLabel`, main-thread `QTimer`, the `qt_sim` precedent): vehicle selector, **terrain** toggle,
  camera modes, live telemetry (speed/lap/tilt), play/pause/reset. `./.venv/bin/python -m hymeko_rl.gui.vehicle_qt`.

**Terrain (MuJoCo heightfields).** MuJoCo supports terrain natively via `<hfield>`; `hymeko_rl/env/terrain.py`
fills a procedural elevation grid (`hills` / `bumps` / `ramps`, with a flattened border skirt) and swaps the
flat floor for a heightfield collision geom. Verified: the diff-drive vehicle and F1TENTH drive over it,
finite + upright (`vehicle_terrain_hills.mp4`). Enabled via `make_vehicle(terrain=...)` / `make_f1tenth(terrain=...)`
and the Qt app's terrain toggle. (Not for the 200 km/h race car — bumps at that speed launch it.)

Videos are compact MP4 (250 KB–1.9 MB).

**Bézier race track.** `hymeko_rl/env/track_gen.py` generates smooth **flat** race tracks from cubic Bézier
splines (Catmull-Rom → Bézier handles → a C1-continuous curve through anchor points, sampled to waypoints —
smooth curvature the pursuit holds at speed, unlike a sharp polygon). `race_circuit(scale)` presets a large
flat GP circuit (long start straight + sweeping corners + a chicane); `make_race_car(course="circuit")` uses
it. Verified: the race car runs the circuit **upright**, hitting **~205 km/h on the straight** and easing to
~30–50 km/h through the chicane — a real racing-line speed profile (a 55 m/s corner needs radius ≳ 300 m, so
the diff-drive can't corner at 200; that's honest physics). Artifacts: `racecar_bezier_track.png` (the
speed-coloured racing line over the Bézier curve), `racecar_bezier_lap.mp4`. The Qt app's racecar now uses it.

**56 tests pass** (`test_locomotion_env` + `test_locomotion_render` + `test_terrain` + `test_track_gen`, incl.
a headless Qt build smoke), ruff clean.

## Open issues / next milestones

4. HyMeKo control substrate: `ControllerSpec`/`HybridAutomaton` FSM (Drive→Recover→Brake) + `Monitor` per
   substrate.
5. `hymeko_transfer` adapters #4 (vehicle) / #5 (humanoid) / cheetah + the **locomotion-contact** extension of
   the `protocol_specificity` taxonomy (do vehicle tire / foot-ground contacts make `CommandSemantics` bind?).
6. **One** campaign entry point (mirror `exp_v25c_asym_rl_campaign`): vehicle {SAC, TD3, TD3+BC, DAgger},
   cheetah/humanoid {SAC, TD3}; observability, GIF/plot/json.
7. Nagare learned-structural-monitor harness (AUROC, rollover/off-track) + CIP/LiNGAM causal diagnosis
   (`DirectLiNGAM` → signed causal `.hymeko`). NB: Nagare is a Rust crate with **no** Python binding — the
   real-crate path is a Rust harness on dumped features; a Python-equivalent is the smoke fallback (plan R1).
8. Local 1-seed production-scale smoke + kato15 launch script (`systemd-run --user -p MemoryMax=16G`,
   multi-seed, resume) — reconcile wall vs the smoke baseline before queuing (§11).

## Provenance

- Git: branch `integration/fanuc-pick-place-canonical`, base SHA `a4e6a9a`; working tree dirty with
  pre-existing branch changes plus the new files above (this work adds only new files; no existing file
  edited except lint auto-fix of the new test).
- Env: Python 3.11 (`.venv`), gymnasium 1.3.0, mujoco 3.10.0, macOS (Darwin 25.5.0, Apple Silicon), CPU.
- Seeds: env determinism seeded (0/7/8 in tests); TD3 smoke seed 0. RL claims will rest on multi-seed
  median/IQR per §3 (none asserted yet — no training beyond the pipeline smoke).
