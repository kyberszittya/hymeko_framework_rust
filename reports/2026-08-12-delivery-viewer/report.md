# Delivery viewer — REAL trained coin-delivery physics (mujoco.viewer)

**Date:** 2026-08-12 · **File:** [hymeko_rl/gui/delivery_viewer.py](../../hymeko_rl/gui/delivery_viewer.py)

Replaces the earlier matplotlib kinematic preview (deleted). This is **real MuJoCo physics**, driven by the **actual
trained delivery pipeline** — no reimplementation, no kinematics. It is thin glue that *composes* existing library
functions.

## What it runs (the real pipeline it composes)

For a chosen object shape + target scenario + strategy:

1. `reconstruct_capture(rig, …)` — exact-zero home → RRT reach → **certified capture** (real physics).
2. the deployed **descriptor-nearest retrieval θ** (the frozen coin policy) *or* the scenario's **teacher (CEM) θ**.
3. `rollout_primitive(snap, θ, CLOSED_LOOP_CFG, frame_hook=…)` — the governed **Δτ transport** (real physics, the
   frozen `COIN_DYNAMICS_CONTRACT_V2` slew/governor).

Every step's full `qpos` is recorded and replayed in a live **`mujoco.viewer`** window (orbit / zoom / pan), or rendered
offscreen via **`mujoco.Renderer`** to a GIF. The K6 verdict is the frozen monitor's (`delivery_success`).

## Run it

Live viewer (macOS needs the viewer on the main thread → **mjpython**):

```bash
cd /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_coin_r9_wt
PYTHONPATH=. /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/mjpython \
    -m hymeko_rl.gui.delivery_viewer
```

Terminal keys: `s` cycles shape, `t` cycles target scenario, `g` cycles strategy, ENTER runs the real delivery (opens
the viewer), `q` quits.

Headless GIFs (no display, canonical offscreen render):

```bash
PYTHONPATH=. …/.venv/bin/python -m hymeko_rl.gui.delivery_viewer --render
```

## Verified — real physics, honest per shape

Measured deliveries (target `bank_c1_+0.03_-0.02`, the frozen deployed policy / teacher θ):

| shape | strategy | K6 | dtz_end | note |
|---|---|---|---|---|
| **coin (O0)** | TD3 (deployed retrieval) | **✓** | 18.4 mm | the deployed policy delivers |
| coin (O0) | teacher (CEM θ) | ✓ | 6.4 mm | teacher tighter |
| **triangle (O6-T)** | teacher (CEM θ) | **✓** | 6.6 mm | a non-circular shape delivers with the teacher θ |
| triangle (O6-T) | TD3 (deployed retrieval) | ✗ | 64.3 mm | coin's retrieval θ doesn't transfer |
| square (O4-S) | either | ✗ | 42 / 69 mm | needs its own bank (physical delivery runs, no K6) |
| ellipse/capsule (O3-E/O9-K) | — | — | — | **no certified grasp** (the round-family certification wall) |

This is the honest state: the **coin is fully deployed** (both strategies K6); a **non-circular shape (triangle)
delivers with the teacher θ**; the box needs its own bank; the round family does not certify a straddle (documented in
`reports/2026-08-12-round-family-ellipse-capsule/`). The demo shows the **real physics** and the **real research state**,
not a scripted animation.

## Notes

- Strategies: `TD3 (deployed retrieval)` = the frozen descriptor→θ table (the coin's deployed policy);
  `teacher (CEM θ)` = the scenario's teacher θ. Both are real and run the same governed physics. (k×n actor-critic / SAC
  are not part of the coin's θ-primitive deploy — Track C is in development.)
- Composition only — imports `reconstruct_capture`, `rollout_primitive`, the retrieval table, `_cam`, `encode_clip`,
  `_rig`, `variant`. No physics/rollout/render logic is reimplemented.
- CORE.YAML: none touched (`hymeko_rl/gui`, non-core); no new dependency (mujoco/imageio/Pillow already present).
