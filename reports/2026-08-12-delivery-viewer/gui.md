# Delivery GUI (tkinter) — the interactive window

**File:** [hymeko_rl/gui/delivery_gui.py](../../hymeko_rl/gui/delivery_gui.py)

A real interactive window over the same real pipeline as `delivery_viewer.py` — a proper GUI, not terminal keys.

## What it is

- **Object shape** dropdown: coin / square / triangle / pentagon / hexagon / ellipse / capsule.
- **Target map** (top-down canvas): the deployable target scenarios are dots; **click** to pick the nearest one (it
  highlights + names the scenario). Coin start + relocated targets are the real bank targets the deploy covers.
- **Strategy** dropdown: TD3 (deployed retrieval) / teacher (CEM θ).
- **RUN ▶**: the real pipeline runs on a **worker thread** (`reconstruct_capture` → retrieval/teacher θ →
  `rollout_primitive`), the rollout is rendered offscreen (`viz.rollout_film`, verified thread-safe on macOS), and the
  frames play back in the window. Status shows the retrieved demo, K6, and final dtz.

## Run it

```bash
cd /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_coin_r9_wt
PYTHONPATH=. /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/python \
    -m hymeko_rl.gui.delivery_gui
```

(A plain window — no `mjpython` needed; the physics is real, the display is the rendered rollout. tkinter + Pillow are
already present.)

## Thin glue only

Composes: `delivery_viewer.DeployedPolicy` / `record_delivery` (the real pipeline), `viz.rollout_film.render_qpos_seq` +
`top_down` (the consolidated renderer), `dataset.scenario_by_id` (the target geometry). No physics/render/retrieval is
reimplemented — the GUI is UI + threading only.

## Verified (worker path, headless)

- render off the main thread works on macOS (the worker renders while tkinter stays responsive).
- The real delivery worker: **coin K6=True 18.4 mm (90 frames)**; **box K6=False 42.4 mm** (honest — the box needs its
  own bank; the physics delivery + render still run); round shapes → "no certified grasp" (the certification wall).
- ruff clean. CORE.YAML none; no new dependency.

## Three ways to view the same real delivery

- `delivery_gui.py` — this interactive tkinter window (shape + click-target + strategy + in-window playback).
- `delivery_viewer.py` (live) — native `mujoco.viewer` 3D window (`mjpython`), orbit/zoom.
- `delivery_viewer.py --render` — headless GIF/MP4 (the canonical offscreen viz).
