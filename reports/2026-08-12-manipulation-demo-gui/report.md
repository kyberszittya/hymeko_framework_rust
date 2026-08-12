# Manipulation demo GUI (interactive, matplotlib)

**Date:** 2026-08-12 · **File:** [hymeko_rl/gui/manipulation_demo.py](../../hymeko_rl/gui/manipulation_demo.py)
**Purpose:** a live, presentable demo of the HyMeKo planar two-arm delivery over the shape families.

![preview](demo_preview.gif)

## What it does

An interactive matplotlib window:

- **Object shape** (radio buttons): coin / square / triangle / pentagon / hexagon / ellipse / capsule — the seven
  curriculum shapes, all HyMeKo-generated and equal-area to the coin.
- **Target**: click anywhere in the workspace to set the delivery target.
- **Strategy** (radio buttons): TD3 / k×n actor-critic / SAC.
- **RUN ▶**: animates the two 2R arms reaching a straddle grasp and carrying the object to the target; **reset** clears.
- A faint green region marks the **both-arm delivery workspace** (object centres where both fingertips can grip) — the
  region to click inside. A target outside it is carried as far as reachable (flagged in the title).

## Run it

```bash
cd /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_coin_r9_wt
PYTHONPATH=. /Users/kyberszittya/hakiko_ai_ws/03_implementation/hymeko_framework_rust/.venv/bin/python \
    -m hymeko_rl.gui.manipulation_demo
```

(Uses matplotlib's native window backend on macOS — no extra install; matplotlib/mujoco already in the venv.)

## Honesty note (state it in the talk)

- The arms use the **real calibrated 2R inverse kinematics** (`PlanarArm2R`, calibrated from the MuJoCo model) and the
  **real HyMeKo-generated object geometry** (the same `ObjectSpec` / boundary the physics uses).
- The carry is a **geometric preview** (kinematic transport of the grasped object), **not** a trained-policy physics
  rollout. Robust delivery of *arbitrary* shapes to *arbitrary* targets is the open research this framework targets
  (the round family does not even certify a straddle yet — see `reports/2026-08-12-round-family-ellipse-capsule/`).
- The **strategy selector** previews the target architectures: **TD3** is the coin's deployed residual policy; **k×n
  actor-critic** (Track C) and **SAC** (option-RL) are in development. The selector changes the labelled strategy; the
  motion shown is the shared geometric preview (the strategies are not yet trained to drive arbitrary shape+target
  delivery).

So the demo honestly shows: *HyMeKo generates any of these shapes; you pick a target and a strategy; the two-arm system
plans a straddle grasp and delivers it* — with the geometric/architectural scaffolding real and the learned control the
stated next step.

## Verification

- Headless smoke: all **7 shapes × 3 strategies** run without crash; delivery to an in-workspace target reaches it
  (0 mm error) with both arms gripping (~55–66 mm straddle standoff); out-of-workspace targets clamp gracefully.
- `ruff` clean. Delivery workspace ≈ `x∈[−0.09, 0.12], y∈[0.02, 0.20]` (209 sampled grip points).

## Process note

Given the same-day demo deadline, the 4-format plan step (CLAUDE.md §2) was **streamlined** for this
presentation/demo tool (non-core, `hymeko_rl/gui/`): recon → build → headless-verify → this report. No core files, no
new dependencies. If the demo graduates into a maintained tool, a retro plan can be added.

## Follow-ups

- Wire the **real coin TD3 rollout** (physics) for shape=coin + strategy=TD3 (the one genuinely deployed policy), so at
  least one selector runs true physics.
- Larger delivery workspace (the both-arm grip region is small — the arm reach limits it); or a base-relocation demo.
- Round-family shapes render + carry, but their real *acquisition* doesn't certify (documented); a round-aware grasp
  would let them be gripped in physics, not just the preview.
