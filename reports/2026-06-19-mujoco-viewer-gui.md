# Report — interactive MuJoCo viewer GUI for the arm simulations

**Date:** 2026-06-19 · **Plan:** `docs/plans/2026-06-19-mujoco-viewer-gui/` (4 artifacts, compiles)
**Author:** Aiko (agent), for Dr.\ Csaba Hajdu
**Status:** ✅ **Built, loop unit-tested.** ⚠️ The live window is **not agent-verified** — it needs
a display and cannot be smoke-launched headlessly. Run command below.

## Summary
Added `hymeko_rl/viewer.py`: a live, interactive window (orbit / zoom / pan / pause) via MuJoCo's
native `mujoco.viewer.launch_passive` (present in 3.9.0; **zero new dependency**). It is a new
*sink* over the existing rollout toolkit — `build_render_env` (emit + beautified scene) and the
`ActionFn` sources from `render_reach.py` — so no logic is duplicated; `render_reach.py` is
imported, not modified.

The loop is split for testability:
- **`drive_viewer(env, action_fn, handle, *, max_steps, realtime, sleep_fn, seed, on_reset)`** —
  pure, GL-free: act → `env.step` → `handle.sync()` → real-time pace → reset on episode end;
  invokes the optional `on_reset(handle, info)` hook after the initial reset and each episode reset;
  returns steps driven. Unit-tested with a fake handle.
- **Target marker** — `draw_target(handle, info)` (re)draws the reach target as a sphere in the
  viewer's persistent `user_scn` (clears `ngeom` then appends one geom, so a new episode's target
  replaces the old). Wired into `launch` via `on_reset=draw_target`, so the live window shows the
  target the policy is reaching for (matching the offscreen render, which already drew it). Reuses
  `render_reach._draw_target` for the geom init — no duplication.
- **`launch(env, action_fn, ...)`** — the thin GL wrapper that opens `launch_passive` (Handle used
  as a context manager) and runs `drive_viewer`. Not unit-tested (needs a display).
- Sources: `expert` / `bc` (reused from `render_reach`) + new `random` ("is it alive") and `zero`
  (settle under gravity).

## Setup (uv)
`pyproject.toml` has `dependencies = []` and `default-groups = ["dev"]`, so a bare `uv sync`
installs only `dev` (pytest/mypy/…) — **not** torch and not the sim deps. Name the groups:

```
uv sync --group ml --group rl       # ml = torch + HSiKAN policy; rl = MuJoCo + Gymnasium
```

Activate the venv (`uv sync` creates `.venv`):

```
# Windows — PowerShell
.venv\Scripts\Activate.ps1
# Windows — cmd
.venv\Scripts\activate.bat
# Linux / macOS — bash/zsh
source .venv/bin/activate
```
(Or skip activation entirely and prefix commands with `uv run`, e.g. `uv run python -m hymeko_rl.viewer …`.)
`viewer.py` → imports `render_reach` → imports `torch`, so `ml` is required even for the expert
source. `mujoco`/`gymnasium` live in the new `rl` dependency group
(`APPROVED-CORE-EDIT: rl-deps-group`, 2026-06-19 chat); `uv lock` resolves them — mujoco 3.9.0,
gymnasium 1.3.0, plus `glfw`/`pyopengl` for the GL backend the live window uses. Add `--group demo`
only if you also want MP4 render (`render_reach.py`).

## Run it (user — needs a display)
```
python -m hymeko_rl.viewer --robot data/robotics/anthropomorphic_arm.hymeko --source expert
python -m hymeko_rl.viewer --source random --no-realtime      # fast sanity
python -m hymeko_rl.viewer --robot default --source bc        # bare 4-DOF arm, learned policy
```
`--control` (position default), `--pretty/--no-pretty`, `--ee-body`, `--seed` mirror `render_reach`.
On a headless host use `render_reach` (GIF/MP4) instead.

## Files touched
| file | change |
|---|---|
| `hymeko_rl/viewer.py` | NEW (~140 LOC) — `drive_viewer` (+`on_reset` hook), `launch`, `draw_target`, `random_source`/`zero_source`, `make_source`, CLI |
| `hymeko_rl/tests/test_viewer.py` | NEW — 11 tests (fake-handle drive loop + target marker) |
| `docs/plans/2026-06-19-mujoco-viewer-gui/{plan.tex,pdf,tikz,mmd}` | NEW plan (compiles) |

## Interface / contract
- `ViewerHandle` — a structural `Protocol` (`is_running`, `sync`); MuJoCo's `Handle` satisfies it,
  a fake satisfies it in tests, keeping the loop GL-free.
- `drive_viewer` postcondition: syncs once per step, stops on `is_running()→False` or `max_steps`,
  resets the env on `terminated ∨ truncated`; mutates no MuJoCo state except via `env.step/reset`.
  `ValueError` on `max_steps < 1`.

## Test results
- `pytest -p no:randomly hymeko_rl/tests/test_viewer.py` — **11 passed**, 10.1 s:
  syncs once per step; stops when the handle stops; `max_steps` bound; `max_steps=0`→`ValueError`;
  **reset-on-done** (FakeEnv terminating every 2 steps → reset_count correct); realtime pacing calls
  `sleep_fn` each step with `dt>0`; sources return actions in `action_space` shape; **target marker**
  — `draw_target` adds exactly one geom to a real (headless) `MjvScene` and replaces (not
  accumulates) on redraw, no-op without a target; **`on_reset` fires** on the initial reset + each
  episode reset (1+3 over 6 steps).
- **Static:** `ruff check` clean. `mypy --strict hymeko_rl/viewer.py` — **clean** (no own errors;
  the only flags are the documented `mujoco` import-untyped baseline). Note: mypy surfaces
  **pre-existing** `imageio` import nits in `render_reach.py` (lines 134/141, optional MP4 import) —
  not introduced here; they belong to the render task's file and are out of scope for this change.

## CORE.YAML / dependencies
The viewer **code** needs no new dependency (`mujoco.viewer` ships with MuJoCo). Separately, on the
user's request, `mujoco`/`gymnasium` were formalised into a new `rl` dependency group in
`pyproject.toml` — a CORE.YAML §1 dependency edit, made under **`APPROVED-CORE-EDIT: rl-deps-group`**
(2026-06-19 chat). They were already installed ad-hoc; this just declares them so `uv sync --group
ml --group rl` reproduces the env. `uv lock` resolves clean (mujoco 3.9.0, gymnasium 1.3.0,
+ glfw/pyopengl). No CORE crate/spec file touched.

## §6.5 anti-patterns
None. Sources are pluggable `ActionFn`s (not a `_kind` branch, #7); the loop is one entry point
with injected `sleep_fn`/handle (Strategy, testable); discovery confirmed no `mujoco.viewer` usage
anywhere before creating the module (#12); `render_reach` reused, not copied (#1).

## Open / follow-up
- **Window not agent-verified** (headless). The user runs the command above; the drive loop's
  correctness is covered by the fake-handle tests.
- A `key_callback` (e.g. `r` to reset, `space` already pauses) is a small follow-up if wanted.
- Carries over unchanged to the Galambos planar grasper and grasp-ball scenes once their envs exist.

## Provenance
- Git SHA `7d16ad0` (working tree dirty; `hymeko_rl` uncommitted increment). MuJoCo 3.9.0,
  Python 3.12, Windows 11. Seeds 0 (tests use fixed counts).
