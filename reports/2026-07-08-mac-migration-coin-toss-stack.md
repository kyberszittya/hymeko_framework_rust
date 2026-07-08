# Mac migration — coin-toss / hymeko_rl stack (2026-07-08)

**Created-at:** 2026-07-08 03:07 JST
**Scope:** stand up a runnable `hymeko_rl` (coin-toss / Galambos) environment natively on this
Apple-Silicon Mac (arm64, macOS 26.4), replacing the checked-in Windows `.venv` that was synced across
parallel sessions. User directive: "Migrate the whole here, `.venv` can be deleted and reinstated with a
mac-oriented framework."

## Summary

The Windows `.venv` (194 MB, `Lib/` + `Scripts/` + `z3.exe` layout, no `pyvenv.cfg`) is gitignored
(`.gitignore:30 .venv*/`) — deleting it touched no git state. It was removed and a fresh CPython 3.11.15
arm64 venv created with `uv`. The ML/RL stack was installed **from PyPI CPU/MPS wheels**, bypassing the
`pyproject.toml` `[tool.uv.sources]` CUDA-13.2 Windows index via `uv pip install --no-sources`. No
`CORE.YAML`-governed file was edited: `pyproject.toml`, `uv.lock`, and the pinned torch version
(`torch==2.12.0`, unchanged) are intact, so the Windows/CUDA workspace resolution is preserved for the
parallel sessions.

The `hymeko` Rust CLI (`hymeko_cli` crate) was built (`cargo build -p hymeko_cli` → `target/debug/hymeko`,
25.6 s) — required by env paths that emit MJCF from `.hymeko` files. `hymeko_rl` does **not** import the
PyO3 bindings (`hymeko_py`) or `signedkan_native`; only `torch`/`mujoco`/`gymnasium`/`numpy` plus the CLI.

## Why standalone venv instead of editing pyproject

`pyproject.toml` routes `torch`/`torchvision` through `[[tool.uv.index]] pytorch-cu132` (CUDA-13.2, Windows
wheels only). A plain `uv sync --group ml` on darwin fails to resolve — no arm64 wheel at that index. Editing
the source routing is a pinned-dependency change (`CORE.YAML` §dependencies.pinned.python, torch under
`APPROVED-CORE-EDIT: torch-cuda13`) and would re-lock `uv.lock`, risking the Windows resolution the parallel
sessions depend on. `uv pip install --no-sources` installs the **same** `torch==2.12.0` from PyPI (arm64
CPU/MPS build) into a standalone venv — non-invasive, `CORE.YAML`-safe, reversible, and leaves the Windows
path exactly as it was.

## Commands (reproducible)

```bash
rm -rf .venv                                   # gitignored Windows venv, user-authorized
uv venv --python 3.11                          # → CPython 3.11.15 arm64
uv pip install --no-sources \
  "torch==2.12.0" "numpy>=2,<3" "mujoco>=3,<4" "gymnasium>=1,<2" \
  "matplotlib>=3.8,<4" "imageio>=2,<3" "imageio-ffmpeg>=0.5,<1" \
  "pytest>=8,<9" "hypothesis>=6,<7"
cargo build -p hymeko_cli                       # → target/debug/hymeko (MJCF emission)
```

Run RL code with repo root on the path (workspace is `package = false`, not installed):
`PYTHONPATH=<repo-root> .venv/bin/python -m hymeko_rl.…` (or `.venv/bin/python -m pytest hymeko_rl/tests/…`).

## Resolved environment

- CPython 3.11.15 (arm64), torch 2.12.0 (**MPS available**), numpy 2.4.6, mujoco 3.10.0, gymnasium 1.3.0,
  matplotlib 3.11.0, imageio 2.37.3 + imageio-ffmpeg 0.6.0. 32 packages total.

## Test results

| Suite | Result |
|---|---|
| torch import + MPS + mujoco `mj_step` + offscreen `Renderer` (GIF path) | PASS |
| `test_galambos_demo.py` + `test_bc.py` + `test_reward_oracle.py` | 29 passed / 4.2 s |
| + `test_ddpg.py` `test_reward.py` `test_tasks.py` `test_scenario_sanity.py` | 81 passed, **2 failed** / 8.0 s |

The 2 failures are `test_scenario_sanity.py::{test_geometry_well_formed,test_observation_contract}[quadruped]`
— quadruped obs shape (9 vs 33 vertices), an unrelated scenario untouched by this migration and by the
coin-toss path. **Not introduced by the Mac env** (no code changed). Flagged for the quadruped thread, not
this one.

## Production-scale smoke (behavioral parity, not just imports)

Scripted push controller, documented eval config (50 eps, seed 9000, difficulty 0.3, 300-step horizon,
5-step dwell), canonical rollout mirroring `test_push_regression_beats_pinch_carry`:

> **`scripted_controller` delivery = 42/50 = 0.840, deaths = 0.**

This matches the cached Windows fact exactly (`reports/2026-07-05-rl-scenario-assumption-audit.md:57`:
`PushDemonstrator: 42/50 = 0.84`). Mac MuJoCo physics is behaviorally identical. Wall: 4.1 s / 50 eps =
82 ms/ep ≈ 0.27 ms/step CPU MuJoCo.

Stage sentence (per stage ledger): **Scripted controller: 0.84. BC clone: 0.44–0.52 (not re-run here).
RL-refined: below BC in prior measured runs. Best saved checkpoint came from `bc_clone`/step-0.** This
migration re-established the `scripted_controller` number only; no learning was run.

## CORE.YAML items touched

None. `pyproject.toml`, `uv.lock`, `torch==2.12.0` pin all unchanged.

## New / removed dependencies

None at the project level (no manifest edit). Local venv only: same package set as the `ml`/`rl`/`demo`
groups, torch sourced from PyPI (CPU/MPS) rather than cu132 — a platform routing difference, not a version
change.

## Open issues / follow-ups

- `uv sync` (the managed workflow) still fails on darwin because of the cu132 source pin. If native Mac
  `uv sync` is wanted (rather than the `--no-sources` venv), it requires a platform-conditional
  `[tool.uv.sources]` marker (`sys_platform != 'darwin'`) — a `CORE.YAML` pinned-dependency edit needing §1
  approval. Deferred; the standalone venv is sufficient to run and train the coin-toss stack.
- 2 pre-existing quadruped `test_scenario_sanity` failures (above) — separate thread.
- Rust CLI is a `debug` build; a `release` build (`cargo build --release -p hymeko_cli` →
  `target/release/hymeko`, checked first by `_hymeko_cli()`) is worth it if env emission becomes a hot path.
- The research-direction fork from the session start is now runnable **on this machine**: (A) fair
  vector-critic re-test with action-diversity replay [recommended], (B) better imitation, (C) gradient-free
  monitor-directed search [needs sign-off]. Pending user choice.

## Provenance

- Host: Apple Silicon (arm64), macOS 26.4 (build 25E246). uv 0.11.26, cargo 1.96.1, rustc 1.96.1.
- Git branch `hymeko-neuro-migration`, working tree dirty (pre-existing WIP; no source changed by this task —
  only `.venv/` recreated, which is gitignored).
- No seeds beyond the eval config (seed 9000 + ep) — no training performed.
