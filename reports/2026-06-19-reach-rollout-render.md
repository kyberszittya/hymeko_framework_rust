# Report — visual simulation of the reaching rollout

**Date:** 2026-06-19 · **Plan:** `docs/plans/2026-06-19-reach-rollout-render/`

## Summary

`hymeko_rl` had physics + BC + matplotlib error curves but **no visual simulation**
(`ArmReachEnv.metadata["render_modes"] == []`). Added `hymeko_rl/render_reach.py`: an offscreen
`mujoco.Renderer` steps an action source (the scripted DLS-IK **expert** or a behaviour-cloned
**policy**) through `ArmReachEnv`, captures one frame per env step with the reach **target drawn
as a 3-D scene marker** (`mjv_initGeom`, not a 2-D overlay), and encodes via an encoder
Strategy: `gif` (Pillow, no new dep) or `mp4` (imageio, §1-gated). First watchable artifact
rendered: `reports/2026-06-19-reach-rollout.gif` (expert, 81 frames).

## Are we close to visual simulation? — yes, now in place

The one genuine risk was the GL context on this Windows host; **measured working** — a real
240×320 frame off the live `ArmReachEnv` model, `MUJOCO_GL` unset. The renderer is wired,
tested, and produced a GIF. The MP4 path is written but inert pending the dependency token.

## Files touched

| file | LOC | change |
|---|---|---|
| `hymeko_rl/render_reach.py` | 188 | NEW — `render_rollout`, `CameraView`, expert/policy sources, `_draw_target`, encoder Strategy (`gif`/`mp4`), `encode`, CLI |
| `hymeko_rl/tests/test_render_reach.py` | 110 | NEW — 7 tests (encoder contract + GL-gated render) |
| `docs/plans/2026-06-19-reach-rollout-render/{plan.tex,pdf,tikz,mmd}` | — | NEW plan (4 formats, compiles) |

Reuses `plot_reach.py`'s rollout pattern and `bc.py` helpers; no edits to existing files.
The `signedkan_wip/.../render_mujoco_video.py` pipeline was used as a **reference only**
(sinusoid-driven, different scene, hand-rolled fovy projection) — not imported, to keep
`hymeko_rl` isolated.

## CORE.YAML items touched

**None** (no locked file). **Dependency added — §1 approved:** `imageio` + `imageio-ffmpeg`,
token `APPROVED-CORE-EDIT: imageio-mp4-render` (2026-06-19 chat). Declared in `pyproject.toml`
`demo` group (`imageio>=2,<3`, `imageio-ffmpeg>=0.5,<1`); installed `imageio 2.37.3`,
`imageio-ffmpeg 0.6.0`. `gif` uses Pillow (already present). When imageio is absent (base
install) `_encode_mp4` raises an actionable `RuntimeError` pointing to `uv sync --group demo`.

## Test results

- `pytest -p no:randomly hymeko_rl/tests/test_render_reach.py` — **7 passed in ~14 s**.
  - Encoder contract (no GL): empty/unknown-kind → `ValueError`; `gif` round-trips (PIL reopens
    with matching `n_frames`); `mp4`-without-imageio → actionable `RuntimeError` (skips now that
    imageio is installed; the contract holds for base installs).
  - Render (GL-gated, ran here): frame shape/dtype/content; determinism (same seed → identical
    first frame); target marker increments `scene.ngeom`; render→gif integration.
  - `requires_gl` skip guard added for headless CI (the encoder-contract tests always run).
- **Static:** `ruff check` — clean. `mypy --strict` — no new error class beyond the package's
  pre-existing `mujoco` `import-untyped` baseline (unsuppressed in 3 sibling files; matched).
  One new scoped `# type: ignore[import-not-found]` on the guarded optional `imageio` import,
  with an inline reason (§6.3).

## Performance results (§3 production smoke)

Expert rollout, 480×360, default episode (81 frames), both encoders:

| encoder | wall | peak RSS | output |
|---|---|---|---|
| gif (Pillow) | 13.7 s | 739.0 MB | `reports/2026-06-19-reach-rollout.gif`, 122 KB |
| mp4 (imageio/H.264) | 9.6 s | 678.4 MB | `reports/2026-06-19-reach-rollout.mp4`, 31 KB |

Budgets (wall < 60 s, RSS < 2 GB, cap 16 GB) — met by both. Peak RSS via Win32
`GetProcessMemoryInfo.PeakWorkingSetSize`. H.264 is ~4× smaller than the GIF at equal frames.

## New / removed dependencies

Added (§1 approved `imageio-mp4-render`): `imageio 2.37.3`, `imageio-ffmpeg 0.6.0`, in the
`pyproject.toml` `demo` group. None removed.

## §6.5 anti-patterns

None. Encoders are a Strategy registry (`_ENCODERS`), not per-format wrappers (#1/#9);
action source is a pluggable `ActionFn`, not a `_kind: str` branch (#7); new module created
only after a discovery pass confirmed no render path in `hymeko_rl` (#12). No `v2`/`_new`
file proliferation (#13).

## Open issues / follow-ups

1. **MP4** — done: dep approved, declared, installed; `--encoder mp4` rendered (table above).
2. **BC-policy video** — `--source bc` renders the learned policy (trains inline); not yet run
   as a smoke (expert was the fast deterministic first artifact).
3. **Grasping (Phase 2)** scene has no gripper/object yet; the renderer is scene-agnostic and
   will carry over.

## Provenance

- Git SHA `7d16ad0` (working tree dirty; `hymeko_rl` is an uncommitted increment — new files
  above are untracked). Seeds: smoke seed 0; tests seeds 0/1/7. `mujoco 3.9.0`, Pillow 11.3.0,
  torch per CORE pin. Host: Windows 11, CPU; offscreen GL (`MUJOCO_GL` unset) functional.
