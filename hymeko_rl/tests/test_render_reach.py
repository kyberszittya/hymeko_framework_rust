"""Tests for the reaching-rollout renderer.

The encoder-contract tests need no GL context and always run; the render tests skip when an
offscreen ``mujoco.Renderer`` cannot be constructed (headless CI without a GL context).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.render_reach import encode, expert_source, render_rollout


def _can_render() -> bool:
    try:
        env = ArmReachEnv(control_mode="torque")
        mujoco.Renderer(env.model, height=64, width=64).close()
        return True
    except Exception:
        return False


requires_gl = pytest.mark.skipif(not _can_render(), reason="no offscreen GL context")


# ── encoder contract (no GL) ─────────────────────────────────────────────────
def test_encode_rejects_empty_and_unknown(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="no frames"):
        encode([], tmp_path / "x", fps=10, kind="gif")
    with pytest.raises(ValueError, match="unknown encoder"):
        encode([frame], tmp_path / "x", fps=10, kind="avi")


def test_encode_gif_roundtrips(tmp_path: Path) -> None:
    frames = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (0, 80, 160, 240)]
    # stamp="" disables the provenance overlay: this test checks the encoder preserves every frame,
    # which a baked-in label (covering the degenerate 8x8 fixture) would defeat.
    path = encode(frames, tmp_path / "clip", fps=10, kind="gif", stamp="")
    assert path.suffix == ".gif" and path.is_file()
    with Image.open(path) as im:
        assert getattr(im, "n_frames", 1) == len(frames)


def test_encode_mp4_without_dep_is_loud(tmp_path: Path) -> None:
    """mp4 without imageio must fail with an actionable error (the `demo`-group install
    instruction), not a bare ImportError. Skips when imageio is installed."""
    try:
        import imageio.v2  # noqa: F401
        pytest.skip("imageio present — the no-dep contract does not apply")
    except ImportError:
        pass
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="uv sync --group demo"):
        encode([frame], tmp_path / "x", fps=10, kind="mp4")


# ── rollout rendering (needs GL) ─────────────────────────────────────────────
@requires_gl
def test_render_rollout_shapes_and_content() -> None:
    env = ArmReachEnv(control_mode="torque")
    frames = render_rollout(env, expert_source(), seed=0, width=96, height=72, max_frames=4)
    assert 0 < len(frames) <= 5            # capped, plus a possible final settled frame
    for f in frames:
        assert f.shape == (72, 96, 3) and f.dtype == np.uint8
    assert int((frames[0] > 0).sum()) > 0  # non-trivial render


@requires_gl
def test_render_rollout_is_deterministic() -> None:
    env = ArmReachEnv(control_mode="torque")
    a = render_rollout(env, expert_source(), seed=7, width=96, height=72, max_frames=3)
    b = render_rollout(env, expert_source(), seed=7, width=96, height=72, max_frames=3)
    assert len(a) == len(b)
    assert np.array_equal(a[0], b[0])


@requires_gl
def test_target_marker_adds_a_scene_geom() -> None:
    """The target marker increments the rendered scene's geom count (it is drawn)."""
    from hymeko_rl.render_reach import CameraView, _draw_target

    env = ArmReachEnv(control_mode="torque")
    _obs, info = env.reset(seed=1)
    renderer = mujoco.Renderer(env.model, height=64, width=64)
    try:
        renderer.update_scene(env.data, camera=CameraView().to_mjv())
        before = renderer.scene.ngeom
        _draw_target(renderer.scene, np.asarray(info["target"], dtype=np.float64))
        assert renderer.scene.ngeom == before + 1
    finally:
        renderer.close()


@requires_gl
def test_render_then_gif_integration(tmp_path: Path) -> None:
    env = ArmReachEnv(control_mode="torque")
    frames = render_rollout(env, expert_source(), seed=0, width=96, height=72, max_frames=5)
    path = encode(frames, tmp_path / "reach", fps=15, kind="gif")
    with Image.open(path) as im:
        assert getattr(im, "n_frames", 1) == len(frames)
