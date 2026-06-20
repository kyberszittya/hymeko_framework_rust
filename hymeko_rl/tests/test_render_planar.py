"""Tests for the Galambos planar-grasp GIF renderer (hymeko_rl/render_planar_gifs.py).

The camera test needs no GL context; the render test skips when an offscreen ``mujoco.Renderer``
cannot be constructed (headless CI without a GL context).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
from PIL import Image

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.evaluate import render_episode_gif
from hymeko_rl.policy import build_policy
from hymeko_rl.render_planar_gifs import policy_action_fn, topdown_camera


def _can_render() -> bool:
    try:
        env = PlanarGraspEnv(max_steps=4)
        mujoco.Renderer(env.model, height=48, width=48).close()
        return True
    except Exception:  # noqa: BLE001 - GL availability probe
        return False


requires_gl = pytest.mark.skipif(not _can_render(), reason="no offscreen GL context")


def test_topdown_camera_is_overhead() -> None:
    cam = topdown_camera()
    assert -90.0 <= cam.elevation <= -85.0          # near-straight-down (top-down table)
    assert cam.distance > 0.0


@requires_gl
def test_renders_a_valid_gif(tmp_path: Path) -> None:
    env = PlanarGraspEnv(max_steps=6)
    feat = int(env.observation_space.shape[-1])
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg, hidden=16)
    out = render_episode_gif(env, policy_action_fn(ac), tmp_path / "clip", seed=0,
                             width=80, height=72, camera=topdown_camera())
    assert out.exists() and out.suffix == ".gif"
    im = Image.open(out)
    assert im.format == "GIF" and im.size == (80, 72)
    assert getattr(im, "n_frames", 1) >= 2          # an actual animation, not a single frame
