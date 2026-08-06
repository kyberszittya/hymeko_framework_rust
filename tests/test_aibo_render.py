"""Smoke test for the AIBO Lyapunov render path (skips where no GL context exists)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv  # noqa: E402

from scenarios.aibo.lyapunov import AIBOLyapunov  # noqa: E402
from scenarios.aibo.render_lyapunov_video import _H, _W, rollout  # noqa: E402


def test_rollout_frames_and_telemetry_smoke() -> None:
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.9,
                           reach_radius=0.12, max_steps=40)
    try:
        frames, telem, ev = rollout(env, AIBOLyapunov(reach_target=0.42), pursue=True)
    except Exception as exc:                                   # no offscreen GL backend
        pytest.skip(f"no MuJoCo render context: {exc}")
    assert frames and frames[0].shape == (_H, _W, 3)
    assert len(telem) == len(frames) and {"d", "herr", "V", "path", "goal"} <= telem[0].keys()
    assert np.all(np.isfinite(frames[0])) and "passes" in ev
