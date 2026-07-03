"""Tests for the interactive viewer's drive loop (hymeko_rl/viewer.py).

The live window (`launch`) needs a display and is NOT exercised here. The loop logic
(`drive_viewer`) is GL-free and tested with a fake handle: it must sync once per step, stop when
the handle stops running, honour max_steps, and reset on episode end. Sources are checked for
shape.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.viz.viewer import (
    draw_target, drive_viewer, expert_source, random_source, zero_source,
)


class FakeHandle:
    """A ViewerHandle that runs for `runs` iterations then stops; counts sync() calls."""

    def __init__(self, runs: int = 5) -> None:
        self._left = runs
        self.sync_count = 0

    def is_running(self) -> bool:
        running = self._left > 0
        self._left -= 1
        return running

    def sync(self) -> None:
        self.sync_count += 1


class _StubModelOpt:
    timestep = 0.002


class _StubModel:
    opt = _StubModelOpt()


class FakeEnv:
    """Minimal stepped env that terminates every `period` steps — for the reset-on-done test."""

    def __init__(self, period: int = 2, n_actions: int = 4) -> None:
        self.model = _StubModel()
        self.frame_skip = 1
        self._n = n_actions
        self._period = period
        self._t = 0
        self.reset_count = 0

    def reset(self, *, seed: int | None = None):  # noqa: ANN201 - test stub
        self.reset_count += 1
        self._t = 0
        return np.zeros(self._n, dtype=np.float32), {}

    def step(self, _action):  # noqa: ANN201 - test stub
        self._t += 1
        terminated = self._t % self._period == 0
        return np.zeros(self._n, dtype=np.float32), 0.0, terminated, False, {}


def _expert_fn(_env, _obs):
    return np.zeros(4, dtype=np.float32)


def test_drive_syncs_once_per_step_and_stops_when_handle_stops() -> None:
    env = ArmReachEnv()          # real env; stepping needs no GL
    handle = FakeHandle(runs=5)
    steps = drive_viewer(env, expert_source(), handle, realtime=False, sleep_fn=lambda _dt: None)
    assert steps == 5
    assert handle.sync_count == 5


def test_max_steps_bounds_the_loop() -> None:
    env = ArmReachEnv()
    handle = FakeHandle(runs=10_000)        # effectively always running
    steps = drive_viewer(env, expert_source(), handle, max_steps=3, realtime=False,
                         sleep_fn=lambda _dt: None)
    assert steps == 3


def test_max_steps_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        drive_viewer(ArmReachEnv(), expert_source(), FakeHandle(), max_steps=0)


def test_resets_on_episode_end() -> None:
    env = FakeEnv(period=2)
    handle = FakeHandle(runs=6)
    steps = drive_viewer(env, _expert_fn, handle, realtime=False, sleep_fn=lambda _dt: None)
    assert steps == 6
    # terminates every 2 steps -> resets at steps 2,4,6; plus the initial reset = 4.
    assert env.reset_count == 4


def test_realtime_pacing_calls_sleep_each_step() -> None:
    env = ArmReachEnv()
    handle = FakeHandle(runs=3)
    calls: list[float] = []
    drive_viewer(env, expert_source(), handle, realtime=True, sleep_fn=calls.append)
    assert len(calls) == 3
    assert all(dt > 0 for dt in calls)      # dt = timestep * frame_skip > 0


@pytest.mark.parametrize("source", [random_source(), zero_source(), expert_source()])
def test_sources_return_action_in_space_shape(source) -> None:
    env = ArmReachEnv()
    obs, _ = env.reset(seed=0)
    action = source(env, obs)
    assert np.asarray(action).shape == env.action_space.shape


class _SceneHandle:
    """A handle exposing a real (headless, CPU) user_scn for the draw_target test."""

    def __init__(self, model: object) -> None:
        self.user_scn = mujoco.MjvScene(model, maxgeom=50)


def test_draw_target_adds_one_marker_and_replaces_on_redraw() -> None:
    env = ArmReachEnv()
    handle = _SceneHandle(env.model)
    draw_target(handle, {"target": np.array([0.1, 0.0, 0.3])})
    assert handle.user_scn.ngeom == 1
    # a new episode's target replaces (not accumulates) the marker.
    draw_target(handle, {"target": np.array([-0.2, 0.1, 0.25])})
    assert handle.user_scn.ngeom == 1


def test_draw_target_noop_without_target() -> None:
    env = ArmReachEnv()
    handle = _SceneHandle(env.model)
    draw_target(handle, {})            # reset info with no "target"
    assert handle.user_scn.ngeom == 0


def test_on_reset_fires_on_initial_and_each_episode_reset() -> None:
    env = FakeEnv(period=2)
    handle = FakeHandle(runs=6)
    calls: list[dict] = []
    drive_viewer(env, _expert_fn, handle, realtime=False, sleep_fn=lambda _dt: None,
                 on_reset=lambda _h, info: calls.append(info))
    # initial reset + a reset every 2 steps over 6 steps = 1 + 3 = 4.
    assert len(calls) == 4
