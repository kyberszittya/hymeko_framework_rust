"""Interactive MuJoCo viewer for the arm simulations.

A live window (orbit / zoom / pan / pause) driven by the *same* rollout toolkit as
:mod:`hymeko_rl.render_reach`: an :data:`ActionFn` (expert / bc / random / zero) steps the
beautified scene built by :func:`build_render_env` while ``mujoco.viewer`` renders it in real
time. Where ``render_reach`` is an offscreen sink (GIF/MP4), this is the live sink.

    python -m hymeko_rl.viewer --robot data/robotics/anthropomorphic_arm.hymeko --source expert
    python -m hymeko_rl.viewer --source random --no-realtime       # quick "is it alive" check

The window needs a display; on a headless host use ``render_reach`` (GIF/MP4) instead. The loop
logic lives in :func:`drive_viewer` (pure, unit-tested with a fake handle); :func:`launch` is the
thin GL wrapper that opens the window.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.env.arm_world import CONTROL_MODES
from hymeko_rl.render_reach import (
    _DEFAULT_ROBOT, ActionFn, _bc_policy, _draw_target, build_render_env, expert_source,
    policy_source,
)

# A reset hook: called with (handle, reset-info) after each env.reset, e.g. to (re)draw the target.
ResetHook = Callable[[Any, "dict[str, Any]"], None]


class ViewerHandle(Protocol):
    """The slice of ``mujoco.viewer.Handle`` that :func:`drive_viewer` needs (a fake satisfies
    it in tests, keeping the loop GL-free)."""

    def is_running(self) -> bool: ...
    def sync(self) -> None: ...


def random_source() -> ActionFn:
    """Uniform-random actions in the env's ctrlrange — the 'is the sim alive' source."""
    def fn(env: ArmReachEnv, _obs: np.ndarray) -> np.ndarray:
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    return fn


def zero_source() -> ActionFn:
    """Zero control — watch the arm settle under gravity / hold."""
    def fn(env: ArmReachEnv, _obs: np.ndarray) -> np.ndarray:
        return np.zeros(env.n_actions, dtype=np.float32)
    return fn


def draw_target(handle: Any, info: dict[str, Any]) -> None:
    """(Re)draw the reach target as a sphere in the viewer's persistent ``user_scn``.

    Clears the previous marker (``ngeom = 0``) then appends one sphere at ``info["target"]`` — so a
    new episode's target replaces the old one. No-op if the reset info carries no target.

    # Preconditions ``handle`` exposes a ``user_scn`` (the ``mujoco.viewer`` Handle does).
    # Postconditions ``user_scn`` holds exactly the current target marker (or is unchanged if no
      target is present).
    """
    target = info.get("target")
    if target is None:
        return
    scn = handle.user_scn
    scn.ngeom = 0
    _draw_target(scn, np.asarray(target, dtype=np.float64))


def drive_viewer(env: ArmReachEnv, action_fn: ActionFn, handle: ViewerHandle, *,
                 max_steps: int = 100_000, realtime: bool = True,
                 sleep_fn: Callable[[float], Any] = time.sleep, seed: int = 0,
                 on_reset: ResetHook | None = None) -> int:
    """Step ``action_fn`` through ``env``, syncing the viewer each step; reset on episode end.

    # Preconditions ``max_steps >= 1``; ``env`` shares ``handle``'s model/data (``launch`` ensures
      this); ``action_fn`` returns an action in the env's ctrlrange.
    # Postconditions steps until the handle stops running or ``max_steps`` is hit, calling
      ``sync()`` once per step and resetting the env on ``terminated`` ∨ ``truncated``; ``on_reset``
      (if given) is invoked with ``(handle, info)`` after the initial reset and after each episode
      reset; returns the number of steps driven.
    # Invariants touches no MuJoCo state except via ``env.step`` / ``env.reset`` and ``on_reset``.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    dt = float(env.model.opt.timestep) * env.frame_skip
    obs, info = env.reset(seed=seed)
    if on_reset is not None:
        on_reset(handle, info)
    steps = 0
    while handle.is_running() and steps < max_steps:
        obs, _reward, terminated, truncated, _info = env.step(action_fn(env, obs))
        handle.sync()
        steps += 1
        if realtime:
            sleep_fn(dt)
        if terminated or truncated:
            obs, info = env.reset()
            if on_reset is not None:
                on_reset(handle, info)
    return steps


def launch(env: ArmReachEnv, action_fn: ActionFn, *, realtime: bool = True,
           max_steps: int = 100_000) -> None:
    """Open an interactive viewer for ``env`` and drive it. Blocks until the window closes.

    Needs a display; not exercised headlessly (see module docstring). The ``Handle`` is used as a
    context manager so it closes on exit/exception.
    """
    import mujoco.viewer  # local import: only the live path needs a display
    with mujoco.viewer.launch_passive(env.model, env.data) as handle:
        drive_viewer(env, action_fn, handle, max_steps=max_steps, realtime=realtime,
                     on_reset=draw_target)


def make_source(name: str, env: ArmReachEnv, *, demos: int = 48, epochs: int = 200,
                hidden: int = 64, seed: int = 0) -> ActionFn:
    """Resolve a source name to an :data:`ActionFn` (reuses ``render_reach`` sources)."""
    if name == "expert":
        return expert_source()
    if name == "random":
        return random_source()
    if name == "zero":
        return zero_source()
    if name == "bc":
        return policy_source(_bc_policy(env, n_demos=demos, n_epochs=epochs, hidden=hidden,
                                        seed=seed))
    raise ValueError(f"unknown source {name!r}; expected expert/bc/random/zero")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", default=_DEFAULT_ROBOT,
                    help="robot .hymeko to view; 'default' = the bare 4-DOF arm_world arm")
    ap.add_argument("--source", default="expert", choices=["expert", "bc", "random", "zero"])
    ap.add_argument("--control", default="position", choices=sorted(CONTROL_MODES))
    ap.add_argument("--ee-body", default="tool")
    ap.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="step as fast as possible instead of wall-clock paced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demos", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    a = ap.parse_args(argv)

    robot = None if a.robot == "default" else a.robot
    env = build_render_env(robot, control_mode=a.control, pretty=a.pretty, ee_body=a.ee_body)
    action_fn = make_source(a.source, env, demos=a.demos, epochs=a.epochs, hidden=a.hidden,
                            seed=a.seed)
    launch(env, action_fn, realtime=a.realtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
