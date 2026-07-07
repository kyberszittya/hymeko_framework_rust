"""Residual-control wrapper: RL learns a bounded CORRECTION on top of a fixed scripted controller.

The measured failure of every off-policy recipe on the coin toss (2026-07-05: noise/anchor/adaptive/warm-up
cells all collapse the moment the actor trains) motivates the standard remedy: keep the working base
controller (0.84 delivery) untouchable, and let the policy act only through a small bounded delta added to
the base action. A zero policy reproduces the base controller exactly, so learning starts AT the base's
performance and exploration is confined to a band around a working trajectory. Any gain is attributable to
RL alone — no warm start to lose, no clone to degrade.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box


class ResidualControllerEnv:
    """A gym-5-tuple env whose action is a bounded DELTA over a scripted controller's action.

    ``step(delta)`` applies ``clip(base + delta, base_space)`` to the inner env, where ``base`` is the
    controller's closed-loop action in the current state. Observations, rewards, termination, and ``info``
    pass through unchanged, so every existing metric (`DwellMetric` etc.) works as-is.

    # Preconditions ``controller`` has the ``reset()`` / ``action(env) -> np.ndarray`` contract;
      ``0 < delta_scale``. # Invariants the executed action stays inside the inner env's action space;
      a zero delta reproduces the base controller's behaviour exactly (regression-tested).
    """

    def __init__(self, env: Any, controller: Any, *, delta_scale: float = 0.15) -> None:
        if delta_scale <= 0.0:
            raise ValueError(f"delta_scale must be positive; got {delta_scale}")
        self.env = env
        self.controller = controller
        inner = env.action_space
        assert isinstance(inner, Box)
        self._inner_space = inner
        self.delta_scale = float(delta_scale)
        self.action_space = Box(low=-delta_scale, high=delta_scale,
                                shape=inner.shape, dtype=np.float32)
        self.observation_space = env.observation_space
        self.n_actions = int(env.n_actions)

    # -- attribute passthrough: metrics/backbones read hg, max_steps, success_steps, _planar_metrics, … ----
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *, seed: "int | None" = None, **kw: Any) -> "tuple[Any, dict]":
        out = self.env.reset(seed=seed, **kw)
        self.controller.reset()
        return out

    def step(self, delta: np.ndarray) -> "tuple[Any, float, bool, bool, dict]":
        base = np.asarray(self.controller.action(self.env), dtype=np.float32)
        act = np.clip(base + np.asarray(delta, dtype=np.float32),
                      self._inner_space.low, self._inner_space.high)
        return self.env.step(act)  # type: ignore[no-any-return]

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
