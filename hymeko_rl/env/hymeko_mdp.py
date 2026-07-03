"""A control MDP whose state, observation, dynamics, and objective are ALL declared in one ``.hymeko``.

The minimal witness of *HyMeKo as a declarative control substrate* (memory
``project-hymeko-as-control-substrate``): a point-mass reach task where the **state space**, the **observed
channels**, the **dynamics parameters**, and the **objective** (reward ``Σ weight·term`` + success radius) are
read from a single ``.hymeko`` profile — no bespoke reward code, the *same* :class:`RewardSpec` the robots use.
Solved here by the *same* off-policy TD3 (:func:`hymeko_rl.train.ddpg.train_offpolicy`) — one substrate, robots and toys.

The dynamics are a D-dimensional point mass (``D = action_dim``): state ``[pos(D), vel(D)]`` (so
``state_dim = 2·D``), ``v ← v + (force_gain·f − damping·v)·dt``, ``p ← p + v·dt``. The reward is evaluated by the
declared spec with ``dist = ‖p − target‖`` (``reach_distance = −dist``; ``action_cost = −‖f‖²``) — reusing the
robot terms verbatim. Runtime-tunable: edit the ``.hymeko`` (git-tracked), the env re-reads it, no code change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from hymeko_rl.env._profile import read_scene_fields
from hymeko_rl.env.reward import RewardSpec


class HymekoReachEnv:
    """Point-mass reach MDP fully specified by a ``.hymeko`` profile. Gym 5-tuple contract; ``obs`` is
    ``(n_observed, 1)`` so the MLP / per-vertex backbones read it unchanged.

    # Preconditions ``state_dim == 2·action_dim``; ``observed`` indices in ``[0, state_dim)``; ``target`` is
      ``action_dim``-dimensional; ``reward_spec`` uses terms that read only ``dist`` / ``action`` (e.g.
      ``reach_distance``, ``action_cost``).
    # Postconditions ``step`` returns a finite reward; ``terminated`` iff ``‖p − target‖ < reach_radius``.
    # Invariants the action is clipped to ``±action_scale``; the observation is the declared state slice.
    """

    def __init__(self, *, action_dim: int, dt: float, damping: float, force_gain: float, action_scale: float,
                 target: "tuple[float, ...]", reach_radius: float, max_steps: int, observed: "tuple[int, ...]",
                 reward_spec: RewardSpec) -> None:
        self.action_dim = int(action_dim)
        self.state_dim = 2 * self.action_dim
        self.dt = float(dt)
        self.damping = float(damping)
        self.force_gain = float(force_gain)
        self.action_scale = float(action_scale)
        self.reach_radius = float(reach_radius)
        self.max_steps = int(max_steps)
        self.target = np.asarray(target, dtype=np.float64)
        if self.target.shape != (self.action_dim,):
            raise ValueError(f"target must be {self.action_dim}-D; got {self.target.shape}")
        self.observed = [int(i) for i in observed]
        if any(i < 0 or i >= self.state_dim for i in self.observed):
            raise ValueError(f"observed indices out of range [0,{self.state_dim}); got {self.observed}")
        self.reward_spec = reward_spec
        self.n_actions = self.action_dim
        self.observation_space = Box(-np.inf, np.inf, (len(self.observed), 1), np.float32)
        self.action_space = Box(-self.action_scale, self.action_scale, (self.action_dim,), np.float32)
        self._x = np.zeros(self.state_dim, np.float64)
        self._t = 0
        self._rng = np.random.default_rng(0)

    @classmethod
    def from_hymeko(cls, path: "str | Path") -> "HymekoReachEnv":
        """Build the env from a ``.hymeko`` profile: scene fields (state/dynamics/objective) + the reward spec."""
        f = read_scene_fields(path, "scene")
        action_dim = int(f["action_dim"])
        if int(f["state_dim"]) != 2 * action_dim:
            raise ValueError(f"{path}: state_dim must be 2·action_dim ({2 * action_dim}); got {int(f['state_dim'])}")
        target = f["target"] if isinstance(f["target"], tuple) else (float(f["target"]),)
        observed = f["observed"] if isinstance(f["observed"], tuple) else (float(f["observed"]),)
        return cls(action_dim=action_dim, dt=float(f["dt"]), damping=float(f["damping"]),
                   force_gain=float(f["force_gain"]), action_scale=float(f["action_scale"]),
                   target=target, reach_radius=float(f["reach_radius"]), max_steps=int(f["max_steps"]),
                   observed=tuple(int(i) for i in observed), reward_spec=RewardSpec.from_hymeko(path))

    def _obs(self) -> np.ndarray:
        return self._x[self.observed].reshape(-1, 1).astype(np.float32)

    def reset(self, *, seed: "int | None" = None) -> "tuple[np.ndarray, dict[str, Any]]":
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        pos = self._rng.uniform(-1.0, 1.0, self.action_dim)
        self._x = np.concatenate([pos, np.zeros(self.action_dim)]).astype(np.float64)
        self._t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict[str, Any]]":
        f = np.clip(np.asarray(action, np.float64).reshape(self.action_dim), -self.action_scale, self.action_scale)
        pos, vel = self._x[:self.action_dim], self._x[self.action_dim:]
        vel = vel + (self.force_gain * f - self.damping * vel) * self.dt
        pos = pos + vel * self.dt
        self._x = np.concatenate([pos, vel])
        dist = float(np.linalg.norm(pos - self.target))
        reward = self.reward_spec.evaluate(self, dist, f)      # reach_distance = −dist, action_cost = −‖f‖²
        self._t += 1
        reached = dist < self.reach_radius
        return self._obs(), reward, bool(reached), self._t >= self.max_steps, {"dist": dist, "reached": reached}


def reach_rate(env: HymekoReachEnv, actor: Any, n_episodes: int, seed0: int = 9000) -> float:
    """Fraction of episodes the greedy actor drives the point mass into the goal radius (the declared objective)."""
    import torch
    hits = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        for _ in range(env.max_steps):
            with torch.no_grad():
                a = actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
            obs, _r, term, trunc, _info = env.step(a)
            if term:
                hits += 1
                break
            if trunc:
                break
    return hits / max(1, n_episodes)
