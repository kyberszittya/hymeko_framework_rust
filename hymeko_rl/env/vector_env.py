"""Lock-step vectorised env wrapper — the launch-bound fix (``B=N`` rollout, dispatch amortised ~17x at large N).

``N`` independent env instances stepped together; the policy forward then sees a batch of ``N`` observations
instead of one, so the per-op dispatch overhead (the dispatch-bound HSiKAN cost at ``B=1``) amortises. Custom
rather than ``gymnasium.vector.SyncVectorEnv``: each env owns its ``MjModel``/``MjData``, needs per-env autoreset
and a per-env failure guard, and the off-policy truncation rule wants ``terminated`` and ``truncated`` kept
separate. Mirrors the proven ``ppo._collect_vec`` pattern.

Autoreset semantics (the subtle part): :meth:`step` returns the **true** next observation for the transition (the
terminal obs when an env is done) so a *truncation* bootstraps from the real state; meanwhile ``self.obs`` is
advanced to the **reset** obs of any done env, ready for the next action. The trainer stores
``done = terminated`` (NOT ``terminated or truncated``), so a time-limit bootstraps and a true terminal does not.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class VectorizedEnv:
    """``N`` Gymnasium envs stepped in lock-step.

    # Preconditions ``make_env()`` returns a Gymnasium env (``reset(seed=) -> (obs, info)``, ``step(action) ->
    (obs, reward, terminated, truncated, info)``); ``n_envs >= 1``.
    # Postconditions ``obs`` is ``(n_envs, *obs_shape)`` float32 and every env is live (finished ones reset)."""

    def __init__(self, make_env: Callable[[], Any], n_envs: int, *, seed: int = 0) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1; got {n_envs}")
        self.envs = [make_env() for _ in range(n_envs)]
        self.n_envs = int(n_envs)
        self.obs = self.reset(seed=seed)

    def reset(self, *, seed: int = 0) -> np.ndarray:
        """Reset every env (per-env seed ``seed+i``); return stacked obs ``(n_envs, *obs_shape)``."""
        self.obs = np.stack([self.envs[i].reset(seed=seed + i)[0]
                             for i in range(self.n_envs)]).astype(np.float32)
        return self.obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step all envs with ``actions (n_envs, action_dim)``. Returns ``(next_obs, reward, terminated,
        truncated)`` each with leading dim ``n_envs``; ``next_obs`` is the TRUE post-step obs (terminal when done,
        for the transition). ``self.obs`` is advanced to the reset obs of any done env (for the next action)."""
        next_obs, rew, term, trunc = [], [], [], []
        cur: list[np.ndarray] = []
        for i, env in enumerate(self.envs):
            try:
                o, r, te, tr, _ = env.step(actions[i])
            except Exception:   # noqa: BLE001 — a physics blow-up ends THIS episode, must not kill the batch
                o, r, te, tr = self.obs[i], -1.0, True, False
            next_obs.append(o)
            rew.append(float(r))
            term.append(bool(te))
            trunc.append(bool(tr))
            cur.append(env.reset()[0] if (te or tr) else o)   # the obs the NEXT action will use
        self.obs = np.stack(cur).astype(np.float32)
        return (np.stack(next_obs).astype(np.float32),
                np.asarray(rew, dtype=np.float32),
                np.asarray(term, dtype=np.float32),
                np.asarray(trunc, dtype=np.float32))

    def close(self) -> None:
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self.envs = []
