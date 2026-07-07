"""A fixed-capacity replay buffer — the off-policy data path (DDPG/TD3/SAC).

Struct-of-arrays ring (CLAUDE.md §5): contiguous float32 columns for ``(obs, action, reward, next_obs,
done)``, O(1) ``add``, uniform-random ``sample``. Capacity-bounded, so peak RSS is fixed regardless of how
long training runs.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity SoA replay ring for transitions ``(s, a, r, s', done)``.

    # Preconditions ``capacity >= 1``; ``obs_shape`` is the per-step observation shape (e.g. ``(N, feat)``);
    ``action_dim >= 1``.
    # Invariants peak memory is ``O(capacity)`` (no growth); ``size <= capacity``; ``sample`` only draws
    from the ``size`` filled slots.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_dim: int, *,
                 priv_dim: int = 0) -> None:
        if capacity < 1 or action_dim < 1:
            raise ValueError(f"capacity/action_dim must be >= 1; got {capacity}/{action_dim}")
        if priv_dim < 0:
            raise ValueError(f"priv_dim must be >= 0; got {priv_dim}")
        self.capacity = int(capacity)
        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._act = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rew = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        # PRIVILEGED (asymmetric CTDE / MADDPG): the centralized critic's extra global state z(s) and z(s'),
        # stored ONLY when priv_dim>0. z is NOT reconstructable from obs (contact needs contact forces, phase
        # is a history latch), so it must be persisted alongside the transition. priv_dim=0 allocates nothing
        # and every method below behaves exactly as the plain (s,a,r,s',d) ring (cart-pole / SAC unchanged).
        self.priv_dim = int(priv_dim)
        self._priv = np.zeros((capacity, priv_dim), dtype=np.float32) if priv_dim else None
        self._priv_next = np.zeros((capacity, priv_dim), dtype=np.float32) if priv_dim else None
        self._ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool, priv: "np.ndarray | None" = None,
            priv_next: "np.ndarray | None" = None) -> None:
        """Append one transition (overwriting the oldest when full). ``priv``/``priv_next`` are the privileged
        critic state ``z(s)``/``z(s')`` — required iff ``priv_dim>0``, ignored (must be ``None``) otherwise.

        # Postconditions ``size`` grows until ``capacity``, then stays; the new transition is at the
        position the next ``add`` will overwrite (ring)."""
        i = self._ptr
        self._obs[i] = obs
        self._act[i] = action
        self._rew[i] = reward
        self._next[i] = next_obs
        self._done[i] = 1.0 if done else 0.0
        if self._priv is not None:
            if priv is None or priv_next is None:
                raise ValueError("priv_dim>0 buffer requires priv and priv_next on add()")
            self._priv[i] = priv
            self._priv_next[i] = priv_next   # type: ignore[index]
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
                  next_obs: np.ndarray, done: np.ndarray, priv: "np.ndarray | None" = None,
                  priv_next: "np.ndarray | None" = None) -> None:
        """Append ``N`` transitions at once (the vectorised-rollout path), equivalent to ``N`` sequential
        :meth:`add` calls in ring order. ``priv``/``priv_next`` are ``(N, priv_dim)`` — required iff ``priv_dim>0``.

        # Preconditions leading dim ``N``: ``obs``/``next_obs`` ``(N, *obs_shape)``, ``action`` ``(N, action_dim)``,
        ``reward``/``done`` ``(N,)``; ``N <= capacity``.
        # Postconditions ``size`` grows by ``N`` (capped at ``capacity``); the ring pointer advances by ``N``."""
        n = int(np.asarray(reward).shape[0])
        if n < 1:
            return
        idx = (self._ptr + np.arange(n)) % self.capacity      # wraps around the ring; same order as N adds
        self._obs[idx] = obs
        self._act[idx] = action
        self._rew[idx] = np.asarray(reward, dtype=np.float32)
        self._next[idx] = next_obs
        self._done[idx] = np.asarray(done, dtype=np.float32)
        if self._priv is not None:
            if priv is None or priv_next is None:
                raise ValueError("priv_dim>0 buffer requires priv and priv_next on add_batch()")
            self._priv[idx] = np.asarray(priv, dtype=np.float32)
            self._priv_next[idx] = np.asarray(priv_next, dtype=np.float32)   # type: ignore[index]
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def column_schema(self) -> list[tuple[str, int]]:
        """The ordered ``(field, width)`` layout of a stored transition — the replay *serialization* schema
        (consumed by the live pipeline tensor-contract guard). ``priv``/``next_priv`` appear iff ``priv_dim>0``."""
        obs_w = int(np.prod(self._obs.shape[1:]))
        cols = [("obs", obs_w), ("action", int(self._act.shape[1])), ("reward", 1),
                ("next_obs", obs_w), ("done", 1)]
        if self._priv is not None:
            cols += [("priv", int(self._priv.shape[1])), ("next_priv", int(self._priv_next.shape[1]))]
        return cols

    def _gather(self, idx: np.ndarray,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialise the base ``(obs, act, rew, next_obs, done)`` tensors for ``idx`` (shared by both
        samplers so the gather logic lives once, §6.1)."""
        t = torch.as_tensor
        return (t(self._obs[idx]), t(self._act[idx]), t(self._rew[idx]),
                t(self._next[idx]), t(self._done[idx]))

    def sample(self, batch_size: int, *, generator: np.random.Generator,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample a minibatch as float32 tensors ``(obs, act, rew, next_obs, done)``.

        # Preconditions ``1 <= batch_size <= size``. # Postconditions ``rew``/``done`` are shape ``(B,)``;
        the others keep their per-item shape with a leading batch dim."""
        if not 1 <= batch_size <= self.size:
            raise ValueError(f"batch_size must be in [1, size={self.size}]; got {batch_size}")
        idx = generator.integers(0, self.size, size=batch_size)
        return self._gather(idx)

    def sample_with_priv(self, batch_size: int, *, generator: np.random.Generator,
                         ) -> tuple[torch.Tensor, ...]:
        """Sample ``(obs, act, rew, next_obs, done, priv, priv_next)`` — the asymmetric-CTDE path. The two
        trailing tensors are the privileged critic state ``z(s)`` and ``z(s')``.

        # Preconditions ``priv_dim>0``; ``1 <= batch_size <= size``. # Errors ``ValueError`` on a non-priv buffer."""
        if self._priv is None:
            raise ValueError("sample_with_priv requires a priv_dim>0 buffer")
        if not 1 <= batch_size <= self.size:
            raise ValueError(f"batch_size must be in [1, size={self.size}]; got {batch_size}")
        idx = generator.integers(0, self.size, size=batch_size)
        t = torch.as_tensor
        return (*self._gather(idx), t(self._priv[idx]), t(self._priv_next[idx]))   # type: ignore[index]
