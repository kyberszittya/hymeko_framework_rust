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
        # per-transition provenance TAG for contact-stratified sampling: 0 = ONLINE (the default; every online add and
        # every legacy caller lands here, so uniform sampling is byte-unchanged), >0 = a demonstration stratum id.
        # The tag never enters the learning signal — only :meth:`sample_stratified` reads it.
        self._tag = np.zeros(capacity, dtype=np.int16)
        self._ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool, priv: "np.ndarray | None" = None,
            priv_next: "np.ndarray | None" = None, *, tag: int = 0) -> None:
        """Append one transition (overwriting the oldest when full). ``priv``/``priv_next`` are the privileged
        critic state ``z(s)``/``z(s')`` — required iff ``priv_dim>0``, ignored (must be ``None``) otherwise.
        ``tag`` is the provenance stratum id (default ``0`` = ONLINE; only :meth:`sample_stratified` reads it).

        # Postconditions ``size`` grows until ``capacity``, then stays; the new transition is at the
        position the next ``add`` will overwrite (ring)."""
        i = self._ptr
        self._obs[i] = obs
        self._act[i] = action
        self._rew[i] = reward
        self._next[i] = next_obs
        self._done[i] = 1.0 if done else 0.0
        self._tag[i] = int(tag)
        if self._priv is not None:
            if priv is None or priv_next is None:
                raise ValueError("priv_dim>0 buffer requires priv and priv_next on add()")
            self._priv[i] = priv
            self._priv_next[i] = priv_next   # type: ignore[index]
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray,
                  next_obs: np.ndarray, done: np.ndarray, priv: "np.ndarray | None" = None,
                  priv_next: "np.ndarray | None" = None, *, tags: "np.ndarray | None" = None) -> None:
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
        self._tag[idx] = np.asarray(tags, dtype=np.int16) if tags is not None else 0
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

    def tag_counts(self) -> dict[int, int]:
        """Number of stored transitions per provenance tag (0 = ONLINE, >0 = demo strata). For preflight/logging."""
        tags, counts = np.unique(self._tag[: self.size], return_counts=True)
        return {int(t): int(c) for t, c in zip(tags, counts)}

    @staticmethod
    def _largest_remainder(total: int, weights: "list[float]") -> np.ndarray:
        """Integer apportionment of ``total`` across ``weights`` (largest-remainder / Hamilton) — exact sum, no drift."""
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        raw = w * total
        base = np.floor(raw).astype(int)
        for k in np.argsort(-(raw - base))[: int(total - base.sum())]:
            base[k] += 1
        return base

    def sample_stratified(self, batch_size: int, *, demo_frac: float, strata_weights: "dict[int, float]",
                          generator: np.random.Generator, shortage: "dict | None" = None,
                          account: "dict | None" = None,
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compose a minibatch as ``demo_frac`` demonstration transitions (tag>0, split by ``strata_weights``) +
        the remainder ONLINE (tag==0). A thin stratum is sampled WITH REPLACEMENT (quota kept, shortage logged); an
        EMPTY stratum's quota is redistributed across the non-empty demo strata (never silently drawn from arbitrary
        approach transitions). Returns the same ``(obs,act,rew,next,done)`` tensors as :meth:`sample`; batch size exact.

        # Preconditions ``1 <= batch_size <= size``; ``strata_weights`` keys are positive tag ids. # Postconditions
        deterministic for a fixed ``generator`` state; ``shortage`` (if given) records per-tag empty/replacement counts."""
        if not 1 <= batch_size <= self.size:
            raise ValueError(f"batch_size must be in [1, size={self.size}]; got {batch_size}")
        demo_frac = float(np.clip(demo_frac, 0.0, 1.0))
        n_demo = int(round(batch_size * demo_frac))
        n_online = batch_size - n_demo
        tag = self._tag[: self.size]
        parts: list[np.ndarray] = []
        if n_online > 0:                                              # ONLINE portion (unchanged uniform behaviour)
            pool = np.flatnonzero(tag == 0)
            pool = pool if pool.size else np.arange(self.size)        # early warmup: no online yet -> whole buffer
            parts.append(pool[generator.integers(0, pool.size, size=n_online)])
            if account is not None:
                account[0] = account.get(0, 0) + n_online
        if n_demo > 0:
            keys = sorted(strata_weights)
            avail = {k: np.flatnonzero(tag == k) for k in keys}
            nonempty = [k for k in keys if avail[k].size > 0]
            for k in keys:                                            # log empty non-zero-weight strata
                if avail[k].size == 0 and strata_weights[k] > 0 and shortage is not None:
                    shortage.setdefault(k, {"empty": 0, "replacement": 0})["empty"] += 1
            if not nonempty:                                          # no demos at all -> uniform fallback
                parts.append(generator.integers(0, self.size, size=n_demo))
            else:
                alloc = self._largest_remainder(n_demo, [strata_weights[k] for k in nonempty])
                for k, n_k in zip(nonempty, alloc):
                    if n_k <= 0:
                        continue
                    pool = avail[k]
                    if pool.size < n_k:                              # thin stratum -> with replacement + log
                        if shortage is not None:
                            shortage.setdefault(k, {"empty": 0, "replacement": 0})["replacement"] += int(n_k - pool.size)
                        parts.append(pool[generator.integers(0, pool.size, size=n_k)])
                    else:
                        parts.append(pool[generator.choice(pool.size, size=n_k, replace=False)])
                    if account is not None:
                        account[int(k)] = account.get(int(k), 0) + int(n_k)
        idx = np.concatenate(parts)[:batch_size] if parts else generator.integers(0, self.size, size=batch_size)
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
