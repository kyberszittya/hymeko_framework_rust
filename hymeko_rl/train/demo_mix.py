"""Demo-mix pools — split held expert demonstrations into a sustained-contact pool and a delivery-completion pool,
then mix them at a chosen ratio.

The previous BC fine-tune over-fit *holding*: trained on all held states of a sustained-pushing expert, the clone
learned to hold two-finger contact but lost the delivery-completion behaviour (ft_dom 0.75→0.417). The fix under
test: label each expert state as inside a sustained-PUSH window (``pool_sustained``) or not (``pool_deliver`` =
approach + push-to-zone + delivery completion), and train on a *balanced* mixture so the clone sees enough
delivery-completion states to still finish. Reuses the single sustained-window definition
(:func:`hymeko_rl.eval.push_audit.sustained_windows_raw`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hymeko_rl.eval.push_audit import sustained_windows_raw


@dataclass
class TaggedPools:
    """Held-only expert states split by sustained-PUSH membership. Each pool is ``(obs, acts)``."""
    sustained_obs: np.ndarray
    sustained_acts: np.ndarray
    deliver_obs: np.ndarray
    deliver_acts: np.ndarray
    n_episodes: int
    n_delivered: int

    @property
    def n_sustained(self) -> int:
        return int(self.sustained_obs.shape[0])

    @property
    def n_deliver(self) -> int:
        return int(self.deliver_obs.shape[0])

    def summary(self) -> dict:
        return {"n_episodes": self.n_episodes, "n_delivered": self.n_delivered,
                "n_sustained_states": self.n_sustained, "n_deliver_states": self.n_deliver}


def collect_tagged_demos(env: Any, expert_factory: Callable[[Any], Any], *, n_episodes: int, seed0: int,
                         k_sustained: int = 5, progress_eps: float = 0.002, body_eps: float = 0.005) -> TaggedPools:
    """Roll the expert for ``n_episodes`` (held-only) and tag each state as sustained-PUSH vs delivery-completion.

    # Preconditions: ``expert_factory(env)`` returns a fresh ``.reset()``/``.action(env)`` controller; ``env`` is a
      PlanarGraspEnv. # Postconditions: a :class:`TaggedPools`; only states from HELD episodes are kept (filter ≡
      grading rule); raises RuntimeError if no held episode was collected."""
    dwell_need = int(getattr(env, "success_steps", 1))
    s_obs: list[np.ndarray] = []
    s_act: list[np.ndarray] = []
    d_obs: list[np.ndarray] = []
    d_act: list[np.ndarray] = []
    n_delivered = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        expert = expert_factory(env)
        expert.reset()
        prev_dist = float(info["disk_to_zone"])
        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        both: list[bool] = []
        toward: list[float] = []
        body_prog: list[float] = []
        body_c: list[bool] = []
        consec = 0
        delivered = False
        for _ in range(env.max_steps):
            action = np.asarray(expert.action(env), dtype=np.float32)
            ep_obs.append(np.asarray(obs, dtype=np.float32))
            ep_act.append(action)
            obs, _r, term, trunc, info = env.step(action)
            dist = float(info["disk_to_zone"])
            tw = max(0.0, prev_dist - dist)
            fingertip = bool(info["fingertip_contact"])
            body_now = bool(info["arm_body_contact_this_step"])
            both.append(bool(info["both_contact"]))
            toward.append(tw)
            body_prog.append(tw if (body_now and not fingertip) else 0.0)
            body_c.append(body_now)
            consec = consec + 1 if bool(info["in_zone"]) else 0
            delivered = delivered or consec >= dwell_need
            prev_dist = dist
            if term or trunc:
                break
        if not delivered:
            continue
        n_delivered += 1
        windows = sustained_windows_raw(np.asarray(both), np.asarray(toward), np.asarray(body_prog),
                                        np.asarray(body_c), progress_eps=progress_eps, body_eps=body_eps,
                                        k=k_sustained)
        in_window = np.zeros(len(ep_obs), dtype=bool)
        for s, e in windows:
            in_window[s:e] = True
        for i in range(len(ep_obs)):
            (s_obs if in_window[i] else d_obs).append(ep_obs[i])
            (s_act if in_window[i] else d_act).append(ep_act[i])
    if n_delivered == 0:
        raise RuntimeError(f"collected no held demos ({n_delivered}/{n_episodes}); adjust expert/seed")
    empty_o = np.empty((0, *np.asarray(s_obs[0] if s_obs else d_obs[0]).shape), dtype=np.float32)
    empty_a = np.empty((0, *np.asarray(s_act[0] if s_act else d_act[0]).shape), dtype=np.float32)
    return TaggedPools(
        sustained_obs=np.asarray(s_obs, dtype=np.float32) if s_obs else empty_o,
        sustained_acts=np.asarray(s_act, dtype=np.float32) if s_act else empty_a,
        deliver_obs=np.asarray(d_obs, dtype=np.float32) if d_obs else empty_o,
        deliver_acts=np.asarray(d_act, dtype=np.float32) if d_act else empty_a,
        n_episodes=n_episodes, n_delivered=n_delivered)


def mix_pools(pools: TaggedPools, frac_sustained: float, *, total: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``total`` (obs, act) samples with ``frac_sustained`` from the sustained pool and the rest from the
    delivery-completion pool (sampled with replacement so any ratio is realizable).

    # Preconditions: 0 ≤ frac_sustained ≤ 1; at least one pool non-empty. # Postconditions: ``(obs, acts)`` of
      length ``total`` (or fewer only if a required pool is empty — the ratio is then clamped, and the caller can
      detect it from the returned counts)."""
    if not 0.0 <= frac_sustained <= 1.0:
        raise ValueError(f"frac_sustained must be in [0,1], got {frac_sustained}")
    rng = np.random.default_rng(seed)
    n_s = int(round(frac_sustained * total))
    n_d = total - n_s
    # clamp to available pools (empty pool → shift the request to the other)
    if pools.n_sustained == 0:
        n_s, n_d = 0, total
    if pools.n_deliver == 0:
        n_s, n_d = total, 0

    def _draw(obs: np.ndarray, acts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0 or obs.shape[0] == 0:
            return obs[:0], acts[:0]
        idx = rng.integers(0, obs.shape[0], size=n)
        return obs[idx], acts[idx]

    so, sa = _draw(pools.sustained_obs, pools.sustained_acts, n_s)
    do, da = _draw(pools.deliver_obs, pools.deliver_acts, n_d)
    obs = np.concatenate([so, do], axis=0) if len(so) or len(do) else pools.deliver_obs[:0]
    acts = np.concatenate([sa, da], axis=0) if len(sa) or len(da) else pools.deliver_acts[:0]
    perm = rng.permutation(obs.shape[0])
    return obs[perm], acts[perm]
