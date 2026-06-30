"""Multi-step + multi-agent RL sanity worlds — the siblings of the 1-step contextual bandit (``sanity_rl``).

Still physics-free and fast (seconds), exercising the real :func:`build_policy` backbones in a *sequential* and a
*cooperative* RL loop, so architecture sanity covers credit assignment and coordination too:

* :class:`LatticeNav` — ``grid`` (4-neighbour) or ``hex`` (6-neighbour) navigation: a point agent reads per-vertex
  features over the lattice hypergraph and moves to a goal landmark over a short horizon (multi-step credit
  assignment). The ``hex`` lattice is the grid-cell / path-integration substrate (ties to the holonomy line).
* :class:`CollabBandit` — a 2-agent coordination task: the two agents' actions must *sum* to a context target
  (neither can solve it alone), the fast analog of the collaborative Galambos scenario.

Reward 0.0 is optimal in every world; report final reward + wall + params per backbone.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from hymeko_rl.policy import ActorCritic, build_policy
from hymeko_rl.topology_zoo import _signed_graph


@dataclass(frozen=True)
class WorldConfig:
    lattice: str = "grid"          # "grid" (4-neighbour) or "hex" (6-neighbour)
    size: int = 3                  # size x size landmark vertices
    horizon: int = 8
    feat: int = 3                  # per-vertex obs: [landmark - agent (2), is_goal (1)]
    seed: int = 0


def _lattice(kind: str, size: int) -> "tuple[torch.Tensor, list[tuple[int, int]]]":
    """Landmark positions ``(N, 2)`` in ``[0,1]^2`` + undirected edges for a ``grid`` or ``hex`` lattice."""
    idx: dict[tuple[int, int], int] = {}
    pos: list[tuple[float, float]] = []
    for r in range(size):
        for c in range(size):
            idx[(r, c)] = len(pos)
            x = c + (0.5 if (kind == "hex" and r % 2) else 0.0)
            pos.append((x, r * (0.866 if kind == "hex" else 1.0)))
    edges: list[tuple[int, int]] = []
    for (r, c), i in idx.items():
        nbrs = [(r, c + 1), (r + 1, c)]                      # right, down (undirected -> also left, up)
        if kind == "hex":
            nbrs.append((r + 1, c + (1 if r % 2 else -1)))   # the hex diagonal -> 6-neighbour
        edges += [(i, idx[nb]) for nb in nbrs if nb in idx]
    p = torch.tensor(pos)
    p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-9)
    return p, edges


class LatticeNav:
    """Vectorised multi-step navigation on a signed lattice hypergraph (batch of agents in parallel)."""

    def __init__(self, cfg: WorldConfig) -> None:
        self.cfg = cfg
        self.pos, edges = _lattice(cfg.lattice, cfg.size)
        self.n = self.pos.shape[0]
        self.hg = _signed_graph(self.n, edges, seed=cfg.seed, tag=cfg.lattice)
        self._gen = torch.Generator().manual_seed(cfg.seed + 1)
        self._agent = torch.zeros(1, 2)
        self._goal = torch.zeros(1, dtype=torch.long)
        self._t = 0

    def reset(self, batch: int) -> torch.Tensor:
        self._agent = torch.rand(batch, 2, generator=self._gen)
        self._goal = torch.randint(0, self.n, (batch,), generator=self._gen)
        self._t = 0
        return self._obs()

    def _obs(self) -> torch.Tensor:
        rel = self.pos[None, :, :] - self._agent[:, None, :]                  # (B, N, 2)
        is_goal = torch.zeros(self._agent.shape[0], self.n, 1)
        is_goal[torch.arange(self._agent.shape[0]), self._goal, 0] = 1.0
        return torch.cat([rel, is_goal], dim=-1)                             # (B, N, 3)

    def step(self, action: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, bool]":
        self._agent = (self._agent + 0.25 * torch.tanh(action)).clamp(0.0, 1.0)
        reward = -(self._agent - self.pos[self._goal]).norm(dim=1)            # (B,), 0 == at goal
        self._t += 1
        return self._obs(), reward, self._t >= self.cfg.horizon


class CollabBandit:
    """2-agent coordination: both agents see the context; reward ``-‖(a_A + a_B) - target‖`` — neither solves it
    alone, only the *sum* matches (the fast analog of the cooperative Galambos delivery). 1-step."""

    def __init__(self, cfg: WorldConfig) -> None:
        self.cfg = cfg
        self.pos, edges = _lattice(cfg.lattice, cfg.size)
        self.n = self.pos.shape[0]
        self.hg = _signed_graph(self.n, edges, seed=cfg.seed, tag=f"{cfg.lattice}collab")
        self._gen = torch.Generator().manual_seed(cfg.seed + 2)

    def sample(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.n, self.cfg.feat, generator=self._gen)

    def target(self, ctx: torch.Tensor) -> torch.Tensor:
        return torch.tanh(ctx.mean(dim=1)[:, :2])                            # (B, 2)

    def reward(self, ctx: torch.Tensor, a_a: torch.Tensor, a_b: torch.Tensor) -> torch.Tensor:
        return -((a_a + a_b - self.target(ctx)) ** 2).mean(dim=1)            # (B,), 0 == coordinated


def _gauss_logp(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (-0.5 * ((action - mean) / std) ** 2 - std.log()).sum(dim=-1)


def _build(kind: str, obs_feat: int, n: int, hg: object, *, hidden: int, action_dim: int) -> ActorCritic:
    kw: dict[str, Any] = {"hidden": hidden}
    if kind != "mlp":
        kw["hg_state"] = hg
    obs_dim = obs_feat if kind != "mlp" else n * obs_feat
    return build_policy(kind, obs_dim=obs_dim, action_dim=action_dim, **kw)


def _train_nav(ac: ActorCritic, world: LatticeNav, *, iters: int, batch: int, lr: float, gamma: float) -> float:
    torch.manual_seed(world.cfg.seed)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    for _ in range(iters):
        obs = world.reset(batch)
        logps: list[torch.Tensor] = []
        rews: list[torch.Tensor] = []
        for _ in range(world.cfg.horizon):
            mean = ac.action_mean(obs)
            std = ac.log_std.exp()
            a = mean + std * torch.randn_like(mean)
            logps.append(_gauss_logp(a, mean, std))
            obs, r, _ = world.step(a)
            rews.append(r)
        ret = torch.zeros(batch)
        rets: list[torch.Tensor] = []
        for r in reversed(rews):
            ret = r + gamma * ret
            rets.insert(0, ret)
        adv = (rt := torch.stack(rets)) - rt.mean(dim=1, keepdim=True)
        loss = -(torch.stack(logps) * adv.detach()).mean()
        opt.zero_grad()
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
        opt.step()
    obs, r = world.reset(256), torch.zeros(256)
    with torch.no_grad():
        for _ in range(world.cfg.horizon):
            obs, r, _ = world.step(ac.action_mean(obs))
    return float(r.mean())


def run_world_sanity(kinds: "tuple[str, ...]" = ("mlp", "hsikan", "sa_hsikan", "mixture"), *,
                     cfg: WorldConfig | None = None, iters: int = 120, batch: int = 64, lr: float = 1e-2,
                     hidden: int = 32, gamma: float = 0.95) -> "dict[str, dict[str, float]]":
    """Train each backbone on the lattice navigation world; report ``{kind: {reward, wall_s, params}}`` (final
    reward 0.0 = sitting on the goal). # Postconditions every kind trains to a finite reward."""
    world = LatticeNav(cfg or WorldConfig())
    out: dict[str, dict[str, float]] = {}
    for kind in kinds:
        ac = _build(kind, world.cfg.feat, world.n, world.hg, hidden=hidden, action_dim=2)
        t0 = time.perf_counter()
        reward = _train_nav(ac, world, iters=iters, batch=batch, lr=lr, gamma=gamma)
        out[kind] = {"reward": round(reward, 4), "wall_s": round(time.perf_counter() - t0, 2),
                     "params": int(ac.n_parameters())}
    return out


def run_collab_sanity(kind: str = "hsikan", *, cfg: WorldConfig | None = None, iters: int = 400, batch: int = 64,
                      lr: float = 1e-2, hidden: int = 32) -> "dict[str, float | str | int]":
    """Two cooperating agents (shared reward = CTDE) on :class:`CollabBandit`; report the joint reward (0.0 =
    coordinated) + total params. The fast analog of the collaborative Galambos delivery."""
    cfg = cfg or WorldConfig()
    bandit = CollabBandit(cfg)
    ac_a = _build(kind, cfg.feat, bandit.n, bandit.hg, hidden=hidden, action_dim=2)
    ac_b = _build(kind, cfg.feat, bandit.n, bandit.hg, hidden=hidden, action_dim=2)
    opt = torch.optim.Adam(list(ac_a.parameters()) + list(ac_b.parameters()), lr=lr)
    for _ in range(iters):
        ctx = bandit.sample(batch)
        ma, mb = ac_a.action_mean(ctx), ac_b.action_mean(ctx)
        sa, sb = ac_a.log_std.exp(), ac_b.log_std.exp()
        aa, ab = ma + sa * torch.randn_like(ma), mb + sb * torch.randn_like(mb)
        logp = _gauss_logp(aa, ma, sa) + _gauss_logp(ab, mb, sb)
        r = bandit.reward(ctx, aa, ab)
        loss = -(logp * (r - r.mean()).detach()).mean()
        opt.zero_grad()
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
        opt.step()
    with torch.no_grad():
        ctx = bandit.sample(512)
        reward = float(bandit.reward(ctx, ac_a.action_mean(ctx), ac_b.action_mean(ctx)).mean())
    return {"kind": kind, "reward": round(reward, 4),
            "params": int(ac_a.n_parameters() + ac_b.n_parameters())}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", default="grid", choices=("grid", "hex", "collab"))
    ap.add_argument("--iters", type=int, default=120)
    a = ap.parse_args()
    if a.world == "collab":
        for kind in ("mlp", "hsikan"):
            print(run_collab_sanity(kind, iters=max(a.iters, 400)))
        return 0
    res = run_world_sanity(cfg=WorldConfig(lattice=a.world), iters=a.iters)
    print(f"{a.world}-nav sanity (final reward 0.0 = on goal):")
    for kind, m in sorted(res.items(), key=lambda kv: -kv[1]["reward"]):
        print(f"  {kind:10} reward={m['reward']:+.4f}  wall={m['wall_s']:5.2f}s  params={m['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
