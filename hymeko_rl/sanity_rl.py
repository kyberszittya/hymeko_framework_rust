"""Tiny, physics-free RL sanity testbeds — architecture sanity + a fast performance leaderboard in *seconds*.

Every other RL env here is MuJoCo (minutes to hours); a backbone regression or a wiring bug should not need an
hour to surface. This is the RL analog of :mod:`hymeko_rl.structural_probe`: a **contextual bandit** on a small
signed hypergraph, trained with a minimal REINFORCE loop through the real :func:`build_policy` backbones
(``mlp`` / ``hsikan`` / ``sa_hsikan`` / ``mixture``). Two targets:

* ``"flat"``       — optimal action = pooled context (no graph): every backbone should solve it (pure sanity).
* ``"structural"`` — optimal action = pooled signed 1-hop aggregation ``mean(A x)``: a second sanity that the
  graph-reading path works. It is still LINEAR in ``x``, so it does **not** discriminate HSiKAN from MLP (an MLP
  represents any linear graph op trivially — measured: both tie ~-0.25). The genuine *accuracy* discriminator
  needs a NONLINEAR graph property (cycle parity / Z2 holonomy) — the planned holonomy-discriminator toy.

Reward is ``-MSE(action, a*)`` so 0.0 is optimal. What this testbed exposes per backbone, in seconds: whether it
learns at all (accuracy / regression-catch) and its wall-time (e.g. HSiKAN's launch-bound shows immediately).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from signed_kan import set_deploy_mode

from hymeko_rl.policy import ActorCritic, build_policy
from hymeko_rl.topology_zoo import ring


@dataclass(frozen=True)
class BanditConfig:
    n_vertices: int = 6
    feat: int = 4
    action_dim: int = 3
    target: str = "structural"          # "structural" (needs the graph) or "flat" (pooled only)
    seed: int = 0


class ContextualBandit:
    """A 1-step continuous contextual bandit on a fixed signed ring. ``sample(batch)`` draws contexts
    ``(B, N, feat)``; ``optimal`` maps a context batch to the best action ``(B, action_dim)``; ``reward`` is
    ``-MSE(action, optimal)``. # Preconditions ``action_dim <= feat``."""

    def __init__(self, cfg: BanditConfig) -> None:
        if cfg.action_dim > cfg.feat:
            raise ValueError(f"action_dim ({cfg.action_dim}) must be <= feat ({cfg.feat})")
        self.cfg = cfg
        self.hg = ring(cfg.n_vertices, seed=cfg.seed)            # the signed structure the policy reads
        a = torch.zeros(cfg.n_vertices, cfg.n_vertices)
        for (i, j), s in zip(self.hg.edges.tolist(), self.hg.signs.tolist()):
            a[i, j] = float(s)
        self._a = a                                             # the signed 1-hop operator (the structural signal)
        self._gen = torch.Generator().manual_seed(cfg.seed + 1)

    def sample(self, batch: int) -> torch.Tensor:
        return torch.randn(batch, self.cfg.n_vertices, self.cfg.feat, generator=self._gen)

    def optimal(self, ctx: torch.Tensor) -> torch.Tensor:
        if self.cfg.target == "structural":                     # pooled signed 1-hop aggregation (graph-reading
            agg = torch.einsum("nm,bmf->bnf", self._a, ctx)     # path sanity; still LINEAR in x so it does NOT
            return torch.tanh(agg.mean(dim=1)[:, : self.cfg.action_dim])   # discriminate HSiKAN from MLP)
        return torch.tanh(ctx.mean(dim=1)[:, : self.cfg.action_dim])     # flat: pooled context, no graph

    def reward(self, ctx: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return -((action - self.optimal(ctx)) ** 2).mean(dim=1)         # (B,), 0.0 == optimal


def _reinforce(ac: ActorCritic, bandit: ContextualBandit, *, steps: int, batch: int, lr: float,
               seed: int) -> float:
    """Minimal REINFORCE (Gaussian policy, batch baseline) — returns the final deterministic eval reward."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    for _ in range(steps):
        ctx = bandit.sample(batch)
        mean = ac.action_mean(ctx)
        std = torch.exp(ac.log_std)
        noise = torch.randn_like(mean)
        action = mean + std * noise
        logp = (-0.5 * ((action - mean) / std) ** 2 - std.log()).sum(dim=1)
        r = bandit.reward(ctx, action)
        loss = -(logp * (r - r.mean()).detach()).mean()
        opt.zero_grad()
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
        opt.step()
    with torch.no_grad():
        ctx = bandit.sample(512)
        return float(bandit.reward(ctx, ac.action_mean(ctx)).mean())


def _deploy_latency_ms(ac: ActorCritic, bandit: ContextualBandit, *, reps: int = 60) -> float:
    """Median B=1 ``action_mean`` latency (ms) — the rollout/launch-bound regime where the recent enhancements
    pay off (SA-HSiKAN's B^L collapse; the cr_cheby cell). Batched training hides this; B=1 deploy exposes it.
    Warm-up + median per the bench discipline."""
    x1 = bandit.sample(1)
    set_deploy_mode(ac, True)                           # train-CR / deploy-Chebyshev: take the fast path (no-op
    try:                                                # for cr / mlp; switches cr_cheby cells to chebyshev_forward)
        with torch.no_grad():
            for _ in range(8):
                ac.action_mean(x1)
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                ac.action_mean(x1)
                ts.append(time.perf_counter() - t)
    finally:
        set_deploy_mode(ac, False)                      # restore the CR train path
    ts.sort()
    return round(ts[len(ts) // 2] * 1e3, 3)


# label -> (policy kind, extra build_policy kwargs). Folds in the recent perf work: the cr_cheby cell on HSiKAN
# (vanilla Catmull-Rom vs Chebyshev-CR control points) and the SA-HSiKAN B^L collapse (cr_cheby by default).
_DEFAULT_VARIANTS: "tuple[tuple[str, str, dict[str, Any]], ...]" = (
    ("mlp", "mlp", {}),
    ("hsikan-cr", "hsikan", {"activation": "cr"}),
    ("hsikan-cheby", "hsikan", {"activation": "cr_cheby"}),
    ("sa_hsikan", "sa_hsikan", {}),
    ("mixture", "mixture", {}),
)


def run_bandit_sanity(variants: "tuple[str | tuple[str, str, dict[str, Any]], ...]" = _DEFAULT_VARIANTS, *,
                      steps: int = 400, batch: int = 64, lr: float = 5e-3, hidden: int = 32,
                      cfg: BanditConfig | None = None) -> "dict[str, dict[str, float]]":
    """Train each variant; report ``{label: {reward, train_s, deploy_ms, params}}`` (reward 0.0 = optimal;
    ``deploy_ms`` = B=1 rollout latency). A variant is a ``kind`` string or a ``(label, kind, kwargs)`` triple.

    # Postconditions every variant trains to a finite reward; a healthy backbone reaches ``> -0.2`` (flat)."""
    cfg = cfg or BanditConfig()
    bandit = ContextualBandit(cfg)
    out: dict[str, dict[str, float]] = {}
    for v in variants:
        label, kind, extra = (v, v, {}) if isinstance(v, str) else v   # noqa: E501
        torch.manual_seed(cfg.seed)
        kw: dict[str, Any] = {"hidden": hidden, **extra}
        if kind != "mlp":
            kw["hg_state"] = bandit.hg
        obs_dim = cfg.feat if kind != "mlp" else cfg.n_vertices * cfg.feat
        ac: ActorCritic = build_policy(kind, obs_dim=obs_dim, action_dim=cfg.action_dim, **kw)
        t0 = time.perf_counter()
        reward = _reinforce(ac, bandit, steps=steps, batch=batch, lr=lr, seed=cfg.seed)
        out[str(label)] = {"reward": round(reward, 4), "train_s": round(time.perf_counter() - t0, 2),
                           "deploy_ms": _deploy_latency_ms(ac, bandit), "params": int(ac.n_parameters())}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="structural", choices=("structural", "flat"))
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    res = run_bandit_sanity(steps=a.steps, cfg=BanditConfig(target=a.target))
    print(f"contextual-bandit ({a.target}; reward 0.0=optimal, deploy_ms=B=1 rollout latency):")
    for label, m in sorted(res.items(), key=lambda kv: kv[1]["deploy_ms"]):
        print(f"  {label:13} reward={m['reward']:+.4f}  train={m['train_s']:5.2f}s  "
              f"deploy={m['deploy_ms']:7.3f}ms  params={m['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
