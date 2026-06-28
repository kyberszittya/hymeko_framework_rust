"""SAC — Soft Actor-Critic: maximum-entropy off-policy control on the HyMeKo cart-pole.

The strong stochastic baseline (survey ``reports/2026-06-21-offpolicy-rl-survey``). A squashed-Gaussian actor
maximises reward *plus* policy entropy ``J = Σ E[r + α·H(π)]``; twin soft-Q critics (clipped double-Q, as TD3)
bound overestimation; the temperature ``α`` is auto-tuned to a target entropy. Reuses the off-policy
scaffolding (``ReplayBuffer``, ``QCritic``, the backbone family, ``_polyak``, ``eval_balance``) — only the
actor (stochastic vs deterministic) and the entropy-augmented update differ from DDPG/TD3, so this is a
separate trainer (§6.5 #8: structural variant) rather than another config of the deterministic core.

**The entropy seat.** The ``α·log π`` term is the maximum-entropy regulariser (signal (1) of three: policy
entropy). It is the explicit insertion point where the *epistemic* signal — critic-ensemble disagreement
``Var_k Q`` (the "TD-k" idea) — or the HyMeKo-native *structural* entropy of the state hypergraph
(``hymeko entropy``; the seat ``ppo.py`` flags) could be substituted or added. SAC instantiates the seat;
those are the planned extensions (see the report).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_rl.ddpg import QCritic, _backbone, _polyak
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.normalize import RunningRMS
from hymeko_rl.policy import POLICY_KINDS, PerNodeActionHead
from hymeko_rl.replay import ReplayBuffer
from hymeko_rl.train_inverted_pendulum import eval_balance

_LOG_STD_MIN, _LOG_STD_MAX = -20.0, 2.0


class _SquashedGaussianActorBase(nn.Module):
    """Shared interface for the SAC actors (pooled or per-node) — the seam ``train_sac``/eval drive.

    Subclasses set ``action_scale``/``action_dim`` and implement ``sample`` (reparameterised, returns
    ``(action, log_prob)``) and ``action_mean`` (deterministic greedy). Declaring the contract here lets
    ``build_sac`` return one type and keeps both actors swappable without per-variant call sites (§6.5 #8)."""

    action_scale: float
    action_dim: int

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SquashedGaussianActor(_SquashedGaussianActorBase):
    """A tanh-squashed diagonal-Gaussian policy: ``a = action_scale · tanh(μ + σ⊙ε)``.

    # Postconditions ``sample`` returns ``(action ∈ [-scale, scale], log_prob)`` with the tanh
    change-of-variables correction; ``action_mean`` is the deterministic ``action_scale·tanh(μ)`` (for eval).
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, action_scale: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.mu = nn.Linear(feat_dim, action_dim)
        self.log_std = nn.Linear(feat_dim, action_dim)
        self.action_scale = float(action_scale)
        self.action_dim = int(action_dim)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample. Returns ``(action, log_prob (B,))``."""
        h = self.backbone(obs)
        mu = self.mu(h)
        std = self.log_std(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()
        dist: Any = torch.distributions.Normal(mu, std)
        pre = dist.rsample()
        log_prob = dist.log_prob(pre).sum(-1)
        squashed = torch.tanh(pre)
        # tanh change-of-variables: log π(a) = log N(pre) − Σ log(1 − tanh²(pre)).
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        return squashed * self.action_scale, log_prob

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic mean action (greedy, for evaluation)."""
        return self.action_scale * torch.tanh(self.mu(self.backbone(obs)))


class PerNodeSquashedGaussianActor(_SquashedGaussianActorBase):
    """A squashed-Gaussian actor whose ``μ``/``log σ`` are **per-node** (the non-collapsing actor head).

    Reads the backbone's per-vertex activations (``node_activations``, not the pooled output), and gives each
    actuated joint its own ``μ/log σ`` from its vertex + the global pool (:class:`hymeko_rl.policy.PerNodeActionHead`).
    A structural variant of :class:`SquashedGaussianActor` (§6.5 #8), duck-typing its interface so
    ``train_sac`` drives it unchanged. The robot payoff test for the multidim-readout finding.

    # Preconditions ``backbone`` exposes ``node_activations(obs) -> (B, N, hidden)``; ``act_vertices`` index
    ``[0, N)``, one per actuator (``len == action_dim``)."""

    mu: PerNodeActionHead          # declared so mypy uses these (not nn.Module.__getattr__'s Tensor|Module union)
    log_std: PerNodeActionHead

    def __init__(self, backbone: nn.Module, hidden: int, action_dim: int, action_scale: float,
                 act_vertices: Sequence[int]) -> None:
        super().__init__()
        if not hasattr(backbone, "node_activations"):
            raise TypeError("PerNodeSquashedGaussianActor needs a backbone exposing node_activations (HSiKAN)")
        if len(list(act_vertices)) != action_dim:
            raise ValueError(f"act_vertices ({len(list(act_vertices))}) must match action_dim ({action_dim})")
        self.backbone = backbone
        self.mu = PerNodeActionHead(hidden, act_vertices)
        self.log_std = PerNodeActionHead(hidden, act_vertices)
        self.action_scale = float(action_scale)
        self.action_dim = int(action_dim)

    def _node_acts(self, obs: torch.Tensor) -> torch.Tensor:
        # __init__ asserts the backbone exposes node_activations (HSiKAN); nn.Module's dynamic attr typing
        # can't see it, so the call is type-checked away here (invariant guaranteed at construction).
        acts: torch.Tensor = self.backbone.node_activations(obs)   # type: ignore[operator]  # (B, N, hidden)
        return acts

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised per-joint sample — identical tanh-Gaussian semantics to the pooled actor."""
        h = self._node_acts(obs)
        mu = self.mu(h)
        std = self.log_std(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp()
        dist: Any = torch.distributions.Normal(mu, std)
        pre = dist.rsample()
        log_prob = dist.log_prob(pre).sum(-1)
        squashed = torch.tanh(pre)
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        return squashed * self.action_scale, log_prob

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic per-joint mean action (greedy, for evaluation)."""
        return self.action_scale * torch.tanh(self.mu(self._node_acts(obs)))


def build_sac(kind: str, *, obs_dim: int, flat_dim: int, action_dim: int, action_scale: float,
              n_critics: int = 2, hidden: int = 64, device: torch.device | str = "cpu",
              actor_head: str = "pooled", act_vertices: Sequence[int] | None = None,
              **kw: object) -> tuple[_SquashedGaussianActorBase, list[QCritic]]:
    """Construct ``(stochastic actor, [critic, …])`` with independent backbones of ``kind``.

    ``actor_head`` selects the actor's readout: ``"pooled"`` (default — ``Linear`` over the mean-pooled
    backbone) or ``"per_node"`` (each joint from its own vertex; needs ``act_vertices`` + a per-vertex
    backbone). The **critic always pools** (a scalar Q/V wants an aggregate). ``device`` moves the nets.

    # Preconditions ``kind in POLICY_KINDS``; ``hsikan``/``signedkan`` require ``hg_state=`` in ``kw``;
    ``actor_head="per_node"`` requires ``act_vertices`` and a per-vertex backbone (not ``mlp``)."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected {POLICY_KINDS}")
    ab, feat = _backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)
    if actor_head == "per_node":
        if kind == "mlp":
            raise ValueError("actor_head='per_node' needs a per-vertex backbone (hsikan/signedkan), not mlp")
        if act_vertices is None:
            raise ValueError("actor_head='per_node' requires act_vertices (one vertex per actuator)")
        actor: _SquashedGaussianActorBase = PerNodeSquashedGaussianActor(
            ab, feat, action_dim, action_scale, act_vertices).to(device)
    elif actor_head == "pooled":
        actor = SquashedGaussianActor(ab, feat, action_dim, action_scale).to(device)
    else:
        raise ValueError(f"actor_head must be 'pooled' or 'per_node'; got {actor_head!r}")
    critics = [QCritic(_backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)[0], feat, action_dim).to(device)
               for _ in range(max(1, n_critics))]
    return actor, critics


@dataclass(frozen=True)
class SACConfig:
    """SAC hyperparameters (cart-pole defaults)."""

    total_steps: int = 30_000
    start_steps: int = 1_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    max_grad_norm: float = 10.0       # gradient-norm clip on actor + critics (prevents the Q-overestimation blow-up)
    reward_norm: bool = True          # normalise rewards by a running RMS → bounds the Q-scale (anti-divergence)
    capacity: int = 100_000
    n_critics: int = 2
    target_entropy: float | None = None   # default: -action_dim (the SAC heuristic)
    eval_every: int = 5_000
    n_eval: int = 10
    seed: int = 0


def train_sac(actor: _SquashedGaussianActorBase, critics: list[QCritic], env: InvertedPendulumEnv,
              cfg: SACConfig, *,
              eval_fn: Callable[[Any, Any], float] | None = None) -> list[float]:
    """Train SAC on ``env``; returns the periodic eval curve.

    Twin soft-Q critics with the entropy-augmented target ``y = r + γ(1-d)(min_i Q̄_i(s',a') − α·logπ(a'))``;
    a reparameterised actor maximising ``min_i Q_i(s,a) − α·logπ(a)``; auto-tuned ``α`` to ``target_entropy``.
    Truncation is stored as non-terminal (bootstrap past the time limit), as in DDPG/TD3.

    ``eval_fn`` scores the curve: ``None`` (default) uses :func:`eval_balance` (cart-pole upright-steps,
    behaviour unchanged); a real-topology task injects a return-based eval. Called ``eval_fn(env, actor)``.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = next(actor.parameters()).device                          # CPU or CUDA — alpha, batches, inference follow it
    t_critics = [copy.deepcopy(c) for c in critics]
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=cfg.critic_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=dev)
    al_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
    action_dim = actor.action_dim                                  # both actor variants expose this
    target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(action_dim)
    scale = actor.action_scale
    space_shape = env.observation_space.shape
    assert space_shape is not None
    buf = ReplayBuffer(cfg.capacity, tuple(int(d) for d in space_shape), action_dim)
    history: list[float] = []
    reward_rms = RunningRMS()                                       # bounds the Q-scale (anti-divergence)
    obs, _ = env.reset(seed=cfg.seed)
    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.start_steps:
            action = rng.uniform(-scale, scale, size=action_dim).astype(np.float32)
        else:
            with torch.no_grad():
                action = actor.sample(torch.as_tensor(obs[None], dtype=torch.float32, device=dev),
                                      )[0].squeeze(0).cpu().numpy()
        nobs, rew, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, float(rew), nobs, bool(terminated))
        obs = nobs if not (terminated or truncated) else env.reset()[0]

        if buf.size >= cfg.batch_size and step > cfg.start_steps:
            s, a, r, s2, d = (x.to(dev) for x in buf.sample(cfg.batch_size, generator=rng))
            alpha = log_alpha.exp()
            with torch.no_grad():
                a2, logp2 = actor.sample(s2)
                q_next = torch.stack([tc(s2, a2) for tc in t_critics], 0).amin(0) - alpha * logp2
                if cfg.reward_norm:
                    r = reward_rms.normalize(r)                     # bounded-scale reward → bounded Q-target
                y = r + cfg.gamma * (1 - d) * q_next
            c_loss = sum(F.mse_loss(c(s, a), y) for c in critics)
            c_opt.zero_grad()
            c_loss.backward()   # type: ignore[union-attr]  # sum() of tensors is a tensor
            torch.nn.utils.clip_grad_norm_([p for c in critics for p in c.parameters()], cfg.max_grad_norm)
            c_opt.step()

            ap, logp = actor.sample(s)
            q_pi = torch.stack([c(s, ap) for c in critics], 0).amin(0)
            a_loss = (alpha.detach() * logp - q_pi).mean()
            a_opt.zero_grad()
            a_loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            a_opt.step()

            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
            al_opt.zero_grad()
            alpha_loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            al_opt.step()

            for tc, c in zip(t_critics, critics):
                _polyak(tc, c, cfg.tau)

        if step % cfg.eval_every == 0:
            history.append(eval_balance(env, actor, cfg.n_eval, seed=20_000)
                           if eval_fn is None else eval_fn(env, actor))
    return history


def run_sac(kind: str = "hsikan", *, hidden: int = 64, seed: int = 0, n_eval: int = 20,
            cfg: SACConfig | None = None, save: str | None = None) -> dict[str, float | str]:
    """Build env + actor/critics, train SAC, evaluate. Mirrors ``run_balance``/``run_offpolicy``."""
    cfg = cfg or SACConfig(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space_shape = env.observation_space.shape
    assert space_shape is not None
    n_vertices, feat = int(space_shape[0]), int(space_shape[1])
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_sac(kind, obs_dim=feat, flat_dim=n_vertices * feat, action_dim=1,
                               action_scale=env.force_mag, n_critics=cfg.n_critics, hidden=hidden, **kw)
    floor = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=10_000)
    history = train_sac(actor, critics, env, cfg)
    upright = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=20_000)
    if save is not None:
        from hymeko_rl.policy_store import policy_to_hymeko
        policy_to_hymeko(actor.state_dict(), save, meta={
            "algo": "sac", "backbone": kind, "upright": round(upright, 1), "seed": seed})
    return dict(algo="sac", policy=kind, n_critics=cfg.n_critics,
                n_params=sum(p.numel() for p in actor.parameters()),
                untrained_upright_steps=round(floor, 2), upright_steps=round(upright, 2),
                curve=json.dumps([round(h, 1) for h in history]), max_steps=float(env.max_steps))


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="hsikan", choices=list(POLICY_KINDS))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="store the trained policy to this .hymeko (with provenance)")
    a = ap.parse_args(argv)
    print(json.dumps(run_sac(a.policy, hidden=a.hidden, seed=a.seed, save=a.save,
                             cfg=SACConfig(total_steps=a.steps, seed=a.seed)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
