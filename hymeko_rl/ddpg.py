"""Off-policy actor-critic — DDPG and TD3 on the HyMeKo cart-pole (one config-driven core).

The off-policy family beside the on-policy PPO baseline (survey:
``reports/2026-06-21-offpolicy-rl-survey``). A deterministic actor ``μ(s)`` is improved *through* a Q-critic
``Q(s,a)`` (the deterministic policy gradient); critics regress to a Polyak-target Bellman backup over a replay
buffer; exploration is additive Gaussian action noise. The state encoder is the *same* swappable backbone
(``mlp``/``hsikan``/``signedkan``) the PPO policy uses — architecture is orthogonal to the algorithm.

**DDPG and TD3 are one trainer** (§6.5 #1: config, not a forked function). TD3 = DDPG + three settings, each a
fix to DDPG's Q-overestimation/fragility: ``n_critics=2`` (clipped double-Q — target uses the min),
``policy_delay=2`` (delayed actor/target updates), ``target_noise>0`` (target-policy smoothing). DDPG is the
degenerate preset (``n_critics=1, policy_delay=1, target_noise=0``).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.normalize import RunningRMS
from hymeko_rl.policy import POLICY_KINDS, _BACKBONES
from hymeko_rl.replay import ReplayBuffer
from hymeko_rl.train_inverted_pendulum import eval_balance


class DeterministicActor(nn.Module):
    """``μ(s) = action_scale · tanh(head(backbone(s)))`` — a bounded deterministic policy.

    # Invariants output is in ``[-action_scale, action_scale]^{action_dim}``."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, action_scale: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, action_dim)
        self.action_scale = float(action_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_scale * torch.tanh(self.head(self.backbone(obs)))

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias so :func:`hymeko_rl.train_inverted_pendulum.eval_balance` can score the greedy policy."""
        return self.forward(obs)


class QCritic(nn.Module):
    """``Q(s,a) = head(concat(backbone(s), a))`` — an action-value over the same backbone family."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(feat_dim + action_dim, feat_dim), nn.ReLU(),
                                  nn.Linear(feat_dim, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.head(torch.cat([self.backbone(obs), action], dim=-1)).squeeze(-1)
        return q


def _backbone(kind: str, obs_dim: int, flat_dim: int, **kw: object) -> tuple[nn.Module, int]:
    """Build one backbone of the requested kind (mlp reads the flat obs; hsikan/signedkan the per-vertex obs).

    ``hidden`` is forwarded to the mlp too, so the off-policy MLP baseline can be widened to **params-match**
    the HSiKAN backbone (the standing rule: always compare against a matched-capacity control). ``hg_state``
    and other hsikan-only kwargs are not passed to the mlp.
    """
    if kind == "mlp":
        hidden = kw.get("hidden")
        if hidden is None:
            return _BACKBONES["mlp"](flat_dim)
        assert isinstance(hidden, int)   # callers pass an int width; narrow for the kwargs forward
        return _BACKBONES["mlp"](flat_dim, hidden=hidden)
    return _BACKBONES[kind](obs_dim, **kw)


def build_offpolicy(kind: str, *, obs_dim: int, flat_dim: int, action_dim: int, action_scale: float,
                    n_critics: int = 1, hidden: int = 64, device: torch.device | str = "cpu",
                    **kw: object) -> tuple[DeterministicActor, list[QCritic]]:
    """Construct ``(actor, [critic, …])`` with independent backbones of ``kind`` (the architecture swap).

    ``n_critics`` is 1 for DDPG, 2 for TD3 (clipped double-Q). ``device`` moves the nets (and the signed
    adjacency buffers/params travel with them) onto CPU (default) or a CUDA device.
    # Preconditions ``kind in POLICY_KINDS``; ``hsikan``/``signedkan`` require ``hg_state=`` in ``kw``."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected {POLICY_KINDS}")
    if n_critics < 1:
        raise ValueError(f"n_critics must be >= 1; got {n_critics}")
    ab, feat = _backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)
    actor = DeterministicActor(ab, feat, action_dim, action_scale).to(device)
    critics = [QCritic(_backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)[0], feat, action_dim).to(device)
               for _ in range(n_critics)]
    return actor, critics


@dataclass(frozen=True)
class OffPolicyConfig:
    """DDPG/TD3 hyperparameters (cart-pole defaults). The TD3 axes default to DDPG (off)."""

    total_steps: int = 30_000
    start_steps: int = 1_000          # uniform-random action warm-up to seed the buffer
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005                # Polyak averaging coefficient for the target nets
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    max_grad_norm: float = 10.0       # gradient-norm clip on actor + critics (prevents the Q-overestimation blow-up)
    reward_norm: bool = True          # normalise rewards by a running RMS → bounds the Q-scale (anti-divergence)
    capacity: int = 100_000
    noise_scale: float = 0.1          # exploration σ as a fraction of action_scale
    eval_every: int = 5_000
    n_eval: int = 10
    seed: int = 0
    # --- TD3 axes (DDPG defaults: a single critic, no delay, no target smoothing) ---
    n_critics: int = 1
    policy_delay: int = 1             # update actor/targets every `policy_delay` critic steps (TD3: 2)
    target_noise: float = 0.0         # target-policy smoothing σ as a fraction of action_scale (TD3: ~0.2)
    noise_clip: float = 0.5           # smoothing clip as a fraction of action_scale
    # --- BC warm-start bridge (default off → from-scratch behaviour unchanged) ---
    warm_start: bool = False          # the actor is pre-trained (BC): act with IT from step 0 (skip the
                                      # uniform-random `start_steps` that would overwrite the cloned behaviour).
    critic_warmup: int = 0            # update the CRITIC alone for this many steps before the actor moves, so a
                                      # cold critic's gradients don't destroy the cloned actor (set ~2000 for BC).


# Back-compat alias (DDPG was the first member); presets below select DDPG vs TD3.
DDPGConfig = OffPolicyConfig


def td3_config(**overrides: object) -> OffPolicyConfig:
    """A TD3 preset: twin critics, delayed policy, target-policy smoothing."""
    base: dict[str, object] = {"n_critics": 2, "policy_delay": 2, "target_noise": 0.2, "noise_clip": 0.5}
    base.update(overrides)
    return OffPolicyConfig(**base)   # type: ignore[arg-type]


def _polyak(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


def train_offpolicy(actor: DeterministicActor, critics: list[QCritic], env: InvertedPendulumEnv,
                    cfg: OffPolicyConfig, *,
                    eval_fn: Callable[[Any, Any], float] | None = None) -> list[float]:
    """Train DDPG/TD3 on ``env`` (one core); returns the periodic eval curve.

    Clipped double-Q when ``n_critics>1`` (target uses ``min`` over the target critics); delayed actor/target
    updates every ``policy_delay`` critic steps; target-policy smoothing when ``target_noise>0``. Time-limit
    truncation is stored as **non-terminal**, so the Bellman backup bootstraps past the time limit.

    ``eval_fn`` scores the curve: ``None`` (default) uses :func:`eval_balance` (cart-pole upright-steps,
    behaviour unchanged); a real-topology task injects a return-based eval (env-agnostic). It is called as
    ``eval_fn(env, actor) -> float`` at each ``eval_every`` checkpoint.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    t_actor = copy.deepcopy(actor)
    t_critics = [copy.deepcopy(c) for c in critics]
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=cfg.critic_lr)
    space_shape = env.observation_space.shape
    assert space_shape is not None
    buf = ReplayBuffer(cfg.capacity, tuple(int(d) for d in space_shape), actor.head.out_features)
    scale = actor.action_scale
    sigma, tn, nc = cfg.noise_scale * scale, cfg.target_noise * scale, cfg.noise_clip * scale
    history: list[float] = []
    updates = 0
    reward_rms = RunningRMS()                                       # bounds the Q-scale (anti-divergence)
    dev = next(actor.parameters()).device                          # CPU or CUDA — batches + inference follow it
    obs, _ = env.reset(seed=cfg.seed)
    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.start_steps and not cfg.warm_start:
            action = rng.uniform(-scale, scale, size=actor.head.out_features).astype(np.float32)
        else:                                                      # warm-started: act with the cloned actor now
            with torch.no_grad():
                mu = actor(torch.as_tensor(obs[None], dtype=torch.float32, device=dev)).squeeze(0).cpu().numpy()
            action = np.clip(mu + rng.normal(0, sigma, mu.shape), -scale, scale).astype(np.float32)
        nobs, rew, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, float(rew), nobs, bool(terminated))   # truncation is NOT a true terminal
        obs = nobs if not (terminated or truncated) else env.reset()[0]

        if buf.size >= cfg.batch_size and (cfg.warm_start or step > cfg.start_steps):
            s, a, r, s2, d = (x.to(dev) for x in buf.sample(cfg.batch_size, generator=rng))
            with torch.no_grad():
                a2 = t_actor(s2)
                if tn > 0:                                         # TD3 target-policy smoothing
                    noise = torch.clamp(torch.randn_like(a2) * tn, -nc, nc)
                    a2 = torch.clamp(a2 + noise, -scale, scale)
                q_next = torch.stack([tc(s2, a2) for tc in t_critics], 0).amin(0)   # clipped double-Q
                if cfg.reward_norm:
                    r = reward_rms.normalize(r)                     # bounded-scale reward → bounded Q-target
                y = r + cfg.gamma * (1 - d) * q_next
            c_loss = sum(F.mse_loss(c(s, a), y) for c in critics)
            c_opt.zero_grad()
            c_loss.backward()   # type: ignore[union-attr]  # sum() of tensors is a tensor
            torch.nn.utils.clip_grad_norm_([p for c in critics for p in c.parameters()], cfg.max_grad_norm)
            c_opt.step()
            updates += 1
            # Hold the actor during the critic warm-up so a cold critic can't wreck a cloned policy.
            if updates > cfg.critic_warmup and updates % cfg.policy_delay == 0:
                a_loss = -critics[0](s, actor(s)).mean()
                a_opt.zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                a_opt.step()
                _polyak(t_actor, actor, cfg.tau)
                for tc, c in zip(t_critics, critics):
                    _polyak(tc, c, cfg.tau)

        if step % cfg.eval_every == 0:
            history.append(eval_balance(env, actor, cfg.n_eval, seed=20_000)
                           if eval_fn is None else eval_fn(env, actor))
    return history


def run_offpolicy(algo: str = "ddpg", kind: str = "hsikan", *, hidden: int = 64, seed: int = 0,
                  n_eval: int = 20, cfg: OffPolicyConfig | None = None,
                  save: str | None = None) -> dict[str, float | str]:
    """Build env + actor/critics, train DDPG or TD3, evaluate. Mirrors ``run_balance`` (PPO) for comparison."""
    if algo not in ("ddpg", "td3"):
        raise ValueError(f"algo must be 'ddpg' or 'td3'; got {algo!r}")
    if cfg is None:
        cfg = td3_config(seed=seed) if algo == "td3" else OffPolicyConfig(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space_shape = env.observation_space.shape
    assert space_shape is not None
    n_vertices, feat = int(space_shape[0]), int(space_shape[1])
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_offpolicy(kind, obs_dim=feat, flat_dim=n_vertices * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=cfg.n_critics, hidden=hidden, **kw)
    floor = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=10_000)
    history = train_offpolicy(actor, critics, env, cfg)
    upright = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=20_000)
    if save is not None:
        from hymeko_rl.policy_store import policy_to_hymeko
        policy_to_hymeko(actor.state_dict(), save, meta={
            "algo": algo, "backbone": kind, "upright": round(upright, 1), "seed": seed})
    return dict(algo=algo, policy=kind, n_critics=cfg.n_critics,
                n_params=sum(p.numel() for p in actor.parameters()),
                untrained_upright_steps=round(floor, 2), upright_steps=round(upright, 2),
                curve=json.dumps([round(h, 1) for h in history]), max_steps=float(env.max_steps))


def run_ddpg(kind: str = "hsikan", **kw: object) -> dict[str, float | str]:
    """DDPG preset (single critic, no delay/smoothing)."""
    return run_offpolicy("ddpg", kind, **kw)   # type: ignore[arg-type]


def run_td3(kind: str = "hsikan", **kw: object) -> dict[str, float | str]:
    """TD3 preset (twin critics, delayed policy, target smoothing)."""
    return run_offpolicy("td3", kind, **kw)   # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algo", default="td3", choices=["ddpg", "td3"])
    ap.add_argument("--policy", default="hsikan", choices=list(POLICY_KINDS))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="store the trained policy to this .hymeko (with provenance)")
    a = ap.parse_args(argv)
    cfg = td3_config(total_steps=a.steps, seed=a.seed) if a.algo == "td3" \
        else OffPolicyConfig(total_steps=a.steps, seed=a.seed)
    print(json.dumps(run_offpolicy(a.algo, a.policy, hidden=a.hidden, seed=a.seed, cfg=cfg,
                                   save=a.save), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
