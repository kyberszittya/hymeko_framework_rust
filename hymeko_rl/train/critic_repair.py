"""Critic-only repair training (NO actor updates) — Strategy losses that attack off-policy OOD overestimation.

The guarded sanity sequence localized the RL failure to the CRITIC (it ranks the body-shove exploit above the
DAgger policy). This trains the centralized twin critics ALONE on DAgger-seeded replay under different critic
losses, so the critic ranking benchmark (:mod:`hymeko_rl.eval.critic_benchmark`) can select a repaired critic
BEFORE any actor is trained again. The actor is frozen throughout (targets use the frozen actor); no actor
optimizer exists here.

Variants (Strategy):
  A baseline          — clipped double-Q Bellman (Huber): the failing setup, the control.
  B behavior_support  — + push Q on OOD (random) actions below the data action by a margin.
  C cql               — + CQL conservative penalty ``logsumexp_a Q(s,a) − Q(s,a_data)`` (designed for this).
  E expectile         — asymmetric (expectile) Bellman that penalises overestimation (``Q>y``) more; never
                        queries Q on OOD actions.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.train.ddpg import _polyak


@dataclass
class CriticRepairConfig:
    variant: str = "A"          # A baseline | B behavior_support | C cql | E expectile
    steps: int = 4000
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    huber: bool = True
    max_grad_norm: float = 10.0
    support_lambda: float = 1.0     # B
    support_margin: float = 1.0     # B
    cql_alpha: float = 1.0          # C
    cql_n_samples: int = 10         # C
    expectile: float = 0.3          # E — <0.5 penalises overestimation more (conservative)
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1000


class CriticLoss(ABC):
    """A critic-loss Strategy. ``critics`` are the twins; ``y`` the shared clipped-double-Q Bellman target."""

    @abstractmethod
    def compute(self, critics: list, y: torch.Tensor, s: torch.Tensor, a: torch.Tensor, z: "torch.Tensor | None",
                actor: Any, lo: torch.Tensor, hi: torch.Tensor, cfg: CriticRepairConfig) -> torch.Tensor:
        ...


def _bellman(critics: list, y: torch.Tensor, s, a, z, huber: bool) -> torch.Tensor:
    lf = F.smooth_l1_loss if huber else F.mse_loss
    return sum(lf(c(s, a, z), y) for c in critics)  # type: ignore[return-value]


class BaselineCriticLoss(CriticLoss):
    def compute(self, critics, y, s, a, z, actor, lo, hi, cfg):
        return _bellman(critics, y, s, a, z, cfg.huber)


class BehaviorSupportCriticLoss(CriticLoss):
    """Push Q on random (OOD) actions below the data action by a margin — a cheap support constraint."""

    def compute(self, critics, y, s, a, z, actor, lo, hi, cfg):
        bellman = _bellman(critics, y, s, a, z, cfg.huber)
        rand = lo + (hi - lo) * torch.rand(s.shape[0], a.shape[1], device=s.device)
        q_ood = critics[0](s, rand, z)
        q_data = critics[0](s, a, z)
        penalty = F.relu(q_ood - q_data + cfg.support_margin).mean()
        return bellman + cfg.support_lambda * penalty


class CQLCriticLoss(CriticLoss):
    """CQL: minimise ``logsumexp_a Q(s,a) − Q(s,a_data)`` (push OOD Q down, data Q up) + Bellman."""

    def compute(self, critics, y, s, a, z, actor, lo, hi, cfg):
        bellman = _bellman(critics, y, s, a, z, cfg.huber)
        b, adim, k = s.shape[0], a.shape[1], cfg.cql_n_samples
        rand = lo + (hi - lo) * torch.rand(k, b, adim, device=s.device)
        q_rand = torch.stack([critics[0](s, rand[i], z) for i in range(k)], 0)     # (K, B)
        with torch.no_grad():
            a_pi = actor(s)
        q_pi = critics[0](s, a_pi, z).unsqueeze(0)                                  # (1, B)
        logsumexp = torch.logsumexp(torch.cat([q_rand, q_pi], 0), 0)               # (B,)
        conservative = (logsumexp - critics[0](s, a, z)).mean()
        return bellman + cfg.cql_alpha * conservative


class ExpectileCriticLoss(CriticLoss):
    """Asymmetric (expectile) Bellman: weight overestimation residuals (``y − Q < 0``) more when tau < 0.5."""

    def compute(self, critics, y, s, a, z, actor, lo, hi, cfg):
        loss = s.new_zeros(())
        for c in critics:
            u = y - c(s, a, z)
            w = torch.where(u < 0, 1.0 - cfg.expectile, cfg.expectile)
            loss = loss + (w * u.pow(2)).mean()
        return loss


_LOSSES = {"A": BaselineCriticLoss, "B": BehaviorSupportCriticLoss, "C": CQLCriticLoss, "E": ExpectileCriticLoss}


def build_critic_loss(cfg: CriticRepairConfig) -> CriticLoss:
    if cfg.variant not in _LOSSES:
        raise ValueError(f"unknown critic-repair variant {cfg.variant!r}; expected one of {sorted(_LOSSES)}")
    return _LOSSES[cfg.variant]()


@dataclass
class ResidualTrainConfig:
    steps: int = 3000
    lr: float = 3e-4
    bc_coef: float = 0.5           # anchor the corrected action toward the stored DAgger action
    batch_size: int = 256
    saturation_abort: float = 0.9  # mean |tanh(r_phi)| above this = residual pinned at its bound → abort
    seed: int = 0
    device: str = "cpu"
    log_every: int = 500


def train_residual(residual_actor: Any, frozen_critic: Any, replay: Any, cfg: ResidualTrainConfig, *,
                   gate_fn: Any) -> dict:
    """Train ONLY the residual net to raise the FROZEN critic's Q + a BC anchor, on a fixed (DAgger-seeded) replay.

    The base policy AND the critic are frozen — so there is no Q-scale runaway (the CQL-smoke failure mode) and the
    correction is structurally bounded/gated by :class:`~hymeko_rl.agents.residual_actor.ResidualActor`. ``gate_fn(z)``
    returns the per-sample phase gate. Returns the residual-magnitude / saturation history for the abort check."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    residual_actor.to(dev)
    for p in frozen_critic.parameters():
        p.requires_grad_(False)
    frozen_critic.to(dev).eval()
    opt = torch.optim.Adam(residual_actor.residual.parameters(), lr=cfg.lr)   # ONLY the residual net
    rng = np.random.default_rng(cfg.seed)
    scale = float(residual_actor.action_scale)
    hist = {"residual_norm_normalized": [], "residual_norm_physical": [], "saturation": [], "q": []}
    aborted = None
    for step in range(1, cfg.steps + 1):
        s, a, r, s2, d, z, z2 = (x.to(dev) for x in replay.sample_with_priv(cfg.batch_size, generator=rng))
        gate = gate_fn(z).to(dev)
        a_res = residual_actor.action_mean(s, gate)
        q = frozen_critic(s, a_res, z).mean()
        bc = F.mse_loss(a_res, a)
        loss = -q + cfg.bc_coef * bc
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            gated = (residual_actor.raw_residual(s) * gate.reshape(-1, 1))
            phys = float(gated.norm(dim=1).mean())
            sat = residual_actor.saturation(s)
        hist["residual_norm_physical"].append(round(phys, 5))
        hist["residual_norm_normalized"].append(round(phys / max(scale, 1e-9), 5))
        hist["saturation"].append(round(sat, 4))
        hist["q"].append(round(float(q.detach()), 3))
        if sat > cfg.saturation_abort:
            aborted = f"residual saturated (mean|tanh|={sat:.3f} > {cfg.saturation_abort}) at step {step}"
            break
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"  [residual] step {step}/{cfg.steps} |res|_norm={phys / scale:.4f} sat={sat:.3f} "
                  f"Q={float(q.detach()):+.3g} bc={float(bc.detach()):.3g}", flush=True)
    return {"aborted": aborted, "final_residual_normalized": hist["residual_norm_normalized"][-1] if hist["q"] else 0.0,
            "final_saturation": hist["saturation"][-1] if hist["q"] else 0.0, "history": hist}


def cql_regularizer(action_lo: np.ndarray, action_hi: np.ndarray, *, alpha: float = 1.0, n_samples: int = 6):
    """A ``critic_regularizer(critics, s, a, z, actor)`` closure = the CQL conservative penalty
    ``alpha * (logsumexp_a Q(s,a) − Q(s,a_data))``, to keep OOD suppression active DURING an actor smoke
    (``train_offpolicy(critic_regularizer=…)``). ``actor`` is the moving actor (its action enters the logsumexp)."""
    lo_np, hi_np = np.asarray(action_lo, np.float32), np.asarray(action_hi, np.float32)

    def reg(critics: list, s: torch.Tensor, a: torch.Tensor, z: "torch.Tensor | None", actor: Any) -> torch.Tensor:
        lo = torch.as_tensor(lo_np, device=s.device)
        hi = torch.as_tensor(hi_np, device=s.device)
        b, adim = s.shape[0], a.shape[1]
        rand = lo + (hi - lo) * torch.rand(n_samples, b, adim, device=s.device)
        q_rand = torch.stack([critics[0](s, rand[i], z) for i in range(n_samples)], 0)   # (K, B)
        with torch.no_grad():
            a_pi = actor(s)
        q_pi = critics[0](s, a_pi, z).unsqueeze(0)                                        # (1, B)
        logsumexp = torch.logsumexp(torch.cat([q_rand, q_pi], 0), 0)
        return alpha * (logsumexp - critics[0](s, a, z)).mean()

    return reg


def train_critic_only(critics: list, frozen_actor: Any, replay: Any, cfg: CriticRepairConfig, *,
                      action_lo: np.ndarray, action_hi: np.ndarray) -> list:
    """Train the twin critics alone on ``replay`` (DAgger-seeded) with the configured loss. The actor is FROZEN —
    targets use ``frozen_actor``; no actor optimiser. Clipped double-Q target, Polyak-tracked twin targets."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    for c in critics:
        c.to(dev).train()
    frozen_actor.to(dev).eval()
    for p in frozen_actor.parameters():
        p.requires_grad_(False)
    t_critics = [copy.deepcopy(c) for c in critics]
    opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=cfg.critic_lr)
    loss_fn = build_critic_loss(cfg)
    rng = np.random.default_rng(cfg.seed)
    lo = torch.as_tensor(action_lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(action_hi, dtype=torch.float32, device=dev)
    priv = getattr(replay, "priv_dim", 0) > 0
    for step in range(1, cfg.steps + 1):
        if priv:
            s, a, r, s2, d, z, z2 = (x.to(dev) for x in replay.sample_with_priv(cfg.batch_size, generator=rng))
        else:
            s, a, r, s2, d = (x.to(dev) for x in replay.sample(cfg.batch_size, generator=rng))
            z = z2 = None
        with torch.no_grad():
            a2 = frozen_actor(s2)
            q_next = torch.stack([tc(s2, a2, z2) for tc in t_critics], 0).amin(0)
            y = r + cfg.gamma * (1 - d) * q_next
        loss = loss_fn.compute(critics, y, s, a, z, frozen_actor, lo, hi, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for c in critics for p in c.parameters()], cfg.max_grad_norm)
        opt.step()
        for tc, c in zip(t_critics, critics):
            _polyak(tc, c, cfg.tau)
        if cfg.log_every and step % cfg.log_every == 0:
            with torch.no_grad():
                qm = float(critics[0](s, a, z).mean())
            print(f"  [critic-repair {cfg.variant}] step {step}/{cfg.steps} loss={float(loss.detach()):.3g} "
                  f"Q(data)={qm:+.3g}", flush=True)
    return critics
