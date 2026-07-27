"""R8 S4 — residual-parameterised OPTION environment over the frozen tip-referenced scaffold.

The Bellman action is the actor's bounded residual ``a ∈ [-1,1]^3`` (forward-velocity / squeeze / stop-gain), applied as a
CONSTANT per-option residual over the whole rollout via `ConstantResidualActor` + `ResidualTipAdapter`. One ``step`` = one
full scaffold+residual rollout = one semi-MDP option (``terminal = 1`` — no downstream skill, so ``γ^τ·Q_next`` is zeroed:
the option-consequence return IS the target). The return is the K6-INDEPENDENT `option_return` (injected, never the frozen
certificate), safety-gated by the unchanged motion contract. Everything except ``a`` is provenance; the R6 release
certificate stays the release shield.

Held-out cradles are NEVER placed in this env — it is constructed from development cradles only, so nothing held-out can
enter the replay, the critic, or model selection. Plugs into the framework `train_semi_mdp` unchanged (it is an `OptionEnv`).

    initiation fingerprint  →  actor residual a (Bellman action)  →  ConstantResidualActor + frozen scaffold  →  physics
                              →  R6 certificate  →  K6-independent option return
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    ConstantResidualActor, ResidualBounds, ResidualTipAdapter, ZeroActor)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.option_rl.agents import SemiMDPConfig

OBS_DIM = 11                          # the t=0 adapter fingerprint (residual_adapter.OBS_KEYS)
ACT_DIM = 3                           # the bounded residual (residual_adapter.RESIDUAL_ROLES)
REWARD_SCALE = 5.0                    # algorithmic critic reward-scale (same across SAC/TD3; selection stays on the metric)


def residual_init_obs(snap: Any, params: Any = TipTransportParams(), bounds: Any = ResidualBounds(),
                      cfg: Any = DELIVERY_CFG) -> np.ndarray:
    """The DETERMINISTIC t=0 initiation observation the actor conditions on — the 11-D adapter fingerprint (dtz + initial
    grip forces + squeeze) BEFORE any residual acts, captured from a zero-residual probe. Identical feature layout on dev
    and held-out cradles. # Postconditions: length-`OBS_DIM` float32 vector; independent of the actor."""
    adapter = ResidualTipAdapter(snap, ZeroActor(), params, bounds, cfg)
    velocity_rollout(snap, adapter, cfg)
    return np.asarray(adapter.provenance[0]["obs"], np.float32)


@dataclass
class ResidualOptionEnv:
    """`OptionEnv` over the residual-parameterised scaffold. ``cradles`` = list of ``(tag, snap, init_obs)`` — DEVELOPMENT
    ONLY. ``reward_fn(metrics) -> float`` is the injected K6-independent option return (kept out of this module so the
    reward never imports the certificate). Each ``step`` executes ONE option and is terminal.

    # Preconditions: every cradle is a development cradle; ``reward_fn`` uses only physical quantities.
    # Invariants: the tag set never contains a held-out cradle; the Bellman action returned in ``info`` equals the clipped
      actor emission and drives the physics — no other value is ever presented as the action."""

    cradles: list
    reward_fn: Callable[[dict], float]
    params: Any = field(default_factory=TipTransportParams)
    bounds: Any = field(default_factory=ResidualBounds)
    cfg: Any = DELIVERY_CFG
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._i = 0

    @property
    def tags(self) -> "list[str]":
        return [c[0] for c in self.cradles]

    def reset(self, idx: int | None = None) -> np.ndarray:
        self._i = int(self.rng.integers(0, len(self.cradles))) if idx is None else int(idx)
        return np.asarray(self.cradles[self._i][2], np.float32)

    def step(self, action: np.ndarray, *, search_seed: int | None = None) -> "tuple[np.ndarray, float, bool, dict]":
        a = np.clip(np.asarray(action, np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # the BELLMAN ACTION (clipped tanh emission)
        tag, snap, init_obs = self.cradles[self._i]
        adapter = ResidualTipAdapter(snap, ConstantResidualActor(a), self.params, self.bounds, self.cfg)
        m = velocity_rollout(snap, adapter, self.cfg)
        prov = adapter.provenance
        reward = float(self.reward_fn(m))
        info = {"tau": float(max(1, len(prov))), "terminal": 1.0, "end": "handoff", "tag": tag,
                "k6": float(bool(delivery_success(m, self.cfg))), "reached_handoff": float(m.get("k6_max_dwell", 0) > 0),
                "bellman_action": a.astype(np.float32),
                "executed_residual": np.asarray(prov[-1]["residual"], np.float32) if prov else np.zeros(ACT_DIM, np.float32),
                "dtz_end": float(m["dtz_end"]), "peak_coin_speed": float(m["peak_coin_speed"]),
                "peak_qdot": float(m["peak_qdot"])}
        return np.asarray(init_obs, np.float32), reward, True, info      # terminal option → s_next unused by the γ^τ target


@dataclass
class ResidualRLConfig:
    """Matched semi-MDP hyperparameters for the residual SAC/TD3 (identical across both algorithms — only the algo differs)."""

    gamma: float = 0.99
    tau_polyak: float = 0.01
    lr: float = 3e-4
    batch: int = 64               # small enough that updates begin at option 64 (2-cradle contextual problem)
    warmup_options: int = 40
    total_options: int = 600
    updates_per_option: int = 2   # more critic/actor steps per option (shallow value signal, §S3)
    eval_every: int = 75
    policy_delay: int = 2
    target_noise: float = 0.15
    noise_clip: float = 0.3
    expl_noise: float = 0.2
    alpha: float = 0.1

    def to_semi_mdp(self) -> SemiMDPConfig:
        return SemiMDPConfig(gamma=self.gamma, tau_polyak=self.tau_polyak, lr=self.lr, batch=self.batch,
                             warmup_options=self.warmup_options, total_options=self.total_options,
                             updates_per_option=self.updates_per_option, eval_every=self.eval_every, reward_scale=REWARD_SCALE,
                             policy_delay=self.policy_delay, target_noise=self.target_noise, noise_clip=self.noise_clip,
                             expl_noise=self.expl_noise, alpha=self.alpha)


def distill_zero_residual(actor: Any, init_obs: np.ndarray, *, epochs: int = 200, lr: float = 1e-3, seed: int = 0) -> float:
    """Identical RL init: distil the actor's MEAN residual to 0 on the dev initiation observations, so the update-0
    checkpoint reproduces the frozen SAFE scaffold (the S2 identity) before any gradient. # Postconditions: returns the
    final MSE; leaves the actor's exploratory head untouched (SAC entropy / TD3 noise still explore)."""
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(init_obs, np.float32))
    if x.ndim == 1:
        x = x[None]
    opt = torch.optim.Adam(actor.parameters(), lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        pred = actor.mean_action(x) if hasattr(actor, "mean_action") else actor(x)
        loss = (pred ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())
