"""Task-independent semi-MDP SAC / TD3 over a proposal-center action.

The critic is Q(s, action) where ``action`` is the proposal center and the fixed local-search wrapper is part of the
environment's stochastic response — so the actor is trained through the critic only, never through the (non-differentiable)
search, and never against the search-selected candidate. Targets use `smdp_target` (γ^τ). Networks are dimension-parameterised;
the loop takes an `OptionEnv` and a `dev_eval_fn(actor) -> (score, aux)` callback, so no task specifics leak in.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hymeko_rl.option_rl.core import OptionReplayBuffer, OptionTransition, smdp_target


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class DetActor(nn.Module):                                               # TD3 (deterministic, tanh-bounded)
    def __init__(self, obs_dim: int, act_dim: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, act_dim), nn.Tanh())

    def forward(self, s):
        return self.net(s)

    def mean_action(self, s):
        return self.net(s)


class GaussActor(nn.Module):                                            # SAC (squashed Gaussian)
    LOG_STD = (-5.0, 2.0)

    def __init__(self, obs_dim: int, act_dim: int, h: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, act_dim); self.log_std = nn.Linear(h, act_dim)

    def forward(self, s):
        x = self.body(s); return self.mu(x), self.log_std(x).clamp(*self.LOG_STD)

    def sample(self, s):
        mu, log_std = self(s); std = log_std.exp()
        n = torch.randn_like(std); pre = mu + std * n; a = torch.tanh(pre)
        logp = (-0.5 * (n ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp

    def mean_action(self, s):
        return torch.tanh(self(s)[0])


@dataclass
class SemiMDPConfig:
    gamma: float = 0.99
    tau_polyak: float = 0.01
    lr: float = 3e-4
    batch: int = 128
    warmup_options: int = 40
    total_options: int = 600
    updates_per_option: int = 1
    eval_every: int = 120
    reward_scale: float = 5.0     # algorithmic critic reward-scale (same across branches; selection stays on the task metric)
    policy_delay: int = 2         # TD3
    target_noise: float = 0.15
    noise_clip: float = 0.3
    expl_noise: float = 0.2       # TD3 exploration
    alpha: float = 0.1            # SAC entropy temperature (fixed)


def _polyak(tgt, src, tau):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(tau * sp.data)


def make_actor(algo: str, obs_dim: int, act_dim: int):
    return GaussActor(obs_dim, act_dim) if algo == "sac" else DetActor(obs_dim, act_dim)


def train_semi_mdp(algo, env, actor, dev_eval_fn, cfg, *, obs_dim, act_dim, log=print, seed=0):
    """Generic semi-MDP SAC/TD3 over the proposal action. ``env`` is an `OptionEnv`; ``dev_eval_fn(actor) -> (score, aux)``
    is the task's held-out selection metric. Returns (checkpoints{update0,early,mid,best_val,final}, history)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q1, q2 = QNet(obs_dim, act_dim), QNet(obs_dim, act_dim)
    q1t, q2t = QNet(obs_dim, act_dim), QNet(obs_dim, act_dim); q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    qopt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), cfg.lr)
    at = DetActor(obs_dim, act_dim) if algo == "td3" else None
    if at is not None:
        at.load_state_dict(actor.state_dict())
    aopt = torch.optim.Adam(actor.parameters(), cfg.lr)
    replay = OptionReplayBuffer(); history = []; ckpts = {"update0": copy.deepcopy(actor.state_dict())}
    best_val = -1e9; ckpts["best_val"] = copy.deepcopy(actor.state_dict()); upd = 0

    def act(o, explore):
        with torch.no_grad():
            ot = torch.as_tensor(o[None]).float()
            if algo == "sac":
                a = (actor.sample(ot)[0] if explore else actor.mean_action(ot))[0].numpy()
            else:
                a = actor.mean_action(ot)[0].numpy()
                if explore:
                    a = np.clip(a + rng.normal(0, cfg.expl_noise, act_dim).astype(np.float32), -1, 1)
        return a

    s = env.reset(); scorerun = []
    for it in range(cfg.total_options):
        a = act(s, explore=True) if it >= cfg.warmup_options else np.clip(rng.uniform(-1, 1, act_dim).astype(np.float32), -1, 1)
        s2, r, done, info = env.step(a)
        replay.add(OptionTransition(s=s, action=np.asarray(a, np.float32), reward=r / cfg.reward_scale, tau=float(info["tau"]),
                                    s_next=s2, terminal=float(info.get("terminal", done)), end=info.get("end", "handoff"),
                                    provenance={k: info.get(k) for k in ("theta_center", "theta_selected", "k6", "reached_handoff", "tau", "contain_exit_ct")}))
        scorerun_v = info.get("k6", 0.0); scorerun.append(scorerun_v); s = env.reset() if done else s2
        if len(replay) >= cfg.batch and it >= cfg.warmup_options:
            for _ in range(cfg.updates_per_option):
                upd += 1
                bs, ba, br, bt, bs2, bd = replay.sample(cfg.batch, rng)
                with torch.no_grad():
                    if algo == "sac":
                        a2, logp2 = actor.sample(bs2); q_next = torch.min(q1t(bs2, a2), q2t(bs2, a2)) - cfg.alpha * logp2
                    else:
                        noise = (torch.randn_like(ba) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                        a2 = (at(bs2) + noise).clamp(-1, 1); q_next = torch.min(q1t(bs2, a2), q2t(bs2, a2))
                    y = smdp_target(br, cfg.gamma, bt, bd, q_next)
                ql = ((q1(bs, ba) - y) ** 2).mean() + ((q2(bs, ba) - y) ** 2).mean()
                qopt.zero_grad(); ql.backward(); qopt.step()
                if algo == "sac":
                    ap, logp = actor.sample(bs); al = (cfg.alpha * logp - torch.min(q1(bs, ap), q2(bs, ap))).mean()
                    aopt.zero_grad(); al.backward(); aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak); _polyak(q2t, q2, cfg.tau_polyak)
                elif upd % cfg.policy_delay == 0:
                    al = -q1(bs, actor(bs)).mean(); aopt.zero_grad(); al.backward(); aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak); _polyak(q2t, q2, cfg.tau_polyak); _polyak(at, actor, cfg.tau_polyak)
        if (it + 1) % cfg.eval_every == 0 or it == cfg.total_options - 1:
            score, aux = dev_eval_fn(actor)
            with torch.no_grad():
                qm = float(q1(*replay.sample(min(64, len(replay)), rng)[:2]).mean()) if len(replay) >= 8 else 0.0
            recent = float(np.mean(scorerun[-cfg.eval_every:])) if scorerun else 0.0
            history.append({"it": it + 1, "dev_score": score, "aux": aux, "train_recent": round(recent, 3), "q_mean": round(qm, 2)})
            log(f"    [{algo} it {it+1}/{cfg.total_options}] dev {score} {aux} | train {round(recent,3)} | Q~{round(qm,2)} | replay {len(replay)}")
            if it + 1 >= cfg.total_options // 4 and "early" not in ckpts:
                ckpts["early"] = copy.deepcopy(actor.state_dict())
            if it + 1 >= cfg.total_options // 2 and "mid" not in ckpts:
                ckpts["mid"] = copy.deepcopy(actor.state_dict())
            if score > best_val:
                best_val = score; ckpts["best_val"] = copy.deepcopy(actor.state_dict())
    ckpts["final"] = copy.deepcopy(actor.state_dict())
    ckpts.setdefault("early", ckpts["update0"]); ckpts.setdefault("mid", ckpts.get("best_val"))
    return ckpts, history
