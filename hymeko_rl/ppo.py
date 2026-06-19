"""Minimal in-repo PPO (clipped surrogate + GAE) — Phase 2.

One on-policy loop trains *either* the HSiKAN-on-hypergraph policy or the MLP baseline on
the same ``ArmReachEnv`` reward — fix the algorithm, ablate the architecture. On-policy
data also closes the behaviour-cloning covariate-shift gap from Phase 1. The standard
entropy bonus (``ent_coef``) is the seat where the **algebraic entropy feedback** signal
(structural entropy of the state hypergraph) will later replace/augment ``H(π)``.

No heavy RL dependency: ~1 file over the pinned torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.bc import _make_policy, eval_reach
from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.policy import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters (shared by both policy arms — the ablation fixes these)."""

    n_iters: int = 40
    n_steps: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 8
    minibatch: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    value_warmup: int = 0   # critic-only warm-up iterations before joint PPO
    seed: int = 0           # seeds the env RNG (reaching targets) for reproducible runs


def _gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, last_val: float,
         gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation. Returns ``(advantages, returns)``."""
    t = len(rews)
    adv = np.zeros(t, dtype=np.float32)
    last = 0.0
    for i in reversed(range(t)):
        next_val = last_val if i == t - 1 else vals[i + 1]
        nonterm = 1.0 - dones[i]
        delta = rews[i] + gamma * next_val * nonterm - vals[i]
        last = delta + gamma * lam * nonterm * last
        adv[i] = last
    return adv, adv + vals


def _mc_returns(rews: np.ndarray, dones: np.ndarray, last_val: float, gamma: float,
                ) -> np.ndarray:
    """Discounted Monte-Carlo return-to-go (no critic bootstrap) — for the critic warm-up,
    where the critic is still cold and a GAE bootstrap would be unreliable."""
    t = len(rews)
    ret = np.zeros(t, dtype=np.float32)
    running = last_val
    for i in reversed(range(t)):
        running = rews[i] + gamma * running * (1.0 - dones[i])
        ret[i] = running
    return ret


def _warmup_critic(ac: ActorCritic, env: ArmReachEnv, obs: np.ndarray, cfg: PPOConfig,
                   ) -> np.ndarray:
    """Fit the (separate) critic network to MC returns under the current policy before joint
    PPO, so advantages are sane from the first update. The critic backbone is independent of
    the actor, so the whole critic network trains freely here without touching the actor.
    Returns the next obs."""
    params = list(ac.critic_backbone.parameters()) + list(ac.critic.parameters())
    crit_opt = torch.optim.Adam(params, lr=1e-3)
    cur = obs
    for _ in range(cfg.value_warmup):
        buf, cur, last_val, _ = _collect(env, ac, cur, cfg.n_steps, cfg.gamma)
        ret = torch.as_tensor(_mc_returns(buf["rew"], buf["done"], last_val, cfg.gamma))
        for _ in range(cfg.update_epochs):
            loss = F.mse_loss(ac.value(buf["obs"]), ret)
            crit_opt.zero_grad()
            loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            crit_opt.step()
    return cur


@torch.no_grad()
def _collect(env: ArmReachEnv, ac: ActorCritic, obs: np.ndarray, n_steps: int,
             gamma: float) -> tuple[dict[str, Any], np.ndarray, float, float]:
    """Roll the current policy for ``n_steps``; return buffers, next obs, bootstrap value,
    and the mean episodic return over completed episodes in this rollout.

    Time-limit truncation is handled correctly: when an episode is *truncated* (hit
    ``max_steps``) rather than *terminated* (reached the goal), the cut-off future value
    ``γ·V(next_obs)`` is folded into that step's reward, so GAE / the MC warm-up do not
    treat the time limit as a true terminal (the bug that made PPO degrade good policies).
    """
    o_buf, a_buf, lp_buf, r_buf, v_buf, d_buf = [], [], [], [], [], []
    ep_ret, ep_rets, cur = 0.0, [], obs
    for _ in range(n_steps):
        ot = torch.as_tensor(cur[None], dtype=torch.float32)
        action, logp, value = ac.act(ot)
        nobs, rew, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
        buf_rew = float(rew)
        if truncated and not terminated:
            v_next = float(ac.act(torch.as_tensor(nobs[None], dtype=torch.float32))[2].item())
            buf_rew += gamma * v_next   # bootstrap the time-limit cut-off
        o_buf.append(cur)
        a_buf.append(action.squeeze(0))
        lp_buf.append(logp.squeeze(0))
        r_buf.append(np.float32(buf_rew))
        v_buf.append(value.squeeze(0))
        d_buf.append(np.float32(terminated or truncated))
        ep_ret += float(rew)   # the TRUE reward, for logging
        cur = nobs
        if terminated or truncated:
            ep_rets.append(ep_ret)
            ep_ret = 0.0
            cur, _ = env.reset()
    last_val = float(ac.act(torch.as_tensor(cur[None], dtype=torch.float32))[2].item())
    buf = {
        "obs": torch.as_tensor(np.asarray(o_buf), dtype=torch.float32),
        "act": torch.stack(a_buf),
        "logp": torch.stack(lp_buf),
        "rew": np.asarray(r_buf, dtype=np.float32),
        "val": torch.stack(v_buf).numpy().astype(np.float32),
        "done": np.asarray(d_buf, dtype=np.float32),
    }
    mean_ret = float(np.mean(ep_rets)) if ep_rets else ep_ret
    return buf, cur, last_val, mean_ret


def _update(ac: ActorCritic, opt: torch.optim.Optimizer, buf: dict[str, Any],
            adv: np.ndarray, ret: np.ndarray, cfg: PPOConfig) -> None:
    obs, act, old_logp = buf["obs"], buf["act"], buf["logp"]
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8))
    ret_t = torch.as_tensor(ret)
    n = len(obs)
    idx = np.arange(n)
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for s in range(0, n, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logp, ent, val = ac.evaluate(obs[mb], act[mb])
            ratio = (logp - old_logp[mb]).exp()
            surr = torch.min(ratio * adv_t[mb],
                             torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[mb])
            loss = (-surr.mean() + cfg.vf_coef * F.mse_loss(val, ret_t[mb])
                    - cfg.ent_coef * ent.mean())
            opt.zero_grad()
            loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()


def train_ppo(ac: ActorCritic, env: ArmReachEnv, cfg: PPOConfig) -> list[float]:
    """Train ``ac`` on ``env`` with PPO. Returns the per-iteration mean episodic return."""
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    # Seed the env RNG so the reaching targets (hence returns) are reproducible; the
    # subsequent in-loop resets advance the same generator deterministically. Without this,
    # `env.reset()` drew targets from system entropy → run-to-run-flaky returns (§3).
    obs, _ = env.reset(seed=cfg.seed)
    if cfg.value_warmup > 0:
        obs = _warmup_critic(ac, env, obs, cfg)
    history: list[float] = []
    for _ in range(cfg.n_iters):
        buf, obs, last_val, mean_ret = _collect(env, ac, obs, cfg.n_steps, cfg.gamma)
        adv, ret = _gae(buf["rew"], buf["val"], buf["done"], last_val, cfg.gamma, cfg.lam)
        _update(ac, opt, buf, adv, ret, cfg)
        history.append(mean_ret)
    return history


def run_ppo(policy_kind: str = "hsikan", *, control_mode: str = "torque",
            hidden: int = 64, seed: int = 0, n_eval: int = 24,
            cfg: PPOConfig | None = None,
            pretrain_demos: int = 0, pretrain_epochs: int = 120,
            ) -> dict[str, float | str]:
    """Build env + policy, optionally behaviour-clone-warm-start, train with PPO, evaluate.

    ``control_mode`` selects the actuator interface (torque / position / velocity) — the
    ablation axis. ``pretrain_demos > 0`` clones the closed-loop expert before PPO
    (imitation → RL): the standard recipe that reaches well *and* closes the BC
    covariate-shift gap. With ``pretrain_demos == 0`` it is PPO from scratch.
    """
    cfg = cfg or PPOConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ArmReachEnv(control_mode=control_mode)
    ac = _make_policy(policy_kind, env, hidden)
    floor, _ = eval_reach(ArmReachEnv(control_mode=control_mode), ac, n_eval, seed=10_000)
    pre_reach = float("nan")
    if pretrain_demos > 0:
        from hymeko_rl.bc import behaviour_clone, collect_demos
        obs, acts = collect_demos(env, pretrain_demos, seed)
        behaviour_clone(ac, obs, acts, n_epochs=pretrain_epochs, seed=seed)
        pre_reach, _ = eval_reach(
            ArmReachEnv(control_mode=control_mode), ac, n_eval, seed=15_000)
    history = train_ppo(ac, env, cfg)
    reach_mean, reach_std = eval_reach(
        ArmReachEnv(control_mode=control_mode), ac, n_eval, seed=20_000)
    return dict(
        policy=policy_kind, control_mode=control_mode,
        n_params=ac.n_parameters(), warm_start=pretrain_demos > 0,
        init_return=round(history[0], 3), final_return=round(history[-1], 3),
        reach_err_m=round(reach_mean, 4), reach_std=round(reach_std, 4),
        pretrain_reach_m=round(pre_reach, 4), untrained_floor_m=round(floor, 4),
    )
