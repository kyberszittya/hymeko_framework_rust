"""Flat-obs PPO for the Stage-B reward fine-tune — on-policy, value-baselined, clipped.

The multi-seed REINFORCE pass was optimizer-variance-dominated (success not robust). PPO is the low-variance,
covariate-shift-correcting fix: a value critic + GAE advantages + a clipped surrogate keep the fine-tune near the
BC skill and let the reward shape it with far less variance. No flat-obs trainer exists in the repo (the trainers
need 2-D hypergraph obs), so this is new, not a duplicate — same justification as the REINFORCE smoke. The actor is
the harness ``_GaussianMLP``; PPO optimizes the pre-tanh Gaussian and applies ``tanh`` as a fixed squash.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


class _RunningNorm:
    """Streaming (Chan's parallel Welford) obs mean/std — the from-scratch substitute for BC-fit normalization."""

    def __init__(self, dim: int) -> None:
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        b_n = len(batch)
        if b_n == 0:
            return
        b_mean = batch.mean(0)
        b_m2 = batch.var(0) * b_n
        if self.n == 0:
            self.mean, self.m2, self.n = b_mean.copy(), b_m2.copy(), b_n
            return
        delta = b_mean - self.mean
        tot = self.n + b_n
        self.mean += delta * b_n / tot
        self.m2 += b_m2 + delta ** 2 * self.n * b_n / tot
        self.n = tot

    def std(self) -> np.ndarray:
        return np.sqrt(self.m2 / max(1, self.n)) if self.n else np.ones_like(self.mean)


class _ValueMLP:
    """A small obs-standardized value critic V(s) for GAE — mirrors the actor's input normalization."""

    def __init__(self, obs_dim: int, hidden: int, seed: int) -> None:
        import torch
        torch.manual_seed(seed + 7)
        self._torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        self.obs_mean = torch.zeros(obs_dim)
        self.obs_std = torch.ones(obs_dim)

    def set_obs_norm(self, mean: Any, std: Any) -> None:
        self.obs_mean = mean
        self.obs_std = std

    def parameters(self) -> Any:
        return self.net.parameters()

    def value(self, obs: Any) -> Any:
        t = self._torch
        x = obs if t.is_tensor(obs) else t.as_tensor(np.asarray(obs, np.float32))
        return self.net((x - self.obs_mean) / self.obs_std).squeeze(-1)


def _collect_rollout(cfg: Any, env: Any, actor: Any, critic: Any, seed: int) -> "dict[str, Any]":
    """Collect ``ppo_rollout_steps`` on-policy transitions (reset on episode end); bootstrap the tail with V(s_T)."""
    import torch
    obs_l: list[np.ndarray] = []
    raw_l: list[Any] = []
    logp_l: list[float] = []
    rew_l: list[float] = []
    val_l: list[float] = []
    done_l: list[float] = []
    ep_returns: list[float] = []
    ep_ret = 0.0
    ep_len = 0
    obs, _ = env.reset(seed=seed)
    for _ in range(cfg.ppo_rollout_steps):
        raw, act, logp = actor.sample_raw(obs)
        obs_l.append(np.asarray(obs, np.float32))
        raw_l.append(raw)
        logp_l.append(logp)
        val_l.append(float(critic.value(obs).detach()))
        obs, r, term, trunc, _info = env.step(act)
        rew_l.append(float(r))
        ep_ret += float(r)
        ep_len += 1
        done = bool(term or trunc) or ep_len >= cfg.max_steps
        done_l.append(1.0 if done else 0.0)
        if done:
            ep_returns.append(ep_ret)
            ep_ret = ep_len = 0
            obs, _ = env.reset(seed=seed + len(ep_returns) + 1_000)
    last_val = float(critic.value(obs).detach())
    return {"obs": np.asarray(obs_l, np.float32), "raw": torch.stack(raw_l),
            "logp": np.asarray(logp_l, np.float32), "rew": np.asarray(rew_l, np.float32),
            "val": np.asarray(val_l, np.float32), "done": np.asarray(done_l, np.float32),
            "last_val": last_val, "ep_returns": ep_returns}


def _gae(buf: "dict[str, Any]", gamma: float, lam: float) -> "tuple[np.ndarray, np.ndarray]":
    """Generalized advantage estimation over the rollout buffer → (advantages, returns)."""
    rew, val, done = buf["rew"], buf["val"], buf["done"]
    n = len(rew)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_val = buf["last_val"] if t == n - 1 else val[t + 1]
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + val


def _ppo_update(cfg: Any, actor: Any, critic: Any, opt: Any, buf: "dict[str, Any]", adv: np.ndarray,
                ret: np.ndarray, seed: int) -> None:
    """K epochs of clipped-surrogate + value-MSE minibatch updates over the rollout."""
    import torch
    obs_t = torch.as_tensor(buf["obs"])
    raw_t = buf["raw"]
    old_logp = torch.as_tensor(buf["logp"])
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    n = len(adv)
    for _ in range(cfg.ppo_epochs):
        idx = rng.permutation(n)
        for s in range(0, n, cfg.ppo_minibatch):
            b = idx[s:s + cfg.ppo_minibatch]
            new_logp, ent = actor.log_prob_raw(obs_t[b], raw_t[b])
            ratio = (new_logp - old_logp[b]).exp()
            surr = torch.min(ratio * adv_t[b],
                             torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv_t[b])
            val_loss = ((critic.value(obs_t[b]) - ret_t[b]) ** 2).mean()
            loss = -surr.mean() + cfg.ppo_value_coef * val_loss - cfg.ppo_entropy_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_ppo_flat(cfg: Any, env: Any, init_state: "dict[str, Any] | None", seed: int, *, log: Any) -> "dict[str, Any]":
    """Warm-started (BC) flat-obs PPO on the reward-override env. Live-logged. Returns the same schema as REINFORCE."""
    import torch

    from .exp_metaworld_reward_stageb import _GaussianMLP
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    scale = float(np.max(np.abs(np.asarray(env.action_space.high, np.float64))))
    log_std_init = (float(np.log(cfg.explore_std)) if init_state is not None
                    else float(np.log(cfg.ppo_from_scratch_std)))     # from-scratch: tunable initial exploration
    actor = _GaussianMLP(obs_dim, act_dim, cfg.hidden, scale, seed=seed, log_std_init=log_std_init)
    if init_state is not None:
        actor.load_state_dict(init_state)
    critic = _ValueMLP(obs_dim, cfg.hidden, seed)
    critic.set_obs_norm(actor.obs_mean, actor.obs_std)
    # from scratch (no BC): estimate obs normalization online (no demos to fit it); warm-started keeps the BC norm.
    running = _RunningNorm(obs_dim) if init_state is None else None
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=cfg.ppo_lr)
    history: list[float] = []
    steps = it = 0
    t0 = time.time()
    while steps < cfg.total_env_steps and (time.time() - t0) < cfg.wall_time_cap_s:
        buf = _collect_rollout(cfg, env, actor, critic, seed + it)
        if running is not None:
            running.update(buf["obs"])
            norm = (torch.as_tensor(running.mean, dtype=torch.float32),
                    torch.as_tensor(np.maximum(running.std(), 1e-3), dtype=torch.float32))
            actor.obs_mean, actor.obs_std = norm
            critic.set_obs_norm(*norm)
        steps += len(buf["rew"])
        adv, ret = _gae(buf, cfg.gamma, cfg.ppo_gae_lambda)
        _ppo_update(cfg, actor, critic, opt, buf, adv, ret, seed + it)
        it += 1
        mean_ret = float(np.mean(buf["ep_returns"])) if buf["ep_returns"] else float("nan")
        history.append(mean_ret)
        log(f"iter {it} step {steps}/{cfg.total_env_steps} ep_ret {mean_ret:.1f} "
            f"eps {len(buf['ep_returns'])} {steps / max(1e-6, time.time() - t0):.0f} st/s")
    return {"policy": actor, "returns": history, "env_steps": steps, "episodes": it,
            "wall_s": round(time.time() - t0, 1)}
