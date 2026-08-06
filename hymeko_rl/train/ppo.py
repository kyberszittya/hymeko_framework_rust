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
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.train.bc import _make_policy, eval_reach
from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.agents.policy import ActorCritic, build_policy


class RolloutEnv(Protocol):
    """The minimal env contract the PPO loop depends on (Strategy / structural typing).

    Both :class:`~hymeko_rl.env.arm_reach_env.ArmReachEnv` and
    :class:`~hymeko_rl.env.inverted_pendulum_env.InvertedPendulumEnv` satisfy it, so the *same*
    ``train_ppo`` trains either — fix the algorithm, vary the task/architecture. ``difficulty`` is read
    opportunistically via ``getattr`` (for the curriculum), so it is not part of this contract."""

    def reset(self, *, seed: int | None = ...,
              ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...


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
    bc_coef: float = 0.0    # BC-anchor weight: keeps PPO near the demos (TD3+BC's lesson) so a warm-started
    #                         policy does not drift off the precise grasp/place skill the rollout rarely reinforces


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


def _warmup_critic(ac: ActorCritic, env: RolloutEnv, obs: np.ndarray, cfg: PPOConfig,
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
def _collect(env: RolloutEnv, ac: ActorCritic, obs: np.ndarray, n_steps: int,
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
            adv: np.ndarray, ret: np.ndarray, cfg: PPOConfig,
            bc_data: "tuple[np.ndarray, np.ndarray] | None" = None) -> dict[str, float]:
    """One PPO update over the rollout; returns the mean diagnostics across minibatches
    (policy/value loss, entropy, approx-KL, clip fraction) for tracing the training curves.

    When ``cfg.bc_coef > 0`` and ``bc_data`` (demo obs, acts) is given, a behaviour-cloning anchor
    ``bc_coef * MSE(action_mean(demo_obs), demo_act)`` is added to each minibatch loss (TD3+BC-style)."""
    obs, act, old_logp = buf["obs"], buf["act"], buf["logp"]
    adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8))
    ret_t = torch.as_tensor(ret)
    anchor = cfg.bc_coef > 0.0 and bc_data is not None
    bc_obs = torch.as_tensor(bc_data[0], dtype=torch.float32) if (anchor and bc_data is not None) else None
    bc_act = torch.as_tensor(bc_data[1], dtype=torch.float32) if (anchor and bc_data is not None) else None
    n = len(obs)
    idx = np.arange(n)
    pl = vl = en = kl = cf = 0.0
    nb = 0
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for s in range(0, n, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logp, ent, val = ac.evaluate(obs[mb], act[mb])
            ratio = (logp - old_logp[mb]).exp()
            surr = torch.min(ratio * adv_t[mb],
                             torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[mb])
            ploss = -surr.mean()
            vloss = F.mse_loss(val, ret_t[mb])
            entm = ent.mean()
            loss = ploss + cfg.vf_coef * vloss - cfg.ent_coef * entm
            if anchor and bc_obs is not None and bc_act is not None:    # keep the policy near the demos
                bi = np.random.randint(0, len(bc_obs), size=min(cfg.minibatch, len(bc_obs)))
                loss = loss + cfg.bc_coef * F.mse_loss(ac.action_mean(bc_obs[bi]), bc_act[bi])
            opt.zero_grad()
            loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                pl += float(ploss.item())
                vl += float(vloss.item())
                en += float(entm.item())
                kl += float((old_logp[mb] - logp).mean().item())        # approx KL(old||new)
                cf += float(((ratio - 1.0).abs() > cfg.clip).float().mean().item())
            nb += 1
    inv = 1.0 / max(1, nb)
    return {"policy_loss": pl * inv, "value_loss": vl * inv, "entropy": en * inv,
            "approx_kl": kl * inv, "clip_frac": cf * inv}


@torch.no_grad()
def _collect_vec(envs: "list[RolloutEnv]", ac: ActorCritic, obs: np.ndarray, cfg: PPOConfig,
                 ) -> tuple[dict[str, Any], np.ndarray, float]:
    """Batched rollout over ``N = len(envs)`` lock-step envs — the dispatch-amortising sibling of
    :func:`_collect`. Measured motivation: the batch-1 ``ac.act`` is ~87% of a single-env iteration, so
    one *batched* forward over ``N`` obs per tick (``T = ⌈n_steps/N⌉`` ticks) cuts the dominant cost
    ~``N``×. Returns ``(buf, next_obs (N,…), mean_ret)`` with flattened ``(T·N,…)`` ``obs/act/logp`` plus
    **per-env** ``adv/ret`` (each env is an independent trajectory; GAE is computed per column).

    The time-limit truncation fix is preserved **per env**: a truncation (not termination) folds
    ``γ·V(next_obs)`` into that env's reward and marks it done, so GAE does not treat the time limit as a
    true terminal (the bug that made PPO degrade good policies).

    # Preconditions ``obs`` is ``(N, *obs_shape)``; every env shares one obs/action shape; ``N >= 1``.
    """
    n = len(envs)
    n_steps, gamma, lam = cfg.n_steps, cfg.gamma, cfg.lam
    ticks = max(1, -(-n_steps // n))                 # ceil(n_steps / N); total transitions = ticks·N
    obs_shape = obs.shape[1:]
    a_dim = int(ac.log_std.numel())          # backbone-agnostic action dim (works for CTDE/multichannel too)
    o_buf = np.zeros((ticks, n, *obs_shape), dtype=np.float32)
    a_buf = torch.zeros((ticks, n, a_dim))
    lp_buf = torch.zeros((ticks, n))
    v_buf = np.zeros((ticks, n), dtype=np.float32)
    r_buf = np.zeros((ticks, n), dtype=np.float32)
    d_buf = np.zeros((ticks, n), dtype=np.float32)
    ep_ret = np.zeros(n, dtype=np.float64)
    ep_rets: list[float] = []
    cur = obs.astype(np.float32, copy=True)
    for t in range(ticks):
        actions, logps, values = ac.act(torch.as_tensor(cur, dtype=torch.float32))  # batched (N,·)
        acts_np = actions.numpy()
        next_obs = np.zeros_like(cur)
        rews = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.float32)
        boot: dict[int, np.ndarray] = {}             # env idx -> terminal obs needing a γV bootstrap
        for i, env in enumerate(envs):
            nobs, rew, terminated, truncated, _ = env.step(acts_np[i])
            ep_ret[i] += float(rew)
            done = bool(terminated or truncated)
            if truncated and not terminated:
                boot[i] = np.asarray(nobs, dtype=np.float32)   # bootstrap before the reset clobbers it
            if done:
                ep_rets.append(float(ep_ret[i]))
                ep_ret[i] = 0.0
                nobs, _ = env.reset()
            next_obs[i] = nobs
            rews[i] = float(rew)
            dones[i] = 1.0 if done else 0.0
        if boot:                                     # one batched value forward for all truncated envs
            idxs = sorted(boot)
            vnext = ac.value(torch.as_tensor(np.stack([boot[i] for i in idxs]), dtype=torch.float32)).numpy()
            for k, i in enumerate(idxs):
                rews[i] += gamma * float(vnext[k])
        o_buf[t] = cur
        a_buf[t] = actions
        lp_buf[t] = logps
        v_buf[t] = values.numpy()
        r_buf[t] = rews
        d_buf[t] = dones
        cur = next_obs
    last_vals = ac.value(torch.as_tensor(cur, dtype=torch.float32)).numpy()   # per-env GAE tail bootstrap
    adv = np.zeros((ticks, n), dtype=np.float32)
    ret = np.zeros((ticks, n), dtype=np.float32)
    for i in range(n):                               # GAE PER ENV COLUMN (independent trajectories)
        adv[:, i], ret[:, i] = _gae(r_buf[:, i], v_buf[:, i], d_buf[:, i], float(last_vals[i]), gamma, lam)
    buf = {
        "obs": torch.as_tensor(o_buf.reshape(ticks * n, *obs_shape), dtype=torch.float32),
        "act": a_buf.reshape(ticks * n, a_dim),
        "logp": lp_buf.reshape(ticks * n),
        "val": v_buf.reshape(ticks * n),
        "adv": adv.reshape(ticks * n),
        "ret": ret.reshape(ticks * n),
    }
    mean_ret = float(np.mean(ep_rets)) if ep_rets else float(np.mean(ep_ret))
    return buf, cur, mean_ret


def train_ppo(ac: ActorCritic, env: RolloutEnv, cfg: PPOConfig,
              *, on_iteration: Callable[[int, int], None] | None = None,
              metrics_out: list[dict[str, float]] | None = None,
              n_envs: int = 1, make_env: "Callable[[], RolloutEnv] | None" = None,
              bc_data: "tuple[np.ndarray, np.ndarray] | None" = None) -> list[float]:
    """Train ``ac`` on ``env`` with PPO. Returns the per-iteration mean episodic return.

    ``on_iteration(i, n_iters)`` (Observer) is called before each of the ``n_iters`` iterations —
    the seat for a curriculum that mutates ``env`` (e.g. its start-state difficulty) as training
    progresses. It is called exactly ``cfg.n_iters`` times with ``i`` in ``0..n_iters-1``.

    ``n_envs > 1`` switches to the **vectorized** rollout (:func:`_collect_vec`): ``make_env`` must build a
    fresh env, ``n_envs`` of them are run lock-step and the policy forward is batched (the throughput win;
    see the perf report). ``n_envs == 1`` (the default) keeps the exact single-env path — reach training is
    unchanged. The vectorized path does not thread ``on_iteration`` mutations into the worker envs
    (curriculum is single-env only) and skips the critic warm-up.

    If ``metrics_out`` is given, one diagnostics dict per iteration is appended (return, losses,
    entropy, approx-KL, clip fraction, action std, curriculum difficulty) for tracing the curves.

    # Preconditions ``n_envs >= 1``; if ``n_envs > 1`` then ``make_env`` is given.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1; got {n_envs}")
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    vec = n_envs > 1
    envs: list[RolloutEnv] = []
    if vec:
        if make_env is None:
            raise ValueError("n_envs > 1 requires make_env to build the worker envs")
        envs = [make_env() for _ in range(n_envs)]
        # Distinct per-env seeds → independent, reproducible trajectory streams.
        obs = np.stack([envs[i].reset(seed=cfg.seed + i)[0] for i in range(n_envs)]).astype(np.float32)
    else:
        # Seed the env RNG so the targets (hence returns) are reproducible; in-loop resets advance the
        # same generator deterministically. Without this, `env.reset()` drew from system entropy (§3).
        obs, _ = env.reset(seed=cfg.seed)
        if cfg.value_warmup > 0:
            obs = _warmup_critic(ac, env, obs, cfg)
    history: list[float] = []
    for i in range(cfg.n_iters):
        if on_iteration is not None:
            on_iteration(i, cfg.n_iters)
        if vec:
            buf, obs, mean_ret = _collect_vec(envs, ac, obs, cfg)
            adv, ret = buf["adv"], buf["ret"]
        else:
            buf, obs, last_val, mean_ret = _collect(env, ac, obs, cfg.n_steps, cfg.gamma)
            adv, ret = _gae(buf["rew"], buf["val"], buf["done"], last_val, cfg.gamma, cfg.lam)
        upd = _update(ac, opt, buf, adv, ret, cfg, bc_data=bc_data)
        history.append(mean_ret)
        if metrics_out is not None:
            metrics_out.append({
                "iter": float(i), "return": float(mean_ret),
                "action_std": float(ac.log_std.exp().mean().item()),
                "difficulty": float(getattr(env, "difficulty", 1.0)),
                "mean_adv": float(adv.mean()), **upd})
    return history


def run_ppo(policy_kind: str = "hsikan", *, control_mode: str = "torque",
            hidden: int = 64, seed: int = 0, n_eval: int = 24,
            cfg: PPOConfig | None = None,
            pretrain_demos: int = 0, pretrain_epochs: int = 120, n_envs: int = 8,
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
        from hymeko_rl.train.bc import behaviour_clone, collect_demos
        obs, acts = collect_demos(env, pretrain_demos, seed)
        behaviour_clone(ac, obs, acts, n_epochs=pretrain_epochs, seed=seed)
        pre_reach, _ = eval_reach(
            ArmReachEnv(control_mode=control_mode), ac, n_eval, seed=15_000)
    # Vectorized rollout is the measured-faster default (no curriculum here); n_envs=1 = single-env path.
    history = train_ppo(ac, env, cfg, n_envs=n_envs,
                        make_env=(lambda: ArmReachEnv(control_mode=control_mode)) if n_envs > 1 else None)
    reach_mean, reach_std = eval_reach(
        ArmReachEnv(control_mode=control_mode), ac, n_eval, seed=20_000)
    return dict(
        policy=policy_kind, control_mode=control_mode,
        n_params=ac.n_parameters(), warm_start=pretrain_demos > 0,
        init_return=round(history[0], 3), final_return=round(history[-1], 3),
        reach_err_m=round(reach_mean, 4), reach_std=round(reach_std, 4),
        pretrain_reach_m=round(pre_reach, 4), untrained_floor_m=round(floor, 4),
    )


# ----------------------------------------------------------------------------------------------------
# Off-policy-harness adapter — lets PPO join offpolicy_eval._ALGOS so the HSiKAN/MLP architecture ablation
# runs across {PPO, DDPG, TD3, SAC} on ONE comparison surface (CLAUDE.md §6.5 #1: extend the harness, not a
# parallel on-policy runner). PPO is iteration-based; the adapter maps the step budget to iterations and scores
# the SAME greedy curve as the off-policy members (via the shared eval_fn), so curve-max is comparable. NOTE the
# params-match (offpolicy_eval.match_mlp_hidden) is computed from the SAC actor; PPO's ActorCritic also carries a
# critic backbone, so absolute counts differ but the HSiKAN-vs-MLP ratio is approximately preserved.
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PPOEvalConfig:
    """PPO budget in the off-policy harness's vocabulary (steps), plus the fields the driver reads.

    The step budget maps to ``n_iters = total_steps // n_steps`` PPO iterations with an eval cadence of
    ``max(1, eval_every // n_steps)`` iterations. ``n_critics`` is unused by PPO (its critic lives inside the
    ActorCritic) and only satisfies the shared ``build`` call.

    # Invariants ``total_steps >= n_steps >= 1``; ``eval_every >= 1``; ``n_eval >= 1``."""

    total_steps: int
    eval_every: int
    n_eval: int
    seed: int = 0
    n_critics: int = 1      # ignored by PPO; the harness passes it to the shared build signature
    n_steps: int = 1024     # PPO rollout length per iteration


def build_ppo(kind: str, *, obs_dim: int, flat_dim: int, action_dim: int, action_scale: float,
              n_critics: int = 1, hidden: int = 64, **kw: Any) -> tuple[ActorCritic, list[Any]]:
    """Build a PPO ActorCritic conforming to the off-policy ``build`` signature (empty critics list).

    Mirrors :func:`hymeko_rl.train.bc._make_policy`: ``mlp`` reads the flat obs (``flat_dim``); ``hsikan``/
    ``signedkan`` read the per-vertex feature dim (``obs_dim``) plus ``hg_state`` (in ``kw``). ``action_scale``
    and ``n_critics`` are accepted for signature parity and unused (PPO is an unbounded Gaussian; the env clips).

    # Preconditions ``kind in {'mlp','hsikan','signedkan'}``; non-mlp requires ``hg_state`` in ``kw``."""
    if kind == "mlp":
        actor = build_policy("mlp", obs_dim=flat_dim, action_dim=action_dim, hidden=hidden)
    else:
        actor = build_policy(kind, obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, **kw)
    return actor, []


def train_ppo_eval(actor: ActorCritic, _critics: list[Any], env: RolloutEnv, cfg: PPOEvalConfig,
                   *, eval_fn: Callable[[Any, Any], float] | None = None) -> list[float]:
    """Train PPO; return the GREEDY-eval curve (one point per ``eval_every``), comparable to the off-policy
    members' curves so curve-max stats are apples-to-apples.

    The eval reuses the training env — the same shared-env eval the off-policy trainers already perform
    (``sac.py``/``ddpg.py``); the single post-eval transition is stale, which is negligible (one transition per
    rollout, as in the off-policy path). # Postconditions a non-empty list of finite greedy returns."""
    n_iters = max(1, cfg.total_steps // cfg.n_steps)
    eval_every_iters = max(1, cfg.eval_every // cfg.n_steps)
    ppo_cfg = PPOConfig(n_iters=n_iters, n_steps=cfg.n_steps, seed=cfg.seed)
    curve: list[float] = []

    def _on_iter(i: int, _n: int) -> None:
        if eval_fn is not None and i % eval_every_iters == 0:
            curve.append(float(eval_fn(env, actor)))

    train_ppo(actor, env, ppo_cfg, on_iteration=_on_iter)
    if eval_fn is not None:
        curve.append(float(eval_fn(env, actor)))   # final greedy point
    return curve or [0.0]
