"""Behaviour cloning for reaching (Phase 1 — imitation, no reward, no RL loop).

Collect privileged-expert demonstrations (drive to the reaching joint config), fit the
policy's actor mean to the expert action by MSE, and evaluate the *learned* policy's
final end-effector error. Trains either the HSiKAN-on-hypergraph policy or the MLP
baseline on the identical node-feature obs — the first data point of the architecture
ablation (on a serial arm the gap is expected to be small; the quadruped is where the
structural prior should pay off).
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.eval.evaluate import FinalScalarMetric, eval_metric, greedy_action_fn
from hymeko_rl.agents.policy import ActorCritic, build_policy


def collect_demos(env: ArmReachEnv, n_episodes: int, seed: int,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Roll the scripted expert; return ``(obs (M, N, feat), acts (M, n_act))``."""
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert = env.expert_action
        done = False
        while not done:
            obs_list.append(obs)
            act_list.append(expert)
            obs, _, terminated, truncated, _ = env.step(expert)
            done = terminated or truncated
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.float32))


def behaviour_clone(ac: ActorCritic, obs: np.ndarray, acts: np.ndarray, *,
                    n_epochs: int = 150, lr: float = 1e-3, batch: int = 128,
                    seed: int = 0, log_every: int = 25,
                    device: "torch.device | str" = "cpu") -> list[float]:
    """Fit ``ac``'s actor mean to the expert actions (MSE). Returns per-epoch loss.

    ``device``: ``"cpu"`` (default), ``"cuda"``, or ``"auto"`` (cuda if available) — same convention as
    :class:`hymeko_rl.train.ddpg.OffPolicyConfig`. The whole demo corpus is moved to the device once (a BC
    corpus is small — tens of MB); the model is moved for training and returned to the CPU after, so eval
    and ``state_dict`` consumers see the usual CPU module. The per-epoch loss is accumulated **on device**
    and synced once per epoch (a per-batch ``.item()`` would serialise every GPU step).

    Never run blind (§3): emits a flushed ``[bc]`` progress line (epoch, MSE loss, epochs/s, ETA) every
    ``log_every`` epochs so the BC phase is not a dark gap before off-policy training. ``log_every=0`` silences it.
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
    ac.to(dev)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    act_t = torch.as_tensor(acts, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    n = len(obs_t)
    losses: list[float] = []
    t0 = time.perf_counter()
    for ep in range(n_epochs):
        perm = torch.randperm(n, device=dev)
        ep_loss = torch.zeros((), device=dev)
        n_batch = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = ac.action_mean(obs_t[idx])
            loss = F.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()   # type: ignore[no-untyped-call]  # torch stub gap
            opt.step()
            ep_loss += loss.detach()
            n_batch += 1
        losses.append(float(ep_loss.item()) / max(1, n_batch))
        if log_every and ((ep + 1) % log_every == 0 or ep + 1 == n_epochs):
            rate = (ep + 1) / max(1e-9, time.perf_counter() - t0)
            print(f"  [bc] epoch {ep + 1:>4}/{n_epochs} | loss={losses[-1]:.4g} "
                  f"| {rate:4.1f} ep/s | ETA {(n_epochs - ep - 1) / max(1e-9, rate):4.0f}s", flush=True)
    ac.to(torch.device("cpu"))
    return losses


def eval_reach(env: ArmReachEnv, ac: ActorCritic, n_episodes: int, seed: int,
               ) -> tuple[float, float]:
    """Roll the deterministic actor mean; return ``(mean, std)`` final EE error (m). Thin wrapper over the shared
    :func:`hymeko_rl.eval.evaluate.eval_metric` with the ``dist`` final-scalar metric."""
    res = eval_metric(env, greedy_action_fn(ac), FinalScalarMetric("dist"), n_episodes=n_episodes, seed0=seed)
    return float(np.mean(res)), float(np.std(res))


def _make_policy(kind: str, env: ArmReachEnv, hidden: int) -> ActorCritic:
    # Size against the env's actual observation width, so a non-default obs profile
    # (more/fewer channels) is handled without touching this code.
    feat = env.obs_spec.obs_dim
    if kind == "hsikan":
        return build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions,
                            hg_state=env.hg, hidden=hidden)
    return build_policy("mlp", obs_dim=env.hg.n_vertices * feat,
                        action_dim=env.n_actions, hidden=hidden)


def run_bc(policy_kind: str = "hsikan", *, n_demos: int = 48, n_epochs: int = 150,
           hidden: int = 64, seed: int = 0, n_eval: int = 24,
           env_factory: Callable[[], ArmReachEnv] | None = None,
           ) -> dict[str, float | str]:
    """Build env + policy, collect demos, behaviour-clone, and evaluate reaching.

    ``env_factory`` selects the scene (default: the 4-DOF ``arm_world`` arm); pass a factory
    that builds a hymeko-emitted arm for the canonical comparison. Fully reproducible: the
    torch seed gates the policy init, ``behaviour_clone`` the training, and the env RNG the
    demo/eval targets — both backbones see identical targets at a given seed.

    Returns the trained reach error vs the untrained floor (BC works iff trained < floor).
    """
    make_env = env_factory if env_factory is not None else ArmReachEnv
    torch.manual_seed(seed)   # reproducible policy initialisation (before _make_policy)
    env = make_env()
    ac = _make_policy(policy_kind, env, hidden)
    floor_mean, _ = eval_reach(make_env(), ac, n_eval, seed=10_000)   # untrained floor
    obs, acts = collect_demos(env, n_demos, seed)
    losses = behaviour_clone(ac, obs, acts, n_epochs=n_epochs, seed=seed)
    reach_mean, reach_std = eval_reach(make_env(), ac, n_eval, seed=20_000)
    return dict(
        policy=policy_kind, n_demos=n_demos, n_samples=int(len(obs)),
        n_params=ac.n_parameters(), final_bc_loss=round(losses[-1], 5),
        reach_err_m=round(reach_mean, 4), reach_std=round(reach_std, 4),
        untrained_floor_m=round(floor_mean, 4),
    )
