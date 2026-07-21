"""Train the HSiKAN actor-critic (or the MLP baseline) to balance the HyMeKo cart-pole.

    python -m hymeko_rl.train.train_inverted_pendulum --policy hsikan
    python -m hymeko_rl.train.train_inverted_pendulum --policy mlp      # the ablation baseline

The cart-pole structure (the 2-vertex kinematic hypergraph the HSiKAN backbone reads) comes from
``data/robotics/inverted_pendulum.hymeko``; the balancing reward/termination is the canonical
InvertedPendulum rule in :class:`InvertedPendulumEnv`. Same shared PPO loop and same backbone swap as the
reaching task — only the env (the task) changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.eval.evaluate import StepCountMetric, eval_metric, greedy_action_fn
from hymeko_rl.agents.policy import POLICY_KINDS, ActorCritic, build_policy
from hymeko_rl.train.ppo import PPOConfig, train_ppo


class GreedyPolicy(Protocol):
    """The minimal contract :func:`eval_balance` needs — a deterministic mean action. Both the PPO
    ``ActorCritic`` and the DDPG ``DeterministicActor`` satisfy it."""

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor: ...


def _make_balance_policy(kind: str, env: InvertedPendulumEnv, hidden: int) -> ActorCritic:
    """Build the actor-critic for ``env``: HSiKAN reads the per-vertex obs over the cart-pole
    hypergraph; the MLP baseline reads the same obs flattened. Same dispatch shape as
    :func:`hymeko_rl.train.bc._make_policy`, sized from the gym spaces so it is env-contract-agnostic.

    # Preconditions ``kind in POLICY_KINDS``; ``env.observation_space`` is ``(n_vertices, feat)``.
    """
    # The env's spaces are Boxes with definite shapes (invariant of InvertedPendulumEnv.__init__),
    # so `.shape` is never None here; assert it to satisfy the gym `shape: tuple | None` annotation.
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    assert obs_shape is not None and act_shape is not None
    n_vertices, feat = int(obs_shape[0]), int(obs_shape[1])
    action_dim = int(act_shape[0])
    if kind == "mlp":
        return build_policy("mlp", obs_dim=n_vertices * feat, action_dim=action_dim, hidden=hidden)
    # hsikan (fixed incidence) and signedkan (learned incidence) both read the per-vertex hypergraph obs.
    return build_policy(kind, obs_dim=feat, action_dim=action_dim, hg_state=env.hg, hidden=hidden)


def eval_balance(env: InvertedPendulumEnv, ac: GreedyPolicy, n_episodes: int, *, seed: int) -> float:
    """Mean upright-steps over ``n_episodes`` greedy (deterministic-mean) rollouts. Thin wrapper over the shared
    :func:`hymeko_rl.eval.evaluate.eval_metric` with the step-count metric.

    # Postconditions returns a value in ``[0, env.max_steps]``; higher = a better balancer."""
    res = eval_metric(env, greedy_action_fn(ac), StepCountMetric(), n_episodes=n_episodes, seed0=seed)
    return float(np.mean(res))


def run_balance(policy_kind: str = "hsikan", *, hidden: int = 64, seed: int = 0, n_eval: int = 20,
                n_envs: int = 16, cfg: PPOConfig | None = None,
                save: str | Path | None = None) -> dict[str, float | str]:
    """Build env + policy, train with the shared PPO loop, evaluate the balancer.

    ``n_envs > 1`` uses the vectorized rollout (batched policy forward over ``n_envs`` worker envs) — the
    throughput win for the dispatch-bound 2-vertex forward. ``n_envs == 1`` is the single-env path.

    # Preconditions ``policy_kind in POLICY_KINDS``; ``n_envs >= 1``.
    # Postconditions returns the run summary (init/final return, mean upright-steps, param count)."""
    if policy_kind not in POLICY_KINDS:
        raise ValueError(f"unknown policy kind {policy_kind!r}; expected one of {POLICY_KINDS}")
    cfg = cfg or PPOConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Emit the cart-pole scene ONCE and share it across all envs (eval + the N rollout workers), so a
    # vectorized run does not pay N+ `hymeko` CLI subprocess calls at construction.
    base_mjcf = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=base_mjcf)
    ac = _make_balance_policy(policy_kind, env, hidden)
    floor = eval_balance(InvertedPendulumEnv(mjcf=base_mjcf), ac, n_eval, seed=10_000)
    history = train_ppo(ac, env, cfg, n_envs=n_envs,
                        make_env=(lambda: InvertedPendulumEnv(mjcf=base_mjcf)) if n_envs > 1 else None)
    upright = eval_balance(InvertedPendulumEnv(mjcf=base_mjcf), ac, n_eval, seed=20_000)
    if save is not None:
        from hymeko_rl.agents.policy_store import policy_to_hymeko
        policy_to_hymeko(ac.state_dict(), save, meta={
            "algo": "ppo", "backbone": policy_kind, "upright": round(upright, 1), "seed": seed})
    return dict(
        policy=policy_kind, n_params=ac.n_parameters(), n_envs=n_envs,
        init_return=round(history[0], 3), final_return=round(history[-1], 3),
        untrained_upright_steps=round(floor, 2), upright_steps=round(upright, 2),
        max_steps=float(env.max_steps), saved=str(save) if save else "",
    )


def main(argv: list[str] | None = None) -> int:
    # Single-threaded torch at the binary boundary (§6.5 #11 "set once at entry" exception): the rollout
    # is batch-1 forwards on a 2-vertex graph, where the thread-pool dispatch overhead exceeds the work —
    # measured ~10% faster per iter at 1 thread vs 8 (335 vs 300 steps/s). Library callers keep the default.
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", default="hsikan", choices=list(POLICY_KINDS))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--envs", type=int, default=16, help="parallel rollout envs (1 = single-env)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="store the trained policy to this .hymeko (with provenance)")
    a = ap.parse_args(argv)
    cfg = PPOConfig(n_iters=a.iters, n_steps=a.steps, seed=a.seed)
    print(json.dumps(run_balance(a.policy, hidden=a.hidden, seed=a.seed, n_envs=a.envs, cfg=cfg,
                                 save=a.save), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
