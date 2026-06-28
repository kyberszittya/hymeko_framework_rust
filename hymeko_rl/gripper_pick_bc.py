"""Behaviour cloning for the floating-gripper pick-and-place (Phase 1, path B).

Prove the expert -> BC -> learned-grasp loop on the easy-control floating gripper before porting to the arm:
collect demos from the direct-Cartesian expert, clone the policy's actor mean (reusing
``bc.behaviour_clone``), and report the learned policy's grasp/lift/place success vs the untrained floor.
HSiKAN-on-hypergraph or the MLP baseline on the identical obs — the same backbone-swap ablation.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.env.gripper_pick_env import GripperPickEnv
from hymeko_rl.policy import ActorCritic, build_policy


class PickEnv(Protocol):
    """The contract the BC helpers need — shared by the floating gripper and the FANUC arm pick envs."""
    observation_space: Any
    hg: Any
    n_actions: int

    def reset(self, *, seed: int | None = ..., options: Any = ...) -> tuple[np.ndarray, dict[str, Any]]: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...
    @property
    def expert_action(self) -> np.ndarray: ...


def collect_demos(env: PickEnv, n_episodes: int, seed: int, *,
                  only_success: bool = False, lift_thresh: float = 0.035) -> tuple[np.ndarray, np.ndarray]:
    """Roll the scripted expert; return ``(obs (M, N, feat), acts (M, n_actions))``. With ``only_success``,
    keep an episode's transitions only if it lifted/placed (filters out the expert's occasional misses → the
    clone learns from clean demos)."""
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        done, lifted, placed = False, False, False
        while not done:
            a = env.expert_action
            ep_obs.append(obs)
            ep_act.append(a)
            obs, _, term, trunc, info = env.step(a)
            lifted = lifted or bool(info.get("lifted", 0.0) > lift_thresh)
            placed = placed or bool(info.get("reached", False))
            done = term or trunc
        if (not only_success) or lifted or placed:
            obs_all.extend(ep_obs)
            act_all.extend(ep_act)
    return np.asarray(obs_all, dtype=np.float32), np.asarray(act_all, dtype=np.float32)


def build(kind: str, env: PickEnv, hidden: int) -> ActorCritic:
    feat = int(env.observation_space.shape[-1])
    if kind == "mlp":
        return build_policy("mlp", obs_dim=env.hg.n_vertices * feat,
                            action_dim=env.n_actions, hidden=hidden)
    return build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions,
                        hg_state=env.hg, hidden=hidden)


@torch.no_grad()
def eval_success(env: PickEnv, ac: ActorCritic, n_episodes: int, seed: int, *,
                 lift_thresh: float = 0.035) -> tuple[float, float]:
    """Roll the deterministic actor mean; return ``(lift_rate, place_rate)`` over ``n_episodes``."""
    lifts = places = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, lifted, placed = False, False, False
        while not done:
            a = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
            obs, _, term, trunc, info = env.step(a)
            lifted = lifted or bool(info["lifted"] > lift_thresh)
            placed = placed or bool(info["reached"])
            done = term or trunc
        lifts += int(lifted)
        places += int(placed)
    return lifts / n_episodes, places / n_episodes


def run_gripper_bc(kind: str = "hsikan", *, n_demos: int = 40, n_epochs: int = 120,
                   hidden: int = 64, seed: int = 0, n_eval: int = 12) -> dict[str, float | str]:
    """Build env + policy, clone the expert, evaluate grasp success vs the untrained floor."""
    torch.manual_seed(seed)
    env = GripperPickEnv()
    ac = build(kind, env, hidden)
    floor_lift, _ = eval_success(GripperPickEnv(), ac, n_eval, seed=10_000)
    obs, acts = collect_demos(env, n_demos, seed)
    behaviour_clone(ac, obs, acts, n_epochs=n_epochs, seed=seed)
    lift, place = eval_success(GripperPickEnv(), ac, n_eval, seed=20_000)
    return dict(policy=kind, n_samples=int(len(obs)), n_params=ac.n_parameters(),
                lift_rate=round(lift, 3), place_rate=round(place, 3), untrained_lift=round(floor_lift, 3))


def main() -> int:
    import json
    print(json.dumps(run_gripper_bc(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
