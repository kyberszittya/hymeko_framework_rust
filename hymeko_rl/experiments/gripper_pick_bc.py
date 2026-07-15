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

from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.env.gripper_pick_env import GripperPickEnv
from hymeko_rl.eval.evaluate import LiftPlaceMetric, eval_metric, greedy_action_fn
from hymeko_rl.agents.policy import ActorCritic, build_policy


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
    return build_policy(kind, obs_dim=feat, action_dim=env.n_actions,   # hsikan / signedkan / sa_hsikan / mixture
                        hg_state=env.hg, hidden=hidden)


class PickPolicyIncompatible(RuntimeError):
    """A pick-place checkpoint does not match the env it is being loaded into (obs schema / action dim / policy
    family / observe_target). Raised instead of silently mis-loading a state_dict or falling back to an untrained
    model. Assimilates the 2026-07-14 provenance findings (metadata-blind FF ckpts, obs-72 vs obs-90, 4 loaders)."""


def _env_obs_dim(env: PickEnv) -> int:
    return env.hg.n_vertices * int(env.observation_space.shape[-1])


def load_pick_actor(path: str, env: PickEnv, *, kind: str = "hsikan", hidden: int = 64) -> ActorCritic:
    """Load a raw **feedforward** ActorCritic checkpoint with FAIL-LOUD compatibility validation. A shape mismatch
    (e.g. the June obs-72 graph loaded into the current obs-90 config) raises :class:`PickPolicyIncompatible` rather
    than mis-loading. # Preconditions ``path`` is a raw FF ``state_dict`` (not a recurrent-metadata blob)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and blob.get("kind") == "recurrent_clone":
        raise PickPolicyIncompatible(f"{path}: recurrent checkpoint — use load_pick_policy(), not load_pick_actor()")
    ac = build(kind, env, hidden)
    try:
        ac.load_state_dict(blob)
    except (RuntimeError, KeyError) as err:                             # shape/key mismatch = wrong obs/action schema
        raise PickPolicyIncompatible(
            f"{path}: feedforward checkpoint incompatible with env (kind={kind}, obs_dim={_env_obs_dim(env)}, "
            f"n_actions={env.n_actions}): {err}") from err
    return ac.eval()


def load_pick_policy(path: str, env: PickEnv, *, kind: str = "hsikan"):
    """THE canonical pick-place checkpoint loader → a deterministic ``action_fn(env, obs) -> action``. Dispatches a
    recurrent-metadata blob (``kind=="recurrent_clone"``) vs a raw feedforward ``state_dict``, validating obs schema /
    action dim / ``observe_target`` against ``env`` and FAILING LOUD on mismatch (never a silent mis-load or untrained
    fallback). Used by the GUI, the renderer, and the residual base — the single loader replacing 4 ad-hoc ones."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and blob.get("kind") == "recurrent_clone":
        from hymeko_rl.experiments.pick_place_recurrent_clone import RecurrentActionFn, RecurrentPickActor
        want, got = int(blob["obs_dim"]), _env_obs_dim(env)
        if want != got:
            raise PickPolicyIncompatible(
                f"{path}: recurrent obs_dim {want} != env {got} — observe_target mismatch "
                f"(env observe_target={getattr(env, '_observe_target', None)}, ckpt={blob.get('observe_target')})")
        if int(blob["action_dim"]) != env.n_actions:
            raise PickPolicyIncompatible(f"{path}: recurrent action_dim {blob['action_dim']} != env {env.n_actions}")
        actor = RecurrentPickActor(blob["obs_dim"], blob["action_dim"], blob["hidden"], cell=blob["cell"])
        actor.load_state_dict(blob["state_dict"])
        return RecurrentActionFn(actor)
    if isinstance(blob, dict) and blob.get("kind") == "residual_clone":          # base + bounded RL residual
        from hymeko_rl.env.residual_pick_env import ResidualActionAdapter
        base = load_pick_actor(blob["base_ckpt"], env, kind=blob.get("base_kind", "hsikan"))
        residual = build(blob.get("residual_kind", "hsikan"), env, int(blob.get("hidden", 64)))
        try:
            residual.load_state_dict(blob["residual_state_dict"])
        except (RuntimeError, KeyError) as err:
            raise PickPolicyIncompatible(f"{path}: residual checkpoint incompatible with env: {err}") from err
        residual.eval()
        lo, hi = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        adapter = ResidualActionAdapter(base, residual, delta_scale=blob["delta_scale"],
                                        grip_scale=blob["grip_scale"], action_dim=env.n_actions, lo=lo, hi=hi)
        return greedy_action_fn(adapter)
    return greedy_action_fn(load_pick_actor(path, env, kind=kind))


_DIVERGE_QACC = 5_000.0   # |qacc| above this = the sim has blown up (normal contact is ~1e2); see below.


def eval_success(env: PickEnv, ac: ActorCritic, n_episodes: int, seed: int, *,
                 lift_thresh: float = 0.035) -> tuple[float, float]:
    """Roll the deterministic actor mean; return ``(lift_rate, place_rate)`` over ``n_episodes``.

    Thin wrapper over the shared :func:`hymeko_rl.eval.evaluate.eval_metric` with :class:`~hymeko_rl.eval.evaluate.
    LiftPlaceMetric` — the divergence guard (a blow-up is a FAILURE, never a counted lift; measured 2026-06-30,
    where it inflated a true 0.125 lift to 0.875) now lives in that reusable metric, not in a duplicated loop."""
    res = eval_metric(env, greedy_action_fn(ac),
                      LiftPlaceMetric(lift_thresh=lift_thresh, diverge_qacc=_DIVERGE_QACC),
                      n_episodes=n_episodes, seed0=seed)
    lifts = sum(lift for lift, _ in res)
    places = sum(place for _, place in res)
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
