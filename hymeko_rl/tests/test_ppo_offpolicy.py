"""Stage 2: PPO joins the off-policy architecture-comparison harness (one comparison surface for all algos)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.offpolicy_eval import _ALGOS, Budget, compare_offpolicy
from hymeko_rl.ppo import build_ppo


def test_ppo_registered_in_algos() -> None:
    """PPO is a first-class member of the algorithm Strategy registry alongside the off-policy trio."""
    assert "ppo" in _ALGOS and set(_ALGOS) >= {"ppo", "sac", "ddpg", "td3"}


def test_build_ppo_both_backbones() -> None:
    """The adapter builds a PPO ActorCritic for the flat MLP and the structural HSiKAN, returning no critics."""
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    space = env.observation_space.shape
    assert space is not None
    nv, feat = int(space[0]), int(space[1])
    ad = int(np.prod(env.action_space.shape))
    mlp, crit_m = build_ppo("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0, hidden=32)
    hsk, crit_h = build_ppo("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=ad, action_scale=1.0,
                            hidden=32, hg_state=env.hg)
    assert crit_m == [] and crit_h == []
    assert mlp.n_parameters() > 0 and hsk.n_parameters() > 0


def test_ppo_runs_through_compare_offpolicy() -> None:
    """The full driver path runs PPO and yields a finite curve-max — the Stage-2 wiring works end-to-end.

    Tiny budget on the 2-vertex control floor: this is a *smoke* of the adapter, not an architecture verdict
    (those are real-topology only — standing rule)."""
    budget = Budget(total_steps=2048, eval_every=1024, n_eval=2,
                    hidden={"mlp": 32, "hsikan": 32, "signedkan": 32})
    out = compare_offpolicy("cartpole", algos=["ppo"], backbones=["mlp"], seeds=[0], budget=budget)
    stats = out["ppo/mlp"]
    assert np.isfinite(stats["curve_max_median"]) and stats["n_params"] > 0
