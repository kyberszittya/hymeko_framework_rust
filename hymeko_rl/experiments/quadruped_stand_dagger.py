"""Aibo standing via DAgger declared as a signed dataflow HYPERGRAPH (``quadruped_stand_dagger.hymeko``).

The strategy is READ from the graph — :func:`hymeko_rl.train.strategy_graph.run_dagger_graph` classifies it as
``dagger`` from the relabel cycle and dispatches to the shared :class:`Dagger` executor; the MDP (plant + reward)
is the sibling ``quadruped_stand.hymeko`` (env ``from_hymeko``). DAgger is the motivated lever past the BC ceiling
(BC best 0.79, seeds 1--2 at 0.29/0.46; TD3+BC collapses to 0.0). The PD-hold expert is **stateless**, so the
lock-step ``label_sanity`` gate is trivially latch-coherent.

    python -m hymeko_rl.experiments.quadruped_stand_dagger --smoke     # 1 seed, 1 round, local path-check
    python -m hymeko_rl.experiments.quadruped_stand_dagger             # the declared 3-seed x iterate-4 DAgger
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, greedy_action_fn
from hymeko_rl.experiments.quadruped_stand_train import _ARCH, _demos, _make_env
from hymeko_rl.train.ddpg import build_offpolicy
from hymeko_rl.train.strategy_graph import run_dagger_graph

_GRAPH = "data/robotics/quadruped_stand_dagger.hymeko"
_N_EVAL = 24


def _build_actor(env: Any) -> Any:
    """The imitation policy (SA-HSiKAN actor only — DAgger has NO critic, so the value drift that collapsed
    TD3+BC cannot recur). Reuses ``build_offpolicy`` and discards the (single, unused) critic."""
    actor, _critics = build_offpolicy(_ARCH, obs_dim=2, flat_dim=env.hg.n_vertices * 2,
                                      action_dim=env.n_actions, action_scale=1.0, n_critics=1,
                                      hg_state=env.hg, critic_layernorm=True)
    return actor


def _measure(env: Any, actor: Any) -> "dict[str, float]":
    """Standing dwell rate on the given eval env (Dagger's ``(env, actor)`` measure signature)."""
    vals = eval_metric(env, greedy_action_fn(actor), DwellMetric("standing", 200),
                       n_episodes=_N_EVAL, seed0=7000)
    return {"standing": float(np.mean(vals))}


def _expert(env: Any) -> np.ndarray:
    return np.asarray(env.expert_action, dtype=np.float32)   # the stateless PD-hold-q0 standing demonstrator


def run(*, smoke: bool = False, seeds: "tuple[int, ...] | None" = None,
        base: "str | Any" = "experiments") -> "dict[str, Any]":
    """Read the DAgger dataflow graph and run it (topology → dagger; the MDP comes from quadruped_stand.hymeko)."""
    probe = _make_env()
    sp = probe.action_space
    assert isinstance(sp, Box)
    lo, hi = np.asarray(sp.low, dtype=np.float64), np.asarray(sp.high, dtype=np.float64)
    return run_dagger_graph(
        _GRAPH, make_env=_make_env, build=_build_actor, demos=_demos, measure=_measure, expert_fn=_expert,
        name="quadruped_stand_dagger", select="standing", smoke=smoke, seeds_override=seeds,
        gif=not smoke, action_bounds=(lo, hi), base=base)


if __name__ == "__main__":
    import sys

    summary = run(smoke=("--smoke" in sys.argv))
    print(f"\n=== Aibo standing (DAgger-as-hypergraph) ===\n  standing median={summary.get('standing_median')} "
          f"per-seed={summary.get('standing_per_seed')}")
