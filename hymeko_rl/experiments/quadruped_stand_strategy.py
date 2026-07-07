"""Aibo standing, with the algorithm chosen BY GRAPH TOPOLOGY.

Reads a strategy dataflow graph, classifies it (``bc`` / ``dagger`` / ``td3_bc`` — acyclic / relabel-cycle /
critic-node), and dispatches to the matching executor. One entry, three algorithms, told apart by their dataflow
shape — the capstone of "the training strategy is a hypergraph". The MDP is always ``quadruped_stand.hymeko``.

    python -m hymeko_rl.experiments.quadruped_stand_strategy --graph data/robotics/quadruped_stand_dagger.hymeko
    python -m hymeko_rl.experiments.quadruped_stand_strategy --graph data/robotics/quadruped_stand_td3bc_graph.hymeko --smoke
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from hymeko_rl.experiments.quadruped_stand_dagger import _build_actor, _expert
from hymeko_rl.experiments.quadruped_stand_dagger import _measure as _measure_dagger
from hymeko_rl.experiments.quadruped_stand_train import _build, _demos, _make_env
from hymeko_rl.experiments.quadruped_stand_train import _measure as _measure_campaign
from hymeko_rl.train.strategy_graph import StrategyGraph, run_dagger_graph, run_td3bc_graph


def run(graph: str, *, smoke: bool = False, seeds: "tuple[int, ...] | None" = None,
        base: "str | Any" = "experiments") -> "dict[str, Any]":
    """Classify the graph by topology and run the matching executor (the MDP is quadruped_stand.hymeko)."""
    algo = StrategyGraph.from_hymeko(graph).classify()
    name = f"quadruped_stand_{algo}"
    print(f"[strategy] {graph} -> classified '{algo}' from topology; dispatching.", flush=True)
    if algo == "dagger":
        probe = _make_env()
        sp = probe.action_space
        assert isinstance(sp, Box)
        lo, hi = np.asarray(sp.low, dtype=np.float64), np.asarray(sp.high, dtype=np.float64)
        return run_dagger_graph(graph, make_env=_make_env, build=_build_actor, demos=_demos,
                                measure=_measure_dagger, expert_fn=_expert, name=name, select="standing",
                                smoke=smoke, seeds_override=seeds, gif=not smoke, action_bounds=(lo, hi), base=base)
    if algo == "td3_bc":
        return run_td3bc_graph(graph, make_env=_make_env, build=_build, demos=_demos, measure=_measure_campaign,
                               name=name, select="standing", smoke=smoke, seeds_override=seeds, base=base)
    raise ValueError(f"{graph}: classified {algo!r}; only 'dagger'/'td3_bc' executors are wired in this entry")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default="data/robotics/quadruped_stand_dagger.hymeko")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    summary = run(a.graph, smoke=a.smoke)
    print(f"\n=== {a.graph}: standing median={summary.get('standing_median')} "
          f"per-seed={summary.get('standing_per_seed')} ===")
