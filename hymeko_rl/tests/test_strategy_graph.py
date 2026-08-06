"""The training STRATEGY as a signed dataflow hypergraph: parse, classify-by-topology, and reject-by-topology.

Covers the core claim that the algorithm's identity is its graph TOPOLOGY, not a label: the DAgger graph
(``quadruped_stand_dagger.hymeko``) has a relabel cycle → ``dagger``; the BC graph (``quadruped_stand_bc_graph.hymeko``)
is acyclic → ``bc``; removing the loop edge flips the classification and the DAgger runner rejects it (load-bearing).
"""
from __future__ import annotations

import pytest

from hymeko_rl.train.strategy_graph import (
    Flow,
    Stage,
    StrategyGraph,
    run_dagger_graph,
    run_td3bc_graph,
)

_DAGGER = "data/robotics/quadruped_stand_dagger.hymeko"
_BC = "data/robotics/quadruped_stand_bc_graph.hymeko"
_TD3BC = "data/robotics/quadruped_stand_td3bc_graph.hymeko"


def test_from_hymeko_parses_stages_flows_and_iterate() -> None:
    g = StrategyGraph.from_hymeko(_DAGGER)
    assert set(s.role for s in g.stages.values()) == {"source", "bc", "rollout", "label", "aggregate", "eval"}
    assert len(g.stages) == 7 and len(g.flows) == 6
    assert g.iterate == 4 and g.seeds == (0, 1, 2)
    roll = next(s for s in g.stages.values() if s.role == "rollout")
    assert roll.knobs["rollouts"] == pytest.approx(12.0) and roll.knobs["beta"] == pytest.approx(0.5)


def test_loop_flow_is_the_relabel_cycle() -> None:
    """The f_loop hyperedge is (+ retrain, - rollout, - evaluate) — the producer→consumer that closes the cycle."""
    g = StrategyGraph.from_hymeko(_DAGGER)
    loop = next(f for f in g.flows if "retrain" in f.producers and "rollout" in f.consumers)
    assert "rollout" in loop.consumers and "evaluate" in loop.consumers


def test_classify_dagger_from_topology() -> None:
    assert StrategyGraph.from_hymeko(_DAGGER).classify() == "dagger"


def test_classify_bc_from_acyclic_topology() -> None:
    assert StrategyGraph.from_hymeko(_BC).classify() == "bc"


def test_classify_is_topology_not_label() -> None:
    """Constructed graphs: a back edge makes it dagger; without it, bc — identity is the cycle, nothing else."""
    stages = {n: Stage(n, r, {}) for n, r in
              [("s", "source"), ("b", "bc"), ("r", "rollout"), ("a", "aggregate"), ("t", "bc"), ("e", "eval")]}
    acyclic = (Flow("f1", ("s",), ("b",)), Flow("f2", ("b",), ("e",)))
    cyclic = (*acyclic, Flow("f3", ("b",), ("r",)), Flow("f4", ("r",), ("a",)),
              Flow("f5", ("a",), ("t",)), Flow("floop", ("t",), ("r",)))   # t -> r closes the loop
    assert StrategyGraph(stages, acyclic, 0, (0,), "x").classify() == "bc"
    assert StrategyGraph(stages, cyclic, 4, (0,), "x").classify() == "dagger"


def test_to_dagger_config_reads_declared_knobs() -> None:
    cfg = StrategyGraph.from_hymeko(_DAGGER).to_dagger_config(name="q", select="standing")
    assert cfg.n_demos == 40 and cfg.bc_epochs == 200 and cfg.dagger_iters == 4   # n_demos: small D0 (relabels bite)
    assert cfg.rollouts_per_iter == 12 and cfg.beta == pytest.approx(0.5)
    assert cfg.expert_replay_ratio == pytest.approx(1.0) and cfg.n_eval == 24 and cfg.seeds == (0, 1, 2)


def test_runner_rejects_bc_graph_topology() -> None:
    """The DAgger runner rejects a graph whose relabel loop is absent (topology is load-bearing, not decorative)."""
    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("must not reach the executor for a bc-classified graph")

    with pytest.raises(ValueError, match="not 'dagger'"):
        run_dagger_graph(_BC, make_env=_boom, build=_boom, demos=_boom, measure=_boom, expert_fn=_boom,
                         name="q", select="standing")


def test_classify_td3bc_from_critic_node() -> None:
    """A ``critic`` node (with Q-target edges) makes the graph td3_bc — the third topology in the family."""
    assert StrategyGraph.from_hymeko(_TD3BC).classify() == "td3_bc"


def test_all_three_algorithms_distinguished_by_topology() -> None:
    """bc / dagger / td3_bc are told apart purely by dataflow shape (acyclic / relabel-cycle / critic-node)."""
    assert StrategyGraph.from_hymeko(_BC).classify() == "bc"
    assert StrategyGraph.from_hymeko(_DAGGER).classify() == "dagger"
    assert StrategyGraph.from_hymeko(_TD3BC).classify() == "td3_bc"


def test_to_campaign_config_reads_declared_knobs() -> None:
    cfg = StrategyGraph.from_hymeko(_TD3BC).to_campaign_config(name="q", select="standing")
    assert cfg.total_steps == 150_000 and cfg.n_demos == 200 and cfg.seeds == (0, 1, 2)
    assert cfg.offpolicy is not None
    assert cfg.offpolicy["critic_huber"] == pytest.approx(1.0) and cfg.offpolicy["bc_coef"] == pytest.approx(2.5)


def test_warm_start_and_aggregate_cap_read_from_stages() -> None:
    """Smoothing knobs are read from stage nodes: warm_start from the retrain bc, aggregate_cap from aggregate.
    (Both were measured no-help on standing — the declared graph now sets warm_start 0, aggregate_cap 2 — but the
    reader must still surface whatever the graph declares, not a hardcoded default.)"""
    on = StrategyGraph({"t": Stage("t", "bc", {"warm_start": 1.0})}, (), 0, (0,), "x")
    off = StrategyGraph({"t": Stage("t", "bc", {})}, (), 0, (0,), "x")
    assert on._warm_start() is True and off._warm_start() is False
    cfg = StrategyGraph.from_hymeko(_DAGGER).to_dagger_config(name="q", select="standing")
    assert cfg.warm_start is False and cfg.aggregate_cap_rounds == 2   # what the declared graph currently sets


def test_td3bc_runner_rejects_dagger_graph() -> None:
    """Symmetric guard: the td3_bc runner rejects a graph with no critic node (it classifies dagger)."""
    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("must not reach the executor")

    with pytest.raises(ValueError, match="not 'td3_bc'"):
        run_td3bc_graph(_DAGGER, make_env=_boom, build=_boom, demos=_boom, measure=_boom,
                        name="q", select="standing")
