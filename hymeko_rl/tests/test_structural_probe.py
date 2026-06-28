"""Tests for the structural discriminating probe (hymeko_rl/structural_probe.py).

Pin the toy graph, the structural-vs-bag target distinction (the structural target genuinely reads the
signed adjacency; the bag target does not), dataset determinism, both backbones' forward, params-matching,
and a fast end-to-end run.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.hypergraph_state import HypergraphState
from hymeko_rl.structural_probe import (
    _LOCAL_NODE, _MULTIDIM_CONFIGS, READOUT_MODES, build_chain_graph, build_model, build_multidim_model,
    build_toy_graph, make_dataset, match_mlp_hidden, run_chain_probe, run_multidim_probe, run_probe,
    run_readout_ablation, run_readout_bakeoff, sweep_n_train,
)


def _flip_one_sign(hg: HypergraphState) -> HypergraphState:
    # flip the LAST arc (the 5-6 edge, adjacent to _LOCAL_NODE=6) so the change is within node 6's 2-hop
    # neighbourhood — the 'local' target reads only that neighbourhood, so a far flip would (correctly) be inert.
    signs = hg.signs.copy()
    signs[-1] = -signs[-1]
    return HypergraphState(hg.vertex_labels, hg.edges.copy(), signs, hg.topo_hash + ":flip")


def test_toy_graph_is_signed_and_cyclic() -> None:
    hg = build_toy_graph()
    assert hg.n_vertices == 7
    assert set(int(s) for s in hg.signs) == {-1, 1}          # both signs present
    # a cycle exists ⇒ #undirected edges >= #vertices (tree would be N-1); here clearly more.
    assert hg.edges.shape[0] // 2 >= hg.n_vertices


def test_structural_target_reads_adjacency_bag_does_not() -> None:
    """Flipping a graph sign changes the structural target (it reads B = A⁺−A⁻) but not the bag target
    (a per-node sum) — the property that makes the probe discriminating."""
    hg, hg2 = build_toy_graph(), _flip_one_sign(build_toy_graph())
    _, y_struct = make_dataset(hg, 64, "structural", seed=7)
    _, y_struct2 = make_dataset(hg2, 64, "structural", seed=7)      # same X (same seed), different signs
    _, y_bag = make_dataset(hg, 64, "bag", seed=7)
    _, y_bag2 = make_dataset(hg2, 64, "bag", seed=7)
    assert not torch.allclose(y_struct, y_struct2), "structural target must depend on the signed adjacency"
    assert torch.allclose(y_bag, y_bag2), "bag target must be structure-independent"


def test_dataset_shapes_and_determinism() -> None:
    hg = build_toy_graph()
    x, y = make_dataset(hg, 32, "structural", seed=3)
    assert x.shape == (32, 7, 1) and y.shape == (32,)
    x2, y2 = make_dataset(hg, 32, "structural", seed=3)
    assert torch.allclose(x, x2) and torch.allclose(y, y2)         # seed-deterministic
    assert torch.isfinite(y).all()


def test_both_backbones_forward_finite() -> None:
    hg = build_toy_graph()
    x, _ = make_dataset(hg, 8, "bag", seed=0)
    for kind in ("hsikan", "mlp"):
        model = build_model(kind, hg, hidden=16)  # type: ignore[arg-type]
        out = model(x)
        assert out.shape == (8,) and torch.isfinite(out).all()


def test_params_match_is_close() -> None:
    hg = build_toy_graph()
    _mlp_h, hk, mlp = match_mlp_hidden(hg, 32)
    assert abs(hk - mlp) / hk < 0.15, f"params should match within 15%: hsikan={hk} mlp={mlp}"


def test_run_probe_smoke_returns_finite_mses() -> None:
    """A fast end-to-end run returns the four (target,backbone) MSEs, all finite, with both ratios present.

    (The directional finding — HSiKAN wins the bag/pooling target, and the structural gap grows with data —
    only converges at full scale (epochs=300, n>=256); it is documented in the report from the full run, not
    asserted here, since at this tiny smoke scale the nets are far from converged.)"""
    report = run_probe(hsikan_hidden=16, n_train=64, n_test=128, seeds=2, epochs=40)
    rows = report["results"]
    assert len(rows) == 4
    assert all(np.isfinite(r["mse_median"]) and r["mse_median"] >= 0.0 for r in rows)
    ratios = report["advantage_ratio_mlp_over_hsikan"]
    assert np.isfinite(ratios["structural"]) and np.isfinite(ratios["bag"])


def test_local_target_reads_adjacency_and_is_node_specific() -> None:
    """The 'local' target reads one node's signed 2-hop value: it depends on the signs (a flip changes it)
    and is distinct from the pooled 'structural' sum — the node-specific signal pooling cannot isolate."""
    hg, hg2 = build_toy_graph(), _flip_one_sign(build_toy_graph())
    _, y_local = make_dataset(hg, 64, "local", seed=5)
    _, y_local2 = make_dataset(hg2, 64, "local", seed=5)          # same X (same seed), flipped adjacent sign
    _, y_struct = make_dataset(hg, 64, "structural", seed=5)
    _, y_bag = make_dataset(hg, 64, "bag", seed=5)
    assert not torch.allclose(y_local, y_local2), "local target must read the signed adjacency (near node)"
    assert not torch.allclose(y_local, y_struct), "local (one node) must differ from the pooled sum"
    assert not torch.allclose(y_local, y_bag), "local must use structure (unlike the bag target)"
    assert 0 <= _LOCAL_NODE < hg.n_vertices


def test_concat_readout_forward_and_dim() -> None:
    """The concat readout keeps node identity: feat-dim is N·hidden (vs hidden for mean-pool), forward finite."""
    hg = build_toy_graph()
    x, _ = make_dataset(hg, 8, "local", seed=0)
    pool = build_model("hsikan", hg, hidden=16, readout="pool")
    concat = build_model("hsikan", hg, hidden=16, readout="concat")
    assert pool.readout.in_features == 16
    assert concat.readout.in_features == hg.n_vertices * 16        # non-collapsing
    out = concat(x)
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_readout_ablation_smoke() -> None:
    """The ablation returns 3 targets × 3 configs, all finite, with the local pool/concat ratio present."""
    report = run_readout_ablation(hsikan_hidden=16, n_train=64, n_test=128, seeds=2, epochs=40)
    rows = report["rows"]
    assert len(rows) == 9
    assert {r["config"] for r in rows} == {"hsikan_pool", "hsikan_concat", "mlp"}
    assert all(np.isfinite(r["mse_median"]) for r in rows)
    assert np.isfinite(report["local_pool_over_concat"])


def test_all_readouts_forward_finite() -> None:
    """Every bake-off readout produces a finite scalar; attention has its own scorer params (≠ mean)."""
    hg = build_toy_graph()
    x, _ = make_dataset(hg, 8, "local", seed=0)
    for ro in READOUT_MODES:
        model = build_model("hsikan", hg, hidden=16, readout=ro)  # type: ignore[arg-type]
        out = model(x)
        assert out.shape == (8,) and torch.isfinite(out).all(), ro
    # attention is a strict generalisation of mean-pool → it carries extra (scorer) params.
    assert build_model("hsikan", hg, 16, readout="attention").n_params() > \
        build_model("hsikan", hg, 16, readout="mean").n_params()


def test_readout_bakeoff_smoke_ranks_readouts() -> None:
    """The bake-off ranks all readouts + MLP by worst-case MSE across targets; winner is first."""
    report = run_readout_bakeoff(hsikan_hidden=16, n_train=64, n_test=128, seeds=2, epochs=40)
    assert set(report["ranking"]) == set(READOUT_MODES) | {"mlp"}
    assert report["winner"] == report["ranking"][0]
    assert all(np.isfinite(v) for v in report["worst_case_mse"].values())


def test_pernode_target_is_a_vector() -> None:
    """The per-node target is y ∈ R^N (one value per node) — the multidimensional shape a per-joint output
    matches; it reads the adjacency (differs from a structure-free per-node map)."""
    hg = build_toy_graph()
    x, y = make_dataset(hg, 16, "pernode", seed=2)
    assert x.shape == (16, hg.n_vertices, 1) and y.shape == (16, hg.n_vertices)
    assert torch.isfinite(y).all()


def test_multidim_heads_forward_and_shape() -> None:
    """All four vector-output heads map (B,N,1) → (B,N); per_node has the *fewest* readout params yet keeps
    node identity (no collapse)."""
    hg = build_toy_graph()
    x, _ = make_dataset(hg, 8, "pernode", seed=0)
    for cfg in _MULTIDIM_CONFIGS:
        model = build_multidim_model(cfg, hg, hidden=16)
        out = model(x)
        assert out.shape == (8, hg.n_vertices) and torch.isfinite(out).all(), cfg


def test_multidim_probe_smoke() -> None:
    """The multidim probe returns a row per config + the collapse-cost ratio (pool_expand / per_node)."""
    report = run_multidim_probe(hsikan_hidden=16, n_train=64, n_test=128, seeds=2, epochs=40)
    assert {r["config"] for r in report["rows"]} == set(_MULTIDIM_CONFIGS)
    assert np.isfinite(report["ratio_poolexpand_over_pernode"])


def test_chain_graph_is_a_sparse_line() -> None:
    """A chain of N nodes is a line: 2(N-1) directed arcs, each interior node degree-bounded (sparse)."""
    hg = build_chain_graph(8, seed=1)
    assert hg.n_vertices == 8
    assert hg.edges.shape[0] == 2 * (8 - 1)            # only nearest-neighbour arcs → sparse
    assert hg.edges.shape[0] < 8 * 8                    # far below dense


def test_chain_probe_smoke() -> None:
    """The chain sweep returns a row per length with HSiKAN vs MLP MSE + ratio, all finite."""
    report = run_chain_probe(lengths=[4, 8], hsikan_hidden=16, n_train=64, n_test=128, seeds=2, epochs=40)
    rows = report["rows"]
    assert [r["n_nodes"] for r in rows] == [4, 8]
    assert all(np.isfinite(r["hsikan_mse"]) and np.isfinite(r["ratio"]) for r in rows)


def test_sweep_smoke_returns_rows_per_size() -> None:
    """The data-scaling sweep returns one finite row per size — the curve that separates a
    representational advantage (gap grows with data) from a sample-efficiency one."""
    sweep = sweep_n_train([32, 64], hsikan_hidden=16, seeds=2, epochs=40, n_test=128)
    rows = sweep["rows"]
    assert [r["n_train"] for r in rows] == [32, 64]
    assert all(np.isfinite(r["bag_hsikan"]) and np.isfinite(r["struct_mlp"]) for r in rows)
