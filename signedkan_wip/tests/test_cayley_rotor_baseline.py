"""Tests for the inductive Cayley-rotor signed-link baseline.

Run: pytest -p no:randomly signedkan_wip/tests/test_cayley_rotor_baseline.py
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.baselines.cayley_rotor_baseline import (
    STRUCT_DIM,
    _structural_features,
)
from signedkan_wip.src.baselines.registry import (
    GraphMeta,
    HParams,
    get_baseline,
    list_baselines,
)


def _toy_graph():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.int64)
    signs = np.array([1, -1, 1, -1, 1], dtype=np.int64)
    return edges, signs, 4


def test_registered():
    assert "cayley_rotor" in list_baselines()


def test_structural_features_shape_and_identity_free():
    edges, signs, n = _toy_graph()
    feats = _structural_features(edges, signs, n)
    assert feats.shape == (n, STRUCT_DIM)
    assert np.isfinite(feats).all()
    # Two nodes with the same signed-degree profile must get identical features
    # (features depend on structure, not node id) — the leakage-free property.
    e2 = np.array([[0, 1], [2, 3]], dtype=np.int64)
    s2 = np.array([1, 1], dtype=np.int64)
    f2 = _structural_features(e2, s2, 4)
    assert np.allclose(f2[0], f2[1]) and np.allclose(f2[2], f2[3])


def test_build_context_and_encode_shapes():
    edges, signs, n = _toy_graph()
    strat = get_baseline("cayley_rotor")
    hp = strat.default_hparams()
    ctx = strat.build_context(edges, signs, n, torch.device("cpu"))
    model = strat.build_model(GraphMeta(n_nodes=n), hp)
    z = model.encode_nodes(*ctx)
    assert z.shape == (n, 2 * hp.hidden)
    assert torch.isfinite(z).all()
    logits = model.edge_logits(z, torch.from_numpy(edges))
    assert logits.shape == (edges.shape[0],)


def test_cyc_registered_and_cycle_features_balance():
    assert "cayley_rotor_cyc" in list_baselines()
    from signedkan_wip.src.baselines.cayley_rotor_baseline import _cycle_features
    # balanced triangle: 0 negative edges (even) → balanced for all 3 nodes.
    e = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    f = _cycle_features(e, np.array([1, 1, 1], dtype=np.int64), 3)
    assert f.shape == (3, 3)                        # 3 features for k=3
    assert np.allclose(f[:, 0], np.log1p(1.0))      # 1 balanced triangle/node
    assert np.allclose(f[:, 1], 0.0)                # 0 unbalanced
    assert np.allclose(f[:, 2], 1.0)                # balanced fraction 1.0
    # one negative edge (odd) → unbalanced triangle.
    f2 = _cycle_features(e, np.array([1, 1, -1], dtype=np.int64), 3)
    assert np.allclose(f2[:, 0], 0.0) and np.allclose(f2[:, 1], np.log1p(1.0))
    assert np.allclose(f2[:, 2], 0.0)


def test_mlp_embed_control_is_a_fair_ablation():
    """The MLP-embed control (rotor geometry ablation) shares the proj/SGCN/
    classifier and the same embedding_dim, with >= the rotor's params (a generous
    control), and is inductive (node-count-free)."""
    assert {"mlp_embed", "mlp_embed_walk"} <= set(list_baselines())
    hp = HParams(hidden=32, n_layers=2)
    meta = GraphMeta(n_nodes=500)
    rotor = get_baseline("cayley_rotor").build_model(meta, hp)
    mlp = get_baseline("mlp_embed").build_model(meta, hp)
    # Same downstream: same embedding_dim feeding proj, same SGCN depth.
    assert rotor.emb.embedding_dim == mlp.emb.embedding_dim
    assert mlp.num_parameters() >= rotor.num_parameters()   # not param-starved
    # Forward works and produces the same 2*hidden node code shape.
    edges, signs, n = _toy_graph()
    strat = get_baseline("mlp_embed")
    ctx = strat.build_context(edges, signs, n, torch.device("cpu"))
    z = strat.build_model(GraphMeta(n_nodes=n), hp).encode_nodes(*ctx)
    assert z.shape == (n, 2 * hp.hidden) and torch.isfinite(z).all()
    # Inductive: param count independent of node count.
    p100 = get_baseline("mlp_embed").build_model(GraphMeta(n_nodes=100), hp).num_parameters()
    p1m = get_baseline("mlp_embed").build_model(GraphMeta(n_nodes=10**6), hp).num_parameters()
    assert p100 == p1m


def test_param_count_independent_of_node_count():
    """The whole point: no per-node table, so model size does not grow with the
    graph — unlike the transductive nn.Embedding(n_nodes, hidden) baselines."""
    strat = get_baseline("cayley_rotor")
    hp = HParams(hidden=32, n_layers=2)
    p_small = strat.build_model(GraphMeta(n_nodes=100), hp).num_parameters()
    p_large = strat.build_model(GraphMeta(n_nodes=1_000_000), hp).num_parameters()
    assert p_small == p_large                     # item-count-free
    assert p_small < 1_000_000 * 32               # far below a dense node table
