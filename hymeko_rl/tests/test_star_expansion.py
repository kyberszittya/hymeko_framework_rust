"""Bipartite star-expansion of the kinematic hypergraph: each joint becomes a hub node (the hyperedge step)."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.agents.policy import HSiKANBackbone
from hymeko_rl.agents.star_entropy import star_expansion_entropy


def _chain3() -> HypergraphState:
    """a—b—c: two joints (a→b, b→c), each as a down(+1)/up(-1) arc pair (the from_mjcf form)."""
    return HypergraphState(
        vertex_labels=("a", "b", "c"),
        edges=np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64),
        signs=np.array([1, -1, 1, -1], dtype=np.int64), topo_hash="chain3")


def test_star_expansion_adds_one_hub_per_joint() -> None:
    """N original vertices + J joint-hubs; hub labels name the bridged links; topo_hash re-encode-gated."""
    star = _chain3().star_expansion()
    assert star.n_vertices == 3 + 2                      # 3 links + 2 joint hubs
    assert star.vertex_labels[3:] == ("hub:a~b", "hub:b~c")
    assert star.topo_hash.endswith(":star")


def test_hub_connects_exactly_its_parent_and_child() -> None:
    """Each hub mediates only its own parent/child, with down(+1)/up(-1) routed through it."""
    star = _chain3().star_expansion()
    hub_ab = 3
    # arcs touching hub_ab: (a→hub +1), (hub→b +1), (b→hub -1), (hub→a -1)
    touching = {(int(s), int(d), int(sg)) for (s, d), sg in zip(star.edges, star.signs)
                if hub_ab in (int(s), int(d))}
    assert touching == {(0, 3, 1), (3, 1, 1), (1, 3, -1), (3, 0, -1)}


def test_star_expansion_feeds_hsikan_and_h_star_over_hubs() -> None:
    """The expanded graph is a drop-in for the HSiKAN backbone; node_activations now span vertices+hubs and
    H★ reads the hub-augmented node set (a finite entropy in [0,1])."""
    star = _chain3().star_expansion()
    a_pos, a_neg = star.dense_signed_adj()
    assert a_pos.shape == (5, 5) and a_neg.shape == (5, 5)
    backbone = HSiKANBackbone(in_feat=2, hidden=4, n_layers=2, hg_state=star)
    acts = backbone.node_activations(torch.randn(2, 5, 2))   # (B, N+J, hidden)
    assert acts.shape == (2, 5, 4)
    h = star_expansion_entropy(acts)
    assert h.shape == (2,) and bool((h >= 0).all()) and bool((h <= 1.0 + 1e-6).all())


def test_star_expansion_requires_a_joint() -> None:
    empty = HypergraphState(vertex_labels=("a",), edges=np.zeros((0, 2), dtype=np.int64),
                            signs=np.zeros((0,), dtype=np.int64), topo_hash="empty")
    try:
        empty.star_expansion()
    except ValueError:
        return
    raise AssertionError("expected ValueError when there is no joint")
