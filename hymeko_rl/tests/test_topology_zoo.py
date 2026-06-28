"""Tests for the topology zoo + the controller benchmark (Phase 1, Kato's isomorphic-controllers program)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.controller_bench import equivariance_check, run_topology_map
from hymeko_rl.topology_zoo import TOPOLOGIES, permuted


@pytest.mark.parametrize("name", list(TOPOLOGIES))
def test_each_topology_is_a_valid_signed_graph(name: str) -> None:
    hg = TOPOLOGIES[name](9, seed=0)
    assert hg.n_vertices == 9
    assert hg.edges.shape[1] == 2 and hg.edges.shape[0] >= 2
    assert set(np.unique(hg.signs).tolist()) <= {-1, 1}
    assert hg.edges.max() < 9 and hg.edges.min() >= 0          # endpoints in range
    assert hg.edges.shape[0] % 2 == 0                          # every undirected edge → two arcs


def test_family_shapes_are_distinct() -> None:
    """The families are genuinely different topologies (different undirected edge counts at N=9)."""
    counts = {name: TOPOLOGIES[name](9, seed=0).edges.shape[0] // 2 for name in TOPOLOGIES}
    assert counts["chain"] == 8 and counts["ring"] == 9 and counts["complete"] == 36
    assert counts["chain"] < counts["grid"] < counts["complete"]


def test_grid_requires_perfect_square() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        TOPOLOGIES["grid"](8, seed=0)


def test_permuted_is_isomorphic() -> None:
    """A relabelling preserves vertex/edge counts and the multiset of signs (an isomorphic copy)."""
    hg = TOPOLOGIES["ring"](9, seed=1)
    perm = np.array([3, 1, 4, 1, 5, 9, 2, 6, 8]) % 9          # not a permutation → must reject
    with pytest.raises(ValueError, match="permutation"):
        permuted(hg, perm)
    good = np.array([2, 0, 1, 3, 4, 5, 6, 7, 8])
    iso = permuted(hg, good)
    assert iso.n_vertices == hg.n_vertices
    assert iso.edges.shape == hg.edges.shape
    assert sorted(iso.signs.tolist()) == sorted(hg.signs.tolist())


def test_topology_map_diagonal_tends_to_win() -> None:
    """The matching controller (controller topology = plant topology) should usually be the best — the core
    Kato hypothesis at small scale. We require it for the structurally distinctive star and complete plants."""
    r = run_topology_map(names=["chain", "star", "complete"], n_nodes=9, seeds=2, epochs=80, hidden=16)
    assert r["diagonal_is_best"]["star"], r["matrix"]["star"]
    assert r["diagonal_is_best"]["complete"], r["matrix"]["complete"]


@pytest.mark.parametrize("topology", ["chain", "star", "small_world", "complete"])
def test_controller_is_permutation_equivariant(topology: str) -> None:
    """The exact well-definedness guard: with identical weights, the controller's pooled output on (H, x)
    equals the output on the isomorphic (π(H), π(x)). A ~0 residual proves C(H) depends only on the topology
    up to isomorphism — so the topology→performance map's structure is real, not a labelling artefact."""
    eq = equivariance_check(topology=topology, n_nodes=9)
    assert eq["equivariant"], eq
