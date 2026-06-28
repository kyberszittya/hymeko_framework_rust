"""The extremal topologies (Phase 4): Petersen, Kneser, Grötzsch (Mycielskian), expander — verified by the graph
invariants that make them interesting (strong-regularity, triangle-freeness, regularity). Plus the hypergraph
LIFTS (Phase 4b): the closed-neighbourhood hypergraph version of each regular graph."""
from hymeko_rl.hypergraph_designs import closed_neighbourhood_blocks, graph_to_kuniform
from hymeko_rl.topology_invariants import invariants
from hymeko_rl.topology_zoo import HYPER_TOPOLOGIES, TOPOLOGIES, petersen


def _adj(hg):
    nb = {i: set() for i in range(hg.n_vertices)}
    for a, b in hg.edges.tolist():
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    return nb


def test_all_registered() -> None:
    for name in ("petersen", "kneser", "grotzsch", "expander"):
        assert name in TOPOLOGIES


def test_petersen_is_srg() -> None:
    hg = TOPOLOGIES["petersen"](seed=0)
    assert hg.n_vertices == 10
    nb = _adj(hg)
    assert all(len(nb[i]) == 3 for i in range(10))           # 3-regular
    for i in range(10):                                       # strongly-regular (10,3,0,1):
        for j in nb[i]:
            assert len(nb[i] & nb[j]) == 0                    #   adjacent pairs share lambda=0 (no triangle)
        for j in range(10):
            if j != i and j not in nb[i]:
                assert len(nb[i] & nb[j]) == 1                #   non-adjacent pairs share mu=1


def test_kneser_5_2_is_petersen() -> None:
    hg = TOPOLOGIES["kneser"](5, seed=0)                      # K(5,2) is the Petersen graph
    assert hg.n_vertices == 10
    nb = _adj(hg)
    assert all(len(nb[i]) == 3 for i in range(10))           # 3-regular, like Petersen


def test_grotzsch_triangle_free() -> None:
    hg = TOPOLOGIES["grotzsch"](seed=0)
    assert hg.n_vertices == 11                               # Mycielskian of C5 (2*5+1)
    nb = _adj(hg)
    for i in range(11):
        for j in nb[i]:
            assert not (nb[i] & nb[j])                       # adjacent pairs share no neighbour -> triangle-free


def test_expander_is_regular() -> None:
    hg = TOPOLOGIES["expander"](12, seed=1)
    nb = _adj(hg)
    assert all(len(nb[i]) == 3 for i in range(hg.n_vertices))   # random 3-regular


def test_closed_neighbourhood_lift_is_k_uniform() -> None:
    blocks = closed_neighbourhood_blocks(petersen(seed=0))      # 3-regular -> closed nbhd has 3+1=4 vertices
    assert len(blocks) == 10                                    # one block per vertex
    assert {len(b) for b in blocks} == {4}                      # 4-uniform
    assert all(i in blocks[i] for i in range(10))               # the vertex itself is in its block (CLOSED)


def test_graph_to_kuniform_star_expands() -> None:
    hg = graph_to_kuniform(petersen(seed=0), tag="petersen_h")
    assert hg.n_vertices == 20                                  # 10 points + 10 block-hubs


def test_hyper_lift_changes_walk_geometry() -> None:
    # The hub-mediated walks raise frame coherence above the 2-uniform graph's (a genuinely different walk basis).
    graph_coh = invariants(petersen(seed=0))["frame_coherence"]
    hyper_coh = invariants(HYPER_TOPOLOGIES["petersen_h"](seed=0))["frame_coherence"]
    assert abs(graph_coh - 1 / 3) < 5e-3                        # graph: the tight frame
    assert hyper_coh > graph_coh + 0.1                          # hyper: hubs add walk overlap


def test_hyper_topologies_registered() -> None:
    for name in ("petersen_h", "ring_h", "expander_h", "complete_h"):
        assert name in HYPER_TOPOLOGIES
        assert HYPER_TOPOLOGIES[name](10, seed=0).n_vertices > 0
