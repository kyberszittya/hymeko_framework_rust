"""Hypergraph zoo — the defining combinatorial property of each canonical family, pinned.

These are benchmark generators for the HyMeKo / Nagare line (signed structures, cycles, holonomy, matroid
constraints), so each test asserts the family's *defining* property (design parameters, uniformity, the cycle
overlap structure, minimal-dependency circuits, …) — not just that a graph was produced.
"""

from __future__ import annotations

from scenarios.hypergraph_zoo import (
    affine_plane,
    complete_uniform,
    fano_plane,
    graphic_matroid_circuits,
    kneser,
    loose_cycle,
    projective_plane,
    random_uniform,
    simplex_boundary,
    steiner_triple_system,
    sunflower,
    tight_cycle,
)


def test_fano_plane_is_the_2_7_3_1_design() -> None:
    f = fano_plane()
    assert f.n_vertices == 7 and f.n_edges == 7
    assert f.is_uniform(3) and f.is_regular(3) and f.is_linear() and f.is_self_dual()
    assert f.is_2_design() == (True, 1)


def test_projective_planes_scale_as_q2_q_1() -> None:
    for q in (2, 3, 5):
        p = projective_plane(q)
        assert p.n_vertices == q * q + q + 1 and p.n_edges == q * q + q + 1
        assert p.is_uniform(q + 1) and p.is_regular(q + 1) and p.is_linear()
        assert p.is_2_design() == (True, 1)


def test_affine_plane_is_s_2_q_q2() -> None:
    a = affine_plane(3)
    assert a.n_vertices == 9 and a.n_edges == 12 and a.is_uniform(3)
    assert a.is_2_design() == (True, 1)                     # AG(2,3) = S(2,3,9)


def test_steiner_triple_systems_cover_every_pair_once() -> None:
    for v in (7, 9, 13, 15):
        s = steiner_triple_system(v)
        assert s.n_vertices == v and s.is_uniform(3)
        assert s.is_2_design() == (True, 1)                 # the defining Steiner property
        assert s.n_edges == v * (v - 1) // 6


def test_complete_uniform_has_all_k_subsets() -> None:
    from math import comb
    k5 = complete_uniform(5, 3)
    assert k5.n_edges == comb(5, 3) and k5.is_uniform(3)


def test_kneser_5_2_is_the_petersen_hypergraph() -> None:
    kg = kneser(5, 2, 2)
    assert kg.n_vertices == 10 and kg.n_edges == 15        # Petersen: 10 vertices, 15 edges


def test_loose_and_tight_cycle_overlap_structure() -> None:
    lc = loose_cycle(3, 5)
    assert lc.is_uniform(3)
    assert all(len(lc.edges[i] & lc.edges[(i + 1) % 5]) == 1 for i in range(5))   # loose: share one vertex
    tc = tight_cycle(3, 7)
    assert tc.is_uniform(3)
    assert all(len(tc.edges[i] & tc.edges[(i + 1) % 7]) == 2 for i in range(7))   # tight: share k−1 vertices


def test_sunflower_has_a_constant_core() -> None:
    sf = sunflower(2, 4, 3)
    pairwise = {a & b for i, a in enumerate(sf.edges) for b in sf.edges[i + 1:]}
    assert len(pairwise) == 1 and len(next(iter(pairwise))) == 2                  # every two edges meet in the core


def test_random_hypergraph_is_seeded_and_reproducible() -> None:
    a = random_uniform(9, 3, 0.3, seed=1)
    b = random_uniform(9, 3, 0.3, seed=1)
    assert a.edges == b.edges and a.is_uniform(3)
    assert random_uniform(9, 3, 0.3, seed=2).edges != a.edges


def test_graphic_matroid_circuits_are_the_simple_cycles() -> None:
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    gm = graphic_matroid_circuits(k4)
    assert gm.n_edges == 7                                  # K4: 4 triangles + 3 four-cycles
    assert sorted(gm.edge_sizes()) == [3, 3, 3, 3, 4, 4, 4]
    assert graphic_matroid_circuits([(0, 1), (1, 2), (0, 2)]).n_edges == 1        # a triangle = one circuit


def test_simplex_boundary_facets() -> None:
    sb = simplex_boundary(4)
    assert sb.n_edges == 4 and sb.is_uniform(3)            # the (n−1)-subsets = the tetrahedron's triangular faces


def test_dual_and_self_duality() -> None:
    assert fano_plane().is_self_dual()
    assert not loose_cycle(3, 4).is_self_dual()            # a loose cycle is not self-dual
    assert loose_cycle(3, 4).dual().dual().incidence_matrix().shape == loose_cycle(3, 4).incidence_matrix().shape
