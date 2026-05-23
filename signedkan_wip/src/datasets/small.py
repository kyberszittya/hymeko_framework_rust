"""Small signed-graph datasets — real (Zachary karate, faction-signed)
and synthetic (SBM-signed, controllable).

Real:
  - karate_faction: Zachary's karate club via NetworkX, sign +1 if
    both endpoints remained in the same faction after the split,
    sign −1 if cross-faction. 34 nodes, 78 edges. Tiny — tests
    pipeline correctness, not architectural discriminators.

Synthetic:
  - sbm_signed: Stochastic Block Model with sign-aware structural
    bias. K communities, within-community edges biased positive,
    between-community edges biased negative, with controllable
    noise. Returns a SignedGraph plus the ground-truth community
    labels for visualisation.

  - hierarchical_signed: 2-level SBM where coarse + fine community
    structure gives k=4 cycles distinguishing signal that k=3
    triads do not — designed to favour mixed-arity HSiKAN.
"""
from __future__ import annotations

import numpy as np

from .legacy import SignedGraph


def karate_faction_signed() -> SignedGraph:
    """Zachary's karate club (NetworkX) with faction-derived signs."""
    import networkx as nx
    g = nx.karate_club_graph()
    factions = nx.get_node_attributes(g, "club")
    edges = []
    signs = []
    for u, v in g.edges():
        edges.append((int(u), int(v)))
        s = +1 if factions[u] == factions[v] else -1
        signs.append(s)
    return SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=g.number_of_nodes(),
    )


def sbm_signed(n_nodes: int = 200, n_communities: int = 4,
                p_in: float = 0.20, p_out: float = 0.05,
                pos_in: float = 0.85, pos_out: float = 0.15,
                noise: float = 0.05, seed: int = 0
              ) -> tuple[SignedGraph, np.ndarray]:
    """Stochastic Block Model with sign-aware structural bias.

    Parameters
    ----------
    n_nodes        : graph size
    n_communities  : number of blocks; nodes assigned uniformly
    p_in, p_out    : edge probability within / between communities
    pos_in         : conditional P(sign=+1 | within-community edge)
    pos_out        : conditional P(sign=+1 | cross-community edge)
    noise          : symmetric per-edge sign-flip probability
    seed           : RNG seed for reproducibility

    Returns
    -------
    (SignedGraph, community_labels)
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_communities, size=n_nodes)
    edges = []
    signs = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            same = labels[u] == labels[v]
            p = p_in if same else p_out
            if rng.random() >= p:
                continue
            base_p_pos = pos_in if same else pos_out
            sign = +1 if rng.random() < base_p_pos else -1
            if rng.random() < noise:
                sign = -sign
            edges.append((u, v))
            signs.append(sign)
    return (SignedGraph(
                edges=np.array(edges, dtype=np.int64),
                signs=np.array(signs, dtype=np.int8),
                n_nodes=n_nodes,
            ),
            labels)


def hierarchical_signed(n_nodes: int = 240,
                         n_coarse: int = 3,
                         n_fine_per_coarse: int = 3,
                         p_within_fine: float = 0.30,
                         p_within_coarse: float = 0.10,
                         p_cross_coarse: float = 0.04,
                         pos_fine: float = 0.92,
                         pos_within_coarse: float = 0.55,
                         pos_cross: float = 0.10,
                         noise: float = 0.03,
                         seed: int = 0
                       ) -> tuple[SignedGraph, dict]:
    """Two-level hierarchical SBM designed so that k=4 motifs can
    distinguish structure k=3 cannot.

    Each node lives in one (coarse, fine) cell. Three regimes:
      - same fine-block:    high edge density, very positive signs
      - same coarse / different fine: medium density, mostly positive
      - cross-coarse:       low density, mostly negative
    A 4-cycle of (a, b, c, d) where a,b,c are in one fine block and
    d is in a sibling fine block of the same coarse block carries
    a sign pattern that any k=3 sub-triangle of those four vertices
    misses (the {a,b,d}, {a,c,d}, {b,c,d} triads each have only one
    cross-fine edge of the four cycle edges).
    """
    rng = np.random.default_rng(seed)
    n_fine = n_coarse * n_fine_per_coarse
    fine = rng.integers(0, n_fine, size=n_nodes)
    coarse = fine // n_fine_per_coarse
    edges = []
    signs = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if fine[u] == fine[v]:
                p_e, p_pos = p_within_fine, pos_fine
            elif coarse[u] == coarse[v]:
                p_e, p_pos = p_within_coarse, pos_within_coarse
            else:
                p_e, p_pos = p_cross_coarse, pos_cross
            if rng.random() >= p_e:
                continue
            sign = +1 if rng.random() < p_pos else -1
            if rng.random() < noise:
                sign = -sign
            edges.append((u, v))
            signs.append(sign)
    return (SignedGraph(
                edges=np.array(edges, dtype=np.int64),
                signs=np.array(signs, dtype=np.int8),
                n_nodes=n_nodes,
            ),
            {"fine": fine, "coarse": coarse})
