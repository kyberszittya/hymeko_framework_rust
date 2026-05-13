"""Robot communication network: synthetic generator + balance theory.

A multi-robot communication network is a signed graph:

  - vertices = robots
  - edges    = pairwise communication attempts in range
  - sign     = ``+`` reliable (high SINR / trusted), ``−`` jammed / lost

Cartwright-Harary (1956) structural balance theory says a signed graph
is *balanced* iff every cycle has an even count of negative edges.
Equivalently, the σ-product around any cycle equals ``+1``. A
**balanced clique** on robots is then a stable communication team: no
internal conflicts, every pairwise link is consistent, every triangle
closes positively (or with paired flips that cancel).

HSiKAN's cycle pool computes σ-products by construction, so the
inductive bias for this prediction problem is already in the model.
v1 of this demo is descriptive — it generates a network and enumerates
its balanced cliques. v0.5 (next pass) trains a small HSiKAN on a
corpus of synthetic networks and predicts edge signs given only
robot positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..datasets import SignedGraph


@dataclass
class Clique:
    """A balanced clique on the robot network."""

    members: tuple[int, ...]            # vertex indices, sorted ascending
    edges: list[tuple[int, int]]        # the (u, v) pairs within the clique
    signs: list[int]                    # ±1 per edge in `edges`
    sigma_product: int                  # ∏ signs — must be +1 for balanced

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def balanced(self) -> bool:
        return self.sigma_product == 1


@dataclass
class RobotNetworkBundle:
    """A snapshot of a synthetic robot communication network."""

    graph: SignedGraph
    positions: np.ndarray                # (n_robots, 2) float
    seed: int
    comm_range: float
    noise_prob: float
    area_size: float
    name: str = "synthetic"

    @property
    def n_robots(self) -> int:
        return self.graph.n_nodes

    @property
    def n_edges(self) -> int:
        return self.graph.edges.shape[0]

    @property
    def n_negative_edges(self) -> int:
        return int((self.graph.signs == -1).sum())

    @property
    def n_positive_edges(self) -> int:
        return int((self.graph.signs == 1).sum())

    def edge_sign(self, u: int, v: int) -> int | None:
        """Return the sign of edge (u, v) or None if no such edge."""
        edges = self.graph.edges
        signs = self.graph.signs
        for i in range(edges.shape[0]):
            a, b = int(edges[i, 0]), int(edges[i, 1])
            if (a == u and b == v) or (a == v and b == u):
                return int(signs[i])
        return None


def make_robot_network(
    n_robots: int = 12,
    area_size: float = 10.0,
    comm_range: float = 3.5,
    noise_prob: float = 0.10,
    seed: int = 0,
    name: str = "synthetic",
) -> RobotNetworkBundle:
    """Generate a synthetic robot communication network.

    - Robots placed uniformly in ``[0, area_size]²``.
    - An edge is created between every pair within ``comm_range``.
    - Each edge starts ``+1`` (reliable link) and is flipped to ``−1``
      with probability ``noise_prob`` (jammed / distrusted).
    - Deterministic given ``seed``.

    Returns ``RobotNetworkBundle``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, area_size, size=(n_robots, 2)).astype(np.float32)
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for u in range(n_robots):
        for v in range(u + 1, n_robots):
            d = float(np.linalg.norm(pos[u] - pos[v]))
            if d <= comm_range:
                s = -1 if rng.random() < noise_prob else +1
                edges.append((u, v))
                signs.append(s)
    edges_arr = (np.array(edges, dtype=np.int64)
                  if edges else np.zeros((0, 2), dtype=np.int64))
    signs_arr = (np.array(signs, dtype=np.int8)
                  if signs else np.zeros((0,), dtype=np.int8))
    g = SignedGraph(edges=edges_arr, signs=signs_arr, n_nodes=n_robots)
    return RobotNetworkBundle(
        graph=g, positions=pos, seed=seed,
        comm_range=comm_range, noise_prob=noise_prob,
        area_size=area_size, name=name,
    )


def enumerate_balanced_cliques(
    bundle: RobotNetworkBundle,
    min_size: int = 3,
    max_size: int = 6,
    limit: int = 20,
) -> list[Clique]:
    """Enumerate balanced cliques in the network.

    Approach:
      1. Build the *unsigned* underlying graph (all communicating pairs).
      2. Enumerate maximal cliques with NetworkX.
      3. For each clique, check σ-product over all its internal edges.
      4. Return balanced ones, sorted by size descending, truncated to
         ``limit``.

    A clique is *balanced* when the product of edge signs along ALL
    pairwise edges is ``+1``. Equivalently: even number of negatives.
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "networkx is required (run `uv sync --group ml --group demo`)."
        ) from e

    G = nx.Graph()
    G.add_nodes_from(range(bundle.n_robots))
    # Sign lookup for fast σ-product evaluation.
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
        G.add_edge(a, b)

    out: list[Clique] = []
    for clique in nx.find_cliques(G):
        size = len(clique)
        if size < min_size or size > max_size:
            continue
        members = tuple(sorted(int(x) for x in clique))
        edges: list[tuple[int, int]] = []
        signs: list[int] = []
        sigma = 1
        for i in range(size):
            for j in range(i + 1, size):
                a, b = members[i], members[j]
                s = sign_of.get((a, b))
                if s is None:
                    sigma = 0
                    break
                edges.append((a, b))
                signs.append(s)
                sigma *= s
            if sigma == 0:
                break
        if sigma == 0:
            continue
        if sigma == 1:
            out.append(Clique(members=members, edges=edges,
                                signs=signs, sigma_product=sigma))

    out.sort(key=lambda c: (-c.size, c.members))
    return out[:limit]


def balance_summary(bundle: RobotNetworkBundle) -> dict[str, float | int]:
    """Quick numeric summary of the network's structural state.

    Useful as a fingerprint above the figure.
    """
    return {
        "n_robots": bundle.n_robots,
        "n_edges": bundle.n_edges,
        "n_positive": bundle.n_positive_edges,
        "n_negative": bundle.n_negative_edges,
        "negative_fraction": (bundle.n_negative_edges / bundle.n_edges
                                if bundle.n_edges else 0.0),
        "mean_degree": (2.0 * bundle.n_edges / bundle.n_robots
                          if bundle.n_robots else 0.0),
    }


__all__ = [
    "Clique",
    "RobotNetworkBundle",
    "balance_summary",
    "enumerate_balanced_cliques",
    "make_robot_network",
]
