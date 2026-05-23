"""Planted balanced-clique generator for the cliques benchmark.

Extends ``cliques.make_robot_network`` by planting *known* balanced
cliques of specified sizes into the network. The planted edges form
the ground truth that recall metrics measure against; the remaining
edges carry the ambient signal (faction-based) plus observation noise.

A balanced clique of size ``c`` has C(c, 2) internal edges. We plant
them with **all-positive signs** by default — the simplest balanced
configuration (σ-product = +1 trivially). The optional
``planted_sign_strategy="split"`` divides each planted clique into two
sub-blocks with internal `+` and across-block `−`, which is still
balanced (Cartwright-Harary's "two factions, friend of friend") and
gives the detector a richer signal to find.

Ground-truth fields:
  - ``planted_cliques`` : list of ``Clique`` records the generator
    inserted (size, members, edges, signs, sigma_product).
  - ``planted_edge_mask`` : per-edge bool, True iff the edge belongs
    to a planted clique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..datasets import SignedGraph
from .cliques import Clique, RobotNetworkBundle, _clique_balance_indicator


@dataclass
class PlantedRobotNetworkBundle(RobotNetworkBundle):
    """Robot network with known balanced cliques planted.

    Inherits all fields of ``RobotNetworkBundle`` (graph, positions,
    seed, comm_range, noise_prob, area_size, name, n_factions,
    factions) and adds the planted-cliques ground truth.
    """

    planted_cliques: list[Clique] = field(default_factory=list)
    planted_edge_mask: np.ndarray | None = None   # (n_edges,) bool

    @property
    def n_planted(self) -> int:
        return len(self.planted_cliques)


def _verify_clique_balanced_via_signs_dict(
    members: tuple[int, ...],
    sign_of: dict[tuple[int, int], int],
) -> int:
    """Same as ``_clique_balance_indicator`` — local alias for readability."""
    return _clique_balance_indicator(members, sign_of)


def make_planted_balanced_cliques(
    n_robots: int = 50,
    clique_sizes: list[int] | None = None,
    area_size: float = 10.0,
    comm_range: float = 4.0,
    noise_prob: float = 0.05,
    n_factions: int = 0,
    planted_sign_strategy: Literal["all_positive", "split"] = "all_positive",
    seed: int = 0,
    name: str = "planted",
) -> PlantedRobotNetworkBundle:
    """Generate a robot network with planted balanced cliques.

    Procedure:

    1. Place ``n_robots`` uniformly in ``[0, area_size]²``.
    2. For each requested ``c_i`` in ``clique_sizes``, pick ``c_i``
       vertices uniformly at random (without replacement across
       cliques — every robot lives in at most one planted clique).
    3. Add ALL C(c_i, 2) internal edges with the chosen sign strategy.
       Verify the σ-product is +1.
    4. For the *remaining* (non-planted) edges: include every pair
       within ``comm_range``, assign signs by faction (if
       ``n_factions ≥ 2``) or all-positive otherwise, apply
       ``noise_prob`` flips. Skip pairs already covered by planted
       edges.

    Returns ``PlantedRobotNetworkBundle`` with ``planted_cliques``
    and ``planted_edge_mask`` populated.
    """
    if clique_sizes is None:
        clique_sizes = [5, 4, 3]
    if any(s < 3 for s in clique_sizes):
        raise ValueError(f"clique sizes must be ≥ 3, got {clique_sizes}")
    if sum(clique_sizes) > n_robots:
        raise ValueError(
            f"sum(clique_sizes)={sum(clique_sizes)} exceeds n_robots"
            f"={n_robots}; planted cliques would overlap."
        )

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, area_size, size=(n_robots, 2)).astype(np.float32)
    factions = None
    if n_factions >= 2:
        factions = rng.integers(0, n_factions, size=n_robots,
                                  dtype=np.int64)

    # Assign each planted clique a disjoint vertex set.
    available = list(range(n_robots))
    rng.shuffle(available)
    cursor = 0
    planted_clique_members: list[tuple[int, ...]] = []
    for size in clique_sizes:
        members = tuple(sorted(available[cursor:cursor + size]))
        cursor += size
        planted_clique_members.append(members)

    # Build planted edges + signs.
    planted_edges: dict[tuple[int, int], int] = {}
    planted_cliques: list[Clique] = []
    for members in planted_clique_members:
        c_size = len(members)
        clique_edges: list[tuple[int, int]] = []
        clique_signs: list[int] = []

        if planted_sign_strategy == "all_positive":
            # Easiest balanced configuration: all + signs.
            for i in range(c_size):
                for j in range(i + 1, c_size):
                    a, b = members[i], members[j]
                    planted_edges[(a, b)] = +1
                    clique_edges.append((a, b))
                    clique_signs.append(+1)
        elif planted_sign_strategy == "split":
            # Split the clique into two sub-blocks; intra-block edges
            # are +, inter-block are −. σ-product around any triangle:
            # either all three vertices in one block (all +, product
            # +1) or one block × two vertices in the other block
            # (two − and one + → product +1). Balanced by construction.
            half = c_size // 2
            block_a = set(members[:half])
            for i in range(c_size):
                for j in range(i + 1, c_size):
                    a, b = members[i], members[j]
                    s = 1 if (a in block_a) == (b in block_a) else -1
                    planted_edges[(a, b)] = s
                    clique_edges.append((a, b))
                    clique_signs.append(s)
        else:
            raise ValueError(
                f"unknown planted_sign_strategy={planted_sign_strategy!r}"
            )

        # Verify balance via the proper triangle-check, using the
        # planted_edges dict as the sign source.
        sigma_prod = _verify_clique_balanced_via_signs_dict(
            members, planted_edges)
        if sigma_prod != 1:
            raise RuntimeError(
                f"Planted clique on members={members} is not balanced "
                f"(σ-product={sigma_prod}). Generator bug."
            )
        planted_cliques.append(Clique(
            members=members, edges=clique_edges,
            signs=clique_signs, sigma_product=sigma_prod,
        ))

    # Build the rest of the network: range-based pairs not already
    # claimed by a planted clique.
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    planted_mask_list: list[bool] = []

    for u in range(n_robots):
        for v in range(u + 1, n_robots):
            key = (u, v)
            if key in planted_edges:
                edges.append(key)
                signs.append(planted_edges[key])
                planted_mask_list.append(True)
                continue
            d = float(np.linalg.norm(pos[u] - pos[v]))
            if d > comm_range:
                continue
            # Ambient signal.
            if factions is not None:
                base = 1 if factions[u] == factions[v] else -1
            else:
                base = 1
            if rng.random() < noise_prob:
                base = -base
            edges.append(key)
            signs.append(int(base))
            planted_mask_list.append(False)

    edges_arr = (np.array(edges, dtype=np.int64)
                  if edges else np.zeros((0, 2), dtype=np.int64))
    signs_arr = (np.array(signs, dtype=np.int8)
                  if signs else np.zeros((0,), dtype=np.int8))
    planted_mask = np.array(planted_mask_list, dtype=bool)
    g = SignedGraph(edges=edges_arr, signs=signs_arr, n_nodes=n_robots)
    return PlantedRobotNetworkBundle(
        graph=g, positions=pos, seed=seed,
        comm_range=comm_range, noise_prob=noise_prob,
        area_size=area_size, name=name,
        n_factions=int(n_factions), factions=factions,
        planted_cliques=planted_cliques,
        planted_edge_mask=planted_mask,
    )


__all__ = [
    "PlantedRobotNetworkBundle",
    "make_planted_balanced_cliques",
]
