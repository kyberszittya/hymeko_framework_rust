"""SDRF — Stochastic Discrete Ricci Flow edge rewiring.

GömbSoma-Ricci-Stim phase 6. A pragmatic implementation of the
SDRF idea from Topping et al. (NeurIPS 2022, "Understanding
Over-Squashing and Bottlenecks on Graphs via Curvature"). The
mechanism:

  * Identify the most strongly negative-curvature edge — the
    worst bottleneck.
  * Add a shortcut edge between two neighbours of the bottleneck's
    endpoints, choosing the pair that creates the most new
    triangles globally (a proxy for maximal Forman κ improvement).
  * Iterate until the minimum Forman κ exceeds a target threshold,
    or the iteration budget is exhausted.

For signed graphs we also compute a sign for each new shortcut
edge using the same feature-inner-product polarity rule as
`StimulusGraphBuilder` (Phase 5).

Notes on the algorithm
----------------------
Topping's original algorithm picks the shortcut that maximises the
curvature improvement at the bottleneck edge itself. For Forman κ,
adding a shortcut (k, l) doesn't change the degree or triangle count
at the bottleneck edge directly — the improvement is global.
We therefore optimise globally: pick the (k, l) that creates the
most new triangles (= largest |adj(k) ∩ adj(l)| count, since the
new edge (k, l) becomes the apex of a new triangle for each
common neighbour). This is a defensible proxy that converges in
practice.

The "stochastic" qualifier in the original SDRF refers to a
Boltzmann tie-breaking among near-optimal candidates; our version
is deterministic for reproducibility. The phase report documents
this design choice.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.vision.forman import (
    FormanCurvatureHead,
)


@dataclass
class SDRFOutput:
    """Output of an SDRF rewiring pass.

    Attributes
    ----------
    edges : LongTensor[n_edges_after, 2]
        Original edges PLUS any shortcut edges added.
    edge_signs : LongTensor[n_edges_after]
        Signs in {-1, +1}. Original signs preserved; new shortcuts
        receive a sign computed from features (or +1 if no features
        were provided).
    n_added : int
        Number of shortcut edges added during the rewiring.
    kappa_min_before : float
        Minimum Forman κ before rewiring.
    kappa_min_after : float
        Minimum Forman κ after rewiring.
    converged : bool
        True if min κ reached the target threshold; False if max
        iterations or no-shortcut-possible terminated early.
    """

    edges: torch.Tensor
    edge_signs: torch.Tensor
    n_added: int
    kappa_min_before: float
    kappa_min_after: float
    converged: bool


class SDRFRewiring(nn.Module):
    """SDRF rewiring: add shortcut edges to relieve κ-bottlenecks.

    Parameters
    ----------
    max_iters : int, default 10
        Cap on shortcut additions.
    min_kappa_target : float, default -2.0
        Stop once min κ ≥ this value. Default -2.0 reflects the
        cycle-Cₙ baseline (every edge in a long cycle has κ = -2).
    sign_threshold : float, default 0.0
        Threshold θ in σ(u, v) = sign(⟨f_u, f_v⟩ − θ) for new
        shortcut edges. Applied only when ``forward`` is given
        ``anchor_features``.

    Preconditions
    -------------
    * ``edges`` is a LongTensor of shape (n_edges, 2).
    * If ``anchor_features`` is provided, it must have shape
      (n_vertices, d).
    * If ``edge_signs`` is provided, its length must match ``edges``.

    Postconditions
    --------------
    * Output ``edges`` is a superset of the input edges
      (no edges are removed; only shortcuts added).
    * ``kappa_min_after >= kappa_min_before`` (the rewiring never
      worsens the bottleneck — pinned by unit test on a path graph).
    """

    def __init__(
        self,
        max_iters: int = 10,
        min_kappa_target: float = -2.0,
        sign_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_iters = max_iters
        self.min_kappa_target = min_kappa_target
        self.sign_threshold = sign_threshold
        self.forman = FormanCurvatureHead()

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        edges: torch.Tensor,
        n_vertices: int,
        anchor_features: Optional[torch.Tensor] = None,
        edge_signs: Optional[torch.Tensor] = None,
    ) -> SDRFOutput:
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                f"edges must have shape (n_edges, 2); got {tuple(edges.shape)}"
            )
        device = edges.device
        n_orig = edges.shape[0]

        if edge_signs is not None and edge_signs.shape[0] != n_orig:
            raise ValueError(
                f"edge_signs has length {edge_signs.shape[0]}; "
                f"expected {n_orig}"
            )
        if anchor_features is not None:
            if (
                anchor_features.ndim != 2
                or anchor_features.shape[0] != n_vertices
            ):
                raise ValueError(
                    f"anchor_features must have shape "
                    f"({n_vertices}, d); got {tuple(anchor_features.shape)}"
                )

        # Working state in Python (sets + lists). Move back to tensors at end.
        edges_list = edges.tolist()
        adj: list[set[int]] = [set() for _ in range(n_vertices)]
        edge_set: set[tuple[int, int]] = set()
        edge_index: dict[tuple[int, int], int] = {}
        for i, (u, v) in enumerate(edges_list):
            adj[u].add(v)
            adj[v].add(u)
            key = (min(u, v), max(u, v))
            edge_set.add(key)
            edge_index[key] = i

        # Compute Forman κ ONCE; maintain it incrementally across SDRF
        # iterations by applying delta updates after each shortcut
        # addition. This avoids the per-iteration O(|E|) Forman
        # recompute that dominated SDRF wall-time after the
        # delta-κ candidate-scan optimisation.
        if edges_list:
            kappa = self.forman(
                torch.tensor(edges_list, dtype=torch.long),
                n_nodes=n_vertices,
            ).edge_kappa.tolist()
        else:
            kappa = []
        kappa_min_before = min(kappa) if kappa else float("inf")
        n_added = 0
        converged = False

        for _ in range(self.max_iters):
            current_min = min(kappa) if kappa else float("inf")
            if current_min >= self.min_kappa_target:
                converged = True
                break

            shortcut = self._find_best_shortcut_with_kappa(
                edges_list, adj, edge_set, edge_index, kappa, current_min,
            )
            if shortcut is None:
                break
            a, b = shortcut
            # Apply incremental κ updates BEFORE mutating adj.
            self._apply_delta_to_kappa(
                kappa, edge_index, adj, a, b,
            )
            # Now mutate adj / edges_list / edge_set / edge_index.
            new_key = (min(a, b), max(a, b))
            edge_index[new_key] = len(edges_list)
            edges_list.append([a, b])
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add(new_key)
            n_added += 1

        # κ is maintained incrementally; the final min is direct.
        kappa_min_after = min(kappa) if kappa else float("inf")
        # Re-check convergence at the final state.
        if kappa_min_after >= self.min_kappa_target:
            converged = True

        new_edges_t = torch.tensor(edges_list, dtype=torch.long, device=device)

        # Signs for the combined edge list.
        if anchor_features is not None:
            u = new_edges_t[:, 0]
            v = new_edges_t[:, 1]
            dot = (anchor_features[u] * anchor_features[v]).sum(dim=-1)
            new_signs = torch.where(
                dot >= self.sign_threshold,
                torch.ones(new_edges_t.shape[0], dtype=torch.long, device=device),
                -torch.ones(new_edges_t.shape[0], dtype=torch.long, device=device),
            )
            # If the caller provided original edge signs, preserve them on
            # original edges; only the new shortcuts get freshly computed.
            if edge_signs is not None:
                final = torch.cat([
                    edge_signs.to(device),
                    new_signs[n_orig:],
                ])
            else:
                final = new_signs
        else:
            if edge_signs is not None:
                # Original signs + +1 default for new shortcuts.
                final = torch.cat([
                    edge_signs.to(device),
                    torch.ones(
                        new_edges_t.shape[0] - n_orig,
                        dtype=torch.long, device=device,
                    ),
                ])
            else:
                final = torch.ones(
                    new_edges_t.shape[0], dtype=torch.long, device=device,
                )

        return SDRFOutput(
            edges=new_edges_t,
            edge_signs=final,
            n_added=n_added,
            kappa_min_before=kappa_min_before,
            kappa_min_after=kappa_min_after,
            converged=converged,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _min_kappa(self, edges_list: list[list[int]], n: int) -> float:
        if not edges_list:
            return float("inf")
        edges_t = torch.tensor(edges_list, dtype=torch.long)
        out = self.forman(edges_t, n_nodes=n)
        return float(out.edge_kappa.min().item())

    def _find_best_shortcut_with_kappa(
        self,
        edges_list: list[list[int]],
        adj: list[set[int]],
        edge_set: set[tuple[int, int]],
        edge_index: dict[tuple[int, int], int],
        kappa: list[float],
        current_min: float,
    ) -> Optional[tuple[int, int]]:
        """Find the best shortcut, using a pre-computed κ array.

        Same algorithm as the legacy ``_find_best_shortcut`` but
        consumes the externally-maintained ``kappa`` array (avoiding
        a per-call Forman recompute).
        """
        sorted_e = sorted(range(len(kappa)), key=lambda i: kappa[i])

        best: Optional[tuple[int, int]] = None
        best_min_after = current_min - 1e-9
        for e_idx in sorted_e:
            u, v = edges_list[e_idx]
            adj_u = adj[u]
            adj_v = adj[v]
            for a in sorted(adj_u):
                if a == v:
                    continue
                for b in sorted(adj_v):
                    if b == u or a == b:
                        continue
                    key = (min(a, b), max(a, b))
                    if key in edge_set:
                        continue
                    new_min = self._delta_min_kappa_after_add(
                        adj, edge_index, kappa, a, b, current_min,
                    )
                    if new_min >= current_min and (
                        new_min > best_min_after
                        or (new_min == best_min_after
                            and best is not None and (a, b) < best)
                    ):
                        best_min_after = new_min
                        best = (a, b)
            if best is not None:
                return best
        return best

    @staticmethod
    def _apply_delta_to_kappa(
        kappa: list[float],
        edge_index: dict[tuple[int, int], int],
        adj: list[set[int]],
        a: int,
        b: int,
    ) -> None:
        """Apply the κ delta of adding edge (a, b) IN PLACE.

        Assumes adj has not yet been updated with the new edge.
        After this call, ``kappa`` contains the post-add κ values for
        all existing edges, and a new entry has been appended for the
        new (a, b) edge.
        """
        adj_a = adj[a]
        adj_b = adj[b]
        common = adj_a & adj_b
        # 1. Update κ of incident edges using OLD adj.
        for c in adj_a:
            # Excludes a → ... wait, c ∈ adj[a], so the edge (a, c)
            # exists. c can be anything in adj_a (b is not yet in adj_a
            # because we update adj AFTER this method returns).
            key = (min(a, c), max(a, c))
            e_idx = edge_index.get(key)
            if e_idx is None:
                continue
            kappa[e_idx] += 1 if c in adj_b else -1
        for d in adj_b:
            key = (min(b, d), max(b, d))
            e_idx = edge_index.get(key)
            if e_idx is None:
                continue
            kappa[e_idx] += 1 if d in adj_a else -1
        # 2. Append κ of the new edge itself.
        deg_a_new = len(adj_a) + 1
        deg_b_new = len(adj_b) + 1
        kappa.append(2.0 - deg_a_new - deg_b_new + 2.0 * len(common))

    @staticmethod
    def _delta_min_kappa_after_add(
        adj: list[set[int]],
        edge_index: dict[tuple[int, int], int],
        kappa: list[float],
        a: int,
        b: int,
        current_min: float,
    ) -> float:
        """O(deg(a) + deg(b)) recomputation of min κ after adding (a, b)."""
        adj_a = adj[a]
        adj_b = adj[b]
        common = adj_a & adj_b
        deg_a_new = len(adj_a) + 1
        deg_b_new = len(adj_b) + 1
        # κ of the new edge itself.
        new_kappa_ab = 2.0 - deg_a_new - deg_b_new + 2.0 * len(common)
        min_after = min(current_min, new_kappa_ab)
        # Edges incident to a (other than the new one): degree-of-a +1
        # adds −1 to κ; if c ∈ adj(b) the new triangle adds +2; net ±1.
        for c in adj_a:
            key = (min(a, c), max(a, c))
            e_idx = edge_index.get(key)
            if e_idx is None:
                continue
            delta = 1 if c in adj_b else -1
            new_k = kappa[e_idx] + delta
            if new_k < min_after:
                min_after = new_k
        # Symmetric for b's incident edges.
        for d in adj_b:
            key = (min(b, d), max(b, d))
            e_idx = edge_index.get(key)
            if e_idx is None:
                continue
            delta = 1 if d in adj_a else -1
            new_k = kappa[e_idx] + delta
            if new_k < min_after:
                min_after = new_k
        return min_after
