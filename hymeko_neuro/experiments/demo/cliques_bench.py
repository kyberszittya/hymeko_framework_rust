"""Benchmark harness for balanced-clique detectors.

Four detectors share a Strategy interface (``Detector.detect``):

  - ``BronKerboschDetector`` — exact, NetworkX-backed, exponential
    worst case. Treated as ground truth on networks where it
    completes inside the timeout.
  - ``TriangleDensityDetector`` — rank vertices by triangle-balance
    score, expand greedy seeds. Poly.
  - ``GreedyBalancedDetector`` — high-degree seed + greedy growth
    along σ-product = +1 edges. Poly.
  - ``SpectralBalancedDetector`` — signed-Laplacian eigenvectors
    cluster vertices, then balance-check each cluster. Poly.

A fifth slot is reserved for ``GombDetector`` (Stage-2 of the NP-hard
plan) — adds Gömb-trained vertex embeddings + greedy expansion. Not
implemented here; lives in the NP-hard pivot plan.

Performance is measured wall-time + recall against the planted ground
truth via Jaccard overlap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cliques import Clique, _clique_balance_indicator
from .cliques_planted import PlantedRobotNetworkBundle


@dataclass
class DetectorResult:
    """One detector's output on one network."""

    detector_name: str
    cliques: list[Clique]
    wall_time_s: float
    timed_out: bool = False
    error: str | None = None

    @property
    def largest_size(self) -> int:
        return max((c.size for c in self.cliques), default=0)


class Detector(Protocol):
    """Strategy interface — every detector implements ``.detect``."""

    name: str

    def detect(
        self,
        bundle: PlantedRobotNetworkBundle,
        min_size: int,
        max_size: int,
        limit: int,
    ) -> list[Clique]:
        ...


# ─── Detector implementations ──────────────────────────────────────


class BronKerboschDetector:
    """Exact via NetworkX ``find_cliques`` + balance verification.

    Exponential worst case in the number of maximal cliques. Bounded by
    ``find_cliques`` itself, which can blow up on dense graphs. The
    benchmark harness applies a timeout *around* the detector call.
    """

    name = "bron_kerbosch_exact"

    def detect(self, bundle, min_size, max_size, limit):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(bundle.n_robots))
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
            valid = True
            for i in range(size):
                for j in range(i + 1, size):
                    a, b = members[i], members[j]
                    s = sign_of.get((a, b))
                    if s is None:
                        valid = False
                        break
                    edges.append((a, b))
                    signs.append(s)
                if not valid:
                    break
            if not valid:
                continue
            sigma = _clique_balance_indicator(members, sign_of)
            if sigma == 1:
                out.append(Clique(
                    members=members, edges=edges,
                    signs=signs, sigma_product=sigma,
                ))
        out.sort(key=lambda c: (-c.size, c.members))
        return out[:limit]


def _build_sign_lookup(bundle):
    """Return ``(sign_of, neighbours)``: sign of every edge + per-vertex
    adjacency for the underlying *unsigned* graph."""
    n = bundle.n_robots
    neighbours: list[set[int]] = [set() for _ in range(n)]
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
        neighbours[a].add(b)
        neighbours[b].add(a)
    return sign_of, neighbours


def _verify_balanced_clique(members: tuple[int, ...],
                              sign_of: dict[tuple[int, int], int]
                              ) -> Clique | None:
    """Check that ``members`` forms a balanced clique, return as
    ``Clique`` or ``None`` if not a clique or not balanced.

    Balance is the triangle-product check, NOT all-edges-product (see
    ``cliques._clique_balance_indicator``).
    """
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            s = sign_of.get((a, b))
            if s is None:
                return None
            edges.append((a, b))
            signs.append(s)
    sigma = _clique_balance_indicator(members, sign_of)
    if sigma != 1:
        return None
    return Clique(members=members, edges=edges,
                    signs=signs, sigma_product=sigma)


class TriangleDensityDetector:
    """Rank vertices by triangle-balance score; greedily expand seeds.

    Score(v) = number of positive triangles incident on v − number of
    negative triangles. High-scoring vertices are likely centres of
    balanced cliques. For each seed in descending score order, greedily
    add neighbours that keep the clique balanced.
    """

    name = "triangle_density_greedy"

    def detect(self, bundle, min_size, max_size, limit):
        sign_of, neighbours = _build_sign_lookup(bundle)
        n = bundle.n_robots

        # Per-vertex triangle-balance score.
        score = np.zeros(n, dtype=np.int64)
        for v in range(n):
            for a in neighbours[v]:
                for b in neighbours[v]:
                    if a >= b:
                        continue
                    if a not in neighbours[b]:
                        continue
                    s = (sign_of[(min(v, a), max(v, a))] *
                         sign_of[(min(v, b), max(v, b))] *
                         sign_of[(min(a, b), max(a, b))])
                    score[v] += 1 if s == 1 else -1

        out: list[Clique] = []
        seen: set[tuple[int, ...]] = set()
        for seed in np.argsort(-score):
            if len(out) >= limit:
                break
            current = {int(seed)}
            # Candidates: neighbours of all current members.
            while True:
                cand = set.intersection(
                    *[neighbours[m] for m in current]) - current
                if not cand:
                    break
                # Pick the candidate that, when added, keeps balance.
                best = None
                best_score = -10**9
                for w in cand:
                    test_members = tuple(sorted(current | {w}))
                    clique = _verify_balanced_clique(test_members, sign_of)
                    if clique is None:
                        continue
                    # Tie-break by score.
                    if int(score[w]) > best_score:
                        best = w
                        best_score = int(score[w])
                if best is None or len(current) + 1 > max_size:
                    break
                current.add(best)
            members = tuple(sorted(current))
            if len(members) >= min_size and members not in seen:
                clique = _verify_balanced_clique(members, sign_of)
                if clique is not None:
                    out.append(clique)
                    seen.add(members)
        out.sort(key=lambda c: (-c.size, c.members))
        return out[:limit]


class GreedyBalancedDetector:
    """High-degree seed + balance-preserving growth.

    Simpler than TriangleDensityDetector — uses degree alone for
    seeding. Useful as a sanity floor: if this detector matches
    triangle-density, the extra triangle-counting buys nothing.
    """

    name = "greedy_balanced"

    def detect(self, bundle, min_size, max_size, limit):
        sign_of, neighbours = _build_sign_lookup(bundle)
        n = bundle.n_robots
        deg = np.array([len(neighbours[v]) for v in range(n)],
                          dtype=np.int64)

        out: list[Clique] = []
        seen: set[tuple[int, ...]] = set()
        for seed in np.argsort(-deg):
            if len(out) >= limit:
                break
            current = {int(seed)}
            while True:
                cand = set.intersection(
                    *[neighbours[m] for m in current]) - current
                if not cand:
                    break
                # Pick any candidate that preserves balance.
                added = False
                for w in sorted(cand, key=lambda x: -int(deg[x])):
                    test_members = tuple(sorted(current | {w}))
                    if _verify_balanced_clique(test_members, sign_of) is None:
                        continue
                    current.add(w)
                    added = True
                    if len(current) >= max_size:
                        break
                    break
                if not added:
                    break
            members = tuple(sorted(current))
            if len(members) >= min_size and members not in seen:
                clique = _verify_balanced_clique(members, sign_of)
                if clique is not None:
                    out.append(clique)
                    seen.add(members)
        out.sort(key=lambda c: (-c.size, c.members))
        return out[:limit]


class SpectralBalancedDetector:
    """Signed-Laplacian eigenvectors → clusters → balance check.

    Builds the signed adjacency / signed Laplacian (Kunegis et al.
    2010), takes the top-K negative eigenvectors as cluster
    coordinates, runs k-means, and tests each resulting cluster for
    balance. Approximate; will miss small cliques.
    """

    name = "spectral_balanced"

    def __init__(self, n_clusters_max: int = 6):
        self.n_clusters_max = n_clusters_max

    def detect(self, bundle, min_size, max_size, limit):
        from sklearn.cluster import KMeans
        sign_of, neighbours = _build_sign_lookup(bundle)
        n = bundle.n_robots
        if n < min_size:
            return []

        # Signed adjacency.
        A = np.zeros((n, n), dtype=np.float64)
        for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
            A[int(u), int(v)] = float(s)
            A[int(v), int(u)] = float(s)
        # Signed Laplacian: L = D̄ - A, where D̄ uses |A|.
        D_bar = np.diag(np.abs(A).sum(axis=1))
        L = D_bar - A
        # Symmetric, take real eigendecomposition.
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return []
        # Take the bottom-k eigenvectors as cluster coords.
        k_use = min(self.n_clusters_max, n)
        coords = eigvecs[:, :k_use]
        try:
            km = KMeans(n_clusters=min(k_use, n), n_init=5,
                          random_state=bundle.seed)
            labels = km.fit_predict(coords)
        except Exception:
            return []

        out: list[Clique] = []
        for cluster_id in np.unique(labels):
            members_raw = tuple(int(i) for i, lb in enumerate(labels)
                                  if lb == cluster_id)
            if len(members_raw) < min_size or len(members_raw) > max_size:
                continue
            # Reduce to the largest balanced clique inside this cluster.
            # Greedy: start from the highest-degree vertex of the
            # cluster, expand within the cluster only.
            cluster_set = set(members_raw)
            cluster_deg = sorted(
                cluster_set,
                key=lambda v: -len(neighbours[v] & cluster_set),
            )
            for seed in cluster_deg:
                current = {seed}
                while True:
                    cand = (set.intersection(
                        *[neighbours[m] for m in current])
                              & cluster_set) - current
                    if not cand:
                        break
                    added = False
                    for w in cand:
                        test_members = tuple(sorted(current | {w}))
                        if _verify_balanced_clique(
                                test_members, sign_of) is None:
                            continue
                        current.add(w)
                        added = True
                        if len(current) >= max_size:
                            break
                        break
                    if not added:
                        break
                members = tuple(sorted(current))
                if len(members) >= min_size:
                    clique = _verify_balanced_clique(members, sign_of)
                    if clique is not None:
                        out.append(clique)
                        break
        # Deduplicate + sort.
        seen: set[tuple[int, ...]] = set()
        unique: list[Clique] = []
        for c in out:
            if c.members in seen:
                continue
            seen.add(c.members)
            unique.append(c)
        unique.sort(key=lambda c: (-c.size, c.members))
        return unique[:limit]


# ─── Benchmark + recall harness ────────────────────────────────────


def jaccard_overlap(a: Clique, b: Clique) -> float:
    """Jaccard similarity between two clique member sets."""
    sa, sb = set(a.members), set(b.members)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def recall_against_planted(
    detected: list[Clique],
    planted: list[Clique],
    overlap_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Match each planted clique to the best-overlapping detected one.

    Recall: fraction of planted cliques matched at overlap ≥ threshold.
    Precision: fraction of detected cliques that match SOME planted one.
    """
    if not planted:
        return {"recall": float("nan"), "precision": float("nan"),
                "n_planted": 0, "n_detected": len(detected),
                "n_matched": 0}
    matched = 0
    for p in planted:
        best = max((jaccard_overlap(p, d) for d in detected),
                   default=0.0)
        if best >= overlap_threshold:
            matched += 1
    precision = 0
    if detected:
        for d in detected:
            best = max((jaccard_overlap(p, d) for p in planted),
                       default=0.0)
            if best >= overlap_threshold:
                precision += 1
    return {
        "recall": matched / len(planted),
        "precision": (precision / len(detected)) if detected else 0.0,
        "n_planted": len(planted),
        "n_detected": len(detected),
        "n_matched": matched,
    }


def benchmark_detector(
    detector: Detector,
    bundle: PlantedRobotNetworkBundle,
    min_size: int = 3,
    max_size: int = 8,
    limit: int = 20,
    timeout_s: float = 60.0,
) -> DetectorResult:
    """Run one detector against one bundle and time it.

    The ``timeout_s`` is enforced via wall-clock comparison after the
    call returns — Python doesn't preempt cleanly inside C extensions,
    so a runaway Bron-Kerbosch may exceed the budget. We record the
    wall-time honestly and mark ``timed_out`` if it ran over.
    """
    t0 = time.perf_counter()
    try:
        cliques = detector.detect(bundle, min_size, max_size, limit)
        elapsed = time.perf_counter() - t0
        return DetectorResult(
            detector_name=detector.name,
            cliques=cliques,
            wall_time_s=elapsed,
            timed_out=elapsed > timeout_s,
        )
    except Exception as e:
        return DetectorResult(
            detector_name=detector.name,
            cliques=[],
            wall_time_s=time.perf_counter() - t0,
            error=repr(e),
        )


def default_detectors() -> list[Detector]:
    return [
        BronKerboschDetector(),
        TriangleDensityDetector(),
        GreedyBalancedDetector(),
        SpectralBalancedDetector(),
    ]


__all__ = [
    "BronKerboschDetector",
    "Detector",
    "DetectorResult",
    "GreedyBalancedDetector",
    "SpectralBalancedDetector",
    "TriangleDensityDetector",
    "benchmark_detector",
    "default_detectors",
    "jaccard_overlap",
    "recall_against_planted",
]
