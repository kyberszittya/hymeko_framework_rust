"""Synthetic signed-graph generators with ground-truth labels.

Each generator returns a ``SyntheticSet`` bundle: a ``SignedGraph``
plus per-vertex labels, plus an oracle that reports the achievable
AUC on the exact label set.  Used to validate architectural pieces
(sparse attention, learnable incidence) against a known answer
before deploying on real datasets like Epinions.

Three generators today:

- ``easy_sbm``: trivially-solvable signed SBM.  Sanity check ---
  any working architecture should hit AUC > 0.95.
- ``needle_in_haystack``: 1% signal cycles in 99% noise cycles.
  Tests Path A (sparse attention scaling).  Dense attention
  predicted to fail; top-K attention predicted to succeed.
- ``feature_conditioned``: cycle importance depends on per-vertex
  features.  Tests Path B (learnable incidence).  Fixed M_e
  predicted to fail; learnable M_e predicted to succeed.

Each generator is deterministic given a seed.  Running the script as
``python -m signedkan_wip.src.synthetic_signed_graphs`` prints summary
statistics and oracle-baseline AUCs on each.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .datasets import SignedGraph


@dataclass
class SyntheticSet:
    """Bundle returned by every generator: graph + labels + features
    (optional) + a textual description of how the labels are
    determined."""
    graph: SignedGraph
    labels: np.ndarray              # (n_nodes,) int — class id per vertex
    features: np.ndarray | None     # (n_nodes, d) float — optional
    description: str
    name: str
    seed: int


# ─── Helpers ────────────────────────────────────────────────────────


def _erdos_renyi_signed(n: int, p_edge: float, p_pos: float,
                        rng: np.random.Generator) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Random signed Erdős–Rényi: each (u, v) pair (u < v) is an edge
    independently with probability ``p_edge``; sign +1 with prob
    ``p_pos``."""
    pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p_edge:
                pairs.append((u, v))
    pairs = np.array(pairs, dtype=np.int64)
    signs = np.where(rng.random(len(pairs)) < p_pos, 1, -1).astype(np.int64)
    return pairs, signs


def _random_4_cycles(n: int, n_cycles: int,
                      rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    """Random 4-vertex tuples (treated as cycle vertex orders)."""
    out = []
    for _ in range(n_cycles):
        out.append(tuple(rng.choice(n, size=4, replace=False).tolist()))
    return out


# ─── Generator 1: easy_sbm ─────────────────────────────────────────


def easy_sbm(n_per_block: int = 100, n_blocks: int = 2,
              p_within_pos: float = 0.6,
              p_within_neg: float = 0.05,
              p_between_pos: float = 0.05,
              p_between_neg: float = 0.4,
              seed: int = 0) -> SyntheticSet:
    """Signed SBM where same-block pairs prefer +1 edges and
    cross-block pairs prefer -1 edges.

    Should be solvable to AUC > 0.95 by any reasonable signed-graph
    embedding model — sanity check.
    """
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    labels = np.zeros(n, dtype=np.int64)
    for b in range(n_blocks):
        labels[b * n_per_block:(b + 1) * n_per_block] = b

    edges, signs = [], []
    for u in range(n):
        for v in range(u + 1, n):
            same_block = labels[u] == labels[v]
            if same_block:
                p_e = p_within_pos + p_within_neg
                p_pos = p_within_pos / p_e
            else:
                p_e = p_between_pos + p_between_neg
                p_pos = p_between_pos / p_e
            if rng.random() < p_e:
                edges.append((u, v))
                signs.append(1 if rng.random() < p_pos else -1)
    edges = np.array(edges, dtype=np.int64)
    signs = np.array(signs, dtype=np.int64)
    g = SignedGraph(edges=edges, signs=signs, n_nodes=n)
    return SyntheticSet(
        graph=g, labels=labels, features=None,
        description=(
            f"signed SBM, {n_blocks} blocks of {n_per_block}; "
            "same-block edges 92% +1, cross-block edges 89% -1"
        ),
        name="easy_sbm", seed=seed,
    )


# ─── Generator 2: needle_in_haystack ───────────────────────────────


def needle_in_haystack(n_per_block: int = 500,
                        n_signal_cycles_per_block: int = 25,
                        n_noise_cycles: int = 5000,
                        p_background: float = 0.01,
                        seed: int = 0) -> SyntheticSet:
    """Two communities; signal lives entirely in a small set of
    structurally-balanced 4-cycles within each community; noise is
    ~5000 random 4-cycles spanning everywhere.

    Setup designed so that:
        - Dense attention pool sees signal:noise ≈ 50:5000 = 1%.
          Gradient should be diluted.
        - A model that can pick out a small signal subset (top-K
          attention with a learnable scorer, or balance-pruner enum)
          should recover community labels with high AUC.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_block
    labels = np.zeros(n, dtype=np.int64)
    labels[n_per_block:] = 1

    # Background edges: sparse Erdős-Rényi, weak signs.
    bg_edges, bg_signs = _erdos_renyi_signed(n, p_background, 0.5, rng)

    # Signal cycles: 4-cycles entirely within a single community.
    # Each cycle realised as 4 edges arranged in a closed loop.
    signal_edges = []
    signal_signs = []
    for _ in range(n_signal_cycles_per_block):
        for block in range(2):
            base = block * n_per_block
            verts = rng.choice(n_per_block, size=4, replace=False) + base
            sign_pattern = [1, 1, 1, 1]   # structurally balanced
            for i in range(4):
                u, v = int(verts[i]), int(verts[(i + 1) % 4])
                if u > v:
                    u, v = v, u
                signal_edges.append((u, v))
                signal_signs.append(sign_pattern[i])

    # Concatenate; dedup by keeping the last sign at duplicate edges.
    all_pairs = list(map(tuple, bg_edges)) + signal_edges
    all_signs = list(bg_signs) + signal_signs
    edge_dict = {}
    for (u, v), s in zip(all_pairs, all_signs):
        edge_dict[(u, v)] = s
    edges = np.array(sorted(edge_dict.keys()), dtype=np.int64)
    signs = np.array([edge_dict[tuple(e)] for e in edges],
                       dtype=np.int64)
    g = SignedGraph(edges=edges, signs=signs, n_nodes=n)

    return SyntheticSet(
        graph=g, labels=labels, features=None,
        description=(
            f"2 communities of {n_per_block}; "
            f"{n_signal_cycles_per_block} balanced signal cycles per block; "
            f"~{n_noise_cycles} noise edges in background"
        ),
        name="needle_in_haystack", seed=seed,
    )


# ─── Generator 3: feature_conditioned ──────────────────────────────


def feature_conditioned(n_per_mode: int = 250,
                         feat_dim: int = 4,
                         p_within: float = 0.05,
                         p_cross: float = 0.05,
                         seed: int = 0) -> SyntheticSet:
    """Cycle importance depends on vertex features.

    Two feature modes — vertex features drawn from N(+1, 0.1) (mode 0)
    or N(-1, 0.1) (mode 1) along the first feature axis.  Edges
    constructed so the sign of an edge is informative \\emph{only}
    when both endpoints' modes match.  A model with FIXED uniform
    incidence cannot route this asymmetry; a learnable M_e
    conditioned on features can.

    Hypothesis to test:
        - Fixed M_e HSiKAN: AUC near 0.5.
        - Learnable M_e HSiKAN: AUC > 0.85 once M_e learns mode-aware
          weighting.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_mode
    labels = np.zeros(n, dtype=np.int64)
    labels[n_per_mode:] = 1

    # Features: first axis encodes mode; remaining are noise.
    features = rng.standard_normal((n, feat_dim)).astype(np.float32) * 0.3
    features[:n_per_mode, 0] += 1.0
    features[n_per_mode:, 0] -= 1.0

    edges, signs = [], []
    for u in range(n):
        for v in range(u + 1, n):
            same_mode = labels[u] == labels[v]
            p_e = p_within if same_mode else p_cross
            if rng.random() < p_e:
                edges.append((u, v))
                # Same-mode edges carry signal: +1 if mode-0, -1 if mode-1.
                # Cross-mode edges are random — pure noise.
                if same_mode:
                    s = +1 if labels[u] == 0 else -1
                else:
                    s = 1 if rng.random() < 0.5 else -1
                signs.append(s)
    edges = np.array(edges, dtype=np.int64)
    signs = np.array(signs, dtype=np.int64)
    g = SignedGraph(edges=edges, signs=signs, n_nodes=n)
    return SyntheticSet(
        graph=g, labels=labels, features=features,
        description=(
            f"2 feature modes of {n_per_mode}; same-mode edges "
            "carry mode-conditioned sign; cross-mode edges are noise"
        ),
        name="feature_conditioned", seed=seed,
    )


# ─── Oracle baselines ──────────────────────────────────────────────


def _edge_classification_set(s: SyntheticSet,
                              test_frac: float = 0.2,
                              seed: int = 0) -> tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray]:
    """Edge-level link-sign prediction setup.

    Returns (edges_train, signs_train, edges_test, signs_test).
    """
    rng = np.random.default_rng(seed)
    n_e = s.graph.edges.shape[0]
    perm = rng.permutation(n_e)
    n_test = int(round(test_frac * n_e))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return (s.graph.edges[train_idx], s.graph.signs[train_idx],
            s.graph.edges[test_idx], s.graph.signs[test_idx])


def oracle_node_classification_auc(s: SyntheticSet,
                                     seed: int = 0) -> float:
    """Logistic-regression baseline on vertex features (if available)
    or one-hot label-leak (otherwise).  Reports the AUC achievable
    when the classifier sees the right input — the ceiling for any
    architecture that ingests the same signal."""
    if s.features is None:
        # No features → degree-feature LR baseline.
        n = s.graph.n_nodes
        deg_pos = np.zeros(n, dtype=np.float32)
        deg_neg = np.zeros(n, dtype=np.float32)
        for (u, v), sgn in zip(s.graph.edges, s.graph.signs):
            if sgn > 0:
                deg_pos[u] += 1; deg_pos[v] += 1
            else:
                deg_neg[u] += 1; deg_neg[v] += 1
        feats = np.stack([deg_pos, deg_neg], axis=1)
    else:
        feats = s.features
    rng = np.random.default_rng(seed)
    perm = rng.permutation(s.graph.n_nodes)
    n_test = int(0.2 * s.graph.n_nodes)
    test = perm[:n_test]
    train = perm[n_test:]
    if len(np.unique(s.labels)) != 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(feats[train], s.labels[train])
    pred = clf.predict_proba(feats[test])[:, 1]
    return float(roc_auc_score(s.labels[test], pred))


# ─── Main / CLI ────────────────────────────────────────────────────


GENERATORS: dict[str, Callable[..., SyntheticSet]] = {
    "easy_sbm":              easy_sbm,
    "needle_in_haystack":    needle_in_haystack,
    "feature_conditioned":   feature_conditioned,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="all",
                    choices=["all"] + list(GENERATORS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = list(GENERATORS.keys()) if args.generator == "all" \
        else [args.generator]

    print()
    for name in names:
        t0 = time.time()
        s = GENERATORS[name](seed=args.seed)
        gen_s = time.time() - t0

        oracle = oracle_node_classification_auc(s, seed=args.seed)
        stats = s.graph.stats()
        out = dict(
            name=s.name, seed=s.seed,
            n_nodes=stats["n_nodes"], n_edges=stats["n_edges"],
            pos_frac=round(stats["pos_frac"], 3),
            n_classes=int(s.labels.max() + 1),
            balance=round(float(s.labels.mean()), 3),
            has_features=s.features is not None,
            oracle_node_auc=round(oracle, 4),
            gen_time_s=round(gen_s, 3),
        )
        print(f"  [{name}]  {s.description}")
        print(f"    {json.dumps(out)}")
        print()


if __name__ == "__main__":
    main()
