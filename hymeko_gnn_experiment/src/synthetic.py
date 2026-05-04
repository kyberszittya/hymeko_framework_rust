"""HyMeKo vs. GNN experiment — synthetic hypergraph generator + labels.

Implements §3.1 of `docs/plans/plans_20260429/hymeko_gnn_experiment_design.md`.

Generates synthetic hypergraphs parameterised by (N_v, N_e, K_max,
signed). Computes ground-truth labels analytically for the four
WL-hard / WL-easy properties listed in §2.

The generator side-steps the `.hko` round-trip for the moment — that is
phase 0.A of the plan and lands as a follow-up. We emit incidence
matrices + labels directly, in a `.npz` shape both PyTorch Geometric
and HyMeKo can consume.

Run:
    python3 -m src.synthetic --out data/synth_n32_k5.npz \
                              --n-vertices 32 --n-hyperedges 32 \
                              --k-max 5 --n-samples 200 --signed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HypergraphSample:
    """One synthetic hypergraph: incidence matrix + ground-truth labels.

    `B` shape: (N_v, N_e). For unsigned, entries ∈ {0, 1}.
    For signed (G-SPHF), entries ∈ {-1, 0, +1}.

    Labels are dict[property_name → int (0/1)]; computed analytically
    by the generator.
    """
    B: np.ndarray
    labels: dict
    meta: dict


# ─── Property labels (§2) ────────────────────────────────────────────


def is_k_regular(B: np.ndarray, k: int) -> bool:
    """Every hyperedge has arity exactly k. WL-hard for k≥3."""
    arities = (B != 0).sum(axis=0)
    return bool((arities == k).all())


def has_triangle_subhypergraph(B: np.ndarray) -> bool:
    """Returns True iff the hypergraph contains a 3-cycle of size-2
    hyperedges. WL-hard test."""
    edges_size2 = [
        np.where(B[:, j] != 0)[0]
        for j in range(B.shape[1])
        if (B[:, j] != 0).sum() == 2
    ]
    if len(edges_size2) < 3:
        return False
    # Build adjacency from size-2 edges.
    adj = {}
    for u, v in edges_size2:
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))
    for u, neigh in adj.items():
        for v in neigh:
            if v <= u:
                continue
            common = (neigh - {v}) & adj.get(v, set())
            if any(w > v for w in common):
                return True
    return False


def n_components_after_random_removal(B: np.ndarray, seed: int = 0) -> int:
    """Component count of the underlying simple graph (clique expansion)
    after removing a random hyperedge. WL-hard."""
    rng = np.random.default_rng(seed)
    if B.shape[1] == 0:
        return B.shape[0]
    drop = int(rng.integers(0, B.shape[1]))
    B2 = np.delete(B, drop, axis=1)
    # Build clique-expansion adjacency from B2.
    n = B2.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            v = stack.pop()
            # Find hyperedges containing v.
            edges_v = np.where(B2[v, :] != 0)[0]
            for e in edges_v:
                members = np.where(B2[:, e] != 0)[0]
                for u in members:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(int(u))
    return components


def arity_distribution(B: np.ndarray) -> np.ndarray:
    """Histogram of hyperedge arities. WL-easy (sanity baseline)."""
    arities = (B != 0).sum(axis=0)
    if B.shape[1] == 0:
        return np.zeros(1, dtype=np.int64)
    bins = np.zeros(arities.max() + 1, dtype=np.int64)
    for a in arities:
        bins[a] += 1
    return bins


# ─── Generator ───────────────────────────────────────────────────────


def generate_one(rng: np.random.Generator,
                 n_vertices: int, n_hyperedges: int,
                 k_max: int, signed: bool) -> HypergraphSample:
    """One synthetic hypergraph + computed labels.

    Each hyperedge gets a uniform-random arity ∈ [2, k_max] and a
    uniform-random subset of vertices of that size. Signs (if signed)
    are 50/50 ± 1 per incidence.
    """
    B = np.zeros((n_vertices, n_hyperedges), dtype=np.int8)
    for j in range(n_hyperedges):
        arity = int(rng.integers(2, k_max + 1))
        members = rng.choice(n_vertices, size=arity, replace=False)
        if signed:
            signs = rng.choice([-1, 1], size=arity)
        else:
            signs = np.ones(arity, dtype=np.int8)
        for m, s in zip(members, signs):
            B[m, j] = s
    # Ground-truth labels.
    labels = dict(
        is_3_regular=int(is_k_regular(B, 3)),
        is_5_regular=int(is_k_regular(B, 5)),
        has_triangle=int(has_triangle_subhypergraph(B)),
        # n_components_after_removal as a binary "≥2" indicator
        n_components_ge2=int(n_components_after_random_removal(B) >= 2),
    )
    arity_hist = arity_distribution(B)
    meta = dict(
        n_vertices=n_vertices,
        n_hyperedges=n_hyperedges,
        k_max=k_max,
        signed=signed,
        arity_histogram=arity_hist.tolist(),
    )
    return HypergraphSample(B=B, labels=labels, meta=meta)


def generate_dataset(rng: np.random.Generator, n_samples: int,
                     n_vertices: int, n_hyperedges: int,
                     k_max: int, signed: bool) -> list[HypergraphSample]:
    return [
        generate_one(rng, n_vertices, n_hyperedges, k_max, signed)
        for _ in range(n_samples)
    ]


# ─── IO ──────────────────────────────────────────────────────────────


def save_npz(samples: list[HypergraphSample], out_path: Path) -> None:
    """Write all incidence matrices + labels to a single .npz.
    Uses object arrays for variable-shape Bs (here all the same shape,
    but kept flexible for future generator variants)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Bs = np.stack([s.B for s in samples], axis=0)
    labels_keys = sorted(samples[0].labels.keys())
    labels_arr = np.stack(
        [np.array([s.labels[k] for k in labels_keys], dtype=np.int8)
         for s in samples],
        axis=0,
    )
    np.savez_compressed(
        out_path,
        B=Bs,
        labels=labels_arr,
        labels_keys=np.array(labels_keys),
        meta=json.dumps(samples[0].meta),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-vertices", type=int, default=32)
    ap.add_argument("--n-hyperedges", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.n_samples} samples, "
          f"|V|={args.n_vertices}, |E|={args.n_hyperedges}, "
          f"k_max={args.k_max}, signed={args.signed}")
    samples = generate_dataset(
        rng, args.n_samples, args.n_vertices, args.n_hyperedges,
        args.k_max, args.signed,
    )
    save_npz(samples, args.out)
    # Per-property label distribution.
    print(f"Wrote {args.out}")
    keys = sorted(samples[0].labels.keys())
    print(f"\nLabel distributions across {args.n_samples} samples:")
    for k in keys:
        pos = sum(s.labels[k] for s in samples)
        print(f"  {k:<22}  pos={pos:>4}  ({pos / len(samples) * 100:.1f}%)")


if __name__ == "__main__":
    main()
