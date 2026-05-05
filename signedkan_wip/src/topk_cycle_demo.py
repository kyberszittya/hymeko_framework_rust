"""Demonstrate axiom-aware top-K cycle enumeration on Slashdot/Epinions.

Compares three strategies:
  1. Full enumeration  (the existing hymeko.enumerate_k_cycles_rs path)
  2. Top-K by frustration  (hymeko.enumerate_top_k_cycles_signed_rs)
  3. Vertex-stratified top-m  (the recommended HSiKAN drop-in)

For each strategy: |cycles|, vertex coverage, edge coverage, wall-time,
and the implied |M_e| memory in bytes (uint32 cycles + float64 scores).

Usage:
    python topk_cycle_demo.py --dataset slashdot --k 3
    python topk_cycle_demo.py --dataset epinions --k 3 --top-n 1000
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import numpy as np

import hymeko

DATA = Path(__file__).resolve().parents[1] / "data"


def load_edges(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / f"{name}.txt"
    u, v, s = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                a, b, sg = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if a == b:
                continue
            u.append(a)
            v.append(b)
            s.append(1 if sg >= 0 else -1)
    return (
        np.array(u, dtype=np.uint32),
        np.array(v, dtype=np.uint32),
        np.array(s, dtype=np.int8),
    )


def induce_top_degree(u, v, s, top_n):
    """Return remapped (u', v', s', kept_idx) on top-N highest-degree."""
    deg = np.zeros(int(max(u.max(), v.max())) + 1, dtype=np.int64)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    keep = np.argsort(-deg)[:top_n]
    keep_set = np.zeros(len(deg), dtype=bool)
    keep_set[keep] = True
    mask = keep_set[u] & keep_set[v]
    u2, v2, s2 = u[mask], v[mask], s[mask]
    # Re-index to 0..top_n
    remap = -np.ones(len(deg), dtype=np.int64)
    remap[np.sort(keep)] = np.arange(top_n, dtype=np.int64)
    u3 = remap[u2].astype(np.uint32)
    v3 = remap[v2].astype(np.uint32)
    return u3, v3, s2, top_n


def m_e_bytes(n_cycles: int, k_len: int) -> int:
    """uint32 cycles + float64 scores."""
    return n_cycles * k_len * 4 + n_cycles * 8


def coverage(cycles: np.ndarray, n_v: int, edges_keys: set) -> tuple[float, float]:
    if cycles.size == 0:
        return 0.0, 0.0
    touched = np.zeros(n_v, dtype=bool)
    touched[cycles.flatten()] = True
    edge_set = set()
    for c in cycles:
        for j in range(len(c)):
            a, b = int(c[j]), int(c[(j + 1) % len(c)])
            edge_set.add((min(a, b), max(a, b)))
    overlap = len(edge_set & edges_keys)
    return touched.mean() * 100.0, overlap / max(len(edges_keys), 1) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["slashdot", "epinions"], required=True)
    ap.add_argument("--k", type=int, default=3, help="cycle length")
    ap.add_argument("--top-n", type=int, default=600,
                    help="restrict to top-N highest-degree vertices "
                         "(use 0 for the full graph; full graph is slow)")
    args = ap.parse_args()

    print(f"loading {args.dataset} …")
    u, v, s = load_edges(args.dataset)
    print(f"  raw edges: {len(u):,}    %neg: "
          f"{(s < 0).mean() * 100:.1f}%")

    if args.top_n > 0:
        print(f"inducing top-{args.top_n} subgraph …")
        u, v, s, n_nodes = induce_top_degree(u, v, s, args.top_n)
    else:
        n_nodes = int(max(u.max(), v.max())) + 1
    print(f"  subgraph: |V|={n_nodes:,}  |E|={len(u):,}")
    edge_keys = set((min(int(a), int(b)), max(int(a), int(b)))
                    for a, b in zip(u, v))
    print()

    # ── Full enumeration (existing path) ──
    print(f"─── full {args.k}-cycle enumeration ───")
    t0 = time.perf_counter()
    full = hymeko.enumerate_k_cycles_rs(
        u, v, n_nodes, args.k, max_cycles=None, seed=0,
        directed=False, early_stop=False, n_threads=None,
    )
    dt = time.perf_counter() - t0
    n_full = len(full)
    cv, ce = coverage(full, n_nodes, edge_keys)
    print(f"  cycles={n_full:,}  vertex={cv:.1f}%  edge={ce:.1f}%  "
          f"|M_e|={m_e_bytes(n_full, args.k) / 1e6:.1f} MB  time={dt:.2f}s")
    print()

    # ── Top-K by fraction_negative ──
    print(f"─── top-K by fraction_negative ───")
    print(f"  {'K':<10} {'kept':<10} {'%vert':<8} {'%edge':<8} "
          f"{'|M_e| MB':<10} {'time':<8}")
    for K in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        cs, ss = hymeko.enumerate_top_k_cycles_signed_rs(
            u, v, s, n_nodes, args.k, K, "fraction_negative",
        )
        dt = time.perf_counter() - t0
        cv, ce = coverage(cs, n_nodes, edge_keys)
        print(f"  {K:<10} {len(cs):<10,} {cv:<8.1f} {ce:<8.1f} "
              f"{m_e_bytes(len(cs), args.k) / 1e6:<10.2f} {dt:<8.2f}")
    print()

    # ── Vertex-stratified top-m ──
    print(f"─── vertex-stratified top-m by fraction_negative ───")
    print(f"  {'m/v':<6} {'kept':<10} {'%vert':<8} {'%edge':<8} "
          f"{'|M_e| MB':<10} {'speedup':<10} {'time':<8}")
    for m in [1, 4, 16, 64]:
        t0 = time.perf_counter()
        cs, ss = hymeko.enumerate_top_k_per_vertex_cycles_signed_rs(
            u, v, s, n_nodes, args.k, m, "fraction_negative",
        )
        dt = time.perf_counter() - t0
        cv, ce = coverage(cs, n_nodes, edge_keys)
        speedup = n_full / max(len(cs), 1)
        print(f"  {m:<6} {len(cs):<10,} {cv:<8.1f} {ce:<8.1f} "
              f"{m_e_bytes(len(cs), args.k) / 1e6:<10.2f} "
              f"{speedup:<10.1f}x {dt:<8.2f}")


if __name__ == "__main__":
    main()
