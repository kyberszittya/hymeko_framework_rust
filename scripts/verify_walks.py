"""Standalone correctness check for `hymeko.enumerate_k_walks_rs`.

Exercises:

  1. Canonical-form invariant: every emitted walk has
     ``path[0] <= path[walk_len]``.
  2. Cardinality on a known graph: hand-counted walks match the
     enumerator's output.
  3. No vertex revisits within a walk (simple-walk constraint).
  4. Reservoir cap honoured.
  5. Cross-check against a pure-Python reference DFS.

Run:
  python3 scripts/verify_walks.py
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np

import hymeko


def reference_walks(edges, n_nodes, walk_len):
    """Ground-truth pure-Python walk enumerator (small-graph only)."""
    adj = [set() for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    out = []

    def dfs(path):
        if len(path) == walk_len + 1:
            if path[0] <= path[-1]:
                out.append(tuple(path))
            return
        tail = path[-1]
        for nxt in sorted(adj[tail]):
            if nxt in path:
                continue
            path.append(nxt)
            dfs(path)
            path.pop()

    for s in range(n_nodes):
        dfs([s])
    return sorted(out)


def case(name: str, edges, n_nodes: int, walk_len: int):
    print(f"\n── {name}  (n={n_nodes}, walk_len={walk_len}) ──")
    eu = [u for u, v in edges]
    ev = [v for u, v in edges]
    arr = hymeko.enumerate_k_walks_rs(eu, ev, n_nodes, walk_len)
    walks = sorted(tuple(int(x) for x in row) for row in arr)
    expected = reference_walks(edges, n_nodes, walk_len)
    print(f"  rust-emitted: {len(walks)}    reference: {len(expected)}")

    # 1. canonical form
    bad = [w for w in walks if w[0] > w[-1]]
    assert not bad, f"canonical-form invariant violated: {bad[:3]}"

    # 2. simple-walk
    bad = [w for w in walks if len(set(w)) != len(w)]
    assert not bad, f"vertex-revisit detected: {bad[:3]}"

    # 3. against reference
    if walks != expected:
        only_rust = set(walks) - set(expected)
        only_ref  = set(expected) - set(walks)
        print(f"  ❌ MISMATCH  only-rust={list(only_rust)[:3]}  "
              f"only-ref={list(only_ref)[:3]}")
        return False
    print(f"  ✓ matches reference, canonical, simple")
    return True


def reservoir_check():
    print("\n── reservoir cap ──")
    edges = [(i, i + 1) for i in range(15)] + [(0, 8), (3, 11), (5, 12)]
    n = 16
    full = hymeko.enumerate_k_walks_rs([u for u, _ in edges],
                                          [v for _, v in edges], n, 4)
    print(f"  full count length-4: {full.shape[0]}")
    cap = 7
    reservoir = hymeko.enumerate_k_walks_rs([u for u, _ in edges],
                                               [v for _, v in edges],
                                               n, 4, cap, 1234)
    assert reservoir.shape == (cap, 5), f"bad reservoir shape: {reservoir.shape}"
    bad = [w.tolist() for w in reservoir if w[0] > w[-1]]
    assert not bad, f"reservoir broke canonical form: {bad}"
    print(f"  ✓ shape ({cap}, 5), all canonical")


def main():
    ok = True
    # 1. Triangle: only length-2 walks (3 vertices) are 0-1-2, 0-2-1
    ok &= case("triangle k=2", [(0, 1), (1, 2), (0, 2)], 3, 2)
    # 2. Path 0-1-2-3-4: walks of length 1, 2, 3, 4
    path = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for L in [1, 2, 3, 4]:
        ok &= case(f"path-5 length-{L}", path, 5, L)
    # 3. 6-cycle plus chord (the smoke-test case)
    six = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
    for L in [2, 3, 5]:
        ok &= case(f"6cycle+chord length-{L}", six, 6, L)
    # 4. K4 (complete on 4 vertices): all simple walks of any length
    k4 = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for L in [1, 2, 3]:
        ok &= case(f"K4 length-{L}", k4, 4, L)
    # 5. Reservoir
    reservoir_check()

    print()
    if ok:
        print("✅ all walk-enumerator cases passed")
        sys.exit(0)
    else:
        print("❌ at least one case failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
