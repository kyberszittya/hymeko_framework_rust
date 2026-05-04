"""Run the GPU force-summation kernel on additional topology fixtures
beyond the three paper baselines: a sparse-arity and a dense-arity
synthetic graph, and one extended-scale fixture. Reports per-iter
time + density-sweep companion numbers so the paper can quote the
topology-axis sensitivity of the kernel.

Output (stdout):
    name | |V_L| | |E_L| | per-iter ms

Caller can paste the numbers into Table 3 of paper/kepaf_v1/.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

import numpy as np

LAYOUT_BIN = REPO / "target" / "release" / "examples" / "layout_from_json"


def run_layout(n_nodes: int, edges: list[list[int]], n_iter: int,
               seed: int = 0) -> float:
    payload = json.dumps({
        "n_nodes": n_nodes,
        "n_iter": n_iter,
        "seed": seed,
        "edges": edges,
    })
    proc = subprocess.run(
        [str(LAYOUT_BIN)], input=payload,
        capture_output=True, text=True, check=True,
    )
    out = json.loads(proc.stdout)
    return out["wall_ms"] / out["n_iter"]


def synth_levi(n_v: int, n_e: int, mean_arity: float, seed: int = 0):
    """Synthetic Levi-graph: |V_L| = n_v + n_e, edges from each
    hyperedge to mean_arity members. Returns (n_nodes, edges)."""
    rng = np.random.default_rng(seed)
    edges = []
    for j in range(n_e):
        e_id = n_v + j
        ar = max(2, int(rng.poisson(mean_arity)))
        members = rng.choice(n_v, size=ar, replace=False)
        for m in members:
            edges.append([e_id, int(m)])
    return n_v + n_e, edges


def main():
    if not LAYOUT_BIN.exists():
        raise SystemExit(f"missing binary: {LAYOUT_BIN}")

    fixtures = [
        # (label, n_v, n_e, mean_arity, n_iter)
        ("sparse-1e4   (arity 2)", 10_000, 25_000, 2.0,  50),
        ("dense-1e4    (arity 8)", 10_000, 25_000, 8.0,  50),
        ("synth-3e4    (arity 4)", 30_000, 75_000, 4.0,  20),
        ("synth-1e5    (arity 4)", 100_000, 250_000, 4.0, 10),
    ]
    print("== topology fixtures ==")
    print(f"{'fixture':30s} {'|V_L|':>7s} {'|E_L|':>8s} "
          f"{'per-iter (ms)':>14s}")
    for label, n_v, n_e, ar, n_iter in fixtures:
        n_nodes, edges = synth_levi(n_v, n_e, ar)
        t = run_layout(n_nodes, edges, n_iter=n_iter)
        print(f"{label:30s} {n_nodes:>7d} {len(edges):>8d} "
              f"{t:>14.3f}")


if __name__ == "__main__":
    main()
