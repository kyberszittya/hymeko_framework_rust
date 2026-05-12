"""Micro-benchmark: v1 vertex prefilter on small signed graphs.

Measures wall time + cycle count for c4 (k=4) per-vertex enumeration
at production config (m=128, balance pruner, fraction_negative
scorer) across three filter strategies:

- "none"                    baseline (no filter)
- "degree" (min_deg=2)      drops leaves + isolated
- "compose:degree,triangle" drops leaves + vertices not in any triangle

Datasets: bitcoin_alpha (3.8k vertices), bitcoin_otc (5.9k vertices).
Both small enough that enumeration is sub-second; the comparison shows
whether the filter has measurable overhead AND whether the cycle pool
is reduced (which it will be if the filter actually fires).

Usage:
    python -m signedkan_wip.src.bench_vertex_filter
"""
from __future__ import annotations
import os
import time
import statistics
import gc

import numpy as np
import hymeko

from . import datasets


N_REPEATS = 5            # 5-iter medians per CLAUDE.md §3
N_WARMUP = 2             # warm-up before measuring (cache + JIT)
K_LEN = 4
M_PER_VERTEX = 128
SCORER = "fraction_negative"
PRUNER = "balance"


def time_run(label: str, fn) -> dict:
    """Run fn n_warmup + n_repeats times; return median + IQR."""
    samples = []
    n_cycles = 0
    for i in range(N_WARMUP + N_REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        n_cycles = fn()
        dt = time.perf_counter() - t0
        if i >= N_WARMUP:
            samples.append(dt)
    return {
        "label": label,
        "n_cycles": n_cycles,
        "median": statistics.median(samples),
        "iqr": (max(samples) - min(samples)),
        "samples": samples,
    }


def bench_dataset(name: str) -> None:
    print(f"\n=== {name} ===")
    g = datasets.load(name)
    print(f"  n_nodes={g.n_nodes}  n_edges={len(g.edges)}")

    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(g.signs, dtype=np.int8)

    # Strategy refactor 2026-05-11 (CLAUDE.md §6.5 #1):
    # both branches dispatch through the unified enumerate_cycles_rs.
    def run_filter(filter_kind: str = "none", min_deg: int = 2):
        arr, _ = hymeko.enumerate_cycles_rs(
            eu, ev, es, g.n_nodes, K_LEN, M_PER_VERTEX,
            score_kind=SCORER, pruner_kind=PRUNER,
            filter_kind=filter_kind, filter_min_degree=min_deg,
        )
        return arr.shape[0]

    run_unfiltered = lambda: run_filter("none")
    run_filtered = run_filter

    results = [
        time_run("baseline (unfiltered binding)", run_unfiltered),
        time_run("filter=none (same path, baseline)",
                 lambda: run_filtered("none")),
        time_run("filter=degree min_deg=2",
                 lambda: run_filtered("degree", 2)),
        time_run("filter=triangle",
                 lambda: run_filtered("triangle")),
        time_run("filter=compose:degree,triangle",
                 lambda: run_filtered("compose:degree,triangle", 2)),
    ]

    # Print table
    base_med = results[0]["median"]
    base_n = results[0]["n_cycles"]
    print(f"\n  {'config':<40s}  {'median (ms)':>12s}  {'iqr (ms)':>10s}  "
          f"{'speedup':>8s}  {'cycles':>8s}  {'%cycles':>8s}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        speedup = base_med / r["median"] if r["median"] > 0 else float("nan")
        cyc_pct = 100.0 * r["n_cycles"] / max(1, base_n)
        print(f"  {r['label']:<40s}  {r['median']*1000:>12.2f}  "
              f"{r['iqr']*1000:>10.2f}  {speedup:>7.2f}x  "
              f"{r['n_cycles']:>8d}  {cyc_pct:>7.1f}%")


def main() -> None:
    print(f"Vertex-prefilter v1 micro-benchmark")
    print(f"  n_warmup={N_WARMUP}  n_repeats={N_REPEATS}")
    print(f"  k_len={K_LEN}  m_per_vertex={M_PER_VERTEX}")
    print(f"  scorer={SCORER}  pruner={PRUNER}")

    for ds in ("bitcoin_alpha", "bitcoin_otc", "slashdot"):
        bench_dataset(ds)


if __name__ == "__main__":
    main()
