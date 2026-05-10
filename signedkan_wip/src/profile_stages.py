"""One-shot stage timer for Epinions production-config enumeration.

Runs each stage of the run_final_cell mixed-arity pipeline independently
(no model training) and prints wall time + cycle count per stage. The
goal: find which arity / call is responsible for the ~115 min/seed wall.

Usage:
    python signedkan_wip/profile_stages.py

Env vars: same as run_final_cell (HSIKAN_TOPK_MODE etc).
"""
from __future__ import annotations
import os
import sys
import time
import resource
import gc


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def timed(label: str, fn, *args, **kw):
    gc.collect()
    rss_before = peak_rss_gb()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    dt = time.perf_counter() - t0
    rss_after = peak_rss_gb()
    n = len(out) if hasattr(out, "__len__") else "n/a"
    print(f"  [{dt:7.2f}s]  {label:<40}  n={n}  peak_rss={rss_after:.2f} GB (+{rss_after-rss_before:.2f})", flush=True)
    return out, dt


def main():
    banner("Setup: import + load Epinions")
    from . import datasets, n_tuples, hyperedges, walks

    g, dt_load = timed("load Epinions graph", datasets.load, "epinions")
    print(f"    n_nodes={g.n_nodes}  n_edges={len(g.edges)}", flush=True)

    # Match run_adaptive_mv_5seed_2026_05_10.sh production-config env.
    os.environ.setdefault("HSIKAN_TOPK_MODE", "per_vertex")
    os.environ.setdefault("HSIKAN_TOPK_K", "128")
    os.environ.setdefault("HSIKAN_TOPK_PRUNER", "balance")
    os.environ.setdefault("HSIKAN_TOPK_SCORER", "fraction_negative")
    # Match run_final_cell.py:144 cap_dict construction at production scale.
    # k=2/3 = HSIKAN_MAX_K2/K3 (script sets 200k); k=4/5/6 = max_k4 (script
    # passes --max-k4 100000). Walks default to max_k4 too.
    cap_k4 = 100_000
    cap_k5 = 100_000   # NOT 200k — production uses cap_dict[5] = max_k4.
    cap_k2_k3 = 200_000
    cap_walk = 100_000

    banner("Per-arity enumeration stages (Epinions, production env)")

    # c2: handled by construct_2 (n_tuples)
    timed("c2 via n_tuples.construct_2", n_tuples.construct_2, g)

    # c3: TWO paths to compare.
    timed("c3 SLOW PATH (hyperedges.construct, pure Python)",
          hyperedges.construct, g)

    timed("c3 FAST PATH (n_tuples.construct_k k=3, Rust per_vertex)",
          n_tuples.construct_k, g, 3, cap_k2_k3, 0)

    # c4: the canonical Rust per_vertex path
    timed("c4 (n_tuples.construct_k k=4, Rust per_vertex)",
          n_tuples.construct_k, g, 4, cap_k4, 0)

    # c5: probably the longest of the cycle arities
    timed("c5 (n_tuples.construct_k k=5, Rust per_vertex)",
          n_tuples.construct_k, g, 5, cap_k5, 0)

    # w2 / w3 walks (production cap = max_k4 = 100k, not 200k)
    timed("w2 (walks.construct_walks L=2)",
          walks.construct_walks, g, 2, cap_walk, 0)

    timed("w3 (walks.construct_walks L=3)",
          walks.construct_walks, g, 3, cap_walk, 0)

    banner("Done")
    print(f"Final peak RSS: {peak_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
