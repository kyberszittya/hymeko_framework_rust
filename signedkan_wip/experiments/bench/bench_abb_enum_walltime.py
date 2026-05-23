"""Bench per-vertex top-m cycle enumeration walltime across three
variants on Slashdot k=4:

    1. ABB OFF (the non-bb filtered_batched binding)
    2. v1 start-ABB (lossy; the 2026-05-10 binding)
    3. Global-min ABB with fullness_gate=1.0 (strictly correct)

Reports the median of N runs after a warm-up pass per CLAUDE.md
benchmark stability rules. CPU only — no torch, no GPU.

Usage:
    python -m signedkan_wip.experiments.bench.bench_abb_enum_walltime
        --dataset slashdot --k 4 --m 128 --iters 5 --warmup 1

A pure-enumeration bench: no training, no model. The number that
matters is the (median, IQR, worst) tuple in seconds, plus output
cycle counts for sanity.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import hymeko
from signedkan_wip.src.datasets import load


def to_uv_signs(g) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(g.signs,        dtype=np.int8)
    return eu, ev, es, int(g.n_nodes)


def bench_one(fn, *args, iters: int, warmup: int):
    # Warm-up — first call may pay JIT / page-in cost.
    for _ in range(warmup):
        out = fn(*args)
        n_cycles = int(out[0].shape[0]) if hasattr(out[0], "shape") else len(out[0])
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return {
        "n_cycles": n_cycles,
        "median_s": statistics.median(times),
        "min_s":    min(times),
        "max_s":    max(times),
        "iqr_s":    statistics.quantiles(times, n=4)[2] - statistics.quantiles(times, n=4)[0]
                    if iters >= 4 else None,
        "n_iters":  iters,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="slashdot")
    ap.add_argument("--k", type=int, default=4, help="cycle length")
    ap.add_argument("--m", type=int, default=128, help="m_per_vertex cap")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--scorer", default="fraction_negative")
    ap.add_argument("--pruner", default="none")
    args = ap.parse_args()

    print(f"[load] {args.dataset} …")
    t0 = time.perf_counter()
    g = load(args.dataset)
    print(f"[load] |V|={g.n_nodes} |E|={len(g.edges)} "
          f"loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    eu, ev, es, n = to_uv_signs(g)

    def call(abb_mode: str, fullness_gate: float = 0.25):
        """Single-axis dispatch over the unified enumerator (Strategy
        refactor 2026-05-11 — see CLAUDE.md §6.5 #1)."""
        return hymeko.enumerate_cycles_rs(
            eu, ev, es, n, args.k, args.m,
            score_kind=args.scorer, pruner_kind=args.pruner,
            filter_kind="none", filter_min_degree=2,
            abb_mode=abb_mode, fullness_gate=fullness_gate,
        )

    print(f"[bench] k={args.k} m={args.m} iters={args.iters} warmup={args.warmup}",
          flush=True)

    results: dict[str, dict] = {}

    print("  variant=abb_off", flush=True)
    results["abb_off"] = bench_one(
        call, "none", iters=args.iters, warmup=args.warmup,
    )
    print(f"    median={results['abb_off']['median_s']:.2f}s "
          f"n_cycles={results['abb_off']['n_cycles']}", flush=True)

    print("  variant=v1_start (start-local ABB, lossy)", flush=True)
    results["v1_start"] = bench_one(
        call, "start_local", iters=args.iters, warmup=args.warmup,
    )
    print(f"    median={results['v1_start']['median_s']:.2f}s "
          f"n_cycles={results['v1_start']['n_cycles']}", flush=True)

    for gate in (1.0, 0.5, 0.25):
        key = f"global_g{gate}".replace(".", "")
        print(f"  variant=global ABB gate={gate}", flush=True)
        results[key] = bench_one(
            call, "global_min", gate,
            iters=args.iters, warmup=args.warmup,
        )
        print(f"    median={results[key]['median_s']:.2f}s "
              f"n_cycles={results[key]['n_cycles']}", flush=True)

    print(json.dumps({
        "dataset": args.dataset, "k": args.k, "m": args.m,
        "scorer": args.scorer, "pruner": args.pruner,
        "n_nodes": n, "n_edges": len(g.edges),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
