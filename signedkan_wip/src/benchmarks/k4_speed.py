"""Speed benchmark for k=4 enumeration.

Tier A + B (correctness-verified): runs reference + fast paths and
reports speedup. Tier C (slashdot, epinions): runs fast path only;
reference is impractical (>30 min in pure Python).

Output JSON includes wall-clock and (when available) peak memory
for both paths, suitable for the paper's speed table.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

from ..datasets import load
from ..n_tuples import construct_k
from ..n_tuples_fast import construct_4_fast_arrays


TIERS = {
    "A_unit": ["karate"],
    "B_correctness": [
        "sbm_n200_k4_s0",
        "sbm_n400_k5_s0",
        "hier_n240_s0",
        "bitcoin_alpha",
        "bitcoin_otc",
    ],
    "C_scale": ["slashdot"],   # epinions is even bigger; opt-in
}


def time_one(fn) -> tuple[float, float, object]:
    """Run fn, return (wall_seconds, peak_mb, result)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    t = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return t, peak / 1e6, out


def bench_one(dataset: str, run_reference: bool) -> dict:
    g = load(dataset)
    rec = dict(dataset=dataset, n_nodes=g.n_nodes, n_edges=len(g.edges))

    if run_reference:
        try:
            t_ref, mem_ref, ref = time_one(lambda: construct_k(g, k=4))
            rec.update(
                ref_time_s=round(t_ref, 3),
                ref_peak_mb=round(mem_ref, 1),
                ref_count=len(ref),
            )
        except MemoryError:
            rec.update(ref_time_s=None, ref_peak_mb=None, ref_count=None,
                        ref_error="MemoryError")

    try:
        t_fast, mem_fast, (cv, *_) = time_one(
            lambda: construct_4_fast_arrays(g)
        )
        rec.update(
            fast_time_s=round(t_fast, 3),
            fast_peak_mb=round(mem_fast, 1),
            fast_count=int(cv.shape[0]),
        )
    except (MemoryError, RuntimeError) as e:
        rec.update(fast_time_s=None, fast_peak_mb=None,
                    fast_count=None, fast_error=repr(e))

    if (rec.get("ref_time_s") is not None
            and rec.get("fast_time_s") is not None
            and rec.get("fast_time_s") > 0):
        rec["speedup_x"] = round(rec["ref_time_s"] / rec["fast_time_s"], 2)
        rec["count_match"] = (rec["ref_count"] == rec["fast_count"])

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+",
                    default=["A_unit", "B_correctness", "C_scale"])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/k4_speed_benchmark.json")
    args = ap.parse_args()

    rows: list[dict] = []
    print(f"=== k=4 speed benchmark ===\n")
    for tier in args.tiers:
        if tier not in TIERS:
            print(f"  unknown tier: {tier}")
            continue
        run_reference = tier in ("A_unit", "B_correctness")
        print(f"-- Tier {tier} ({'reference + fast' if run_reference else 'fast only'}) --")
        for ds in TIERS[tier]:
            print(f"  running {ds}...")
            r = bench_one(ds, run_reference=run_reference)
            r["tier"] = tier
            rows.append(r)
            ref_s = (f"ref={r.get('ref_time_s')}s ({r.get('ref_count', '?'):,})"
                     if r.get('ref_time_s') is not None else "ref=skip")
            fast_s = (f"fast={r.get('fast_time_s')}s ({r.get('fast_count', '?'):,})"
                      if r.get('fast_time_s') is not None else "fast=FAILED")
            speedup = (f"  {r.get('speedup_x')}x"
                       if r.get('speedup_x') is not None else "")
            print(f"    {ds:18s}  {ref_s:>30s}  {fast_s:>30s}{speedup}")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
