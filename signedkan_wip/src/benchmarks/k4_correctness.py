"""Correctness benchmark for fast k=4 enumeration.

Asserts the five invariants from BENCHMARK_PLAN.md:
  1. cycle count matches reference
  2. cycle vertex-set matches (after canonicalisation)
  3. edge_signs match per cycle
  4. sigma assignment matches per cycle
  5. balanced flag matches per cycle

Tier A + B fixtures: karate, SBM-200, SBM-400, hier-SBM, bitcoin_alpha,
bitcoin_otc. These are the cells where the reference is tractable.
"""
from __future__ import annotations

import time

from ..datasets import load
from ..n_tuples import construct_k
from ..n_tuples_fast import construct_4_fast


TIER_A_B = [
    "karate",
    "sbm_n200_k4_s0",
    "sbm_n400_k5_s0",
    "hier_n240_s0",
    "bitcoin_alpha",
    "bitcoin_otc",
]


def _norm_v(v):
    """Canonical rotation/reflection — smallest-vertex-first, lex-smaller
    direction."""
    v = tuple(int(x) for x in v)
    n = len(v)
    rmin = min(range(n), key=lambda i: v[i])
    fwd = tuple(v[(rmin + j) % n] for j in range(n))
    rev = tuple(v[(rmin - j) % n] for j in range(n))
    return min(fwd, rev)


def verify_one(dataset: str) -> dict:
    g = load(dataset)
    t0 = time.time(); ref = construct_k(g, k=4); t_ref = time.time() - t0
    t0 = time.time(); fast = construct_4_fast(g); t_fast = time.time() - t0

    # 1. count
    count_ok = len(ref) == len(fast)

    # 2. vertex set
    ref_keys = {_norm_v(t.v) for t in ref}
    fast_keys = {_norm_v(t.v) for t in fast}
    set_ok = ref_keys == fast_keys
    only_ref = ref_keys - fast_keys
    only_fast = fast_keys - ref_keys

    # 3-5. per-cycle attribute match (over the intersection)
    ref_by_v = {_norm_v(t.v): t for t in ref}
    fast_by_v = {_norm_v(t.v): t for t in fast}
    common = ref_keys & fast_keys
    edge_signs_mismatch = 0
    sigma_mismatch = 0
    balanced_mismatch = 0
    for k in common:
        r = ref_by_v[k]; f = fast_by_v[k]
        # Edge signs are in cycle order; the canonical rotation may
        # rotate the cycle, so compare as multisets (signed multiset).
        if sorted(r.edge_signs) != sorted(f.edge_signs):
            edge_signs_mismatch += 1
        if sorted(r.sigma) != sorted(f.sigma):
            sigma_mismatch += 1
        if r.balanced != f.balanced:
            balanced_mismatch += 1

    return dict(
        dataset=dataset,
        ref_count=len(ref), fast_count=len(fast),
        count_ok=count_ok,
        set_ok=set_ok, only_ref=len(only_ref), only_fast=len(only_fast),
        edge_signs_mismatch=edge_signs_mismatch,
        sigma_mismatch=sigma_mismatch,
        balanced_mismatch=balanced_mismatch,
        ref_time_s=round(t_ref, 3),
        fast_time_s=round(t_fast, 3),
        speedup=round(t_ref / max(t_fast, 1e-3), 2),
    )


def main():
    print(f"=== k=4 correctness benchmark — {len(TIER_A_B)} fixtures ===\n")
    all_ok = True
    for ds in TIER_A_B:
        try:
            r = verify_one(ds)
        except Exception as e:
            print(f"  {ds:18s}  FAILED: {e!r}")
            all_ok = False
            continue
        invariants = [r["count_ok"], r["set_ok"],
                      r["edge_signs_mismatch"] == 0,
                      r["sigma_mismatch"] == 0,
                      r["balanced_mismatch"] == 0]
        ok = all(invariants)
        marker = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{marker}] {r['dataset']:18s}  "
              f"ref={r['ref_count']:>9,}  fast={r['fast_count']:>9,}  "
              f"only_ref={r['only_ref']:>3}  only_fast={r['only_fast']:>3}  "
              f"sig_diff={r['sigma_mismatch']:>3}  "
              f"bal_diff={r['balanced_mismatch']:>3}  "
              f"speedup={r['speedup']}x  ({r['ref_time_s']}s → {r['fast_time_s']}s)")
    print()
    print("=== overall: " + ("PASS" if all_ok else "FAIL") + " ===")


if __name__ == "__main__":
    main()
