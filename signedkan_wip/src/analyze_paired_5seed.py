"""Paired 5-seed AUC summary for the overnight 2026-05-11 queue.

Reads `reports/overnight_2026_05_11_stage5/epinions_baseline_s{0..4}.json`
and `epinions_kitchen_sink_s{0..4}.json`, computes per-seed delta,
mean, paired SE, t-statistic, and a sign-test.  Prints a compact
table and the paired stats line a paper would want.

Usage:
    python -m signedkan_wip.src.analyze_paired_5seed \
        --dir reports/overnight_2026_05_11_stage5 \
        --baseline epinions_baseline \
        --treatment epinions_kitchen_sink

Designed for the 2026-05-11 walks-augmented story (kitchen-sink config:
walks c3,c4,w2,w3 + CPG-3 + g10 ABB + h=32). Generalises to any other
A vs B paired comparison via the --baseline / --treatment prefixes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load_auc(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last = ""
    for line in path.read_text().strip().splitlines():
        last = line
    try:
        d = json.loads(last)
        return float(d.get("auc"))
    except Exception:
        return None


def paired_stats(baseline: list[float],
                   treatment: list[float]) -> dict[str, float]:
    """Return dict with mean_A, mean_B, mean_delta, sd_delta,
    paired_se, paired_t, n, plus sign-test p-value."""
    assert len(baseline) == len(treatment), "lengths must match"
    n = len(baseline)
    deltas = [b - a for a, b in zip(baseline, treatment)]
    mean_a = statistics.mean(baseline)
    mean_b = statistics.mean(treatment)
    mean_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if n >= 2 else 0.0
    paired_se = sd_d / math.sqrt(n) if n >= 2 else float("nan")
    paired_t = mean_d / paired_se if paired_se > 0 else float("inf")
    # Two-sided sign test (binomial)
    n_pos = sum(1 for d in deltas if d > 0)
    # P(X >= n_pos | H0: p=0.5) — exact small-N
    sign_p = sum(
        math.comb(n, k) * 0.5 ** n
        for k in range(min(n_pos, n - n_pos), -1, -1)
    ) + sum(
        math.comb(n, k) * 0.5 ** n
        for k in range(max(n_pos, n - n_pos), n + 1)
    )
    return {
        "n": n,
        "mean_baseline": mean_a,
        "mean_treatment": mean_b,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "paired_se": paired_se,
        "paired_t": paired_t,
        "n_pos_delta": n_pos,
        "sign_test_p_two_sided": sign_p,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="reports/overnight_2026_05_11_stage5",
        help="Directory containing the per-seed JSON results",
    )
    ap.add_argument(
        "--baseline", default="epinions_baseline",
        help="Filename prefix for baseline runs (suffix _s{seed}.json)",
    )
    ap.add_argument(
        "--treatment", default="epinions_kitchen_sink",
        help="Filename prefix for treatment runs",
    )
    ap.add_argument(
        "--seeds", default="0,1,2,3,4",
        help="Comma-separated seed list",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    log_dir = Path(args.dir)
    a_aucs: list[float] = []
    b_aucs: list[float] = []
    missing: list[int] = []
    for seed in seeds:
        a = load_auc(log_dir / f"{args.baseline}_s{seed}.json")
        b = load_auc(log_dir / f"{args.treatment}_s{seed}.json")
        if a is None or b is None:
            missing.append(seed)
            continue
        a_aucs.append(a)
        b_aucs.append(b)

    if missing:
        print(f"[warn] missing seeds: {missing}")
    if not a_aucs:
        print("[error] no complete pairs found")
        return

    # Per-seed table
    print(f"\n{'seed':>5}  {args.baseline:>32}  {args.treatment:>32}  {'delta':>10}")
    print("-" * 95)
    for seed, a, b in zip(seeds, a_aucs, b_aucs):
        print(f"{seed:>5}  {a:>32.4f}  {b:>32.4f}  {b - a:>+10.4f}")

    s = paired_stats(a_aucs, b_aucs)
    print()
    print(f"n = {s['n']}, mean baseline = {s['mean_baseline']:.4f}, "
          f"mean treatment = {s['mean_treatment']:.4f}")
    print(f"paired mean delta = {s['mean_delta']:+.4f}  "
          f"(sd of delta = {s['sd_delta']:.4f})")
    print(f"paired SE = {s['paired_se']:.4f}  "
          f"paired t = {s['paired_t']:+.2f}")
    print(f"sign test: {s['n_pos_delta']}/{s['n']} positive, "
          f"two-sided p ≈ {s['sign_test_p_two_sided']:.4f}")
    print()
    # Compact one-liner for paper / memory
    print(
        f"PAIRED: {args.treatment} {s['mean_treatment']:.4f} ± {s['sd_delta']:.4f}  "
        f"vs {args.baseline} {s['mean_baseline']:.4f};  "
        f"delta = {s['mean_delta']:+.4f}  (paired σ = {s['paired_t']:+.2f}, n = {s['n']})"
    )


if __name__ == "__main__":
    main()
