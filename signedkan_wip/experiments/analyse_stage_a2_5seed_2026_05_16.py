"""Aggregator for the 2026-05-16 HyMeYOLO Stage A-2 5-seed run.

Reads `+ricci-mod` rows from the Stage A-2 results dir and the
Stage A-1 (warm-start) results dir, prints per-seed table, paired-Δ,
and verdict against the pre-registered criterion
(paired mean ≥ 0.03 AND z ≥ 2).

Usage:
    python -m signedkan_wip.experiments.analyse_stage_a2_5seed_2026_05_16 \\
        signedkan_wip/experiments/results/hymeyolo_stage_a2_5seed_<STAMP>/ \\
        signedkan_wip/experiments/results/hymeyolo_warmstart_5seed_20260516T101835Z/
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def _load_map50_by_seed(d: Path) -> dict[int, float]:
    """Each .jsonl line is one config row; we filter to `+ricci-mod`."""
    out = {}
    for f in sorted(d.glob("*.jsonl")):
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("label") == "+ricci-mod":
                out[int(r["seed"])] = float(r.get("mAP_50", 0.0))
    return out


def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _verdict(mean_d: float, sd_d: float, n: int) -> str:
    if n < 2 or sd_d == 0:
        return "n/a"
    z = mean_d / (sd_d / math.sqrt(n))
    if mean_d > 0.03 and z >= 2.0:
        return f"WIN   mean Δ={mean_d:+.4f}  z={z:+.2f}"
    if mean_d < -0.03 and z <= -2.0:
        return f"LOSS  mean Δ={mean_d:+.4f}  z={z:+.2f}"
    return f"tie   mean Δ={mean_d:+.4f}  z={z:+.2f}"


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: analyse_stage_a2_5seed_2026_05_16.py "
              "<stage-a2-dir> <stage-a1-dir>")
        sys.exit(2)
    a2 = _load_map50_by_seed(Path(sys.argv[1]))
    a1 = _load_map50_by_seed(Path(sys.argv[2]))
    if not a2:
        print(f"no rows in {sys.argv[1]}")
        sys.exit(1)

    print(f"# Stage A-2 vs Stage A-1 (paired by seed)")
    print(f"# Stage A-2 dir: {sys.argv[1]}")
    print(f"# Stage A-1 dir: {sys.argv[2]}")
    print()
    print(f"  {'seed':>4}  {'Stage A-2 mAP':>14}  {'Stage A-1 mAP':>14}"
          f"  {'paired Δ':>10}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}")
    diffs = []
    shared = sorted(set(a2.keys()) & set(a1.keys()))
    for s in shared:
        d = a2[s] - a1[s]
        diffs.append(d)
        print(f"  {s:>4}  {a2[s]:>14.4f}  {a1[s]:>14.4f}  {d:>+10.4f}")
    print()
    a2_vals = [a2[s] for s in shared]
    a1_vals = [a1[s] for s in shared]
    print(f"  Stage A-2 (n={len(a2_vals)}): mean {statistics.mean(a2_vals):.4f}"
          f"  pstdev {_pstdev(a2_vals):.4f}"
          f"  min {min(a2_vals):.4f}  max {max(a2_vals):.4f}")
    print(f"  Stage A-1 (n={len(a1_vals)}): mean {statistics.mean(a1_vals):.4f}"
          f"  pstdev {_pstdev(a1_vals):.4f}"
          f"  min {min(a1_vals):.4f}  max {max(a1_vals):.4f}")
    if diffs:
        d_m = statistics.mean(diffs)
        d_sd = _pstdev(diffs)
        n_wins = sum(1 for d in diffs if d > 0)
        print(f"  paired Δ (n={len(diffs)}): mean {d_m:+.4f}"
              f"  pstdev {d_sd:.4f}"
              f"  wins {n_wins}/{len(diffs)}")
        print()
        print(f"  verdict: {_verdict(d_m, d_sd, len(diffs))}")

    # mAP cap sanity (the bug fix from 2026-05-16 morning).
    over = [s for s, v in a2.items() if v > 1.0 + 1e-6]
    if over:
        print(f"\n!! mAP_50 > 1.0 on Stage A-2 seeds: {over}")


if __name__ == "__main__":
    main()
