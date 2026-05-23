"""Unified paired analyser for HyMeYOLO YOLO-parity-ladder stages.

Compares a target stage against an arbitrary baseline stage (typically
the immediate predecessor on the ladder). Reads `+ricci-mod` rows
from the two result dirs, pairs by seed, reports mean / pstdev / min /
max per stage + the paired-Δ statistics + a verdict against the
pre-registered criterion (paired mean ≥ 0.03 AND z ≥ 2).

Usage:
    python -m signedkan_wip.experiments.analyse_hymeyolo_ladder_paired \\
        <target-stage-dir> <baseline-stage-dir>

Replaces the per-stage analyser scripts (analyse_stage_a2_5seed_*.py)
with a single source of truth. The protocol-pair logic is identical.

The mAP@0.5 GT-consumption fix is verified inline: if any row reports
mAP_50 > 1.0, the analyser flags it loudly (which would indicate the
buggy metric had crept back in).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def _load_map50_by_seed(d: Path) -> dict[int, float]:
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


def _stage_label(p: Path) -> str:
    """Recover the stage name from a dir like `hymeyolo_ladder_a2_<STAMP>`
    or the older `hymeyolo_warmstart_5seed_<STAMP>` etc."""
    name = p.name
    if "ladder_" in name:
        return name.split("ladder_", 1)[1].split("_", 1)[0]
    for key in ("stage_a2", "stage_a3", "warmstart", "ricci_scale_sweep"):
        if key in name:
            return key
    return name


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: analyse_hymeyolo_ladder_paired.py "
              "<target-dir> <baseline-dir>")
        sys.exit(2)
    t_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    t = _load_map50_by_seed(t_dir)
    b = _load_map50_by_seed(b_dir)
    if not t:
        print(f"no +ricci-mod rows in target {t_dir}")
        sys.exit(1)
    if not b:
        print(f"no +ricci-mod rows in baseline {b_dir}")
        sys.exit(1)

    t_label = _stage_label(t_dir)
    b_label = _stage_label(b_dir)

    print(f"# HyMeYOLO ladder paired comparison")
    print(f"# target   ({t_label}): {t_dir}")
    print(f"# baseline ({b_label}): {b_dir}")
    print()
    print(f"  {'seed':>4}  {t_label+' mAP':>14}  {b_label+' mAP':>14}"
          f"  {'paired Δ':>10}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*10}")

    diffs = []
    shared = sorted(set(t.keys()) & set(b.keys()))
    for s in shared:
        d = t[s] - b[s]
        diffs.append(d)
        print(f"  {s:>4}  {t[s]:>14.4f}  {b[s]:>14.4f}  {d:>+10.4f}")

    print()
    t_vals = [t[s] for s in shared]
    b_vals = [b[s] for s in shared]
    print(f"  {t_label} (n={len(t_vals)}): mean {statistics.mean(t_vals):.4f}"
          f"  pstdev {_pstdev(t_vals):.4f}"
          f"  min {min(t_vals):.4f}  max {max(t_vals):.4f}")
    print(f"  {b_label} (n={len(b_vals)}): mean {statistics.mean(b_vals):.4f}"
          f"  pstdev {_pstdev(b_vals):.4f}"
          f"  min {min(b_vals):.4f}  max {max(b_vals):.4f}")
    if diffs:
        d_m = statistics.mean(diffs)
        d_sd = _pstdev(diffs)
        n_wins = sum(1 for d in diffs if d > 0)
        print(f"  paired Δ (n={len(diffs)}): mean {d_m:+.4f}"
              f"  pstdev {d_sd:.4f}"
              f"  wins {n_wins}/{len(diffs)}")
        print()
        print(f"  verdict: {_verdict(d_m, d_sd, len(diffs))}")

    over = [(p, v) for p, v in t.items() if v > 1.0 + 1e-6]
    if over:
        print(f"\n!! mAP_50 > 1.0 in {t_label}: {over}")
        print("   The GT-consumption metric fix may have regressed; "
              "investigate train_circles_ricci.py:compute_detection_metrics.")


if __name__ == "__main__":
    main()
