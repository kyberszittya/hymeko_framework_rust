"""Analyser for the 2026-05-16 HyMeYOLO ricci-scale sweep.

Reads the per-run jsonl rows under

    signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_<STAMP>/

and produces:

  * a per-scale aggregate table (n, mean ± pstdev, min, max);
  * a paired-Δ table vs scale=1.0 control, with the σ estimate over
    seed-paired differences;
  * a one-line verdict per scale (`+0.03 σ≥2` / `tie` / `-0.03 σ≥2`).

Usage:
    python -m signedkan_wip.experiments.analyse_ricci_scale_sweep_2026_05_16 \\
        signedkan_wip/experiments/results/hymeyolo_ricci_scale_sweep_<STAMP>/
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _verdict(mean_delta: float, sigma_delta: float, n: int) -> str:
    """Statistically separate scale from the 1.0 control.

    'Winner': mean lift > 0.03 AND σ ≥ 2.0 from paired diffs.
    'Loser':  mean drop > 0.03 AND σ ≥ 2.0.
    'Tie':    otherwise.
    """
    if n < 2 or sigma_delta == 0:
        return "n/a"
    z = mean_delta / (sigma_delta / math.sqrt(n))
    if mean_delta > 0.03 and z >= 2.0:
        return f"WIN  mean +{mean_delta:.3f} z={z:.2f}"
    if mean_delta < -0.03 and z <= -2.0:
        return f"LOSS mean {mean_delta:+.3f} z={z:.2f}"
    return f"tie  mean {mean_delta:+.3f} z={z:+.2f}"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: analyse_ricci_scale_sweep_2026_05_16.py <results-dir>")
        sys.exit(2)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}")
        sys.exit(2)

    # Each row is one (scale, seed) measurement.
    rows: list[dict] = []
    for fpath in sorted(root.glob("*.jsonl")):
        with fpath.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Filter to +ricci-mod rows only (the sweep target).
                if rec.get("label") != "+ricci-mod":
                    continue
                rows.append(rec)

    if not rows:
        print(f"no +ricci-mod rows under {root}")
        sys.exit(1)

    # Group by ricci_scale.
    by_scale: dict[float, dict[int, float]] = {}
    for r in rows:
        scale = float(r.get("ricci_scale", 1.0))
        seed = int(r["seed"])
        mAP = r.get("mAP_50")
        if mAP is None:
            continue
        by_scale.setdefault(scale, {})[seed] = float(mAP)

    scales = sorted(by_scale.keys())
    n_seeds = max((len(v) for v in by_scale.values()), default=0)
    print(f"# Ricci-scale sweep aggregate — {len(rows)} rows, {len(scales)} scales, "
          f"up to {n_seeds} seeds per scale")
    print()

    # 1. Per-scale aggregate.
    print("## Per-scale aggregate (5-seed)")
    print()
    print(f"  {'scale':>6s}  {'n':>2s}  {'mean':>7s}  {'pstdev':>7s}  "
          f"{'min':>7s}  {'max':>7s}")
    print(f"  {'-'*6}  {'-'*2}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for s in scales:
        vals = list(by_scale[s].values())
        if not vals:
            continue
        m = _mean(vals); sd = _pstdev(vals)
        print(f"  {s:6.3f}  {len(vals):2d}  {m:7.4f}  {sd:7.4f}  "
              f"{min(vals):7.4f}  {max(vals):7.4f}")
    print()

    # 2. Paired-Δ vs scale=1.0 control.
    if 1.0 in by_scale:
        print("## Paired Δ vs scale=1.0 control")
        print()
        print(f"  {'scale':>6s}  {'n_paired':>8s}  {'mean Δ':>9s}  "
              f"{'σ Δ':>7s}  verdict")
        print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*40}")
        ctrl = by_scale[1.0]
        for s in scales:
            if s == 1.0:
                continue
            this = by_scale[s]
            shared = sorted(set(ctrl.keys()) & set(this.keys()))
            if not shared:
                continue
            diffs = [this[k] - ctrl[k] for k in shared]
            md = _mean(diffs); sd = _pstdev(diffs)
            print(f"  {s:6.3f}  {len(shared):8d}  {md:+9.4f}  {sd:7.4f}  "
                  f"{_verdict(md, sd, len(shared))}")
    else:
        print("# No scale=1.0 rows found — paired Δ table skipped.")
    print()

    # 3. Sanity: any mAP > 1?
    over_1 = [r for r in rows if r.get("mAP_50") is not None
              and r["mAP_50"] > 1.0 + 1e-6]
    if over_1:
        print(f"!! {len(over_1)} rows still have mAP_50 > 1.0 — the metric "
              f"fix did not land:")
        for r in over_1:
            print(f"   seed={r['seed']} scale={r.get('ricci_scale')} "
                  f"mAP_50={r['mAP_50']}")
    else:
        print("# Sanity: all mAP_50 values in [0, 1].")


if __name__ == "__main__":
    main()
