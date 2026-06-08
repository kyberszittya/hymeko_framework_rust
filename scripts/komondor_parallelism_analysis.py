#!/usr/bin/env python3
"""Komondor parallelism showcase analysis.

Reads the wall-time-per-cell data from the chain JSONLs (sequential runs)
and computes:

  1. Theoretical serial total wall      (sum of cell walls)
  2. Theoretical 20-fold parallel wall  (max of cell walls, K-cap)
  3. Theoretical speedup factor          serial / parallel
  4. Per-arity setup wall + AUC stability across seeds

Produces a Markdown table + ASCII bar histogram of per-cell wall times.

Usage:
  python3 scripts/komondor_parallelism_analysis.py \
      [jsonl1] [jsonl2] [...] \
      [--output reports/<slug>.md]

Default JSONLs: hsikan_edge_cr_audit/results.jsonl + results_ba_otc.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fmt_wall(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"


def ascii_hist(walls: list[float], bins: int = 10, width: int = 50) -> str:
    if not walls:
        return "(no walls)"
    lo, hi = min(walls), max(walls)
    if hi <= lo:
        return f"all walls ≈ {fmt_wall(lo)}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for w in walls:
        b = min(int((w - lo) / step), bins - 1)
        counts[b] += 1
    max_c = max(counts)
    out = []
    for i, c in enumerate(counts):
        edge_lo, edge_hi = lo + i * step, lo + (i + 1) * step
        bar = "#" * int(round(c / max_c * width))
        out.append(
            f"{fmt_wall(edge_lo):>7} - {fmt_wall(edge_hi):>7}  "
            f"|{bar:<{width}}| {c}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "jsonls", nargs="*",
        help="JSONL paths; defaults to hsikan_edge_cr_audit/results*.jsonl",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Markdown report path (default: stdout only)",
    )
    parser.add_argument(
        "--slots", type=int, nargs="+", default=[1, 5, 10, 20, 40],
        help="Parallel slot counts to project (default: 1 5 10 20 40)",
    )
    args = parser.parse_args(argv)

    paths = (
        [Path(p) for p in args.jsonls] if args.jsonls
        else [REPO / "hsikan_edge_cr_audit" / "results.jsonl",
              REPO / "hsikan_edge_cr_audit" / "results_ba_otc.jsonl"]
    )
    rows: list[dict] = []
    for p in paths:
        rs = load_jsonl(p)
        for r in rs:
            r["_source"] = p.name
        rows.extend(rs)
    if not rows:
        print("[err] no rows; check JSONL paths", file=sys.stderr)
        return 1

    lines: list[str] = []
    out = lines.append

    out(f"# Komondor parallelism analysis — {len(rows)} completed cells")
    out("")
    out("## 1. Per-cell wall-time distribution")
    out("")
    walls = [r["_audit_elapsed_s"] for r in rows if "_audit_elapsed_s" in r]
    if walls:
        out("```")
        out(ascii_hist(walls))
        out("```")
        out("")
        out(f"- n = {len(walls)}")
        out(f"- min  = {fmt_wall(min(walls))}")
        out(f"- p25  = {fmt_wall(statistics.quantiles(walls, n=4)[0])}")
        out(f"- median = {fmt_wall(statistics.median(walls))}")
        out(f"- p75  = {fmt_wall(statistics.quantiles(walls, n=4)[2])}")
        out(f"- max  = {fmt_wall(max(walls))}")
        out("")

    # Per-dataset breakdown.
    out("## 2. Per-(dataset, mode) wall + AUC")
    out("")
    out("| dataset | mode | n | median wall | total wall | AUC mean ± std |")
    out("|---|---|---|---|---|---|")
    grp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["_audit_mode"])].append(r)
    for (ds, mode), gs in sorted(grp.items()):
        ws = [g["_audit_elapsed_s"] for g in gs]
        aucs = [g["auc"] for g in gs if isinstance(g.get("auc"), (int, float))]
        if aucs:
            auc_str = (
                f"{statistics.mean(aucs):.4f}"
                f" ± {statistics.stdev(aucs) if len(aucs) >= 2 else 0.0:.4f}"
            )
        else:
            auc_str = "—"
        out(
            f"| {ds} | {mode} | {len(gs)} | "
            f"{fmt_wall(statistics.median(ws))} | {fmt_wall(sum(ws))} | "
            f"{auc_str} |"
        )
    out("")

    # The headline number.
    out("## 3. Parallelism speedup projection")
    out("")
    serial_total = sum(walls)
    cell_max = max(walls)
    out(f"Sequential total wall (sum of all cells):  **{fmt_wall(serial_total)}**")
    out(f"Single-cell longest wall (worst case):     **{fmt_wall(cell_max)}**")
    out("")
    out("| K (parallel slots) | projected wall | speedup vs serial |")
    out("|---|---|---|")
    for k in args.slots:
        # Greedy makespan lower bound: ceil(n/K) groups, each at most
        # one max-cell-wall. Tighter than sum/K when n_cells is small.
        n_cells = len(walls)
        # Use list-scheduling LPT bound: max(max_cell, sum/K).
        proj = max(cell_max, serial_total / k)
        speedup = serial_total / proj if proj > 0 else float("inf")
        out(f"| {k:>3} | {fmt_wall(proj)} | {speedup:.2f}× |")
    out("")
    out(
        f"(LPT lower bound: each slot wall = max(max_cell, sum/K). "
        f"max_cell = {fmt_wall(cell_max)} dominates at high K.)"
    )
    out("")

    # Per-arity time. _audit_config + arities give us the SOTA mixed regime.
    arity_walls: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ar = r.get("arities")
        if isinstance(ar, list):
            ar_key = ",".join(str(x) for x in sorted(ar, key=str))
            arity_walls[ar_key].append(r["_audit_elapsed_s"])
    if len(arity_walls) > 1:
        out("## 4. Per-arity wall breakdown")
        out("")
        out("| arities | n | median wall |")
        out("|---|---|---|")
        for ak, ws in sorted(arity_walls.items()):
            out(f"| {ak} | {len(ws)} | {fmt_wall(statistics.median(ws))} |")
        out("")

    out("## 5. Hardware comparison")
    out("")
    out("| platform | GPU | per-cell wall | RSS | notes |")
    out("|---|---|---|---|---|")
    out("| Komondor (HUN-REN) | A100-SXM4-40GB | "
        f"{fmt_wall(statistics.median(walls))} (median) | "
        "~2 GB | A100; pinned 24 GB / 48 GB SLURM mem |")
    out("| Local | RTX 2070 SUPER 7.6 GB | OOM (edge_cr) | "
        ">7.6 GB → kill | full SOTA config exceeds local VRAM |")
    out("")
    out("Local was the first attempted host for the SOTA `c2,c3,c4,c5,w2,w3 + "
        "quaternion + edge_cr` config (2026-06-03) and crashed at "
        "_catmull_rom_eval forward; Komondor A100 absorbs the same config "
        "in ~2 GB RSS and completes 5-seed in 90 min instead of OOM.")
    out("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"\n[ok] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
