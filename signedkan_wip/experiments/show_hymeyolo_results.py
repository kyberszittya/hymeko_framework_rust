"""View HyMeYOLO results across all run dirs in one place.

Walks ``signedkan_wip/experiments/results/`` looking for HyMeYOLO
output jsonls (Cluttered MNIST and PASCAL VOC), aggregates them
into a single table sorted by run timestamp, and surfaces the
most-recent run per stage label.

Usage::

    python -m signedkan_wip.experiments.show_hymeyolo_results
    python -m signedkan_wip.experiments.show_hymeyolo_results --all
    python -m signedkan_wip.experiments.show_hymeyolo_results --csv

Outputs a Markdown-ish table to stdout; pass --csv to emit CSV
for spreadsheet pasting. Default shows the most recent run per
(stage, dataset) tuple; --all shows every seed of every run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev


RESULTS_ROOT = Path(__file__).parent / "results"

# Pattern: hymeyolo_ladder_<stage>_<stamp> or hymeyolo_<expt>_<stamp>
STAGE_FROM_DIR = re.compile(
    r"(?:hymeyolo_ladder_(?P<stage>[a-z0-9_]+?)_|hymeyolo_(?P<expt>[a-z0-9_]+?)_|stage_d_(?P<staged>[a-z0-9_]+?)_)"
    r"(?P<stamp>\d{8}T\d{6}Z)"
)


def discover_runs() -> list[dict]:
    """Find every HyMeYOLO run dir and parse its jsonls into rows."""
    if not RESULTS_ROOT.is_dir():
        return []
    rows: list[dict] = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = STAGE_FROM_DIR.match(d.name)
        if not m:
            continue
        stage = m.group("stage") or m.group("expt") or m.group("staged")
        stamp = m.group("stamp")
        for jf in sorted(d.glob("*.jsonl")):
            try:
                lines = [l for l in jf.read_text().splitlines() if l.strip()]
            except OSError:
                continue
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_run_dir"] = d.name
                rec["_stage"] = stage
                rec["_stamp"] = stamp
                rec["_jsonl"] = jf.name
                rows.append(rec)
    return rows


def fmt_map(x) -> str:
    if x is None:
        return "    —"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "    —"


def aggregate_per_stage(rows: list[dict]) -> dict:
    """Group rows by (stage, dataset) and compute n / mean / pstdev."""
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        dataset = r.get("dataset", "cmnist")
        # +ricci-mod is the production config; other labels are
        # ablations from train_circles_ricci's 5-config sweep.
        label = r.get("label", r.get("_stage", "?"))
        key = (r["_stage"], dataset, label)
        by_key.setdefault(key, []).append(r)
    summary: dict = {}
    for key, items in by_key.items():
        maps = [it["mAP_50"] for it in items if isinstance(it.get("mAP_50"), (int, float))]
        latest_stamp = max(it["_stamp"] for it in items)
        latest_items = [it for it in items if it["_stamp"] == latest_stamp]
        latest_maps = [it["mAP_50"] for it in latest_items
                       if isinstance(it.get("mAP_50"), (int, float))]
        summary[key] = {
            "n_total": len(items),
            "all_maps": maps,
            "latest_stamp": latest_stamp,
            "n_latest": len(latest_items),
            "latest_maps": latest_maps,
            "latest_run_dir": latest_items[0]["_run_dir"] if latest_items else "",
        }
    return summary


def print_latest_per_stage(summary: dict) -> None:
    print("# Latest HyMeYOLO results — most-recent run per (stage, dataset, label)\n")
    header = (
        f"{'stage':<14s} {'dataset':<22s} {'label':<18s} {'n':>3s} "
        f"{'mAP_50_mean':>12s} {'pstdev':>8s} {'min':>7s} {'max':>7s} "
        f"  {'latest run':<40s}"
    )
    print(header)
    print("-" * len(header))
    rows_sorted = sorted(summary.items(), key=lambda kv: kv[1]["latest_stamp"], reverse=True)
    for (stage, dataset, label), s in rows_sorted:
        maps = s["latest_maps"]
        if not maps:
            mean_s = "    —"
            sd_s = "    —"
            mn_s = mx_s = "    —"
        else:
            mean_s = fmt_map(mean(maps))
            sd_s = fmt_map(pstdev(maps)) if len(maps) >= 2 else "  0.0000"
            mn_s = fmt_map(min(maps))
            mx_s = fmt_map(max(maps))
        print(
            f"{stage:<14s} {dataset[:22]:<22s} {label[:18]:<18s} "
            f"{s['n_latest']:>3d} {mean_s:>12s} {sd_s:>8s} "
            f"{mn_s:>7s} {mx_s:>7s}   {s['latest_run_dir']:<40s}"
        )


def print_all(rows: list[dict]) -> None:
    print("# All HyMeYOLO results — one line per seed across all runs\n")
    print(f"{'stamp':<16s} {'stage':<14s} {'dataset':<22s} {'label':<18s} "
          f"{'seed':>4s} {'mAP_50':>10s}  run_dir")
    rows_sorted = sorted(rows, key=lambda r: (r["_stamp"], r.get("seed", 0)))
    for r in rows_sorted:
        seed = r.get("seed", -1)
        m = fmt_map(r.get("mAP_50"))
        print(
            f"{r['_stamp']:<16s} {r['_stage']:<14s} "
            f"{r.get('dataset', 'cmnist')[:22]:<22s} "
            f"{r.get('label', '?')[:18]:<18s} {seed:>4d} {m:>10s}  "
            f"{r['_run_dir']}"
        )


def print_csv(rows: list[dict]) -> None:
    import csv
    fields = ["stamp", "stage", "dataset", "label", "seed", "mAP_50",
              "mAP_50_95", "box_cls_acc", "loss_start", "loss_end",
              "wall_s", "epochs", "n_images", "n_params", "run_dir"]
    w = csv.DictWriter(sys.stdout, fieldnames=fields)
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["_stamp"], r.get("seed", 0))):
        w.writerow({
            "stamp": r["_stamp"],
            "stage": r["_stage"],
            "dataset": r.get("dataset", "cmnist"),
            "label": r.get("label", ""),
            "seed": r.get("seed"),
            "mAP_50": r.get("mAP_50"),
            "mAP_50_95": r.get("mAP_50_95"),
            "box_cls_acc": r.get("box_cls_acc"),
            "loss_start": r.get("loss_start"),
            "loss_end": r.get("loss_end"),
            "wall_s": r.get("wall_s"),
            "epochs": r.get("epochs"),
            "n_images": r.get("n_images"),
            "n_params": r.get("n_params"),
            "run_dir": r["_run_dir"],
        })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true",
                   help="Show every seed across every run.")
    p.add_argument("--csv", action="store_true",
                   help="Emit CSV instead of human-readable table.")
    p.add_argument("--stage", default=None,
                   help="Filter to a specific stage (a2, b_resnet, b_hsikan, "
                        "c_fpn, stage_d_voc2007, ...).")
    args = p.parse_args()

    rows = discover_runs()
    if args.stage:
        rows = [r for r in rows if r["_stage"] == args.stage]
    if not rows:
        print(f"No HyMeYOLO results under {RESULTS_ROOT}", file=sys.stderr)
        return 1

    if args.csv:
        print_csv(rows)
    elif args.all:
        print_all(rows)
    else:
        summary = aggregate_per_stage(rows)
        print_latest_per_stage(summary)
        print()
        print(f"  Total runs found: {len({r['_run_dir'] for r in rows})}")
        print(f"  Total seeds:      {len(rows)}")
        print(f"  Results root:     {RESULTS_ROOT}")
        print(f"  Use --all for every seed; --csv for spreadsheet output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
