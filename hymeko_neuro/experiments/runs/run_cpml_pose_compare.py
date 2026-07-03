"""Paired 3-seed comparison: CPMLPose vs PositionRegHSiKAN on
``pose_k4`` and ``pose_k6``.

For each seed in ``--seeds``, run both ``cell_pose`` (baseline,
PositionRegHSiKAN) and ``cell_cpml_pose`` (new, CPMLPose) on each arity
in ``--arities``. Cell function is the existing data + training pipeline
in ``run_final_cell.py`` — only the model class differs, so this is a
clean isolation of CPML's tier-stratified routing as a per-vertex
regression backbone.

Outputs a single jsonl row per (model, arity, seed) into
``--results-file``; the script's final stage aggregates per-(model,
arity) MSE/MAE means ± sd, paired Δ, and σ. Plan:
``docs/plans/2026-05-29-cpml-pose-regression/``.

Usage
-----
    PYTHONPATH=$PWD systemd-run --user --scope -p MemoryMax=16G \\
        /home/kyberszittya/miniconda3/bin/python -m \\
        hymeko_neuro.experiments.runs.run_cpml_pose_compare \\
            --arities 4,6 --seeds 0,1,2 --n-epochs 80 --hidden 16 \\
            --results-file /tmp/cpml_pose/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path


def _seed_everything(seed: int) -> None:
    """torch + numpy + python seeds, in that order."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _run_cell(model_kind: str, arity: int, hidden: int, n_epochs: int,
              device, seed: int) -> dict | None:
    """Dispatch to either ``cell_pose`` (baseline) or ``cell_cpml_pose``
    (new). Returns the cell's result dict augmented with ``seed`` and
    ``wall_s``; ``None`` if the cell rejects (no candidates of this
    arity)."""
    from hymeko_neuro.experiments.runs.run_final_cell import (
        cell_pose,
        cell_cpml_pose,
    )
    _seed_everything(seed)
    t0 = time.monotonic()
    if model_kind == "baseline":
        row = cell_pose(arity, hidden, n_epochs, device)
    elif model_kind == "cpml":
        row = cell_cpml_pose(arity, hidden, n_epochs, device)
    else:
        raise ValueError(f"unknown model_kind {model_kind!r}")
    wall = time.monotonic() - t0
    if row is None:
        return None
    row["seed"] = seed
    row["wall_s"] = round(wall, 2)
    row["model_kind"] = model_kind
    return row


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _paired_delta(a: list[float], b: list[float]) -> dict:
    """``a - b`` paired delta; both must be same-length, same-seed-ordered."""
    if len(a) != len(b) or len(a) == 0:
        return {"n_paired": 0}
    diffs = [ai - bi for ai, bi in zip(a, b)]
    m, sd = _mean_sd(diffs)
    sigma = m / (sd / math.sqrt(len(diffs))) if sd > 0 else 0.0
    return {
        "delta_mean": m, "delta_sd": sd, "sigma": sigma,
        "n_paired": len(diffs),
        "wins_a": sum(1 for d in diffs if d < 0),  # lower mse = a wins
    }


def aggregate(rows: list[dict]) -> dict:
    """Per (model_kind, arity) mean/sd of MSE + MAE; paired Δ (cpml − base)."""
    by: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        key = (r["model_kind"], int(r["dataset"].rsplit("k", 1)[1]))
        by.setdefault(key, []).append(r)
    per = {}
    for (k, a), rs in by.items():
        mse = [r["mse"] for r in rs]
        mae = [r["mae"] for r in rs]
        m_mse, sd_mse = _mean_sd(mse)
        m_mae, sd_mae = _mean_sd(mae)
        per[f"{k}|k{a}"] = {
            "mse_mean": m_mse, "mse_sd": sd_mse,
            "mae_mean": m_mae, "mae_sd": sd_mae,
            "n": len(rs),
            "params": rs[0].get("n_params"),
        }
    # Paired Δ for each arity.
    paired = {}
    for a in {int(r["dataset"].rsplit("k", 1)[1]) for r in rows}:
        # Seed-aligned ordering: assume same seed list, sorted.
        base_pairs = sorted(
            [(r["seed"], r["mse"]) for r in rows
             if r["model_kind"] == "baseline" and r["dataset"].endswith(f"k{a}")]
        )
        cpml_pairs = sorted(
            [(r["seed"], r["mse"]) for r in rows
             if r["model_kind"] == "cpml" and r["dataset"].endswith(f"k{a}")]
        )
        if len(base_pairs) == len(cpml_pairs) and base_pairs:
            paired[f"k{a}"] = _paired_delta(
                [v for _, v in cpml_pairs], [v for _, v in base_pairs],
            )
        else:
            paired[f"k{a}"] = {"n_paired": 0, "note": "uneven seeds"}
    return {"per_model_arity": per, "paired_cpml_minus_baseline_mse": paired}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arities", default="4,6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--results-file", default="/tmp/cpml_pose/results.jsonl")
    ap.add_argument("--log-dir", default="/tmp/cpml_pose")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    arities = [int(a) for a in args.arities.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results_file = Path(args.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
        print(json.dumps(aggregate(rows), indent=2))
        return 0

    cells = [(k, a, s) for k in ("baseline", "cpml") for a in arities for s in seeds]

    if args.dry_run:
        print(json.dumps({
            "n_cells": len(cells),
            "cells": [f"{k}|k{a}|seed{s}" for k, a, s in cells],
        }, indent=2))
        return 0

    # Lazy import torch / device only when actually running.
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done: set[tuple[str, int, int]] = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((r["model_kind"], int(r["dataset"].rsplit("k", 1)[1]),
                      int(r["seed"])))

    for i, (kind, arity, seed) in enumerate(cells, 1):
        if (kind, arity, seed) in done:
            print(f"[cpml-pose] skip {kind}|k{arity}|seed{seed} (done) "
                  f"[{i}/{len(cells)}]", file=sys.stderr)
            continue
        try:
            row = _run_cell(kind, arity, args.hidden, args.n_epochs, device, seed)
        except Exception as err:  # noqa: BLE001
            print(f"[cpml-pose] FAIL {kind}|k{arity}|seed{seed}: {err}",
                  file=sys.stderr)
            rec = {"model_kind": kind, "dataset": f"pose_k{arity}",
                   "seed": seed, "error": str(err)}
            with open(results_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            continue
        if row is None:
            print(f"[cpml-pose] none {kind}|k{arity}|seed{seed} "
                  f"(no candidates) [{i}/{len(cells)}]", file=sys.stderr)
            continue
        with open(results_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[cpml-pose] {kind}|k{arity}|seed{seed} "
              f"mse={row['mse']:.4f} mae={row['mae']:.4f} "
              f"wall={row['wall_s']:.0f}s [{i}/{len(cells)}]",
              file=sys.stderr)

    rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()
            and "mse" in json.loads(x)]
    summary = aggregate(rows)
    print(json.dumps(summary, indent=2))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.log_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
