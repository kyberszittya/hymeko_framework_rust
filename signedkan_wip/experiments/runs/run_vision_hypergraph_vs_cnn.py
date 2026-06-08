"""Matrix orchestrator: hypergraph-vision operators vs CNN/MLP re-benchmark.

Runs model x dataset x seed cells (each a subprocess of
``vision_bench_cell``), checkpoints every row to jsonl (resumable), and
aggregates per (model, dataset) mean +/- sd test accuracy. Answers: does
the hypergraph->CNN gap survive FAIR training (the 2026-05-06 null was
5 epochs / 1 seed)?

Usage
-----
    PYTHONPATH=$PWD systemd-run --user --scope -p MemoryMax=16G \\
        /home/kyberszittya/miniconda3/bin/python -m \\
        signedkan_wip.experiments.runs.run_vision_hypergraph_vs_cnn \\
            --seeds 0,1,2 --n-epochs 15 --train-subset 8000 \\
            --results-file /tmp/vision_bench/results.jsonl

``--dry-run`` lists the cell matrix (no torch). ``--analyze-only``
re-aggregates an existing jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Reuse generic helpers (torch-free import chain) rather than re-deriving.
from signedkan_wip.experiments.runs.run_hsikan_mixed_composite_smoke import (
    _git_sha,
    _last_json_line,
    _parse_time_v_rss_gib,
    _repo_root,
)
from signedkan_wip.experiments.runs.run_regime_abc_5seed import _mean_sd
from signedkan_wip.src.vision.vision_bench_cell import DATASETS, MODEL_NAMES

CELL_MODULE = "signedkan_wip.src.vision.vision_bench_cell"


def enumerate_cells(
    models: list[str], datasets: list[str], seeds: list[int]
) -> list[tuple[str, str, int]]:
    """Full (model, dataset, seed) matrix in a stable order."""
    return [(m, d, s) for m in models for d in datasets for s in seeds]


def cell_key(model: str, dataset: str, seed: int) -> str:
    """Display-only key used in logs. Resume detection uses
    ``full_cell_key`` so multi-config sweeps writing to one results
    file don't false-positive-skip across configs."""
    return f"{model}|{dataset}|seed{seed}"


# Fields that distinguish two cells with the same (model, dataset, seed)
# under different sweep configs. Any field here MUST be present in the
# row that ``vision_bench_cell`` emits.
_CONFIG_FIELDS_FOR_RESUME = (
    "hidden", "n_layers", "spatial_filter", "tie_we",
    "n_epochs", "train_subset", "compile", "amp",
)


def full_cell_key(row_or_cfg: dict[str, Any], model: str, dataset: str,
                  seed: int) -> str:
    """Resume key including all config axes — fixes the 2026-05-29 bug
    where multi-(hidden, n_layers) sweeps to one results file
    false-positive-skipped after the first config landed."""
    parts = [model, dataset, f"seed{seed}"]
    for f in _CONFIG_FIELDS_FOR_RESUME:
        if f in row_or_cfg:
            parts.append(f"{f}={row_or_cfg[f]}")
    return "|".join(parts)


def _load_done(results_file: Path) -> set[str]:
    done: set[str] = set()
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add(full_cell_key(r, r["model"], r["dataset"],
                                       int(r["seed"])))
    return done


def run_cell_subprocess(
    repo: Path, *, model: str, dataset: str, seed: int, cfg: dict[str, Any],
    log_path: Path,
) -> tuple[dict[str, Any], float, float]:
    """Subprocess one vision cell; return (row, wall_s, peak_rss_gib)."""
    import os
    import time

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    inner = [
        sys.executable, "-m", CELL_MODULE,
        "--model", model, "--dataset", dataset, "--seed", str(seed),
        "--n-epochs", str(cfg["n_epochs"]),
        "--train-subset", str(cfg["train_subset"]),
        "--batch-size", str(cfg["batch_size"]),
        "--hidden", str(cfg["hidden"]),
        "--n-layers", str(cfg.get("n_layers", 2)),
    ]
    if cfg.get("tie_we"):
        inner.append("--tie-we")
    sf = cfg.get("spatial_filter", "none")
    if sf and sf != "none":
        inner.extend(["--spatial-filter", str(sf)])
    if cfg.get("compile"):
        inner.append("--compile")
    if cfg.get("amp"):
        inner.append("--amp")
    gnu_time = Path("/usr/bin/time")
    cmd = ([str(gnu_time), "-v", *inner] if gnu_time.is_file() else inner)
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(repo), env=env, check=False
    )
    wall_s = time.monotonic() - t0
    rss = _parse_time_v_rss_gib(proc.stderr)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n=== stdout ===\n{proc.stdout}\n=== stderr ===\n{proc.stderr}\n"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vision cell failed rc={proc.returncode}; see {log_path}")
    return _last_json_line(proc.stdout), wall_s, rss


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (model, dataset) mean/sd accuracy + per-dataset CNN gap."""
    by: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc = r.get("test_accuracy")
        if acc is not None:
            by.setdefault((r["model"], r["dataset"]), []).append(float(acc))
    per = {
        f"{m}|{d}": {"mean": _mean_sd(v)[0], "sd": _mean_sd(v)[1], "n": len(v)}
        for (m, d), v in by.items()
    }
    # Per dataset: gap of each model's mean below the CNN mean (positive
    # = below CNN). The headline measurement.
    gaps: dict[str, dict[str, float]] = {}
    for d in {ds for (_, ds) in by}:
        cnn = per.get(f"cnn|{d}", {}).get("mean")
        if cnn is None:
            continue
        gaps[d] = {
            m: round(cnn - per[f"{m}|{d}"]["mean"], 4)
            for (m, ds) in by
            if ds == d
        }
    return {"per_model_dataset": per, "cnn_gap": gaps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODEL_NAMES))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--train-subset", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--results-file", default="/tmp/vision_bench/results.jsonl")
    ap.add_argument("--log-dir", default="/tmp/vision_bench")
    ap.add_argument("--tie-we", action="store_true",
                    help="HSiKAN translation-equivariance variant; passed "
                         "through to each cell (silently ignored by non-hsikan).")
    ap.add_argument("--spatial-filter", default="none",
                    choices=["none", "scalar", "per_channel"],
                    help="HSiKAN within-RF spatial filter mode; passed "
                         "through (silently ignored by non-hsikan).")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(model) — profile-justified win on "
                         "HSiKAN at n_epochs >= 10.")
    ap.add_argument("--amp", action="store_true",
                    help="autocast + GradScaler. NULL on HSiKAN MNIST per "
                         "Tier-1 probe; included for non-HSiKAN models or "
                         "larger configs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)

    repo = _repo_root()
    models = [m for m in args.models.split(",") if m.strip()]
    datasets = [d for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results_file = Path(args.results_file)
    cells = enumerate_cells(models, datasets, seeds)

    if args.analyze_only:
        rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
        print(json.dumps(aggregate(rows), indent=2))
        return 0

    if args.dry_run:
        print(json.dumps({
            "n_cells": len(cells),
            "cells": [cell_key(m, d, s) for (m, d, s) in cells],
        }, indent=2))
        return 0

    cfg = {"n_epochs": args.n_epochs, "train_subset": args.train_subset,
           "batch_size": args.batch_size, "hidden": args.hidden,
           "n_layers": args.n_layers,
           "tie_we": args.tie_we, "spatial_filter": args.spatial_filter,
           "compile": args.compile, "amp": args.amp}
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(results_file)
    sha = _git_sha(repo)

    for i, (model, dataset, seed) in enumerate(cells, 1):
        key = cell_key(model, dataset, seed)
        # Resume check: must match the FULL config, not just (m,d,s).
        # Otherwise multi-config sweeps writing to one jsonl will
        # false-positive-skip after the first config lands (2026-05-29
        # depth+narrow bug).
        full_key = full_cell_key(cfg, model, dataset, seed)
        if full_key in done:
            print(f"[vbench] skip {key} (done) [{i}/{len(cells)}]", file=sys.stderr)
            continue
        log_path = log_dir / f"{key.replace('|', '_')}.log"
        try:
            row, wall_s, rss = run_cell_subprocess(
                repo, model=model, dataset=dataset, seed=seed, cfg=cfg,
                log_path=log_path,
            )
        except RuntimeError as err:
            print(f"[vbench] FAIL {key}: {err}", file=sys.stderr)
            rec = {"model": model, "dataset": dataset, "seed": seed,
                   "test_accuracy": None, "error": str(err), "git_sha": sha}
            with open(results_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
            continue
        rec = {**row, "wall_s": round(wall_s, 2), "peak_rss_gib": round(rss, 3),
               **cfg, "git_sha": sha}
        with open(results_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[vbench] {key} acc={row.get('test_accuracy'):.4f} "
              f"wall={wall_s:.0f}s rss={rss:.2f}G [{i}/{len(cells)}]", file=sys.stderr)

    rows = [json.loads(x) for x in results_file.read_text().splitlines() if x.strip()]
    summary = aggregate(rows)
    print(json.dumps(summary, indent=2))
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
