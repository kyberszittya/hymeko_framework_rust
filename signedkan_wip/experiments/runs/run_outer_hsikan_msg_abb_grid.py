"""MSG → ABB → run + aggregate driver for the outer-HSIKAN-Gömb
architecture search (2026-05-21).

Enumerates the chosen axis-product via MSG, prunes by predicted
memory / wall via ABB, runs the survivors through
``run_gomb_smoke`` under ``systemd-run --user --scope -p
MemoryMax=14G``, harvests the JSONL line per cell, and prints a
paired-Δ aggregate vs the established plain-Gömb baselines.

This is the runtime entry point — the search algorithms live in
:mod:`signedkan_wip.src.arch_search.abb`.

Usage::

    python -m signedkan_wip.experiments.runs.run_outer_hsikan_msg_abb_grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from signedkan_wip.src.arch_search import (   # noqa: E402
    ArchCandidate, msg_enumerate, abb_prune, ssg_pareto,
)


# --- Dataset-specific Gömb-strict-bench hyperparameters --------------
# (matches the configs we've used all session for plain-Gömb baseline.)
_DATASET_CFG = {
    "bitcoin_alpha": dict(
        d_embed=32, M_outer=8, d_outer=20, d_middle=24, d_core=48,
        n_tiers=4, topk=56, lr=0.005, pos_weight_auto=True,
    ),
    "bitcoin_otc": dict(
        d_embed=32, M_outer=8, d_outer=20, d_middle=24, d_core=48,
        n_tiers=4, topk=56, lr=0.005, pos_weight_auto=True,
    ),
    "slashdot": dict(
        d_embed=16, M_outer=12, d_outer=8, d_middle=16, d_core=32,
        n_tiers=2, topk=32, lr=0.005, pos_weight_auto=False,
    ),
    "epinions": dict(
        d_embed=16, M_outer=8, d_outer=8, d_middle=16, d_core=32,
        n_tiers=2, topk=32, lr=0.005, pos_weight_auto=False,
    ),
}


def _build_cmd(c: ArchCandidate, log_dir: Path) -> tuple[list[str], Path]:
    """systemd-run + python -m + Gömb-strict-bench config + ABB cell."""
    label = c.name
    logf = log_dir / f"{label}.log"
    ds_cfg = _DATASET_CFG[c.dataset]
    base_args = [
        "--d-embed", str(ds_cfg["d_embed"]),
        "--M-outer", str(ds_cfg["M_outer"]),
        "--d-outer", str(ds_cfg["d_outer"]),
        "--d-middle", str(ds_cfg["d_middle"]),
        "--d-core", str(ds_cfg["d_core"]),
        "--n-tiers", str(ds_cfg["n_tiers"]),
        "--topk", str(ds_cfg["topk"]),
        "--lr", str(ds_cfg["lr"]),
    ]
    if ds_cfg["pos_weight_auto"]:
        base_args.append("--pos-weight-auto")
    cmd = [
        "systemd-run", "--user", "--scope", "-p", "MemoryMax=14G",
        "env",
        f"PATH=/home/kyberszittya/miniconda3/bin:{os.environ.get('PATH','')}",
        f"PYTHONPATH={REPO_ROOT}",
        "HYMEKO_CYCLE_CACHE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "python", "-m", "signedkan_wip.experiments.runs.run_gomb_smoke",
        *c.to_cli_args(),
        *base_args,
    ]
    return cmd, logf


def _harvest(logf: Path) -> dict | None:
    for line in reversed(logf.read_text().splitlines()):
        if line.startswith('{"dataset"'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/outer_hsikan_msg_abb")
    ap.add_argument("--results_file", default=
                    "signedkan_wip/experiments/results/"
                    "outer_hsikan_msg_abb_2026_05_21.jsonl")
    ap.add_argument("--mem_cap_gib", type=float, default=6.5)
    ap.add_argument("--wall_cap_s", type=float, default=120.0)
    ap.add_argument("--n_epochs", type=int, default=60)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print MSG → ABB → SSG pipeline output "
                         "without running cells.")
    args = ap.parse_args()

    # ---------- MSG: enumerate axis-product ----------------------
    # Targeted search around the established lift point:
    # - Bitcoin Alpha: extend depth (4 → 6 → 8); combine with cr_highway
    # - Bitcoin OTC: validate the lever at d=4
    # - Epinions: try shallow d=2 (the dataset is big)
    axes_ba = {
        "dataset": ["bitcoin_alpha"],
        "outer_hsikan_n_layers": [4, 6, 8],
        "inner_skip": ["highway", "cr_highway"],
        "grad_checkpoint": [False, True],
    }
    axes_otc = {
        "dataset": ["bitcoin_otc"],
        "outer_hsikan_n_layers": [2, 4],
        "inner_skip": ["highway"],
        "grad_checkpoint": [False, True],
    }
    axes_epi = {
        "dataset": ["epinions"],
        "outer_hsikan_n_layers": [2],
        "inner_skip": ["highway"],
        "grad_checkpoint": [False, True],
    }
    cands = (msg_enumerate(axes_ba, seeds=(0, 1, 2),
                              n_epochs=args.n_epochs)
             + msg_enumerate(axes_otc, seeds=(0, 1, 2),
                              n_epochs=args.n_epochs)
             + msg_enumerate(axes_epi, seeds=(0, 1, 2),
                              n_epochs=args.n_epochs))
    print(f"[msg] enumerated {len(cands)} candidates")

    # ---------- ABB: prune by memory / wall ----------------------
    survivors, pruned = abb_prune(
        cands, mem_cap_gib=args.mem_cap_gib,
        wall_cap_s=args.wall_cap_s, param_cap=20_000_000,
    )
    print(f"[abb] {len(survivors)} survivors / "
          f"{len(pruned)} pruned (caps: "
          f"mem={args.mem_cap_gib} GiB, wall={args.wall_cap_s} s)")
    if pruned:
        # Group pruned by reason for one-line summary.
        by_reason = {}
        for c, r in pruned:
            tag = r.split()[0]
            by_reason.setdefault(tag, 0)
            by_reason[tag] += 1
        print(f"[abb] pruned reasons: {by_reason}")

    # ---------- SSG: keep wall-minimal per bucket ----------------
    survivors = ssg_pareto(survivors)
    print(f"[ssg] {len(survivors)} after Pareto filter")

    if args.dry_run:
        for c in survivors:
            m = c.predicted_peak_mem_gib()
            w = c.predicted_wall_s()
            print(f"  {c.name:35s}  mem≈{m:.2f} GiB  wall≈{w:.1f} s")
        return

    # ---------- Run survivors -------------------------------------
    log_dir = Path(args.out_dir) / time.strftime("%Y%m%dT%H%M%SZ",
                                                    time.gmtime())
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("")  # truncate

    print(f"[run] log_dir={log_dir}")
    t_total = time.time()
    n_done, n_fail = 0, 0
    for c in survivors:
        cmd, logf = _build_cmd(c, log_dir)
        t0 = time.time()
        rc = subprocess.run(cmd, stdout=logf.open("w"),
                              stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0
        result = _harvest(logf)
        if result and rc == 0:
            result["candidate_name"] = c.name
            result["elapsed_s"] = elapsed
            result["outer_depth"] = c.outer_hsikan_n_layers
            result["inner_skip"] = c.inner_skip
            result["grad_checkpoint"] = c.grad_checkpoint
            result["predicted_mem_gib"] = c.predicted_peak_mem_gib()
            result["predicted_wall_s"] = c.predicted_wall_s()
            with results_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            auc = result.get("val_auc_best") or result.get("val_auroc")
            print(f"[run] DONE {c.name:35s} AUC={auc:.4f} "
                  f"actual={elapsed:.1f}s pred={c.predicted_wall_s():.1f}s")
            n_done += 1
        else:
            print(f"[run] FAIL {c.name:35s} rc={rc} actual={elapsed:.1f}s")
            n_fail += 1
    print(f"[run] {n_done} done / {n_fail} fail in "
          f"{time.time() - t_total:.1f} s total")

    # ---------- Aggregate ----------------------------------------
    rows = [json.loads(l) for l in results_path.read_text().splitlines()
            if l.strip()]

    # Load plain-Gömb baselines (per dataset, depth=1 of stacked-middle).
    base_files = [
        "signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl",
        "signedkan_wip/experiments/results/stacked_gomb_overnight_slashdot_2026_05_20.jsonl",
    ]
    base_rows = []
    for bf in base_files:
        bp = Path(bf)
        if not bp.exists(): continue
        for l in bp.read_text().splitlines():
            if l.strip() and json.loads(l).get("depth") == 1:
                base_rows.append(json.loads(l))

    def auc(r):
        return r.get("val_auc_best") or r.get("val_auroc")

    print("\n=== MSG/ABB grid summary ===")
    for ds in sorted({r["dataset"] for r in rows}):
        cells = [r for r in rows if r["dataset"] == ds]
        print(f"\n--- {ds} ---")
        by_key = {}
        for r in cells:
            key = (r["outer_depth"], r["inner_skip"],
                    r.get("grad_checkpoint", False))
            by_key.setdefault(key, []).append(r)
        for key in sorted(by_key):
            rs = by_key[key]
            aucs = [auc(r) for r in rs if auc(r) is not None]
            if not aucs: continue
            mu = statistics.mean(aucs)
            sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
            wall = statistics.mean(r["elapsed_s"] for r in rs)
            d, skip, ckpt = key
            tag = (f"d={d:<2}  {skip:11s}"
                   f"{'  ckpt' if ckpt else '      '}")
            print(f"  {tag}  AUC {mu:.4f} ± {sd:.4f}  "
                  f"wall {wall:5.1f}s  n={len(aucs)}")
        # Paired Δ vs plain Gömb baseline.
        base_by_seed = {r["seed"]: auc(r)
                          for r in base_rows if r["dataset"] == ds}
        if base_by_seed:
            base_mu = statistics.mean(base_by_seed.values())
            print(f"  vs PLAIN GÖMB {base_mu:.4f} "
                  f"(n={len(base_by_seed)}):")
            for key in sorted(by_key):
                deltas = [auc(r) - base_by_seed[r["seed"]]
                           for r in by_key[key]
                           if auc(r) is not None
                           and r["seed"] in base_by_seed]
                if not deltas: continue
                mu_d = statistics.mean(deltas)
                sd_d = (statistics.stdev(deltas)
                          if len(deltas) > 1 else 0.0)
                se = sd_d / math.sqrt(len(deltas)) if sd_d > 0 else 0.0
                z = mu_d / se if se > 0 else float("inf")
                wins = sum(1 for x in deltas if x > 0)
                d, skip, ckpt = key
                tag = (f"d={d}  {skip:11s}"
                       f"{'  ckpt' if ckpt else ''}")
                print(f"    {tag}  Δ={mu_d:+.4f} ± {sd_d:.4f}  "
                      f"σ_d={z:+.2f}  wins={wins}/{len(deltas)}")


if __name__ == "__main__":
    main()
