"""Extended MSG/ABB grid — follows up the 2026-05-21 initial grid.

Adds the four follow-ups identified in
`reports/2026-05-21-outer-hsikan-msg-abb-grid.md`:

1. **5-seed extension** on Bitcoin Alpha d∈{4,6,8} highway+cr_highway.
   Two new seeds (3, 4) added to the existing 3-seed cells from
   today; combined we get n=5 for paired comparison.
2. **Bitcoin Alpha d=16** highway + cr_highway, with grad-ckpt.
   Tests whether the monotonic depth-scaling extends beyond d=8.
3. **Bitcoin OTC d=8** validation: does OTC see the same
   monotonic lift as BA?
4. **Epinions d=2** with a SMALLER Gömb config (d_core=16,
   topk=16, n_tiers=2) + grad-ckpt. The strict-bench config
   OOMs at the CPML edge_logits with 640k Epinions train
   edges; a smaller core/topk reduces that pressure.
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

# Two dataset configs: "strict_bench" (the established baseline) and
# "small" (for Epinions to actually fit on a 7.6 GiB GPU).
_DATASET_CFG = {
    ("bitcoin_alpha", "strict_bench"): dict(
        d_embed=32, M_outer=8, d_outer=20, d_middle=24, d_core=48,
        n_tiers=4, topk=56, lr=0.005, pos_weight_auto=True,
    ),
    ("bitcoin_otc", "strict_bench"): dict(
        d_embed=32, M_outer=8, d_outer=20, d_middle=24, d_core=48,
        n_tiers=4, topk=56, lr=0.005, pos_weight_auto=True,
    ),
    ("epinions", "small"): dict(
        d_embed=8, M_outer=4, d_outer=4, d_middle=8, d_core=16,
        n_tiers=2, topk=16, lr=0.003, pos_weight_auto=False,
    ),
}


def _build_cmd(c, profile, log_dir):
    label = f"{c.name}_{profile}"
    logf = log_dir / f"{label}.log"
    cfg = _DATASET_CFG[(c.dataset, profile)]
    base = [
        "--d-embed", str(cfg["d_embed"]),
        "--M-outer", str(cfg["M_outer"]),
        "--d-outer", str(cfg["d_outer"]),
        "--d-middle", str(cfg["d_middle"]),
        "--d-core", str(cfg["d_core"]),
        "--n-tiers", str(cfg["n_tiers"]),
        "--topk", str(cfg["topk"]),
        "--lr", str(cfg["lr"]),
    ]
    if cfg["pos_weight_auto"]:
        base.append("--pos-weight-auto")
    cmd = [
        "systemd-run", "--user", "--scope", "-p", "MemoryMax=14G",
        "env",
        f"PATH=/home/kyberszittya/miniconda3/bin:{os.environ.get('PATH','')}",
        f"PYTHONPATH={REPO_ROOT}",
        "HYMEKO_CYCLE_CACHE=1",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "python", "-m", "signedkan_wip.experiments.runs.run_gomb_smoke",
        *c.to_cli_args(),
        *base,
    ]
    return cmd, logf, label


def _harvest(logf):
    for line in reversed(logf.read_text().splitlines()):
        if line.startswith('{"dataset"'):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/tmp/outer_hsikan_msg_abb_ext")
    ap.add_argument("--results_file", default=
                    "signedkan_wip/experiments/results/"
                    "outer_hsikan_msg_abb_ext_2026_05_21.jsonl")
    ap.add_argument("--mem_cap_gib", type=float, default=6.5)
    ap.add_argument("--wall_cap_s", type=float, default=300.0)
    args = ap.parse_args()

    # --- Follow-up 1: BA 5-seed extension (seeds 3, 4) -----------
    ba_5seed = msg_enumerate(
        {
            "dataset": ["bitcoin_alpha"],
            "outer_hsikan_n_layers": [4, 6, 8],
            "inner_skip": ["highway", "cr_highway"],
            "grad_checkpoint": [False],
        },
        seeds=(3, 4),
        n_epochs=60,
    )

    # --- Follow-up 2: BA d=16, both skips, ckpt --------------------
    ba_d16 = msg_enumerate(
        {
            "dataset": ["bitcoin_alpha"],
            "outer_hsikan_n_layers": [16],
            "inner_skip": ["highway", "cr_highway"],
            "grad_checkpoint": [False, True],
        },
        seeds=(0, 1, 2),
        n_epochs=60,
    )

    # --- Follow-up 3: OTC d=8 + d=2 cr_highway extension ---------
    otc_extend = msg_enumerate(
        {
            "dataset": ["bitcoin_otc"],
            "outer_hsikan_n_layers": [2, 4, 8],
            "inner_skip": ["highway", "cr_highway"],
            "grad_checkpoint": [False, True],
        },
        seeds=(0, 1, 2),
        n_epochs=60,
    )

    # --- Follow-up 4: Epinions with smaller Gömb config ----------
    epi_small = msg_enumerate(
        {
            "dataset": ["epinions"],
            "outer_hsikan_n_layers": [2],
            "inner_skip": ["highway"],
            "grad_checkpoint": [True],
        },
        seeds=(0, 1, 2),
        n_epochs=40,   # fewer epochs since cycles are big
    )

    # Tag each candidate with which dataset-config profile to use.
    tagged = (
        [(c, "strict_bench") for c in ba_5seed]
        + [(c, "strict_bench") for c in ba_d16]
        + [(c, "strict_bench") for c in otc_extend]
        + [(c, "small") for c in epi_small]
    )
    print(f"[msg] enumerated {len(tagged)} candidates")

    # ABB prune (use predicted-memory from ArchCandidate; profile
    # adjustment isn't reflected in the predictor — conservative).
    survivors_tagged = []
    pruned = []
    for c, profile in tagged:
        m = c.predicted_peak_mem_gib()
        w = c.predicted_wall_s()
        if m > args.mem_cap_gib:
            pruned.append((c, profile, f"mem {m:.2f} > {args.mem_cap_gib}"))
            continue
        if w > args.wall_cap_s:
            pruned.append((c, profile, f"wall {w:.1f} > {args.wall_cap_s}"))
            continue
        survivors_tagged.append((c, profile))
    print(f"[abb] {len(survivors_tagged)} survivors / "
          f"{len(pruned)} pruned")

    # SSG: prefer ckpt-off within each (dataset, depth, skip, seed)
    # bucket when both survive.
    bucket = {}
    for c, profile in survivors_tagged:
        key = (c.dataset, c.outer_hsikan_n_layers, c.inner_skip,
               c.use_arc_weights, c.seed, profile)
        prev = bucket.get(key)
        if prev is None or c.predicted_wall_s() < prev[0].predicted_wall_s():
            bucket[key] = (c, profile)
    final = list(bucket.values())
    print(f"[ssg] {len(final)} after Pareto filter")

    # Run.
    log_dir = Path(args.out_dir) / time.strftime("%Y%m%dT%H%M%SZ",
                                                    time.gmtime())
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("")

    print(f"[run] log_dir={log_dir}")
    t_total = time.time()
    n_done = n_fail = 0
    for c, profile in final:
        cmd, logf, label = _build_cmd(c, profile, log_dir)
        t0 = time.time()
        rc = subprocess.run(cmd, stdout=logf.open("w"),
                              stderr=subprocess.STDOUT).returncode
        elapsed = time.time() - t0
        result = _harvest(logf)
        if result and rc == 0:
            result["candidate_name"] = label
            result["elapsed_s"] = elapsed
            result["outer_depth"] = c.outer_hsikan_n_layers
            result["inner_skip"] = c.inner_skip
            result["grad_checkpoint"] = c.grad_checkpoint
            result["dataset_profile"] = profile
            with results_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
            auc = result.get("val_auc_best") or result.get("val_auroc")
            print(f"[run] DONE {label:50s} AUC={auc:.4f} actual={elapsed:.1f}s")
            n_done += 1
        else:
            print(f"[run] FAIL {label:50s} rc={rc} actual={elapsed:.1f}s")
            n_fail += 1
    print(f"[run] {n_done}/{n_done + n_fail} done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
