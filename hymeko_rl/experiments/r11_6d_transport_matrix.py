"""R11.6D Phase 4 — build the full train theta x handoff transfer matrix (the transport signatures come from this).

For every panel handoff (44 train + 7 dev; TEST sealed), reconstruct it once and roll ALL 44 train theta from it,
recording the rich transport metrics per cell. The train rows calibrate + measure retrieval (leave-one-scenario-out);
the dev rows evaluate. Diagnostic build only — no policy is selected here. Fanout by handoff via --offset/--limit.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.transport_retrieval import query_features, roll_full
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-6d-matrix")


def _handoffs(dataset_dir: Path) -> "list[tuple[str, str, int]]":
    """(scenario_id, split, seed) for the 44 train + 7 dev handoffs; TEST is sealed (excluded)."""
    return [(s.scenario_id, s.split, int(s.seed)) for s in _load_dataset(dataset_dir) if s.split in ("train", "dev")]


def _slice(seq: list, offset: int, limit: int) -> list:
    return seq[offset:(offset + limit if limit else len(seq))]


def _run_eval(args: argparse.Namespace) -> None:
    fp = json.loads(args.frozen.read_text())
    theta_ids, thetas = fp["table"]["scenario_ids"], [np.asarray(t, np.float64) for t in fp["table"]["theta"]]
    cfg, conf, obj = bc_context()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"matrix_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for hid, split, seed in _slice(_handoffs(args.dataset_dir), args.offset, args.limit):
            rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(hid), seed)
            if rc is None:
                fh.write(json.dumps({"handoff": hid, "split": split, "error": "reconstruction_failed"}) + "\n")
                continue
            snap = rc.result.outcome.snapshot
            qf = query_features(snap)
            for tid, theta in zip(theta_ids, thetas):
                cell = {"handoff": hid, "split": split, "theta": tid, "bearing": qf["bearing"], **roll_full(snap, theta)}
                fh.write(json.dumps(cell) + "\n")
            fh.flush()
            print(f"{hid:26s} {split:5s} d_req={qf['d_required_mm']:.1f}mm bearing={qf['bearing']:.2f} rolled {len(thetas)}",
                  flush=True)


def _run_merge(args: argparse.Namespace) -> None:
    cells: list[dict] = []
    for f in sorted(glob.glob(str(args.out / "matrix_*.jsonl"))):
        cells += [json.loads(line) for line in Path(f).open() if line.strip()]
    good = [c for c in cells if "error" not in c]
    handoffs = sorted({c["handoff"] for c in good})
    thetas = sorted({c["theta"] for c in good})
    (args.out / "matrix.json").write_text(json.dumps({"cells": good, "handoffs": handoffs, "thetas": thetas},
                                                     indent=2), encoding="utf-8")
    k6 = sum(1 for c in good if c["k6"])
    print(f"matrix: {len(handoffs)} handoffs x {len(thetas)} theta = {len(good)} cells; {k6} strict-K6 cells", flush=True)
    print("R11_6D_MATRIX_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("eval", "merge"), required=True)
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"eval": _run_eval, "merge": _run_merge}[args.phase](args)


if __name__ == "__main__":
    main()
