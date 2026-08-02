"""R11.4B basin-robustness diagnostic — WHY conditioned BC fails: are the certified teacher theta narrow-basin?

For each scenario, reconstruct the certified handoff and perturb the stored (verified-K6) theta by Gaussian noise at
increasing relative box scales, measuring K6 survival. If survival collapses at a small scale, the teacher schedules are
chaotically sensitive / narrow-basin, so no smooth descriptor->theta policy can reproduce them (a learned policy's finite
prediction error lands off the basin) — this is the mechanism behind BC_REPRESENTATION_INSUFFICIENT, distinguishing
"targets not smoothly learnable" from "model too weak". Parallelizable via --offset/--limit.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import BcSample, bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import rollout_theta, theta_basin_survival

SCALES = (0.005, 0.01, 0.02, 0.05)
DEFAULT_DATASET = Path("reports/2026-08-03-r11-4b-bc/dataset")
DEFAULT_OUT = Path("reports/2026-08-03-r11-4b-bc/basin")


def _load(dataset_dir: Path) -> "list[BcSample]":
    out: list[BcSample] = []
    for f in sorted(glob.glob(str(dataset_dir / "extract_*.jsonl"))):
        for line in Path(f).open():
            if line.strip():
                d = json.loads(line)
                if not d.get("omitted"):
                    out.append(BcSample.from_json(d))
    return out


def _one(ctx: tuple, smp: BcSample, scales: tuple, k: int) -> dict[str, Any]:
    cfg, conf, obj = ctx
    scen = scenario_by_id(smp.scenario_id)
    rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scen, smp.seed)
    if rc is None:
        return {"scenario_id": smp.scenario_id, "split": smp.split, "error": "reconstruction_failed"}
    snap = rc.result.outcome.snapshot
    theta = np.array(smp.theta, np.float64)
    survival = theta_basin_survival(snap, theta, scales, k)
    return {"scenario_id": smp.scenario_id, "split": smp.split, "dtz_mm": smp.dtz_mm,
            "base_k6": bool(rollout_theta(snap, theta)["k6"]),
            "survival": {str(sc): survival[sc] for sc in scales}}


def _read_basin(out_dir: Path) -> list:
    rows: list = []
    for f in sorted(glob.glob(str(out_dir / "basin_*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).open() if line.strip()]
    return [r for r in rows if "error" not in r]


def _summary(out_dir: Path, scales: tuple) -> dict[str, Any]:
    rows = _read_basin(out_dir)
    mean = {str(sc): round(float(np.mean([r["survival"][str(sc)] for r in rows])), 3) for sc in scales}
    narrow = sum(1 for r in rows if r["survival"][str(scales[1])] < 0.5)     # <50% survival at the 2nd (small) scale
    return {"n": len(rows), "base_k6_all": all(r["base_k6"] for r in rows), "mean_survival_by_scale": mean,
            "narrow_basin_count": narrow, "narrow_basin_frac": round(narrow / len(rows), 3) if rows else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("run", "summary"), default="run")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "summary":
        s = _summary(args.out_dir, SCALES)
        (args.out_dir / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(json.dumps(s, indent=2), flush=True)
        print("R11_4B_BASIN_DONE", flush=True)
        return
    ctx = bc_context()
    samples = _load(args.dataset_dir)
    shard = samples[args.offset:(args.offset + args.limit if args.limit else len(samples))]
    rows = [_one(ctx, smp, SCALES, args.k) for smp in shard]
    for r in rows:
        tag = r.get("error") or "  ".join(f"{sc}={r['survival'][sc]}" for sc in r["survival"])
        print(f"{r['scenario_id']:26s} {r['split']:5s} {tag}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"basin_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
