"""R11.5R density-ablation learning curve — does held-out delivery improve as the (robust) training set grows?

The B0-vs-B1 A/B left one fork: the robust targets fixed narrow-basin learnability but the parametric
descriptor->theta map still does not generalize held-out. The held-out geometry (misses farther from the train
manifold, threshold ~nn 2.6) and the imperfect wide-only train fit (0.605) both point to a DENSITY-limited map. This
settles it: subsample the WIDE-recertified train scenarios to k in {10,20,30,38}, refit the SAME BC policies, and
measure the FIXED held-out (dev+test) closed-loop strict-K6 + delivered dtz.

  * held-out RISES with k  -> DENSITY-limited: densifying the (robust) demos is the indicated lever.
  * held-out FLAT with k    -> DESCRIPTOR-limited: more demos at this descriptor won't help (need a better descriptor
    or a retrieval policy form).

The held-out capture snapshots are reconstructed ONCE (the expensive step) and reused across every (k, seed) fit, so
the whole curve is one cheap process. Only the training subset changes; the harness (fit + rollout) is unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import rollout_theta
from hymeko_rl.experiments.r11_4b_conditioned_bc import _fit_policies, _load_dataset

RECERT = Path("reports/2026-08-05-r11-5r-robust-teacher/merged.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-05-r11-5r-density-curve")
K_GRID = (10, 20, 30, 38)
SEEDS = (0, 1, 2)
POLICIES = ("mean_theta", "nearest_schedule", "ridge", "mlp_bc")
_WIDE = "WIDE_RECERTIFIED"


def wide_train_ids(recert: Path) -> "set[str]":
    """Scenario ids of the WIDE-recertified TRAIN demonstrations (the fittable, wide-basin training pool)."""
    rows = json.loads(recert.read_text())["rows"]
    return {r["scenario_id"] for r in rows if r.get("split") == "train" and r.get("status") == _WIDE}


def subsample(pool: "list[Any]", k: int, seed: int) -> "list[Any]":
    """Deterministic size-``min(k, len(pool))`` subset of ``pool`` (order-preserving). ``seed`` varies which subset."""
    n = min(k, len(pool))
    idx = np.random.default_rng(seed).permutation(len(pool))[:n]
    return [pool[i] for i in sorted(int(j) for j in idx)]


def _summ(rows: "list[dict]", policy: str) -> "dict[str, float]":
    """Held-out K6 rate and mean delivered dtz (mm) for one policy over the fixed held-out set."""
    k6 = [r[policy]["k6"] for r in rows if policy in r]
    dtz = [r[policy]["dtz_mm"] for r in rows if policy in r]
    return {"k6_rate": round(float(np.mean(k6)), 3) if k6 else 0.0,
            "dtz_mm": round(float(np.mean(dtz)), 2) if dtz else 0.0}


class HeldoutSnapshots:
    """Reconstruct the fixed held-out (dev+test) capture snapshots ONCE; reused across every (k, seed) fit."""

    def __init__(self, ctx: tuple, held: "list[Any]", limit: int = 0) -> None:
        cfg, conf, obj = ctx
        self.items: list[tuple] = []
        for smp in (held[:limit] if limit else held):
            rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(smp.scenario_id), smp.seed)
            if rc is not None:                                       # a non-reconstructable held-out is skipped (flagged)
                self.items.append((smp, rc.result.outcome.snapshot))

    def evaluate(self, policies: "list[Any]") -> "list[dict]":
        rows = []
        for smp, snap in self.items:
            x = np.array(smp.x, np.float64)
            row: dict[str, Any] = {"scenario_id": smp.scenario_id, "split": smp.split}
            for pol in policies:
                row[pol.name] = rollout_theta(snap, pol.predict(x))
            rows.append(row)
        return rows


def _curve_verdict(agg: "dict[int, dict]") -> str:
    """DENSITY-limited if the PARAMETRIC held-out improves from the smallest to the largest k (K6 up >=0.10 or dtz down
    >=5mm, averaged over ridge+mlp); else DESCRIPTOR-limited."""
    lo, hi = min(agg), max(agg)
    d_k6 = float(np.mean([agg[hi][p]["k6_rate"] - agg[lo][p]["k6_rate"] for p in ("ridge", "mlp_bc")]))
    d_dtz = float(np.mean([agg[lo][p]["dtz_mm"] - agg[hi][p]["dtz_mm"] for p in ("ridge", "mlp_bc")]))
    if d_k6 >= 0.10 or d_dtz >= 5.0:
        return "R11_5R_DENSITY_LIMITED_DENSIFY_INDICATED"
    return "R11_5R_DESCRIPTOR_LIMITED_DENSIFY_UNLIKELY_TO_HELP"


def aggregate(points: "list[dict]") -> "dict[str, Any]":
    """Per-k mean (over seeds) held-out K6 rate + dtz per policy, plus the density verdict."""
    by_k: dict[int, dict] = {}
    for k in sorted({p["k"] for p in points}):
        pts = [p for p in points if p["k"] == k]
        by_k[k] = {pol: {"k6_rate": round(float(np.mean([q["held"][pol]["k6_rate"] for q in pts])), 3),
                         "dtz_mm": round(float(np.mean([q["held"][pol]["dtz_mm"] for q in pts])), 2)}
                   for pol in POLICIES}
    return {"curve": {str(k): by_k[k] for k in by_k}, "n_seeds": len({p["seed"] for p in points}),
            "verdict": _curve_verdict(by_k)}


def _run(args: argparse.Namespace) -> None:
    ctx = bc_context()
    samples = _load_dataset(args.dataset_dir)
    wide = wide_train_ids(args.recert)
    train_pool = [s for s in samples if s.split == "train" and s.scenario_id in wide]
    held = [s for s in samples if s.split in ("dev", "test")]
    snaps = HeldoutSnapshots(ctx, held, limit=args.limit_held)
    print(f"train_pool(wide)={len(train_pool)}  held-out reconstructed={len(snaps.items)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    k_grid = [k for k in K_GRID if k <= len(train_pool)] or [len(train_pool)]
    points: list[dict] = []
    for k in k_grid:
        for seed in SEEDS:
            rows = snaps.evaluate(_fit_policies(subsample(train_pool, k, seed)))
            rec = {"k": k, "seed": seed, "held": {p: _summ(rows, p) for p in POLICIES}}
            points.append(rec)
            print(f"k={k:2d} seed={seed}  " + "  ".join(
                f"{p}={rec['held'][p]['k6_rate']:.2f}/{rec['held'][p]['dtz_mm']:.1f}mm" for p in POLICIES), flush=True)
    result = {"train_pool_wide": len(train_pool), "held_out": len(snaps.items), **aggregate(points), "points": points}
    (args.out / "curve.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== R11.5R DENSITY CURVE ===", flush=True)
    print(json.dumps({"curve": result["curve"], "verdict": result["verdict"]}, indent=2), flush=True)
    print("R11_5R_DENSITY_CURVE_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--recert", type=Path, default=RECERT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit-held", type=int, default=0, help="smoke: reconstruct only the first N held-out scenarios")
    _run(ap.parse_args())


if __name__ == "__main__":
    main()
