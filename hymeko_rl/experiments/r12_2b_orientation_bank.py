"""R12.2-B — orientation-aware θ bank generator.

The pooled bank is axis-aligned-tuned (delivers only near yaw≈0 ⇒ a degenerate K6≈is-yaw-0 label). This builds an
ORIENTATION-AWARE bank: per (yaw × target) certified handoff, the CEM's top-k DELIVERING θ are collected. Because the
feasibility probe proved these θ are orientation-specific (a θ tuned at one yaw fails at others), a single bank of such
θ, applied to a handoff at yaw Y, yields BOTH positives (θ tuned near Y) and negatives (θ tuned far from Y) — the
non-trivial orientation×θ structure R12.2-B needs, with no need to inject random negatives.

Run:  python -m hymeko_rl.experiments.r12_2b_orientation_bank [family] [keep] [pop] [iters] [restarts] [max_seeds]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_2b_theta_feasibility import _TARGETS, _YAWS, _cem_search, _certified_handoff

_OUT = Path("reports/2026-08-08-r12-2-orientation")


def _dedup(entries: list[dict[str, Any]], tol: float) -> list[dict[str, Any]]:
    """Drop near-duplicate θ (keeps the first, which is higher-scored since cells are appended best-first)."""
    kept: list[dict[str, Any]] = []
    for e in entries:
        th = np.asarray(e["theta"])
        if all(np.linalg.norm(th - np.asarray(k["theta"])) > tol for k in kept):
            kept.append(e)
    return kept


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    keep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    restarts = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    max_seeds = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(fam).object_spec)
    t0 = time.perf_counter()
    print(f"R12.2-B orientation-aware bank [{fam}]: {len(_YAWS)}×{len(_TARGETS)} cells, keep {keep} K6 θ/cell", flush=True)

    raw: list[dict[str, Any]] = []
    for yaw in _YAWS:
        for ti, sid in enumerate(_TARGETS):
            snap, _seed = _certified_handoff(rig, scenario_by_id(sid), yaw, cfg, conf, obj, max_seeds)
            if snap is None:
                print(f"  [y{yaw:4.0f}° {sid:20s}] no certified handoff", flush=True)
                continue
            rng = np.random.default_rng(int(yaw) * 100 + ti)
            top, n_eval = _cem_search(snap, pop, iters, restarts, rng, keep=keep)
            k6s = [t for t in top if t["k6"]]
            chosen = k6s if k6s else top[:2]              # delivering θ; if the cell can't deliver, its 2 nearest
            for t in chosen:
                raw.append({"theta": t["theta"], "tuning_yaw": yaw, "target": sid, "k6": bool(t["k6"]),
                            "dtz_mm": round(float(t["dtz_mm"]), 1)})
            print(f"  [y{yaw:4.0f}° {sid:20s}] kept {len(chosen)} (K6={len(k6s)}) of {keep} "
                  f"({n_eval} evals, {time.perf_counter() - t0:.0f}s)", flush=True)

    box_diag = float(np.linalg.norm(THETA_HI - THETA_LO))
    bank = _dedup(raw, 0.03 * box_diag)                   # 3%-of-box distinctness
    by_yaw = {int(y): sum(1 for b in bank if b["tuning_yaw"] == y) for y in _YAWS}
    n_k6 = sum(1 for b in bank if b["k6"])
    summary = {"family": fam, "n_theta": len(bank), "n_k6": n_k6, "by_tuning_yaw": by_yaw,
               "keep": keep, "cem": {"pop": pop, "iters": iters, "restarts": restarts},
               "thetas": [b["theta"] for b in bank], "provenance": bank, "wall_s": round(time.perf_counter() - t0, 1)}
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / f"orientation_bank_{fam}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nBANK [{fam}]: {len(bank)} θ ({n_k6} K6-delivering) after dedup, by tuning-yaw {by_yaw}, "
          f"{summary['wall_s'] / 60:.1f} min → orientation_bank_{fam}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
