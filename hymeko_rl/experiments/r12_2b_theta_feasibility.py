"""R12.2-B feasibility probe — is there a real orientation×θ interaction? (capture-independent)

The decisive go/no-go before investing in the full orientation-aware dataset. From CERTIFIED handoff snapshots (so
orientation-dependent GRASPABILITY does not confound the DELIVERY question), a small CEM searches θ-space per
(yaw × target) cell for a delivering θ, then a yaw×θ CROSS-TRANSFER matrix applies each yaw's best θ to every yaw's
handoff. Because `_delivery_signals` rolls out delivery from an already-certified grasp, θ only drives delivery — the
grasp problem is isolated out.

GO for the full dataset only if ALL hold: (i) a K6 θ exists in ≥3/4 yaws; (ii) the delivering θ genuinely DIFFER across
yaws (not one-θ-fits-all); (iii) cross-yaw transfer is NON-TRIVIAL (θ(yaw=a) works at a, degrades at b). The ideal:
θ0 works@0° fails@60° and θ60 works@60° fails@0° ⇒ real orientation×action interaction ⇒ the ranker question (MLP vs
random-sparse vs task-HSiKAN vs Steiner) is meaningful. One-θ-fits-all ⇒ no structural signal. CEM finds nothing at
60/90° ⇒ a controller/action-coverage problem, not a ranker question.

Run:  python -m hymeko_rl.experiments.r12_2b_theta_feasibility [family] [pop] [iters] [restarts] [max_seeds]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, clip_theta, _from01
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.r12_orientation import object_yaw, reach_capture_at_yaw
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_YAWS = (0.0, 30.0, 60.0, 90.0)
# 3 representative target geometries (distinct delivery zones): near / lateral / far — from the scenario bank.
_TARGETS = ("bank_c2_+0.015_+0.000", "bank_c2_+0.015_-0.015", "bank_c3_r7_a-15")
_K6_BONUS = 1.0e6                        # CEM prefers any K6, then minimal dtz


def _score(sig: Any) -> float:
    return (_K6_BONUS if sig.k6 else 0.0) - float(sig.dtz_mm)


def _cem_theta(snap: Any, pop: int, iters: int, restarts: int, rng: np.random.Generator) -> dict[str, Any]:
    """CEM over the 6-D certified θ-box from a certified handoff snap. Returns the best θ + its delivery signals."""
    dim = int(THETA_LO.shape[0])
    best: dict[str, Any] = {"score": -math.inf}
    n_eval = 0
    for _r in range(restarts):
        mu, sigma = np.full(dim, 0.5), np.full(dim, 0.35)               # unit-space Gaussian
        for _it in range(iters):
            z = np.clip(rng.normal(mu, sigma, size=(pop, dim)), 0.0, 1.0)
            evals = []
            for zi in z:
                sig = _delivery_signals(snap, _from01(zi))
                n_eval += 1
                s = _score(sig)
                evals.append((s, zi, sig))
                if s > best["score"]:
                    best = {"score": s, "theta": _from01(zi).tolist(), "k6": bool(sig.k6),
                            "dtz_mm": float(sig.dtz_mm), "gap_closed": float(sig.gap_closed),
                            "contact_lost_steps": int(sig.contact_lost_steps), "safe": bool(sig.safe)}
            evals.sort(key=lambda e: e[0], reverse=True)
            elite = np.array([e[1] for e in evals[:max(2, pop // 4)]])
            mu, sigma = elite.mean(0), elite.std(0) + 1e-3
    best["n_eval"] = n_eval
    return best


def _certified_handoff(rig: dict[str, Any], scen: Any, yaw: float, cfg: Any, conf: Any, obj: Any,
                       max_seeds: int) -> "tuple[Any, int] | tuple[None, int]":
    """First seed whose yaw-varied reach/capture certifies (isolates the grasp problem: we only study delivery where a
    certified grasp EXISTS). Returns (snap, seed) or (None, tries)."""
    for seed in range(max_seeds):
        h = reach_capture_at_yaw(rig, scen, seed, math.radians(yaw), cfg, conf, obj)
        if h.record is None:
            return h.snap, seed
    return None, max_seeds


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    max_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(fam).object_spec)
    t0 = time.perf_counter()
    print(f"R12.2-B θ-feasibility [{fam}]: {len(_YAWS)}×{len(_TARGETS)} cells, CEM pop{pop}×it{iters}×r{restarts}",
          flush=True)

    cells: list[dict[str, Any]] = []
    cell_snap: dict[tuple[float, str], Any] = {}          # (yaw, target) → its certified handoff snap
    cell_theta: dict[tuple[float, str], list[float]] = {}  # (yaw, target) → CEM best θ (tuned on THAT handoff)
    for yaw in _YAWS:
        for ti, sid in enumerate(_TARGETS):
            snap, seed = _certified_handoff(rig, scenario_by_id(sid), yaw, cfg, conf, obj, max_seeds)
            if snap is None:
                cells.append({"yaw": yaw, "target": sid, "certified": False})
                print(f"  [y{yaw:4.0f}° {sid:20s}] NO CERTIFIED HANDOFF in {max_seeds} seeds", flush=True)
                continue
            rng = np.random.default_rng(int(yaw) * 100 + ti)
            b = _cem_theta(snap, pop, iters, restarts, rng)
            cell_snap[(yaw, sid)] = snap
            cell_theta[(yaw, sid)] = b["theta"]
            cells.append({"yaw": yaw, "target": sid, "certified": True, "seed": seed,
                          "post_grasp_yaw_deg": round(math.degrees(object_yaw(snap)), 3), **b})
            print(f"  [y{yaw:4.0f}° {sid:20s}] K6={b['k6']} dtz={b['dtz_mm']:.1f}mm gap={b['gap_closed']:.2f} "
                  f"({b['n_eval']} evals, {time.perf_counter() - t0:.0f}s)", flush=True)

    # YAW-ISOLATED cross-transfer: PER TARGET, apply (yaw_a, target) θ to (yaw_b, SAME target) handoff — so ONLY the
    # orientation differs (same coin position + same delivery zone). A confound-free test of orientation×θ.
    per_target: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    interactions: list[tuple[str, int, int]] = []        # (target, θ-yaw delivers here, but fails at that yaw)
    for sid in _TARGETS:
        yh = [y for y in _YAWS if (y, sid) in cell_theta]
        mat: dict[int, dict[int, dict[str, Any]]] = {}
        for ya in yh:
            row: dict[int, dict[str, Any]] = {}
            for yb in yh:
                sig = _delivery_signals(cell_snap[(yb, sid)], clip_theta(np.asarray(cell_theta[(ya, sid)])))
                row[int(yb)] = {"k6": bool(sig.k6), "dtz_mm": round(float(sig.dtz_mm), 1)}
            mat[int(ya)] = row
        per_target[sid] = mat
        for ya in yh:                                     # orientation-specific θ: delivers at own yaw, fails elsewhere
            if mat[int(ya)][int(ya)]["k6"]:
                for yb in yh:
                    if yb != ya and not mat[int(ya)][int(yb)]["k6"]:
                        interactions.append((sid, int(ya), int(yb)))

    # θ-spread across the per-yaw delivering θ (secondary diagnostic)
    k6_thetas = [cell_theta[k] for k, c in [((c["yaw"], c["target"]), c) for c in cells if c.get("certified")]
                 if c["k6"]]
    theta_box_diag = float(np.linalg.norm(THETA_HI - THETA_LO))
    theta_spread = float(np.max([np.linalg.norm(np.array(a) - np.array(b))
                                 for a in k6_thetas for b in k6_thetas])) if len(k6_thetas) > 1 else 0.0

    yaws_with_k6 = sorted({c["yaw"] for c in cells if c.get("certified") and c.get("k6")})
    go = len(yaws_with_k6) >= 3 and len(interactions) > 0

    verdict = ("GO — real orientation×θ interaction: K6 θ in ≥3 yaws AND ≥1 orientation-specific θ (delivers at its yaw, "
               "FAILS at another, SAME target) ⇒ full orientation-aware dataset + ranker test warranted" if go else
               "NO-GO — " + ("CEM found no K6 at ≥2 yaws (controller/action-coverage limit, not a ranker question)"
                             if len(yaws_with_k6) < 3 else
                             "one-θ-fits-all: every delivering θ also transfers across yaws (no orientation-specific θ) "
                             "⇒ no structural orientation signal in this substrate"))
    print(f"\nyaws with K6 θ: {yaws_with_k6}  |  θ-spread {theta_spread:.2f}/{theta_box_diag:.2f}  |  "
          f"orientation-specific θ (interactions): {interactions}", flush=True)
    for sid, mat in per_target.items():
        print(f"per-target yaw×yaw K6 matrix [{sid}] (row θ-yaw → col handoff-yaw, SAME target):", flush=True)
        for ya, row in mat.items():
            cs = [f"h{yb}:" + ("K6" if v["k6"] else f"{v['dtz_mm']:.0f}") for yb, v in row.items()]
            print(f"  θ{ya:>3}  " + "  ".join(cs), flush=True)
    print(f"\n⇒ {verdict}", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    summary = {"family": fam, "cem": {"pop": pop, "iters": iters, "restarts": restarts}, "cells": cells,
               "per_target_matrix": {sid: {str(ya): {str(yb): v for yb, v in row.items()} for ya, row in mat.items()}
                                     for sid, mat in per_target.items()},
               "interactions": interactions, "yaws_with_k6": yaws_with_k6,
               "theta_spread": round(theta_spread, 4), "theta_box_diag": round(theta_box_diag, 4),
               "go": go, "verdict": verdict, "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "b_theta_feasibility.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote b_theta_feasibility.json ({summary['wall_s'] / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
