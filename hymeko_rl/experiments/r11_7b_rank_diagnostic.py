"""R11.7B — selection rank diagnostic: is the delivering θ FINDABLE, or a fundamental outlier?

Descriptor-nearest, k3-blend and physics-match all fail to select the delivering θ (which provably exists — the
dense θ×handoff matrix shows 4/6 dev snapshots have one). The decisive question before investing in any smarter
selector: where does the delivering θ RANK among the 66 stored θ under each score? If it ranks ~random/middle, it is
a fundamental transfer outlier (the coin's conclusion — unlearnable), and heuristic selectors are a dead end. If it
ranks near the top (2nd–4th), a refined score could find it.

No deliveries: the delivering-θ indices come from the dense matrix; this only re-acquires the dev descriptors and
ranks the 66 stored θ by each score. Random baseline: a delivering θ would rank at ~(66+1)/(n_delivering+1) by chance.

Run:  python -m hymeko_rl.experiments.r11_7b_rank_diagnostic [n_seeds]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer
from hymeko_rl.coin_delivery.exact_zero_composition import reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_7a_u6b_box_pilot import BOX_DEV, _VARIANT
from hymeko_rl.experiments.r11_7b_physics_selector import _PHYS_IDX, _PHYS_W

_OUT = Path("reports/2026-08-06-r11-7a-u6b-box-pilot")


def _ranks(dist: np.ndarray, delivering: list[int]) -> "tuple[int | None, list[int]]":
    """1-indexed rank (ascending distance) of each delivering θ; returns (best_rank, all_ranks)."""
    order = list(np.argsort(dist))                      # nearest first
    rank_of = {idx: order.index(idx) + 1 for idx in delivering}
    ranks = sorted(rank_of.values())
    return (ranks[0] if ranks else None), ranks


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    bank = json.loads((_OUT / "bank_dense.json").read_text())
    X = np.asarray([s["x"] for s in bank["samples"]], np.float64)
    n_theta = len(X)
    std_full, std_phys = Standardizer.fit(X), Standardizer.fit(X[:, _PHYS_IDX])
    Zf = std_full.transform(X)
    Zp = std_phys.transform(X[:, _PHYS_IDX]) * _PHYS_W

    mat = json.loads((_OUT / "theta_handoff_matrix_dense.json").read_text())
    deliver_idx = {(r["scenario_id"], r["seed"]): [i for i, m in enumerate(r["matrix"]) if m["k6"]]
                   for r in mat["rows"] if r.get("reached_delivery")}

    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    rows: list[dict[str, Any]] = []
    for sid in BOX_DEV:
        for seed in range(n_seeds):
            h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
            if h.record is not None:
                continue
            deliv = deliver_idx.get((sid, seed), [])
            if not deliv:                                # no delivering θ (coverage gap) — nothing to rank
                rows.append({"scenario_id": sid, "seed": seed, "n_delivering": 0})
                print(f"[{sid:22s} s{seed}] no delivering θ (coverage gap)", flush=True)
                continue
            x = np.asarray(h.x, np.float64)
            zf = std_full.transform(x[None, :])[0]
            zp = std_phys.transform(x[None, _PHYS_IDX])[0] * _PHYS_W
            best_n, ranks_n = _ranks(np.linalg.norm(Zf - zf, axis=1), deliv)
            best_p, ranks_p = _ranks(np.linalg.norm(Zp - zp, axis=1), deliv)
            chance = round((n_theta + 1) / (len(deliv) + 1), 1)
            rows.append({"scenario_id": sid, "seed": seed, "n_delivering": len(deliv),
                         "nearest_best_rank": best_n, "physics_best_rank": best_p, "chance_rank": chance,
                         "nearest_ranks": ranks_n, "physics_ranks": ranks_p})
            print(f"[{sid:22s} s{seed}] delivering={len(deliv)}/{n_theta} | best rank: "
                  f"nearest={best_n} physics={best_p} (chance≈{chance})", flush=True)

    res = _summarize(rows, n_theta)
    (_OUT / "rank_diagnostic.json").write_text(json.dumps({"rows": rows, **res}, indent=1, default=str))
    print(f"\n{res['verdict']}: {res['finding']}")
    print(f"  snapshots with a delivering θ: {res['n_with_delivering']} | median best rank: "
          f"nearest={res['median_nearest_best_rank']} physics={res['median_physics_best_rank']} "
          f"(of {n_theta}; top-{res['findable_top_k']} = findable)")
    return 0


def _summarize(rows: list[dict], n_theta: int) -> dict[str, Any]:
    have = [r for r in rows if r.get("n_delivering", 0) > 0]
    nr = [r["nearest_best_rank"] for r in have]
    pr = [r["physics_best_rank"] for r in have]
    med_n = float(np.median(nr)) if nr else None
    med_p = float(np.median(pr)) if pr else None
    findable_k = max(3, n_theta // 10)                   # "near the top" = top-10% (or 3)
    best = min([x for x in (med_n, med_p) if x is not None], default=None)
    if best is None:
        verdict, finding = "RANK_DIAGNOSTIC_INCONCLUSIVE", "no dev snapshot had a delivering θ"
    elif best <= findable_k:
        verdict = "BOX_DELIVERING_THETA_FINDABLE"
        finding = f"a delivering θ ranks within the top {findable_k} under a tested score — a refined selector could find it"
    else:
        verdict = "BOX_DELIVERING_THETA_IS_OUTLIER"
        finding = (f"the delivering θ ranks deep (median best {best:.0f} of {n_theta}, ~chance) under both nearest and "
                   "physics scores — a fundamental transfer outlier, as the coin concluded; heuristic selectors are a dead end")
    return {"verdict": verdict, "finding": finding, "n_with_delivering": len(have),
            "median_nearest_best_rank": med_n, "median_physics_best_rank": med_p, "findable_top_k": findable_k}


if __name__ == "__main__":
    sys.exit(main())
