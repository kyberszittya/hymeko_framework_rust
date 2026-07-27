"""K0 positive-control smoke — is a legal receding-horizon teacher a strict-K6 positive control from the frozen KINETIC entry?

Before any KINETIC feedback-policy learning (K1--K4), the teacher that will LABEL the transport segment must be verified to
actually deliver strict K6 when replanned from the state the learner will occupy. This experiment measures that, on dev s1
only, with no state edit / teleport / hidden force / oracle-in-deployment. It compares two start states -- the proven at-rest
straddle HANDOFF (the scout positive control) and the frozen KINETIC ENTRY -- crossed with the teacher budget (canonical theta
executed directly, a full deterministic CEM, and a warm-started local CEM), plus the frozen KINETIC scaffold's own reach for
reference. The output is a verdict, not a controller: it localises WHERE the positive control lives so K1 attaches the
relabeler to the right start.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_positive_control`` (writes the JSON snapshot + prints the verdict table).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, cem_search, delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams, HybridApproachController
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

OUT = Path("reports/2026-07-27-coin-r9-learned-s1-kinetic-delivery-k0/positive_control.json")


def _min_dtz_mm(snap: Any, metrics: dict) -> float:
    """Minimum distance-to-zone over the coin trace (mm). The zone centre is recovered from the frozen start geometry
    (``coin0 + e_par * dtz_start``, exact since ``e_par`` is the unit direction to the zone and ``dtz_start`` its distance)."""
    rl = snap.branch()
    e_par, dtz_start = rl.inner.direction_to_zone()
    coin0 = _coin_xy(rl)
    zone = coin0 + np.asarray(e_par, np.float64)[:2] * float(dtz_start)
    trace = np.asarray(metrics.get("coin_trace", []), np.float64)
    if trace.size == 0:
        return float(metrics.get("dtz_end", 9.0)) * 1000.0
    return float(np.min(np.linalg.norm(trace - zone[None, :], axis=1))) * 1000.0


def _verdict_row(name: str, snap: Any, metrics: dict) -> dict:
    """A single (start, budget) row: strict-K6 delivery + the physical reach the teacher achieved from that start."""
    return {"case": name, "delivers_k6": bool(delivery_success(metrics, DELIVERY_CFG)),
            "k6_max_dwell": int(metrics.get("k6_max_dwell", 0)), "min_dtz_mm": round(_min_dtz_mm(snap, metrics), 2),
            "dtz_end_mm": round(float(metrics.get("dtz_end", 0.0)) * 1000, 2),
            "peak_qdot": round(float(metrics.get("peak_qdot", 0.0)), 3),
            "peak_coin_speed": round(float(metrics.get("peak_coin_speed", 0.0)), 3)}


def _kinetic_scaffold_reach(snap: Any) -> dict:
    """The frozen KINETIC scaffold's own reach from the handoff (reference: G0 report ~48 mm, does not deliver)."""
    ap = ApproachParams(qdot_approach=2.4, acquire_squeeze=0.10, kinetic_transport=True, v_floor=0.26,
                        kinetic_vcap=1.8, kinetic_squeeze=0.10, kinetic_entry_v=0.10, kinetic_max_steps=50)
    c = HybridApproachController(snap, TipTransportParams(), ap, DELIVERY_CFG)
    return velocity_rollout(snap, c, DELIVERY_CFG)


def run() -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    entry = kc.freeze_kinetic_entry(snap)
    rows: list[dict] = []

    # (1) proven positive control reproduction: canonical theta from the AT-REST handoff.
    m = rollout_primitive(snap, kc.S1_CANONICAL_THETA, DELIVERY_CFG)
    rows.append(_verdict_row("handoff+canonical_theta", snap, m))

    # (2) teacher full CEM from the handoff (the scout teacher).
    r = cem_search(snap, DELIVERY_CFG)
    rows.append({**_verdict_row("handoff+full_cem", snap, r["best_metrics"]), "theta": r["best_theta"]})

    # (3) canonical theta from the frozen KINETIC entry (the moving, firm-grip start).
    m = rollout_primitive(entry.tsnap, kc.S1_CANONICAL_THETA, DELIVERY_CFG)
    rows.append(_verdict_row("entry+canonical_theta", entry.tsnap, m))

    # (4) teacher full CEM from the frozen KINETIC entry -- the load-bearing question.
    r = cem_search(entry.tsnap, DELIVERY_CFG)
    rows.append({**_verdict_row("entry+full_cem", entry.tsnap, r["best_metrics"]), "theta": r["best_theta"]})

    # (5) warm-started local CEM from the entry (the K0 relabeler at full budget).
    rl = kc.receding_horizon_relabel(entry.tsnap, budget=1, pop=32, iters=6)
    rows.append({**_verdict_row("entry+warm_local_cem", entry.tsnap, rl.metrics), "theta": rl.theta})

    # (6) reference: the frozen KINETIC scaffold's own reach from the handoff.
    ks = _kinetic_scaffold_reach(snap)
    rows.append(_verdict_row("handoff+kinetic_scaffold", snap, ks))

    verdict = _classify(rows)
    out = {"contract": "COIN_KINETIC_POSITIVE_CONTROL_SMOKE_V1", "seed": kc.S1_SEED,
           "entry": entry.summary(), "rows": rows, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


def _classify(rows: list[dict]) -> str:
    """Where does the strict-K6 positive control live? handoff-yes/entry-yes = REPLAN_FROM_KINETIC_ENTRY_DELIVERS_S1 (a
    CEM replan from the frozen entry delivers -- the interface-side positive control for a LEARNED skill, NOT yet a learned
    delivery: the teacher/CEM is still what runs); handoff-yes/entry-no = POSITIVE_CONTROL_ONLY_FROM_HANDOFF (the entry start
    is out of the teacher basin); handoff-no = TEACHER_NOT_REPRODUCED (investigate before proceeding)."""
    by = {r["case"]: r for r in rows}
    handoff = by.get("handoff+full_cem", {}).get("delivers_k6") or by.get("handoff+canonical_theta", {}).get("delivers_k6")
    entry = by.get("entry+full_cem", {}).get("delivers_k6") or by.get("entry+warm_local_cem", {}).get("delivers_k6")
    if not handoff:
        return "TEACHER_NOT_REPRODUCED"
    return "REPLAN_FROM_KINETIC_ENTRY_DELIVERS_S1" if entry else "POSITIVE_CONTROL_ONLY_FROM_HANDOFF"


if __name__ == "__main__":
    res = run()
    print(f"\nVERDICT: {res['verdict']}  (entry {res['entry']['entry_dtz_mm']}mm @ v_par {res['entry']['entry_v_par']}, "
          f"wall {res['wall_s']}s)\n")
    for r in res["rows"]:
        print(f"  {r['case']:28s}  K6={str(r['delivers_k6']):5s}  dwell={r['k6_max_dwell']:2d}  "
              f"min_dtz={r['min_dtz_mm']:6.1f}mm  peak_qdot={r['peak_qdot']:.2f}  peak_v={r['peak_coin_speed']:.2f}")
