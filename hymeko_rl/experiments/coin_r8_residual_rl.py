"""R8 residual-RL harness (corrected gate) — S2 update-zero identity, S3 learnability audit, S4 matched SAC/TD3, S5 held-out.

One harness, mode flags (§6.5 #13). Reuses the frozen R8 tip-referenced scaffold (`tip_transport`) + the bounded residual
adapter (`residual_adapter`) + the frozen `velocity_rollout` physics + the R6 release certificate. Every integrity
constraint of `reports/2026-07-27-coin-r8-corrected-rl-gate-contract.md` is kept hard (no teleport / hidden force / teacher
fallback / oracle injection; exact Bellman provenance = the actor emission only; held-out s4/s7 excluded from all
training/tuning/selection; oracle a feasibility witness only). S2 runs on the development cradles.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, _coin_xy, delivery_success
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    RESIDUAL_ROLES, ResidualBounds, ResidualTipAdapter, ZeroActor)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipReferencedController, TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

OUT = "reports/2026-07-27-coin-r8-residual-rl"
REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"
DEV_TAGS = ("s1", "s3")                                 # development cradles (held-out s4/s7 excluded from all of S2-S4)
_METRIC_KEYS = ("dtz_start", "dtz_end", "forward", "cross", "peak_qdot", "peak_coin_speed", "terminal_coin_speed",
                "k6_max_dwell", "contact_lost_steps", "lost_before_release", "release_step", "gap_closed")


def _rich_trace(snap: Any, controller: Any, cfg: Any = DELIVERY_CFG) -> "tuple[dict, list]":
    """Roll a controller and capture a per-step physical trace (coin pose/velocity, joint velocity, contact fₙ, dtz) via a
    frame_hook — the full-trace comparison surface for the update-zero identity."""
    rows: list[dict[str, Any]] = []

    def hook(rl: Any, t: int) -> None:
        d = rl.inner.data
        from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
        con = primary_fingertip_contacts(rl)
        _u, dtz = rl.inner.direction_to_zone()
        rows.append({"t": int(t), "coin": np.asarray(_coin_xy(rl), np.float64).copy(),
                     "coin_vel": np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2].copy(),
                     "qvel": np.asarray(d.qvel[:4], np.float64).copy(), "dtz": float(dtz), "speed": float(_coin_speed(rl)),
                     "fn_l": float(con["left"]["fn"]) if con["left"] else 0.0,
                     "fn_r": float(con["right"]["fn"]) if con["right"] else 0.0})

    m = velocity_rollout(snap, controller, cfg, frame_hook=hook)
    return m, rows


def _max_diff(a: "list[dict]", b: "list[dict]") -> "dict[str, float]":
    """Max absolute difference of the per-step physical arrays between two rich traces (must be equal length)."""
    keys = ("coin", "coin_vel", "qvel")
    scal = ("dtz", "speed", "fn_l", "fn_r")
    out = {k: 0.0 for k in (*keys, *scal)}
    for ra, rb in zip(a, b):
        for k in keys:
            out[k] = max(out[k], float(np.max(np.abs(ra[k] - rb[k]))))
        for k in scal:
            out[k] = max(out[k], abs(ra[k] - rb[k]))
    return out


def _s2_state(snap: Any, params: Any, bounds: Any, tol: float) -> dict:
    """Compare the frozen scaffold vs the zero-residual adapter on ONE cradle over the full physical trace + provenance."""
    m_base, tr_base = _rich_trace(snap, TipReferencedController(snap, params, DELIVERY_CFG))
    adapter = ResidualTipAdapter(snap, ZeroActor(), params, bounds, DELIVERY_CFG)
    m_adpt, tr_adpt = _rich_trace(snap, adapter)
    diffs = _max_diff(tr_base, tr_adpt)
    metric_diff = {k: abs(float(m_base[k]) - float(m_adpt[k])) for k in _METRIC_KEYS}
    coin_equal = bool(np.array_equal(np.asarray(m_base["coin_trace"]), np.asarray(m_adpt["coin_trace"])))
    prov_ok = all(all(abs(x) < tol for x in p["residual"]) and abs(p["corrected_qref"] - p["base_qref"]) < tol
                  and abs(p["corrected_sqz"] - p["base_sqz"]) < tol and abs(p["kv"] - params.k_v) < tol
                  and all(abs(x) < tol for x in p["bellman_action"]) and not p["clip_flags"]["a"]
                  for p in adapter.provenance)
    trace_ok = coin_equal and max(diffs.values()) < tol and max(metric_diff.values()) < tol   # release_step ∈ metrics
    return {"split": "development", "trace_identity": trace_ok, "provenance_zero_effect": prov_ok,
            "coin_trace_bit_equal": coin_equal, "max_step_diff": {k: round(v, 12) for k, v in diffs.items()},
            "max_metric_diff": {k: round(v, 12) for k, v in metric_diff.items()}, "n_steps": len(tr_base),
            "delivery_scaffold": bool(delivery_success(m_base, DELIVERY_CFG)),
            "release_step_base": m_base["release_step"], "release_step_adapter": m_adpt["release_step"],
            "_max": max(max(diffs.values()), max(metric_diff.values()))}


def s2_update_zero_identity(smoke: bool = False) -> dict:
    """S2 — the zero-residual adapter reproduces the frozen scaffold over the FULL trace (dev cradles). # Postconditions:
    writes s2_update_zero_identity.json; verdict UPDATE_ZERO_RESIDUAL_IDENTITY_{PASS,FAILS}."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    params, bounds, tol = TipTransportParams(), ResidualBounds(), 1e-9
    per: dict[str, Any] = {}
    for tag in DEV_TAGS:
        st = _s2_state(panel[tag].snap, params, bounds, tol)
        mx = st.pop("_max")
        per[tag] = st
        print(f"   {tag}: trace_identity={st['trace_identity']} prov_zero={st['provenance_zero_effect']} "
              f"coin_equal={st['coin_trace_bit_equal']} max_diff={mx:.2e}", flush=True)
    passed = all(v["trace_identity"] and v["provenance_zero_effect"] for v in per.values())
    verdict = "UPDATE_ZERO_RESIDUAL_IDENTITY_PASS" if passed else "UPDATE_ZERO_RESIDUAL_IDENTITY_FAILS"
    out = {"contract": "COIN_R8_S2_UPDATE_ZERO_IDENTITY", "date": "2026-07-27", "tolerance": tol,
           "residual_roles": list(RESIDUAL_ROLES),
           "residual_bounds": {"d_fwd_vel": bounds.d_fwd_vel, "d_squeeze": bounds.d_squeeze, "d_stop_gain": bounds.d_stop_gain,
                               "kv_lo": bounds.kv_lo, "kv_hi": bounds.kv_hi},
           "bellman_action": "actor emission a in [-1,1]^3 ONLY; base/corrected/clipped/torque = provenance",
           "per_state": per, "passed": passed, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/s2_update_zero_identity.json", "w"), indent=1, default=float)
    print(f"\n== S2 ==\n  {verdict} | dev {sum(v['trace_identity'] and v['provenance_zero_effect'] for v in per.values())}/"
          f"{len(per)} | wall {out['wall_s']}s\nR8_S2_DONE", flush=True)
    return out


def _load_harness() -> Any:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    return load_harness()


def main(argv: "list[str]") -> None:
    if "--s2" in argv:
        s2_update_zero_identity(smoke="--smoke" in argv)
    else:
        print("usage: coin_r8_residual_rl.py --s2 | (S3/S4/S5 modes added as gates pass)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
