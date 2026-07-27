"""LOCAL-ATLAS reconnaissance — DATA ACQUISITION ONLY.

The go-forward reframe (2026-07-27): not "one global KINETIC controller for all cradles" but a LOCAL CONTROL ATLAS + a causal
gate (different local contact regimes → different local torque policies). This run is the cheap reconnaissance for that: scout
N candidate straddle cradles, certify a frozen-CEM teacher θ per deliverable cradle, and — crucially — record the REALIZED
teacher transport PHYSICS (peak/mean v_par, mean/peak Fn, contact fraction, release dtz/v_par, sign reversals, min_dtz, K6), so
the transport MODES can later be discovered from the physics, NOT from the θ vectors (which mislead: the same param gives a
different physical response in a different state).

HARD SCOPE: this file scouts + delivers + profiles + writes ONE versioned artifact, then STOPS. It does NOT build the D1
perturbation feedback-banks, the receding-horizon teacher relabelling, the BC, the strategy clustering, or any validation —
those are the next (fresh-head) steps. s4/s7 (seeds 15000, 15750) stay held-out validation-only; f1-f4 sealed. Reuses
`scout_certified_cradles` / `acquire_snapshot` / `deliver_on_snapshot` / `rollout_primitive` / `_phys_hook` — no new algorithm.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
from hymeko_rl.coin_delivery.theta_option.teacher_bank import (
    FROZEN_SEEDS, acquire_snapshot, deliver_on_snapshot, enumerate_seeds, load_harness, scout_certified_cradles)
from hymeko_rl.experiments.coin_r9_causal_rl import DELIVERY_CFG, _phys_hook

_HELD_OUT_SEEDS = (15000, 15750)                 # s4, s7 — validation-only, must never enter the local-atlas training set
_OUT_DIR = "reports/2026-07-27-coin-local-atlas-scout"


def _sign_reversals(vpar: list) -> int:
    """Count v_par sign flips (>0.02 ↔ <-0.02) — a stiction/bounce signature that distinguishes a smooth transport from a stall."""
    signs = [1 if v > 0.02 else (-1 if v < -0.02 else 0) for v in vpar]
    return sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0)


def _profile_from_trace(tr: list, rel: int) -> dict:
    """Reduce a teacher transport trace to its mechanical fingerprint (extracted to keep _teacher_physics under CC15)."""
    vpar = [r["v_par"] for r in tr]
    fn_contact = [min(r["fn_l"], r["fn_r"]) for r in tr if r["contact"]] or [0.0]
    rel_row = next((r for r in tr if r["t"] == rel), tr[-1])
    return {"peak_vpar": round(max(vpar), 4), "mean_vpar": round(float(np.mean(vpar)), 4),
            "peak_fn": round(max(min(r["fn_l"], r["fn_r"]) for r in tr), 3),
            "mean_fn_contact": round(float(np.mean(fn_contact)), 3),
            "contact_fraction": round(float(np.mean([r["contact"] for r in tr])), 3),
            "release_dtz_mm": round(rel_row["dtz_mm"], 1), "release_vpar": round(rel_row["v_par"], 4),
            "sign_reversals": _sign_reversals(vpar), "min_dtz_mm": round(min(r["dtz_mm"] for r in tr), 1)}


def _teacher_physics(snap: Any, theta: Any) -> dict:
    """Realized transport-physics signature of a certified teacher θ (run it, read the trace) — the mechanical fingerprint the
    local modes will be clustered on (light-slip vs firm-guided …). Release step = θ[4]."""
    tr: list = []
    rollout_primitive(snap, tuple(theta), DELIVERY_CFG, frame_hook=_phys_hook(tr))
    return _profile_from_trace(tr, int(round(float(theta[4])))) if tr else {}


def _atlas_entry(harness: tuple, seed: int) -> "dict | None":
    """Re-acquire the certified snapshot, run the frozen-CEM delivery, and (if deliverable) profile the realized teacher physics."""
    snap, _meta = acquire_snapshot(harness, seed)
    if snap is None:
        return None
    dv = deliver_on_snapshot(snap, basin_seed=seed)
    theta = dv.get("canonical_theta")
    entry = {"seed": seed, "is_frozen_seed": bool(seed in FROZEN_SEEDS), "held_out": bool(seed in _HELD_OUT_SEEDS),
             "deliverable": bool(dv.get("deliverable")), "canonical_theta": theta,
             "canonical_source": dv.get("canonical_source")}
    if entry["deliverable"] and theta is not None and not entry["held_out"]:      # profile dev-eligible deliverable cradles only
        entry["teacher_physics"] = _teacher_physics(snap, theta)
    return entry


def scout_local_atlas(n: int = 24) -> dict:
    """Scout n cradles, certify + deliver + physics-profile each, write ONE versioned artifact, then STOP (no clustering/BC)."""
    t0 = time.time()
    os.makedirs(_OUT_DIR, exist_ok=True)
    harness = load_harness()
    seeds = enumerate_seeds(n)
    print(f"LOCAL-ATLAS SCOUT — {n} candidate cradles (seed=14000+250·si) | held-out {list(_HELD_OUT_SEEDS)} excluded from "
          f"training | reconnaissance only, hard stop after", flush=True)

    def _prog(r: dict, dt: float) -> None:
        print(f"   [scout si={r['si']:2d} seed={r['seed']}] certified={r['certified']} "
              f"n_dot={r['n_dot']} frozen={r['is_frozen_seed']} ({dt:.1f}s)", flush=True)
    rows = scout_certified_cradles(harness, seeds, progress=_prog)
    certified = [r for r in rows if r["certified"]]
    print(f"   scouted {n} → certified {len(certified)}; now frozen-CEM delivery + physics profile per certified cradle…",
          flush=True)
    atlas = []
    for i, r in enumerate(certified):
        entry = _atlas_entry(harness, r["seed"])
        if entry is None:
            continue
        atlas.append(entry)
        ph = entry.get("teacher_physics", {})
        print(f"   [deliver {i + 1}/{len(certified)} seed={entry['seed']}] deliverable={entry['deliverable']} "
              f"held_out={entry['held_out']} | vpar~{ph.get('mean_vpar')} fn~{ph.get('mean_fn_contact')} "
              f"rel_dtz={ph.get('release_dtz_mm')}mm min_dtz={ph.get('min_dtz_mm')}mm", flush=True)
    out = _assemble(n, len(certified), atlas, round(time.time() - t0, 1))
    json.dump(out, open(f"{_OUT_DIR}/scout_n{n}.json", "w"), indent=1, default=float)
    _summary(out)
    return out


def _assemble(n: int, n_certified: int, atlas: list, wall: float) -> dict:
    dev_deliverable = [e for e in atlas if e["deliverable"] and not e["held_out"]]
    return {"contract": "COIN_LOCAL_ATLAS_SCOUT_V1", "scope": "reconnaissance_only_no_clustering_no_bc",
            "n_scouted": n, "n_certified": n_certified, "n_deliverable_dev": len(dev_deliverable),
            "held_out_seeds": list(_HELD_OUT_SEEDS), "blind_status": "f1-f4 SEALED (untouched)",
            "s4_s7_status": "held-out validation-only (excluded from training profiles)", "atlas": atlas, "wall_s": wall}


def _summary(out: dict) -> None:
    dev = [e for e in out["atlas"] if e["deliverable"] and not e["held_out"]]
    print(f"\n== LOCAL-ATLAS SCOUT SUMMARY ==\n  scouted {out['n_scouted']} · certified {out['n_certified']} · "
          f"dev-deliverable {out['n_deliverable_dev']} (s4/s7 held-out, f1-f4 sealed)", flush=True)
    for e in dev:
        ph = e.get("teacher_physics", {})
        th = [round(float(x), 3) for x in e["canonical_theta"]] if e.get("canonical_theta") else None
        print(f"   seed {e['seed']}: θ={th}\n      physics: mean_vpar {ph.get('mean_vpar')} peak_vpar {ph.get('peak_vpar')} "
              f"mean_fn {ph.get('mean_fn_contact')} peak_fn {ph.get('peak_fn')} contact_frac {ph.get('contact_fraction')} "
              f"rel_dtz {ph.get('release_dtz_mm')}mm rel_vpar {ph.get('release_vpar')} min_dtz {ph.get('min_dtz_mm')}mm "
              f"sign_rev {ph.get('sign_reversals')}", flush=True)
    print(f"  artifact: {_OUT_DIR}/scout_n{out['n_scouted']}.json\n  HARD STOP — clustering / feedback-banks / BC are the next "
          f"(fresh-head) steps.\nLOCAL_ATLAS_SCOUT_DONE", flush=True)


def main(argv: "list[str]") -> None:
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 24
    if "--scout" in argv:
        scout_local_atlas(n=n)
    else:
        print("usage: coin_local_atlas_scout.py --scout [--n 24]", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
