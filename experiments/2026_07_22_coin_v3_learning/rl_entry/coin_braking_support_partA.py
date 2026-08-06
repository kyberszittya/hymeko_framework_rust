"""CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 — Part A entry: braking action-support discovery + gate (no training).
Builds the deterministic braking-state snapshot manifest from the 31 dev handoffs, branch-searches bounded candidate
offsets around the exact pi_0 action at each, and emits BRAKING_SAFE_BENEFICIAL_SUPPORT_{FOUND,INSUFFICIENT}."""
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_braking_support import (  # noqa: E402
    SafeBeneficialConfig,
    candidate_offsets,
    capture_braking_states,
    evaluate_state_support,
    support_gate,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart, replay_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
PI0 = f"{D}/frozen/pi0_shared_clip_actor.pt"
TDCFG = f"{D}/transport_dwell_config.json"
V1 = f"{D}/primitive_mpc_qualify_v1.json"
MANIFEST = f"{D}/braking_snapshot_manifest.json"
OUT = f"{D}/braking_support_partA.json"


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _v1_outcome(row):
    """Classify each dev handoff's V1 primitive outcome (for the 'support must reach failing states' gate)."""
    if row["primitive"]["strict_success"] and row["primitive"]["required_contact_retention"] >= 0.5:
        return "delivered_contact_preserving"
    if row["primitive"].get("exit_before_k6"):
        return "target_exit"
    if row["primitive"]["lost_required_contact"] or row["primitive"]["required_contact_retention"] < 0.4:
        return "contact_losing"
    return "no_delivery"


def main():
    cfg = SafeBeneficialConfig(); log = lambda *a: print(*a, flush=True)
    tc = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True)
    dev = [ls for m in CONTROL_MODES for ls in _bank(tc["banks"]["dev"][m])]
    v1 = {(s["seed"], s["prefix"]): _v1_outcome(s) for s in json.load(open(V1))["per_state"]}

    log(f"[{time.strftime('%H:%M:%S')}] capturing braking states from {len(dev)} dev handoffs...")
    caps = capture_braking_states(pi0, dev)
    man = {"n_states": len(caps), "spacing": cfg.branch_horizon, "branch_horizon": cfg.branch_horizon,
           "states": caps, "sha16": hashlib.sha256(json.dumps(caps, sort_keys=True).encode()).hexdigest()[:16]}
    json.dump(man, open(MANIFEST, "w"), indent=1, default=float)
    log(f"  {len(caps)} braking states (manifest sha {man['sha16']})")

    offsets = candidate_offsets()
    log(f"[{time.strftime('%H:%M:%S')}] branch-searching {len(offsets)-1} candidates × {len(caps)} states "
        f"(horizon {cfg.branch_horizon})...")
    t = time.time(); rows = []
    for c in caps:
        rl, _g, _h, _r = replay_pi0(pi0, c["seed"], stop_at=c["abs_step"])       # exact reconstruction
        sup = evaluate_state_support(rl, offsets, pi0, cfg)
        rows.append({"seed": c["seed"], "family": c["family"], "abs_step": c["abs_step"],
                     "v1_outcome": v1.get((c["seed"], c["handoff_prefix"]), "no_delivery"),
                     "dtz": c["dtz"], "pi0_radial_vel": c["radial_vel"], "support": sup})
    log(f"  ({time.time()-t:.0f}s) done")

    gate = support_gate(rows, cfg)
    # support broken down by V1 outcome class (successful vs failing braking states)
    from collections import defaultdict
    byout = defaultdict(lambda: {"n": 0, "with_support": 0, "safe_per": []})
    for r in rows:
        o = byout[r["v1_outcome"]]; o["n"] += 1; o["with_support"] += int(r["support"]["n_safe_beneficial"] > 0)
        o["safe_per"].append(r["support"]["n_safe_beneficial"])
    by_outcome = {k: {"n": v["n"], "with_support": v["with_support"],
                      "mean_safe": round(float(np.mean(v["safe_per"])), 2)} for k, v in byout.items()}

    out = {"campaign": "CONTACT_PRESERVING_BRAKING_PRIMITIVE_V2 Part A", "date": "2026-07-23", "no_training": True,
           "frozen_thresholds": {"branch_horizon": cfg.branch_horizon, "radial_decel_eps": cfg.radial_decel_eps,
                                 "progress_tol": cfg.progress_tol, "meaningful_fraction": cfg.meaningful_fraction,
                                 "min_failing_states_with_support": cfg.min_failing_states_with_support},
           "manifest_sha16": man["sha16"], "n_candidates": len(offsets) - 1, "gate": gate, "by_v1_outcome": by_outcome,
           "per_state": rows, "verdict": gate["verdict"]}
    json.dump(out, open(OUT, "w"), indent=1, default=float)

    log("\n== BRAKING SUPPORT (Part A) ==")
    log(f"  braking states: {gate['n_braking_states']}  with ≥1 safe-beneficial: {round(gate['fraction_with_support']*100)}%  "
        f"mean/state {gate['mean_safe_beneficial_per_state']}")
    log(f"  failing states with support: {gate['n_failing_states_with_support']}/{gate['n_failing_states']}  "
        f"best radial decel {gate['best_radial_decel_overall']} m/s")
    for k, v in by_outcome.items():
        log(f"    {k:30} n={v['n']:<3} with_support={v['with_support']:<3} mean_safe={v['mean_safe']}")
    log(f"\n→ {gate['verdict']}\nwrote {OUT}\nPARTA_DONE")


if __name__ == "__main__":
    main()
