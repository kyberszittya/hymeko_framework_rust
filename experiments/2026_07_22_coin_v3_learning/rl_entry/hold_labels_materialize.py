"""Materialize the RAW per-candidate hold-sweep labels from the FROZEN preregistration (state-manifest SHA verified),
so BENEFICIAL_SUPPORT_AUDIT_V1 can audit them. The completed sweep (config 8d85923, results 5023cf1) is NOT modified —
these labels are the identical deterministic (×2-certified) output that Stage C under-serialized. The captured
observation / base action / causal critic-state / gate state are stored as the AUTHORITATIVE state identity; nothing is
paired against a post-restore recomputed observation.

Writes hold_sweep_v1_labels.json + verifies median|ΔG| per K/family matches the completed results aggregates.
"""
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_counterfactual_labels import capture_state_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_hold_sweep import hold_candidates, residual_hold_return  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/hold_sweep_v1_config.json"
RES = "experiments/2026_07_22_coin_v3_learning/rl_entry/hold_sweep_v1_results.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/hold_sweep_v1_labels.json"
DEV_BANK = (6100, 6200); N_ISO = 3


def _state_manifest_sha(groups):
    rows = [[g.group_id, g.seed, g.family, g.t] for g in groups]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def main():
    cfg = json.load(open(CFG)); res = json.load(open(RES))
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    log = lambda *a: print(*a, flush=True)
    log(f"[{time.strftime('%H:%M:%S')}] recapturing frozen panel (per_family={cfg['per_family']})...")
    groups = capture_state_panel(pi0, range(*DEV_BANK), per_family=cfg["per_family"], label=False)
    sm = _state_manifest_sha(groups)
    assert sm == cfg["state_manifest"]["sha16"], f"state manifest drift {sm} != {cfg['state_manifest']['sha16']}"
    log(f"  state manifest SHA {sm} MATCHES frozen preregistration")
    cands = hold_candidates(N_ISO); K_values = tuple(cfg["K_values"])
    rl = CoinRL4Dof(); out_groups = {}; t0 = time.time()
    for i, g in enumerate(groups):
        rec = {"family": g.family, "seed": g.seed, "t": g.t,
               "identity": {"obs48": [round(float(x), 6) for x in g.obs], "base": [round(float(x), 6) for x in g.base],
                            "causal_state_sha": hashlib.sha256(np.asarray(g.causal_state).tobytes()).hexdigest()[:12],
                            "gate_state": g.cstate}, "K": {}}
        for K in K_values:
            names, dG, cp, te, dwell, ss, mag, direc, gact = ([] for _ in range(9))
            G0 = None
            for name, d, meta in cands:
                G, o = residual_hold_return(rl, pi0, g.snap, g.gate_snap, g.base, d, K)  # 1x (x2 identity certified in sweep)
                if name == "zero":
                    G0 = G
                names.append(name); dG.append(round(G - G0, 6) if G0 is not None else 0.0)
                cp.append(bool(o["contact_persist"])); te.append(bool(o["target_exit"]))
                dwell.append(int(o["max_dwell"])); ss.append(bool(o["strict_success"]))
                mag.append(meta["magnitude"]); direc.append(meta["dir"]); gact.append(int(o["gate_active_steps"]))
            rec["K"][str(K)] = {"names": names, "dG": dG, "contact_persist": cp, "target_exit": te,
                                "max_dwell": dwell, "strict_success": ss, "magnitude": mag, "dir": direc,
                                "gate_active_steps": gact, "G0": round(float(G0), 6),
                                "dwell0": int(dwell[0]), "ss0": bool(ss[0])}
        out_groups[str(g.group_id)] = rec
        if i % 5 == 0 or i == len(groups) - 1:
            log(f"    group {i+1}/{len(groups)} ({g.family}) {time.time()-t0:.0f}s")

    # identity cross-check: median|dG| per K/family must match the completed results aggregates
    ok = True
    for K in K_values:
        for fam in ("transport", "entry", "settling", "contact_retention"):
            dgs = []
            for gid, rec in out_groups.items():
                if rec["family"] == fam:
                    dgs += [abs(x) for x in rec["K"][str(K)]["dG"][1:]]
            med = round(float(np.median(dgs)), 3) if dgs else 0.0
            ref = res["metrics_by_K_family"][str(K)][fam]["median_abs_dG"]
            if abs(med - ref) > 1e-3:
                ok = False; log(f"  MISMATCH K{K} {fam}: {med} != {ref}")
    payload = {"sweep": "RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1", "source": "re-materialized from frozen prereg (deterministic)",
               "state_manifest_sha": sm, "candidate_manifest_sha": res["candidate_manifest_sha"],
               "matches_completed_aggregates": ok, "deterministic_x2_certified_in_sweep": res["deterministic_x2"],
               "K_values": list(K_values), "groups": out_groups}
    json.dump(payload, open(OUT, "w"))
    log(f"\naggregates match completed sweep: {ok}\nwrote {OUT}\nLABELS_MATERIALIZED")


if __name__ == "__main__":
    main()
