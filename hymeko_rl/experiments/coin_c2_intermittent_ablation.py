"""C2 (intermittent) — coin option-language ablation on the FROZEN V4_INTERMITTENT_CONTACT dynamics, with the option
language that FITS the coin's real contact class (push-and-coast). Supersedes the continuous-transport C2
(``coin_c2_ablation.py``, arms feedback/brake/replan) which fought the physics. Same panel/seeds/V4/governor for every arm;
the primary result is the PHASE LADDER (acquire → transport distance → zone entry → K6), not final K6.

Arms — progressive intermittent options (each adds ONE element to the one before):
  A legacy          — the open-loop push→brake→release macro (expert search) under V4 (reference)
  B impulse+coast   — a SINGLE short bounded impulse toward the zone, then release + coast (no re-contact/brake/settle)
  C +re-contact     — B + re-acquire after a stalled coast (many short impulses)
  D +brake          — C + velocity-aware braking near the zone
  E +settle         — D + low-speed settling in the zone (the full intermittent controller)

No RL. No proposal refit. No dynamics tuning. K6/zone are outputs, never inputs to the controller.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")

from hymeko_rl.coin_delivery.intermittent_carry import IntermittentConfig, intermittent_carry  # noqa: E402
from hymeko_rl.experiments.coin_c2_ablation import EVAL_H, _coin_progress, _legacy_arm, _stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"


def _intermittent_arm(rl, gate, pi0, base, stack, cfg: IntermittentConfig):
    """One intermittent-controller arm. intermittent_carry sets V4 dynamics + governor itself and reports the phase
    ladder + delivery + realised motion. Transport distance is re-derived from a coin-progress hook for parity with the
    legacy arm's measurement."""
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    u, _n = rl.inner.direction_to_zone()
    prog = {"max": 0.0}

    def hook(_ph, _s):
        prog["max"] = max(prog["max"], _coin_progress(r2, disk0, np.asarray(u, np.float32)))
    r2 = copy.deepcopy(rl)
    o = intermittent_carry(r2, copy.deepcopy(gate), pi0, base, stack, horizon=EVAL_H, cfg=cfg, frame_hook=hook)
    return {"k6": int(o["k6"]), "acquired": int(o["acquired_contact"]), "transport_dist": round(prog["max"], 4),
            "zone_entry": int(o["entered_zone"]), "max_dwell": int(o["max_dwell"]), "episodes": o["n_contact_episodes"],
            "peak_vel": o["peak_joint_vel"], "peak_coin": o["peak_coin_speed"]}


ARMS = {
    "A_legacy": lambda rl, g, p, b, s: _legacy_arm(rl, g, p, b, s, slow=False),
    "B_impulse_coast": lambda rl, g, p, b, s: _intermittent_arm(rl, g, p, b, s, IntermittentConfig(
        enable_recontact=False, enable_brake=False, enable_settle=False)),
    "C_recontact": lambda rl, g, p, b, s: _intermittent_arm(rl, g, p, b, s, IntermittentConfig(
        enable_recontact=True, enable_brake=False, enable_settle=False)),
    "D_brake": lambda rl, g, p, b, s: _intermittent_arm(rl, g, p, b, s, IntermittentConfig(
        enable_recontact=True, enable_brake=True, enable_settle=False)),
    "E_settle": lambda rl, g, p, b, s: _intermittent_arm(rl, g, p, b, s, IntermittentConfig(
        enable_recontact=True, enable_brake=True, enable_settle=True)),
}


def _verdict(td, ze, k6):
    """Pre-registered interpretation on the phase ladder (transport distance / zone entry / K6). Which intermittent
    element FIRST bears the transport is the mechanism."""
    base_td = td["A_legacy"]
    if k6["E_settle"] > 0.3 and k6["E_settle"] > k6["D_brake"] + 0.15:
        return "LOW_SPEED_SETTLING_REQUIRED_FOR_STABLE_DELIVERY"
    if k6["D_brake"] > 0.3 and k6["D_brake"] > k6["C_recontact"] + 0.15:
        return "VELOCITY_AWARE_BRAKING_LOAD_BEARING_UNDER_INTERMITTENT_CONTACT"
    if ze["C_recontact"] > max(ze["A_legacy"], ze["B_impulse_coast"]) + 0.15:
        return "RECONTACT_REQUIRED_FOR_TRANSPORT"
    if td["B_impulse_coast"] > base_td + 0.02:
        return "IMPULSE_COAST_ALREADY_IMPROVES_TRANSPORT_OVER_LEGACY"
    return "INTERMITTENT_OPTION_LANGUAGE_INSUFFICIENT_UNDER_REALISTIC_CONTACT_DYNAMICS"


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    stack, v4 = _stack()
    assert v4["dynamics_contract"] == "COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT", v4["dynamics_contract"]
    n_states = 6 if smoke else 16
    rows = []
    for si in range(n_states):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 250 * si, tries=3)
        rec = {"state": si}
        for name, fn in ARMS.items():
            rec[name] = fn(rl, gate, pi0, base, stack)
        rows.append(rec)
        print("  s%d: " % si + " | ".join(
            f"{k.split('_')[0]} acq{rec[k]['acquired']} td{rec[k]['transport_dist']} zone{rec[k]['zone_entry']} k6{rec[k]['k6']}"
            for k in ARMS), flush=True)

    def agg(name, key):
        return round(float(np.mean([r[name][key] for r in rows])), 3)
    summary = {name: {k: agg(name, k) for k in ("acquired", "transport_dist", "zone_entry", "k6")} for name in ARMS}
    td = {n: summary[n]["transport_dist"] for n in ARMS}
    ze = {n: summary[n]["zone_entry"] for n in ARMS}
    k6 = {n: summary[n]["k6"] for n in ARMS}
    verdict = _verdict(td, ze, k6)
    manifest = {"contract": "COIN_C2_INTERMITTENT_OPTION_LANGUAGE_ABLATION", "date": "2026-07-25",
                "dynamics_contract": v4, "n_states": n_states, "summary": summary, "rows": rows, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/c2_intermittent_ablation.json", "w"), indent=1, default=float)
    print("\n== C2 (intermittent) phase ladder (mean) ==")
    for n in ARMS:
        print(f"  {n:18s} acquire {summary[n]['acquired']} transport-dist {summary[n]['transport_dist']} "
              f"zone-entry {summary[n]['zone_entry']} K6 {summary[n]['k6']}")
    print(f"\n  → {verdict}\n  artifact: {OUT}/c2_intermittent_ablation.json\nC2_INTERMITTENT_DONE")
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
