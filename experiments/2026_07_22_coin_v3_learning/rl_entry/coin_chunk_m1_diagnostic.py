"""CHUNK_SUPERVISED_M1_FEEDBACK_V1 — isolated execution-horizon diagnostic (no TD3). Reproduces the exact V2 supervised
chunk actor (same dataset/DAgger/prefix-weighted regression) and evaluates the SAME actor with executed prefix M=1 vs
M=2 vs frozen pi_0. Only the execution horizon changes; the actor still predicts K=8 (item 7 — no retraining)."""
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_chunk_td3 import chunk_metrics, eval_receding_horizon  # noqa: E402
from hymeko_rl.coin_delivery.coin_feedback_chunk_v2 import reproduce_v2_actor  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
TDCFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/chunk_m1_diagnostic.json"


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main():
    cfg = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True); log = lambda *a: print(*a, flush=True)
    train = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["train"][m])]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    horizon = cfg["horizon"]
    log(f"[{time.strftime('%H:%M:%S')}] reproducing V2 actor...")
    t = time.time(); actor, X, Y, stats = reproduce_v2_actor(pi0, train, horizon=horizon, log=log)
    log(f"  ({time.time()-t:.0f}s) first-action MSE {chunk_metrics(actor, X, Y)['first_action_mse']}")

    log(f"[{time.strftime('%H:%M:%S')}] eval M=2 and M=1 (same actor) vs pi_0 on {len(dev)} dev starts...")
    ev2 = eval_receding_horizon(pi0, actor, dev, horizon=horizon, m=2)
    ev1 = eval_receding_horizon(pi0, actor, dev, horizon=horizon, m=1)
    for tag, ev in (("M2", ev2), ("M1", ev1)):
        dl = ev["delta_vs_pi0"]
        log(f"  {tag} vs pi_0: Δcontact {dl['contact_retention']:+.3f} Δexit {dl['exited']:+.3f} Δdwell {dl['max_dwell']:+.2f} "
            f"Δstrict {dl['strict_success']:+.3f} Δprogress {dl['progress']:+.4f}  | chunk contact {ev['chunk']['contact_retention']}")

    d1 = ev1["delta_vs_pi0"]; c1, c2, cp = ev1["chunk"]["contact_retention"], ev2["chunk"]["contact_retention"], ev1["pi0"]["contact_retention"]
    gate = (d1["contact_retention"] >= -0.05 and d1["exited"] <= 0.05 and d1["max_dwell"] >= -0.10 and d1["strict_success"] >= -0.05)
    if gate:
        verdict = "M1_FEEDBACK_RECOVERS_CONTACT"
    elif c1 > c2 + 0.05:                                             # M=1 materially better than M=2 but not yet at pi_0
        verdict = "M1_FEEDBACK_PARTIALLY_RECOVERS_CONTACT"
    else:
        verdict = "M1_FEEDBACK_NO_GAIN"

    out = {"contract": "CHUNK_SUPERVISED_M1_FEEDBACK_V1", "date": "2026-07-23", "pi0_sha": cfg["pi0_sha"], "no_td3": True,
           "changed_only": "executed prefix M: 2 -> 1 (K=8 unchanged, same V2 actor)",
           "dataset_provenance": stats["by_provenance"],
           "contact_retention": {"M1_chunk": c1, "M2_chunk": c2, "pi0": cp},
           "eval_M1_vs_pi0": ev1, "eval_M2_vs_pi0": ev2, "acceptance_gate_M1": bool(gate), "verdict": verdict}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    log(f"\ncontact: M1 {c1} vs M2 {c2} vs pi0 {cp}  gate_M1={gate}  → {verdict}\nwrote {OUT}\nM1_DIAGNOSTIC_DONE")


if __name__ == "__main__":
    main()
