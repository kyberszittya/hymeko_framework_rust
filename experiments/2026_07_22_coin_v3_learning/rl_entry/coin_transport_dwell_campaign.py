"""TRANSPORT_TO_DWELL_TD3_BASELINE_V1 — one bounded campaign (Arm A). Re-scoped ontology {transport,braking,
settling_dwell} + target-entry event features + contact flag; horizon 60; 50/30/20 balanced sampling. Everything else
(pi_0, reward, gate, 4-step, smoothing, exploration, transactional caps, term/trunc, eval) UNCHANGED. No old
target_entry curriculum, no Stage 2, no Arm B, no SAC, no neutral-reset, no final-test."""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart, late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES, train_transport_dwell  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = sys.argv[2] if len(sys.argv) > 2 else "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_results.json"


def _bank(m):
    b = [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]
    assert late_start_bank_manifest(b)["sha16"] == m["sha16"], "bank SHA drift"
    return b


def main():
    cfg = json.load(open(CFG)); tcfg = TransactionalConfig()
    assert cfg["meets_thresholds"], "bank thresholds not met — refuse to train"
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    train = {m: _bank(cfg["banks"]["train"][m]) for m in CONTROL_MODES}
    dev_flat = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    log = lambda *a: print(*a, flush=True)
    launch = {
        "campaign": cfg["campaign"], "config_sha": cfg["config_sha"], "pi0_sha": cfg["pi0_sha"],
        "exact_command": f"python experiments/2026_07_22_coin_v3_learning/rl_entry/coin_transport_dwell_campaign.py {PI0} {CFG} {OUT}",
        "rng_seeds": cfg["rng_seeds"], "ontology": cfg["ontology"], "horizon": cfg["horizon"], "n_step": cfg["n_step"],
        "sample_target": cfg["sample_target"], "bank_counts": cfg["train_bank_counts"],
        "bank_shas": {"train": {m: cfg["banks"]["train"][m]["sha16"] for m in CONTROL_MODES},
                      "dev": {m: cfg["banks"]["dev"][m]["sha16"] for m in CONTROL_MODES}},
        "stage": cfg["stage"], "transactional_UNCHANGED": tcfg.manifest(), "unchanged": cfg["unchanged"],
        "stop_conditions": cfg["stop_conditions"], "constraints": cfg["no"],
    }
    json.dump(launch, open(OUT.replace("_results.json", "_launch.json"), "w"), indent=2)
    log("=" * 78); log("LAUNCH MANIFEST — TRANSPORT_TO_DWELL_TD3_BASELINE_V1 (Arm A)"); log("=" * 78)
    log(json.dumps(launch, indent=1)); log("=" * 78)

    st = dict(cfg["stage"]); st.update(horizon=cfg["horizon"], n_step=cfg["n_step"])
    t0 = time.time()
    res = train_transport_dwell(pi0, st, train, dev_flat, seeds=cfg["rng_seeds"], tcfg=tcfg, log=log)
    res["campaign"] = cfg["campaign"]; res["config_sha"] = cfg["config_sha"]; res["wall_s"] = round(time.time() - t0, 1)
    if not res["critic_ever_authorized"]:
        res["verdict"] = "TD3_CRITIC_NOT_AUTHORIZED_FOR_ACTOR_UPDATE"
    elif res["td_pass"]:
        res["verdict"] = "TRANSPORT_DWELL_TD3_PASS"
    else:
        res["verdict"] = "TRANSPORT_DWELL_TD3_NO_IMPROVEMENT"
    res["launch_sha"] = hashlib.sha256(json.dumps(launch, sort_keys=True).encode()).hexdigest()[:12]
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    log(f"\ntd_pass={res['td_pass']} ever_auth={res['critic_ever_authorized']} acc/rej {res['accepted']}/{res['rejected']} "
        f"wall={res['wall_s']}s → {res['verdict']}\nwrote {OUT}\nTRANSPORT_DWELL_DONE")


if __name__ == "__main__":
    main()
