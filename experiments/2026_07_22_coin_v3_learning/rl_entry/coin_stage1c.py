"""Stage-1c campaign — DYNAMIC phase-conditioned transactional TD3 (Arm A). SAME banks, reward, horizon, n-step,
smoothing, exploration, and transactional trust-region thresholds as Stage-1b — the ONLY change is dynamic phase
conditioning. Trust-region limits are NOT relaxed (isolates the phase effect). Drift is reported in the constrained
anchor-L2 metrics; probe-L∞ is a separate diagnostic. No Stage 2, no Arm B, no SAC, no neutral-reset, no final-test.
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart, late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_phase_stage1c import train_stage1c  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = sys.argv[2] if len(sys.argv) > 2 else "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_config.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_stage1c_results.json"


def _bank(m):
    bank = [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]
    assert late_start_bank_manifest(bank)["sha16"] == m["sha16"], "bank SHA drift"
    return bank


def main():
    cfg = json.load(open(CFG)); tcfg = TransactionalConfig()
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    tb = _bank(cfg["banks"]["late_train"]); db = _bank(cfg["banks"]["late_dev"])
    log = lambda *a: print(*a, flush=True)
    launch = {
        "campaign": "PHASE_SWITCHED_TD3_STAGE1C (DYNAMIC phase conditioning)", "arm": cfg["arm"],
        "config_sha": cfg["config_sha"], "pi0_sha": cfg["pi0_sha"],
        "exact_command": f"python experiments/2026_07_22_coin_v3_learning/rl_entry/coin_stage1c.py {PI0} {CFG} {OUT}",
        "rng_seeds": cfg["rng_seeds"],
        "bank_shas": {"late_train": cfg["banks"]["late_train"]["sha16"], "late_dev": cfg["banks"]["late_dev"]["sha16"]},
        "stage1": {k: cfg["stage1"][k] for k in ("families", "horizon", "n_step", "critic_warmup_steps",
                                                 "total_updates", "checkpoints")},
        "transactional_config_UNCHANGED": tcfg.manifest(),
        "target_smoothing_UNCHANGED": cfg["amendments"]["target_smoothing"],
        "exploration_schedule_UNCHANGED": cfg["exploration_schedule"],
        "only_change": "DYNAMIC per-transition phase conditioning (actor+critic+target); trust-region NOT relaxed",
        "drift_reporting": "anchor-L2 per-step (over accepted) + cumulative {median,p95,max}; probe-Linf = diagnostic only",
        "constraints": ["no Stage 2", "no Arm B", "no SAC", "no neutral-reset composition", "no final-test seeds",
                        "no trust-region relaxation", "reward/gate/pi_0 unchanged"],
    }
    json.dump(launch, open(OUT.replace("_results.json", "_launch.json"), "w"), indent=2)
    log("=" * 78); log("LAUNCH MANIFEST — Stage-1c DYNAMIC phase-conditioned transactional TD3 (Arm A)"); log("=" * 78)
    log(json.dumps(launch, indent=1)); log("=" * 78)

    t0 = time.time()
    res = train_stage1c(pi0, cfg["stage1"], tb, db, seeds=cfg["rng_seeds"], tcfg=tcfg, log=log)
    res["campaign"] = "PHASE_SWITCHED_TD3_STAGE1C"; res["config_sha"] = cfg["config_sha"]; res["wall_s"] = round(time.time() - t0, 1)
    if not res["critic_ever_authorized"]:
        res["verdict"] = "TD3_CRITIC_NOT_AUTHORIZED_FOR_ACTOR_UPDATE"
    elif res["stage1c_pass"]:
        res["verdict"] = "PHASE_SWITCHED_TD3_STAGE1C_PASS"
    else:
        res["verdict"] = "PHASE_SWITCHED_TD3_STAGE1C_NO_IMPROVEMENT"
    res["launch_sha"] = hashlib.sha256(json.dumps(launch, sort_keys=True).encode()).hexdigest()[:12]
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    log(f"\nstage1c_pass={res['stage1c_pass']} ever_auth={res['critic_ever_authorized']} acc/rej {res['accepted']}/{res['rejected']} "
        f"wall={res['wall_s']}s → {res['verdict']}\nwrote {OUT}\nTD3_STAGE1C_DONE")


if __name__ == "__main__":
    main()
