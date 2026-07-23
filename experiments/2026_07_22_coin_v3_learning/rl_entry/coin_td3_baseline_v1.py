"""PHASE_SWITCHED_TD3_BASELINE_V1 — Stage-1 campaign entry (Arm A only). Prints+saves the launch manifest (§15), rebuilds
the frozen banks (verifying SHAs), runs the smallest complete Stage-1 campaign, and writes results + verdict. No
neutral-reset composed eval, no Arm B, no SAC, final-test bank unopened.

Usage: python coin_td3_baseline_v1.py <pi0.pt> <config.json> <out.json>
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart, late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_trainer import train_stage1  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
CFG = sys.argv[2] if len(sys.argv) > 2 else "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_config.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_results.json"


def _bank_from_rows(manifest) -> list:
    """Rebuild LateStart objects from the frozen manifest rows (gate_state unused for training/eval)."""
    bank = [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
            for r in manifest["rows"]]
    assert late_start_bank_manifest(bank)["sha16"] == manifest["sha16"], "bank SHA drift from frozen manifest"
    return bank


def main():
    cfg = json.load(open(CFG))
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    train_bank = _bank_from_rows(cfg["banks"]["late_train"]); dev_bank = _bank_from_rows(cfg["banks"]["late_dev"])
    log = lambda *a: print(*a, flush=True)

    launch = {
        "campaign": cfg["campaign"], "arm": cfg["arm"], "config_sha": cfg["config_sha"], "pi0_sha": cfg["pi0_sha"],
        "exact_command": f"python experiments/2026_07_22_coin_v3_learning/rl_entry/coin_td3_baseline_v1.py {PI0} {CFG} {OUT}",
        "rng_seeds": cfg["rng_seeds"],
        "bank_shas": {"late_train": cfg["banks"]["late_train"]["sha16"], "late_dev": cfg["banks"]["late_dev"]["sha16"]},
        "stage1": {k: cfg["stage1"][k] for k in ("families", "horizon", "n_step", "critic_warmup_steps", "policy_delay",
                                                 "total_updates", "checkpoints")},
        "target_smoothing": cfg["amendments"]["target_smoothing"], "exploration_schedule": cfg["exploration_schedule"],
        "checkpoint_schedule": cfg["stage1"]["checkpoints"], "stop_conditions": cfg["stage1_stop_conditions"],
        "constraints": cfg["no"],
    }
    json.dump(launch, open(OUT.replace("_results.json", "_launch.json"), "w"), indent=2)
    log("=" * 78); log("LAUNCH MANIFEST — PHASE_SWITCHED_TD3_BASELINE_V1 (Stage 1, Arm A)"); log("=" * 78)
    log(json.dumps(launch, indent=1)); log("=" * 78)

    t0 = time.time()
    res = train_stage1(pi0, cfg["stage1"], train_bank, dev_bank, seeds=cfg["rng_seeds"], log=log)
    res["campaign"] = cfg["campaign"]; res["config_sha"] = cfg["config_sha"]; res["wall_s"] = round(time.time() - t0, 1)
    res["verdict"] = "PHASE_SWITCHED_TD3_STAGE1_PASS" if res["stage1_pass"] else "PHASE_SWITCHED_TD3_STAGE1_NO_IMPROVEMENT"
    res["launch_sha"] = hashlib.sha256(json.dumps(launch, sort_keys=True).encode()).hexdigest()[:12]
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    log(f"\nstage1_pass={res['stage1_pass']}  wall={res['wall_s']}s  → {res['verdict']}\nwrote {OUT}\nTD3_BASELINE_V1_DONE")


if __name__ == "__main__":
    main()
