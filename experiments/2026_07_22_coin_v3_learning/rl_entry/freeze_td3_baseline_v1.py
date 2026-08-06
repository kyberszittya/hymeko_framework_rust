"""Freeze the PHASE_SWITCHED_TD3_BASELINE_V1 preregistration (amended smoothing + merged settling_dwell), BEFORE
launching. Builds the merged-family late-start banks, records the frozen Stage-1/Stage-2 spec, exploration schedule,
checkpoint schedule, and stop conditions, and stamps a config SHA. No training. Writes td3_baseline_v1_config.json."""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank, late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_contracts import HISTORICAL_SCALED_DEFAULT_SMOOTHING, TD3Config  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/td3_baseline_v1_config.json"
LATE_TRAIN = (6000, 6100); LATE_DEV = (6100, 6160)


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    log = lambda *a: print(*a, flush=True)
    t = time.time(); train = build_late_start_bank(pi0, range(*LATE_TRAIN), per_family=8)
    log(f"late_train {len(train)} starts ({time.time()-t:.0f}s)")
    t = time.time(); dev = build_late_start_bank(pi0, range(*LATE_DEV), per_family=6)
    log(f"late_dev {len(dev)} starts ({time.time()-t:.0f}s)")
    tb = {"seed_range": list(LATE_TRAIN), **late_start_bank_manifest(train)}
    db = {"seed_range": list(LATE_DEV), **late_start_bank_manifest(dev)}
    assert not ({r[0] for r in tb["rows"]} & {r[0] for r in db["rows"]}), "train/dev seed overlap"

    cfg = {
        "campaign": "PHASE_SWITCHED_TD3_BASELINE_V1", "date": "2026-07-23", "arm": "A (pi_late = exact pi_0 copy)",
        "pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
        "amendments": {
            "target_smoothing": {"std": 0.10, "clip": 0.25, "units": "full-action",
                                 "note": "local critic regularization, NOT broad exploration"},
            "historical_scaled_default_control_only": HISTORICAL_SCALED_DEFAULT_SMOOTHING,
            "merged_family": "settling + dwell -> settling_dwell",
            "deferred": "contact_loss_reacquisition -> separate recovery curriculum after a basic late controller improves",
        },
        "exploration_schedule": {"type": "temporally_coherent", "std_init": 0.15, "std_max": 0.30,
                                 "hold_steps": [2, 4], "note": "SEPARATE from target smoothing; frozen before training"},
        "td3_contract": TD3Config().frozen_manifest(),
        "stage1": {"pi_late_init": "exact pi_0 copy", "horizon": 30, "n_step": 4,
                   "families": ["target_entry", "braking", "settling_dwell"],
                   "critic_warmup_steps": 2000, "policy_delay": 2, "phase_balanced_replay": True,
                   "total_updates": 8000, "collect_every": 500, "episodes_per_collect": 24,
                   "checkpoints": [0, 2000, 4000, 6000, 8000], "dev_eval_at_checkpoints": True},
        "stage2": {"horizon": 60, "families": ["target_entry", "braking", "settling_dwell", "transport", "overshoot"],
                   "note": "entered ONLY after Stage 1 passes; same controller/reward/gate/action contracts"},
        "stage1_stop_conditions": [
            "two CONSECUTIVE dev checkpoints improve strict-K6 success OR the preregistered max-dwell metric vs frozen pi_0 continuation",
            "no material increase in target_exit (<= +0.05 vs pi_0)",
            "no material degradation in contact_retention (>= -0.05 vs pi_0)",
            "stable Q1/Q2 calibration (finite, |Q1-Q2| bounded, no divergence)",
        ],
        "eval_metrics": ["strict_K6_success", "max_dwell", "target_entry", "target_exit", "contact_retention",
                         "return_vs_frozen_pi0_continuation", "Q1_Q2_calibration", "actor_action_drift_from_update0"],
        "banks": {"late_train": tb, "late_dev": db},
        "rng_seeds": {"torch": 0, "numpy_collect": 0, "numpy_replay": 1},
        "reward": "data/robotics/galambos_task_deliver_v3.hymeko (UNCHANGED)",
        "gate": "STABLE_OBJECT_ENGAGEMENT_V1 (7633dd3c, UNCHANGED)",
        "no": ["neutral-reset composed eval before a late-start stage passes", "Arm B planner warm-start", "SAC",
               "open final-test bank (8000-8049)", "modify reward/gate/frozen pi_0", "residual-critic route"],
    }
    # config SHA over the frozen spec (bank SHAs included; bank rows excluded to keep it stable)
    stable = {k: v for k, v in cfg.items() if k != "banks"}
    stable["bank_shas"] = {"late_train": tb["sha16"], "late_dev": db["sha16"]}
    cfg["config_sha"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(cfg, open(OUT, "w"), indent=1)
    log(f"FROZEN {OUT}  config_sha {cfg['config_sha']}")
    log(f"  late_train n={tb['n']} sha {tb['sha16']} coverage {tb['coverage']}")
    log(f"  late_dev   n={db['n']} sha {db['sha16']} coverage {db['coverage']}")
    log(f"  stage1 families {cfg['stage1']['families']} horizon {cfg['stage1']['horizon']} n_step {cfg['stage1']['n_step']}")


if __name__ == "__main__":
    main()
