"""Freeze the PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1 preregistration (before any training): mechanism-balanced
late-start bank manifests (deterministic replay-to-handoff), short late-phase horizons, and the frozen TD3 contract.
No training. Writes phase_switched_late_config.json."""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank, late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_contracts import TD3Config  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/phase_switched_late_config.json"
LATE_TRAIN = (6000, 6060)          # disjoint from policy VALIDATION(7000-7029)/FINAL_TEST(8000-8049)/HEADLINE
LATE_DEV = (6060, 6100)


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    log = lambda *a: print(*a, flush=True)
    t = time.time(); log("building late_train bank (replay-to-handoff)...")
    train = build_late_start_bank(pi0, range(*LATE_TRAIN), per_family=6)
    log(f"  train {len(train)} starts ({time.time()-t:.0f}s)")
    t = time.time(); log("building late_dev bank...")
    dev = build_late_start_bank(pi0, range(*LATE_DEV), per_family=4)
    log(f"  dev {len(dev)} starts ({time.time()-t:.0f}s)")

    cfg = {
        "controller": "PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1", "date": "2026-07-23",
        "pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
        "structure": "gate_t==0 ⇒ clip(pi_0(obs),-4,4); gate_t==1 ⇒ clip(pi_late(obs),-4,4); pi_late full-action, "
                     "initialized EXACT copy of pi_0 (update-0 reproduces 3/9,2/30,9/9)",
        "handoff_construction": "deterministic REPLAY (neutral reset → frozen pi_0 prefix → begin late episode); "
                                "NOT a MuJoCo snapshot restore (which does not reproduce node_features' velocity buffer)",
        "late_phase_horizons_preregistered": [30, 60],        # §7 short late-phase horizons initially
        "reward": "data/robotics/galambos_task_deliver_v3.hymeko (UNCHANGED)",
        "gate": "STABLE_OBJECT_ENGAGEMENT_V1 (7633dd3c, UNCHANGED)",
        "banks": {
            "late_train": {"seed_range": list(LATE_TRAIN), **late_start_bank_manifest(train)},
            "late_dev": {"seed_range": list(LATE_DEV), **late_start_bank_manifest(dev)},
        },
        "td3_contract": TD3Config().frozen_manifest(),
        "td3_init_arms": {
            "A": "pi_late = EXACT pi_0 copy",
            "B": "pi_late warm-started from H=30 feedback-planner / certified late-policy data (provenance REPLAY_ONLY)",
        },
        "sac_deferred": "SAC is a SECOND matched campaign, ONLY after TD3 plumbing passes; tanh-squash (4*tanh(z)) with "
                        "exact change-of-variables log-prob (NOT gaussian+hard-clip); low initial entropy; not implemented here",
        "policy_final_test_bank": {"range": [8000, 8050], "status": "UNOPENED"},
        "constraints_this_stage": ["contracts+tests only", "no training", "no SAC", "no residual-critic route",
                                   "no safety-filter head", "reward unchanged", "gate unchanged", "residual bound n/a",
                                   "final-test bank unopened"],
    }
    json.dump(cfg, open(OUT, "w"), indent=1)
    tb = cfg["banks"]["late_train"]; db = cfg["banks"]["late_dev"]
    log(f"FROZEN {OUT}")
    log(f"  late_train n={tb['n']} sha {tb['sha16']} coverage {tb['coverage']}")
    log(f"  late_dev   n={db['n']} sha {db['sha16']} coverage {db['coverage']}")
    # disjointness guard: train and dev seeds must not overlap
    tr_seeds = {r[0] for r in tb["rows"]}; dv_seeds = {r[0] for r in db["rows"]}
    assert not (tr_seeds & dv_seeds), "late_train and late_dev share seeds"
    log("  train/dev seed-disjoint: True")


if __name__ == "__main__":
    main()
