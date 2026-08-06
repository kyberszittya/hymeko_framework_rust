"""Freeze the TRANSPORT_TO_DWELL_TD3_BASELINE_V1 preregistration (before training): persistent control-mode banks
(transport/braking/settling_dwell), the re-scoped ontology, horizon 60, 50/30/20 sampling, and the UNCHANGED
reward/gate/smoothing/exploration/trust-region/n-step/term-trunc/eval. Verifies the §6 bank thresholds. No training."""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import late_start_bank_manifest  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import (  # noqa: E402
    BANK_MIN,
    CONTACT_FLAGS,
    CONTROL_MODES,
    N_COND,
    SAMPLE_TARGET,
    rebuild_control_mode_bank,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
TRAIN_SEEDS = (6000, 6200); DEV_SEEDS = (6200, 6300)
TRAIN_PER = {"transport": 30, "braking": 20, "settling_dwell": 12}
DEV_PER = {"transport": 15, "braking": 10, "settling_dwell": 6}


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); log = lambda *a: print(*a, flush=True)
    t = time.time(); train, tc = rebuild_control_mode_bank(pi0, range(*TRAIN_SEEDS), min_persist=2, per_mode=TRAIN_PER)
    log(f"train banks {tc} ({time.time()-t:.0f}s)")
    t = time.time(); dev, dc = rebuild_control_mode_bank(pi0, range(*DEV_SEEDS), min_persist=2, per_mode=DEV_PER)
    log(f"dev banks {dc} ({time.time()-t:.0f}s)")
    tman = {m: {**late_start_bank_manifest(train[m]), "seed_range": list(TRAIN_SEEDS)} for m in CONTROL_MODES}
    dman = {m: {**late_start_bank_manifest(dev[m]), "seed_range": list(DEV_SEEDS)} for m in CONTROL_MODES}
    tr_seeds = {r[0] for m in CONTROL_MODES for r in tman[m]["rows"]}
    dv_seeds = {r[0] for m in CONTROL_MODES for r in dman[m]["rows"]}
    meets = all(tc[m] >= BANK_MIN[m] for m in CONTROL_MODES)

    cfg = {
        "campaign": "TRANSPORT_TO_DWELL_TD3_BASELINE_V1", "date": "2026-07-23",
        "pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8],
        "ontology": {"control_modes": list(CONTROL_MODES), "contact_flags": list(CONTACT_FLAGS),
                     "target_entry": "DEMOTED to event features (not a control mode)",
                     "event_features": ["inside_target_zone", "just_entered", "just_exited", "distance_to_target", "radial_velocity"],
                     "conditioning": f"onehot3(control) ++ onehot2(contact) ++ event(5) = {N_COND}; actor/critic input obs_48 ++ {N_COND} = {48+N_COND}",
                     "update0_identity": "conditioning weights ZERO-init ⇒ update-0 == pi_0"},
        "horizon": 60, "n_step": 4,
        "target_smoothing_UNCHANGED": {"std": 0.10, "clip": 0.25},
        "exploration_UNCHANGED": {"std_init": 0.15, "std_max": 0.30, "hold": [2, 4]},
        "transactional_UNCHANGED": TransactionalConfig().manifest(),
        "sample_target": SAMPLE_TARGET, "sample_note": "dynamically balanced (not forced equal)",
        "bank_min": BANK_MIN, "train_bank_counts": tc, "dev_bank_counts": dc, "meets_thresholds": meets,
        "banks": {"train": tman, "dev": dman}, "train_dev_seed_disjoint": not (tr_seeds & dv_seeds),
        "stage": {"total_updates": 8000, "critic_warmup_steps": 2000, "policy_delay": 4, "collect_every": 500,
                  "episodes_per_collect": 24, "checkpoints": [0, 2000, 4000, 6000, 8000]},
        "rng_seeds": {"torch": 0, "numpy_collect": 0, "numpy_replay": 1},
        "unchanged": ["frozen pi_0", "canonical reward", "stable-engagement gate", "4-step TD", "smoothing 0.10/0.25",
                      "coherent exploration 0.15->0.30 held 2-4", "transactional per-step+cumulative caps",
                      "terminated/truncated semantics", "development eval protocol"],
        "stop_conditions": ["accepted actor updates > 0",
                            "two consecutive TRAINED checkpoints improve >=1 of {entry-rate, braking/entry-speed, "
                            "max-dwell, strict} without degrading contact retention (>=-0.05) or target exit (<=+0.05)"],
        "no": ["old target_entry curriculum", "Stage 2", "Arm B", "SAC", "neutral-reset composition", "final-test seeds",
               "trust-region relaxation"],
    }
    stable = {k: v for k, v in cfg.items() if k != "banks"}
    stable["bank_shas"] = {"train": {m: tman[m]["sha16"] for m in CONTROL_MODES}, "dev": {m: dman[m]["sha16"] for m in CONTROL_MODES}}
    cfg["config_sha"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
    json.dump(cfg, open(OUT, "w"), indent=1)
    log(f"FROZEN {OUT} config_sha {cfg['config_sha']}")
    log(f"  train {tc} meets_thresholds={meets} (min {BANK_MIN})  dev {dc}  seed-disjoint={cfg['train_dev_seed_disjoint']}")
    if not meets:
        log("  WARNING: bank thresholds NOT met — do not train")


if __name__ == "__main__":
    main()
