"""RECEDING_HORIZON_ACTION_CHUNK_TD3_V1 — warm-start dataset + supervised chunk baseline (D/E). No TD3 fine-tuning.
Builds sequence targets (planner CEM + pi_0 rehearsal) from the frozen transport-dwell late-start banks, trains the
supervised chunk actor, evaluates it (chunk MSE + receding-horizon rollout vs frozen pi_0) on disjoint dev states, and
freezes the warm-start manifest. Planner data is WARM_START_ONLY (no planner runs during evaluation)."""
import hashlib
import json
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_chunk_td3 import (  # noqa: E402
    CHUNK_DIM,
    K,
    M,
    STATE_DIM,
    build_warmstart_dataset,
    chunk_metrics,
    eval_receding_horizon,
    train_supervised_chunk,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
TDCFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/chunk_warmstart_v1.json"


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main():
    cfg = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True); log = lambda *a: print(*a, flush=True)
    train = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["train"][m])]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    log(f"[{time.strftime('%H:%M:%S')}] building warm-start dataset ({len(train)} starts, planner+pi_0)...")
    t = time.time(); X, Y, prov = build_warmstart_dataset(pi0, train, use_planner=True, planner_frac=0.35)
    log(f"  dataset {X.shape} provenance {dict(Counter(prov))} ({time.time()-t:.0f}s)")
    ds_sha = hashlib.sha256(X.tobytes() + Y.tobytes()).hexdigest()[:16]

    log(f"[{time.strftime('%H:%M:%S')}] supervised chunk regression...")
    actor = train_supervised_chunk(X, Y, steps=3000, log=log)
    train_m = chunk_metrics(actor, X, Y)
    log(f"  chunk MSE train: seq {train_m['sequence_mse']} first {train_m['first_action_mse']} prefix {train_m['two_step_prefix_mse']}")

    log(f"[{time.strftime('%H:%M:%S')}] receding-horizon eval (chunk vs frozen pi_0) on {len(dev)} dev starts...")
    ev = eval_receding_horizon(pi0, actor, dev, horizon=cfg["horizon"])
    dl = ev["delta_vs_pi0"]
    log(f"  supervised baseline vs pi_0: Δstrict {dl['strict_success']:+.3f} Δdwell {dl['max_dwell']:+.2f} "
        f"Δenter {dl['entered']:+.3f} Δcontact {dl['contact_retention']:+.3f} Δexit {dl['exited']:+.3f} Δprogress {dl['progress']:+.4f}")

    out = {"contract": "RECEDING_HORIZON_ACTION_CHUNK_TD3_V1", "date": "2026-07-23", "pi0_sha": cfg["pi0_sha"],
           "chunk": {"K": K, "M": M, "state_dim": STATE_DIM, "chunk_dim": CHUNK_DIM,
                     "note": "each chunk step independently predicted; execute first M, replan (NOT a residual hold)"},
           "warm_start_sources": {"A_pi0_rollouts": "USED (pi_0 rehearsal)",
                                  "B_H30_feedback_planner": "USED via plan_chunk CEM (WARM_START_ONLY; no planner at eval)",
                                  "C_certified_open_loop_continuations": "declared REPLAY_ONLY (not integrated this turn)"},
           "dataset": {"n": int(X.shape[0]), "provenance": dict(Counter(prov)), "sha16": ds_sha,
                       "bank_shas": {"train": {m: cfg["banks"]["train"][m]["sha16"] for m in CONTROL_MODES},
                                     "dev": {m: cfg["banks"]["dev"][m]["sha16"] for m in CONTROL_MODES}}},
           "supervised_chunk_metrics_train": train_m,
           "supervised_baseline_eval_vs_pi0": ev,
           "no_td3": True, "planner_at_eval": False,
           "verdict": "CHUNK_WARMSTART_AND_EXECUTION_CONTRACTS_PASS"}
    out["config_sha"] = hashlib.sha256(json.dumps({k: v for k, v in out.items() if k not in ("supervised_baseline_eval_vs_pi0",)}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    log(f"\nwrote {OUT}\nCHUNK_WARMSTART_DONE  dataset_sha {ds_sha}")


if __name__ == "__main__":
    main()
