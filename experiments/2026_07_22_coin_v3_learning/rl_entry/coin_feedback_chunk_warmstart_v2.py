"""FEEDBACK_CHUNK_WARMSTART_V2 — dense feedback-planner warm-start + prefix-weighted regression + supervised DAgger +
acceptance gate (NO TD3). Trajectory-level disjoint train/dev. Planner is WARM_START_ONLY (no planner at eval)."""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_chunk_td3 import chunk_metrics, eval_receding_horizon  # noqa: E402
from hymeko_rl.coin_delivery.coin_feedback_chunk_v2 import (  # noqa: E402
    PREFIX_WEIGHTS,
    build_feedback_dataset,
    dagger_collect,
    train_supervised_v2,
)
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
import numpy as np  # noqa: E402

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
TDCFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"
OUT = "experiments/2026_07_22_coin_v3_learning/rl_entry/feedback_chunk_warmstart_v2.json"
DAGGER_ROUNDS = 2


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def main():
    cfg = json.load(open(TDCFG)); pi0 = load_frozen_clip_actor(PI0, freeze=True); log = lambda *a: print(*a, flush=True)
    train = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["train"][m])]
    dev = [ls for m in CONTROL_MODES for ls in _bank(cfg["banks"]["dev"][m])]
    assert not ({ls.seed for ls in train} & {ls.seed for ls in dev}), "train/dev seed overlap"
    horizon = cfg["horizon"]

    log(f"[{time.strftime('%H:%M:%S')}] building DENSE feedback-planner dataset ({len(train)} train trajectories)...")
    t = time.time(); X, Y, prov, stats = build_feedback_dataset(pi0, train, horizon=horizon)
    log(f"  {stats} ({time.time()-t:.0f}s)")
    ds_sha = hashlib.sha256(X.tobytes() + Y.tobytes()).hexdigest()[:16]
    # supervised MSE is reported on the training set; the DISJOINT dev-trajectory receding-horizon rollout is the gate.
    log(f"[{time.strftime('%H:%M:%S')}] prefix-weighted supervised regression...")
    actor = train_supervised_v2(X, Y, steps=4000, log=log)
    m0 = chunk_metrics(actor, X, Y)
    log(f"  base chunk MSE: seq {m0['sequence_mse']} first {m0['first_action_mse']} prefix {m0['two_step_prefix_mse']}")

    agg_prov = dict(stats["by_provenance"])
    for rd in range(DAGGER_ROUNDS):
        log(f"[{time.strftime('%H:%M:%S')}] DAgger round {rd+1}/{DAGGER_ROUNDS} (roll learned actor, query planner)...")
        dX, dY, dprov = dagger_collect(pi0, actor, train, horizon=horizon, plan_seed=7000 + 1000 * rd)
        if dX is not None:
            X = np.concatenate([X, dX]); Y = np.concatenate([Y, dY])
            from collections import Counter
            for k, v in Counter(dprov).items():
                agg_prov[k] = agg_prov.get(k, 0) + v
            log(f"  +{len(dX)} DAgger examples ({dict(Counter(dprov))}); dataset now {len(X)}")
            actor = train_supervised_v2(X, Y, steps=3000, actor=actor, log=log)

    mf = chunk_metrics(actor, X, Y)
    log(f"[{time.strftime('%H:%M:%S')}] receding-horizon eval (V2 chunk vs frozen pi_0) on {len(dev)} dev starts...")
    ev = eval_receding_horizon(pi0, actor, dev, horizon=horizon); dl = ev["delta_vs_pi0"]
    log(f"  V2 vs pi_0: Δstrict {dl['strict_success']:+.3f} Δdwell {dl['max_dwell']:+.2f} Δenter {dl['entered']:+.3f} "
        f"Δcontact {dl['contact_retention']:+.3f} Δexit {dl['exited']:+.3f} Δprogress {dl['progress']:+.4f}")

    # §10 acceptance gate: must not materially underperform pi_0 in contact/exit/dwell/strict
    gate = (dl["contact_retention"] >= -0.05 and dl["exited"] <= 0.05 and dl["max_dwell"] >= -0.10 and dl["strict_success"] >= -0.05)
    reproduces_advantage = (dl["progress"] > 0 or dl["max_dwell"] > 0 or dl["strict_success"] > 0 or dl["entered"] > 0)
    out = {"contract": "FEEDBACK_CHUNK_WARMSTART_V2", "date": "2026-07-23", "pi0_sha": cfg["pi0_sha"], "no_td3": True,
           "planner_at_eval": False, "prefix_weights": PREFIX_WEIGHTS.tolist(), "dagger_rounds": DAGGER_ROUNDS,
           "teacher": "H-receding-horizon FEEDBACK planner (plan_chunk CEM at every replanning state), WARM_START_ONLY",
           "dataset": {**stats, "with_dagger_provenance": agg_prov, "n_final": int(len(X)), "sha16": ds_sha,
                       "bank_shas": {"train": {m: cfg["banks"]["train"][m]["sha16"] for m in CONTROL_MODES},
                                     "dev": {m: cfg["banks"]["dev"][m]["sha16"] for m in CONTROL_MODES}}},
           "supervised_chunk_metrics_base": m0, "supervised_chunk_metrics_final": mf,
           "eval_vs_pi0": ev, "acceptance_gate_pass": bool(gate), "reproduces_some_planner_advantage": bool(reproduces_advantage)}
    out["verdict"] = "FEEDBACK_CHUNK_WARMSTART_V2_ACCEPTED" if gate else "FEEDBACK_CHUNK_WARMSTART_V2_STILL_UNDERPERFORMS"
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    log(f"\nacceptance_gate={gate} reproduces_advantage={reproduces_advantage} → {out['verdict']}\nwrote {OUT}\nV2_DONE")


if __name__ == "__main__":
    main()
