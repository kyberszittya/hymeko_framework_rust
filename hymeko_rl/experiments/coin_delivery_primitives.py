"""COIN-DELIVERY new-primitives — Track B centering primitive oracle (the cleanest first hard-state test).

Recommended order (F-COIN-DELIVERY-RL2 follow-up): the precision failures are near the goal with a dense control signal
— the cleanest first off-policy target. This runs the Track-B **centering-primitive CEM oracle** on the precision
failures (ZONE_ONLY + NEAR_MISS) and gates whether off-policy RL on the centering subproblem is justified:

    GATE: the new centering primitive recovers >= 4 additional center-reaches on the 11 REQUIRES_NEW_PRIMITIVE precision
          failures (i.e. beats the residual oracle, which recovered 0 of them).

NO RL launched here — this is the primitive-oracle gate that precedes any RL. NO env/reward/dynamics/CORE change.
Reuses train.coin_delivery_primitives (the centering primitive + CEM) and the RL-2 classification manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from hymeko_rl.train.coin_delivery_primitives import (
    CenteringOracleConfig,
    CenteringParams,
    cem_centering,
    eval_centering,
)
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, eval_delivery, make_delivery_rl_env, scripted_action_fn

_OUT = Path("experiments/2026_07_20_coin_delivery_primitives")
_RL2_MANIFEST = Path("experiments/2026_07_20_coin_delivery_rl2/manifests/coin_delivery_rl2.json")
_HELD = range(64_000, 64_090)
_GATE_MIN_RECOVER = 4


def _log(m: str) -> None:
    print(m, flush=True)


def _target_seeds() -> dict:
    """Load the Track-B target seeds from the RL-2 classification: precision failures (ZONE_ONLY+NEAR_MISS), the 11
    REQUIRES_NEW_PRIMITIVE subset (the gate set), the 6 residual-recoverable, and the 54 easy states."""
    d = json.loads(_RL2_MANIFEST.read_text())
    probs = d["problems"]
    precision = [int(p["state_id"]) for p in probs if p["failure_class"] in ("ZONE_ONLY", "NEAR_MISS")]
    new_prim = [int(p["state_id"]) for p in probs
                if p["failure_class"] in ("ZONE_ONLY", "NEAR_MISS") and p["recoverability"] == "REQUIRES_NEW_PRIMITIVE"]
    residual_recov = [int(p["state_id"]) for p in probs
                      if p["failure_class"] in ("ZONE_ONLY", "NEAR_MISS") and p["recoverability"].startswith("RECOVERABLE")]
    easy = [int(p["state_id"]) for p in probs if p["failure_class"] == "SCRIPTED_CENTER_SUCCESS"]
    return {"precision": precision, "requires_new_primitive": new_prim, "residual_recoverable": residual_recov, "easy": easy}


def run(*, ocfg: CenteringOracleConfig | None = None, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    cfg = DeliveryRLConfig()
    ocfg = ocfg or CenteringOracleConfig()
    env = make_delivery_rl_env(cfg)
    tg = _target_seeds()
    precision, new_prim, easy = tg["precision"], tg["requires_new_primitive"], tg["easy"]
    _log(f"=== Track B centering-primitive oracle === precision failures n={len(precision)} "
         f"(REQUIRES_NEW_PRIMITIVE n={len(new_prim)}, residual-recoverable n={len(tg['residual_recoverable'])}); easy n={len(easy)}")

    # baseline: scripted on the precision failures (all failures → 0 center by construction)
    sc = eval_delivery(scripted_action_fn(), precision, cfg, env=env)
    _log(f"  scripted on precision failures: center_reach={sc['center_reach']} final_dtz_med={sc['final_dtz_med']}")

    # CEM the centering primitive on the 17 precision failures
    _log(f"  CEM centering (pop{ocfg.pop}×it{ocfg.iters}) over {len(precision)} precision failures ...")
    best = cem_centering(precision, cfg, ocfg, np.random.default_rng(seed), env=env, log=_log)
    p: CenteringParams = best["params"]

    # evaluate the tuned primitive on the gate set (11 REQUIRES_NEW_PRIMITIVE), all precision (17), and easy (54)
    on_new = eval_centering(p, new_prim, cfg, env=env)
    on_prec = eval_centering(p, precision, cfg, env=env)
    on_easy = eval_centering(p, easy, cfg, env=env)
    recovered_new = on_new["n_center"]
    gate = recovered_new >= _GATE_MIN_RECOVER
    easy_preserved = on_easy["center_reach"]                   # what fraction of the 54 easy states the primitive keeps

    verdict = ("TRACK_B_GATE_PASS_centering_beats_residual_oracle" if gate
               else "TRACK_B_GATE_FAIL_centering_does_not_beat_precision_wall")
    out = {"config": {"gate_min_recover": _GATE_MIN_RECOVER, "cem": {"pop": ocfg.pop, "iters": ocfg.iters},
                      "center_tol": cfg.center_tol},
           "targets": {"precision_n": len(precision), "requires_new_primitive": new_prim, "easy_n": len(easy)},
           "best_params": p.__dict__,
           "scripted_precision": {k: sc[k] for k in ("center_reach", "final_dtz_med")},
           "centering_on_requires_new_primitive": on_new,
           "centering_on_all_precision": on_prec,
           "centering_on_easy": on_easy,
           "recovered_on_requires_new_primitive": recovered_new,
           "easy_state_preservation": easy_preserved,
           "gate_pass": gate, "verdict": verdict,
           "off_policy_rl_justified": gate, "kato15_justified": False,
           "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "coin_delivery_primitives.json").write_text(json.dumps(out, indent=2, default=float))
    _log(f"\n  best centering params: {p.__dict__}")
    _log(f"  recovered on 11 REQUIRES_NEW_PRIMITIVE: {recovered_new}/{len(new_prim)} "
         f"(center_reach={on_new['center_reach']}, final_dtz_med={on_new['final_dtz_med']})")
    _log(f"  on all {len(precision)} precision failures: center_reach={on_prec['center_reach']} "
         f"n_center={on_prec['n_center']} final_dtz_med={on_prec['final_dtz_med']}")
    _log(f"  easy-state preservation ({len(easy)} states): center_reach={easy_preserved}")
    _log(f"[TRACK-B] gate(recover≥{_GATE_MIN_RECOVER}) {'PASS' if gate else 'FAIL'} → {verdict} | "
         f"off_policy_rl_justified={gate} | {out['wall_s']}s")
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="COIN-DELIVERY Track-B centering-primitive oracle")
    ap.add_argument("--fast", action="store_true", help="smaller CEM budget")
    a = ap.parse_args(argv)
    run(ocfg=CenteringOracleConfig(pop=10, iters=5) if a.fast else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
