"""COIN-DELIVERY-RL-2 — hard-state targeted residual RL (oracle-gated).

RL-1 was CASE D (uniform PPO residual matched but did not beat scripted). RL-2 asks whether the remaining HARD states
are recoverable within the residual abstraction, and launches targeted RL ONLY if the residual oracle says a
meaningful subset is. Stages: 0 classify (7 failure classes) → 1 residual oracle sweep on failures (δ∈{.3,.5,.75,1},
segmented open-loop CEM, NO RL) → GATE (≥20-30% recoverable) → 2 hard-state generator → 3 targeted PPO → 4 eval all 90.

NO kato15, NO broad multi-seed, NO env/reward/dynamics/CORE change. Matching the scripted baseline is NOT improvement.
Reuses train.coin_delivery_rl (harness), train.coin_delivery_hardstate (classify/oracle/generator), train.ppo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.train.coin_delivery_hardstate import (
    FailureClass,
    OracleConfig,
    Recoverability,
    build_problems,
    classify_held,
    oracle_recoverability,
)
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, make_delivery_rl_env

_OUT = Path("experiments/2026_07_20_coin_delivery_rl2")
_HELD = range(64_000, 64_090)
_DELTAS = (0.30, 0.50, 0.75, 1.00)
_GATE_FRACTION = 0.20            # proceed to targeted RL only if >= this fraction of failures is recoverable


def _log(m: str) -> None:
    print(m, flush=True)


# ── Stage 0 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _class_stats(rows: list[dict]) -> dict:
    """Per-class aggregate signals (start/min/final dtz, handoff, contact-loss, progress)."""
    agg: dict[str, list] = {}
    for c in rows:
        agg.setdefault(c["failure_class"], []).append(c["row"])
    out = {}
    for cls, rs in agg.items():
        out[cls] = {"n": len(rs),
                    "start_dtz_med": round(float(np.median([r["start_dtz"] for r in rs])), 4),
                    "min_dtz_med": round(float(np.median([r["min_dtz"] for r in rs])), 4),
                    "final_dtz_med": round(float(np.median([r["final_dtz"] for r in rs])), 4),
                    "handoff_rate": round(float(np.mean([r["handoff_event"] for r in rs])), 3),
                    "contact_lost_rate": round(float(np.mean([r["contact_lost"] for r in rs])), 3),
                    "progress_med": round(float(np.median([r["start_dtz"] - r["min_dtz"] for r in rs])), 4)}
    return out


def stage0_classify(cfg: DeliveryRLConfig, held, env) -> dict:
    _log("=== Stage 0 — classify 90 held states (scripted rollout) ===")
    classified = classify_held(cfg, held, env=env)
    hist = Counter(c["failure_class"] for c in classified)
    stats = _class_stats(classified)
    for cls in [c.value for c in FailureClass]:
        if cls in hist:
            _log(f"  {cls:26s} n={hist[cls]:2d}  {stats[cls]}")
    return {"classified": classified, "histogram": dict(hist), "class_stats": stats}


# ── Stage 1 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def stage1_oracle(cfg: DeliveryRLConfig, classified: list[dict], env, ocfg: OracleConfig,
                  deltas: tuple[float, ...], seed: int = 0) -> dict:
    _log(f"\n=== Stage 1 — residual ORACLE sweep on failures (δ={deltas}, H<={ocfg.horizon}, "
         f"CEM pop{ocfg.pop}×it{ocfg.iters}×seg{ocfg.segments}) ===")
    rng = np.random.default_rng(seed)
    failures = [c for c in classified if c["failure_class"] != FailureClass.CENTER_SUCCESS.value]
    oracle: dict[int, dict] = {}
    t0 = time.perf_counter()
    for i, c in enumerate(failures):
        res = oracle_recoverability(env, c["seed"], deltas, ocfg, rng, zone_half=cfg.zone_half)
        oracle[c["seed"]] = res
        el = time.perf_counter() - t0
        eta = el / (i + 1) * (len(failures) - i - 1)
        _log(f"  [{i + 1}/{len(failures)}] seed {c['seed']} {c['failure_class']:20s} → {res['recoverability']:34s} "
             f"δ={res['recovered_at_delta']} H={res['min_H']} | {el:.0f}s ETA {eta:.0f}s")
    rec_hist = Counter(r["recoverability"] for r in oracle.values())
    recoverable = sum(rec_hist.get(k.value, 0) for k in (Recoverability.CURRENT, Recoverability.WIDER))
    n_fail = len(failures)
    frac = round(recoverable / n_fail, 4) if n_fail else 0.0
    gate = frac >= _GATE_FRACTION
    _log(f"  recoverability histogram: {dict(rec_hist)}")
    _log(f"  recoverable_failure_fraction = {recoverable}/{n_fail} = {frac} → gate(≥{_GATE_FRACTION}) "
         f"{'PASS' if gate else 'FAIL'}")
    return {"oracle": oracle, "recoverability_histogram": dict(rec_hist), "n_failures": n_fail,
            "recoverable_count": recoverable, "recoverable_failure_fraction": frac, "gate_pass": gate}


# ── driver (Stage 0 + Stage 1 + gate; Stages 2-4 gated) ──────────────────────────────────────────────────────────────
def run(*, held=_HELD, deltas: tuple[float, ...] = _DELTAS, ocfg: OracleConfig | None = None,
        stage3: bool = True) -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    cfg = DeliveryRLConfig()
    ocfg = ocfg or OracleConfig()
    env = make_delivery_rl_env(cfg)

    s0 = stage0_classify(cfg, held, env)
    s1 = stage1_oracle(cfg, s0["classified"], env, ocfg, deltas)
    problems = build_problems(s0["classified"], s1["oracle"])

    out: dict[str, Any] = {
        "config": {"deltas": list(deltas), "gate_fraction": _GATE_FRACTION, "horizon": cfg.horizon,
                   "oracle": {"segments": ocfg.segments, "pop": ocfg.pop, "iters": ocfg.iters, "H": ocfg.horizon}},
        "stage0_histogram": s0["histogram"], "stage0_class_stats": s0["class_stats"],
        "stage1_recoverability_histogram": s1["recoverability_histogram"],
        "stage1_recoverable_failure_fraction": s1["recoverable_failure_fraction"],
        "stage1_gate_pass": s1["gate_pass"],
        "problems": [p.__dict__ for p in problems],
        "oracle_per_seed": {str(k): v for k, v in s1["oracle"].items()},
    }
    if not s1["gate_pass"]:
        out["verdict"] = "CASE_C_residual_abstraction_exhausted_no_RL"
        out["kato15_justified"] = False
        _log("\n[COIN-DELIVERY-RL-2] Stage-1 gate FAILED → CASE C (residual abstraction exhausted). "
             "Next lever = a NEW transport/settling primitive, NOT more RL. Stage 3 NOT run.")
    elif not stage3:
        out["verdict"] = "GATE_PASS_stage234_deferred"
        out["kato15_justified"] = False
        _log("\n[COIN-DELIVERY-RL-2] Stage-1 gate PASSED; Stage 2-4 deferred (--stage01-only).")
    else:
        from hymeko_rl.experiments.coin_delivery_rl2_train import stage234
        out.update(stage234(cfg, problems, env, held, run_stage3=stage3))
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    (_OUT / "manifests" / "coin_delivery_rl2.json").write_text(json.dumps(out, indent=2, default=float))
    _log(f"\n[COIN-DELIVERY-RL-2] verdict={out.get('verdict')} | recoverable_fraction="
         f"{s1['recoverable_failure_fraction']} | kato15_justified={out.get('kato15_justified')} | {out['wall_s']}s")
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="COIN-DELIVERY-RL-2 — hard-state targeted residual RL (oracle-gated)")
    ap.add_argument("--stage01-only", action="store_true", help="run only Stage 0+1 (classify + oracle gate)")
    ap.add_argument("--fast-oracle", action="store_true", help="smaller CEM budget for a quick gate estimate")
    a = ap.parse_args(argv)
    ocfg = OracleConfig(pop=10, iters=4) if a.fast_oracle else OracleConfig()
    run(ocfg=ocfg, stage3=not a.stage01_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
