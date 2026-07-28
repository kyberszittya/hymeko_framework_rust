"""R9 C1 component load-bearing ablation driver (frozen-policy intervention + matched retraining).

Objective `C1_COMPONENT_LOAD_BEARING_ABLATION` from immutable `2478a35d`: how much of the reproduced teacher-free K6 is the frozen
clone/R2 heritage vs the authority-unlock TD3? See `kinetic_ablation` for the two probes. All `8a0c1c7b` modules are imported
unchanged; the frozen seed-0 champion and its tag are never touched; s4/s7 and f1–f4 are never opened.

Run:  ``python -m hymeko_rl.experiments.coin_kinetic_ablation --frozen``     (intervention on the 22 K6 checkpoints)
      ``python -m hymeko_rl.experiments.coin_kinetic_ablation --retrain``    (matched F0/F1/F2, 8 seeds each)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option import kinetic_ablation as ab
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import ALPHA0
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-ablation")
CKPT_DIR = Path("reports/2026-07-28-coin-r9-r3c-multiseed")
K6_SEEDS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def _wilson(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def _rebuild(state: dict, obs_dim: int = AUG_DIM) -> Any:
    a = make_actor("td3", obs_dim, ACT_DIM)
    a.load_state_dict({k: torch.tensor(v) for k, v in state.items()})
    a.eval()
    return a


class _Env:
    def __init__(self) -> None:
        from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
        model, norm = _load_clone()
        snap, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
        self.model, self.norm = model, norm
        self.entry = kc.freeze_kinetic_entry(snap, seed=kc.S1_SEED).tsnap
        self.bounds = ResidualBounds(alpha=ALPHA0)

    def cf(self) -> CloneActor:
        return CloneActor(self.model, self.norm)


def _rollout_metrics(env: _Env, controller: Any) -> dict:
    """Canonical K6 verdict + cleanliness/safety of one teacher-free rollout (the frozen `velocity_rollout` physics)."""
    m = velocity_rollout(env.entry, controller, DELIVERY_CFG)
    kin = [r for r in controller.clone_trace if r["kind"] == "KINETIC_CLONE"]
    vpar = [r["v_par"] for r in kin]
    stalls = sum(1 for v in vpar if v <= 0.0)
    clamps = sum(1 for r in kin if min(r["fn_l"], r["fn_r"]) > krl2.FN_CLAMP)
    reversals = sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0.0)
    return {"k6": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
            "min_dtz_mm": round(_min_dtz_mm(env.entry, m), 2), "dtz_end_mm": round(m["dtz_end"] * 1000, 2),
            "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5), "peak_qdot": round(float(m["peak_qdot"]), 3),
            "stalls": stalls, "clamps": clamps, "reversals": reversals,
            "clean": bool(stalls == 0 and clamps == 0 and reversals == 0), "coin_trace": m["coin_trace"]}


# ------------------------------------------------------------------------------------------------------------------------
# A. Frozen-policy intervention over the 22 verified K6 checkpoints
# ------------------------------------------------------------------------------------------------------------------------
def _intervene_one(env: _Env, ckpt: dict, mask: tuple) -> dict:
    r2_fn = deterministic_residual(_rebuild(ckpt["r2_champ_state"]))
    exp_fn = deterministic_residual(_rebuild(ckpt["expansion_state"]))
    ctrl = ab.AblationUnlockController(env.entry, env.cf(), exp_fn, env.bounds, r2_fn=r2_fn, beta=ckpt["beta"], include=mask)
    m = _rollout_metrics(env, ctrl)
    m.pop("coin_trace")
    return m


def _summarise_mode(rows: list) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["k6"])
    return {"n": n, "k6": k, "k6_rate": round(k / n, 4) if n else 0.0, "wilson95": _wilson(k, n),
            "safe_all": all(r["safe"] for r in rows), "clean_all": all(r["clean"] for r in rows),
            "min_dtz_median": round(float(np.median([r["min_dtz_mm"] for r in rows])), 2) if rows else None,
            "dwell_median": round(float(np.median([r["k6_dwell"] for r in rows])), 1) if rows else None,
            "stall_clamp_reversal": [sum(r["stalls"] for r in rows), sum(r["clamps"] for r in rows),
                                     sum(r["reversals"] for r in rows)]}


def run_frozen_intervention(out: Path = OUT) -> dict:
    env = _Env()
    ckpts = {s: json.load(open(CKPT_DIR / f"seed_{s:02d}" / "checkpoint.json")) for s in K6_SEEDS}
    per_mode: dict = {}
    for name, mask in ab.INTERVENTIONS.items():
        rows = [{"seed": s, **_intervene_one(env, ckpts[s], mask)} for s in K6_SEEDS]
        per_mode[name] = {"summary": _summarise_mode(rows), "rows": rows}
        sm = per_mode[name]["summary"]
        print(f"  {name:15s} K6 {sm['k6']:2d}/{sm['n']} rate {sm['k6_rate']:.3f} {sm['wilson95']}  "
              f"min_dtz~{sm['min_dtz_median']}mm dwell~{sm['dwell_median']}  clean_all {sm['clean_all']} safe_all {sm['safe_all']}  "
              f"s/c/r {sm['stall_clamp_reversal']}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"contract": "C1_FROZEN_POLICY_INTERVENTION_V1", "immutable_source": "2478a35d", "n_checkpoints": len(K6_SEEDS),
               "interventions": {k: v["summary"] for k, v in per_mode.items()}, "per_mode": per_mode}
    (out / "frozen_intervention.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list) -> None:
    t0 = time.time()
    if "--frozen" in argv or not any(a in argv for a in ("--frozen", "--retrain")):
        print("C1 FROZEN-POLICY INTERVENTION (22 verified K6 checkpoints; no training)")
        r = run_frozen_intervention()
        full = r["interventions"]["FULL"]
        print(f"\n  FULL is the control ({full['k6']}/{full['n']} K6). Load-bearing = the drop when a term is removed.")
    if "--retrain" in argv:
        from hymeko_rl.experiments.coin_kinetic_ablation_retrain import run_retraining
        run_retraining()
    print(f"  wall {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main(sys.argv)
