"""R9 — delivery-focused CAUSAL residual TD3 harness (one file, mode flags; §6.5 #13).

R9 learns a bounded causal per-step increment Δa over the FROZEN R8 champion (`coin-r8-bounded-residual-heldout-improvement
-v1`). Modes: `--stage2` update-zero identity (Δa≡0 reproduces the R8 champion bit-for-bit); later stages add the delivery
curriculum, dev gate, validation delivery and the single blind final-panel eval. The blind final panel is SEALED separately
(`coin_r9_blind_panel.py`) and is NEVER touched here until STAGE 6. Every integrity constraint is kept hard (no teleport /
hidden force / teacher fallback / free release bit; unchanged torque/motion/certificate; reward independent from K6; exact
per-step Bellman provenance = the Δa emission only).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import (
    DeltaBounds, R9CausalResidualAdapter, ZeroDeltaActor)
from hymeko_rl.coin_delivery.theta_option.residual_adapter import ConstantResidualActor, ResidualBounds, ResidualTipAdapter
from hymeko_rl.coin_delivery.theta_option.residual_option_env import OBS_DIM, ACT_DIM, residual_init_obs
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.experiments.coin_r8_residual_rl import _max_diff, _rich_trace, _load_harness
from hymeko_rl.option_rl.agents import make_actor

OUT = "reports/2026-07-27-coin-r9-causal-residual-delivery"
BANK = "reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"
R8_CKPT = "reports/2026-07-27-coin-r8-residual-rl/ckpts/td3_seed2_best_val.pt"
DEV_TAGS = ("s1", "s3")
_METRIC_KEYS = ("dtz_start", "dtz_end", "forward", "cross", "peak_qdot", "peak_coin_speed", "terminal_coin_speed",
                "k6_max_dwell", "contact_lost_steps", "lost_before_release", "release_step", "gap_closed")


def _r8_champion() -> Any:
    """Load the FROZEN R8 dev-selected champion (TD3 seed2) actor."""
    actor = make_actor("td3", OBS_DIM, ACT_DIM)
    actor.load_state_dict(torch.load(R8_CKPT))
    actor.eval()
    return actor


def _r8_base_residual(actor: Any, snap: Any, params: Any, bounds: Any) -> np.ndarray:
    """The frozen R8 champion's CONSTANT residual for a cradle = mean_action(t=0 init obs) — the R9 base `a_R8`."""
    with torch.no_grad():
        a = actor.mean_action(torch.as_tensor(residual_init_obs(snap, params, bounds)[None]))[0].numpy()
    return np.asarray(a, np.float64)


def _identity_on_cradle(snap: Any, a_r8: np.ndarray, params: Any, bounds: Any, dbounds: Any, tol: float) -> dict:
    """Compare the frozen R8 champion (ConstantResidualActor(a_R8)) vs the R9 causal adapter at Δa≡0 over the full trace."""
    m_r8, tr_r8 = _rich_trace(snap, ResidualTipAdapter(snap, ConstantResidualActor(a_r8), params, bounds, DELIVERY_CFG))
    r9 = R9CausalResidualAdapter(snap, a_r8, ZeroDeltaActor(), params, bounds, dbounds, control_interval=4, cfg=DELIVERY_CFG)
    m_r9, tr_r9 = _rich_trace(snap, r9)
    diffs = _max_diff(tr_r8, tr_r9)
    metric_diff = {k: abs(float(m_r8[k]) - float(m_r9[k])) for k in _METRIC_KEYS}
    coin_equal = bool(np.array_equal(np.asarray(m_r8["coin_trace"]), np.asarray(m_r9["coin_trace"])))
    # every R9 step must have Δa == 0 and a_exec == a_R8 (the base) — provenance check
    prov_ok = all(all(abs(x) < tol for x in p["bellman_action"]) and
                  all(abs(ax - ar) < tol for ax, ar in zip(p["a_exec"], p["a_r8_base"])) for p in r9.provenance)
    trace_ok = coin_equal and max(diffs.values()) < tol and max(metric_diff.values()) < tol
    return {"a_r8": [round(float(x), 6) for x in a_r8], "trace_identity": trace_ok, "coin_trace_bit_equal": coin_equal,
            "delta_zero_and_a_exec_is_base": prov_ok, "n_steps": len(tr_r8),
            "max_step_diff": {k: round(v, 12) for k, v in diffs.items()},
            "max_metric_diff": {k: round(v, 12) for k, v in metric_diff.items()},
            "_max": max(max(diffs.values()), max(metric_diff.values()))}


def stage2_update_zero_identity() -> dict:
    """STAGE 2 — Δa≡0 reproduces the frozen R8 champion trajectory bit-for-bit on the dev cradles. Verdict
    R9_UPDATE_ZERO_REPRODUCES_R8_CHAMPION / …_FAILS."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    actor = _r8_champion()
    params, bounds, dbounds, tol = TipTransportParams(), ResidualBounds(), DeltaBounds(), 1e-9
    per: dict[str, Any] = {}
    for tag in DEV_TAGS:
        a_r8 = _r8_base_residual(actor, panel[tag].snap, params, bounds)
        st = _identity_on_cradle(panel[tag].snap, a_r8, params, bounds, dbounds, tol)
        mx = st.pop("_max")
        per[tag] = st
        print(f"   {tag}: identity={st['trace_identity']} coin_equal={st['coin_trace_bit_equal']} "
              f"delta0={st['delta_zero_and_a_exec_is_base']} a_r8={st['a_r8']} max_diff={mx:.2e}", flush=True)
    passed = all(v["trace_identity"] and v["delta_zero_and_a_exec_is_base"] for v in per.values())
    verdict = "R9_UPDATE_ZERO_REPRODUCES_R8_CHAMPION" if passed else "R9_UPDATE_ZERO_FAILS"
    out = {"contract": "COIN_R9_STAGE2_UPDATE_ZERO", "date": "2026-07-27", "tolerance": tol,
           "base": "frozen R8 champion TD3 seed2 (coin-r8-bounded-residual-heldout-improvement-v1)",
           "delta_bounds": {"d_fwd_vel": dbounds.d_fwd_vel, "d_squeeze": dbounds.d_squeeze,
                            "d_stop_gain": dbounds.d_stop_gain, "slew": dbounds.slew},
           "bellman_action": "Delta a in [-1,1]^3 (causal increment) ONLY; a_R8 base / a_exec / targets / torque = provenance",
           "per_state": per, "passed": passed, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/stage2_update_zero.json", "w"), indent=1, default=float)
    print(f"\n== R9 STAGE 2 ==\n  {verdict} | dev {sum(v['trace_identity'] for v in per.values())}/{len(per)} | "
          f"wall {out['wall_s']}s\nR9_STAGE2_DONE", flush=True)
    return out


def main(argv: "list[str]") -> None:
    if "--stage2" in argv:
        stage2_update_zero_identity()
    else:
        print("usage: coin_r9_causal_rl.py --stage2 | (stage3-6 added as gates pass)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
