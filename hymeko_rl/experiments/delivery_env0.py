"""DELIVERY-ENV-0 — coin delivery termination / action-abstraction correction.

COIN-DELIVERY-1 showed the env TERMINATES on handoff (4-step both-contact) before the coin reaches the zone, so
grasp-and-carry is interrupted; `carry_pulse` helps only by AVOIDING handoff termination. Hypothesis: the delivery
bottleneck is partly an EPISODE-SEMANTICS artifact. This module adds a NON-INVASIVE evaluation wrapper that ignores
handoff-termination and continues until delivery (in_zone) / max horizon / safety — WITHOUT changing dynamics or reward —
and re-scores the scripted primitives to see whether post-handoff continuation raises the delivery ceiling.

NO RL, NO CORE change, NO reward change, NO env-dynamics change. Handoff is logged as a PHASE event, never task success.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hymeko_rl.experiments.coin_delivery1 import PRIMS, _dir_to_zone, _rand_prim  # noqa: F401  (reuse primitives)
from hymeko_rl.experiments.pedc_selection import _env

E = Path("experiments/2026_07_20_delivery_env0")
_HELD = range(64_000, 64_090)


class CoinDeliveryEvalWrapper:
    """Non-invasive EVAL wrapper: ignores handoff-termination; the episode continues until delivery (in_zone), the
    (optionally extended) horizon, or a safety hard-failure. Dynamics and reward are UNCHANGED — only the terminal
    signal is re-mapped, and `max_horizon` (if given) extends the eval horizon (Stage-3 longer-horizon test). Handoff is
    logged as a phase event, never treated as terminal success."""

    def __init__(self, env, max_horizon: "int | None" = None, center_tol: "float | None" = None) -> None:
        self.env = env; self.inner = env._env; self._max_h = max_horizon; self._center_tol = center_tol

    def reset(self, seed: int):
        obs, info = self.env.reset(seed=int(seed))
        if self._max_h is not None:
            self.env._horizon = int(self._max_h)                                   # eval-semantics: how long we may run
        return obs, info

    def step(self, action):
        obs, r, _term_handoff, trunc, info = self.env.step(action)                 # dynamics + reward UNCHANGED
        m = self.inner._planar_metrics
        # delivery = coin in zone; with `center_tol` set, delivery/termination requires the coin at the zone CENTRE
        # (disk_to_zone <= center_tol), a tighter criterion than in_zone (disk_to_zone <= zone_half).
        delivered = (float(m.disk_to_zone) <= self._center_tol) if self._center_tol is not None else bool(m.in_zone)
        info = {**info, "handoff_event": bool(info.get("handoff_ready")),          # handoff = PHASE event only
                "delivery_success": bool(delivered), "in_zone": bool(m.in_zone)}
        terminated = bool(delivered or info.get("safety_violation"))               # terminate on DELIVERY-to-centre (or safety)
        return obs, r, terminated, trunc, info


def rollout(wrapper: CoinDeliveryEvalWrapper, seed: int, prim, max_h: int) -> dict:
    """Roll one scripted primitive under delivery semantics; capture delivery + the phase-separated metrics."""
    wrapper.reset(seed); inner = wrapper.inner; m = inner._planar_metrics
    start_dtz = float(m.disk_to_zone); deliv = handoff = False; min_dtz = start_dtz; final_dtz = start_dtz
    t_handoff = t_deliv = None; dtz_at_handoff = post_handoff_min = None; contact_dwell = 0
    for t in range(max_h):
        _o, _r, term, trunc, info = wrapper.step(np.clip(prim(inner, t), -1, 1).astype(np.float32))
        mm = inner._planar_metrics; dtz = float(mm.disk_to_zone)
        min_dtz = min(min_dtz, dtz); final_dtz = dtz; contact_dwell += int(bool(mm.left_contact and mm.right_contact))
        if info["handoff_event"] and not handoff:
            handoff = True; t_handoff = t; dtz_at_handoff = post_handoff_min = dtz
        if handoff:
            post_handoff_min = min(post_handoff_min, dtz)
        if info["delivery_success"]:
            deliv = True
            if t_deliv is None:
                t_deliv = t
        if term or trunc:
            break
    post_prog = round(dtz_at_handoff - post_handoff_min, 4) if handoff else 0.0
    return {"deliv": deliv, "handoff": handoff, "start_dtz": round(start_dtz, 4), "min_dtz": round(min_dtz, 4),
            "final_dtz": round(final_dtz, 4), "progress": round(start_dtz - min_dtz, 4),
            "time_to_handoff": t_handoff, "time_to_delivery": t_deliv, "post_handoff_transport": post_prog,
            "contact_dwell": contact_dwell}


def _summ(rs: list, n: int) -> dict:
    deliv = [r for r in rs if r["deliv"]]
    return {"delivery_success": round(sum(r["deliv"] for r in rs) / n, 4),
            "handoff_event_rate": round(sum(r["handoff"] for r in rs) / n, 4),
            "grasp_no_delivery_rate": round(sum(r["handoff"] and not r["deliv"] for r in rs) / n, 4),
            "post_handoff_transport_med": round(float(np.median([r["post_handoff_transport"] for r in rs])), 4),
            "final_dtz_med": round(float(np.median([r["final_dtz"] for r in rs])), 4),
            "time_to_delivery_med": (round(float(np.median([r["time_to_delivery"] for r in deliv])), 1) if deliv else None),
            "moved_coin_rate": round(sum(r["progress"] > 0.01 for r in rs) / n, 4)}


def _sweep(seeds, max_h: int) -> dict:
    env = _env(); w = CoinDeliveryEvalWrapper(env, max_horizon=max_h); rng = np.random.default_rng(0)
    rows = {name: [] for name in PRIMS}; rows["random_ctrl"] = []; n = 0
    for sd in seeds:
        n += 1
        for name, prim in PRIMS.items():
            rows[name].append(rollout(w, sd, prim, max_h))
        rows["random_ctrl"].append(rollout(w, sd, _rand_prim(rng), max_h))
        if n % 30 == 0:
            print(f"  [env0 h{max_h}] {n}/{len(list(seeds))} | deliv Σ "
                  f"{ {k: sum(r['deliv'] for r in v) for k, v in rows.items()} }", flush=True)
    per = {name: _summ(rows[name], n) for name in rows}
    ceil = [int(any(rows[name][i]["deliv"] for name in PRIMS)) for i in range(n)]
    per["delivery_primitives_CEILING"] = {"delivery_success": round(sum(ceil) / n, 4)}
    return per


def run(seeds=_HELD) -> dict:
    E.mkdir(parents=True, exist_ok=True); (E / "manifests").mkdir(exist_ok=True)
    print("=== DELIVERY-ENV-0: Stage 2 (delivery-semantics wrapper, horizon 40) ===", flush=True)
    s2 = _sweep(seeds, max_h=40); c2 = s2["delivery_primitives_CEILING"]["delivery_success"]
    gate2 = c2 >= 0.40 or (c2 - 0.30) >= 0.10
    stage3 = {}
    if not gate2:                                                                  # Stage 3: longer horizons
        print("\n=== Stage 3 (longer horizons) ===", flush=True)
        for h in (80, 120):
            stage3[f"h{h}"] = _sweep(seeds, max_h=h)
    ceilings = {"h40": c2, **{k: v["delivery_primitives_CEILING"]["delivery_success"] for k, v in stage3.items()}}
    best = max(ceilings.values()); gain = round(best - 0.30, 4)
    termination_mismatch = s2["grasp_carry"]["delivery_success"] > 0.178 + 0.05    # grasp_carry improves vs COIN-DELIVERY-1
    verdict = ("ENV0_PASS_delivery_feasible_after_termination_fix" if best >= 0.40 or gain >= 0.10
               else "ENV0_FAIL_action_abstraction_insufficient_even_with_delivery_termination")
    out = {"stage2_h40": s2, "stage3_longer_horizon": stage3, "ceilings": ceilings,
           "coin_delivery1_ceiling": 0.30, "best_delivery_ceiling": best, "gain_over_primitive_ceiling": gain,
           "termination_mismatch_confirmed": bool(termination_mismatch), "verdict": verdict}
    (E / "manifests" / "delivery_env0.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n=== DELIVERY-ENV-0 results (delivery-semantics wrapper) ===", flush=True)
    for name, m in s2.items():
        print(f"  h40 {name:28s} {m}", flush=True)
    for h, per in stage3.items():
        print(f"  {h} CEILING {per['delivery_primitives_CEILING']} | grasp_carry {per['grasp_carry']['delivery_success']} "
              f"| carry_pulse {per['carry_pulse']['delivery_success']}", flush=True)
    print(f"[DELIVERY-ENV-0] COIN-DELIVERY-1 ceiling 0.30 → wrapper best {best} (gain {gain}) | "
          f"termination_mismatch {termination_mismatch} → {verdict}", flush=True)
    return out


if __name__ == "__main__":
    run()
