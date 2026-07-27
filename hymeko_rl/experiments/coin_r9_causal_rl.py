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

from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import (
    CAUSAL_DIM, DeltaBounds, R9CausalResidualAdapter, ZeroDeltaActor)
from hymeko_rl.coin_delivery.theta_option.r9_delivery_train import (
    ACT_DIM as _R9_ACT, DeliveryReward, R9TD3Config, _dev_eval, train_causal_td3)
from hymeko_rl.option_rl.agents import DetActor
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


# ── STAGE 3 — per-decision delivery-focused TD3 (development cradles only) ────────────────────────────────────────────
CURRICULA = {
    "A": DeliveryReward(w_terminal=2.0, w_lowspeed=0.0, w_reversal=3.0),   # trajectory shaping (reduce fling/reversal, hold contact)
    "B": DeliveryReward(w_terminal=6.0, w_lowspeed=3.0, w_reversal=3.0),   # terminal approach (slow near target, don't stop short)
    "C": DeliveryReward(w_terminal=10.0, w_lowspeed=5.0, w_reversal=2.0)}  # strict delivery (enter 20mm zone, settle)


def _dev_cradles(panel: dict, actor: Any, params: Any, bounds: Any) -> list:
    """(tag, snap, a_R8) for the DEV cradles — a_R8 = the frozen R8 champion's constant residual per cradle."""
    return [(t, panel[t].snap, _r8_base_residual(actor, panel[t].snap, params, bounds)) for t in DEV_TAGS]


def _eval_ckpt(state: dict, dev: list, cfg: Any) -> dict:
    pol = DetActor(CAUSAL_DIM, _R9_ACT)
    pol.load_state_dict(state)
    return _dev_eval(pol, dev, cfg)


def stage3_train(curriculum: str = "A", seeds: tuple = (0, 1, 2), smoke: bool = False) -> dict:
    """STAGE 3 — train the causal residual TD3 on dev cradles under a curriculum reward. Reports update-0 (≈R8 champion),
    best_val and final dev delivery per seed. Dev-only; s4/s7 and the blind panel are untouched."""
    t0 = time.time()
    os.makedirs(f"{OUT}/ckpts", exist_ok=True)
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    params, bounds = TipTransportParams(), ResidualBounds()
    dev = _dev_cradles(panel, _r8_champion(), params, bounds)
    reward = CURRICULA[curriculum]
    cfg = R9TD3Config(total_rollouts=30, warmup_rollouts=8, eval_every=10) if smoke else R9TD3Config()
    if smoke:
        seeds = (0,)
    runs = []
    for seed in seeds:
        print(f"\n  ── curriculum {curriculum} · seed {seed} ──", flush=True)
        ckpts, info = train_causal_td3(dev, reward, cfg, seed=seed, log=lambda s: print(s, flush=True))
        upd0, best, final = (_eval_ckpt(ckpts[k], dev, cfg) for k in ("update0", "best_val", "final"))
        torch.save(ckpts["best_val"], f"{OUT}/ckpts/r9_{curriculum}_seed{seed}_best_val.pt")
        runs.append({"seed": seed, "distill_loss": info["distill_loss"], "update0_dev": upd0, "best_val_dev": best,
                     "final_dev": final, "ckpt": f"{OUT}/ckpts/r9_{curriculum}_seed{seed}_best_val.pt"})
        print(f"    seed {seed}: update0 K6 {upd0['k6']}/{upd0['n']} dtz {upd0['mean_dtz_mm']}mm -> "
              f"best_val K6 {best['k6']}/{best['n']} dtz {best['mean_dtz_mm']}mm (distill {info['distill_loss']})", flush=True)
    out = {"contract": "COIN_R9_STAGE3_DELIVERY_TD3", "curriculum": curriculum, "smoke": bool(smoke),
           "reward_weights": reward.__dict__, "config": cfg.__dict__, "dev_tags": list(DEV_TAGS), "runs": runs,
           "dev_k6_best_median": int(np.median([r["best_val_dev"]["k6"] for r in runs])),
           "gate_strict_k6_dev_2of2": bool(any(r["best_val_dev"]["k6"] == 2 for r in runs)), "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/stage3_{curriculum}{'_smoke' if smoke else ''}.json", "w"), indent=1, default=float)
    print(f"\n== R9 STAGE 3 ({curriculum}) ==\n  dev best K6 median {out['dev_k6_best_median']}/2 | "
          f"any 2/2 {out['gate_strict_k6_dev_2of2']} | wall {out['wall_s']}s\nR9_STAGE3_DONE", flush=True)
    return out


def _ceiling_on_cradle(snap: Any, a_r8: np.ndarray, params: Any, bounds: Any, dbounds: Any, n: int, seed: int) -> dict:
    """Sweep CONSTANT bounded Δa over the frozen R8 champion on ONE cradle; return the best achievable dtz_end + whether any
    bounded residual delivers strict K6. This is the residual-DELIVERY CEILING (measure it before blaming the optimiser)."""
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import ConstantDeltaActor
    rng = np.random.default_rng(seed)
    grid = [np.zeros(3)] + [np.eye(3)[i] * s for i in range(3) for s in (-1.0, 1.0)] + \
           [np.clip(rng.uniform(-1, 1, 3), -1, 1) for _ in range(n)]
    best_dtz, delivered, best_a = 1e9, False, None
    for a in grid:
        adapter = R9CausalResidualAdapter(snap, a_r8, ConstantDeltaActor(a), params, bounds, dbounds, control_interval=1, cfg=DELIVERY_CFG)
        from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
        m = velocity_rollout(snap, adapter, DELIVERY_CFG)
        safe = m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0
        if safe and float(m["dtz_end"]) < best_dtz:
            best_dtz, best_a = float(m["dtz_end"]), [round(float(x), 3) for x in a]
        if safe and bool(delivery_success(m, DELIVERY_CFG)):
            delivered = True
    return {"best_dtz_end_mm": round(best_dtz * 1000, 1), "any_bounded_residual_delivers": delivered, "best_a": best_a}


def stage3_ceiling_diag(n: int = 120) -> dict:
    """Discriminating test: the residual-delivery CEILING per dev cradle. If s1 is NOT deliverable by ANY bounded Δa over
    the R8 champion, the STAGE-3 stall is STRUCTURAL (bound too small / need a different base), not an optimiser failure."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    actor = _r8_champion()
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import DeltaBounds
    params, bounds, dbounds = TipTransportParams(), ResidualBounds(), DeltaBounds()
    per = {}
    for tag in DEV_TAGS:
        a_r8 = _r8_base_residual(actor, panel[tag].snap, params, bounds)
        c = _ceiling_on_cradle(panel[tag].snap, a_r8, params, bounds, dbounds, n, seed=hash(tag) % 9999)
        per[tag] = c
        print(f"   {tag}: best_dtz {c['best_dtz_end_mm']}mm delivers={c['any_bounded_residual_delivers']} best_a={c['best_a']}", flush=True)
    diag = "STRUCTURAL_BOUND_INSUFFICIENT" if not all(v["any_bounded_residual_delivers"] for v in per.values()) \
        else "OPTIMISER_OR_REWARD_LIMITED"
    out = {"contract": "COIN_R9_STAGE3_RESIDUAL_CEILING", "delta_bounds": dbounds.__dict__, "n_samples": n, "per_state": per,
           "all_dev_deliverable_by_bounded_residual": all(v["any_bounded_residual_delivers"] for v in per.values()),
           "diagnosis": diag, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/stage3_residual_ceiling.json", "w"), indent=1, default=float)
    print(f"\n== R9 STAGE 3 CEILING ==\n  diagnosis: {diag} | wall {out['wall_s']}s\nR9_DIAG_DONE", flush=True)
    return out


def main(argv: "list[str]") -> None:
    if "--stage2" in argv:
        stage2_update_zero_identity()
    elif "--diag" in argv:
        stage3_ceiling_diag()
    elif "--stage3" in argv:
        cur = next((a.split("=")[1] for a in argv if a.startswith("--curriculum=")), "A")
        stage3_train(curriculum=cur, smoke="--smoke" in argv)
    else:
        print("usage: coin_r9_causal_rl.py --stage2 | --stage3 [--curriculum=A|B|C] [--smoke]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
