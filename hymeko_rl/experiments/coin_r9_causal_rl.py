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
    # NOTE: a CONSTANT-Δa sweep is not a reachability CERTIFICATE — it does not prove no TEMPORAL sequence delivers s1.
    # It measures whether the searched (constant) family demonstrates delivery. The decisive test is `--reach` (R10-0).
    diag = "NO_DEMONSTRATED_DELIVERY_IN_CONSTANT_BOUNDED_FAMILY" \
        if not all(v["any_bounded_residual_delivers"] for v in per.values()) else "CONSTANT_BOUNDED_RESIDUAL_DELIVERS"
    out = {"contract": "COIN_R9_STAGE3_RESIDUAL_CEILING", "delta_bounds": dbounds.__dict__, "n_samples": n, "per_state": per,
           "all_dev_deliverable_by_bounded_residual": all(v["any_bounded_residual_delivers"] for v in per.values()),
           "diagnosis": diag, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/stage3_residual_ceiling.json", "w"), indent=1, default=float)
    print(f"\n== R9 STAGE 3 CEILING ==\n  diagnosis: {diag} | wall {out['wall_s']}s\nR9_DIAG_DONE", flush=True)
    return out


# ── R10-0 — residual REACHABILITY audit (teacher control + temporal-sequence CEM at declared vs full-range bound) ──────
def _teacher_theta(bank: dict, tag: str) -> "np.ndarray | None":
    for s in bank["states"]:
        if s["tag"] == tag:
            return np.asarray(s["canonical_theta_vec"], np.float64)
    return None


def _teacher_delivers(snap: Any, theta: np.ndarray) -> dict:
    """Positive control: does the frozen K6 TEACHER θ deliver this cradle (it is deliverable by SOME controller)?"""
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    m = rollout_primitive(snap, tuple(theta), DELIVERY_CFG)
    return {"dtz_end_mm": round(float(m["dtz_end"]) * 1000, 1), "k6": bool(delivery_success(m, DELIVERY_CFG))}


def _reach_search(snap: Any, a_r8: np.ndarray, dbounds: Any, params: Any, bounds: Any, *, n_seg: int = 6, pop: int = 48,
                  n_iter: int = 6, elite: int = 8, seed: int = 0) -> dict:
    """CEM over a per-frame Δa SEQUENCE (n_seg piecewise segments) minimising s-cradle dtz_end within `dbounds`. Returns the
    best safe dtz_end + whether strict K6 was reached by ANY searched temporal sequence."""
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import SegmentDeltaActor
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    rng = np.random.default_rng(seed)
    n_dec, dim = int(DELIVERY_CFG.horizon), n_seg * 3
    mu, sig = np.zeros(dim), 0.7 * np.ones(dim)
    best_dtz, best = 1e9, {"dtz_end_mm": None, "k6": False, "safe": False}
    for _ in range(n_iter):
        cand = np.clip(mu[None] + sig[None] * rng.standard_normal((pop, dim)), -1, 1)
        scores = []
        for c in cand:
            adapter = R9CausalResidualAdapter(snap, a_r8, SegmentDeltaActor(c.reshape(n_seg, 3), n_dec), params, bounds,
                                              dbounds, control_interval=1, cfg=DELIVERY_CFG)
            m = velocity_rollout(snap, adapter, DELIVERY_CFG)
            safe = m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0
            dtz = float(m["dtz_end"]) if safe else 1e9
            scores.append(dtz)
            if dtz < best_dtz:
                best_dtz = dtz
                best = {"dtz_end_mm": round(dtz * 1000, 1), "k6": bool(safe and delivery_success(m, DELIVERY_CFG)),
                        "safe": bool(safe)}
        order = np.argsort(scores)[:elite]
        mu = cand[order].mean(0)
        sig = np.clip(cand[order].std(0), 0.05, 1.0)
    return best


def _reach_case(s1: dict) -> str:
    if s1["reach_declared_bound"]["k6"]:
        return "A_LEARNABLE_WITHIN_DECLARED_BOUND_TD3_MISSED"      # a bounded temporal residual DOES deliver → learning-limited
    if s1["reach_full_range"]["k6"]:
        return "B_RESIDUAL_MAGNITUDE_LIMIT_CONFIRMED"              # only a larger residual delivers → per-channel bound growth
    return "C_BASE_OR_RESIDUAL_BASIS_INSUFFICIENT"                 # even full-range temporal residual fails → base/basis change


def reach_audit(n_seg: int = 6, pop: int = 48, n_iter: int = 6) -> dict:
    """R10-0 — the DECISIVE test. Confirms the teacher delivers (feasible), then CEM-searches the temporal residual sequence
    over the frozen R8 base at the DECLARED ±bound and at FULL range. Classifies A (learnable) / B (magnitude) / C (base)."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import DeltaBounds
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    actor = _r8_champion()
    params, bounds = TipTransportParams(), ResidualBounds()
    declared = DeltaBounds()
    full = DeltaBounds(d_fwd_vel=1.0, d_squeeze=0.15, d_stop_gain=1.0, slew=1.0)   # a_exec spans the full R8 residual range
    per: dict[str, Any] = {}
    for tag in DEV_TAGS:
        snap, a_r8 = panel[tag].snap, _r8_base_residual(actor, panel[tag].snap, params, bounds)
        per[tag] = {"teacher": _teacher_delivers(snap, _teacher_theta(bank, tag)),
                    "reach_declared_bound": _reach_search(snap, a_r8, declared, params, bounds, n_seg=n_seg, pop=pop, n_iter=n_iter, seed=1),
                    "reach_full_range": _reach_search(snap, a_r8, full, params, bounds, n_seg=n_seg, pop=pop, n_iter=n_iter, seed=2)}
        print(f"   {tag}: teacher {per[tag]['teacher']} | declared {per[tag]['reach_declared_bound']} | "
              f"full {per[tag]['reach_full_range']}", flush=True)
    case = _reach_case(per["s1"])
    out = {"contract": "COIN_R9_R10_0_REACHABILITY_AUDIT", "n_seg": n_seg, "search": {"pop": pop, "n_iter": n_iter},
           "per_state": per, "s1_case": case,
           "note": "a POSITIVE (teacher) control confirms feasibility; the CEM is an existence search, not a proof of "
                   "non-existence — a null result is NO_DEMONSTRATED_DELIVERY, strengthened by the full-range search.",
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10_0_reachability.json", "w"), indent=1, default=float)
    print(f"\n== R10-0 REACHABILITY ==\n  s1 CASE: {case} | wall {out['wall_s']}s\nR9_REACH_DONE", flush=True)
    return out


# ── R10-B0 — M0 base-coverage audit (find a NEAR base for s1 by re-tuning the scaffold; far base = R8/default) ─────────
_M0_LO = np.array([1.0, 0.04, 2.0, 0.2, 0.02, 0.08])       # k_d, v_max, k_q, k_v, settle_speed, squeeze_hold
_M0_HI = np.array([10.0, 0.25, 10.0, 2.0, 0.12, 0.22])


def _tp_from_vec(v: np.ndarray) -> Any:
    return TipTransportParams(k_d=float(v[0]), v_max=float(v[1]), k_q=float(v[2]), k_v=float(v[3]),
                              settle_speed=float(v[4]), squeeze_hold=float(v[5]))


def _base_delivers(snap: Any, tp: Any) -> dict:
    """Roll the tip-transport scaffold (zero residual) with params `tp`; return dtz_end + strict K6 + safety."""
    from hymeko_rl.coin_delivery.theta_option.tip_transport import TipReferencedController
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    m = velocity_rollout(snap, TipReferencedController(snap, tp, DELIVERY_CFG), DELIVERY_CFG)
    safe = bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0)
    return {"dtz_end_mm": round(float(m["dtz_end"]) * 1000, 1), "k6": bool(safe and delivery_success(m, DELIVERY_CFG)), "safe": safe}


def _init_geo(snap: Any) -> dict:
    from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
    rl = snap.branch()
    _u, dtz0 = rl.inner.direction_to_zone()
    return {"dtz0_mm": round(float(dtz0) * 1000, 1), "coin_x0": round(float(np.asarray(_coin_xy(rl), np.float64)[0]), 4)}


def _near_base_search(snap: Any, *, n_iter: int = 8, pop: int = 40, elite: int = 8, seed: int = 0) -> dict:
    """CEM over the scaffold params (dev s1 ONLY) for a NEAR base that delivers s1 with ZERO residual."""
    rng = np.random.default_rng(seed)
    mu, sig = (_M0_LO + _M0_HI) / 2, (_M0_HI - _M0_LO) / 4
    best_dtz, best = 1e9, {"dtz_end_mm": None, "k6": False, "params": None}
    for _ in range(n_iter):
        cand = np.clip(mu[None] + sig[None] * rng.standard_normal((pop, 6)), _M0_LO, _M0_HI)
        scores = []
        for c in cand:
            r = _base_delivers(snap, _tp_from_vec(c))
            dtz = r["dtz_end_mm"] if r["safe"] else 1e9
            scores.append(dtz)
            if dtz < best_dtz:
                best_dtz = dtz
                best = {"dtz_end_mm": r["dtz_end_mm"], "k6": r["k6"], "params": [round(float(x), 4) for x in c]}
        order = np.argsort(scores)[:elite]
        mu = cand[order].mean(0)
        sig = np.clip(cand[order].std(0), (_M0_HI - _M0_LO) * 0.02, _M0_HI - _M0_LO)
    return best


def m0_base_coverage() -> dict:
    """R10-B0 — can a re-tuned NEAR base deliver s1 (dev-s1-only) while the FAR (default) base delivers s3, and are s1/s3
    causally separable for a gate? Answers whether the two-base R10-B is viable BEFORE any TD3."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    far = TipTransportParams()
    far_s3, far_s1 = _base_delivers(panel["s3"].snap, far), _base_delivers(panel["s1"].snap, far)
    near_s1 = _near_base_search(panel["s1"].snap)
    near_on_s3 = _base_delivers(panel["s3"].snap, _tp_from_vec(near_s1["params"])) if near_s1["params"] else None
    geo = {t: _init_geo(panel[t].snap) for t in DEV_TAGS}
    separable = bool(abs(geo["s1"]["dtz0_mm"] - geo["s3"]["dtz0_mm"]) > 10.0)
    viable = bool(near_s1["k6"] and far_s3["k6"] and separable)
    verdict = "TWO_BASE_R10B_VIABLE" if viable else (
        "NEAR_BASE_NOT_IN_SCAFFOLD_FAMILY" if not near_s1["k6"] else "GATE_OR_FAR_BASE_ISSUE")
    out = {"contract": "COIN_R9_R10B0_BASE_COVERAGE", "far_base": "default TipTransportParams",
           "far_base_s3": far_s3, "far_base_s1": far_s1, "near_base_s1_search": near_s1, "near_base_on_s3": near_on_s3,
           "init_geometry": geo, "gate_separable_by_dtz0": separable, "two_base_viable": viable, "verdict": verdict,
           "note": "near base built from dev s1 ONLY; far base = default; s4/s7 untouched. If NEAR_BASE_NOT_IN_SCAFFOLD_"
                   "FAMILY, s1 needs a different near CONTROLLER / 4th channel, not just a recentered base.",
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10b0_base_coverage.json", "w"), indent=1, default=float)
    print(f"   far base: s3 {far_s3} · s1 {far_s1}\n   near base (s1-tuned): {near_s1} · on s3 {near_on_s3}\n"
          f"   geometry {geo} separable={separable}\n== R10-B0 ==\n  {verdict} | wall {out['wall_s']}s\nR9_M0_DONE", flush=True)
    return out


# ── R10-C0 — event-aligned teacher/scaffold trace audit (localise the physical d.o.f. the scaffold cannot express) ─────
def _phys_hook(store: list):
    """A frame_hook capturing the SAME physical signals for any rollout (teacher via rollout_primitive OR scaffold via
    velocity_rollout): dtz, coin speed, along/cross-track velocity, spin, L/R contact force + imbalance, contact, qdot."""
    from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
    from hymeko_rl.coin_delivery.forward_displacement import _coin_speed
    from hymeko_rl.coin_delivery.theta_option.residual_adapter import _coin_spin

    def hook(rl: Any, t: int) -> None:
        u, dtz = rl.inner.direction_to_zone()
        uu = np.asarray(u, np.float64)[:2]
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        con = primary_fingertip_contacts(rl)
        fnl = float(con["left"]["fn"]) if con["left"] is not None else 0.0
        fnr = float(con["right"]["fn"]) if con["right"] is not None else 0.0
        store.append({"t": int(t), "dtz_mm": float(dtz) * 1000, "speed": float(_coin_speed(rl)),
                      "v_par": float(np.dot(vel, uu)), "v_lat": float(np.dot(vel, np.array([-uu[1], uu[0]]))),
                      "spin": float(_coin_spin(rl)), "fn_l": fnl, "fn_r": fnr,
                      "contact": int(fnl > 0 or fnr > 0), "qdot_max": float(np.max(np.abs(rl.inner.data.qvel[:4])))})
    return hook


def _summ(tr: list) -> dict:
    """PHASE-structured signature of a physical trace — the APPROACH→HOLD/TRANSPORT→BRAKE/SETTLE→RELEASE phase events +
    the signals that distinguish each phase (peak speed, deceleration, stored energy at release, transport reach, lateral)."""
    dtz = [r["dtz_mm"] for r in tr]
    spd = [r["speed"] for r in tr]
    con = [r["contact"] for r in tr]
    pk = max(spd) if spd else 0.0
    pk_t = int(np.argmax(spd)) if spd else 0
    t_brake = next((r["t"] for r in tr[pk_t:] if r["speed"] < 0.8 * pk), None)       # first sustained decel after peak speed
    return {"peak_vpar": round(max(r["v_par"] for r in tr), 4), "peak_speed": round(pk, 4),
            "terminal_speed": round(spd[-1], 4), "decel_ratio": round(1.0 - spd[-1] / (pk + 1e-9), 3),
            "max_abs_vlat": round(max(abs(r["v_lat"]) for r in tr), 4),
            "max_abs_imbalance": round(max(abs(r["fn_l"] - r["fn_r"]) for r in tr), 4),
            "min_dtz_mm": round(min(dtz), 1), "reaches_zone": bool(min(dtz) <= 20.0), "terminal_dtz_mm": round(dtz[-1], 1),
            "contact_frac": round(float(np.mean(con)), 3),
            "t_first_contact": next((r["t"] for r in tr if r["contact"] == 1), None), "t_peak_speed": pk_t,
            "t_brake_onset": t_brake, "t_zone_entry": next((r["t"] for r in tr if r["dtz_mm"] <= 20.0), None),
            "t_contact_lost": next((r["t"] for r in tr if r["contact"] == 0 and r["t"] > 3), len(tr))}


def _c0_classify(teach: dict, scaf: dict) -> "tuple[str, list]":
    """Localise which trajectory PHASE the monolithic scaffold fails to represent (the hybrid-mode axis)."""
    clues = []
    if not scaf["reaches_zone"] and teach["reaches_zone"]:
        clues.append(f"APPROACH/TRANSPORT: scaffold never reaches zone (min_dtz {scaf['min_dtz_mm']} vs {teach['min_dtz_mm']})")
    if scaf["t_brake_onset"] is None and teach["t_brake_onset"] is not None:
        clues.append("BRAKE: scaffold shows NO distinct deceleration phase (no brake primitive)")
    if scaf["decel_ratio"] < 0.5 * teach["decel_ratio"]:
        clues.append(f"BRAKE: scaffold barely decelerates (decel_ratio {scaf['decel_ratio']} vs {teach['decel_ratio']})")
    if scaf["terminal_speed"] > 2 * teach["terminal_speed"] + 1e-3:
        clues.append(f"RELEASE/SETTLE: scaffold retains stored energy (terminal_speed {scaf['terminal_speed']} vs {teach['terminal_speed']})")
    if teach["max_abs_vlat"] > 2 * scaf["max_abs_vlat"] + 1e-3:
        clues.append(f"HOLD: teacher uses lateral trim the scaffold suppresses (|v_lat| {teach['max_abs_vlat']} vs {scaf['max_abs_vlat']})")
    if not scaf["reaches_zone"]:
        prim = "APPROACH_TRANSPORT_PHASE_INSUFFICIENT"
    elif scaf["terminal_speed"] > teach["terminal_speed"] + 0.05:
        prim = "BRAKE_SETTLE_PHASE_INSUFFICIENT"
    else:
        prim = "RELEASE_TIMING_PHASE"
    return prim, clues


def c0_trace_audit() -> dict:
    """R10-C0 — compare the DELIVERING s1 teacher vs the non-delivering scaffold (R8 base) on the same physical signals;
    localise which d.o.f. the tip-transport servo cannot express. No training; s1 (dev) only."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    snap = panel["s1"].snap
    a_r8 = _r8_base_residual(_r8_champion(), snap, TipTransportParams(), ResidualBounds())
    t_store: list = []
    rollout_primitive(snap, tuple(_teacher_theta(bank, "s1")), DELIVERY_CFG, frame_hook=_phys_hook(t_store))
    s_store: list = []
    velocity_rollout(snap, ResidualTipAdapter(snap, ConstantResidualActor(a_r8), TipTransportParams(), ResidualBounds(),
                                              DELIVERY_CFG), DELIVERY_CFG, frame_hook=_phys_hook(s_store))
    teach, scaf = _summ(t_store), _summ(s_store)
    case, clues = _c0_classify(teach, scaf)
    out = {"contract": "COIN_R9_R10C0_TRACE_AUDIT", "cradle": "s1", "teacher": teach, "scaffold_r8_base": scaf,
           "primary_case": case, "missing_dof_clues": clues,
           "note": "delivering teacher vs non-delivering scaffold on identical physical signals; localises the d.o.f. gap "
                   "that R10-C2's minimal new near primitive must supply (NOT a proof).", "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c0_trace_audit.json", "w"), indent=1, default=float)
    print(f"   teacher s1: {teach}\n   scaffold  s1: {scaf}\n   CASE: {case}\n   clues: {clues}\n"
          f"== R10-C0 ==\n  wall {out['wall_s']}s\nR9_C0_DONE", flush=True)
    return out


def _scaffold_peak_vpar(snap: Any, a_r8: np.ndarray, dbounds: Any, seg: np.ndarray, n_dec: int) -> "tuple[float, float, bool]":
    """Roll the scaffold+segment residual; return (peak forward coin velocity, dtz_end, safe). Peak v_par = APPROACH effort."""
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import SegmentDeltaActor
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    adapter = R9CausalResidualAdapter(snap, a_r8, SegmentDeltaActor(seg, n_dec), TipTransportParams(), ResidualBounds(),
                                      dbounds, control_interval=1, cfg=DELIVERY_CFG)
    peak = {"v": -1e9}

    def hook(rl: Any, t: int) -> None:
        u, _d = rl.inner.direction_to_zone()
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        peak["v"] = max(peak["v"], float(np.dot(vel, np.asarray(u, np.float64)[:2])))
    m = velocity_rollout(snap, adapter, DELIVERY_CFG, frame_hook=hook)
    return peak["v"], float(m["dtz_end"]), bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0)


def c1_projection(n_seg: int = 6, pop: int = 40, n_iter: int = 5) -> dict:
    """R10-C1 — is the APPROACH forward-effort/impulse the UNEXPRESSIBLE component of the current 3-channel basis? CEM
    MAXIMISES the scaffold's peak forward coin velocity on s1 over the full residual range; compares to the teacher's peak.
    A large gap localises r_perp to APPROACH forward-effort ⇒ justifies a dedicated APPROACH impulse primitive (C2)."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import DeltaBounds
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    snap = panel["s1"].snap
    a_r8 = _r8_base_residual(_r8_champion(), snap, TipTransportParams(), ResidualBounds())
    t_store: list = []
    rollout_primitive(snap, tuple(_teacher_theta(bank, "s1")), DELIVERY_CFG, frame_hook=_phys_hook(t_store))
    teach_peak = round(max(r["v_par"] for r in t_store), 4)
    full = DeltaBounds(d_fwd_vel=1.0, d_squeeze=0.15, d_stop_gain=1.0, slew=1.0)
    rng, n_dec, dim = np.random.default_rng(3), int(DELIVERY_CFG.horizon), n_seg * 3
    mu, sig = np.zeros(dim), 0.7 * np.ones(dim)
    best = {"peak_vpar": -1e9, "dtz_end_mm": None}
    for _ in range(n_iter):
        cand = np.clip(mu[None] + sig[None] * rng.standard_normal((pop, dim)), -1, 1)
        pk = []
        for c in cand:
            v, dtz, safe = _scaffold_peak_vpar(snap, a_r8, full, c.reshape(n_seg, 3), n_dec)
            pk.append(v if safe else -1e9)
            if safe and v > best["peak_vpar"]:
                best = {"peak_vpar": round(v, 4), "dtz_end_mm": round(dtz * 1000, 1)}
        order = np.argsort(pk)[::-1][:8]
        mu = cand[order].mean(0)
        sig = np.clip(cand[order].std(0), 0.05, 1.0)
    gap = round(teach_peak - best["peak_vpar"], 4)
    localised = bool(best["peak_vpar"] < 0.7 * teach_peak)
    out = {"contract": "COIN_R9_R10C1_APPROACH_EFFORT_PROJECTION", "cradle": "s1",
           "teacher_peak_vpar": teach_peak, "scaffold_max_peak_vpar_full_residual": best["peak_vpar"],
           "gap": gap, "scaffold_best_effort_dtz_mm": best["dtz_end_mm"],
           "r_perp_localised_to_approach_forward_effort": localised,
           "reading": ("even MAXIMISING forward effort over the full residual range, the scaffold cannot approach the "
                       "teacher's peak forward velocity ⇒ the APPROACH momentum-build is the unexpressible component of the "
                       "current 3-channel basis; a dedicated APPROACH impulse primitive (C2) is justified" if localised else
                       "scaffold can match the teacher's peak forward velocity ⇒ APPROACH effort is expressible; look "
                       "elsewhere (HOLD/BRAKE/RELEASE phase)"), "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c1_approach_projection.json", "w"), indent=1, default=float)
    print(f"   teacher peak v_par {teach_peak} | scaffold MAX peak v_par (full residual) {best['peak_vpar']} "
          f"(best-effort dtz {best['dtz_end_mm']}mm) | gap {gap} | approach-unexpressible {localised}\n"
          f"== R10-C1 ==\n  wall {out['wall_s']}s\nR9_C1_DONE", flush=True)
    return out


# ── R10-C2 — APPROACH momentum-build mechanism gate (dev s1 only; staged C2-A..D) ────────────────────────────────────
def _run_approach(snap: Any, ap: Any, *, enabled: bool = True) -> dict:
    """Roll the HybridApproachController on a cradle; capture peak v_par, min dtz, K6, safety, and the approach exit."""
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import HybridApproachController
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    ctrl = HybridApproachController(snap, TipTransportParams(), ap, DELIVERY_CFG, enabled=enabled)
    acc = {"v": -1e9, "d": 1e9}

    def hook(rl: Any, t: int) -> None:
        u, d = rl.inner.direction_to_zone()
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        acc["v"] = max(acc["v"], float(np.dot(vel, np.asarray(u, np.float64)[:2])))
        acc["d"] = min(acc["d"], float(d))
    m = velocity_rollout(snap, ctrl, DELIVERY_CFG, frame_hook=hook)
    safe = bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0)
    carry_dist = (round(ctrl.approach_end["dtz_mm"] - ctrl.brake_start["dtz_mm"], 1)
                  if getattr(ctrl, "approach_end", None) and getattr(ctrl, "brake_start", None) else None)
    return {"qdot_approach": ap.qdot_approach, "launch": [ap.launch_vlo, ap.launch_vhi], "peak_vpar": round(acc["v"], 4),
            "min_dtz_mm": round(acc["d"] * 1000, 1), "dtz_end_mm": round(float(m["dtz_end"]) * 1000, 1),
            "k6": bool(safe and delivery_success(m, DELIVERY_CFG)), "safe": safe, "exit_step": ctrl.approach_exit_step,
            "exit_reason": ctrl.exit_reason, "peak_qdot": round(float(m["peak_qdot"]), 3),
            "peak_coin": round(float(m["peak_coin_speed"]), 3),
            "approach_end": getattr(ctrl, "approach_end", None), "brake_start": getattr(ctrl, "brake_start", None),
            "carry_distance_mm": carry_dist, "release_state": getattr(ctrl, "release_state", None),
            "reacquire_start": getattr(ctrl, "reacquire_start", None), "reacquire_end": getattr(ctrl, "reacquire_end", None)}


def c28_reacquire_audit() -> dict:
    """R10-C2.8 (R0/R1) — does a post-coast RE-ACQUIRE primitive EXIST? After RELEASED_COAST, gently catch the coasting coin
    (bounded forward velocity + ramped squeeze) once it slows into the reachable corridor, then hand to the frozen settle.
    Measures reachability (R0), gentle re-grip without fling (R1), and the full chain K6 (R4). Dev s1 only; settle/cert frozen."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    snap = {ps.tag: ps for ps in build_panel(_load_harness(), json.load(open(BANK)))}["s1"].snap
    base = {"qdot_approach": 1.6, "launch_vlo": 5.0, "launch_vhi": 6.0, "impulse_budget": 10.0, "max_steps": 40,
            "acquire_squeeze": 0.2}
    runs = []
    for k in range(3, 15, 1):                                                # release at various steps → coast to various dtz
        r = _run_approach(snap, ApproachParams(**base, release_at_step=k, reacquire=True))
        rs, re = r.get("reacquire_start"), r.get("reacquire_end")
        dtz_push = (round(re["dtz_mm"] - rs["dtz_mm"], 1) if rs and re else None)   # +ve ⇒ re-acquire pushed the coin AWAY
        runs.append({"release_step": k, "reached_corridor": bool(rs), "regrip_success": bool(re and re.get("success")),
                     "dtz_push_mm": dtz_push, "v_par_at_regrip": (re or {}).get("v_par_at_regrip"),
                     "dtz_end_mm": r["dtz_end_mm"], "k6": r["k6"], "safe": r["safe"], "min_dtz_mm": r["min_dtz_mm"]})
    reached = [r for r in runs if r["reached_corridor"] and r["safe"]]
    gentle = [r for r in reached if r["regrip_success"] and (r["dtz_push_mm"] is None or r["dtz_push_mm"] <= 5.0)]
    delivered = [r for r in gentle if r["k6"]]
    best = min([r for r in runs if r["safe"]], key=lambda r: r["dtz_end_mm"]) if runs else None
    if delivered:
        verdict = "REACQUIRE_CHAIN_DELIVERS_S1"                               # R4 pass
    elif gentle:
        verdict = "REACQUIRE_FEASIBLE_SETTLE_NOT_YET"                         # R0/R1 pass, R3/R4 tuning
    elif reached:
        verdict = "REACQUIRE_CONTACT_NOT_STABILISED"                         # R0 pass, R1 fails (fling / no stable re-grip)
    else:
        verdict = "REACQUIRE_GEOMETRICALLY_UNREACHABLE"                      # R0 fails
    out = {"contract": "COIN_R9_R10C28_REACQUIRE", "cradle": "s1", "R0_reached": len(reached), "R1_gentle": len(gentle),
           "R4_delivered": len(delivered), "verdict": verdict, "best_chain": best, "n_runs": len(runs), "runs": runs,
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c28_reacquire.json", "w"), indent=1, default=float)
    print(f"   R0 reached corridor {len(reached)}/{len(runs)} | R1 gentle re-grip {len(gentle)} | R4 delivered {len(delivered)}\n"
          f"   best chain: {best['dtz_end_mm'] if best else None}mm K6 {best['k6'] if best else None}\n"
          f"== R10-C2.8 ==\n  {verdict} | wall {out['wall_s']}s\nR9_C28_DONE", flush=True)
    return out


def _coast_branches(snap: Any, coll: dict) -> list:
    """C2.6 data-collection: force RELEASED_COAST at a grid of steps; from each (v_release, d_release, landing) estimate the
    passive coast deceleration a_coast = v²/2·coast_dist. No online a estimate is USED for control — this only bounds it."""
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    out = []
    for k in range(2, 16, 1):
        r = _run_approach(snap, ApproachParams(**coll, release_at_step=k))
        rs = r.get("release_state")
        if not (rs and r["safe"]):
            continue
        v, d, land = rs["v_par"], rs["dtz_mm"] / 1000.0, r["min_dtz_mm"] / 1000.0
        coast = d - land
        if v > 0.03 and coast > 0.003:
            out.append({"release_step": k, "v": round(v, 4), "d_release_mm": rs["dtz_mm"], "landing_mm": r["min_dtz_mm"],
                        "a_coast": round(v * v / (2.0 * coast), 4), "k6": r["k6"]})
    return out


def c26_coast_guard_audit() -> dict:
    """R10-C2.6 — a ROBUST coast-entry guard (R5 lesson: interval, not point, coast estimate). Collect passive-coast branches
    on s1, bound the deceleration [a_min,a_max], then deploy `coast_guard` (RELEASE when the predicted landing ⊆ corridor) on
    s1 (target) + s3 (check). Verdicts: RELEASED_COAST_MODE_LOAD_BEARING / RELEASE_GUARD_PREDICTION_INSUFFICIENT /
    COAST_DYNAMICS_TOO_UNCERTAIN_FOR_CURRENT_GUARD."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), json.load(open(BANK)))}
    coll = {"qdot_approach": 1.6, "launch_vlo": 5.0, "launch_vhi": 6.0, "impulse_budget": 10.0, "max_steps": 40,
            "acquire_squeeze": 0.2}                                       # gentler push + firmer grip → contact holds to release
    branches = _coast_branches(panel["s1"].snap, coll)
    if len(branches) < 3:
        out = {"contract": "COIN_R9_R10C26_COAST_GUARD", "verdict": "COAST_DATA_INSUFFICIENT", "branches": branches,
               "wall_s": round(time.time() - t0, 1)}
        json.dump(out, open(f"{OUT}/r10c26_coast_guard.json", "w"), indent=1, default=float)
        print(f"== R10-C2.6 ==\n  COAST_DATA_INSUFFICIENT ({len(branches)} branches)\nR9_C26_DONE", flush=True)
        return out
    a_vals = [b["a_coast"] for b in branches]
    a_min, a_max = round(float(np.percentile(a_vals, 20)), 3), round(float(np.percentile(a_vals, 80)), 3)
    spread = round(a_max / max(a_min, 1e-3), 2)
    gp = {**coll, "coast_guard": True, "guard_amin": a_min, "guard_amax": a_max}
    dep_s1 = _run_approach(panel["s1"].snap, ApproachParams(**gp))
    dep_s3 = _run_approach(panel["s3"].snap, ApproachParams(**gp))
    if dep_s1["k6"]:
        verdict = "RELEASED_COAST_MODE_LOAD_BEARING"
    elif spread > 3.5:
        verdict = "COAST_DYNAMICS_TOO_UNCERTAIN_FOR_CURRENT_GUARD"
    else:
        verdict = "RELEASE_GUARD_PREDICTION_INSUFFICIENT"
    out = {"contract": "COIN_R9_R10C26_COAST_GUARD", "n_branches": len(branches), "a_coast_range": [a_min, a_max],
           "a_spread": spread, "branches": branches, "deploy_s1": dep_s1, "deploy_s3": dep_s3, "verdict": verdict,
           "s1_k6": dep_s1["k6"], "s3_k6": dep_s3["k6"], "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c26_coast_guard.json", "w"), indent=1, default=float)
    print(f"   coast a∈[{a_min},{a_max}] spread {spread} from {len(branches)} branches\n"
          f"   deploy s1: dtz {dep_s1['dtz_end_mm']}mm K6 {dep_s1['k6']} rel@{(dep_s1.get('release_state') or {}).get('t')} | "
          f"s3: dtz {dep_s3['dtz_end_mm']}mm K6 {dep_s3['k6']}\n== R10-C2.6 ==\n  {verdict} | wall {out['wall_s']}s\n"
          f"R9_C26_DONE", flush=True)
    return out


def c25_handoff_audit() -> dict:
    """R10-C2.5 — handoff phase-existence audit: is the second mode HELD_MOMENTUM_CARRY (grip retained) or a teacher-like
    PASSIVE_RELEASE coast? Sweeps carry (held forward-effort × duration) + a diagnostic released-coast branch after a fixed
    momentum-building APPROACH; classifies which handoff reaches s1 K6. Dev s1 only; brake/cert/physics frozen."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    snap = {ps.tag: ps for ps in build_panel(_load_harness(), json.load(open(BANK)))}["s1"].snap
    base = {"qdot_approach": 2.4, "launch_vlo": 0.28, "launch_vhi": 0.45}
    variants = [("HELD", cq, cs, ApproachParams(**base, carry_qref=cq, carry_steps=cs))
                for cq in (0.0, 0.5, 1.0) for cs in (0, 2, 4, 6, 8, 10)]
    variants += [("PASSIVE_RELEASE", 0.0, cs, ApproachParams(**base, carry_steps=cs, carry_release=True))
                 for cs in (2, 4, 6, 8, 10)]
    runs = [{"mode": m, "carry_qref": cq, "carry_steps": cs, **_run_approach(snap, ap)} for m, cq, cs, ap in variants]
    held = [r for r in runs if r["mode"] == "HELD" and r["safe"]]
    rel = [r for r in runs if r["mode"] == "PASSIVE_RELEASE" and r["safe"]]
    best_held = max(held, key=lambda r: (r["k6"], -r["dtz_end_mm"])) if held else None
    best_rel = max(rel, key=lambda r: (r["k6"], -r["dtz_end_mm"])) if rel else None
    if best_held and best_held["k6"]:
        verdict = "HELD_MOMENTUM_CARRY_DELIVERS_S1"                       # keep the R6 rest certificate
    elif best_rel and best_rel["k6"]:
        verdict = "RELEASED_COAST_DELIVERS_S1"                           # need launch/release guard ≠ final settle cert
    else:
        verdict = "HANDOFF_TRANSPORT_MODE_NOT_YET_IDENTIFIED"            # active carry primitive still missing
    out = {"contract": "COIN_R9_R10C25_HANDOFF_AUDIT", "cradle": "s1", "approach_base": base,
           "best_held": best_held, "best_passive_release": best_rel, "verdict": verdict, "n_variants": len(runs),
           "runs": runs, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c25_handoff_audit.json", "w"), indent=1, default=float)
    bh = f"{best_held['dtz_end_mm']}mm K6 {best_held['k6']} (qref {best_held['carry_qref']} steps {best_held['carry_steps']})" if best_held else "none-safe"
    br = f"{best_rel['dtz_end_mm']}mm K6 {best_rel['k6']} (steps {best_rel['carry_steps']})" if best_rel else "none-safe"
    print(f"   best HELD {bh}\n   best PASSIVE_RELEASE {br}\n== R10-C2.5 ==\n  {verdict} | wall {out['wall_s']}s\nR9_C25_DONE",
          flush=True)
    return out


def c2_mechanism() -> dict:
    """R10-C2 — does the APPROACH momentum-build mode materially exceed the 0.154 forward-velocity ceiling (safely), exit via
    a causal guard, hand off to the frozen brake, and (staged) reach s1 K6? Dev s1 only; blind/validation states untouched."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    bank = json.load(open(BANK))
    snap = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}["s1"].snap
    grid = [ApproachParams(qdot_approach=q, launch_vlo=lo, launch_vhi=lo + 0.20)
            for q in (1.6, 2.0, 2.4, 2.8) for lo in (0.15, 0.25, 0.35)]
    runs = [_run_approach(snap, ap) for ap in grid]
    scaffold = _run_approach(snap, ApproachParams(), enabled=False)          # disabled ⇒ the frozen scaffold
    best = max(runs, key=lambda r: (r["k6"], -r["dtz_end_mm"]))
    max_v = max(runs, key=lambda r: r["peak_vpar"] if r["safe"] else -1e9)
    c2a = bool(any(r["peak_vpar"] > 0.154 and r["safe"] for r in runs))
    c2b = bool(any(r["exit_reason"] in ("LAUNCH", "REACHABILITY") and r["safe"] for r in runs))
    c2c = bool(any(r["exit_step"] is not None and r["safe"] for r in runs))
    c2d = bool(best["k6"])
    verdict = ("APPROACH_MOMENTUM_MODE_DELIVERS_S1" if c2d else
               "APPROACH_BUILDS_MOMENTUM_BUT_NO_S1_K6" if c2a else "APPROACH_PRIMITIVE_DOES_NOT_BUILD_MOMENTUM")
    out = {"contract": "COIN_R9_R10C2_APPROACH_MECHANISM", "cradle": "s1", "scaffold_ceiling_peak_vpar": 0.154,
           "scaffold_disabled_repro": scaffold, "max_peak_vpar_run": max_v, "best_delivery_run": best,
           "C2A_peak_vpar_exceeds_ceiling_safely": c2a, "C2B_exits_via_causal_guard": c2b,
           "C2C_hands_off_to_brake": c2c, "C2D_reaches_s1_k6": c2d, "verdict": verdict, "n_configs": len(runs),
           "runs": runs, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c2_approach_mechanism.json", "w"), indent=1, default=float)
    print(f"   scaffold(disabled) peak_vpar {scaffold['peak_vpar']} dtz {scaffold['dtz_end_mm']}mm | "
          f"MAX approach peak_vpar {max_v['peak_vpar']} (safe {max_v['safe']}) | best {best['dtz_end_mm']}mm K6 {best['k6']} "
          f"exit {best['exit_reason']}@{best['exit_step']}\n"
          f"   C2-A {c2a} | C2-B {c2b} | C2-C {c2c} | C2-D {c2d}\n== R10-C2 ==\n  {verdict} | wall {out['wall_s']}s\n"
          f"R9_C2_DONE", flush=True)
    return out


def c27_guided_coast_audit() -> dict:
    """R10-C2.7 — post-coast feasibility: is a GUIDED_COAST (light contact = low squeeze + tiny forward effort, so the coin
    coasts but keeps observability/correction authority) between full-grip (over-dissipates, 48mm) and full-release (no
    authority, 33mm) the missing mode? Sweeps guided-coast (squeeze × effort × duration) then the frozen settle on s1; the
    contrast baselines are the C2.5 held/released numbers. Dev s1 only; settle/cert/physics frozen."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    snap = {ps.tag: ps for ps in build_panel(_load_harness(), json.load(open(BANK)))}["s1"].snap
    base = {"qdot_approach": 2.4, "launch_vlo": 0.28, "launch_vhi": 0.45}
    variants = [("GUIDED", sq, cq, cs, ApproachParams(**base, carry_steps=cs, carry_qref=cq, carry_squeeze=sq))
                for sq in (0.02, 0.04, 0.06, 0.10) for cq in (0.0, 0.2) for cs in (4, 8, 12, 16)]
    variants += [("FULL_GRIP", 0.14, 1.0, cs, ApproachParams(**base, carry_steps=cs, carry_qref=1.0, carry_squeeze=0.14))
                 for cs in (4, 8, 12)]                                       # C2.5 held baseline
    variants += [("RELEASED", 0.0, 0.0, cs, ApproachParams(**base, carry_steps=cs, carry_release=True))
                 for cs in (6, 10)]                                          # C2.5 released baseline
    runs = [{"mode": m, "carry_squeeze": sq, "carry_qref": cq, "carry_steps": cs, **_run_approach(snap, ap)}
            for m, sq, cq, cs, ap in variants]
    guided = [r for r in runs if r["mode"] == "GUIDED" and r["safe"]]
    best_g = min(guided, key=lambda r: r["dtz_end_mm"]) if guided else None
    best_all = min([r for r in runs if r["safe"]], key=lambda r: r["dtz_end_mm"])
    if best_g and best_g["k6"]:
        verdict = "POST_COAST_GUIDED_COAST_MODE_LOAD_BEARING"
    elif best_g and best_g["dtz_end_mm"] < 33.0 - 1.0:                        # meaningfully closer than the released-coast 33mm
        verdict = "GUIDED_COAST_PROMISING_NEEDS_SETTLE_TUNING"
    else:
        verdict = "GUIDED_COAST_INSUFFICIENT_REACQUIRE_NEEDED"
    out = {"contract": "COIN_R9_R10C27_GUIDED_COAST", "cradle": "s1", "baselines": {"held_48mm": 48.4, "released_33mm": 33.0},
           "best_guided": best_g, "best_overall_safe": best_all, "verdict": verdict, "n_variants": len(runs), "runs": runs,
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/r10c27_guided_coast.json", "w"), indent=1, default=float)
    bg = (f"{best_g['dtz_end_mm']}mm K6 {best_g['k6']} (sqz {best_g['carry_squeeze']} qref {best_g['carry_qref']} steps "
          f"{best_g['carry_steps']})" if best_g else "none-safe")
    print(f"   best GUIDED {bg}\n   best overall-safe {best_all['mode']} {best_all['dtz_end_mm']}mm K6 {best_all['k6']}\n"
          f"== R10-C2.7 ==\n  {verdict} | wall {out['wall_s']}s\nR9_C27_DONE", flush=True)
    return out


def main(argv: "list[str]") -> None:
    if "--stage2" in argv:
        stage2_update_zero_identity()
    elif "--c27" in argv:
        c27_guided_coast_audit()
    elif "--c28" in argv:
        c28_reacquire_audit()
    elif "--diag" in argv:
        stage3_ceiling_diag()
    elif "--reach" in argv:
        reach_audit()
    elif "--m0" in argv:
        m0_base_coverage()
    elif "--c0" in argv:
        c0_trace_audit()
    elif "--c1" in argv:
        c1_projection()
    elif "--c2" in argv:
        c2_mechanism()
    elif "--c25" in argv:
        c25_handoff_audit()
    elif "--c26" in argv:
        c26_coast_guard_audit()
    elif "--stage3" in argv:
        cur = next((a.split("=")[1] for a in argv if a.startswith("--curriculum=")), "A")
        stage3_train(curriculum=cur, smoke="--smoke" in argv)
    else:
        print("usage: coin_r9_causal_rl.py --stage2 | --stage3 [--curriculum=A|B|C] [--smoke]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
