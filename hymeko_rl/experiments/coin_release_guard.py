"""Release-guard program (G0–G3) — learn the state-conditioned coast-landing map.

The R10 K6-decomposition localised the s1 barrier to *coast-landing precision*: the coin moves only by free coast, the coast
stops short of the 20 mm zone, and the teacher delivers only by a well-timed release. C2.6 found a wide global coast spread
(a_coast ∈ [0.51, 1.26] m/s²) that defeated a point-estimate release guard. This module tests the decisive question the user
posed: is that spread **irreducible**, or **conditionally narrowable** given the pre-release state + short history?

The LAUNCH/RELEASE guard built here is a DIFFERENT object from the R6 final settle certificate
(`theta_option/release_certificate.py`): the guard decides *when to let go so the free coast lands in the corridor*; the R6
cert decides *whether the coin ended at rest in the zone*. This module never touches the held-out s4/s7 panel — dev s1/s3 only.

G0 (this file): conditional-observability audit — global vs state-conditioned coast-landing spread + determinism check.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.experiments.coin_r9_causal_rl import (
    BANK, DELIVERY_CFG, OUT, TipTransportParams, _load_harness, _phys_hook, _teacher_theta, build_panel)

# G0 sweep grid: the pre-release-state distribution is spanned by APPROACH momentum × release timing (dev cradles s1, s3 only).
_G0_CRADLES = ("s1", "s3")
_G0_QDOT = (1.6, 2.0, 2.4)
_G0_RELEASE_STEPS = tuple(range(2, 18))
# The guard's observable inputs (NO future value, NO K6 outcome, NO teacher) — the columns fed to the conditional predictor.
_FEATURE_KEYS = ("d_remaining_mm", "v_par", "dv_par", "spin", "v_lat", "squeeze", "imbalance", "impulse_hist", "t_since_contact")


def _release_rollout(snap: Any, qdot: float, k: int) -> "tuple[list, dict, dict | None]":
    """Force APPROACH(qdot) → passive release at step k on a cradle; return the full per-step trace, metrics, release-state."""
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams, HybridApproachController
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    ap = ApproachParams(qdot_approach=qdot, launch_vlo=5.0, launch_vhi=6.0, impulse_budget=10.0, max_steps=45,
                        acquire_squeeze=0.2, release_at_step=k)
    ctrl = HybridApproachController(snap, TipTransportParams(), ap, DELIVERY_CFG)
    trace: list = []
    m = velocity_rollout(snap, ctrl, DELIVERY_CFG, frame_hook=_phys_hook(trace))
    return trace, m, getattr(ctrl, "release_state", None)


def _pre_release_features(trace: list, k: int) -> dict:
    """Observable pre-release state + short history at the release step — the guard's inputs (no future/outcome leak)."""
    at = next((r for r in trace if r["t"] == k), trace[min(k, len(trace) - 1)])
    prev = next((r for r in trace if r["t"] == k - 2), at)                          # short v_par derivative
    hist = [r for r in trace if r["t"] <= k]
    t_contact = next((r["t"] for r in trace if r["contact"] == 1), None)
    return {"d_remaining_mm": at["dtz_mm"], "v_par": at["v_par"], "dv_par": at["v_par"] - prev["v_par"],
            "spin": at["spin"], "v_lat": at["v_lat"], "squeeze": at["fn_l"] + at["fn_r"],
            "imbalance": at["fn_l"] - at["fn_r"], "impulse_hist": sum(r["fn_l"] + r["fn_r"] for r in hist),
            "t_since_contact": float(k - t_contact) if t_contact is not None else -1.0}


def _outcome(trace: list, m: dict) -> dict:
    """The post-release coast outcome the guard must predict: where the coin comes to rest and whether it reached the zone."""
    dtz = [r["dtz_mm"] for r in trace]
    return {"terminal_dtz_mm": round(float(m["dtz_end"]) * 1000, 1), "min_dtz_mm": round(min(dtz), 1),
            "terminal_speed": round(trace[-1]["speed"], 4), "k6z_reached_zone": bool(min(dtz) <= CENTER_TOL * 1000)}


def _collect(panel: dict) -> list:
    """Build the (features → coast-landing) dataset over dev cradles × APPROACH momentum × release timing (valid coasts only)."""
    rows = []
    for tag in _G0_CRADLES:
        if tag not in panel:
            continue
        snap = panel[tag].snap
        for q in _G0_QDOT:
            for k in _G0_RELEASE_STEPS:
                trace, m, rel = _release_rollout(snap, q, k)
                safe = bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0)
                if not (rel and safe and len(trace) > k + 3 and rel.get("v_par", 0.0) > 0.03):   # a genuine post-release coast
                    continue
                rows.append({"cradle": tag, "qdot": q, "release_step": k, **_pre_release_features(trace, k), **_outcome(trace, m)})
    return rows


def _standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def _knn_loo_residual(feats: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Leave-one-out k-NN regression residual on standardised features: median |predicted − actual| landing (mm)."""
    xs, res = _standardise(feats), []
    for i in range(len(y)):
        d = np.linalg.norm(xs - xs[i], axis=1)
        d[i] = np.inf
        res.append(abs(float(y[np.argsort(d)[:k]].mean()) - float(y[i])))
    return float(np.median(res))


def _knn_loo_accuracy(feats: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Leave-one-out k-NN classification accuracy for 'coast reaches the zone' — can the guard SELECT delivering releases?"""
    xs, correct = _standardise(feats), 0
    for i in range(len(labels)):
        d = np.linalg.norm(xs - xs[i], axis=1)
        d[i] = np.inf
        correct += int(int(labels[np.argsort(d)[:k]].mean() >= 0.5) == int(labels[i]))
    return correct / len(labels)


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _determinism_check(panel: dict) -> bool:
    """Same snapshot + same release ⇒ bit-identical landing? If yes, the coast spread is state-dependence, NOT aleatoric noise."""
    tag = next((t for t in _G0_CRADLES if t in panel), None)
    if tag is None:
        return False
    a, _, _ = _release_rollout(panel[tag].snap, 2.0, 8)
    b, _, _ = _release_rollout(panel[tag].snap, 2.0, 8)
    return bool(a and b and a[-1]["dtz_mm"] == b[-1]["dtz_mm"] and a[-1]["speed"] == b[-1]["speed"])


def _g0_verdict(n_zone: int, ratio: float, sel_acc: float) -> str:
    if n_zone == 0:
        return "NO_RELEASE_LANDS_IN_ZONE"                       # no free coast reaches the zone → guard alone cannot deliver
    if ratio < 0.4 and sel_acc >= 0.8:
        return "COAST_LANDING_CONDITIONALLY_OBSERVABLE"         # state-conditioned spread ≪ global ⇒ learnable guard → G1/G2
    return "COAST_LANDING_NOT_STATE_REDUCIBLE"                  # observable state aliases landings ⇒ need more sensing/probing


def _feature_corr(rows: list, y: np.ndarray) -> dict:
    """Which observable dominates the landing — |Pearson corr| of each feature with the terminal coast dtz."""
    out = {}
    for key in _FEATURE_KEYS:
        col = np.array([r[key] for r in rows], np.float64)
        out[key] = round(abs(float(np.corrcoef(col, y)[0, 1])) if col.std() > 1e-9 else 0.0, 3)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _plot_g0(rows: list, y: np.ndarray, corr: dict, path: str) -> "str | None":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    d = np.array([r["d_remaining_mm"] for r in rows])
    v = np.array([r["v_par"] for r in rows])
    sc = ax0.scatter(d, y, c=v, cmap="viridis", s=28)
    ax0.axhspan(0, 20, color="green", alpha=0.15, label="target zone ≤20mm")
    ax0.set(xlabel="d_remaining at release (mm)", ylabel="terminal coast landing (mm)", title="G0 coast-landing map")
    fig.colorbar(sc, ax=ax0, label="v_par at release")
    ax0.legend(loc="upper left", fontsize=8)
    ax1.barh(list(corr.keys())[::-1], list(corr.values())[::-1], color="steelblue")
    ax1.set(xlabel="|corr| with terminal landing", title="pre-release feature ↔ landing")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def g0_conditional_observability() -> dict:
    """G0 — is the coast-landing spread irreducible or state-conditionally narrowable? Measures global vs k-NN-conditional
    landing spread + a determinism check + release-selection accuracy, on dev s1/s3 (held-out s4/s7 untouched)."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), json.load(open(BANK)))}
    deterministic = _determinism_check(panel)
    rows = _collect(panel)
    if len(rows) < 8:
        out = {"contract": "COIN_RELEASE_GUARD_G0", "verdict": "COAST_DATA_INSUFFICIENT", "n": len(rows),
               "wall_s": round(time.time() - t0, 1)}
        json.dump(out, open(f"{OUT}/release_guard_g0.json", "w"), indent=1, default=float)
        print(f"== G0 ==\n  COAST_DATA_INSUFFICIENT ({len(rows)} rows)\nRELEASE_G0_DONE", flush=True)
        return out
    feats = np.array([[r[k] for k in _FEATURE_KEYS] for r in rows], np.float64)
    y = np.array([r["terminal_dtz_mm"] for r in rows], np.float64)
    labels = np.array([int(r["k6z_reached_zone"]) for r in rows], np.float64)
    global_spread, cond_spread = _iqr(y), _knn_loo_residual(feats, y)
    ratio = round(cond_spread / max(global_spread, 1e-9), 3)
    sel_acc = round(_knn_loo_accuracy(feats, labels), 3) if 0 < labels.sum() < len(labels) else 1.0
    n_zone = int(labels.sum())
    corr = _feature_corr(rows, y)
    verdict = _g0_verdict(n_zone, ratio, sel_acc)
    plot = _plot_g0(rows, y, corr, f"{OUT}/release_guard_g0.png")
    out = {"contract": "COIN_RELEASE_GUARD_G0", "cradles": list(_G0_CRADLES), "n_coast_branches": len(rows),
           "determinism_bit_identical": deterministic, "global_landing_iqr_mm": round(global_spread, 1),
           "conditional_landing_residual_mm": round(cond_spread, 1), "conditional_over_global": ratio,
           "n_reach_zone": n_zone, "zone_selection_loo_accuracy": sel_acc, "feature_landing_corr": corr,
           "landing_min_mm": round(float(y.min()), 1), "landing_max_mm": round(float(y.max()), 1),
           "verdict": verdict, "plot": plot, "wall_s": round(time.time() - t0, 1),
           "reading": ("deterministic sim ⇒ the coast spread is a FUNCTION of the release state; if the state-conditioned "
                       "residual ≪ the global spread and delivering releases are selectable, a learned release guard is "
                       "legitimate (G1/G2). If not, the observable state aliases landings and new sensing/probing is needed.")}
    json.dump(out, open(f"{OUT}/release_guard_g0.json", "w"), indent=1, default=float)
    top = list(corr.items())[:3]
    print(f"   determinism bit-identical {deterministic} | {len(rows)} coast branches on {list(_G0_CRADLES)}\n"
          f"   global landing IQR {global_spread:.1f}mm → conditional residual {cond_spread:.1f}mm (ratio {ratio})\n"
          f"   reach-zone {n_zone}/{len(rows)} | zone-selection LOO acc {sel_acc} | landing∈[{y.min():.1f},{y.max():.1f}]mm\n"
          f"   top features↔landing: {top}\n== G0 (conditional observability) ==\n  {verdict} | wall {out['wall_s']}s\n"
          f"RELEASE_G0_DONE", flush=True)
    return out


# G-VF search space — the 6-param position-varying KINETIC transport profile (v_hi, v_lo, taper_dtz, sqz_transport, decay_onset, release_hi)
_VF_LO = np.array([0.18, 0.10, 0.028, 0.05, 0.026, 0.020])
_VF_HI = np.array([0.34, 0.16, 0.044, 0.16, 0.040, 0.030])


def _vf_ap(v: np.ndarray) -> Any:
    """Profile candidate → ApproachParams (taper/decay breakpoints clamped strictly above the release corridor)."""
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams
    rel = float(v[5])
    return ApproachParams(qdot_approach=2.6, acquire_squeeze=0.12, impulse_budget=12.0, max_steps=30, fn_min_safe=0.02,
                          kinetic_transport=True, kinetic_profile=True, kinetic_vcap=1.9, kinetic_entry_v=0.10,
                          kinetic_max_steps=55, kinetic_release_vmin=0.05, kinetic_release_hi=rel,
                          k_v_hi=float(v[0]), k_v_lo=float(v[1]), k_taper_dtz=max(float(v[2]), rel + 0.005),
                          k_sqz_transport=float(v[3]), k_decay_onset=max(float(v[4]), rel + 0.003))


def _kinetic_rollout(snap: Any, ap: Any) -> "tuple[list, dict, Any]":
    """Roll the KINETIC controller with a PHASE-aware hook capturing per-step (dtz, v_par, realized fn, contact, is-KINETIC)."""
    from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC, HybridApproachController
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    ctrl = HybridApproachController(snap, TipTransportParams(), ap, DELIVERY_CFG)
    traj: list = []

    def hook(rl: Any, t: int) -> None:
        u, d = rl.inner.direction_to_zone()
        uu = np.asarray(u, np.float64)[:2]
        vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        con = primary_fingertip_contacts(rl)
        fnl = float(con["left"]["fn"]) if con["left"] else 0.0
        fnr = float(con["right"]["fn"]) if con["right"] else 0.0
        traj.append({"dtz_mm": float(d) * 1000, "v_par": float(np.dot(vel, uu)), "fn_min": min(fnl, fnr),
                     "contact": int(fnl > 0 or fnr > 0), "kinetic": ctrl.phase == KINETIC})
    m = velocity_rollout(snap, ctrl, DELIVERY_CFG, frame_hook=hook)
    return traj, m, ctrl


def _kinetic_metrics(kine: list, rel_hi_mm: float) -> dict:
    """REALIZED-contact quality over the KINETIC steps: sign reversals, light-contact fraction, clamp (fn too high) + early
    contact-loss (fn=0 before the corridor) penalties — scored on measured fn, not the squeeze command."""
    signs = [1 if r["v_par"] > 0.02 else (-1 if r["v_par"] < -0.02 else 0) for r in kine]
    return {"reversals": sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0),
            "contact_frac": sum(1 for r in kine if 0.03 < r["fn_min"] < 1.8) / max(len(kine), 1),
            "clamp_pen": sum(max(0.0, r["fn_min"] - 1.8) for r in kine),
            "loss_pen": sum(1 for r in kine if r["contact"] == 0 and r["dtz_mm"] > rel_hi_mm + 3.0)}


def _vf_score(traj: list, m: dict, ctrl: Any, rel_hi_mm: float) -> "tuple[float, dict]":
    """Lexicographic-ish scalar over the REALIZED trajectory: safety ≻ K6 ≻ reached-moving-release ≻ light-contact-maintained ≻
    no-sign-reversal ≻ closeness (clamp / early-contact-loss penalised via _kinetic_metrics)."""
    if not (m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0):
        return -1e9, {"safe": False}
    rel = getattr(ctrl, "release_state", None)
    k6 = bool(delivery_success(m, DELIVERY_CFG))
    q = _kinetic_metrics([r for r in traj if r["kinetic"]], rel_hi_mm)
    min_dtz = min((r["dtz_mm"] for r in traj), default=999.0)
    rel_vpar = float((rel or {}).get("v_par", 0.0))
    score = (1e6 * int(k6) + 1e4 * int(rel is not None) + 2e3 * q["contact_frac"] - 300 * q["reversals"]
             - 40 * q["clamp_pen"] - 60 * q["loss_pen"] - min_dtz + 50 * max(0.0, rel_vpar))
    return score, {"safe": True, "k6": k6, "released": rel is not None, "rel_dtz": (rel or {}).get("dtz_mm"),
                   "rel_vpar": round(rel_vpar, 3), "min_dtz_mm": round(min_dtz, 1),
                   "contact_frac": round(q["contact_frac"], 2), "reversals": q["reversals"],
                   "clamp_pen": round(q["clamp_pen"], 2), "loss_pen": q["loss_pen"],
                   "dtz_end_mm": round(float(m["dtz_end"]) * 1000, 1)}


def _vf_generation(snap: Any, cand: np.ndarray, best: dict) -> "tuple[list, dict]":
    scored = []
    for c in cand:
        traj, m, ctrl = _kinetic_rollout(snap, _vf_ap(c))
        s, info = _vf_score(traj, m, ctrl, float(c[5]) * 1000)
        scored.append(s)
        if s > best["score"]:
            best = {"score": round(s, 1), "params": [round(float(x), 4) for x in c], **info}
    return scored, best


def gvf_search(pop: int = 36, n_iter: int = 6, elite: int = 8) -> dict:
    """G-VF — bounded CEM over the 6-param KINETIC transport PROFILE: a feasibility/oracle search for a close+moving release
    whose ACTUAL passive coast delivers s1 K6. Dev s1 only; physics/K6/coast/settle/blind/s4-s7 all frozen (only the in-grip
    transport profile is searched). Teacher = positive control from the same cradle."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    snap = panel["s1"].snap
    rng = np.random.default_rng(77)
    mu, sig, best, hist = (_VF_LO + _VF_HI) / 2, (_VF_HI - _VF_LO) / 4, {"score": -1e9}, []
    for it in range(n_iter):
        cand = (_VF_LO[None] + (_VF_HI - _VF_LO)[None] * rng.random((pop, 6)) if it == 0
                else np.clip(mu[None] + sig[None] * rng.standard_normal((pop, 6)), _VF_LO, _VF_HI))
        scored, best = _vf_generation(snap, cand, best)
        idx = np.argsort(scored)[-elite:]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 1e-4
        hist.append({"iter": it, "k6": best.get("k6"), "min_dtz": best.get("min_dtz_mm"), "contact": best.get("contact_frac")})
        print(f"[gvf {it + 1}/{n_iter}] k6 {best.get('k6')} released {best.get('released')} rel_dtz {best.get('rel_dtz')} "
              f"min_dtz {best.get('min_dtz_mm')}mm contact {best.get('contact_frac')} rev {best.get('reversals')}", flush=True)
    teacher_k6 = bool(delivery_success(rollout_primitive(snap, tuple(_teacher_theta(bank, "s1")), DELIVERY_CFG), DELIVERY_CFG))
    verdict = ("VELOCITY_FLOOR_RELEASE_MANIFOLD_PASS" if best.get("k6") else
               "KINETIC_REACHES_MOVING_RELEASE_NO_K6" if best.get("released") else
               "KINETIC_CANNOT_HOLD_CONTACT_TO_CORRIDOR")
    out = {"contract": "COIN_RELEASE_GUARD_GVF_SEARCH", "cradle": "s1", "n_evals": pop * n_iter, "best": best,
           "teacher_positive_control_k6": teacher_k6, "history": hist, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/release_guard_gvf.json", "w"), indent=1, default=float)
    print(f"   teacher positive control K6 {teacher_k6} | {pop * n_iter} evals\n"
          f"   best: k6 {best.get('k6')} released {best.get('released')} rel_dtz {best.get('rel_dtz')}mm rel_vpar "
          f"{best.get('rel_vpar')} min_dtz {best.get('min_dtz_mm')}mm contact_frac {best.get('contact_frac')}\n"
          f"== G-VF (KINETIC transport search) ==\n  {verdict} | wall {out['wall_s']}s\nRELEASE_GVF_DONE", flush=True)
    return out


def main(argv: "list[str]") -> None:
    if "--g0" in argv:
        g0_conditional_observability()
    elif "--gvfsearch" in argv:
        gvf_search()
    else:
        print("usage: coin_release_guard.py --g0 | --gvfsearch", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
