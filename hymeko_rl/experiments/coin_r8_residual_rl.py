"""R8 residual-RL harness (corrected gate) — S2 update-zero identity, S3 learnability audit, S4 matched SAC/TD3, S5 held-out.

One harness, mode flags (§6.5 #13). Reuses the frozen R8 tip-referenced scaffold (`tip_transport`) + the bounded residual
adapter (`residual_adapter`) + the frozen `velocity_rollout` physics + the R6 release certificate. Every integrity
constraint of `reports/2026-07-27-coin-r8-corrected-rl-gate-contract.md` is kept hard (no teleport / hidden force / teacher
fallback / oracle injection; exact Bellman provenance = the actor emission only; held-out s4/s7 excluded from all
training/tuning/selection; oracle a feasibility witness only). S2 runs on the development cradles.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, _coin_xy, delivery_success
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    RESIDUAL_ROLES, ConstantResidualActor, ResidualBounds, ResidualTipAdapter, SequenceResidualActor, ZeroActor)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipReferencedController, TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

OUT = "reports/2026-07-27-coin-r8-residual-rl"
REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"
DEV_TAGS = ("s1", "s3")                                 # development cradles (held-out s4/s7 excluded from all of S2-S4)
_METRIC_KEYS = ("dtz_start", "dtz_end", "forward", "cross", "peak_qdot", "peak_coin_speed", "terminal_coin_speed",
                "k6_max_dwell", "contact_lost_steps", "lost_before_release", "release_step", "gap_closed")


def _rich_trace(snap: Any, controller: Any, cfg: Any = DELIVERY_CFG) -> "tuple[dict, list]":
    """Roll a controller and capture a per-step physical trace (coin pose/velocity, joint velocity, contact fₙ, dtz) via a
    frame_hook — the full-trace comparison surface for the update-zero identity."""
    rows: list[dict[str, Any]] = []

    def hook(rl: Any, t: int) -> None:
        d = rl.inner.data
        from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
        con = primary_fingertip_contacts(rl)
        _u, dtz = rl.inner.direction_to_zone()
        rows.append({"t": int(t), "coin": np.asarray(_coin_xy(rl), np.float64).copy(),
                     "coin_vel": np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2].copy(),
                     "qvel": np.asarray(d.qvel[:4], np.float64).copy(), "dtz": float(dtz), "speed": float(_coin_speed(rl)),
                     "fn_l": float(con["left"]["fn"]) if con["left"] else 0.0,
                     "fn_r": float(con["right"]["fn"]) if con["right"] else 0.0})

    m = velocity_rollout(snap, controller, cfg, frame_hook=hook)
    return m, rows


def _max_diff(a: "list[dict]", b: "list[dict]") -> "dict[str, float]":
    """Max absolute difference of the per-step physical arrays between two rich traces (must be equal length)."""
    keys = ("coin", "coin_vel", "qvel")
    scal = ("dtz", "speed", "fn_l", "fn_r")
    out = {k: 0.0 for k in (*keys, *scal)}
    for ra, rb in zip(a, b):
        for k in keys:
            out[k] = max(out[k], float(np.max(np.abs(ra[k] - rb[k]))))
        for k in scal:
            out[k] = max(out[k], abs(ra[k] - rb[k]))
    return out


def _s2_state(snap: Any, params: Any, bounds: Any, tol: float) -> dict:
    """Compare the frozen scaffold vs the zero-residual adapter on ONE cradle over the full physical trace + provenance."""
    m_base, tr_base = _rich_trace(snap, TipReferencedController(snap, params, DELIVERY_CFG))
    adapter = ResidualTipAdapter(snap, ZeroActor(), params, bounds, DELIVERY_CFG)
    m_adpt, tr_adpt = _rich_trace(snap, adapter)
    diffs = _max_diff(tr_base, tr_adpt)
    metric_diff = {k: abs(float(m_base[k]) - float(m_adpt[k])) for k in _METRIC_KEYS}
    coin_equal = bool(np.array_equal(np.asarray(m_base["coin_trace"]), np.asarray(m_adpt["coin_trace"])))
    prov_ok = all(all(abs(x) < tol for x in p["residual"]) and abs(p["corrected_qref"] - p["base_qref"]) < tol
                  and abs(p["corrected_sqz"] - p["base_sqz"]) < tol and abs(p["kv"] - params.k_v) < tol
                  and all(abs(x) < tol for x in p["bellman_action"]) and not p["clip_flags"]["a"]
                  for p in adapter.provenance)
    trace_ok = coin_equal and max(diffs.values()) < tol and max(metric_diff.values()) < tol   # release_step ∈ metrics
    return {"split": "development", "trace_identity": trace_ok, "provenance_zero_effect": prov_ok,
            "coin_trace_bit_equal": coin_equal, "max_step_diff": {k: round(v, 12) for k, v in diffs.items()},
            "max_metric_diff": {k: round(v, 12) for k, v in metric_diff.items()}, "n_steps": len(tr_base),
            "delivery_scaffold": bool(delivery_success(m_base, DELIVERY_CFG)),
            "release_step_base": m_base["release_step"], "release_step_adapter": m_adpt["release_step"],
            "_max": max(max(diffs.values()), max(metric_diff.values()))}


def s2_update_zero_identity(smoke: bool = False) -> dict:
    """S2 — the zero-residual adapter reproduces the frozen scaffold over the FULL trace (dev cradles). # Postconditions:
    writes s2_update_zero_identity.json; verdict UPDATE_ZERO_RESIDUAL_IDENTITY_{PASS,FAILS}."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    params, bounds, tol = TipTransportParams(), ResidualBounds(), 1e-9
    per: dict[str, Any] = {}
    for tag in DEV_TAGS:
        st = _s2_state(panel[tag].snap, params, bounds, tol)
        mx = st.pop("_max")
        per[tag] = st
        print(f"   {tag}: trace_identity={st['trace_identity']} prov_zero={st['provenance_zero_effect']} "
              f"coin_equal={st['coin_trace_bit_equal']} max_diff={mx:.2e}", flush=True)
    passed = all(v["trace_identity"] and v["provenance_zero_effect"] for v in per.values())
    verdict = "UPDATE_ZERO_RESIDUAL_IDENTITY_PASS" if passed else "UPDATE_ZERO_RESIDUAL_IDENTITY_FAILS"
    out = {"contract": "COIN_R8_S2_UPDATE_ZERO_IDENTITY", "date": "2026-07-27", "tolerance": tol,
           "residual_roles": list(RESIDUAL_ROLES),
           "residual_bounds": {"d_fwd_vel": bounds.d_fwd_vel, "d_squeeze": bounds.d_squeeze, "d_stop_gain": bounds.d_stop_gain,
                               "kv_lo": bounds.kv_lo, "kv_hi": bounds.kv_hi},
           "bellman_action": "actor emission a in [-1,1]^3 ONLY; base/corrected/clipped/torque = provenance",
           "per_state": per, "passed": passed, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/s2_update_zero_identity.json", "w"), indent=1, default=float)
    print(f"\n== S2 ==\n  {verdict} | dev {sum(v['trace_identity'] and v['provenance_zero_effect'] for v in per.values())}/"
          f"{len(per)} | wall {out['wall_s']}s\nR8_S2_DONE", flush=True)
    return out


# ── S3 — development-only learnability audit ─────────────────────────────────────────────────────────────────────────
def _safe_metrics(m: dict) -> bool:
    """Motion-contract safety (unchanged limits): coin speed ≤ 1.5, joints ≤ 3.0, finite."""
    return bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0 and np.isfinite(m["dtz_end"]))


def option_return(m: dict) -> float:
    """K6-INDEPENDENT option-consequence return: closer to the zone + settled = higher; a safety breach is a hard floor.
    Uses ONLY physical quantities (dtz, terminal speed) — never the frozen K6 certificate flag/dwell."""
    if not _safe_metrics(m):
        return -10.0
    return -float(m["dtz_end"]) - 0.05 * float(m["terminal_coin_speed"])


def _eval_actor(snap: Any, actor: Any, params: Any, bounds: Any) -> dict:
    """Roll the residual adapter with an actor; return the safety-gated K6-independent return + key metrics + the executed
    residual magnitude by role."""
    adapter = ResidualTipAdapter(snap, actor, params, bounds, DELIVERY_CFG)
    m = velocity_rollout(snap, adapter, DELIVERY_CFG)
    res = np.asarray([p["residual"] for p in adapter.provenance], np.float64) if adapter.provenance else np.zeros((1, 3))
    vpars = [r["v_par"] for r in adapter.trace]
    return {"ret": option_return(m), "dtz_end": float(m["dtz_end"]), "min_dtz": float(min(r["dtz_mm"] for r in adapter.trace) / 1000.0),
            "safe": _safe_metrics(m), "k6": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
            "peak_coin": float(m["peak_coin_speed"]), "peak_q": float(m["peak_qdot"]), "reversal": bool(min(vpars) < -0.05),
            "residual_mag": [round(float(x), 5) for x in np.mean(np.abs(res), axis=0)]}


def _seq(kind: str, a: np.ndarray, horizon: int, rng: Any) -> np.ndarray:
    """Build a per-step residual sequence: constant, piecewise (few segments), or temporally-coherent random walk."""
    if kind == "constant":
        return np.tile(a, (horizon, 1))
    if kind == "piecewise":
        seg = np.array_split(np.arange(horizon), 4)
        return np.concatenate([np.tile(np.clip(a + 0.5 * rng.standard_normal(3), -1, 1), (len(s), 1)) for s in seg])
    walk = np.clip(np.cumsum(0.25 * rng.standard_normal((horizon, 3)), axis=0) + a, -1, 1)   # coherent
    return walk


def s3_sensitivity(dev: list, params: Any, bounds: Any) -> dict:
    """S3.1 — per-role ±ε constant perturbations vs update-zero on the dev cradles; classify each residual role."""
    base = {ps.tag: _eval_actor(ps.snap, ZeroActor(), params, bounds) for ps in dev}
    out: dict[str, Any] = {}
    for d, role in enumerate(RESIDUAL_ROLES):
        eff, unsafe = [], False
        for eps in (-0.6, 0.6):
            a = np.zeros(3)
            a[d] = eps
            for ps in dev:
                r = _eval_actor(ps.snap, ConstantResidualActor(a), params, bounds)
                eff.append(abs(r["dtz_end"] - base[ps.tag]["dtz_end"]))
                unsafe = unsafe or (not r["safe"])
        m_eff = float(np.mean(eff))
        cls = "UNSAFE" if unsafe else ("EFFECTIVE" if m_eff > 0.02 else ("WEAK" if m_eff > 0.003 else "INERT"))
        out[role] = {"mean_abs_dtz_change_m": round(m_eff, 5), "unsafe": bool(unsafe), "class": cls}
    return out


def _eval_candidate(dev: list, a: np.ndarray, kind: str, seq: np.ndarray, base: dict, params: Any,
                    bounds: Any) -> "tuple[list, list]":
    """Evaluate one candidate (a, kind, seq) across the dev cradles → (dataset rows, (safe, positive, Δreturn) tuples)."""
    rows, kres = [], []
    for ps in dev:
        actor = ConstantResidualActor(a) if kind == "constant" else SequenceResidualActor(seq)
        r = _eval_actor(ps.snap, actor, params, bounds)
        dret = r["ret"] - base[ps.tag]
        rows.append({"tag": ps.tag, "kind": kind, "a": [round(float(x), 4) for x in a], "residual_mag": r["residual_mag"],
                     "ret": round(r["ret"], 5), "d_return": round(dret, 5), "safe": r["safe"], "k6": r["k6"],
                     "dtz_end": round(r["dtz_end"], 4)})
        kres.append((r["safe"], dret > 1e-4, dret))
    return rows, kres


def _safe_positive_rate(flags: list) -> float:
    """Fraction of (safe AND positive) over `(safe, positive, *rest)` tuples; 0.0 if empty. # Post: in [0,1]."""
    return round(float(np.mean([a and b for a, b, *_ in flags])), 4) if flags else 0.0


def _improvement_stats(dvals: list) -> "tuple[float, float]":
    """(median, max) of the positive-improvement values (metres); (0.0, 0.0) if empty."""
    if not dvals:
        return 0.0, 0.0
    return round(float(np.median(dvals)), 5), round(float(np.max(dvals)), 5)


def _candidate_summary(rows: list, per_kind: dict) -> dict:
    """Safe / positive / safe-positive rates + improvement distribution + coherence effect over the candidate rows."""
    all_t = [(x["safe"], x["d_return"] > 1e-4) for x in rows]
    dvals = [x["d_return"] for x in rows if x["safe"] and x["d_return"] > 1e-4]
    med, mx = _improvement_stats(dvals)
    by_kind = {k: _safe_positive_rate(v) for k, v in per_kind.items()}
    return {"n_candidates_evaluated": len(rows), "safe_rate": round(float(np.mean([s for s, _ in all_t])), 4),
            "positive_rate": round(float(np.mean([p for _, p in all_t])), 4),
            "safe_positive_rate": round(float(np.mean([s and p for s, p in all_t])), 4),
            "improvement_median_m": med, "improvement_max_m": mx,
            "best_dtz_end_m": round(float(min(x["dtz_end"] for x in rows)), 4),
            "safe_positive_by_kind": by_kind,
            "temporal_coherence_helps": bool(by_kind["coherent"] > by_kind["constant"] + 0.02)}


def s3_candidates(dev: list, params: Any, bounds: Any, seed: int = 7001) -> "tuple[dict, list]":
    """S3.2 — sample bounded residual candidates (constant / piecewise / coherent), find SAFE POSITIVE ones vs update-zero.
    Returns (summary, dataset rows for S3.3). No teacher actions are used as labels."""
    rng = np.random.default_rng(seed)
    base = {ps.tag: _eval_actor(ps.snap, ZeroActor(), params, bounds)["ret"] for ps in dev}
    rows: list[dict[str, Any]] = []
    per_kind: dict[str, list] = {"constant": [], "piecewise": [], "coherent": []}
    for _ in range(40):
        a = np.clip(rng.standard_normal(3) * 0.6, -1, 1)
        for kind in ("constant", "piecewise", "coherent"):
            rws, kres = _eval_candidate(dev, a, kind, _seq(kind, a, DELIVERY_CFG.horizon, rng), base, params, bounds)
            rows += rws
            per_kind[kind] += kres
    return _candidate_summary(rows, per_kind), rows


def _rank_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """Cross-validated ridge DIFFERENCE-predictor rankability: Spearman, pairwise accuracy, top-10% enrichment."""
    n = len(y)
    if n < 8:
        return {"spearman": 0.0, "pairwise_acc": 0.5, "top10_enrichment": 1.0, "n": n}
    rng = np.random.default_rng(11)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.linalg.lstsq(xb[tr].T @ xb[tr] + 1e-3 * np.eye(xb.shape[1]), xb[tr].T @ y[tr], rcond=None)[0]
    pred = xb[te] @ w
    def _rank(v: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(v)).astype(float)
    rp, ry = _rank(pred), _rank(y[te])
    spear = float(np.corrcoef(rp, ry)[0, 1]) if len(te) > 2 else 0.0
    base_pos = float(np.mean(y[te] > 0))
    k = max(1, len(te) // 10)
    top = np.argsort(pred)[::-1][:k]
    enrich = round(float(np.mean(y[te][top] > 0) / (base_pos + 1e-9)), 3) if base_pos > 0 else 0.0
    pairs = rng.integers(0, len(te), (200, 2))
    acc = float(np.mean([(pred[i] > pred[j]) == (y[te][i] > y[te][j]) for i, j in pairs if y[te][i] != y[te][j]]))
    return {"spearman": round(spear, 3), "pairwise_acc": round(acc, 3), "top10_enrichment": enrich, "n": int(n)}


def _s3_rank(rows: list) -> "tuple[dict, int]":
    """S3.3 rankability over the SAFE candidate rows. Features = the SIGNED candidate direction `a` (sign matters: +δ_fwd is
    more push, −δ_fwd is less — opposite effects; magnitude alone cannot rank them) + the kind + the state (cradle tag). The
    predictor is the cross-validated ridge difference-predictor Δreturn(state, residual)."""
    safe = [r for r in rows if r["safe"]]
    kinds = {"constant": 0.0, "piecewise": 1.0, "coherent": 2.0}
    tags = {t: float(i) for i, t in enumerate(sorted({r["tag"] for r in safe}))}
    feats = (np.array([[*r["a"], kinds[r["kind"]], tags[r["tag"]]] for r in safe], np.float64)
             if safe else np.zeros((0, 5)))
    return _rank_metrics(feats, np.array([r["d_return"] for r in safe], np.float64)), len(safe)


def _s3_verdict(sens: dict, cand: dict, rank: dict) -> "tuple[str, dict]":
    """Combine S3.1/S3.2/S3.3 into the frozen verdict + the three flags."""
    effective = [r for r in sens if sens[r]["class"] in ("EFFECTIVE", "WEAK")]
    has_effect = bool(effective) and not any(sens[r]["class"] == "UNSAFE" for r in sens)
    has_positive = cand["safe_positive_rate"] > 0.02
    rankable = rank["spearman"] > 0.2 or rank["top10_enrichment"] > 1.3 or rank["pairwise_acc"] > 0.6
    if has_effect and has_positive and rankable:
        v = "RESIDUAL_LEARNABILITY_SIGNAL_PASS"
    elif has_effect and not has_positive:
        v = "RESIDUAL_INTERFACE_HAS_EFFECT_BUT_NO_POSITIVE_CANDIDATES"
    elif has_positive and not rankable:
        v = "POSITIVE_CANDIDATES_EXIST_BUT_ARE_NOT_RANKABLE"
    else:
        v = "CURRENT_RESIDUAL_PARAMETERISATION_INSUFFICIENT"
    return v, {"has_effect": bool(has_effect), "has_safe_positive": bool(has_positive), "rankable": bool(rankable),
               "effective_roles": effective}


def _s3_write(sens: dict, cand: dict, rows: list, rank: dict, n_safe: int, flags: dict) -> None:
    json.dump({"contract": "COIN_R8_S3_1_SENSITIVITY", "per_role": sens, "effective_roles": flags["effective_roles"],
               "inert_or_unsafe": [r for r in sens if sens[r]["class"] in ("INERT", "UNSAFE")]},
              open(f"{OUT}/s3_sensitivity.json", "w"), indent=1, default=float)
    json.dump({"contract": "COIN_R8_S3_2_CANDIDATE_SEARCH", **cand, "n_rows": len(rows)},
              open(f"{OUT}/s3_candidate_search.json", "w"), indent=1, default=float)
    json.dump({"contract": "COIN_R8_S3_3_RANKABILITY", **rank, "n_safe_rows": n_safe,
               "predictor": "cross-validated ridge difference-predictor delta_return(state,residual)"},
              open(f"{OUT}/s3_rankability.json", "w"), indent=1, default=float)


def s3_learnability(smoke: bool = False) -> dict:
    """S3 — dev-only learnability audit (sensitivity + candidate existence + rankability). Writes the artifacts + a verdict;
    NO actor update. # Postconditions: one of RESIDUAL_LEARNABILITY_SIGNAL_PASS / …_HAS_EFFECT_BUT_NO_POSITIVE_CANDIDATES /
    POSITIVE_CANDIDATES_EXIST_BUT_ARE_NOT_RANKABLE / CURRENT_RESIDUAL_PARAMETERISATION_INSUFFICIENT."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    dev = [panel[t] for t in DEV_TAGS]
    params, bounds = TipTransportParams(), ResidualBounds()
    sens = s3_sensitivity(dev, params, bounds)
    print("  S3.1 sensitivity:", {r: sens[r]["class"] for r in sens}, flush=True)
    cand, rows = s3_candidates(dev, params, bounds)
    print(f"  S3.2 safe={cand['safe_rate']} pos={cand['positive_rate']} safe_pos={cand['safe_positive_rate']} "
          f"impr_med={cand['improvement_median_m']} coherence_helps={cand['temporal_coherence_helps']}", flush=True)
    rank, n_safe = _s3_rank(rows)
    print(f"  S3.3 rankability: spearman={rank['spearman']} pairwise={rank['pairwise_acc']} top10={rank['top10_enrichment']}", flush=True)
    verdict, flags = _s3_verdict(sens, cand, rank)
    _s3_write(sens, cand, rows, rank, n_safe, flags)
    passed = verdict == "RESIDUAL_LEARNABILITY_SIGNAL_PASS"
    out = {"contract": "COIN_R8_S3_LEARNABILITY", "sensitivity": {r: sens[r]["class"] for r in sens}, "candidate": cand,
           "rankability": rank, **{k: flags[k] for k in ("has_effect", "has_safe_positive", "rankable")},
           "verdict": verdict, "rl_authorised": bool(passed), "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/s3_learnability.json", "w"), indent=1, default=float)
    print(f"\n== S3 ==\n  VERDICT: {verdict} | rl_authorised={passed} | wall {out['wall_s']}s\nR8_S3_DONE", flush=True)
    return out


def _load_harness() -> Any:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    return load_harness()


def main(argv: "list[str]") -> None:
    if "--s2" in argv:
        s2_update_zero_identity(smoke="--smoke" in argv)
    elif "--s3" in argv:
        s3_learnability(smoke="--smoke" in argv)
    else:
        print("usage: coin_r8_residual_rl.py --s2 | --s3 | (S4/S5 modes added as gates pass)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
