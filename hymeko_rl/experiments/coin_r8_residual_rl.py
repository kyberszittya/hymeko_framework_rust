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
import torch

from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, _coin_xy, delivery_success
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.residual_adapter import (
    RESIDUAL_ROLES, ConstantResidualActor, ResidualBounds, ResidualTipAdapter, SequenceResidualActor, ZeroActor)
from hymeko_rl.coin_delivery.theta_option.residual_option_env import (
    ACT_DIM, OBS_DIM, ResidualOptionEnv, ResidualRLConfig, distill_zero_residual, residual_init_obs)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipReferencedController, TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.option_rl import bootstrap_ci
from hymeko_rl.option_rl.agents import make_actor, train_semi_mdp

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


def _rank_once(x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, rng: Any) -> "tuple[float, float, float]":
    """One train/test half-split of the ridge difference-predictor → (spearman, pairwise_acc, top10_enrichment)."""
    xb = np.hstack([x, np.ones((len(y), 1))])
    w = np.linalg.lstsq(xb[tr].T @ xb[tr] + 1e-3 * np.eye(xb.shape[1]), xb[tr].T @ y[tr], rcond=None)[0]
    pred, yt = xb[te] @ w, y[te]
    rp = np.argsort(np.argsort(pred)).astype(float)
    ry = np.argsort(np.argsort(yt)).astype(float)
    spear = float(np.corrcoef(rp, ry)[0, 1]) if len(te) > 2 else 0.0
    base_pos = float(np.mean(yt > 0))
    top = np.argsort(pred)[::-1][: max(1, len(te) // 10)]
    enrich = float(np.mean(yt[top] > 0) / (base_pos + 1e-9)) if base_pos > 0 else 0.0
    pairs = rng.integers(0, len(te), (200, 2))
    hits = [(pred[i] > pred[j]) == (yt[i] > yt[j]) for i, j in pairs if yt[i] != yt[j]]
    return spear, (float(np.mean(hits)) if hits else 0.5), enrich


def _rank_metrics(x: np.ndarray, y: np.ndarray, n_splits: int = 25) -> dict:
    """ROBUST rankability of the ridge difference-predictor Δreturn(state, residual): median + IQR + threshold-hit
    fraction over `n_splits` random half-splits (§3 multi-seed discipline — a single split is a point estimate, not a
    verdict). # Post: `spearman`/`pairwise_acc`/`top10_enrichment` are MEDIANS; `frac_*` are the share of splits crossing
    each single-metric threshold."""
    n = len(y)
    if n < 8:
        return {"spearman": 0.0, "pairwise_acc": 0.5, "top10_enrichment": 1.0, "n": n, "n_splits": 0}
    rng = np.random.default_rng(11)
    sp, pa, en = [], [], []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        s, a, e = _rank_once(x, y, idx[: n // 2], idx[n // 2:], rng)
        sp.append(s)
        pa.append(a)
        en.append(e)
    sp_a, pa_a, en_a = np.array(sp), np.array(pa), np.array(en)
    def _iqr(v: np.ndarray) -> list:
        return [round(float(np.percentile(v, 25)), 3), round(float(np.percentile(v, 75)), 3)]
    return {"spearman": round(float(np.median(sp_a)), 3), "spearman_iqr": _iqr(sp_a),
            "pairwise_acc": round(float(np.median(pa_a)), 3), "pairwise_iqr": _iqr(pa_a),
            "top10_enrichment": round(float(np.median(en_a)), 3), "top10_iqr": _iqr(en_a),
            "frac_spear_gt_0p2": round(float(np.mean(sp_a > 0.2)), 3),
            "frac_top10_gt_1p3": round(float(np.mean(en_a > 1.3)), 3),
            "frac_pair_gt_0p6": round(float(np.mean(pa_a > 0.6)), 3), "n": int(n), "n_splits": int(n_splits)}


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
    # ROBUST rankability: a MAJORITY of resampled splits must cross a single-metric threshold (a marginal median on one
    # split is not a verdict, §3). Falls back to the single-split keys when the robust `frac_*` are absent (n<8 case).
    rankable = (rank.get("frac_spear_gt_0p2", 0.0) > 0.5 or rank.get("frac_top10_gt_1p3", 0.0) > 0.5
                or rank.get("frac_pair_gt_0p6", 0.0) > 0.5)
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


# ── S4 — matched SAC/TD3 residual-option RL (DEVELOPMENT ONLY; authorised by S2 & S3) ────────────────────────────────
HELD_OUT_TAGS = ("s4", "s7")                            # excluded from S2–S4; seen ONCE, frozen, at S5


def _dev_cradles(panel: dict, params: Any, bounds: Any) -> list:
    """(tag, snap, init_obs) for the DEV cradles only — the residual option env's entire world."""
    return [(t, panel[t].snap, residual_init_obs(panel[t].snap, params, bounds)) for t in DEV_TAGS]


def _actor_residual(actor: Any, obs: np.ndarray) -> np.ndarray:
    """The actor's DETERMINISTIC mean residual for an initiation obs (the deploy-time action)."""
    with torch.no_grad():
        ot = torch.as_tensor(np.asarray(obs, np.float32)[None])
        a = actor.mean_action(ot)[0].numpy() if hasattr(actor, "mean_action") else actor(ot)[0].numpy()
    return np.asarray(a, np.float64)


def _eval_on_cradles(actor: Any, cradles: list, params: Any, bounds: Any) -> "tuple[float, list]":
    """Mean K6-independent option_return of the actor's MEAN residual over a cradle set + per-cradle detail. The dev
    selection metric (S4); reused UNCHANGED for the frozen held-out eval (S5)."""
    rets, per = [], []
    for tag, snap, obs in cradles:
        r = _eval_actor(snap, ConstantResidualActor(_actor_residual(actor, obs)), params, bounds)
        rets.append(r["ret"])
        per.append({"tag": tag, "ret": round(r["ret"], 4), "dtz_end_mm": round(r["dtz_end"] * 1000, 1),
                    "k6": bool(r["k6"]), "safe": bool(r["safe"]), "peak_coin": round(r["peak_coin"], 3)})
    return round(float(np.mean(rets)), 4), per


def _dev_eval_fn(cradles: list, params: Any, bounds: Any):
    """dev_eval_fn(actor) -> (score, aux) for train_semi_mdp; selection = mean dev option_return (higher is better)."""
    def _fn(actor: Any) -> "tuple[float, dict]":
        score, per = _eval_on_cradles(actor, cradles, params, bounds)
        return score, {"deliv": sum(p["k6"] for p in per), "safe": sum(p["safe"] for p in per)}
    return _fn


def _scaffold_return(cradles: list, params: Any, bounds: Any) -> float:
    """Mean K6-independent option_return of the ZERO residual (the frozen safe scaffold = the update-0 baseline)."""
    return round(float(np.mean([_eval_actor(s, ZeroActor(), params, bounds)["ret"] for _t, s, _o in cradles])), 4)


def _s4_reward_cert(cradles: list, params: Any, bounds: Any, base: float) -> dict:
    """Certify the training reward is NOT anti-aligned: the best of a small residual probe out-returns the update-zero
    scaffold on the dev cradles (a positive residual IS rewarded). The §3 pre-launch reward gate, at S4 launch."""
    best = base
    for a in (np.array([-0.6, 0.0, 0.0]), np.array([0.6, 0.0, 0.0]), np.array([-0.6, 0.0, 0.6]), np.array([0.0, 0.5, -0.4])):
        best = max(best, float(np.mean([_eval_actor(s, ConstantResidualActor(a), params, bounds)["ret"] for _t, s, _o in cradles])))
    return {"update0_return": round(base, 4), "best_probe_return": round(best, 4), "reward_certified": bool(best > base + 1e-4)}


def _s4_replay_audit(env: Any, bounds: Any) -> dict:
    """PROVE (a) DEV-ONLY world (env tags ⊆ DEV_TAGS, disjoint from held-out ⇒ nothing held-out can enter replay); and
    (b) BELLMAN = RESIDUAL — over a probe grid, the recorded Bellman action equals the clipped emission and the EXECUTED
    residual equals clip(a)·bounds; no torque/target is ever the action; a=0 executes a zero residual (the S2 identity)."""
    tags = set(env.tags)
    dev_only = bool(tags <= set(DEV_TAGS) and tags.isdisjoint(HELD_OUT_TAGS))
    vec = bounds.vec()
    probes, ok = [], True
    for a in (np.zeros(3), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]),
              np.array([0, 0, 1.0]), np.array([1.5, -1.5, 0.7]), np.array([0.3, -0.4, 0.9])):
        env.reset(0)
        _s2, _r, _d, info = env.step(a)
        ac = np.clip(a, -1.0, 1.0)
        ba_ok = bool(np.allclose(info["bellman_action"], ac.astype(np.float32), atol=1e-6))
        res_ok = bool(np.allclose(info["executed_residual"], (ac * vec).astype(np.float32), atol=1e-6))
        probes.append({"a": [round(float(x), 2) for x in a], "bellman==clip(a)": ba_ok, "residual==clip(a)*bounds": res_ok})
        ok = ok and ba_ok and res_ok
    return {"contract": "COIN_R8_S4_REPLAY_AUDIT", "dev_only_world": dev_only, "dev_tags": sorted(tags),
            "held_out_excluded": list(HELD_OUT_TAGS), "bellman_action_is_actor_residual": ok,
            "framework_note": "train_semi_mdp stores OptionTransition.action = the actor emission a; the env applies exactly "
                              "that a — executed torque / corrected targets are provenance, never presented as the action.",
            "probes": probes}


def _write_training_contract(cfg: Any, seeds: tuple, cert: dict, audit: dict, smoke: bool) -> None:
    hp = ("gamma", "lr", "batch", "warmup_options", "total_options", "updates_per_option", "eval_every", "alpha",
          "policy_delay", "expl_noise")
    json.dump({"contract": "COIN_R8_S4_TRAINING_CONTRACT", "date": "2026-07-27", "gate": "S4_MATCHED_SAC_TD3_DEV",
               "authorised_by": ["S2 UPDATE_ZERO_RESIDUAL_IDENTITY_PASS", "S3 RESIDUAL_LEARNABILITY_SIGNAL_PASS"],
               "bellman_action": "actor residual a in [-1,1]^3 (constant per option); everything else is provenance",
               "reward": "K6-INDEPENDENT option_return (−dtz_end − 0.05·terminal_coin_speed), safety-gated; NEVER the frozen K6 flag",
               "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "dev_cradles": list(DEV_TAGS), "held_out_excluded": list(HELD_OUT_TAGS),
               "algorithms": ["sac", "td3"], "seeds": list(seeds), "smoke": bool(smoke),
               "hyperparameters": {k: getattr(cfg, k) for k in hp}, "reward_certification": cert,
               "replay_dev_only": audit["dev_only_world"], "update0": "actor mean distilled to 0 = the S2 safe scaffold",
               "selection": "best_val on DEV option_return; held-out s4/s7 one-shot at S5 only",
               "integrity": "no teleport / hidden force / teacher fallback / oracle injection; torque/slew/collision/motion "
                            "contracts unchanged; release stays R6-certificate-gated"},
              open(f"{OUT}/training_contract.json", "w"), indent=1, default=float)


def _s4_run_seed(algo: str, cradles: list, cfg: Any, seed: int, params: Any, bounds: Any) -> dict:
    """One matched run: make actor, distil update-0 = scaffold, train_semi_mdp on the dev env, evaluate the dev-selected
    best_val. Returns the per-run record (with the best_val state_dict under `_ckpt`)."""
    actor = make_actor(algo, OBS_DIM, ACT_DIM)
    obs0 = np.stack([c[2] for c in cradles]).astype(np.float32)
    distill_loss = distill_zero_residual(actor, obs0, seed=seed)
    upd0 = _eval_on_cradles(actor, cradles, params, bounds)[0]
    env = ResidualOptionEnv(cradles, option_return, params=params, bounds=bounds, seed=seed)
    ckpts, hist = train_semi_mdp(algo, env, actor, _dev_eval_fn(cradles, params, bounds), cfg.to_semi_mdp(),
                                 obs_dim=OBS_DIM, act_dim=ACT_DIM, log=lambda s: print(s, flush=True), seed=seed)
    best = make_actor(algo, OBS_DIM, ACT_DIM)
    best.load_state_dict(ckpts["best_val"])
    best_score, best_per = _eval_on_cradles(best, cradles, params, bounds)
    return {"algo": algo, "seed": seed, "distill_loss": round(distill_loss, 6), "update0_dev_return": round(upd0, 4),
            "best_val_dev_return": best_score, "delta_over_update0": round(best_score - upd0, 4), "per_cradle": best_per,
            "eval_points": len(hist), "_ckpt": ckpts["best_val"]}


def _s4_summarise(runs: list) -> dict:
    """Median/IQR + bootstrap CI of best_val dev-return and Δ-over-update0 across seeds (RL carve-out: multi-seed median)."""
    dev = [r["best_val_dev_return"] for r in runs]
    dlt = [r["delta_over_update0"] for r in runs]
    return {"n_seeds": len(runs), "best_val_dev_return_median": round(float(np.median(dev)), 4),
            "best_val_dev_return_iqr": [round(float(np.percentile(dev, 25)), 4), round(float(np.percentile(dev, 75)), 4)],
            "delta_over_update0_median": round(float(np.median(dlt)), 4), "delta_ci": bootstrap_ci(dlt),
            "per_seed": [{k: r[k] for k in ("seed", "update0_dev_return", "best_val_dev_return", "delta_over_update0")} for r in runs]}


def _s4_compare(results: dict, base: float) -> dict:
    """Matched SAC-vs-TD3 comparison on DEV + the dev-selected champion (selection is on DEV return ONLY — held-out unseen)."""
    sac = [r["best_val_dev_return"] for r in results["sac"]]
    td3 = [r["best_val_dev_return"] for r in results["td3"]]
    sac_m, td3_m = float(np.median(sac)), float(np.median(td3))
    champ = max(((a, r) for a in ("sac", "td3") for r in results[a]), key=lambda x: x[1]["best_val_dev_return"])
    return {"contract": "COIN_R8_S4_MATCHED_COMPARISON", "update0_dev_return": round(base, 4),
            "sac_dev_median": round(sac_m, 4), "td3_dev_median": round(td3_m, 4),
            "sac_dev_ci": bootstrap_ci(sac), "td3_dev_ci": bootstrap_ci(td3),
            "sac_beats_update0": bool(sac_m > base + 1e-4), "td3_beats_update0": bool(td3_m > base + 1e-4),
            "dev_champion": {"algo": champ[0], "seed": champ[1]["seed"], "dev_return": champ[1]["best_val_dev_return"],
                             "ckpt": champ[1]["ckpt"]},
            "note": "champion selected on DEV return only; held-out s4/s7 remain unseen until the single frozen S5 eval."}


def s4_matched_rl(seeds: tuple = (0, 1, 2), smoke: bool = False) -> dict:
    """S4 — matched SAC/TD3 residual-option RL on the DEV cradles. Pre-training gates (dev-only replay, Bellman=residual,
    reward alignment) → matched multi-seed runs → SAC-vs-TD3 comparison + dev-selected champion. Held-out untouched."""
    t0 = time.time()
    os.makedirs(f"{OUT}/ckpts", exist_ok=True)
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    params, bounds = TipTransportParams(), ResidualBounds()
    cradles = _dev_cradles(panel, params, bounds)
    cfg = (ResidualRLConfig(total_options=60, warmup_options=15, eval_every=20, batch=16)   # smoke exercises the update path
           if smoke else ResidualRLConfig())
    if smoke:
        seeds = (0,)
    audit = _s4_replay_audit(ResidualOptionEnv(cradles, option_return, params=params, bounds=bounds, seed=0), bounds)
    base = _scaffold_return(cradles, params, bounds)
    cert = _s4_reward_cert(cradles, params, bounds, base)
    _write_training_contract(cfg, seeds, cert, audit, smoke)
    json.dump(audit, open(f"{OUT}/replay_audit.json", "w"), indent=1, default=float)
    print(f"  S4 pre-gates: dev_only={audit['dev_only_world']} bellman==residual={audit['bellman_action_is_actor_residual']} "
          f"reward_cert={cert['reward_certified']} (update0 {base})", flush=True)
    if not (audit["dev_only_world"] and audit["bellman_action_is_actor_residual"] and cert["reward_certified"]):
        out = {"contract": "COIN_R8_S4", "status": "HALTED_PRE_TRAINING_GATE", "audit": audit, "reward_cert": cert}
        json.dump(out, open(f"{OUT}/matched_comparison.json", "w"), indent=1, default=float)
        print("  S4 HALTED — a pre-training integrity gate failed; no training run started.\nR8_S4_DONE", flush=True)
        return out
    results: dict[str, list] = {}
    for algo in ("sac", "td3"):
        runs = []
        for seed in seeds:
            print(f"\n  ── {algo.upper()} seed {seed} ──", flush=True)
            r = _s4_run_seed(algo, cradles, cfg, seed, params, bounds)
            torch.save(r.pop("_ckpt"), f"{OUT}/ckpts/{algo}_seed{seed}_best_val.pt")
            r["ckpt"] = f"{OUT}/ckpts/{algo}_seed{seed}_best_val.pt"
            runs.append(r)
        results[algo] = runs
        json.dump({"contract": f"COIN_R8_S4_{algo.upper()}", "runs": runs, "summary": _s4_summarise(runs)},
                  open(f"{OUT}/{algo}_results.json", "w"), indent=1, default=float)
    comp = _s4_compare(results, base)
    comp["wall_s"] = round(time.time() - t0, 1)
    json.dump(comp, open(f"{OUT}/matched_comparison.json", "w"), indent=1, default=float)
    print(f"\n== S4 ==\n  update0 dev-return {base} | SAC {comp['sac_dev_median']} | TD3 {comp['td3_dev_median']} | "
          f"champion {comp['dev_champion']['algo']} seed {comp['dev_champion']['seed']} ({comp['dev_champion']['dev_return']}) "
          f"| wall {comp['wall_s']}s\nR8_S4_DONE", flush=True)
    return comp


def _load_harness() -> Any:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    return load_harness()


def main(argv: "list[str]") -> None:
    if "--s2" in argv:
        s2_update_zero_identity(smoke="--smoke" in argv)
    elif "--s3" in argv:
        s3_learnability(smoke="--smoke" in argv)
    elif "--s4" in argv:
        s4_matched_rl(smoke="--smoke" in argv)
    else:
        print("usage: coin_r8_residual_rl.py --s2 | --s3 | --s4 [--smoke] | (S5 added when S4 passes)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
