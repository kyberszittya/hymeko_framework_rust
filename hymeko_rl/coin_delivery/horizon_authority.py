"""H2 SESSION 2 — LIFTED-HORIZON AUTHORITY versus CRADLE COLLAPSE (measurement, NOT the H2 QP).

Session 1 proved ``ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT`` (frozen, committed): a
one-step joint-position-target perturbation Δq_target has NULL contact-velocity authority because the torque-rate
limiter's per-step slew budget (τ̇·dt) is entirely consumed unwinding the acquisition torque debt |raw_pd − prev_tau|,
so the ±ε branches receive bit-identical post-governor torque ⇒ Δv_rel = 0.

The rate limiter is an INTEGRATOR. Held for H>1 steps it catches up to the raw PD demand; once joint j's debt is spent
(~⌈debt_j/(τ̇·dt)⌉ steps) the ±ε command difference kp·ε survives the clip and authority appears. This module measures
the RACE between authority arrival and cradle collapse — the horizon hierarchy

    H_tau      first step the ±ε POST-GOVERNOR torque sequences diverge
    H_state    first step q/q̇ measurably diverge
    H_vrel     first step measurable contact-velocity authority (‖Δv_rel‖ > floor, both branches same contact mode)
    H_collapse first step the cradle/contact mode is lost (tip drop / straddle inversion / motion-contract breach)

and selects the control-interface route (per the frozen decision tree):
    H_vrel + margin < H_collapse, rank(B_v(H*))≥2, both ε, branch/straddle/cert retained  → ROUTE_C (lifted-horizon H2)
    H_vrel appears within margin of H_collapse                                            → ROUTE_A (lower-debt handoff)
    no H_vrel before H_collapse                                                           → ROUTE_B (slew-admissible Δτ)

Semantics: primary HELD_OFFSET (offset held on the servo target for all H steps); control ONE_STEP_PULSE (offset on
step 1 only). Frozen horizons H ∈ {1,2,4,8,12,16,24,32,40}. Reuses the Session-1 governed-stepping primitives EXACTLY
(pd_governed_torque + per-sub-step governor callback + step_ablation + prev_tau←a threading) so trace-equivalence to the
frozen result holds — the H=1 held rollout MUST reproduce B_v ≡ 0.

NO pinning in evaluated rollouts. NO ε enlargement. NO τ_rate change. V2/V4 physics & motion contracts read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import (
    BvConfig, CradleSnapshot, measure_contact_velocities)
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

HORIZONS = (1, 2, 4, 8, 12, 16, 24, 32, 40)


@dataclass(frozen=True)
class HorizonConfig:
    """Frozen lifted-horizon measurement parameters. Shared cradle/FD params come from an embedded ``BvConfig`` so the
    contact-mode floor, ε scales, and motion-contract cap are IDENTICAL to Session 1 (no divergent second contract)."""

    bv: BvConfig = field(default_factory=BvConfig)
    horizons: tuple = HORIZONS
    h_max: int = 40
    tau_tol: float = 1e-6            # N·m — first ±ε post-governor torque divergence (5-decimal provenance floor)
    state_tol: float = 1e-6         # rad, rad/s — first q/q̇ divergence
    vrel_tol: float = 1e-4          # m/s — first MEASURABLE contact-velocity authority (‖Δv_rel‖)
    rank_abs_tol: float = 1e-3      # absolute singular-value floor for numeric rank of B_v(H) (m/s per rad)
    rank_rel_tol: float = 1e-2      # singular value must also exceed rel_tol·σ_max
    margin: int = 3                 # control steps that must remain after authority onset (usability)
    repro_rel_gap: float = 0.35     # max per-column relative gap between the two ε scales (reproducibility)


# --------------------------------------------------------------------------------------------------------------------
# PURE helpers (unit-testable, no MuJoCo)
# --------------------------------------------------------------------------------------------------------------------
def first_crossing(series, threshold: float) -> "int | None":
    """1-based index of the first element of ``series`` strictly above ``threshold`` (None if never). Used for the
    H_tau / H_state / H_vrel onsets over a per-step scalar series. # Postconditions: result in [1, len] or None."""
    for i, x in enumerate(series):
        if float(x) > threshold:
            return i + 1
    return None


def numeric_rank(mat: np.ndarray, abs_tol: float, rel_tol: float) -> int:
    """Numeric rank of a matrix over its FINITE columns: singular values above max(abs_tol, rel_tol·σ_max). NaN columns
    (invalid at this horizon) are dropped (never zero-filled — a zeroed column would understate rank harmlessly but a
    NaN must not poison the SVD). # Postconditions: 0 ≤ rank ≤ min(shape of the finite sub-matrix)."""
    m = np.asarray(mat, np.float64)
    cols = m[:, np.all(np.isfinite(m), axis=0)]
    if cols.size == 0:
        return 0
    sv = np.linalg.svd(cols, compute_uv=False)
    thr = max(abs_tol, rel_tol * float(sv[0])) if sv.size else abs_tol
    return int(np.sum(sv > thr))


def cradle_alive(meas: dict, peak_qdot: float, cfg: HorizonConfig) -> tuple[bool, dict]:
    """The MULTI-STEP cradle-survival predicate (NOT the one-step FD linearisation gate): both tips present with the
    same fingertip–disk identity, both Fn ≥ floor, straddle retained (n_L·n_R < 0), exactly one tip–disk contact per
    side (unambiguous primary), motion contract held. Benign settling drift over many steps is DELIBERATELY not gated
    here — the xc_drift / normal-angle gates are for one-step FD validity only and would false-trip collapse.

    # Preconditions: ``meas`` from ``measure_contact_velocities``; peak_qdot = post-step max|q̇[:4]|.
    """
    lft, rgt = meas["per"]["left"], meas["per"]["right"]
    sig = {"present": bool(lft["present"] and rgt["present"]),
           "same_identity": bool(lft["same_identity"] and rgt["same_identity"]),
           "fn_ok": bool(lft["fn"] >= cfg.bv.fn_floor and rgt["fn"] >= cfg.bv.fn_floor),
           "unambiguous_primary": bool(lft["count"] == 1 and rgt["count"] == 1),
           "straddle_retained": bool(meas["straddle_live"] < 0.0),
           "contract_ok": bool(peak_qdot <= cfg.bv.joint_vel_hard)}
    return bool(all(sig.values())), sig


# --------------------------------------------------------------------------------------------------------------------
# The governed HORIZON rollout — the Session-1 stepping, threaded H steps, holding (or pulsing) the servo offset
# --------------------------------------------------------------------------------------------------------------------
def rollout_offset(snap: CradleSnapshot, dq_target, horizon: int, mode: str, cfg: HorizonConfig) -> dict:
    """Advance a fresh branch ``horizon`` control steps under servo target q_hold + Δq, threading prev_tau ← applied
    torque exactly as the shared stack does. ``mode='held'`` holds Δq every step; ``mode='pulse'`` applies Δq on step 1
    only (control). Per step records the pre-governor rate-limited step ctrl ``a``, the TRUE per-sub-step post-governor
    ctrl trace (what MuJoCo integrates), post-step q/q̇, the contact-relative velocity vrel4 in the frozen frame, and the
    cradle-alive signature.

    # Preconditions: dq_target length-4; mode ∈ {'held','pulse'}; horizon ≥ 1. # Postconditions: dict of per-step arrays
    of length ``horizon``; the stepping is byte-for-byte the Session-1 path (govern_torque cb + step_ablation).
    """
    assert mode in ("held", "pulse"), mode
    rl = snap.branch()
    d = rl.inner.data
    dq = np.asarray(dq_target, np.float64)
    prev_tau = snap.prev_tau.copy()
    gov_traces: list[list] = []

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
        gov_traces[-1].append(dt.ctrl[:4].copy())

    a_seq, q_seq, qd_seq, vrel_seq, alive_seq, sig_seq, fn_seq, straddle_seq, qdot_seq = ([] for _ in range(9))
    mujoco.set_mjcb_control(_gcb)
    try:
        for h in range(1, int(horizon) + 1):
            q0, qd0 = d.qpos[:4].copy(), d.qvel[:4].copy()
            offset = dq if (mode == "held" or h == 1) else np.zeros(4)
            q_des = snap.q_hold + offset
            a = pd_governed_torque(q0, qd0, q_des, snap.stack, prev_tau, snap.lo, snap.hi)
            gov_traces.append([])
            step_ablation(rl, np.asarray(a, np.float32), "A")
            prev_tau = np.asarray(a, np.float64)
            peak_qdot = float(np.max(np.abs(d.qvel[:4])))
            meas = measure_contact_velocities(rl, snap.frames)
            alive, sig = cradle_alive(meas, peak_qdot, cfg)
            a_seq.append(np.asarray(a, np.float64))
            q_seq.append(d.qpos[:4].copy())
            qd_seq.append(d.qvel[:4].copy())
            vrel_seq.append(meas["vrel4"].copy())
            alive_seq.append(alive)
            sig_seq.append(sig)
            fn_seq.append((meas["fn_left"], meas["fn_right"]))
            straddle_seq.append(meas["straddle_live"])
            qdot_seq.append(peak_qdot)
    finally:
        mujoco.set_mjcb_control(None)
    return {"a": np.asarray(a_seq), "q": np.asarray(q_seq), "qd": np.asarray(qd_seq),
            "vrel4": np.asarray(vrel_seq), "alive": alive_seq, "sig": sig_seq,
            "gov_traces": [np.asarray(g) for g in gov_traces], "fn": fn_seq, "straddle": straddle_seq,
            "peak_qdot": np.asarray(qdot_seq), "mode": mode}


def collapse_step(alive_seq) -> int:
    """First 1-based step at which the cradle is not alive, else len+1 (survives the whole rollout)."""
    for i, a in enumerate(alive_seq):
        if not a:
            return i + 1
    return len(alive_seq) + 1


# --------------------------------------------------------------------------------------------------------------------
# Per-column authority — ±ε held (or pulse) rollouts, the four H_* onsets, per-step Jacobian column
# --------------------------------------------------------------------------------------------------------------------
def _torque_divergence(gp: list[np.ndarray], gm: list[np.ndarray]) -> np.ndarray:
    """Per-step max|post-governor ctrl(+ε) − post-governor ctrl(−ε)| over the sub-step traces (length = #steps)."""
    out = []
    for tp, tm in zip(gp, gm):
        n = min(len(tp), len(tm))
        out.append(float(np.max(np.abs(tp[:n] - tm[:n]))) if n else 0.0)
    return np.asarray(out)


def column_authority(snap: CradleSnapshot, j: int, eps: float, mode: str, cfg: HorizonConfig) -> dict:
    """±ε rollouts on joint j to H_MAX; the per-step Jacobian column B_v[:,j](h) = (vrel(+ε) − vrel(−ε))/(2ε), the raw
    authority ‖Δv_rel‖(h), both-branch aliveness, and the H_tau/H_state/H_vrel onsets for this column."""
    ej = np.zeros(4)
    ej[j] = eps
    rp = rollout_offset(snap, ej, cfg.h_max, mode, cfg)
    rm = rollout_offset(snap, -ej, cfg.h_max, mode, cfg)
    tau_div = _torque_divergence(rp["gov_traces"], rm["gov_traces"])
    state_div = np.maximum(np.max(np.abs(rp["q"] - rm["q"]), axis=1), np.max(np.abs(rp["qd"] - rm["qd"]), axis=1))
    dvrel = rp["vrel4"] - rm["vrel4"]
    dvrel_norm = np.linalg.norm(dvrel, axis=1)
    both_alive = np.array([bool(a and b) for a, b in zip(rp["alive"], rm["alive"])])
    bv_col = dvrel / (2.0 * eps)                                  # (H_MAX, 4) per-step column
    # H_vrel for this column: first step with measurable authority AND both branches on the same contact mode
    h_vrel = next((i + 1 for i in range(cfg.h_max)
                   if dvrel_norm[i] > cfg.vrel_tol and both_alive[i]), None)
    return {"j": j, "eps": float(eps), "mode": mode,
            "H_tau": first_crossing(tau_div, cfg.tau_tol), "H_state": first_crossing(state_div, cfg.state_tol),
            "H_vrel": h_vrel, "collapse_plus": collapse_step(rp["alive"]), "collapse_minus": collapse_step(rm["alive"]),
            "bv_col": bv_col, "dvrel_norm": dvrel_norm, "both_alive": both_alive,
            "tau_div": tau_div, "state_div": state_div}


def _bv_at(cols: list[dict], h: int) -> np.ndarray:
    """Assemble B_v(h) (4×4) from per-column authorities: column j is finite only if both ±ε branches are alive at
    step h; otherwise NaN (invalid at this horizon)."""
    Bv = np.full((4, 4), np.nan)
    for c in cols:
        if c["both_alive"][h - 1]:
            Bv[:, c["j"]] = c["bv_col"][h - 1]
    return Bv


def _repro_gap(cols_a: list[dict], cols_b: list[dict], h: int) -> "float | None":
    """Max per-column relative gap between the two ε scales' B_v(h) columns (both branches alive at h in both)."""
    gaps = []
    for ca, cb in zip(cols_a, cols_b):
        if ca["both_alive"][h - 1] and cb["both_alive"][h - 1]:
            va, vb = ca["bv_col"][h - 1], cb["bv_col"][h - 1]
            denom = np.linalg.norm(va) + np.linalg.norm(vb)
            if denom > 1e-9:
                gaps.append(float(np.linalg.norm(va - vb) / denom))
    return round(max(gaps), 4) if gaps else None


# --------------------------------------------------------------------------------------------------------------------
# State-level analysis + route decision
# --------------------------------------------------------------------------------------------------------------------
def _rank_series(cols: list[dict], cfg: HorizonConfig) -> np.ndarray:
    """Per-step numeric rank of B_v(h) assembled from the columns (h = 1..H_MAX)."""
    return np.asarray([numeric_rank(_bv_at(cols, h), cfg.rank_abs_tol, cfg.rank_rel_tol) for h in range(1, cfg.h_max + 1)])


def _horizon_table(cols: list[dict], cfg: HorizonConfig) -> list[dict]:
    """B_v rank + valid-column count at each FROZEN horizon (for the artifact/plot)."""
    out = []
    for h in cfg.horizons:
        Bv = _bv_at(cols, h)
        out.append({"H": h, "valid_cols": int(np.sum(np.all(np.isfinite(Bv), axis=0))),
                    "rank": numeric_rank(Bv, cfg.rank_abs_tol, cfg.rank_rel_tol),
                    "Bv": [[(round(float(x), 5) if np.isfinite(x) else None) for x in row] for row in Bv]})
    return out


def analyse_state_horizon(snap: CradleSnapshot, cfg: HorizonConfig) -> dict:
    """Full lifted-horizon analysis for one certified cradle: baseline collapse, per-ε held column authorities, the
    rank-2 authority onset, reproducibility across ε, the pulse control, and the per-state route candidate."""
    base = rollout_offset(snap, np.zeros(4), cfg.h_max, "held", cfg)
    H_collapse_baseline = collapse_step(base["alive"])
    held = {e: [column_authority(snap, j, e, "held", cfg) for j in range(4)] for e in cfg.bv.eps_scales}
    eps_list = list(cfg.bv.eps_scales)
    rank_series = {e: _rank_series(held[e], cfg) for e in eps_list}
    # aggregate H_* onsets (first over any column, first ε where it appears)
    def _agg(key):
        vals = [c[key] for e in eps_list for c in held[e] if c[key] is not None]
        return int(min(vals)) if vals else None
    H_tau, H_state, H_vrel = _agg("H_tau"), _agg("H_state"), _agg("H_vrel")
    # rank-2 authority onset must hold on BOTH ε scales, sustain `margin` steps, and reproduce across ε
    onset = _rank2_onset(rank_series, eps_list, cfg)
    repro_gap = _repro_gap(held[eps_list[0]], held[eps_list[1]], onset) if (onset and len(eps_list) > 1) else None
    reproducible = bool(repro_gap is not None and repro_gap <= cfg.repro_rel_gap)
    sustains = _sustains(rank_series, eps_list, onset, cfg) if onset else False
    usable = bool(onset is not None and onset < H_collapse_baseline and sustains and reproducible)
    rank2_before_collapse = _rank2_before_collapse(rank_series, eps_list, H_collapse_baseline)
    pulse = [column_authority(snap, j, eps_list[0], "pulse", cfg) for j in range(4)]
    pulse_rank = _rank_series(pulse, cfg)
    route, reason = _state_route(usable, H_vrel, H_collapse_baseline, rank2_before_collapse, onset, reproducible, sustains)
    return {
        "operating_point": snap.operating_point, "released": snap.released,
        "handoff_prev_tau": [round(float(x), 4) for x in snap.prev_tau], "arm_saturated": snap.arm_saturated,
        "fn0": [round(x, 4) for x in snap.fn0], "straddle0": round(snap.straddle0, 4),
        "H_tau": H_tau, "H_state": H_state, "H_vrel": H_vrel, "H_collapse_baseline": H_collapse_baseline,
        "rank2_onset": onset, "rank2_onset_reproducible": reproducible, "repro_rel_gap": repro_gap,
        "rank2_sustains_margin": sustains, "rank2_before_collapse": rank2_before_collapse,
        "usable_authority": usable, "route": route, "route_reason": reason,
        "baseline_peak_qdot": [round(float(x), 4) for x in base["peak_qdot"]],
        "baseline_fn": [[round(float(x), 4) for x in fn] for fn in base["fn"]],
        "rank_series": {f"eps_{e}": [int(x) for x in rank_series[e]] for e in eps_list},
        "pulse_rank_series": [int(x) for x in pulse_rank],
        "horizon_table": {f"eps_{e}": _horizon_table(held[e], cfg) for e in eps_list},
        "columns": {f"eps_{e}": [_col_summary(c) for c in held[e]] for e in eps_list},
        "pulse_columns": [_col_summary(c) for c in pulse]}


def _col_summary(c: dict) -> dict:
    """Machine-readable per-column authority summary (onsets, collapse, peak authority) — arrays dropped for the JSON."""
    return {"j": c["j"], "eps": c["eps"], "mode": c["mode"], "H_tau": c["H_tau"], "H_state": c["H_state"],
            "H_vrel": c["H_vrel"], "collapse_plus": c["collapse_plus"], "collapse_minus": c["collapse_minus"],
            "peak_dvrel_norm": round(float(np.max(c["dvrel_norm"])), 6),
            "dvrel_norm_at_horizons": [round(float(c["dvrel_norm"][h - 1]), 6) for h in HORIZONS]}


def _rank2_onset(rank_series: dict, eps_list: list, cfg: HorizonConfig) -> "int | None":
    """First step (1-based) at which B_v(h) has rank ≥ 2 on ALL ε scales."""
    for h in range(cfg.h_max):
        if all(rank_series[e][h] >= 2 for e in eps_list):
            return h + 1
    return None


def _sustains(rank_series: dict, eps_list: list, onset: int, cfg: HorizonConfig) -> bool:
    """Rank ≥ 2 (all ε) persists for `margin` steps from onset (a few control steps to actually act)."""
    end = min(onset + cfg.margin, cfg.h_max)
    return all(all(rank_series[e][h] >= 2 for e in eps_list) for h in range(onset - 1, end))


def _rank2_before_collapse(rank_series: dict, eps_list: list, h_collapse: int) -> bool:
    """Some step before baseline collapse reaches rank ≥ 2 on all ε (authority exists, timing aside)."""
    end = min(h_collapse - 1, len(next(iter(rank_series.values()))))
    return any(all(rank_series[e][h] >= 2 for e in eps_list) for h in range(end))


def _state_route(usable, H_vrel, H_collapse, rank2_before_collapse, onset, reproducible, sustains) -> tuple[str, str]:
    """The frozen route decision tree, per state."""
    if usable:
        return ("ROUTE_C_LIFTED_HORIZON", f"rank-2 authority at H={onset}, reproducible, sustains margin, before "
                f"collapse H={H_collapse}")
    if H_vrel is None or H_vrel >= H_collapse:
        return ("ROUTE_B_SLEW_ADMISSIBLE_DTAU", f"no measurable authority before collapse "
                f"(H_vrel={H_vrel}, H_collapse={H_collapse})")
    if rank2_before_collapse:                     # rank-2 authority exists but unusable (timing / margin / reproducibility)
        return ("ROUTE_A_LOWER_DEBT_HANDOFF", f"rank-2 authority appears (onset={onset}) but unusable before collapse "
                f"H={H_collapse} (reproducible={reproducible}, sustains={sustains}) — reduce acquisition torque debt")
    return ("ROUTE_B_SLEW_ADMISSIBLE_DTAU", f"authority appears (H_vrel={H_vrel}) but never reaches rank≥2 before "
            f"collapse H={H_collapse} — Δq_target interface is rank-deficient")


def decide_campaign_route(state_results: dict, dev_ids: list, heldout_ids: list) -> dict:
    """Aggregate the per-state routes into ONE campaign decision. ROUTE_C is claimed only if every development state is
    usable AND at least one held-out state is usable (the frozen gate). Otherwise the most-conservative required route
    wins (A before C, B before A): a single blocked state downgrades the campaign."""
    dev = [state_results[i]["route"] for i in dev_ids if i in state_results]
    held = [state_results[i]["route"] for i in heldout_ids if i in state_results]
    dev_usable = [state_results[i]["usable_authority"] for i in dev_ids if i in state_results]
    held_usable = [state_results[i]["usable_authority"] for i in heldout_ids if i in state_results]
    all_routes = dev + held
    if dev and all(dev_usable) and any(held_usable):
        route = "ROUTE_C_LIFTED_HORIZON"
        reason = "all development states usable AND ≥1 held-out usable — lifted-horizon H2 authorised"
    elif "ROUTE_B_SLEW_ADMISSIBLE_DTAU" in all_routes:
        route = "ROUTE_B_SLEW_ADMISSIBLE_DTAU"
        reason = "≥1 state has no rank-2 authority before collapse — Δq_target interface blocked, need slew-admissible Δτ"
    elif "ROUTE_A_LOWER_DEBT_HANDOFF" in all_routes:
        route = "ROUTE_A_LOWER_DEBT_HANDOFF"
        reason = "rank-2 authority exists but unusable (timing) — lower the acquisition torque debt first"
    else:
        route = "AUTHORITY_RECOVERY_NOT_ESTABLISHED"
        reason = "no state analysed / inconclusive"
    return {"route": route, "reason": reason, "dev_routes": dev, "heldout_routes": held,
            "dev_usable": dev_usable, "heldout_usable": held_usable}
