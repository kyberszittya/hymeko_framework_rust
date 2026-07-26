"""Route B — SLEW-ADMISSIBLE TORQUE-INCREMENT interface B_τ (one-step authority; NOT the H2 QP).

Session 1 proved the one-step POSITION-target Jacobian B_v ≡ 0: the torque-rate limiter absorbs a Δq_target
perturbation because the per-step slew budget (τ̇·dt = 0.3 N·m) is already saturated unwinding the acquisition debt.
Route A (mobile conditioning) could not make the fragile held-out cradles usable. Route B changes the DECISION
VARIABLE itself, as both the frozen decision tree and the Session-1 report recommend:

    decision variable  Δτ_cmd ∈ R^4   with   |Δτ_cmd_j| ≤ τ̇·dt   (slew-admissible by construction)
    applied step ctrl  a = clip(prev_tau + Δτ_cmd, actuator_lo, actuator_hi)   then the per-sub-step governor
    identify           B_τ = ∂ v_rel,t+1 / ∂ Δτ_cmd   at the EXACT certified handoff, one control step

Because Δτ_cmd parameterises the rate-limited step DIRECTLY (it does not re-enter the rate limiter — it is already
admissible), it has authority the instant the actuator is not clipped and the governor is not fully suppressing the
joint. The nominal (Δτ_cmd = Δτ_0) reproduces the hold: Δτ_0 = clip(raw_pd − prev_tau, ±step) = a_hold − prev_tau. At
the handoff a saturated joint sits at Δτ_0 = ±step (the slew edge), so its admissible perturbation is ONE-SIDED into
the box; an interior joint uses a central difference. A ONE-STEP measurement only needs the cradle alive for one step,
so the fragile held-out cradles (alive 9–12 steps) qualify — the property Route A's lifted-horizon route could not use.

Reuses the Session-1 governed stepping EXACTLY (per-sub-step govern_torque callback + step_ablation + the FROZEN
contact frames / common-contact-point v_rel), so the ONLY change from B_v is the torque-increment decision variable.
NO pin, NO ε enlargement, NO τ_rate change, V2/V4 read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import BvConfig, CradleSnapshot, measure_contact_velocities
from hymeko_rl.coin_delivery.horizon_authority import HorizonConfig, cradle_alive, numeric_rank
from hymeko_rl.env.motion_contract import govern_torque


@dataclass(frozen=True)
class TorqueAuthorityConfig:
    """Frozen Route-B identification parameters. Cradle-alive floor + motion cap come from the embedded HorizonConfig
    (shared with Session 2, not duplicated); ε is in N·m as a fraction of the slew budget."""

    hz: HorizonConfig = field(default_factory=lambda: HorizonConfig(bv=BvConfig()))
    eps_scales_nm: tuple = (0.05, 0.10)     # N·m — two admissible torque-increment perturbation scales (< slew step 0.3)
    vrel_tol: float = 1e-4                   # m/s — a column below this response norm is inactive
    rank_abs_tol: float = 1e-3
    rank_rel_tol: float = 1e-2


def _slew_step(snap: CradleSnapshot) -> float:
    return float(snap.stack.tau_rate * snap.stack.control_dt)


def nominal_dtau(snap: CradleSnapshot) -> np.ndarray:
    """The hold's rate-limited increment Δτ_0 = clip(raw_pd − prev_tau, ±step) = a_hold − prev_tau — the operating point
    the FD linearises around (Δτ_cmd = Δτ_0 reproduces the nominal hold)."""
    d = snap.branch().inner.data
    q0, qd0 = d.qpos[:4].copy(), d.qvel[:4].copy()
    raw_pd = snap.stack.kp * (snap.q_hold - q0) - snap.stack.kv * qd0
    step = _slew_step(snap)
    return np.clip(raw_pd - snap.prev_tau, -step, step)


def apply_dtau_step(snap: CradleSnapshot, dtau_cmd, cfg: TorqueAuthorityConfig) -> tuple[np.ndarray, bool, dict]:
    """Advance a fresh branch EXACTLY one control step under the torque-increment command: applied step ctrl
    a = clip(prev_tau + Δτ_cmd, lo, hi), then the per-sub-step directional governor; measure v_rel,t+1 at the common
    contact point in the frozen frame. ``valid`` = the cradle is still alive (same contact mode, straddle, motion
    contract). # Preconditions: dtau_cmd length-4. # Postconditions: vrel4 length-4; diag carries the applied torque
    and the post-governor trace."""
    rl = snap.branch()
    d = rl.inner.data
    a = np.clip(snap.prev_tau + np.asarray(dtau_cmd, np.float64), snap.lo, snap.hi)
    trace: list = []

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
        trace.append(dt.ctrl[:4].copy())

    mujoco.set_mjcb_control(_gcb)
    try:
        step_ablation(rl, np.asarray(a, np.float32), "A")
        peak_qdot = float(np.max(np.abs(d.qvel[:4])))
        meas = measure_contact_velocities(rl, snap.frames)
    finally:
        mujoco.set_mjcb_control(None)
    alive, sig = cradle_alive(meas, peak_qdot, cfg.hz)
    tr = np.asarray(trace) if trace else np.asarray(a)[None, :]
    clipped = bool(np.any((np.asarray(a) <= snap.lo + 1e-6) | (np.asarray(a) >= snap.hi - 1e-6)))
    diag = {"applied_tau": [round(float(x), 5) for x in a], "actuator_clipped": clipped,
            "peak_qdot": round(peak_qdot, 4), "alive": alive, "mode": sig, "ctrl_trace": tr}
    return meas["vrel4"], alive, diag


def _admissible_column(snap, dtau0, j, eps, step, cfg) -> tuple[np.ndarray, bool, dict]:
    """One admissible FD column of B_τ for joint j: central where Δτ_0[j] is interior to [−step, step]; one-sided into
    the box where it sits on the slew edge (the saturated-joint case). Returns (column, both_branches_alive, diag)."""
    can_plus = dtau0[j] + eps <= step + 1e-12
    can_minus = dtau0[j] - eps >= -step - 1e-12
    ep = np.zeros(4)
    ep[j] = eps
    if can_plus and can_minus:
        gp, vp, dp = apply_dtau_step(snap, dtau0 + ep, cfg)
        gm, vm, dm = apply_dtau_step(snap, dtau0 - ep, cfg)
        col, mode = (gp - gm) / (2.0 * eps), "central"
        both = bool(vp and vm)
        clipped = bool(dp["actuator_clipped"] or dm["actuator_clipped"])
    else:
        g0, v0, d0 = apply_dtau_step(snap, dtau0, cfg)
        if can_minus:                                         # at +edge → one-sided down
            gm, vm, dm = apply_dtau_step(snap, dtau0 - ep, cfg)
            col, both, mode = (g0 - gm) / eps, bool(v0 and vm), "onesided_down"
            clipped = bool(d0["actuator_clipped"] or dm["actuator_clipped"])
        else:                                                 # at −edge → one-sided up
            gp, vp, dp = apply_dtau_step(snap, dtau0 + ep, cfg)
            col, both, mode = (gp - g0) / eps, bool(v0 and vp), "onesided_up"
            clipped = bool(d0["actuator_clipped"] or dp["actuator_clipped"])
    return col, both, {"j": j, "fd_mode": mode, "dtau0": round(float(dtau0[j]), 4),
                       "col_norm": round(float(np.linalg.norm(np.nan_to_num(col))), 6), "actuator_clipped": clipped}


def identify_Btau(snap: CradleSnapshot, eps: float, cfg: TorqueAuthorityConfig) -> dict:
    """B_τ (4×4) at perturbation scale ``eps`` (N·m) around the nominal hold increment Δτ_0. Column j valid iff both FD
    branches keep the cradle alive; every column carries its FD mode + actuator-clip flag + response norm."""
    step = _slew_step(snap)
    dtau0 = nominal_dtau(snap)
    Btau = np.full((4, 4), np.nan)
    col_valid = [False] * 4
    cols = {}
    for j in range(4):
        col, both, diag = _admissible_column(snap, dtau0, j, eps, step, cfg)
        if both:
            Btau[:, j] = col
            col_valid[j] = True
        diag["valid"] = bool(both)
        diag["active"] = bool(both and diag["col_norm"] > cfg.vrel_tol)
        cols[j] = diag
    rank = numeric_rank(Btau, cfg.rank_abs_tol, cfg.rank_rel_tol)
    n_active = int(sum(1 for j in range(4) if cols[j]["active"]))
    return {"Btau": Btau, "col_valid": col_valid, "cols": cols, "eps": float(eps), "rank": rank,
            "n_active": n_active, "dtau0": [round(float(x), 4) for x in dtau0], "slew_step": round(step, 4)}


def _repro_gap(Ba, Bb, va, vb) -> "float | None":
    gaps = []
    for j in range(4):
        if va[j] and vb[j]:
            denom = np.linalg.norm(Ba[:, j]) + np.linalg.norm(Bb[:, j])
            if denom > 1e-9:
                gaps.append(float(np.linalg.norm(Ba[:, j] - Bb[:, j]) / denom))
    return round(max(gaps), 4) if gaps else None


def analyse_state_btau(snap: CradleSnapshot, cfg: TorqueAuthorityConfig) -> dict:
    """Route-B one-step B_τ authority for one certified cradle: identify at both ε, rank, reproducibility, and the gate
    (rank ≥ 2 both ε, reproducible, actuator not fully clipping the active columns)."""
    ids = [identify_Btau(snap, e, cfg) for e in cfg.eps_scales_nm]
    eps_list = list(cfg.eps_scales_nm)
    rank_min = int(min(i["rank"] for i in ids))
    repro_gap = _repro_gap(ids[0]["Btau"], ids[1]["Btau"], ids[0]["col_valid"], ids[1]["col_valid"]) if len(ids) > 1 else None
    reproducible = bool(repro_gap is not None and repro_gap <= cfg.hz.repro_rel_gap)
    usable = bool(rank_min >= 2 and reproducible)
    return {
        "operating_point": snap.operating_point, "released": snap.released,
        "handoff_prev_tau": [round(float(x), 4) for x in snap.prev_tau], "arm_saturated": snap.arm_saturated,
        "fn0": [round(x, 4) for x in snap.fn0], "straddle0": round(snap.straddle0, 4),
        "slew_step_Nm": ids[0]["slew_step"], "dtau0": ids[0]["dtau0"],
        "rank_by_eps": {f"eps_{e}": i["rank"] for e, i in zip(eps_list, ids)},
        "n_active_by_eps": {f"eps_{e}": i["n_active"] for e, i in zip(eps_list, ids)},
        "rank_min": rank_min, "reproducible": reproducible, "repro_rel_gap": repro_gap,
        "usable_authority": usable,
        "Btau": {f"eps_{e}": [[(round(float(x), 5) if np.isfinite(x) else None) for x in row] for row in i["Btau"]]
                 for e, i in zip(eps_list, ids)},
        "cols": {f"eps_{e}": i["cols"] for e, i in zip(eps_list, ids)}}


def decide_btau_route(state_results: dict, dev_ids: list, heldout_ids: list) -> dict:
    """Route-B campaign gate: usable one-step Δτ authority on ≥1 development AND ≥1 held-out state (rank ≥ 2 both ε,
    reproducible). If met, Δτ_cmd is the H2 QP decision variable."""
    dev_usable = [state_results[i]["usable_authority"] for i in dev_ids if i in state_results]
    held_usable = [state_results[i]["usable_authority"] for i in heldout_ids if i in state_results]
    passed = bool(any(dev_usable) and any(held_usable))
    if passed:
        route, reason = "ROUTE_B_SLEW_ADMISSIBLE_DTAU_ESTABLISHED", (
            "usable one-step Δτ authority on ≥1 development AND ≥1 held-out state — Δτ_cmd is the H2 decision variable")
    elif any(dev_usable):
        route, reason = "ROUTE_B_DEV_ONLY", "Δτ authority usable on development but NOT held-out"
    else:
        route, reason = "AUTHORITY_RECOVERY_NOT_ESTABLISHED", "no usable Δτ authority even on development"
    return {"route": route, "reason": reason, "dev_usable": dev_usable, "heldout_usable": held_usable}
