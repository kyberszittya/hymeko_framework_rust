"""R1 signed DIRECTIONAL authority — reachable-set object + contact/internal authority (equivariant, deploy-faithful).

The magnitude-only B_τ features (R1 v2) improved the acceptable-set distance (0.87→0.70) but not to within budget-8 reach:
they say *how much* one-step authority exists, not *which* signed torque direction pushes the coin forward vs brakes it.
This module adds the signed information, with the physics corrections the task requires:

  1. REACHABLE SET, not raw Jacobian columns. For a target-frame object direction d, the achievable one-step response is
         a_d± = max_{Δτ ∈ U_slew}  ± dᵀ B_coin Δτ,
     a linear program over the ASYMMETRIC slew-admissible box U_slew (per-joint ±headroom from prev_tau & the rate limit).
     Closed form over a box; captures joint cooperation and the asymmetric headroom. Directions: forward push / reverse,
     lateral ±, and the velocity-opposed brake d_brake = −v_coin/‖v_coin‖ (with a stable fallback at low speed).
  2. SQUEEZE IS NOT IN B_coin. An ideal internal squeeze lives in the NULL space of the object-velocity Jacobian
     (B_coin Δτ ≈ 0), so it is invisible in the coin response. Squeeze / preload / balance authority is measured on the
     CONTACT side (B_τ = ∂vrel4/∂Δτ): total-normal-force ±, left-right balance, and internal/null-motion reach. The two
     spaces stay separate — B_coin → object motion, B_τ → internal force.
  3. DEPLOY-FAITHFUL FD. B_coin is measured through the EXACT executed interface (requested Δτ → rate-limiter → governor →
     actuator clip → MuJoCo), reusing the frozen governed step. The realised-vs-requested post-governor torque ratio is
     recorded as provenance so a governor-attenuated (paper-only) authority is visible, not silently trusted.

Mirror equivariance (target frame): B_coin(Mx) ≈ S_y B_coin(x) P_τ with S_y = diag(1,−1[,−1]) (along invariant, perp — and
the planar-spin pseudoscalar — flip) and P_τ the left-right joint permutation. ⇒ forward push/brake invariant, lateral ±
swap, squeeze ± invariant, L/R balance sign-flips. (Spin ± is a deferred enrichment: the disk spin DOF is not surfaced in
the planar metrics; the teacher's push/brake are in-plane translation, which this covers.)
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.env.motion_contract import govern_torque

BCOIN_EPS = 0.05                                      # N·m FD perturbation (< the 0.3 slew step; matches identify_Btau)


def target_frame(snap: CradleSnapshot) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """(e_par = coin→zone unit, e_perp = ⊥, coin velocity in target frame [v_along, v_perp], coin speed) at the handoff."""
    rl = snap.branch()
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float64)[:2]
    nrm = float(np.linalg.norm(e_par))
    e_par = e_par / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
    e_perp = np.array([-e_par[1], e_par[0]])
    v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    return e_par, e_perp, np.array([v @ e_par, v @ e_perp]), float(np.linalg.norm(v))


def _governed_coin_step(snap: CradleSnapshot, dtau_cmd: np.ndarray, e_par: np.ndarray, e_perp: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Advance a fresh branch EXACTLY one governed control step under Δτ_cmd (a = clip(prev_tau+Δτ, lo, hi) → per-sub-step
    directional governor), then read the coin velocity in the target frame [v_along, v_perp] AND the realised post-governor
    torque (trace mean). Same executed interface as `torque_authority.apply_dtau_step`. # Postconditions: returns
    (coin_vel_target_frame(2), realised_tau(4))."""
    rl = snap.branch()
    d = rl.inner.data
    a = np.clip(snap.prev_tau + np.asarray(dtau_cmd, np.float64), snap.lo, snap.hi)
    trace: list = []

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
        trace.append(dt.ctrl[:4].copy())

    mujoco.set_mjcb_control(_gcb)
    try:
        step_ablation(rl, np.asarray(a, np.float32), "A")
        vx = float(d.qvel[rl.inner._disk_dofx])
        vy = float(d.qvel[rl.inner._disk_dofy])
    finally:
        mujoco.set_mjcb_control(None)
    realised = np.asarray(trace, np.float64).mean(0) if trace else np.asarray(a, np.float64)
    return np.array([vx * e_par[0] + vy * e_par[1], vx * e_perp[0] + vy * e_perp[1]]), realised


def identify_Bcoin(snap: CradleSnapshot, eps: float = BCOIN_EPS) -> dict[str, Any]:
    """B_coin = ∂(coin [v_along, v_perp]) / ∂(requested Δτ) at the handoff, by central finite differences THROUGH the
    executed governed interface (so an authority the governor kills reads ≈0). Returns the 2×4 Jacobian + the realised/
    requested post-governor torque attenuation per column (provenance: a low ratio ⇒ paper-only authority). Columns whose
    realised perturbation vanishes are zeroed. # Postconditions: B (2,4) finite; attenuation ∈ [0,~1]."""
    e_par, e_perp, _v0, _sp = target_frame(snap)
    B = np.zeros((2, 4))
    atten = np.zeros(4)
    for j in range(4):
        ep = np.zeros(4)
        ep[j] = eps
        vp, tp = _governed_coin_step(snap, ep, e_par, e_perp)
        vm, tm = _governed_coin_step(snap, -ep, e_par, e_perp)
        realised = float(abs(tp[j] - tm[j]))            # realised post-governor torque change on joint j
        atten[j] = realised / (2.0 * eps)               # 1 ⇒ fully executed; ~0 ⇒ governor-absorbed
        if realised > 1e-6:
            B[:, j] = (vp - vm) / (2.0 * eps)           # per REQUESTED Δτ (governor inside the numerator = executed)
    return {"B": B, "attenuation": atten, "e_par": e_par, "e_perp": e_perp, "eps": float(eps)}


def admissible_dtau_box(snap: CradleSnapshot) -> "tuple[np.ndarray, np.ndarray]":
    """The per-joint slew-admissible Δτ range [lb, ub]: |Δτ_j| ≤ slew (rate limit) AND prev_tau_j+Δτ_j ∈ [lo_j, hi_j]
    (actuator). Asymmetric when a joint sits near an actuator/slew edge (the saturated-joint case). # Postconditions:
    lb ≤ 0 ≤ ub componentwise."""
    slew = float(snap.stack.tau_rate * snap.stack.control_dt)
    lb = np.maximum(-slew, snap.lo - snap.prev_tau)
    ub = np.minimum(slew, snap.hi - snap.prev_tau)
    return np.minimum(lb, 0.0), np.maximum(ub, 0.0)


def reachable(c: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """max_{Δτ ∈ [lb,ub]} cᵀΔτ — the closed-form reachable response along coefficient vector c over the asymmetric box
    (each joint independently picks its range endpoint by the sign of c_j). ``c = dᵀ B_coin`` for a target-frame direction
    d. # Postconditions: ≥ 0 when 0 ∈ [lb,ub] (Δτ=0 is admissible)."""
    c = np.asarray(c, np.float64)
    return float(np.sum(np.where(c > 0, c * np.asarray(ub), c * np.asarray(lb))))


def object_reaches(B: np.ndarray, lb: np.ndarray, ub: np.ndarray, v_tf: np.ndarray, sp: float,
                   min_atten: float = 1.0) -> dict[str, np.ndarray]:
    """PURE reachable object authority from a precomputed B_coin (2×4) + admissible box + coin target-frame velocity. Split
    out for testability (the Jacobian equivariance test) and so the physics is computed ONCE. forward ±along (invariant),
    lateral ±perp (swap pair), brake along −v/‖v‖ (low-speed fallback −along)."""
    d_fwd = np.array([1.0, 0.0])
    d_lat = np.array([0.0, 1.0])
    d_brake = (-np.asarray(v_tf, np.float64) / sp) if sp > 1e-3 else np.array([-1.0, 0.0])
    return {
        "forward_push_reach": np.array([reachable(d_fwd @ B, lb, ub)]),
        "forward_reverse_reach": np.array([reachable(-d_fwd @ B, lb, ub)]),
        "lateral_reach_pair": np.array([reachable(d_lat @ B, lb, ub), reachable(-d_lat @ B, lb, ub)]),
        "brake_opposed_reach": np.array([reachable(d_brake @ B, lb, ub)]),
        "bcoin_min_attenuation": np.array([float(min_atten)]),
    }


def contact_reaches(Btau: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> dict[str, np.ndarray]:
    """PURE contact/internal authority from a precomputed B_τ (4×4, rows [Lvn,Lvt,Rvn,Rvt]) + admissible box. total-normal ±
    = reachable ±(Lvn+Rvn) (symmetric, mirror-INVARIANT); balance = reachable(Lvn−Rvn) − reachable(Rvn−Lvn) (antisymmetric,
    mirror SIGN-FLIP). Split out for testability + single-physics."""
    sym = Btau[0] + Btau[2]
    anti = Btau[0] - Btau[2]
    return {"normal_force_reach_pair": np.array([reachable(sym, lb, ub), reachable(-sym, lb, ub)]),
            "balance_reach_signed": np.array([reachable(anti, lb, ub) - reachable(-anti, lb, ub)])}


def object_authority(snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """Signed object-motion reachable authority (target frame) — physics wrapper over `object_reaches` (computes B_coin +
    box once). # Postconditions: all reaches ≥ 0; equivariant per the layout kinds."""
    info = identify_Bcoin(snap)
    lb, ub = admissible_dtau_box(snap)
    _ep, _epp, v_tf, sp = target_frame(snap)
    return object_reaches(info["B"], lb, ub, v_tf, sp, float(info["attenuation"].min()))


def contact_internal_authority(snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """Squeeze / preload / balance authority on the CONTACT side (B_τ, where the internal force lives — it is ~null in the
    object response). Physics wrapper over `contact_reaches`. # Postconditions: reaches ≥ 0; balance signed."""
    from hymeko_rl.coin_delivery.torque_authority import TorqueAuthorityConfig, identify_Btau
    B = np.nan_to_num(np.asarray(identify_Btau(snap, BCOIN_EPS, TorqueAuthorityConfig())["Btau"], np.float64), nan=0.0)
    lb, ub = admissible_dtau_box(snap)
    return contact_reaches(B, lb, ub)
