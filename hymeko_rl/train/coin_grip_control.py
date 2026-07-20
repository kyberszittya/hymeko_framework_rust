"""COIN-GRIP-CONTROL-1 — active clamp-load regulation during midpoint motion (position-based, no CORE change).

COIN-FINGERTIP-PAD-2 (F-COIN-FINGERTIP-PAD, Case F) showed a position-controlled clamp holds the cylinder statically
(Z-capsule B_p dwell 16) but the coin slips the instant the midpoint moves (micro-transport 0). Hypothesis: static
position targets establish contact, but TRANSPORT needs active regulation of aperture / normal preload / compliance
during midpoint motion.

Actuator inventory (measured): 4 POSITION servos on the arm joints (gaintype fixed, biastype affine, no force limit);
NO torque/force/impedance actuator. BUT contact forces ARE observable via mj_contactForce, and aperture/squeeze/
differential are separate action dims (decoupled position targets). So aperture-feedback (C1), preload-proxy (C2), and a
force-feedback→position hybrid (C4h) need NO CORE change — only a true torque/impedance actuator would. All controllers
here are position-based. Signals are geometric/dynamic + OBSERVED contact force (mj_contactForce, labelled as force).
NO reward/CORE change; the coin stays cylindrical.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from hymeko_rl.train.coin_delivery_acquisition import APERTURE, BOTH_C
from hymeko_rl.train.coin_transport import obj_to_env


def normal_contact_forces(inner) -> tuple[float, float]:
    """Observed left/right fingertip↔coin normal contact-force magnitudes (mj_contactForce; a REAL force, not a proxy)."""
    import mujoco
    model, data = inner.model, inner.data
    fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    dg = int(getattr(inner, "_disk_geom", -1))
    FL = FR = 0.0
    buf = np.zeros(6)
    for i in range(int(data.ncon)):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if dg not in (g1, g2):
            continue
        other = g2 if g1 == dg else g1
        mujoco.mj_contactForce(model, data, i, buf)
        fn = abs(float(buf[0]))                            # contact-frame normal component
        if other == fl:
            FL += fn
        elif other == fr:
            FR += fn
    return FL, FR


class GripController(str, Enum):
    C0_CONSTANT = "C0_constant_squeeze"          # the pad-2 failure: fixed squeeze, no regulation
    C1_APERTURE_FB = "C1_aperture_feedback"      # close-loop aperture target (position-based)
    C2_PRELOAD_PROXY = "C2_geometric_preload"    # regulate a clamp-quality proxy (both/aperture/slip)
    C4H_FORCE_HYBRID = "C4h_force_feedback_hybrid"  # read normal contact force → command squeeze (no CORE change)


@dataclass(frozen=True)
class GripParams:
    v_translate: float = 0.15    # SLOW midpoint translation (transport needs a gentle move)
    aperture_star: float = -0.4  # target aperture (closed)
    k_a: float = 1.2             # aperture-feedback gain
    squeeze_base: float = 0.6    # baseline squeeze
    squeeze_boost: float = 0.4   # extra squeeze on clamp-deficit / low force
    force_star: float = 0.5      # target normal force (C4h)


def grip_action(inner, obs: np.ndarray, ctrl: GripController, p: GripParams, d: np.ndarray) -> np.ndarray:
    """Object-centric action: SLOW midpoint translation toward the target + the controller's clamp regulation."""
    dc = p.v_translate * d
    both = bool(obs[BOTH_C] > 0.5)
    ap = float(obs[APERTURE])
    if ctrl == GripController.C0_CONSTANT:
        return obj_to_env(dc, 0.0, 0.0, p.squeeze_base)
    if ctrl == GripController.C1_APERTURE_FB:
        da = p.k_a * (p.aperture_star - ap)                # drive aperture toward the closed target
        return obj_to_env(dc, da, 0.0, p.squeeze_base + (p.squeeze_boost if not both else 0.0))
    if ctrl == GripController.C2_PRELOAD_PROXY:
        deficit = (0.0 if both else 1.0)                   # clamp-quality proxy deficit (both-contact lost)
        return obj_to_env(dc, p.k_a * (p.aperture_star - ap) * deficit, 0.0, p.squeeze_base + p.squeeze_boost * deficit)
    # C4h: read the REAL normal contact force; boost squeeze when below target (force→position hybrid)
    FL, FR = normal_contact_forces(inner)
    fn = min(FL, FR)                                       # the weaker side governs clamp integrity
    return obj_to_env(dc, 0.0, 0.0, p.squeeze_base + p.squeeze_boost * float(np.clip(p.force_star - fn, 0.0, 1.0)))


def controlled_micro_transport(env, handoff, ctrl: GripController, p: GripParams, *,
                               establish=20, move_steps=16, scramble=None) -> dict:
    """From a handoff, establish a grasp, then translate the midpoint while the controller regulates the clamp. Report
    retention + coupling ρ = |Δcoin| / |Δmidpoint|. The env is built with the fingertip geometry by the caller."""
    from hymeko_rl.train.coin_grasp_cert import ContactModeClassifier, GraspFamily, GraspParams, grasp_action
    from hymeko_rl.train.coin_transport import restore_planar
    inner = env._env
    restore_planar(inner, handoff.snap)
    env._horizon = establish + move_steps + 400
    gp = GraspParams()
    clf = ContactModeClassifier(gp.dwell)
    obs = handoff.obs.copy()
    for _ in range(establish):                             # establish the grasp (fixed fingertip geometry)
        mode = clf.classify(inner)
        obs, _r, _t, _tr, _i = env.step(np.clip(grasp_action(inner, obs, mode, GraspFamily.G1_SYMMETRIC, gp), -1, 1).astype(np.float32))
    d, _n = inner.direction_to_zone()
    coin0 = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    mid0 = obs[[20, 21]].copy()                            # mid_to_coin proxy for midpoint tracking
    both_run = 0
    for t in range(move_steps):
        a = np.clip(grip_action(inner, obs, ctrl, p, np.asarray(d)), -1, 1).astype(np.float32)
        if scramble is not None:
            a = scramble(a, t)
        obs, _r, _t, _tr, _i = env.step(a)
        both_run += int(bool(inner._planar_metrics.left_contact and inner._planar_metrics.right_contact))
    mid_disp = float(np.linalg.norm(obs[[20, 21]] - mid0)) + p.v_translate * 0.012 * move_steps  # commanded midpoint move
    return _transport_verdict(handoff, ctrl, inner._planar_metrics, coin0, np.asarray(d, np.float64),
                              mid_disp, both_run, move_steps)


def _transport_verdict(handoff, ctrl, m, coin0, d, mid_disp, both_run, move_steps) -> dict:
    """Retention + coupling ρ + the STRICT grasp-integrity predicate. The strict guard requires the coin to stay
    BILATERALLY clamped throughout AND end bilateral AND travel WITH the midpoint (align≥0.7) — ruling out the single-
    finger bulldoze/squirt the loose predicate can be inflated by (evaluation-integrity)."""
    coin1 = np.asarray(m.disk_pos[:2], np.float64)
    coin_disp = float(np.linalg.norm(coin1 - coin0))
    retained = bool(m.left_contact or m.right_contact)
    final_bilateral = bool(m.left_contact and m.right_contact)
    both_frac = both_run / max(1, move_steps)
    align = float((coin1 - coin0) @ d) / (coin_disp * float(np.linalg.norm(d)) + 1e-9)   # drag vs perpendicular squirt
    rho = round(coin_disp / (mid_disp + 1e-9), 3)
    micro = bool(retained and coin_disp >= 0.01 and rho >= 0.3 and both_run >= move_steps // 3)
    micro_strict = bool(final_bilateral and both_frac >= 0.75 and coin_disp >= 0.01 and rho >= 0.3 and align >= 0.7)
    return {"seed": handoff.seed, "controller": ctrl.value, "coin_disp": round(coin_disp, 4), "rho": rho,
            "retained": retained, "both_frac": round(both_frac, 3), "align": round(align, 3),
            "final_bilateral": final_bilateral, "micro_transport": micro, "micro_strict": micro_strict}
