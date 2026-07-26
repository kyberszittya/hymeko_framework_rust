"""Route A — MOBILE LOW-DEBT HANDOFF conditioning (free coin, no pin in the evaluated handoff).

H2 Session 2 measured that lifted-horizon Δq_target authority is REAL and usable on well-conditioned cradles (s1, s3)
but the two held-out cradles (s4, s7) COLLAPSE in 9–12 control steps before the authority is usable. The diagnosed
collapse mechanism (from `authority_recovery.json` baseline rollouts) is GRIP LOSS — the tip normal force decays below
the contact floor under the hold (s4: chronic weak left contact fn≈0.05–0.13 N; s7: progressive preload decay 5→0 N) —
NOT a velocity or straddle-inversion failure (peak q̇ ≤ 0.9 rad/s ≪ 3.0).

The directed-straddle acquisition (`straddle_directed_acquire`) stops at a TRANSIENT: it pushes the tips to opposite
sides along the squeeze axis and hands off before the preload settles into a sustained equilibrium. This module adds a
short MOBILE (free-coin) conditioning phase that reuses the acquisition's PENETRATION-SERVO preload law (from
`acquire_clean_preload`) to drive BOTH tips to a balanced, bounded, SUSTAINED preload while damping velocity — the same
δ setpoint that already gives the development cradles their healthy 2–5 N grip, so it is a PHYSICAL setpoint developed on
the model, not a held-out-tuned threshold.

The conditioning runs on the FREE coin (the pin is released first); no pin remains active in the evaluated handoff. It
does NOT touch V2/V4 physics, morphology, friction, the torque-rate limit, or the actuator range. It only re-runs the
existing squeeze controller onto the free coin until forces settle.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.cooperative_launch import (
    _LEFT_DOF, _RIGHT_DOF, _tip_contacts, CooperativeConfig, release_pin)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.env.governed_arm import V3Stack, pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque


@dataclass(frozen=True)
class ConditionConfig:
    """Mobile conditioning parameters. MONOTONE squeeze: hold the current gripping pose and only ADD inward squeeze to
    an UNDER-loaded side (never retract) — on a free coin, more squeeze is safe but relieving preload releases the coin,
    so a healthy cradle (fn already ≥ ``fn_target``) is left untouched and only a weak side is rescued. ``fn_target`` is
    a physical 'safely-gripped' setpoint well above the 0.05 N collapse floor and below the development cradles' 2–5 N
    band (so s1/s3 need no conditioning); it is NOT a held-out-tuned threshold."""

    max_steps: int = 20          # SHORT bounded budget — minimise geometry drift (over-processing degrades good cradles)
    dwell: int = 4               # consecutive force-healthy steps → freeze & export (stop as soon as the cradle is safe)
    fn_target: float = 1.5       # N — squeeze a side inward only while its normal force is below this
    squeeze_inc: float = 0.004   # rad/step — inward joint increment added to an under-loaded arm
    balance_min: float = 0.30    # min(Fn_l,Fn_r)/max — the early-stop force-balance gate (qdot is NOT gated: the free-coin
    #                              contact limit-cycle spikes q̇ above the acquisition floor, so gating settle on it never
    #                              triggers and the loop over-processes; the horizon cradle-alive check bounds motion)


def _fingertip_geoms(m):
    return (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right"))


def arm_inward_geom(rl, g, dofs, coin) -> np.ndarray:
    """The ANALYTIC joint-space direction (over ONLY this arm's dofs) that moves tip geom ``g`` toward ``coin`` — the
    descent direction of ½‖tip−coin‖², i.e. ``Jₚ,dofsᵀ·(coin−tip)`` from the instantaneous MuJoCo geom Jacobian
    ``mj_jacGeom``. Unlike the acquisition's FD probe (which steps the sim and, on a FREE coin, shoves the coin so the
    measured gradient is corrupted), this is instantaneous and coin-motion-independent — valid on the mobile coin.

    # Postconditions: unit-norm 4-vector, nonzero only on ``dofs`` (0 if the tip is kinematically decoupled here).
    """
    m, d = rl.inner.model, rl.inner.data
    jacp = np.zeros((3, m.nv), np.float64)
    mujoco.mj_jacGeom(m, d, jacp, None, int(g))
    to_coin = np.asarray(coin, np.float64) - d.geom_xpos[int(g)][:2].astype(np.float64)
    v = np.zeros(4, np.float64)
    for j in dofs:
        v[j] = jacp[0, j] * to_coin[0] + jacp[1, j] * to_coin[1]
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else v


def condition_mobile_handoff(rl, stack: V3Stack, saved, prev_tau0, q_target0, *,
                             coop: CooperativeConfig | None = None, ccfg: ConditionConfig | None = None) -> dict:
    """Release the pin (MOBILE free coin) and, starting from the EXACT acquisition hold (base servo target
    ``q_target0``, threaded ``prev_tau0`` — so the STANDING grip torque is preserved, exactly as the baseline
    horizon rollout does), add a MONOTONE inward squeeze to any under-loaded side until both tips hold a balanced,
    settled preload. Leaves ``rl`` in the conditioned FREE-coin state and returns the conditioned handoff controller
    state (``prev_tau``, ``q_target``) plus settle diagnostics — a drop-in replacement for the raw acquisition handoff.

    The base is the acquisition hold (NOT the current qpos with a fresh torque): the grip is sustained by the standing
    torque, so re-seeding it from zero would release the coin. A healthy cradle (fn already ≥ ``fn_target``) gets zero
    squeeze ⇒ conditioning ≡ the surviving baseline hold; only a weak side is squeezed in.

    # Preconditions: ``rl`` at a dual-contact straddle cradle (soft pin still on, ``saved`` its handle); ``prev_tau0`` /
    ``q_target0`` the acquisition's ``final_tau`` / ``final_q_target``. # Postconditions: pin released; ``rl`` a
    free-coin cradle; returns prev_tau length-4, q_target length-4, ``settled``.
    """
    coop = coop or CooperativeConfig()
    ccfg = ccfg or ConditionConfig()
    release_pin(rl, saved)                                        # MOBILE — no pin remains in the evaluated handoff
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl, gr = _fingertip_geoms(m)
    arms = (("left", gl, _LEFT_DOF), ("right", gr, _RIGHT_DOF))

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)

    mujoco.set_mjcb_control(_gcb)
    base = np.asarray(q_target0, np.float64).copy()               # the acquisition hold target (sustains the grip)
    squeeze = np.zeros(4)                                         # accumulated inward offset (monotone, never retracts)
    target = base.copy()
    prev_tau = None if prev_tau0 is None else np.asarray(prev_tau0, np.float64).copy()   # thread the standing grip torque
    dwell, settled, steps_used = 0, False, 0
    fn_hist, fnl, fnr, qdot, balance = [], 0.0, 0.0, 0.0, 0.0
    try:
        for k in range(ccfg.max_steps):
            steps_used = k + 1
            coin = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]
            mpl = rl.inner._planar_metrics
            tc = _tip_contacts(rl)
            for side, g, dofs in arms:
                fn, pen = float(tc[side][0]), float(tc[side][2])
                contact = bool(mpl.left_contact if side == "left" else mpl.right_contact)
                buried = pen <= -coop.clean_pen_max              # already at the burial limit — stop squeezing
                if (not contact) or (fn < ccfg.fn_target and not buried):   # under-loaded (or dropped): squeeze IN
                    inward = arm_inward_geom(rl, g, dofs, coin)  # ANALYTIC direction (valid on the free coin)
                    for j in dofs:
                        squeeze[j] += ccfg.squeeze_inc * inward[j]
            target = base + squeeze
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), target, stack, prev_tau, lo, hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
            tc = _tip_contacts(rl)
            fnl, fnr = float(tc["left"][0]), float(tc["right"][0])
            qdot = float(np.max(np.abs(d.qvel[:4])))
            fn_hist.append((round(fnl, 4), round(fnr, 4)))
            mpl = rl.inner._planar_metrics
            balance = min(fnl, fnr) / max(fnl, fnr, 1e-6)
            ok = bool(mpl.left_contact and mpl.right_contact and fnl >= ccfg.fn_target and fnr >= ccfg.fn_target
                      and balance >= ccfg.balance_min)              # force-health only — freeze the moment the grip is safe
            dwell = dwell + 1 if ok else 0
            if dwell >= ccfg.dwell:
                settled = True
                break
    finally:
        mujoco.set_mjcb_control(None)
    return {"prev_tau": (None if prev_tau is None else np.asarray(prev_tau, np.float64).copy()),
            "q_target": np.asarray(target, np.float64).copy(), "settled": bool(settled), "steps_used": steps_used,
            "fn_final": (round(fnl, 4), round(fnr, 4)), "balance_final": round(balance, 3),
            "qdot_final": round(qdot, 4), "fn_hist": fn_hist}
