"""Cooperative two-contact launch with a REACHABILITY-GATED, DECOUPLED, synchronized close (the (1) fix after BIMANUAL_V1
found two-contact acquisition unreliable). Two facts drove the design:
  * the 4 DoF split cleanly — DoF 0–1 = LEFT arm, 2–3 = RIGHT arm — so each arm is driven INDEPENDENTLY (a coupled 4-DoF
    gradient let one arm dominate);
  * on part of the panel the RIGHT arm cannot reach the coin at all (workspace limit) ⇒ two-contact is geometrically
    impossible there — a REACHABILITY fact that must be checked, not assumed.

Acquire: each arm closes toward the coin CENTRE with its own DoF at equal gain (a SYNCHRONISED symmetric pinch — if the
light coin drifts toward one tip, that tip contacts sooner, so the close is self-correcting). Once both tips contact, run
the cooperative twist-Jacobian allocation (translate toward the zone, zero spin). A per-state REACHABILITY PROBE reports
whether both arms can individually reach the coin — the relational fact the structural prior would predict.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.bimanual_launch import _coin_twist, _coin_twist_jacobian, _unit2
from hymeko_rl.coin_delivery.coin_carry_structured import HELD_DWELL, _det, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_G = 9.81
_LEFT_DOF, _RIGHT_DOF = (0, 1), (2, 3)


@dataclass(frozen=True)
class CooperativeConfig:
    coast_mu: float = 0.179
    coin_radius: float = 0.02
    close_gain: float = 0.10
    launch_gain: float = 0.06
    lam: float = 0.06
    w_omega: float = 3.0
    grasp_lam: float = 0.05            # A2: Tikhonov damping on the grasp-matrix force solve
    replan_every: int = 2
    reach_steps: int = 90
    contact_tol: float = 0.03          # m — tip within this of the coin centre counts as reach-capable
    probe_mag: float = 2.0
    probe_steps: int = 3
    preload_depth: float = 0.005       # m of surface overlap the tips COMMAND — a BOUNDED solid preload (not the coin
    #                                    centre, which is unreachable and marches the arm in until the torque saturates)
    approach_gain: float = 0.05        # rad/step — per-arm distance-servo gain in E0a (a fixed 0.10 step overshot the thin
    #                                    contact margin and buried the tip; the servo scales the step by the reach error)
    servo_scale: float = 0.02          # m — reach error that saturates the servo step (finer control near the setpoint)
    acquire_dwell: int = 12            # frames of stable in-band co-contact required before the preload counts as acquired
    settle_qdot: float = 0.45          # rad/s — gate above the ~0.33 CONTACT limit-cycle floor (simulator-calibrated: the
    #                                    tips vibrate against the near-rigid damped coin; E0b release-sanity is the physical
    #                                    backstop that stored energy is not launching the coin, not this joint-speed gate)
    settle_frames: int = 60            # extra frames the servo may run to reach a stable in-band settled co-contact
    # --- clean-preload gate (FROZEN before the run; a release is only honest from a preload that passes all of these) ---
    clean_pen_min: float = 0.001       # m — min |surface overlap| (a real preload, not a marginal flickering touch)
    clean_pen_max: float = 0.010       # m — max |surface overlap| (bounded, no burial); accepted band ≠ commanded depth
    clean_qdot_max: float = 0.45       # rad/s — max pre-release joint speed (calibrated to the contact floor; see settle_qdot)
    clean_fn_min: float = 0.15         # N — each tip must carry at least this normal force (both really in contact)
    clean_fn_balance_min: float = 0.30  # min(Fn_l,Fn_r)/max — the two normals are balanced, not one-sided
    # --- E1 authority-balance existence oracle (a balanced frame must carry real force AND be near-symmetric) ---
    preload_min_total: float = 0.60    # N — min Fn_L + Fn_R, so the trivial Fn≈0 (touch-but-no-load) point cannot win
    imbalance_max: float = 0.25         # |Fn_L − Fn_R|/(Fn_L + Fn_R) — scale-free balance target, tighter than the clean gate
    search_span: float = 0.04          # m — half-range of the 1-D coin sweep along the fingertip–fingertip axis
    search_n: int = 9                  # candidates in the sweep
    # --- LAUNCH_FEASIBLE_ACQUISITION lexicographic gate (the option-postcondition; Fn-balance is only a diagnostic) ---
    g2_force_max: float = 0.30         # N — G2 max realized NET contact force on the coin (a clean preload = null net wrench)
    g2_torque_max: float = 0.010       # N·m — G2 max realized net coin torque (moment arms can spin an Fn-balanced pair)
    g3_fpar_min: float = 0.10          # (unit grasp force) — G3 min achievable forward net force under the cone
    g3_cross_max: float = 0.20         # G3 max achievable cross-force fraction of the directed grasp solve
    # --- E3 active net-wrench-nulling acquisition (object-level feedback: min‖w_coin‖ s.t. G1 dual contact ∧ G3 feasible) ---
    wrench_null_kf: float = 0.004      # m per N — TANGENTIAL tip slide per unit net force (radial stays the δ servo → keeps G1)
    wrench_null_ktau: float = 0.15     # m per N·m — tangential slide per unit net coin torque
    wrench_null_slide_max: float = 0.005  # m — clamp on the tangential slide, so it cannot walk the tip off the round coin
    wrench_null_gain: float = 0.03     # rad/step — gentle nulling target step (< approach_gain; a big step lost contact)
    wrench_null_steps: int = 200       # feedback iterations to drive ‖w_coin‖ into the G2 band
    wrench_null_dwell: int = 12        # frames the wrench must stay in the G2 band before E3a terminates


def _tip_xy(rl, g):
    return rl.inner.data.geom_xpos[g][:2].astype(np.float32)


def _arm_dir(rl, g, dofs, target, cfg):
    """FD joint direction over ONLY this arm's DoF driving its tip to ``target`` (decoupled per-arm control)."""
    c0 = float(np.linalg.norm(_tip_xy(rl, g) - target))
    grad = np.zeros(4, np.float32)
    for j in dofs:
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = cfg.probe_mag
        for _ in range(cfg.probe_steps):
            step_ablation(r2, a, "A")
        grad[j] = (float(np.linalg.norm(_tip_xy(r2, g) - target)) - c0) / (cfg.probe_mag * cfg.probe_steps)
    g2 = -grad
    n = float(np.linalg.norm(g2))
    return (g2 / n).astype(np.float32) if n > 1e-6 else np.zeros(4, np.float32)


def _coin_radius(rl):
    """The coin (disk) geom radius read from the MuJoCo model — never hardcoded, so a different coin geometry cannot
    reintroduce the centre-vs-surface preload bug."""
    dj = mujoco.mj_name2id(rl.inner.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    return float(rl.inner.model.geom_size[dj][0])


def _surface_target(rl, g, coin, cfg):
    """The tip-CENTRE position that gives a bounded ``preload_depth`` surface overlap on this tip's side. Two spheres/disks
    touch when their centre distance drops below (r_coin + r_tip) — BOTH radii read from the model; aiming the tip centre
    at that minus ``preload_depth`` makes the close self-limit at a light balanced contact (the FD gradient vanishes at the
    target, so the arm stops marching in instead of burying the tip, which ignoring r_tip did, until the torque saturates)."""
    r_tip = float(rl.inner.model.geom_size[g][0])
    v = _tip_xy(rl, g) - np.asarray(coin, np.float32)
    reach = _coin_radius(rl) + r_tip - cfg.preload_depth
    return (np.asarray(coin, np.float32) + reach * (v / (float(np.linalg.norm(v)) + 1e-9))).astype(np.float32)


def place_coin_at(rl, xy):
    """EASY-SCENARIO helper: move the coin to a chosen xy (e.g. the two-arm-reachable tip midpoint), zero its velocity."""
    adr = int(rl.inner._disk_x_adr)
    rl.inner.data.qpos[adr:adr + 2] = np.asarray(xy, np.float64)
    rl.inner.data.qvel[adr:adr + 3] = 0.0
    mujoco.mj_forward(rl.inner.model, rl.inner.data)


def tip_midpoint(rl):
    """The midpoint of the two fingertips — a coin position both arms can reach (the easy scenario)."""
    m, d = rl.inner.model, rl.inner.data
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    return (0.5 * (d.geom_xpos[gl][:2] + d.geom_xpos[gr][:2])).astype(np.float32)


def _pin_coin(rl, on: bool, saved=None):
    """Kinematically FIX the coin (huge slide damping) so a reachability probe measures GEOMETRIC reach, not whether the
    arm shoves a mobile coin away. Returns the saved damping to restore. STATIC_GEOMETRIC_REACHABILITY."""
    m = rl.inner.model
    adr = int(rl.inner._disk_x_adr)
    if on:
        saved = m.dof_damping[adr:adr + 3].copy()
        m.dof_damping[adr:adr + 3] = 1e5
        return saved
    m.dof_damping[adr:adr + 3] = saved
    return None


def static_reachability_probe(rl, stack, cfg: CooperativeConfig | None = None):
    """STATIC_GEOMETRIC_REACHABILITY: pin the coin, then drive each arm alone to the coin's contact arc — can the tip
    reach it geometrically (min tip-coin ≤ contact_tol), independent of the mobile-coin dynamics? The honest geometric
    verdict (vs the dynamic ``reachability_probe`` which chases a mobile coin)."""
    cfg = cfg or CooperativeConfig()
    out = {}
    for side, g_name, dofs in (("left", "fingertip_left", _LEFT_DOF), ("right", "fingertip_right", _RIGHT_DOF)):
        r = copy.deepcopy(rl)
        m, d = r.inner.model, r.inner.data
        m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
        saved = _pin_coin(r, True)
        lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g_name)

        def _gcb(_mo, dt, _gov=stack.gov):
            dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], _gov)
        mujoco.set_mjcb_control(_gcb)
        prev, touched, min_d = None, 0, 9.0
        try:
            for _ in range(cfg.reach_steps):
                coin = np.asarray(r.inner._planar_metrics.disk_pos, np.float32)[:2]
                min_d = min(min_d, float(np.linalg.norm(_tip_xy(r, g) - coin)))
                v = _tip_xy(r, g) - coin
                tgt = coin + cfg.coin_radius * (v / (np.linalg.norm(v) + 1e-9))
                delta = cfg.close_gain * _arm_dir(r, g, dofs, tgt, cfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev, lo, hi)
                prev = a
                step_ablation(r, np.asarray(a, np.float32), "A")
                c = r.inner._planar_metrics.left_contact if side == "left" else r.inner._planar_metrics.right_contact
                touched += int(bool(c))
        finally:
            mujoco.set_mjcb_control(None)
            _pin_coin(r, False, saved)
        out[side] = {"reachable": bool(touched > 0 or min_d <= cfg.contact_tol), "contact_frames": touched,
                     "min_tip_coin": round(min_d, 4)}
    out["two_contact_reachable"] = bool(out["left"]["reachable"] and out["right"]["reachable"])
    return out


def reachability_probe(rl, stack, cfg: CooperativeConfig | None = None):
    """DYNAMIC_ACQUISITION_REACHABILITY: drive EACH arm alone toward the (MOBILE) coin — can it contact given the coin can
    be pushed away. Complements ``static_reachability_probe`` (geometric reach)."""
    cfg = cfg or CooperativeConfig()
    out = {}
    for side, g_name, dofs in (("left", "fingertip_left", _LEFT_DOF), ("right", "fingertip_right", _RIGHT_DOF)):
        r = copy.deepcopy(rl)
        m, d = r.inner.model, r.inner.data
        m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
        lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g_name)

        def _gcb(_mo, dt, _gov=stack.gov):
            dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], _gov)
        mujoco.set_mjcb_control(_gcb)
        prev, touched, min_d = None, 0, 9.0
        try:
            for _ in range(cfg.reach_steps):
                coin = np.asarray(r.inner._planar_metrics.disk_pos, np.float32)[:2]
                min_d = min(min_d, float(np.linalg.norm(_tip_xy(r, g) - coin)))
                v = _tip_xy(r, g) - coin
                tgt = coin + cfg.coin_radius * (v / (np.linalg.norm(v) + 1e-9))
                delta = cfg.close_gain * _arm_dir(r, g, dofs, tgt, cfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + delta, stack, prev, lo, hi)
                prev = a
                step_ablation(r, np.asarray(a, np.float32), "A")
                c = r.inner._planar_metrics.left_contact if side == "left" else r.inner._planar_metrics.right_contact
                touched += int(bool(c))
        finally:
            mujoco.set_mjcb_control(None)
        out[side] = {"reachable": bool(touched > 0), "contact_frames": touched, "min_tip_coin": round(min_d, 4)}
    out["two_contact_reachable"] = bool(out["left"]["reachable"] and out["right"]["reachable"])
    return out


def _preload_sanity(rl, prev_tau, lo, hi, cfg, dwell):
    """The E0 clean-preload gate: a release is only honest from a BOUNDED, BALANCED, SETTLED bilateral preload — never a
    spring-load explosion accumulated against the pin. Records per-tip normal/tangential force, normal balance,
    penetration, the settled joint speed, torque saturation and the both-contact dwell; ``clean`` gates ALL of the frozen
    ``cfg.clean_*`` thresholds. (Feature history: an earlier centre-target close buried the tips 18–34 mm — deeper than
    the coin — so the positive launch velocity was a pin-release artifact; this gate exists to reject exactly that.)"""
    tc = _tip_contacts(rl)
    fnl, fnr = tc["left"][0], tc["right"][0]
    prev = np.asarray(prev_tau, np.float64) if prev_tau is not None else np.zeros(4)
    p = {"fn_left": round(fnl, 3), "fn_right": round(fnr, 3), "ft_left": round(tc["left"][1], 3),
         "ft_right": round(tc["right"][1], 3), "fn_balance": round(min(fnl, fnr) / max(fnl, fnr, 1e-6), 3),
         "penetration_left": round(tc["left"][2], 4), "penetration_right": round(tc["right"][2], 4),
         "qdot_prerelease": round(float(np.max(np.abs(rl.inner.data.qvel[:4]))), 3),
         "torque_saturated": bool(np.any((prev <= lo + 1e-6) | (prev >= hi - 1e-6))), "both_contact_dwell": int(dwell)}
    pens = (p["penetration_left"], p["penetration_right"])
    in_band = all(-cfg.clean_pen_max <= pen <= -cfg.clean_pen_min for pen in pens)   # accepted band, not exact depth
    p["clean"] = bool(p["qdot_prerelease"] <= cfg.clean_qdot_max and not p["torque_saturated"]
                      and p["fn_balance"] >= cfg.clean_fn_balance_min and fnl >= cfg.clean_fn_min and fnr >= cfg.clean_fn_min
                      and in_band)
    return p


def release_pin(rl, saved):
    """Undo the E0a soft pin — restore the coin's slide damping so a subsequent release-only or launch stage runs on a
    genuinely FREE coin. Pairs with the ``saved`` handle returned by ``acquire_clean_preload``."""
    _pin_coin(rl, False, saved)


def acquire_clean_preload(rl, stack, *, cfg: CooperativeConfig | None = None, acquire_cap: int = 160, coin_xy=None):
    """E0a — SOFT-PINNED, servo-held bilateral acquisition. Place the coin at ``coin_xy`` (default: the two-arm-reachable
    tip midpoint; the E1 authority search passes other positions), hold it with a heavy slide-DAMPING pin (not a hard
    kinematic clamp — the clamp is an infinitely stiff wall that traps the tips in a qdot limit cycle and never settles),
    then servo EACH arm's tip-centre distance to its model-read light-preload setpoint (r_coin + r_tip − δ). The servo
    command vanishes at the setpoint, so a bounded, balanced, low-drift co-contact holds stably. NO launch command is
    issued. Leaves the sim at the settled pinned preload and returns the saved damping so a later stage can ``release_pin``
    onto a free coin. The clean-preload sanity gates the frozen ``cfg.clean_*`` thresholds."""
    cfg = cfg or CooperativeConfig()
    place_coin_at(rl, tip_midpoint(rl) if coin_xy is None else coin_xy)
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    reach = {side: _coin_radius(rl) + float(m.geom_size[g][0]) - cfg.preload_depth for side, g in (("left", gl), ("right", gr))}
    arms = (("left", gl, _LEFT_DOF), ("right", gr, _RIGHT_DOF))
    adr = int(rl.inner._disk_x_adr)
    coin0 = d.qpos[adr:adr + 2].copy()
    saved = _pin_coin(rl, True)                                   # SOFT damping pin — coin held, but can yield micro-scale so tips settle

    def _in_band(pen):                                           # pen ≤ 0 (overlap); accept a bounded light preload
        return -cfg.clean_pen_max <= pen <= -cfg.clean_pen_min

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    peak_joint, prev_tau, acquired, dwell = 0.0, None, False, 0
    try:
        contact_of = {"left": lambda mp: mp.left_contact, "right": lambda mp: mp.right_contact}
        for _ in range(acquire_cap + cfg.settle_frames):
            coin = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]
            mpl = rl.inner._planar_metrics
            tc = _tip_contacts(rl)
            target = d.qpos[:4].copy()
            for side, g, dofs in arms:
                inward = _arm_dir(rl, g, dofs, coin, cfg)
                if contact_of[side](mpl):                         # in contact: servo the TRUE penetration to δ (equalises the
                    err = cfg.preload_depth + tc[side][2]         # two normals → balanced preload; con.dist ≤ 0, want = −δ
                else:                                             # not yet touching: close the geometric gap to the setpoint
                    err = float(np.linalg.norm(_tip_xy(rl, g) - coin)) - reach[side]
                gstep = cfg.approach_gain * float(np.clip(err / cfg.servo_scale, -1.0, 1.0))
                for j in dofs:
                    target[j] = d.qpos[j] + gstep * inward[j]
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), target, stack, prev_tau, lo, hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")     # coin held by damping, NOT hard-clamped
            qdot = float(np.max(np.abs(d.qvel[:4])))
            peak_joint = max(peak_joint, qdot)
            mpl = rl.inner._planar_metrics
            tc = _tip_contacts(rl)
            both_in_band = bool(mpl.left_contact and mpl.right_contact and _in_band(tc["left"][2]) and _in_band(tc["right"][2]))
            dwell = dwell + 1 if both_in_band else 0
            if dwell >= cfg.acquire_dwell and qdot < cfg.settle_qdot:
                acquired = True
                break
    finally:
        preload = _preload_sanity(rl, prev_tau, lo, hi, cfg, dwell) if acquired else None
        acq_disp = float(np.linalg.norm(d.qpos[adr:adr + 2] - coin0))
        mujoco.set_mjcb_control(None)
    return {"acquired": acquired, "clean": bool(acquired and preload and preload["clean"]),
            "acquisition_displacement": round(acq_disp, 4), "peak_joint_vel": round(peak_joint, 2), "preload": preload,
            "pin_saved": saved}


def release_only_sanity(rl, stack, saved, *, cfg: CooperativeConfig | None = None, frames: int = 40, retract: bool = False):
    """E0b/E2 — release-only sanity. From the settled preload (call ``acquire_clean_preload`` first), RELEASE the pin
    (restore the coin's damping via ``saved``) and step with NO launch command. A genuinely clean preload must NOT fling
    the coin: report peak coin speed and net displacement. Two release controllers (the analysis's hold/retract-neutral
    distinction): ``retract=False`` HOLDS the tip positions (the tips keep squeezing → stored preload energy can eject the
    coin); ``retract=True`` backs each tip OFF the coin by δ as it releases (relieves the squeeze → isolates whether a
    residue is a hold artifact or a fundamental spring). ``jumped`` = coin moved more than a bounded quiescent tol."""
    cfg = cfg or CooperativeConfig()
    release_pin(rl, saved)                                        # genuine free coin — the pin is OFF for the sanity check
    m, d = rl.inner.model, rl.inner.data
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    target = d.qpos[:4].copy()
    if retract:                                                   # back each tip OFF the coin (relieve the squeeze)
        coin = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]
        for g, dofs in ((gl, _LEFT_DOF), (gr, _RIGHT_DOF)):
            inward = _arm_dir(rl, g, dofs, coin, cfg)
            for j in dofs:
                target[j] = d.qpos[j] - 2.0 * cfg.preload_depth * inward[j]   # outward = −inward
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    peak_speed, prev_tau = 0.0, None
    try:
        for _ in range(frames):
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), target, stack, prev_tau, lo, hi)  # hold/retract, no launch
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")                # coin FREE (no clamp) — genuine release
            peak_speed = max(peak_speed, float(np.linalg.norm(rl.inner._planar_metrics.disk_vel[:2])))
    finally:
        mujoco.set_mjcb_control(None)
    disp = float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0))
    return {"coin_peak_speed": round(peak_speed, 3), "coin_displacement": round(disp, 4), "retract": retract,
            "jumped": bool(peak_speed > cfg.settle_qdot or disp > 0.01)}


def measure_release_branch(rl, stack, saved, *, cfg: CooperativeConfig | None = None, allocator=None, horizon: int = 40):
    """E2B — measure ONE release branch from a settled preload over a SHORT fixed horizon. Release the pin, then either HOLD
    (``allocator=None`` → the passive-drift baseline P) or apply the launch allocation (``TwistAllocator`` A0 / ``GraspAllocator``
    A2). All three branches share this identical code path — bit-identical but for the allocator — and the horizon is kept
    SHORT so the comparison is made BEFORE the P/A0/A2 trajectories diverge into different contact modes (baseline
    subtraction is not literally linear afterwards; the incremental = branch − P isolates the allocator's added wrench only
    while the contact topology is still shared). Reports target-directed / lateral / spin coin velocity (peak + final),
    motion-contract and torque-saturation over the window."""
    cfg = cfg or CooperativeConfig()
    release_pin(rl, saved)
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    e_cross = np.array([-e_par[1], e_par[0]], np.float32)
    v_target = float(min(0.8, np.sqrt(max(0.0, 2.0 * cfg.coast_mu * _G * rl._dtz()))))
    hold = d.qpos[:4].copy()
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    adr = int(rl.inner._disk_x_adr)

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    peak_vp, peak_vc, peak_om, peak_joint, sat, prev_tau, final_vp = 0.0, 0.0, 0.0, 0.0, 0, None, 0.0
    try:
        for _ in range(horizon):
            if allocator is None:                                # passive baseline P — HOLD, no launch command
                target = hold
            else:                                                # A0 / A2 — apply the launch allocation
                dq = allocator(rl, e_par, e_cross, v_target)
                target = d.qpos[:4] + cfg.launch_gain * _unit2(dq)
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), target, stack, prev_tau, lo, hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
            cvn = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            final_vp = float(cvn @ e_par)
            peak_vp = max(peak_vp, final_vp)
            peak_vc = max(peak_vc, abs(float(cvn @ e_cross)))
            peak_om = max(peak_om, abs(float(d.qvel[adr + 2])))
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            sat += int(bool(np.any((np.asarray(a) <= lo + 1e-6) | (np.asarray(a) >= hi - 1e-6))))
    finally:
        mujoco.set_mjcb_control(None)
    disp = float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0))
    return {"v_target": round(v_target, 3), "peak_v_parallel": round(peak_vp, 4), "final_v_parallel": round(final_vp, 4),
            "peak_v_cross": round(peak_vc, 4), "peak_omega": round(peak_om, 4), "peak_joint_vel": round(peak_joint, 3),
            "saturation_frac": round(sat / horizon, 3), "motion_contract_pass": bool(peak_joint <= 3.45),
            "coin_displacement": round(disp, 4)}


def balanced_preload_search(rl, stack, *, cfg: CooperativeConfig | None = None):
    """E1 existence oracle — a delivery-blind 1-D search along the fingertip→fingertip axis for a coin position where BOTH
    arms hold a BALANCED light preload. The geometric tip-midpoint is not in general the authority-balanced point (the two
    arms differ in Jacobian conditioning, torque headroom, and reachable normal force); this search asks whether a balanced
    frame EXISTS at all, separately from whether a controller can drive to it. At each candidate the preload is acquired on
    a deep copy (non-destructive, no launch) and scored by force imbalance |Fn_L − Fn_R|/(Fn_L + Fn_R), subject to the clean
    gate AND a minimum TOTAL normal force (so the trivial Fn≈0 touch-without-load point cannot win). Returns every candidate
    plus the most-balanced one that qualifies (or ``None``)."""
    cfg = cfg or CooperativeConfig()
    m = rl.inner.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    mid0 = tip_midpoint(rl).astype(np.float64)
    axis = _unit2((_tip_xy(rl, gr) - _tip_xy(rl, gl)).astype(np.float64))     # left → right; sweeping trades arm authority
    cands, best, best_env, best_saved = [], None, None, None
    for s in np.linspace(-cfg.search_span, cfg.search_span, cfg.search_n):
        coin_xy = (mid0 + float(s) * axis).astype(np.float32)
        env = copy.deepcopy(rl)
        acq = acquire_clean_preload(env, stack, cfg=cfg, coin_xy=coin_xy)     # env left at the settled pinned preload
        pl = acq["preload"] or {}
        fnl, fnr = float(pl.get("fn_left", 0.0)), float(pl.get("fn_right", 0.0))
        total = fnl + fnr
        imb = abs(fnl - fnr) / max(total, 1e-6)
        balanced = bool(acq["clean"] and total >= cfg.preload_min_total and imb <= cfg.imbalance_max)
        cand = {"s": round(float(s), 4), "coin_xy": [round(float(x), 4) for x in coin_xy], "acquired": acq["acquired"],
                "clean": acq["clean"], "fn_left": round(fnl, 3), "fn_right": round(fnr, 3), "total_fn": round(total, 3),
                "imbalance": round(imb, 3), "penetration_left": pl.get("penetration_left"),
                "penetration_right": pl.get("penetration_right"), "qdot": pl.get("qdot_prerelease"),
                "saturated": pl.get("torque_saturated"), "balanced": balanced}
        cands.append(cand)
        if balanced and (best is None or imb < best["imbalance"]):            # KEEP the validated env for E2 (release from the
            best, best_env, best_saved = cand, env, acq["pin_saved"]          # EXACT state the search validated, not a re-acquire)
    return {"mid": [round(float(x), 4) for x in mid0], "axis": [round(float(x), 4) for x in axis],
            "candidates": cands, "best": best, "exists": bool(best is not None), "_best_env": best_env, "_best_saved": best_saved}


def realized_coin_wrench(rl):
    """G2 — the NET contact wrench [Fx, Fy, τ] the two FINGERTIPS exert on the coin right now (summed from the MuJoCo
    contacts, world/planar frame, torque about the coin centre). ONLY the fingertip–disk contacts count — floor / boundary
    / arm-link reactions are excluded, since it is the two-tip PRELOAD that must be net-null (those other contacts are
    static reactions that do not drive the release drift). A CLEAN preload has both ‖(Fx,Fy)‖ and |τ| small — the coin is
    squeezed with no net push and no net spin. This is the physical preload target; Fn-balance is only a proxy (asymmetric
    contact points can be Fn-balanced yet carry a net tangential force / coin torque via the moment arms). Sign: force ON
    the coin (``mj_contactForce`` returns the force on geom2 along the contact frame)."""
    m, d = rl.inner.model, rl.inner.data
    dj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    tips = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")}
    c = d.qpos[int(rl.inner._disk_x_adr):int(rl.inner._disk_x_adr) + 2]
    net_f, tau, f6 = np.zeros(2), 0.0, np.zeros(6, np.float64)
    for i in range(d.ncon):
        con = d.contact[i]
        pair = {con.geom1, con.geom2}
        if dj not in pair or not (pair & tips):                  # keep only fingertip–disk contacts (the preload)
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = (con.frame.reshape(3, 3).T @ f6[:3])[:2]            # contact-frame force → world
        fw = fw if con.geom2 == dj else -fw                      # force ON the coin
        r = con.pos[:2] - c
        net_f += fw
        tau += float(r[0] * fw[1] - r[1] * fw[0])
    return {"Fx": round(float(net_f[0]), 4), "Fy": round(float(net_f[1]), 4),
            "force_norm": round(float(np.linalg.norm(net_f)), 4), "torque": round(float(tau), 5)}


def launch_feasibility_certificate(rl, cfg: CooperativeConfig | None = None):
    """G3 — is a target-directed launch WRENCH feasible from this contact geometry? Fast pre-filter: the forward-feasibility
    coefficient > 0 (the +e_par direction is inside the friction cone). Full certificate: solve the grasp allocation for a
    forward target and require the realized net force to have F∥ ≥ g3_fpar_min and |F⊥| ≤ g3_cross_max·F∥ with a small
    wrench residual (cone-feasible, low cross). This is the LAUNCH entry condition the acquisition must satisfy."""
    cfg = cfg or CooperativeConfig()
    m = rl.inner.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    c = np.asarray(rl.inner._planar_metrics.disk_pos, np.float64)[:2]
    p_l, p_r = _tip_xy(rl, gl).astype(np.float64), _tip_xy(rl, gr).astype(np.float64)
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float64)
    e_cross = np.array([-e_par[1], e_par[0]], np.float64)
    fa = forward_feasibility(c, p_l, p_r, e_par, cfg.coast_mu)
    _fl, _fr, diag = _grasp_solve(c, p_l, p_r, e_par, cfg.coast_mu, cfg.grasp_lam)
    net = np.asarray(diag["net_force"], np.float64)
    f_par, f_cross = float(net @ e_par), abs(float(net @ e_cross))
    feasible = bool(fa["feasible"] and f_par >= cfg.g3_fpar_min and f_cross <= cfg.g3_cross_max * max(f_par, 1e-9))
    return {"forward_coeff": fa["coeff"], "f_parallel": round(f_par, 4), "f_cross": round(f_cross, 4),
            "wrench_residual": diag["wrench_residual"], "feasible": feasible}


def null_coin_wrench(rl, stack, *, cfg: CooperativeConfig | None = None):
    """E3a nulling phase — an OBJECT-LEVEL feedback controller that drives the realized coin wrench w=[Fx,Fy,τ]→0 on an
    ALREADY-G1-acquired, soft-pinned preload (call ``acquire_clean_preload`` / pass its validated env first). The passive
    servo balances PENETRATION but not the NET WRENCH (LFA: G2 0/72), so this actively nulls it. Three channels separated
    so they do not fight: the RADIAL target stays the δ penetration servo (common-mode → keeps G1 depth); a TANGENTIAL
    slide on each tip cancels the net force's tangential projection and the net torque (sliding along tᵢ changes the
    friction force without changing penetration). Leaves the settled wrench-null preload. Returns whether ‖w‖ reached the
    G2 band, the final wrench, the preload sanity, and the launch-feasibility certificate (G3)."""
    cfg = cfg or CooperativeConfig()
    m, d = rl.inner.model, rl.inner.data
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    reach = {side: _coin_radius(rl) + float(m.geom_size[g][0]) - cfg.preload_depth for side, g in (("left", gl), ("right", gr))}
    arms = (("left", gl, _LEFT_DOF), ("right", gr, _RIGHT_DOF))
    adr = int(rl.inner._disk_x_adr)
    pin_qpos = d.qpos[adr:adr + 3].copy()

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    prev_tau, dwell, nulled = None, 0, False
    try:
        for _ in range(cfg.wrench_null_steps):
            w = realized_coin_wrench(rl)
            f_net = np.array([w["Fx"], w["Fy"]], np.float64)
            tau = w["torque"]
            coin = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].astype(np.float64)
            contact = {"left": bool(rl.inner._planar_metrics.left_contact), "right": bool(rl.inner._planar_metrics.right_contact)}
            target = d.qpos[:4].copy()
            for side, g, dofs in arms:
                n_out = _unit2(_tip_xy(rl, g).astype(np.float64) - coin)   # live outward radial
                t_hat = np.array([-n_out[1], n_out[0]], np.float64)
                # TANGENTIAL wrench-null slide, CLAMPED so it cannot walk the tip off the round coin; zeroed if this tip
                # lost contact (pure radial re-acquire restores G1 before nulling resumes)
                slide = -cfg.wrench_null_kf * float(f_net @ t_hat) - cfg.wrench_null_ktau * tau
                slide = float(np.clip(slide, -cfg.wrench_null_slide_max, cfg.wrench_null_slide_max)) if contact[side] else 0.0
                target_pt = (coin + reach[side] * n_out + slide * t_hat).astype(np.float32)
                dq = _arm_dir(rl, g, dofs, target_pt, cfg)
                for j in dofs:
                    target[j] = d.qpos[j] + cfg.wrench_null_gain * dq[j]
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), target, stack, prev_tau, lo, hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
            d.qpos[adr:adr + 3] = pin_qpos                        # keep the coin held while nulling
            d.qvel[adr:adr + 3] = 0.0
            mujoco.mj_forward(m, d)
            mpl = rl.inner._planar_metrics
            in_band = bool(w["force_norm"] <= cfg.g2_force_max and abs(w["torque"]) <= cfg.g2_torque_max
                           and mpl.left_contact and mpl.right_contact)
            dwell = dwell + 1 if in_band else 0
            if dwell >= cfg.wrench_null_dwell:
                nulled = True
                break
    finally:
        preload = _preload_sanity(rl, prev_tau, lo, hi, cfg, dwell)
        cert = launch_feasibility_certificate(rl, cfg)
        wr = realized_coin_wrench(rl)
        mujoco.set_mjcb_control(None)
    return {"wrench_nulled": nulled, "preload": preload, "realized_wrench": wr, "launch_cert": cert,
            "done": bool(nulled and cert["feasible"] and preload and preload["clean"])}


def active_wrench_null_acquire(rl, stack, *, cfg: CooperativeConfig | None = None, coin_xy=None):
    """E3a standalone — acquire a G1 clean preload at ``coin_xy`` then actively null the coin wrench (``null_coin_wrench``).
    The terminal state is the E3 acquisition whose next option (LAUNCH) is executable. Returns the nulling result + the pin
    handle so a downstream release / launch can proceed on the same env."""
    cfg = cfg or CooperativeConfig()
    acq = acquire_clean_preload(rl, stack, cfg=cfg, coin_xy=coin_xy)
    if not acq["acquired"]:
        return {"acquired": False, "g1_clean": False, "wrench_nulled": False, "preload": acq["preload"],
                "realized_wrench": None, "launch_cert": None, "done": False, "pin_saved": acq["pin_saved"]}
    out = null_coin_wrench(rl, stack, cfg=cfg)
    out.update({"acquired": True, "g1_clean": bool(acq["clean"]), "pin_saved": acq["pin_saved"]})
    return out


def launch_feasible_acquisition_search(rl, stack, *, cfg: CooperativeConfig | None = None):
    """LAUNCH_FEASIBLE_ACQUISITION_SEARCH_V1 — the corrected acquisition: sweep the fingertip axis and apply the
    LEXICOGRAPHIC option-postcondition at each candidate, so ``done`` means the next option (LAUNCH) is executable — not
    merely that the contact is Fn-balanced (E1's mis-abstraction). G1 clean contact (``acquire_clean_preload``) → G2 null
    realized preload wrench (``realized_coin_wrench``) → G3 launch-wrench feasibility (``launch_feasibility_certificate``).
    The far-side sign is only an ORDERING PRIOR; the decision is G1∧G2∧G3. Returns per-candidate gate breakdown + the best
    (lowest-residual) candidate passing all three, with its validated env for a downstream launch."""
    cfg = cfg or CooperativeConfig()
    m = rl.inner.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    mid0 = tip_midpoint(rl).astype(np.float64)
    axis = _unit2((_tip_xy(rl, gr) - _tip_xy(rl, gl)).astype(np.float64))
    cands, best, best_env, best_saved = [], None, None, None
    for s in np.linspace(-cfg.search_span, cfg.search_span, cfg.search_n):
        coin_xy = (mid0 + float(s) * axis).astype(np.float32)
        env = copy.deepcopy(rl)
        acq = acquire_clean_preload(env, stack, cfg=cfg, coin_xy=coin_xy)
        g1 = bool(acq["clean"])
        w = realized_coin_wrench(env) if acq["acquired"] else {"force_norm": None, "torque": None}
        g2 = bool(g1 and w["force_norm"] is not None and w["force_norm"] <= cfg.g2_force_max and abs(w["torque"]) <= cfg.g2_torque_max)
        cert = launch_feasibility_certificate(env, cfg) if acq["acquired"] else {"feasible": False, "wrench_residual": None}
        g3 = bool(g1 and cert["feasible"])
        done = bool(g1 and g2 and g3)
        cand = {"s": round(float(s), 4), "coin_xy": [round(float(x), 4) for x in coin_xy], "acquired": acq["acquired"],
                "G1_clean_contact": g1, "G2_null_preload_wrench": g2, "G3_launch_feasible": g3, "done": done,
                "realized_wrench": w, "launch_cert": cert}
        cands.append(cand)
        if done and (best is None or cert["wrench_residual"] < best["launch_cert"]["wrench_residual"]):
            best, best_env, best_saved = cand, env, acq["pin_saved"]
    return {"candidates": cands, "best": best, "exists": bool(best is not None), "_best_env": best_env, "_best_saved": best_saved}


def _tip_contacts(rl):
    """Per-tip peak (normal force, tangential force, min penetration depth) vs the coin THIS step (MuJoCo contact frame).
    Feeds the E0 pre-release sanity gate: the positive launch velocity must come from a BOUNDED preload, not from a
    spring-load explosion accumulated against the hard pin (large |penetration| + saturated torque)."""
    m, d = rl.inner.model, rl.inner.data
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    dj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    acc = {gl: [0.0, 0.0, 0.0], gr: [0.0, 0.0, 0.0]}
    f = np.zeros(6, np.float64)
    for ci in range(d.ncon):
        con = d.contact[ci]
        pair = (con.geom1, con.geom2)
        if dj not in pair:
            continue
        other = pair[0] if pair[1] == dj else pair[1]
        if other not in acc:
            continue
        mujoco.mj_contactForce(m, d, ci, f)
        acc[other][0] = max(acc[other][0], abs(float(f[0])))
        acc[other][1] = max(acc[other][1], float(np.hypot(f[1], f[2])))
        acc[other][2] = min(acc[other][2], float(con.dist))
    return {"left": tuple(acc[gl]), "right": tuple(acc[gr])}


def _contact_frames(c, p_l, p_r):
    """The FROZEN sign convention for both contacts: normal nᵢ = unit(coin_centre − tip) — the direction tip i pushes INTO
    the coin (Fn ≥ 0 presses along +nᵢ); tangent tᵢ = nᵢ rotated +90°. Returned per contact as (r=tip−centre, n, t)."""
    out = []
    for p in (p_l, p_r):
        r = (np.asarray(p) - np.asarray(c)).astype(np.float64)
        n = _unit2((np.asarray(c) - np.asarray(p)).astype(np.float64))
        out.append((r, n, np.array([-n[1], n[0]], np.float64)))
    return out


def _grasp_solve(c, p_l, p_r, e_par, mu, lam, f_par=1.0):
    """A2 resultant-FORCE allocation with full diagnostics. Grasp matrix G maps [Fn_L,Ft_L,Fn_R,Ft_R] to the coin wrench
    [Fx,Fy,τ]; solve min‖G f − w*‖² + λ‖f‖², w* = [f_par·e_par ; 0], then project onto the friction cone Fn ≥ 0 ∧
    |Ft| ≤ μ·Fn. Returns the two WORLD contact forces AND a diagnostic dict (unclipped/clipped solution, realized net force,
    realized forward force e_par·F_net, wrench residual, whether the cone clip changed the solution) — so a zero output can
    be classified as an HONEST refusal (infeasible wrench) versus a solver/scale/clip artifact."""
    e = np.asarray(e_par, np.float64)
    frames = _contact_frames(c, p_l, p_r)
    cols = []
    for r, n, t in frames:
        cols.append([n[0], n[1], r[0] * n[1] - r[1] * n[0]])
        cols.append([t[0], t[1], r[0] * t[1] - r[1] * t[0]])
    G = np.array(cols, np.float64).T
    w = np.array([f_par * e[0], f_par * e[1], 0.0], np.float64)
    f_unclipped = np.linalg.solve(G.T @ G + lam * np.eye(4), G.T @ w)
    f_clipped, forces = f_unclipped.copy(), []
    for k, (_r, n, t) in zip((0, 2), frames):
        fn = max(0.0, float(f_unclipped[k]))
        ft = float(np.clip(f_unclipped[k + 1], -mu * fn, mu * fn))
        f_clipped[k], f_clipped[k + 1] = fn, ft
        forces.append(fn * n + ft * t)
    net = forces[0] + forces[1]
    diag = {"f_unclipped": [round(float(x), 4) for x in f_unclipped], "f_clipped": [round(float(x), 4) for x in f_clipped],
            "net_force": [round(float(x), 4) for x in net], "forward_force": round(float(e @ net), 4),
            "wrench_residual": round(float(np.linalg.norm(G @ f_clipped - w)), 4),
            "clip_changed": bool(not np.allclose(f_unclipped, f_clipped, atol=1e-9))}
    return forces[0], forces[1], diag


def _grasp_allocation(c, p_l, p_r, e_par, mu, lam, f_par=1.0):
    """Thin wrapper over ``_grasp_solve`` returning just the two WORLD contact forces (the A2 launch command path)."""
    f_l, f_r, _ = _grasp_solve(c, p_l, p_r, e_par, mu, lam, f_par)
    return f_l, f_r


def forward_feasibility(c, p_l, p_r, e_par, mu, fn_max=None):
    """The UNIT-NORMALISED forward-feasibility coefficient — a SIGN gate, NOT a physical force magnitude. With a unit
    per-contact normal cap (Fn ≤ 1), the friction cone's max forward projection has the closed form
    ``coeff = Σᵢ max(0, e_par·nᵢ + μ|e_par·tᵢ|)`` (contacts independent in the objective). Its ROLE is directional
    feasibility: ``coeff == 0`` ⇒ the contact geometry admits NO forward-pushing compressive direction (a grasp allocator
    returning zero is then a physically HONEST refusal); ``coeff > 0`` ⇒ the forward direction lies inside the friction
    cone (feasible). Per-contact terms let a MIXED pair be feasible via friction — so the gate is feasibility, not the
    hardcoded "both tips far-side" sign. It does NOT say how much force the ACTUATORS can produce: pass ``fn_max`` (the
    per-contact realizable normal force from torque/Jacobian/motion-contract) to also get the bounded physical magnitude
    ``bounded = Σᵢ Fn_maxᵢ·max(0, …)``. Without ``fn_max`` the magnitude is unbounded, hence the unit normalisation."""
    e = np.asarray(e_par, np.float64)
    e = e / (float(np.linalg.norm(e)) + 1e-12)
    fn_max = fn_max if fn_max is not None else (1.0, 1.0)
    per, bounded = [], 0.0
    for (_r, n, t), fmax in zip(_contact_frames(c, p_l, p_r), fn_max):
        support = max(0.0, float(e @ n) + mu * abs(float(e @ t)))
        per.append(support)
        bounded += float(fmax) * support
    return {"coeff": round(sum(per), 4), "per_contact": [round(x, 4) for x in per], "feasible": bool(sum(per) > 1e-6),
            "bounded_forward_force": round(bounded, 4)}


class TwistAllocator:
    """A0 baseline (the 0/8 reference): coin-twist Jacobian → desired object twist [v_target·e_par ; 0], DLS solve with a
    zero-spin weight. Resultant-TWIST allocation (velocity level)."""

    def __init__(self, cfg: CooperativeConfig):
        self.cfg = cfg
        self._J = None
        self._t = 0

    def __call__(self, rl, e_par, e_cross, v_target):
        cfg = self.cfg
        if self._J is None or self._t % cfg.replan_every == 0:
            self._J = _coin_twist_jacobian(rl, cfg)
        self._t += 1
        w = np.diag([1.0, 1.0, cfg.w_omega])
        desired = np.array([v_target * e_par[0], v_target * e_par[1], 0.0], np.float64) - _coin_twist(rl)
        return self._J.T @ np.linalg.solve((w @ self._J) @ self._J.T + cfg.lam ** 2 * np.eye(3), w @ desired)


class GraspAllocator:
    """A2 resultant-FORCE allocation (force level). Build the grasp matrix from the two live contact points, solve for the
    two contact forces that produce a zero-torque resultant along e_par (``_grasp_allocation``), then drive each tip ALONG
    its allocated world force — scaled by the relative force magnitude so the A1 normal balance is realised on the arms."""

    def __init__(self, cfg: CooperativeConfig):
        self.cfg = cfg

    def __call__(self, rl, e_par, e_cross, v_target):
        cfg = self.cfg
        m = rl.inner.model
        gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
        gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
        c = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].astype(np.float64)
        p_l, p_r = _tip_xy(rl, gl).astype(np.float64), _tip_xy(rl, gr).astype(np.float64)
        f_l, f_r = _grasp_allocation(c, p_l, p_r, np.asarray(e_par, np.float64), cfg.coast_mu, cfg.grasp_lam)
        fmax = max(float(np.linalg.norm(f_l)), float(np.linalg.norm(f_r)), 1e-6)
        dl = (float(np.linalg.norm(f_l)) / fmax) * _arm_dir(rl, gl, _LEFT_DOF, (p_l + cfg.coin_radius * _unit2(f_l)).astype(np.float32), cfg)
        dr = (float(np.linalg.norm(f_r)) / fmax) * _arm_dir(rl, gr, _RIGHT_DOF, (p_r + cfg.coin_radius * _unit2(f_r)).astype(np.float32), cfg)
        return (dl + dr).astype(np.float64)


def cooperative_launch_carry(rl, gate, pi0, base, stack, *, horizon: int, cfg: CooperativeConfig | None = None,
                             allocator=None):
    """Decoupled synchronized close (both arms → coin centre) then a cooperative RESULTANT allocation (Strategy:
    ``TwistAllocator`` A0 by default, or ``GraspAllocator`` A2). Returns launch quality + both-tips contact/simultaneity +
    force-line-miss ω. K6/zone are outputs."""
    cfg = cfg or CooperativeConfig()
    allocator = allocator or TwistAllocator(cfg)
    m, d = rl.inner.model, rl.inner.data
    m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
    lo, hi = m.actuator_ctrlrange[:4, 0].copy(), m.actuator_ctrlrange[:4, 1].copy()
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    u, _dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float32)
    e_cross = np.array([-e_par[1], e_par[0]], np.float32)

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], stack.gov)
    mujoco.set_mjcb_control(_gcb)
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    peak_vp, peak_vc, peak_om, peak_joint, both_frames, both_seen, prev_tau = 0.0, 0.0, 0.0, 0.0, 0, 0, None
    md, touched, t = int(rl._strict), rl._touched, 0
    try:
        for t in range(horizon):
            mpl = rl.inner._planar_metrics
            coin = np.asarray(mpl.disk_pos, np.float32)[:2]
            lc, rcn = bool(mpl.left_contact), bool(mpl.right_contact)
            dtz = rl._dtz()
            v_target = float(min(0.8, np.sqrt(max(0.0, 2.0 * cfg.coast_mu * _G * dtz))))
            if gate is not None and gate.gate != 1.0:
                a = _det(pi0, rl.obs())
            elif not (lc and rcn):                                # synchronized decoupled close: BOTH arms → coin centre
                dl = _arm_dir(rl, gl, _LEFT_DOF, coin, cfg)
                dr = _arm_dir(rl, gr, _RIGHT_DOF, coin, cfg)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + cfg.close_gain * (dl + dr), stack, prev_tau, lo, hi)
                prev_tau = a
            else:                                                 # cooperative resultant allocation (A0 twist / A2 grasp)
                both_frames += 1
                both_seen = 1
                dq = allocator(rl, e_par, e_cross, v_target)
                a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), d.qpos[:4] + cfg.launch_gain * _unit2(dq), stack, prev_tau, lo, hi)
                prev_tau = a
            _r, term, trunc = step_ablation(rl, np.asarray(a, np.float32), "A")
            if gate is not None:
                l2, r2c, coin_s, ltp, rtp = stable_engagement_signals(rl.inner)
                gate.update(l2, r2c, coin_s, ltp, rtp, terminated=bool(term))
            md = max(md, int(rl._strict))
            touched = touched or rl._touched
            cvn = np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]
            peak_vp = max(peak_vp, float(cvn @ e_par))
            peak_vc = max(peak_vc, abs(float(cvn @ e_cross)))
            peak_om = max(peak_om, abs(float(rl.inner.data.qvel[int(rl.inner._disk_x_adr) + 2])))
            peak_joint = max(peak_joint, float(np.max(np.abs(d.qvel[:4]))))
            if term or trunc:
                break
    finally:
        mujoco.set_mjcb_control(None)
    disp = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0
    return {"peak_v_parallel": round(peak_vp, 3), "peak_v_cross": round(peak_vc, 3),
            "cross_ratio": round(peak_vc / max(1e-6, peak_vp), 3), "peak_omega": round(peak_om, 3),
            "both_tips_contact": both_seen, "both_contact_frames": both_frames,
            "signed_target_displacement": round(float(disp @ e_par), 4), "peak_joint_vel": round(peak_joint, 2),
            "k6": int(md >= HELD_DWELL and touched), "steps": t + 1}
