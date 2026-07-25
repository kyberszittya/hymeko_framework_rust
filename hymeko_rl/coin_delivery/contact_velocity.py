"""H2 — CONTROL-TO-CONTACT-VELOCITY response Jacobian B_v: identification & validation (NOT the H2 QP).

    B_v = ∂ v_rel,t+1 / ∂ Δq_target  ∈ R^{4×4}

maps a joint POSITION-SERVO target perturbation Δq_target ∈ R^4 to the two contacts' frozen-frame relative velocity one
control step later, [v_n,L, v_t,L, v_n,R, v_t,R]. This is the GOVERNED-SERVO response — it runs through the full shared
low-level stack (PD position servo → torque-rate limit → per-sub-step directional velocity governor → actuator saturation
→ MuJoCo contact solver → FREE-coin motion at the contacts), NOT the geometric fingertip Jacobian J_tip(q). The coin is a
free body held only by the two tip contacts (the pin is released), so the contact-solver response and coin motion at the
contacts are real; a stiff pin would suppress them and collapse B_v onto J_tip (the guard the whole exercise turns on).

FROZEN sign / kinematic conventions (see docs/plans/2026-07-26-h2-bv-identification/plan.md §1):
  * v_rel := velocity of the TIP relative to the COIN, evaluated at the COMMON MuJoCo contact point x_c (NOT a geom
    origin): v_rel = v_tip(x_c) − v_coin(x_c), each body's rigid-body velocity transported to x_c. Positive v_n =
    CLOSING / compression (tip approaching the coin along the inward normal).
  * The projection frame (n0_i, t0_i) is the ACTUAL MuJoCo contact normal at the snapshot t (oriented tip→coin so v_n>0
    is closing), FROZEN; every branch projects its live t+1 relative velocity onto that SAME frame, isolating the
    velocity response from frame rotation in the finite difference.
  * Ordering [v_n,L, v_t,L, v_n,R, v_t,R].

Layering (preferred paradigm hierarchy): PURE kinematic transport (no MuJoCo, unit-testable) → THIN MuJoCo readers over
the ACTUAL contacts (fresh state, never the cached _planar_metrics) → IDENTIFICATION (exact deepcopy branching, the TRUE
acquisition controller state, central FD, contact-mode-identity validity).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.cooperative_launch import internal_force_feasibility, release_pin
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque

_TIP_LEFT, _TIP_RIGHT, _DISK = "fingertip_left", "fingertip_right", "disk"


# --------------------------------------------------------------------------------------------------------------------
# PURE kinematic transport (no MuJoCo)
# --------------------------------------------------------------------------------------------------------------------
def planar_point_velocity(v_origin, omega, origin, x_point) -> np.ndarray:
    """Rigid-body velocity transport in the plane: the velocity of the material point at ``x_point`` on a body whose
    frame origin moves at ``v_origin`` and spins at ``omega`` is ``v_origin + ω·ẑ×(x_point − origin)``, ẑ×r=[−r_y, r_x].

    # Preconditions: v_origin, origin, x_point length-2; omega scalar. # Postconditions: length-2 float64.
    """
    r = np.asarray(x_point, np.float64) - np.asarray(origin, np.float64)
    return np.asarray(v_origin, np.float64) + float(omega) * np.array([-r[1], r[0]], np.float64)


def contact_relative_velocity(v_tip, v_center, omega, p, c) -> np.ndarray:
    """Contact-relative velocity of the TIP w.r.t. the coin at contact point ``p`` (coin transported from its centre):
    ``v_rel = v_tip − (v_center + ω·ẑ×(p − c))``. Convenience wrapper over ``planar_point_velocity`` (coin-only transport,
    point tip); the full path transports BOTH bodies to the common contact point.

    # Postconditions: v_tip == coin-surface-velocity ⇒ v_rel == 0.
    """
    return np.asarray(v_tip, np.float64) - planar_point_velocity(v_center, omega, c, p)


def resolve_nt(v_rel, n, t) -> tuple[float, float]:
    """Resolve a planar relative velocity into the (fixed) contact frame: ``(v_n, v_t) = (nᵀ v_rel, tᵀ v_rel)``.
    n is oriented tip→coin, so v_n>0 == closing/compression; t = R₊₉₀ n.
    """
    v = np.asarray(v_rel, np.float64)
    return float(np.asarray(n, np.float64) @ v), float(np.asarray(t, np.float64) @ v)


# --------------------------------------------------------------------------------------------------------------------
# THIN MuJoCo readers — fresh state over the ACTUAL contacts (never the cached planar metrics)
# --------------------------------------------------------------------------------------------------------------------
def _geom_ids(m):
    return (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, _TIP_LEFT),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, _TIP_RIGHT),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, _DISK))


def geom_spatial_velocity(m, d, gid) -> tuple[float, np.ndarray]:
    """Planar spatial velocity of geom ``gid`` at its frame origin, world orientation, via ``mj_objectVelocity``:
    returns ``(ω_z, v_xy)`` (res[2] is the z-angular rate; res[3:5] the linear x,y).

    # Preconditions: gid ≥ 0. # Postconditions: ω_z float, v_xy length-2.
    """
    res = np.zeros(6, np.float64)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(gid), res, 0)
    return float(res[2]), res[3:5].copy()


def geom_planar_velocity(m, d, gid) -> np.ndarray:
    """Planar (x,y) WORLD velocity of geom ``gid`` at its frame origin (linear part only). Per-geom generalisation of
    ``force_slip_carry._tip_world_velocity`` (which averages over both tips)."""
    return geom_spatial_velocity(m, d, gid)[1]


def geom_point_velocity(m, d, gid, x_point) -> np.ndarray:
    """The velocity of the material point ``x_point`` on geom ``gid``'s body (rigid-body transport of the geom's spatial
    velocity to ``x_point``). Used to evaluate both bodies at the COMMON contact point."""
    omega, v_origin = geom_spatial_velocity(m, d, gid)
    return planar_point_velocity(v_origin, omega, d.geom_xpos[int(gid)][:2], x_point)


def coin_twist(m, d, disk_adr, disk_gid) -> tuple[np.ndarray, float]:
    """Planar coin twist ``(v_center, ω)``: v_center = disk-geom world velocity; ω = coin hinge dof ``qvel[disk_adr+2]``
    (cross-checked against ``geom_spatial_velocity`` res[2] in the reader unit test)."""
    return geom_planar_velocity(m, d, disk_gid), float(d.qvel[int(disk_adr) + 2])


def primary_fingertip_contacts(rl) -> dict:
    """Per side, the PRIMARY (deepest) fingertip–disk MuJoCo contact: its identity (the sorted geom pair + the tip geom),
    common contact point ``x_c`` (``contact.pos``), inward normal ``n`` (the ACTUAL contact normal ``contact.frame[:3]``,
    oriented tip→coin so v_n>0 = closing), tangent ``t = R₊₉₀ n``, penetration ``dist`` and normal force ``fn``. Returns
    ``{"left": rec|None, "right": rec|None}`` — ``None`` when that tip has no disk contact this step.

    # Postconditions: uses the real contact point/normal, NOT a geom origin (the 20 mm tip radius makes the ω×r transport
    otherwise ~2× wrong).
    """
    m, d = rl.inner.model, rl.inner.data
    gl, gr, dj = _geom_ids(m)
    side_of = {gl: "left", gr: "right"}
    c = d.geom_xpos[dj][:2].astype(np.float64)
    best = {"left": None, "right": None}
    count = {"left": 0, "right": 0}
    f6 = np.zeros(6, np.float64)
    for i in range(d.ncon):
        con = d.contact[i]
        pair = (int(con.geom1), int(con.geom2))
        if dj not in pair:
            continue
        other = pair[0] if pair[1] == dj else pair[1]
        if other not in side_of:
            continue
        side = side_of[other]
        count[side] += 1
        if best[side] is not None and con.dist >= best[side]["dist"]:
            continue
        x_c = con.pos[:2].astype(np.float64)
        n = con.frame[:3].astype(np.float64)[:2].copy()          # world contact normal (contact.frame row 0)
        n = n / (np.linalg.norm(n) + 1e-12)
        if n @ (c - x_c) < 0.0:                                  # orient tip→coin (v_n>0 == closing/compression)
            n = -n
        mujoco.mj_contactForce(m, d, i, f6)
        best[side] = {"pair": tuple(sorted(pair)), "geom_tip": int(other), "x_c": x_c, "n": n,
                      "t": np.array([-n[1], n[0]], np.float64), "dist": float(con.dist), "fn": abs(float(f6[0]))}
    best["count_left"], best["count_right"] = count["left"], count["right"]
    return best


def state_hash(rl) -> str:
    """A short content hash of the dynamical state (qpos+qvel) — to record that the pin-release + mj_forward is a HYBRID
    RESET MAP (the post-release state differs from the pre-release one, and must not be documented as unchanged)."""
    d = rl.inner.data
    h = hashlib.md5()
    h.update(np.ascontiguousarray(d.qpos).tobytes())
    h.update(np.ascontiguousarray(d.qvel).tobytes())
    return h.hexdigest()[:16]


def recertify_cradle(rl, mu, cfg: "BvConfig") -> dict:
    """Re-certify the ACTUAL current state as an admissible dual-contact straddle cradle. Applied to the POST-RELEASE
    handoff: pin-release + mj_forward is a hybrid reset, so the pinned certificate is NOT inherited — dual contact,
    straddle, internal-force certificate, and a joint-speed bound are all re-checked on the post-release state."""
    con = primary_fingertip_contacts(rl)
    dual = bool(con["left"] is not None and con["right"] is not None)
    m, d = rl.inner.model, rl.inner.data
    gl, gr, dj = _geom_ids(m)
    if dual:
        c = d.geom_xpos[dj][:2].astype(np.float64)
        p_l, p_r = d.geom_xpos[gl][:2].astype(np.float64), d.geom_xpos[gr][:2].astype(np.float64)
        cert_feasible = bool(mu is not None and internal_force_feasibility(c, p_l, p_r, mu)["feasible"])
        straddle = float(con["left"]["n"] @ con["right"]["n"])
    else:
        cert_feasible, straddle = False, 1.0
    qdot = float(np.max(np.abs(d.qvel[:4])))
    return {"dual_contact": dual, "straddle_dot": round(straddle, 4), "straddle": bool(straddle < 0.0),
            "cert_feasible": cert_feasible, "qdot_max": round(qdot, 4), "qdot_ok": bool(qdot <= cfg.recert_qdot_max),
            "admissible": bool(dual and straddle < 0.0)}


def frozen_contact_frames(rl) -> dict:
    """The FROZEN projection frames + contact identities taken at the snapshot t from the ACTUAL contacts.

    # Preconditions: both tips are in disk contact (a certified straddle cradle). # Raises: ValueError if a tip has no
    disk contact at the snapshot (an inadmissible source state — must be gated out upstream).
    """
    con = primary_fingertip_contacts(rl)
    if con["left"] is None or con["right"] is None:
        raise ValueError("snapshot is not a dual-contact cradle (a tip has no disk contact)")
    m, d = rl.inner.model, rl.inner.data
    _gl, _gr, dj = _geom_ids(m)
    return {"left": con["left"], "right": con["right"], "c0": d.geom_xpos[dj][:2].astype(np.float64),
            "straddle0": float(con["left"]["n"] @ con["right"]["n"]),
            "count_left": con["count_left"], "count_right": con["count_right"]}


def measure_contact_velocities(rl, frozen) -> dict:
    """Per-tip contact-relative velocity at the COMMON contact point, resolved in the FROZEN frame. For each side, find
    the same-identity fingertip–disk contact at the CURRENT (t+1) state, transport BOTH the tip body and the coin body
    to the live contact point ``x_c``, form ``v_rel = v_tip(x_c) − v_coin(x_c)``, and project on the frozen ``(n0, t0)``.
    Reports the live contact-mode signature for the validity gate (present, same identity, fn, straddle sign).

    # Preconditions: ``frozen`` from ``frozen_contact_frames`` at the snapshot. # Postconditions: ``vrel4`` =
    [v_n,L, v_t,L, v_n,R, v_t,R]; entries for a lost side are best-effort (frozen x_c) and flagged not-present.
    """
    m, d = rl.inner.model, rl.inner.data
    gl, gr, dj = _geom_ids(m)
    live = primary_fingertip_contacts(rl)
    counts = {"left": live["count_left"], "right": live["count_right"]}
    per, comps, n_live = {}, {}, {}
    for side, g_tip, fr in (("left", gl, frozen["left"]), ("right", gr, frozen["right"])):
        lc = live[side]
        present = bool(lc is not None)
        same_identity = bool(present and lc["pair"] == fr["pair"])
        x_c = lc["x_c"] if present else fr["x_c"]                 # live common contact point (fallback: frozen)
        n_l = lc["n"] if present else fr["n"]
        xc_drift = float(np.linalg.norm(x_c - fr["x_c"]))
        cosang = float(np.clip(n_l @ fr["n"], -1.0, 1.0))
        n_angle = float(np.degrees(np.arccos(cosang)))           # contact-normal rotation over the step
        v_rel = geom_point_velocity(m, d, g_tip, x_c) - geom_point_velocity(m, d, dj, x_c)
        v_n, v_t = resolve_nt(v_rel, fr["n"], fr["t"])
        per[side] = {"v_rel": v_rel, "v_n": v_n, "v_t": v_t, "present": present, "same_identity": same_identity,
                     "fn": (lc["fn"] if present else 0.0), "xc_drift": xc_drift, "n_angle_deg": n_angle,
                     "count": counts[side]}
        comps[side] = (v_n, v_t)
        n_live[side] = n_l
    vrel4 = np.array([comps["left"][0], comps["left"][1], comps["right"][0], comps["right"][1]], np.float64)
    straddle_live = float(n_live["left"] @ n_live["right"])
    return {"per": per, "vrel4": vrel4, "fn_left": per["left"]["fn"], "fn_right": per["right"]["fn"],
            "straddle_live": straddle_live}


# --------------------------------------------------------------------------------------------------------------------
# IDENTIFICATION — exact-state finite differences of the governed one-step response
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BvConfig:
    """Frozen identification/validation parameters (see plan §1)."""

    eps_scales: tuple = (0.0015, 0.003)          # rad — two FD perturbation scales (central difference)
    fn_floor: float = 0.05                       # N — a tip below this after the step ⇒ contact LOST ⇒ branch invalid
    dq_levels: tuple = (0.002, 0.006, 0.012)     # rad — held-out |Δq| per component: small / medium / near-boundary
    n_val_dirs: int = 8                          # random unit directions per validation level
    joint_vel_hard: float = 3.0                  # rad/s — motion-contract hard cap (post-step max|qvel| proxy)
    val_seed: int = 20260726
    cos_ok: float = 0.9                          # cosine agreement threshold for "linearisation valid"
    rel_ok: float = 0.35                         # relative-error threshold for "linearisation valid"
    # --- contact-mode continuity (a same-geom-pair contact can still SLIDE along the rim → a different local branch) ---
    xc_drift_max: float = 0.004                  # m — max common-contact-point drift over the step (else mode changed)
    normal_angle_max_deg: float = 20.0           # deg — max contact-normal rotation over the step
    recert_qdot_max: float = 1.2                 # rad/s — post-release cradle joint-speed bound for re-certification


class InvalidCradleSnapshot(ValueError):
    """The env handed to ``CradleSnapshot`` is not an admissible dual-contact straddle cradle with an explicit handoff
    controller state — so it must be SKIPPED, never carried forward as four invalid FD columns."""


def _assert_admissible_frames(frames: dict, straddle0: float) -> None:
    """Raise InvalidCradleSnapshot unless the POST-RELEASE frames are a straddling, finite, unambiguous dual contact."""
    if straddle0 >= 0.0:
        raise InvalidCradleSnapshot(f"snapshot is not straddling (n_L·n_R = {straddle0:.3f} ≥ 0)")
    if not (np.all(np.isfinite(frames["left"]["n"])) and np.all(np.isfinite(frames["right"]["n"]))):
        raise InvalidCradleSnapshot("contact frames are not finite")
    if frames["count_left"] != 1 or frames["count_right"] != 1:
        raise InvalidCradleSnapshot(f"ambiguous primary contact at snapshot "
                                    f"(counts {frames['count_left']}/{frames['count_right']})")


class CradleSnapshot:
    """A frozen cradle state for B_v identification with EXACT deepcopy branching, built from the EXPLICIT acquisition
    HANDOFF state — no re-synthesis, no settle. Holds a deepcopy of the env (pin released for the free-coin measurement),
    the TRUE handoff controller state (``prev_tau`` = the acquisition's last applied rate-limited torque; ``q_hold`` =
    the acquisition's last servo target), the FROZEN contact frames + identities (ACTUAL MuJoCo contacts), and the V3
    stack. Every branch restores this identical state; the stored env is NEVER stepped in place.

    Two operating points are NOT mixed: a re-synthesised ``−kv·q̇`` seed spuriously saturates the actuator (fabricating
    dead columns), and a passive settle DROPS a tip (PASSIVE_CRADLE_VIABILITY=FAIL — the H2 problem itself). So the
    operating point is EXACTLY the acquisition handoff, with the controller state exported from acquisition.

    ``release_coin=False`` keeps the soft pin → a PINNED POSITIVE CONTROL: the coin barely moves, so B_v isolates the
    governed tip-velocity FD pipeline (a pipeline check), NOT active-stabilizability evidence.

    # Preconditions: ``rl`` a certified dual-contact straddle cradle; ``prev_tau``/``q_target`` from the acquisition
    ``final_tau``/``final_q_target``; ``saved_damping`` the pin handle. # Raises: InvalidCradleSnapshot if the source is
    not a dual-contact straddle or the controller state is missing.
    """

    def __init__(self, rl, stack, saved_damping, prev_tau, q_target, *, release_coin: bool = True,
                 coast_mu: float | None = None, cfg: "BvConfig | None" = None):
        if prev_tau is None or q_target is None:
            raise InvalidCradleSnapshot("missing explicit handoff controller state (prev_tau / q_target)")
        cfg = cfg or BvConfig()
        self._rl = copy.deepcopy(rl)
        m, d = self._rl.inner.model, self._rl.inner.data
        m.dof_armature[:4], m.dof_damping[:4], m.dof_frictionloss[:4] = stack.armature, stack.damping, stack.friction
        mujoco.mj_forward(m, d)
        self.pre_release_hash = state_hash(self._rl)             # pin-release + mj_forward is a HYBRID RESET MAP:
        self.released = bool(release_coin and saved_damping is not None)
        if self.released:
            release_pin(self._rl, saved_damping)                 # free coin — held only by the two tip contacts (handoff)
        mujoco.mj_forward(m, d)
        self.post_release_hash = state_hash(self._rl)            # …record BOTH states; do not pretend nothing changed
        self.operating_point = "EXACT_POST_RELEASE_HANDOFF" if self.released else "PINNED_POSITIVE_CONTROL"
        self.stack = stack
        self.q_hold = np.asarray(q_target, np.float64).copy()    # the acquisition's last servo target (nominal Δq=0)
        self.lo = m.actuator_ctrlrange[:4, 0].copy()
        self.hi = m.actuator_ctrlrange[:4, 1].copy()
        self.prev_tau = np.asarray(prev_tau, np.float64).copy()
        self.recert = recertify_cradle(self._rl, coast_mu, cfg)  # re-certify the POST-RELEASE state, do NOT inherit
        try:
            self.frames = frozen_contact_frames(self._rl)        # HARD gate: raises if a tip has no disk contact
        except ValueError as e:
            raise InvalidCradleSnapshot(str(e)) from e
        self.straddle0 = self.frames["straddle0"]
        self.fn0 = (float(self.frames["left"]["fn"]), float(self.frames["right"]["fn"]))
        _assert_admissible_frames(self.frames, self.straddle0)
        tol = 1e-4
        self.arm_saturated = [bool(t <= lo + tol or t >= hi - tol) for t, lo, hi in zip(self.prev_tau, self.lo, self.hi)]

    def branch(self):
        """A fresh deepcopy restoring the identical snapshot state (MuJoCo data + warm-start + contacts + pin)."""
        return copy.deepcopy(self._rl)


def _contact_mode_valid(snap: CradleSnapshot, meas: dict, peak_qdot: float, cfg: BvConfig) -> tuple[bool, dict]:
    """A derivative branch is valid only on the SAME smooth dynamical branch. Contact retention alone is insufficient —
    a same-geom-pair contact can SLIDE along the coin rim (identity unchanged, dynamics different), or the "deepest"
    primary contact can swap when a side carries several tip–disk contacts. So the gate requires, per side: presence,
    same geom-pair identity, Fn ≥ floor, common-contact-point drift ‖x_c,t+1 − x_c,0‖ ≤ xc_drift_max, contact-normal
    rotation ≤ normal_angle_max_deg, exactly ONE tip–disk contact (unambiguous primary) — plus the straddle topology
    retained and the motion contract held.
    """
    lft, rgt = meas["per"]["left"], meas["per"]["right"]
    sig = {"present": bool(lft["present"] and rgt["present"]),
           "same_identity": bool(lft["same_identity"] and rgt["same_identity"]),
           "fn_ok": bool(lft["fn"] >= cfg.fn_floor and rgt["fn"] >= cfg.fn_floor),
           "xc_continuous": bool(lft["xc_drift"] <= cfg.xc_drift_max and rgt["xc_drift"] <= cfg.xc_drift_max),
           "normal_continuous": bool(lft["n_angle_deg"] <= cfg.normal_angle_max_deg and rgt["n_angle_deg"] <= cfg.normal_angle_max_deg),
           "unambiguous_primary": bool(lft["count"] == 1 and rgt["count"] == 1),
           "straddle_retained": bool(meas["straddle_live"] < 0.0), "contract_ok": bool(peak_qdot <= cfg.joint_vel_hard)}
    return bool(all(sig.values())), sig


def one_step_vrel(snap: CradleSnapshot, dq_target, cfg: BvConfig) -> tuple[np.ndarray, bool, dict]:
    """Advance a fresh branch EXACTLY one control step under servo target ``q_hold + dq_target`` (governed stack, free
    coin), then measure ``v_rel,t+1`` at the common contact point in the frozen frame. ``valid`` requires the SAME
    contact mode (identity + Fn floor + straddle + motion contract), not mere contact retention; on any violation NO
    derivative is fabricated — the caller drops the column/perturbation.

    # Preconditions: dq_target length-4. # Postconditions: vrel4 length-4; diag carries the mode signature.
    """
    rl = snap.branch()
    d = rl.inner.data
    dq = np.asarray(dq_target, np.float64)
    q0, qd0 = d.qpos[:4].copy(), d.qvel[:4].copy()
    raw_pd = snap.stack.kp * (snap.q_hold + dq - q0) - snap.stack.kv * qd0   # PD torque BEFORE rate-limit/governor
    trace = []                                                   # the TRUE per-sub-step post-governor ctrl (MuJoCo sees this)

    def _gcb(_mo, dt):
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
        trace.append(dt.ctrl[:4].copy())
    mujoco.set_mjcb_control(_gcb)
    try:
        a = pd_governed_torque(q0, qd0, snap.q_hold + dq, snap.stack, snap.prev_tau, snap.lo, snap.hi)  # rate-limited step ctrl
        step_ablation(rl, np.asarray(a, np.float32), "A")
        peak_qdot = float(np.max(np.abs(d.qvel[:4])))
        meas = measure_contact_velocities(rl, snap.frames)
    finally:
        mujoco.set_mjcb_control(None)
    valid, sig = _contact_mode_valid(snap, meas, peak_qdot, cfg)
    tr = np.asarray(trace) if trace else np.asarray(a)[None, :]
    gov_active = float(np.mean(np.any(np.abs(tr - np.asarray(a)) > 1e-9, axis=1))) if trace else 0.0
    clipped = bool(np.any((np.asarray(a) <= snap.lo + 1e-6) | (np.asarray(a) >= snap.hi - 1e-6)))
    diag = {"fn_left": round(meas["fn_left"], 4), "fn_right": round(meas["fn_right"], 4),
            "straddle_live": round(meas["straddle_live"], 4), "peak_qdot": round(peak_qdot, 4),
            "contract_ok": sig["contract_ok"], "mode": sig, "raw_pd_tau": [round(float(x), 5) for x in raw_pd],
            "applied_tau": [round(float(x), 5) for x in a], "actuator_clipped": clipped,
            "governor_active_frac": round(gov_active, 3), "ctrl_trace": tr}
    return meas["vrel4"], valid, diag


def _classify_column(Bv_col, vp, vm, dp, dm) -> dict:
    """Physically classify a B_v column using the TRUE post-governor per-sub-step ctrl sequences of the ±ε branches
    (not just the rate-limited step torque ``a``). A genuine one-step DEAD ZONE requires the ±ε post-governor sequences
    to be IDENTICAL — then the coincident zero response is real, not a measurement artifact."""
    seq_identical = bool(dp["ctrl_trace"].shape == dm["ctrl_trace"].shape
                         and np.max(np.abs(dp["ctrl_trace"] - dm["ctrl_trace"])) <= 1e-12)
    col_norm = float(np.linalg.norm(np.nan_to_num(Bv_col)))
    if not (vp and vm):
        cls = "INVALID_MODE_SWITCH"
    elif seq_identical and col_norm < 1e-9:
        cls = "DEAD_ZONE_IDENTICAL_POST_GOVERNOR_TORQUE"   # ±ε absorbed (rate-limit / saturation) → true zero authority
    elif col_norm < 1e-9:
        cls = "ZERO_BUT_TORQUE_DIFFERED"                   # torque differed yet no response — flag for scrutiny
    else:
        cls = "ACTIVE"
    return {"plus_valid": vp, "minus_valid": vm, "post_governor_seq_identical": seq_identical,
            "actuator_clipped": bool(dp["actuator_clipped"] or dm["actuator_clipped"]),
            "governor_active_frac": round(0.5 * (dp["governor_active_frac"] + dm["governor_active_frac"]), 3),
            "col_norm": round(col_norm, 8), "tau_differs": (not seq_identical), "class": cls}


def identify_Bv(snap: CradleSnapshot, eps: float, cfg: BvConfig) -> dict:
    """Central-FD ``B_v`` (4×4) at perturbation scale ``eps`` plus the baseline ``g0 = one_step_vrel(0)``. Column j is
    INVALID (left NaN) unless BOTH ±ε branches stay on the same contact mode. Every column is physically CLASSIFIED
    (active / dead-zone / mode-switch) from the true post-governor torque sequences."""
    g0, v0, d0 = one_step_vrel(snap, np.zeros(4), cfg)
    Bv = np.full((4, 4), np.nan)
    col_valid = [False] * 4
    cols = {}
    for j in range(4):
        ej = np.zeros(4)
        ej[j] = eps
        gp, vp, dp = one_step_vrel(snap, ej, cfg)
        gm, vm, dm = one_step_vrel(snap, -ej, cfg)
        if vp and vm:
            Bv[:, j] = (gp - gm) / (2.0 * eps)
            col_valid[j] = True
        cols[j] = _classify_column(Bv[:, j], vp, vm, dp, dm)
    return {"Bv": Bv, "g0": g0, "g0_valid": v0, "col_valid": col_valid, "cols": cols, "eps": float(eps), "diag0": d0}


def replay_consistency(rl_pre_release, stack, prev_tau, q_target, snap: CradleSnapshot) -> dict:
    """G4 — prove the operating point was TRANSFERRED, not just two fields copied: the control-step torque
    ``a = pd_governed_torque(q, q̇, q_target, prev_tau)`` depends only on (q, q̇, q_target, prev_tau), NOT on the pin —
    so continuing the pre-release acquisition state one step must produce the SAME ``a`` as the snapshot's Δq=0 branch
    (the pin-release changes only the RESULTING state, handled explicitly). Returns the max torque mismatch."""
    d_pre = rl_pre_release.inner.data
    a_acq = pd_governed_torque(d_pre.qpos[:4].copy(), d_pre.qvel[:4].copy(), q_target, stack, prev_tau,
                               stack_lo(rl_pre_release), stack_hi(rl_pre_release))
    _g, _v, diag = one_step_vrel(snap, np.zeros(4), BvConfig())
    a_snap = np.asarray(diag["applied_tau"], np.float64)
    mismatch = float(np.max(np.abs(np.asarray(a_acq, np.float64) - a_snap)))
    return {"max_applied_tau_mismatch": round(mismatch, 8), "consistent": bool(mismatch <= 1e-4),
            "a_acquisition_continue": [round(float(x), 5) for x in a_acq], "a_snapshot_dq0": [round(float(x), 5) for x in a_snap]}


def stack_lo(rl):
    return rl.inner.model.actuator_ctrlrange[:4, 0].copy()


def stack_hi(rl):
    return rl.inner.model.actuator_ctrlrange[:4, 1].copy()


def onesided_Bv(snap: CradleSnapshot, eps: float, cfg: BvConfig) -> dict:
    """FORWARD one-sided FD ``B_v[:,j] = (g(+ε e_j) − g0)/ε`` — the near-constraint comparison against the central
    estimate (a large central-vs-one-sided gap flags a nearby contact-mode boundary)."""
    g0, v0, _d0 = one_step_vrel(snap, np.zeros(4), cfg)
    Bv = np.full((4, 4), np.nan)
    col_valid = [False] * 4
    for j in range(4):
        ej = np.zeros(4)
        ej[j] = eps
        gp, vp, _ = one_step_vrel(snap, ej, cfg)
        if vp and v0:
            Bv[:, j] = (gp - g0) / eps
            col_valid[j] = True
    return {"Bv": Bv, "col_valid": col_valid, "eps": float(eps)}


def _valid_indices(col_valid) -> list[int]:
    return [j for j, ok in enumerate(col_valid) if ok]


def _prediction_record(level, mag, dq, pred, actual, valid, diag) -> dict:
    """Per-perturbation prediction metrics: abs / rel / cosine / per-component error + retention flags."""
    err = pred - actual
    abs_err = float(np.linalg.norm(err))
    a_norm = float(np.linalg.norm(actual))
    rel_err = float(abs_err / a_norm) if a_norm > 1e-9 else None
    denom = float(np.linalg.norm(pred) * a_norm)
    cosine = float(pred @ actual / denom) if denom > 1e-12 else None
    return {"level": level, "dq_mag": round(float(mag), 5), "abs_err": round(abs_err, 6),
            "rel_err": (round(rel_err, 4) if rel_err is not None else None),
            "cosine": (round(cosine, 4) if cosine is not None else None),
            "actual_norm": round(a_norm, 6), "pred_norm": round(float(np.linalg.norm(pred)), 6),
            "per_component_abs_err": [round(float(x), 6) for x in np.abs(err)],
            "contact_retained": bool(valid), "contract_ok": bool(diag["contract_ok"])}


def validate_Bv(snap: CradleSnapshot, Bv: np.ndarray, g0: np.ndarray, col_valid, cfg: BvConfig) -> list[dict]:
    """Held-out prediction check: for random Δq restricted to the VALID column subspace at each level
    (small/medium/near-boundary), compare predicted ``B_v·Δq`` against the simulated ``g(Δq) − g0``. Deterministic
    (fixed RNG seed). Returns one record per perturbation (empty if no column is observable)."""
    idx = _valid_indices(col_valid)
    if not idx:
        return []
    rng = np.random.default_rng(cfg.val_seed)
    Bv_use = np.nan_to_num(Bv, nan=0.0)                          # zeroed invalid columns are never excited (dq ⊂ idx)
    records = []
    for level, mag in zip(("small", "medium", "near_boundary"), cfg.dq_levels):
        for _k in range(cfg.n_val_dirs):
            u = np.zeros(4)
            u[idx] = rng.standard_normal(len(idx))
            u = u / (np.linalg.norm(u) + 1e-12)
            dq = mag * u
            g, valid, diag = one_step_vrel(snap, dq, cfg)
            records.append(_prediction_record(level, mag, dq, Bv_use @ dq, g - g0, valid, diag))
    return records
