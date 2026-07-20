"""COIN-COMPLIANT-PAD-1 — contact-patch & tangential-coupling audit (simulated compliant contact, no CORE change).

COIN-GRIP-CONTROL-1 (F-COIN-GRIP-CONTROL, Case E) rejected position-space clamp REGULATION as the coin-transport lever
(strict micro-transport 0/13; a scramble beat the correct controller ⇒ bulldoze artifacts). The remaining OFF-axis,
non-CORE lever is *simulated compliant contact*: softer fingertip normal response → (hypothesised) larger effective
contact patch → better normal-load distribution → higher tangential coupling → micro-transport.

Actuator/contact facts (measured): 4 position servos, contact force OBSERVABLE (mj_contactForce), timestep 5e-4,
implicitfast integrator, pyramidal cone. The coin's SUPPORT resistance is its slide-joint damping (2.5, viscous) +
frictionloss (0, dry) — NOT floor contact. Compliance is set on the fingertip geom's solref/solimp (priority=1 so the
fingertip governs the fingertip↔coin contact); the coin/floor/friction stay canonical. This is *simulated compliant
contact*, NOT literal material elasticity. NO reward/CORE change; the coin stays cylindrical.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np

from hymeko_rl.experiments.coin_delivery1 import _dir_to_zone
from hymeko_rl.train.coin_grasp_cert import ContactModeClassifier, GraspFamily, GraspParams, grasp_action
from hymeko_rl.train.coin_grip_control import GripController, GripParams, grip_action
from hymeko_rl.train.coin_transport import restore_planar

_EPS = 1e-9
Solref = tuple  # (timeconst, dampratio)


class Compliance(str, Enum):
    """A small predeclared compliance ladder (§6). P0 = canonical rigid; P3 = diagnostic-only, not canonically adoptable."""
    P0_RIGID = "P0_rigid"
    P1_MILD = "P1_mild"
    P2_MODERATE = "P2_moderate"
    P3_HIGH = "P3_high_diagnostic"


# solref = (timeconst, dampratio); solimp = standard 5-tuple or None (keep default). Larger timeconst = softer contact.
_COMPLIANCE: dict[Compliance, tuple | None] = {
    Compliance.P0_RIGID: None,
    Compliance.P1_MILD: ((0.04, 1.0), None),
    Compliance.P2_MODERATE: ((0.08, 1.0), (0.85, 0.92, 0.002, 0.5, 2.0)),
    Compliance.P3_HIGH: ((0.15, 1.0), (0.80, 0.90, 0.003, 0.5, 2.0)),
}


def compliance_params(level: Compliance) -> tuple | None:
    """The (solref, solimp) for a compliance level, or None for canonical rigid contact. # Postconditions: None ⇒ P0."""
    return _COMPLIANCE[level]


def set_runtime_compliance(inner, left: Compliance, right: Compliance) -> None:
    """Per-side fingertip compliance on the LIVE model (S4 unilateral / S5 mismatched structural controls) — mutates
    geom_solref/solimp/priority for fingertip_left/right, no MJCF rebuild. # Preconditions env built rigid (P0)."""
    m = inner.model
    for side, level in (("left", left), ("right", right)):
        cp = compliance_params(level)
        if cp is None:
            continue
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fingertip_{side}")
        solref, solimp = cp
        m.geom_solref[gid] = solref
        m.geom_priority[gid] = 1
        if solimp is not None:
            m.geom_solimp[gid] = solimp


class Mechanism(str, Enum):
    """Failure-mechanism taxonomy for a non-transporting episode (§5)."""
    OK = "ok_transport"
    M0_LOW_NORMAL = "M0_insufficient_normal_load"
    M1_CONE_SAT = "M1_friction_cone_saturation"
    M2_SUPPORT_PIN = "M2_support_pinning"
    M3_PATCH_UNSTABLE = "M3_contact_patch_instability"
    M4_CTRL_BANDWIDTH = "M4_controller_bandwidth"
    M5_ROLLING = "M5_geometric_rolling"


def contact_model_inventory(inner) -> dict:
    """§3 contact-mechanics capability table: solver/timestep + fingertip/disk contact params + coin support resistance.
    All values are read from the LIVE model — the ground truth the compliance variants perturb."""
    m = inner.model
    gid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fingertip_left", "disk", "floor")}
    dof = {}
    for jn in ("disk_x", "disk_y", "disk_rz"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        adr = int(m.jnt_dofadr[jid])
        dof[jn] = {"damping": float(m.dof_damping[adr]), "frictionloss": float(m.dof_frictionloss[adr])}
    return {
        "timestep": float(m.opt.timestep), "integrator": int(m.opt.integrator), "cone": int(m.opt.cone),
        "iterations": int(m.opt.iterations), "default_solref": m.opt.o_solref.tolist(), "default_solimp": m.opt.o_solimp.tolist(),
        "fingertip_solref": m.geom_solref[gid["fingertip_left"]].tolist(),
        "fingertip_solimp": m.geom_solimp[gid["fingertip_left"]].tolist(),
        "fingertip_friction": m.geom_friction[gid["fingertip_left"]].tolist(),
        "fingertip_priority": int(m.geom_priority[gid["fingertip_left"]]),
        "disk_solref": m.geom_solref[gid["disk"]].tolist(), "disk_friction": m.geom_friction[gid["disk"]].tolist(),
        "coin_support_dofs": dof,
    }


def _fingertip_coin_contacts(inner) -> list[dict]:
    """Per fingertip↔coin contact: normal force F_N, tangential magnitude F_T, penetration (=-dist>0), side."""
    m, d = inner.model, inner.data
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    dg = int(getattr(inner, "_disk_geom", -1))
    out, buf = [], np.zeros(6)
    for i in range(int(d.ncon)):
        con = d.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if dg not in (g1, g2):
            continue
        other = g2 if g1 == dg else g1
        if other not in (fl, fr):
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        fn, ft = abs(float(buf[0])), float(np.hypot(buf[1], buf[2]))
        out.append({"side": "L" if other == fl else "R", "F_N": fn, "F_T": ft,
                    "eta": ft / (m.geom_friction[other][0] * fn + _EPS) if fn > 0 else 0.0,
                    "penetration": max(0.0, -float(con.dist))})
    return out


def _coin_support_resistance(inner) -> float:
    """Support tangential resistance the drag must overcome = slide-joint viscous damping·|v| + dry frictionloss (§4)."""
    m, d = inner.model, inner.data
    total = 0.0
    for jn in ("disk_x", "disk_y"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        adr = int(m.jnt_dofadr[jid])
        total += (m.dof_damping[adr] * abs(float(d.qvel[adr])) + m.dof_frictionloss[adr]) ** 2
    return float(np.sqrt(total))


def contact_wrench_budget(inner) -> dict:
    """§4 contact-wrench budget at the current state: aggregate fingertip F_N/F_T, friction utilization η (per-contact,
    from the geom friction), contact count, and the finger-vs-support tangential inequality. # Postconditions η≥0; support≥0."""
    cs = _fingertip_coin_contacts(inner)
    sides = {c["side"] for c in cs}
    fn = min((c["F_N"] for c in cs), default=0.0)                 # weaker side governs bilateral clamp integrity
    ft = sum(c["F_T"] for c in cs)
    eta = max((c["eta"] for c in cs), default=0.0)                # per-contact friction utilization (worst side); >=1 = slip
    support = _coin_support_resistance(inner)
    return {"n_contacts": len(cs), "bilateral": len(sides) == 2, "F_N_min": fn, "F_T_total": ft,
            "eta": float(eta), "penetration_max": max((c["penetration"] for c in cs), default=0.0),
            "support_resistance": support, "finger_exceeds_support": bool(ft > support)}


def _classify(stats: dict, move_steps: int) -> Mechanism:
    """Dominant failure mechanism (§5), priority-ordered on the run aggregates."""
    if stats["micro_strict"]:
        return Mechanism.OK
    if stats["both_frac"] < 0.75 and stats["min_FN"] >= 0.15:
        return Mechanism.M4_CTRL_BANDWIDTH                        # contact lost mid-motion despite adequate start load
    if stats["min_FN"] < 0.1:
        return Mechanism.M0_LOW_NORMAL
    if stats["coin_rot"] > 0.35 and stats["both_frac"] >= 0.75:
        return Mechanism.M5_ROLLING                              # coin rotates out while nominally clamped
    if stats["max_eta"] >= 0.95:
        return Mechanism.M1_CONE_SAT
    if stats["mean_support"] > stats["mean_FT"]:
        return Mechanism.M2_SUPPORT_PIN
    return Mechanism.M3_PATCH_UNSTABLE


@dataclass(frozen=True)
class MicroResult:
    seed: int
    compliance: str
    micro_strict: bool
    rho: float
    both_frac: float
    align: float
    penetration_max: float
    min_FN: float
    max_eta: float
    coin_disp: float
    coin_rot: float
    mechanism: str
    solver_ok: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compliant_micro_transport(env, handoff, compliance: Compliance, controller: GripController, gp_grip: GripParams,
                              *, establish: int = 20, move_steps: int = 16,
                              eps_target: float = 0.010, scramble=None) -> MicroResult:
    """From a certified handoff, establish the grasp, translate the midpoint under ``controller`` while logging the
    per-step contact wrench, then apply the STRENGTHENED strict micro-transport predicate (§11) and classify the
    dominant failure mechanism (§5). The env is built with the fingertip geometry+compliance by the caller."""
    inner = env._env
    restore_planar(inner, handoff.snap)
    env._horizon = establish + move_steps + 400
    obs = _establish_grasp(env, inner, handoff.obs.copy(), establish)
    d, _n = _dir_to_zone(inner)
    coin0, rot0 = _coin_pose(inner)
    log = _run_transport(env, inner, obs, controller, gp_grip, np.asarray(d, np.float64), move_steps, scramble)
    return _verdict(handoff, compliance, inner, coin0, rot0, np.asarray(d, np.float64), log, move_steps, eps_target)


def _establish_grasp(env, inner, obs, establish: int):
    gp = GraspParams()
    clf = ContactModeClassifier(gp.dwell)
    for _ in range(establish):
        mode = clf.classify(inner)
        a = np.clip(grasp_action(inner, obs, mode, GraspFamily.G1_SYMMETRIC, gp), -1, 1).astype(np.float32)
        obs, _r, _t, _tr, _i = env.step(a)
    return obs


def _coin_pose(inner) -> tuple[np.ndarray, float]:
    m = inner._planar_metrics
    jid = mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_JOINT, "disk_rz")
    rz = float(inner.data.qpos[int(inner.model.jnt_qposadr[jid])])
    return np.asarray(m.disk_pos[:2], np.float64), rz


def _run_transport(env, inner, obs, controller, gp_grip, d, move_steps, scramble) -> dict:
    """Step the midpoint translation, logging both-contact / F_N / η / support / penetration / solver-health per step."""
    both, fns, etas, supp, fts, pens, solver_ok = 0, [], [], [], [], [], True
    for t in range(move_steps):
        a = np.clip(grip_action(inner, obs, controller, gp_grip, d), -1, 1).astype(np.float32)
        if scramble is not None:
            a = scramble(a, t)
        obs, _r, _t, _tr, _i = env.step(a)
        wb = contact_wrench_budget(inner)
        both += int(wb["bilateral"])
        fns.append(wb["F_N_min"])
        etas.append(wb["eta"])
        supp.append(wb["support_resistance"])
        fts.append(wb["F_T_total"])
        pens.append(wb["penetration_max"])
        if not np.all(np.isfinite(inner.data.qacc)) or wb["penetration_max"] > 0.02:
            solver_ok = False                                    # NaN/inf accel or runaway penetration = instability
    return {"both": both, "obs": obs, "min_FN": min(fns, default=0.0), "max_eta": max(etas, default=0.0),
            "mean_support": float(np.mean(supp) if supp else 0.0), "mean_FT": float(np.mean(fts) if fts else 0.0),
            "penetration_max": max(pens, default=0.0), "solver_ok": solver_ok}


def strict_micro_transport_pass(*, coin_disp: float, align: float, rho: float, both_frac: float, final_bilateral: bool,
                                max_eta: float, min_FN: float, penetration_max: float, solver_ok: bool,
                                eps_target: float = 0.010) -> bool:
    """The STRENGTHENED strict micro-transport predicate (§11), as a pure function so it can be calibrated on synthetic
    inputs. A trial passes ONLY if: coin travels ≥ eps_target, in-direction (align≥0.7), coupling ρ≥0.30, clamped
    throughout (both_frac≥0.75), ends bilateral, is NOT at the friction limit (max_eta<0.98 — excludes slip/support-only
    motion), carries adequate normal load (min_FN≥0.05 — not a soft slack grip), penetration bounded, solver stable.
    # Postconditions: a firm-grip drag (high ρ, low η, bilateral) passes; a bulldoze / slip / soft-grip is rejected."""
    return bool(coin_disp >= eps_target and align >= 0.7 and rho >= 0.30 and both_frac >= 0.75
                and final_bilateral and max_eta < 0.98 and min_FN >= 0.05
                and penetration_max <= 0.02 and solver_ok)


def _verdict(handoff, compliance, inner, coin0, rot0, d, log, move_steps, eps_target) -> MicroResult:
    m = inner._planar_metrics
    coin1, rot1 = _coin_pose(inner)
    coin_disp = float(np.linalg.norm(coin1 - coin0))
    coin_rot = abs(rot1 - rot0)
    both_frac = log["both"] / max(1, move_steps)
    align = float((coin1 - coin0) @ d) / (coin_disp * float(np.linalg.norm(d)) + _EPS)
    mid_disp = 0.15 * 0.012 * move_steps                        # commanded midpoint translation (GripParams.v_translate)
    rho = coin_disp / (mid_disp + _EPS)
    final_bilateral = bool(m.left_contact and m.right_contact)
    micro = strict_micro_transport_pass(coin_disp=coin_disp, align=align, rho=rho, both_frac=both_frac,
                                        final_bilateral=final_bilateral, max_eta=log["max_eta"], min_FN=log["min_FN"],
                                        penetration_max=log["penetration_max"], solver_ok=log["solver_ok"],
                                        eps_target=eps_target)
    stats = {"micro_strict": micro, "both_frac": both_frac, "min_FN": log["min_FN"], "max_eta": log["max_eta"],
             "coin_rot": coin_rot, "mean_support": log["mean_support"], "mean_FT": log["mean_FT"]}
    return MicroResult(handoff.seed, compliance.value, micro, round(rho, 3), round(both_frac, 3), round(align, 3),
                       round(log["penetration_max"], 5), round(log["min_FN"], 4), round(log["max_eta"], 3),
                       round(coin_disp, 4), round(coin_rot, 4), _classify(stats, move_steps).value, log["solver_ok"])
