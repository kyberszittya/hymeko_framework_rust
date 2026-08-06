"""CLAMP-ORACLE-0 — isolated physical diagnostic: is strict clamp-driven transport of the CYLINDRICAL coin achievable
ANYWHERE in the current simulator/contact model, before any CORE actuator change? (Track B side audit; NOT RL, NOT a
controller campaign, NOT task success.)

The rig is a small NON-CANONICAL MuJoCo model (`diagnostic_physical_oracle`): the same cylindrical coin (geometry,
mass/inertia, slide-joint support damping, contact/friction conventions) between TWO INDEPENDENT opposing fingertip
pads on their own slide joints — controlled independently of the wrist-less canonical arm, with no arm-body collision
and no reward/monitor. It lets us apply a clamp (by position OR by a diagnostic force/impedance actuator unavailable in
the canonical arm) and translate the midpoint, then score the SAME strict grasp-integrity predicate. The canonical task
is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np

from hymeko_rl.train.coin_compliant_pad import (
    Compliance,
    compliance_params,
    contact_wrench_budget,
    strict_micro_transport_pass,
)

# canonical coin constants (mirror env.planar_grasp_env.compose_planar_scene — kept in sync, not imported to stay isolated)
_COIN_R, _COIN_H = 0.035, 0.02
_PAD_R = 0.014
_COIN_FRICTION = "1.0 0.05 0.001"
_CONTACT_X = _COIN_R + _PAD_R          # pad-centre |x| at surface contact (0.049)


class Actuation(str, Enum):
    POSITION = "position"              # O1: position-controlled parallel jaw (bounded preload via servo stiffness)
    FORCE = "force"                    # O2/O3: diagnostic force/impedance clamp (equal+opposite), midpoint by position
    OBJECT_DIRECT = "object_direct"    # O5: move the coin directly along its own slide joints (calibration ONLY)


@dataclass(frozen=True)
class OracleParams:
    squeeze_target: float = 0.046      # pad |x| target when clamping (0.003 inside contact → bounded penetration)
    home_x: float = 0.062              # pad |x| start (outside contact)
    clamp_force: float = 2.0           # O2 diagnostic equal+opposite clamp effort (N)
    transport_y: float = 0.030         # midpoint translation along +y
    establish: int = 300               # steps to establish the clamp (~0.15 s ≫ coin settling τ≈0.06 s)
    move_steps: int = 1000             # steps to translate the midpoint (~0.5 s: slow, damped-coin-following)
    kp_pad: float = 300.0              # pad position-servo stiffness (bounds the preload)


def _pad_geom(shape: str, compliance: Compliance, friction: float) -> str:
    fr = f' friction="{friction:g} 0.05 0.001"'
    comp = ""
    cp = compliance_params(compliance)
    if cp is not None:
        (sr0, sr1), si = cp
        comp = f' solref="{sr0:g} {sr1:g}" priority="1"'
        if si is not None:
            comp += ' solimp="' + " ".join(f"{v:g}" for v in si) + '"'
    if shape == "capsule":                                    # Z-aligned line contact (the orientation-robust pad)
        return f'<geom name="fingertip_{{side}}" type="capsule" fromto="0 0 -0.03 0 0 0.03" size="0.012"{fr}{comp}/>'
    return f'<geom name="fingertip_{{side}}" type="sphere" size="{_PAD_R:g}"{fr}{comp}/>'


def _coin_geom(coin_shape: str) -> str:
    if coin_shape == "box":
        return f'<geom name="disk" type="box" size="{_COIN_R:g} {_COIN_R:g} {_COIN_H:g}" friction="{_COIN_FRICTION}"/>'
    return f'<geom name="disk" type="cylinder" size="{_COIN_R:g} {_COIN_H:g}" friction="{_COIN_FRICTION}"/>'


def _actuators(actuation: Actuation) -> str:
    if actuation is Actuation.OBJECT_DIRECT:                  # O5 calibration: drive the coin's own joints
        return ('<position name="disk_x" joint="disk_x" kp="400"/>'
                '<position name="disk_y" joint="disk_y" kp="400"/>')
    pads = ["pad_left_x", "pad_left_y", "pad_right_x", "pad_right_y"]
    if actuation is Actuation.POSITION:                       # O1: all four pad joints position-controlled
        return "".join(f'<position name="{j}" joint="{j}" kp="300"/>' for j in pads)
    # O2/O3: clamp EFFORT on x (motor/force), midpoint on y (position) — force clamp, independent midpoint motion
    return ('<motor name="pad_left_x" joint="pad_left_x" gear="1" ctrlrange="-20 20"/>'
            '<motor name="pad_right_x" joint="pad_right_x" gear="1" ctrlrange="-20 20"/>'
            '<position name="pad_left_y" joint="pad_left_y" kp="300"/>'
            '<position name="pad_right_y" joint="pad_right_y" kp="300"/>')


def build_oracle_mjcf(*, coin_shape: str = "cylinder", fingertip_shape: str = "sphere",
                      compliance: Compliance = Compliance.P0_RIGID, fingertip_friction: float = 1.0,
                      coin_damping: float = 2.5, spin_damping: float = 0.8, coin_frictionloss: float = 0.0,
                      actuation: Actuation = Actuation.POSITION, p: OracleParams = OracleParams()) -> str:
    """Emit the isolated `diagnostic_physical_oracle` MJCF. # Preconditions coin_shape in {cylinder,box};
    fingertip_shape in {sphere,capsule}. # Postconditions cylinder coin + 2 opposing pads, no arm, no floor-drop."""
    pad = _pad_geom(fingertip_shape, compliance, fingertip_friction)
    fl = f' frictionloss="{coin_frictionloss:g}"' if coin_frictionloss > 0.0 else ""
    return f'''<mujoco model="diagnostic_physical_oracle">
  <option timestep="0.0005" integrator="implicitfast" cone="pyramidal" iterations="100"/>
  <worldbody>
    <body name="disk" pos="0 0 0">
      <joint name="disk_x" type="slide" axis="1 0 0" damping="{coin_damping:g}"{fl}/>
      <joint name="disk_y" type="slide" axis="0 1 0" damping="{coin_damping:g}"{fl}/>
      <joint name="disk_rz" type="hinge" axis="0 0 1" damping="{spin_damping:g}"/>
      {_coin_geom(coin_shape)}
    </body>
    <body name="pad_left" pos="-{p.home_x:g} 0 0">
      <joint name="pad_left_x" type="slide" axis="1 0 0" damping="3"/>
      <joint name="pad_left_y" type="slide" axis="0 1 0" damping="3"/>
      {pad.format(side="left")}
    </body>
    <body name="pad_right" pos="{p.home_x:g} 0 0">
      <joint name="pad_right_x" type="slide" axis="1 0 0" damping="3"/>
      <joint name="pad_right_y" type="slide" axis="0 1 0" damping="3"/>
      {pad.format(side="right")}
    </body>
  </worldbody>
  <actuator>{_actuators(actuation)}</actuator>
</mujoco>'''


class OracleRig:
    """Live wrapper exposing the ``model``/``data``/``_disk_geom`` surface the wrench-budget helpers expect, plus the
    pad/coin joint addresses and the clamp+transport protocol. The pads move the coin ONLY by contact (except O5)."""

    def __init__(self, mjcf: str, actuation: Actuation, p: OracleParams = OracleParams()) -> None:
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.actuation = actuation
        self.p = p
        self.control = "normal"                               # "normal" | "open_right" (scramble) | "left_only"
        self._disk_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
        self._act = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(self.model.nu)}
        mujoco.mj_forward(self.model, self.data)

    def _qadr(self, joint: str) -> int:
        return int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)])

    def _set(self, actuator: str, value: float) -> None:
        if actuator in self._act:
            self.data.ctrl[self._act[actuator]] = value

    def _coin_xy_rz(self) -> tuple[np.ndarray, float]:
        return (np.array([self.data.qpos[self._qadr("disk_x")], self.data.qpos[self._qadr("disk_y")]]),
                float(self.data.qpos[self._qadr("disk_rz")]))

    def _midpoint_y(self) -> float:
        """The fingertip-midpoint y — the true reference the coupling ρ is measured against. For O5 (no pads) the coin
        IS driven directly, so the midpoint reference is the commanded transport target."""
        if self.actuation is Actuation.OBJECT_DIRECT:
            return float(self.data.qpos[self._qadr("disk_y")])
        return 0.5 * (self.data.qpos[self._qadr("pad_left_y")] + self.data.qpos[self._qadr("pad_right_y")])

    def _clamp_cmd(self, y_cmd: float) -> None:
        """Command the clamp (bounded preload) + midpoint y. Position mode drives all pad joints; force mode drives an
        equal+opposite clamp effort on x and positions on y; object-direct drives the coin itself (O5 calibration)."""
        if self.actuation is Actuation.OBJECT_DIRECT:
            self._set("disk_x", 0.0)
            self._set("disk_y", y_cmd)
            return
        self._set("pad_left_y", y_cmd)
        self._set("pad_right_y", y_cmd)
        # "open_right" (structural scramble, §T9): the right pad OPENS instead of clamping → unilateral, should NOT
        # strict-pass. A correct bilateral clamp must beat this to prove the pass is a real grasp, not a one-sided shove.
        open_right = self.control == "open_right"
        if self.actuation is Actuation.POSITION:
            # pad x-joint qpos is measured from the body home (±home_x); the INWARD displacement to reach |x|=squeeze
            # is (home_x - squeeze_target): +for the left pad (home -home_x → +x), −for the right pad (home +home_x → −x).
            close = self.p.home_x - self.p.squeeze_target
            self._set("pad_left_x", close)
            self._set("pad_right_x", 0.0 if open_right else -close)   # 0 = right pad stays at its outward home
        else:                                                # FORCE: equal+opposite clamp effort toward the coin
            self._set("pad_left_x", self.p.clamp_force)       # +x drives the left pad toward the coin at 0
            self._set("pad_right_x", self.p.clamp_force if open_right else -self.p.clamp_force)  # +force = away from coin

    def _actuator_saturation(self) -> float:
        """Max |ctrl| against range as a saturation proxy (position servos are unbounded; motors clamp to ctrlrange)."""
        if self.model.nu == 0:
            return 0.0
        return float(np.max(np.abs(self.data.ctrl[: self.model.nu])))

    def run_condition(self, *, eps_target: float = 0.010, control: str = "normal") -> "OracleResult":
        """Establish the clamp, translate the midpoint +y, score the STRICT grasp-integrity predicate + wrench audit.
        ``control`` = "normal" | "open_right" (structural scramble: right pad opens → unilateral, should fail)."""
        self.control = control
        for _ in range(self.p.establish):                    # phase 1: establish the clamp (no midpoint motion)
            self._clamp_cmd(0.0)
            mujoco.mj_step(self.model, self.data)
        coin0, rz0 = self._coin_xy_rz()
        mid0 = self._midpoint_y()
        log = self._transport()
        coin1, rz1 = self._coin_xy_rz()
        mid_disp = abs(self._midpoint_y() - mid0)
        return _score(self, coin0, rz0, coin1, rz1, mid_disp, log, eps_target)

    def _transport(self) -> dict:
        """Phase 2: ramp the midpoint to +transport_y over move_steps, logging per-step contact wrench + support."""
        both, fns, etas, fts, supp, pens, solver_ok = 0, [], [], [], [], [], True
        n = self.p.move_steps
        for t in range(n):
            self._clamp_cmd(self.p.transport_y * (t + 1) / n)
            mujoco.mj_step(self.model, self.data)
            wb = contact_wrench_budget(self)
            both += int(wb["bilateral"])
            fns.append(wb["F_N_min"])
            etas.append(wb["eta"])
            fts.append(wb["F_T_total"])
            supp.append(wb["support_resistance"])
            pens.append(wb["penetration_max"])
            if not np.all(np.isfinite(self.data.qacc)) or wb["penetration_max"] > 0.02:
                solver_ok = False
        return {"both": both, "min_FN": min(fns, default=0.0), "max_eta": max(etas, default=0.0),
                "mean_FT": float(np.mean(fts) if fts else 0.0), "mean_support": float(np.mean(supp) if supp else 0.0),
                "penetration_max": max(pens, default=0.0), "solver_ok": solver_ok, "sat": self._actuator_saturation()}


@dataclass(frozen=True)
class OracleResult:
    condition: str
    strict_pass: bool
    rho: float
    both_frac: float
    align: float
    coin_disp: float
    coin_rot: float
    min_FN: float
    max_eta: float
    penetration_max: float
    finger_exceeds_support: bool
    actuator_saturation: float
    object_actuated: bool          # True for O5 → NEVER counts as grasp capability
    solver_ok: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _score(rig: OracleRig, coin0, rz0, coin1, rz1, mid_disp, log, eps_target) -> OracleResult:
    coin_disp = float(np.linalg.norm(coin1 - coin0))
    d = np.array([0.0, 1.0])                                  # intended transport direction (+y)
    align = float((coin1 - coin0) @ d) / (coin_disp + 1e-9)
    rho = coin_disp / (mid_disp + 1e-9)                       # coupling vs the ACTUAL fingertip-midpoint travel
    both_frac = log["both"] / max(1, rig.p.move_steps)
    wb = contact_wrench_budget(rig)
    object_actuated = rig.actuation is Actuation.OBJECT_DIRECT
    # STRICT oracle predicate: the strengthened grasp-integrity predicate AND (for a grasp claim) NOT direct-object-driven.
    grasp_ok = strict_micro_transport_pass(coin_disp=coin_disp, align=align, rho=rho, both_frac=both_frac,
                                           final_bilateral=wb["bilateral"], max_eta=log["max_eta"], min_FN=log["min_FN"],
                                           penetration_max=log["penetration_max"], solver_ok=log["solver_ok"],
                                           eps_target=eps_target)
    strict_pass = bool(grasp_ok and not object_actuated)
    return OracleResult(condition="", strict_pass=strict_pass, rho=round(rho, 3), both_frac=round(both_frac, 3),
                        align=round(align, 3), coin_disp=round(coin_disp, 4), coin_rot=round(abs(rz1 - rz0), 4),
                        min_FN=round(log["min_FN"], 4), max_eta=round(log["max_eta"], 3),
                        penetration_max=round(log["penetration_max"], 5),
                        finger_exceeds_support=bool(log["mean_FT"] > log["mean_support"]),
                        actuator_saturation=round(log["sat"], 3), object_actuated=object_actuated,
                        solver_ok=log["solver_ok"])
