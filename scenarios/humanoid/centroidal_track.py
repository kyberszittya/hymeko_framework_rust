"""Whole-body tracking of the centroidal running plan — the humanoid RUNS the Tedrake momentum trajectory.

``centroidal_run`` gives a dynamically-feasible CoM/momentum plan for a running stride (STANCE contact force
+ FLIGHT ballistic). This module executes it on the full sagittal humanoid with the existing whole-body
controller (``wbc.WholeBodyController``):

- STANCE: contact = the stance foot; the WBC realises the plan's CoM acceleration (feed-forward the planned
  CoM accel + a PD correction) — the ground reaction that emerges is the planned contact force — while the
  swing foot is driven forward to the next foothold and lifted for clearance.
- FLIGHT: NO contact; the CoM is ballistic (gravity), and the WBC positions BOTH legs for the landing
  (a joint-space task) so the next stance starts cleanly.

The stance alternates L/R each stride and the whole plan translates forward by one stride length each cycle.
No RL, no hand-tuned gait — the motion is the momentum plan tracked by the WBC.

# Preconditions: the floating humanoid model emits; the plan is feasible (see ``centroidal_run``).
# Postconditions: ``run`` returns (net forward, flight fraction, upright fraction, fell) over the strides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenarios.humanoid.balance_env import _build
from scenarios.humanoid.centroidal_run import CentroidalRunConfig, RunTrajectory, solve_run
from scenarios.humanoid.wbc import Task, WholeBodyController

G = 9.81


@dataclass(frozen=True)
class TrackConfig:
    kp_com: float = 120.0            # horizontal CoM-tracking PD (over the plan feed-forward)
    kd_com: float = 22.0
    kp_com_z: float = 260.0          # VERTICAL CoM gain — higher, to recover the height lost each flight (anti-sink)
    kd_com_z: float = 34.0
    pel_w: float = 40.0              # pelvis-orientation task weight (raise to hold the torso UPRIGHT vs the forward dive)
    pel_kp: float = 140.0
    post_w: float = 0.4              # posture-task weight (raise to resist the progressive running crouch / sink)
    post_kp: float = 6.0
    push_boost: float = 1.0          # vertical stance push-off boost (>1 injects extra energy to offset per-cycle loss)
    com_w: float = 150.0             # WBC CoM-task weight (raise so the WBC realises the plan's CoM accel over swing/posture)
    swing_h: float = 0.07            # swing-foot apex clearance during stance (m)
    swing_w: float = 120.0           # WBC swing-foot task weight
    land_w: float = 60.0             # WBC both-feet task weight during flight (reach for landing)
    base_z0: float = 0.80            # reset pelvis height; settle plants the feet
    settle: int = 60
    fall_z: float = 0.45             # pelvis-z below this = fallen


class CentroidalRunner:
    """Execute a centroidal running plan on the humanoid via the whole-body controller."""

    def __init__(self, run_cfg: CentroidalRunConfig | None = None, cfg: TrackConfig | None = None,
                 model_src: str = "humanoid.hymeko") -> None:
        self.cfg = cfg or TrackConfig()
        self._mj, self.model = _build(model_src)
        self.data = self._mj.MjData(self.model)
        self._mj.mj_forward(self.model, self.data)
        self._mass = float(self.model.body_mass.sum())        # the REAL total mass — the plan + feed-forward must use it
        run_cfg = run_cfg or CentroidalRunConfig()
        self.plan: RunTrajectory = solve_run(CentroidalRunConfig(  # solve the plan for the ACTUAL humanoid mass
            **{**run_cfg.__dict__, "mass": self._mass}))
        self._q0 = self.data.qpos.copy()
        self._base = int(self.model.jnt_qposadr[self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_JOINT, "base")])
        self._pel = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "pelvis")
        self._fl = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_l")
        self._fr = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_r")
        self._act_dof = [int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]])
                         for i in range(self.model.nu)]
        self._act_q = [int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]])
                       for i in range(self.model.nu)]
        self._q0j = np.array([self.data.qpos[a] for a in self._act_q])
        self.wbc = WholeBodyController(self.model, self.data, self._act_dof, "base")
        self._pelR0 = self.data.xmat[self._pel].reshape(3, 3).copy()
        self._dt = float(self.model.opt.timestep) * int(self.model.opt.timestep and 1)  # per mj_step
        self._floor = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, "floor")

    def reset(self) -> None:
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._q0
        self.data.qpos[self._base + 2] = self.cfg.base_z0
        self._mj.mj_forward(self.model, self.data)
        for _ in range(self.cfg.settle):                     # settle onto the floor at q0
            self._pd_hold()
            self._mj.mj_step(self.model, self.data)

    def _pd_hold(self) -> None:
        tau = np.array([60.0 * (self._q0j[i] - self.data.qpos[qa]) - 6.0 * self.data.qvel[dof]
                        + float(self.data.qfrc_bias[dof])
                        for i, (dof, qa) in enumerate(zip(self._act_dof, self._act_q))])
        self.data.ctrl[:] = np.clip(tau, -150.0, 150.0)

    def _airborne(self) -> bool:
        return not any(int(self.data.contact[c].geom1) == self._floor or int(self.data.contact[c].geom2) == self._floor
                       for c in range(int(self.data.ncon)))

    def _com(self) -> np.ndarray:
        return np.asarray(self.data.subtree_com[1])

    def _com_vel(self) -> np.ndarray:
        return self.wbc.com_jacobian() @ np.asarray(self.data.qvel)

    def _pelvis_task(self) -> Task:
        _jp, jr = self.wbc.body_jacobian(self._pel)
        acc = self.cfg.pel_kp * self.wbc.orientation_error(self.data.xmat[self._pel].reshape(3, 3), self._pelR0) \
            - 2.0 * np.sqrt(self.cfg.pel_kp) * (jr @ np.asarray(self.data.qvel))
        return Task(jr, acc, self.cfg.pel_w)

    def _foot_task(self, body: int, target: np.ndarray, weight: float, kp: float = 220.0, kd: float = 28.0) -> Task:
        jp, _ = self.wbc.body_jacobian(body)
        pos = np.asarray(self.data.xpos[body])
        acc = kp * (target - pos) - kd * (jp @ np.asarray(self.data.qvel))
        return Task(jp, acc, weight)

    def _com_accel_task(self, com_des: np.ndarray, comv_des: np.ndarray, ff: np.ndarray) -> np.ndarray:
        com, comv = self._com(), self._com_vel()
        kp = np.array([self.cfg.kp_com, self.cfg.kp_com, self.cfg.kp_com_z])
        kd = np.array([self.cfg.kd_com, self.cfg.kd_com, self.cfg.kd_com_z])
        return ff + kp * (com_des - com) + kd * (comv_des - comv)

    def run(self, n_strides: int = 6, render_cb=None) -> "tuple[float, float, float, bool]":
        """Track the plan for ``n_strides`` alternating strides. Returns (net forward, flight frac, upright, fell)."""
        self.reset()
        p = self.plan
        dt = float(self.model.opt.timestep)
        ns_st = max(int(round(p.t_stance / dt)), 1)
        ns_fl = max(int(round(p.t_flight / dt)), 1)
        px0 = float(self.data.xpos[self._pel, 0])
        com0 = self._com().copy()
        stance, swing = self._fl, self._fr                   # start on the left foot
        n_up = n_air = n_tot = 0
        base_shift = 0.0
        for _stride in range(n_strides):
            foot0 = np.asarray(self.data.xpos[stance]).copy()   # planted stance foot (world)
            sw_start = np.asarray(self.data.xpos[swing]).copy()
            sw_land = sw_start + np.array([p.stride, 0.0, 0.0])  # next foothold, one stride ahead
            # --- STANCE: track CoM plan on the stance foot, swing the other leg forward ---
            for k in range(ns_st):
                frac = k / ns_st
                ip = min(int(frac * (p.com.shape[0] * p.t_stance / (p.t_stance + p.t_flight))), len(p.com) - 1)
                com_des = com0 + np.array([base_shift + p.com[ip, 0], 0.0, p.com[ip, 1] - p.com[0, 1]])
                comv_des = np.array([p.vel[ip, 0], 0.0, p.vel[ip, 1]])
                ff = np.array([p.force[ip, 0] / self._mass, 0.0,
                               self.cfg.push_boost * p.force[ip, 1] / self._mass - G])
                acc_com = self._com_accel_task(com_des, comv_des, ff)
                sw = sw_start + frac * (sw_land - sw_start)
                sw[2] = sw_start[2] + self.cfg.swing_h * np.sin(np.pi * frac)   # clearance arc
                tasks = [Task(self.wbc.com_jacobian(), acc_com, self.cfg.com_w),
                         self._foot_task(swing, sw, self.cfg.swing_w), self._pelvis_task(),
                         self.wbc.posture_task(self._q0j, self._act_q, self.cfg.post_kp, self.cfg.post_kp*0.5, self.cfg.post_w)]
                self.data.ctrl[:] = self.wbc.solve([stance], tasks)   # single-support: one 6D contact, no unload
                self._mj.mj_step(self.model, self.data)
                n_up += int(self.data.xmat[self._pel].reshape(3, 3)[2, 2] > 0.6)
                n_air += int(self._airborne())
                n_tot += 1
                if render_cb:
                    render_cb(self)
                if float(self.data.xpos[self._pel, 2]) < self.cfg.fall_z:
                    return float(self.data.xpos[self._pel, 0]) - px0, n_air / max(n_tot, 1), n_up / max(n_tot, 1), True
            # --- FLIGHT: no contact, ballistic CoM, both legs reach for landing ---
            for k in range(ns_fl):
                land_sw = sw_land.copy()
                land_st = foot0 + np.array([p.stride, 0.0, 0.0])   # the (old) stance foot swings up too, next-next
                tasks = [self._foot_task(swing, land_sw, self.cfg.land_w),
                         self._foot_task(stance, land_st, self.cfg.land_w * 0.5), self._pelvis_task(),
                         self.wbc.posture_task(self._q0j, self._act_q, self.cfg.post_kp, self.cfg.post_kp*0.5, self.cfg.post_w)]
                self.data.ctrl[:] = self.wbc.solve([], tasks)     # NO contact = flight
                self._mj.mj_step(self.model, self.data)
                n_up += int(self.data.xmat[self._pel].reshape(3, 3)[2, 2] > 0.6)
                n_air += int(self._airborne())
                n_tot += 1
                if render_cb:
                    render_cb(self)
                if float(self.data.xpos[self._pel, 2]) < self.cfg.fall_z:
                    return float(self.data.xpos[self._pel, 0]) - px0, n_air / max(n_tot, 1), n_up / max(n_tot, 1), True
            base_shift += p.stride
            stance, swing = swing, stance                    # alternate feet
        fwd = float(self.data.xpos[self._pel, 0]) - px0
        return fwd, n_air / max(n_tot, 1), n_up / max(n_tot, 1), False
