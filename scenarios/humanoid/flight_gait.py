"""Dynamic FLIGHT-PHASE running gait for the humanoid — push-off + flight + land, direct PD gait.

The quasi-static WBC footstep stack cannot walk visibly (mechanism wall: small feet + static stability —
12 configs / 2 katolab sweeps confirmed lunge-fall vs over-conservative-shuffle). A flight-phase gait is
MOMENTUM-based: the stance leg pushes off (knee + ankle extension) to launch the CoM into a brief flight
(both feet airborne), the swing leg reaches forward, and the humanoid lands and absorbs. This env applies a
cyclic PD-to-phase-target gait (NO contact-consistency constraint — the body may leave the ground) with a
compact CEM-tunable parameterization; the reward rewards forward progress + staying upright + an explicit
FLIGHT bonus (both feet off the ground), which the static stack structurally cannot earn.

# Preconditions: the floating-base humanoid model emits (see ``balance_env._build``).
# Postconditions: ``rollout(theta)`` returns (return, forward, max_flight_frac, upright_frac).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .balance_env import _build

# CEM parameter layout (theta): compact, physically-meaningful running-gait knobs.
_P = ("freq", "hip_amp", "hip_off", "knee_amp", "knee_off", "knee_crouch",
      "ankle_amp", "ankle_off", "push_amp", "lean", "arm_amp")
PDIM = len(_P)


@dataclass(frozen=True)
class FlightGaitConfig:
    kp: float = 80.0                 # PD-to-target stiffness (higher than balance for dynamic push-off)
    kv: float = 8.0
    tau_max: float = 200.0           # per-joint torque cap (N·m) — push-off needs authority
    base_z0: float = 0.76            # reset pelvis height; the reset SETTLES the feet onto the floor from here
    settle: int = 60                 # q0-hold steps at reset to plant the feet before the gait starts
    steps: int = 800                 # sim steps per rollout (~0.8 s at 1 ms)
    w_forward: float = 6.0
    w_upright: float = 1.0
    w_flight: float = 2.0            # explicit reward for a genuine flight phase (both feet up)
    w_ctrl: float = 0.0005
    fall_z: float = 0.55             # pelvis-z below this = fallen


class FlightGaitEnv:
    """Floating humanoid + floor; a cyclic PD running gait applied as direct torque (no WBC / no contact
    constraint), so a flight phase is representable. Legs run in antiphase; arms swing counter for balance."""

    #: sagittal running DOF actuator indices (hip, knee, ankle per leg) + arms; frontal DOF held at q0
    LEG = {"hip": (3, 8), "knee": (4, 9), "ankle": (6, 11)}
    ARM = (12, 14)                                             # shoulder_l, shoulder_r (counter-swing)

    def __init__(self, cfg: FlightGaitConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or FlightGaitConfig()
        self._mj, self.model = _build("humanoid.hymeko")
        self.data = self._mj.MjData(self.model)
        self._mj.mj_forward(self.model, self.data)
        self._q0 = self.data.qpos.copy()
        self._base = int(self.model.jnt_qposadr[self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_JOINT, "base")])
        self._pelvis = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "pelvis")
        self._fl = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_l")
        self._fr = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "foot_r")
        self._act_dof = [int(self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]])
                         for i in range(self.model.nu)]
        self._act_qadr = [int(self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]])
                          for i in range(self.model.nu)]
        self._q0j = np.array([self.data.qpos[a] for a in self._act_qadr])
        self._floor = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, "floor")
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)          # full clear (incl. warmstart) → deterministic
        self.data.qpos[:] = self._q0
        self.data.qpos[self._base + 2] = self.cfg.base_z0
        self.data.qvel[:] = 0.0
        self._mj.mj_forward(self.model, self.data)
        for _ in range(self.cfg.settle):                      # settle the feet onto the floor (q0-hold under gravity)
            tau = np.array([self.cfg.kp * (self._q0j[i] - self.data.qpos[qa]) - self.cfg.kv * self.data.qvel[dof]
                            + float(self.data.qfrc_bias[dof])
                            for i, (dof, qa) in enumerate(zip(self._act_dof, self._act_qadr))])
            self.data.ctrl[:] = np.clip(tau, -self.cfg.tau_max, self.cfg.tau_max)
            self._mj.mj_step(self.model, self.data)

    def _phase_targets(self, theta: np.ndarray, ph: float) -> np.ndarray:
        """Per-actuator joint target for the running cycle at phase ``ph`` (rad). Legs antiphase.

        Running cycle per leg: hip swings (fore-aft), knee crouches then EXTENDS for push-off (``push_amp``
        pulse at late stance), ankle plantarflexes to launch, then the leg flexes to clear + reach in swing.
        """
        p = dict(zip(_P, theta))
        tgt = self._q0j.copy()
        for side, (hi, ki, ai) in enumerate(zip(self.LEG["hip"], self.LEG["knee"], self.LEG["ankle"])):
            legph = ph + side * np.pi                          # right leg is half a cycle behind the left
            s, c = np.sin(legph), np.cos(legph)
            # push-off pulse: sharp positive lobe in late stance (legph near 0/2pi), zero in swing
            push = max(0.0, np.cos(legph)) ** 3
            tgt[hi] = self._q0j[hi] + p["lean"] - p["hip_amp"] * s               # hip: swing fore-aft + lean
            tgt[ki] = self._q0j[ki] + p["knee_crouch"] + p["knee_amp"] * (0.5 - 0.5 * c) \
                - p["push_amp"] * push                                          # knee: crouch, extend to push off
            tgt[ai] = self._q0j[ai] + p["ankle_amp"] * s + p["push_amp"] * 0.5 * push  # ankle: plantarflex to launch
        for arm in self.ARM:                                                    # arms counter-swing (momentum)
            side = 0 if arm == self.ARM[0] else 1
            tgt[arm] = self._q0j[arm] + p["arm_amp"] * np.sin(ph + side * np.pi)
        return tgt

    def _both_feet_airborne(self) -> bool:
        """True FLIGHT: NO contact with the floor at all (fully airborne). Robust to which foot geom touches —
        the anatomical model's foot-body origin sits ~0.22 m up, so body-z is useless; floor contact is truth."""
        for c in range(int(self.data.ncon)):
            con = self.data.contact[c]
            if int(con.geom1) == self._floor or int(con.geom2) == self._floor:
                return False
        return True

    def rollout(self, theta: np.ndarray, seed: int = 0) -> "tuple[float, float, float, float]":
        """One episode; returns (return, net forward, max flight fraction over a stride, upright fraction)."""
        self.reset(seed=seed)
        cfg = self.cfg
        freq = float(np.clip(theta[0], 0.5, 4.0))
        px0 = float(self.data.xpos[self._pelvis, 0])
        ret = 0.0
        n_up = n_flight = 0
        dt = float(self.model.opt.timestep)
        for k in range(cfg.steps):
            ph = 2.0 * np.pi * freq * k * dt
            q_target = self._phase_targets(theta, ph)
            tau = np.empty(self.model.nu)
            for i, (dof, qa) in enumerate(zip(self._act_dof, self._act_qadr)):
                servo = cfg.kp * (q_target[i] - self.data.qpos[qa]) - cfg.kv * self.data.qvel[dof]
                tau[i] = np.clip(servo + float(self.data.qfrc_bias[dof]), -cfg.tau_max, cfg.tau_max)
            self.data.ctrl[:] = tau
            self._mj.mj_step(self.model, self.data)
            up = float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2])
            pz = float(self.data.xpos[self._pelvis, 2])
            flight = self._both_feet_airborne()                                 # BOTH feet airborne (contact-based)
            n_up += int(up > 0.6)
            n_flight += int(flight)
            fwd_rate = float(self.data.cvel[self._pelvis][3])                    # pelvis forward velocity
            ret += (cfg.w_forward * fwd_rate * dt + cfg.w_upright * (up - 1.0)
                    + cfg.w_flight * float(flight) * dt - cfg.w_ctrl * float(np.sum((tau / cfg.tau_max) ** 2)))
            if pz < cfg.fall_z or not np.all(np.isfinite(self.data.qpos)):       # fell → terminate
                ret -= 20.0
                break
        fwd = float(self.data.xpos[self._pelvis, 0]) - px0
        return ret, fwd, n_flight / max(k + 1, 1), n_up / max(k + 1, 1)
