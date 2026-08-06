"""Floating-base HyMeKo humanoid balance env (gym 5-tuple), position-servo action.

Two measured corrections over the first (torque) version, both diagnosed by
measurement, not guessed (`reports/2026-07-27-humanoid-sac-diagnosis`):

1. **`h_ref` was mis-referenced** — the reward/Lyapunov reference was 0.818 (a
   pelvis-top height) but the true *standing COM* height is ~0.645, so the old
   reward asked the humanoid to lift its COM 14 cm above where it stands. Fixed.
2. **Torque action model had too little authority** (SAC saturated ±50 N·m on 40 %
   of steps). Replaced with a **position-servo (PD-to-target) action space**:
   ``tau = clip(kp·(q0 + a·delta − q) − kv·qvel + qfrc_bias, ±tau_max)``. The action
   commands a bounded joint-target *offset* around the nominal pose; a stiff PD
   tracks it. This is the standard high-authority action space for RL balance.

By construction **a = 0 is the PD-hold-``q0`` controller**, which is a *certified*
balance baseline (passes the unchanged Lyapunov certificate for pitch-rate
perturbations up to ~0.30). SAC therefore learns a **bounded residual over a
certified scaffold** (coin-R8 regime), not from scratch. The reward is driven by the
COM Lyapunov energy; the reward-independent ``lyapunov_certificate`` is the eval-only
gate. Gains (kp=60, kv=10) are the stable ceiling for this model's Euler @ 1 ms
integrator — kp≳150 diverges.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .lyapunov import HumanoidCOMLyapunov

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "data" / "robotics" / "humanoid.hymeko"


@dataclass(frozen=True)
class BalanceConfig:
    """Env + position-servo + task parameters (all measured/justified in the module doc)."""

    max_steps: int = 500
    h_ref: float = 0.645             # measured standing COM height (was mis-set 0.818)
    kp: float = 60.0                 # PD-target stiffness (stable ceiling ~80 at Euler 1 ms)
    kv: float = 10.0                 # PD-target damping
    tau_max: float = 150.0           # per-joint torque limit (N·m)
    delta_scale: float = 0.4         # action -> joint-target offset (rad) around q0
    base_z0: float = 0.80            # reset base height (feet in contact, COM over support)
    perturb_lo: float = 0.0          # reset pitch-rate perturbation range (rad/s), sign random
    perturb_hi: float = 0.3
    push_lat_lo: float = 0.0         # reset LATERAL push range (base y-velocity m/s), sign random
    push_lat_hi: float = 0.0         # >0 exercises the frontal-plane (abduction/roll) protective response
    w_step: float = 0.0              # capture-point step-shaping reward weight (>0 encourages a protective step)
    w_velocity: float = 0.0          # reward weight on forward (+x) base velocity — >0 turns the balance
    #   task into a DIRECT LOCOMOTION task on the position-servo action (see train_balance_walk)
    vel_cap: float = 0.6             # forward-velocity reward is capped here (m/s) so a lunge/fall can't game it
    torque_action: bool = False      # action = DIRECT gravity-compensated torque (no q0 anchoring) instead of
    #   the position-servo — lets a policy leave the standing pose and SUSTAIN a walking gait
    w_stability: float = 0.0         # viability-band HEALTHY shaping: a HINGE penalty applied only when the
    #   torso leans PAST `upright_safe` (exiting the hysteresis region toward a fall) — NOT a continuous
    #   lean penalty, so a dynamic gait may lean freely within the safe band. Keeps the walk from toppling.
    upright_safe: float = 0.68       # inner (hysteresis) uprightness threshold: a NARROW band just above
    #   fall_uprightness (0.6), so the hinge fires only near the fall — NOT during the normal walking lean
    periodic_gait: bool = False      # PERIODIC-GAIT PRIOR: a phase clock in the obs + a cyclic reward that
    #   rewards L/R foot alternation in sync with the phase, so the policy learns a REPEATING gait (a
    #   sustained cycle) rather than a one-shot dynamic lunge. Default off (obs/reward unchanged).
    gait_period: int = 400           # env steps per full L→R gait cycle (phase advances 2π/gait_period/step)
    w_gait: float = 0.0              # reward weight on phase-synchronised foot alternation (periodic_gait only)
    fall_uprightness: float = 0.6
    fall_pelvis_z: float = 0.55
    model_src: str = "humanoid.hymeko"   # model variant to emit (e.g. "humanoid_toe.hymeko" for the toe/push-off model)


def _cli() -> Path:
    for prof in ("release", "debug"):
        p = _REPO / "target" / prof / "hymeko"
        if p.is_file():
            return p
    raise FileNotFoundError("hymeko CLI not built")


def _build(model_src: str = "humanoid.hymeko"):
    import mujoco
    src = _REPO / "data" / "robotics" / model_src
    name = Path(model_src).stem                        # the .hymeko model block name = the file stem
    xml = subprocess.run([str(_cli()), "emit", "-f", "mjcf", str(src), "-n", name],
                         capture_output=True, text=True, check=True).stdout
    xml = xml.replace('<joint name="base" type="hinge" axis="0 0 1"/>', '<freejoint name="base"/>')
    xml = xml.replace('<motor name="act_base" joint="base" gear="1"/>\n    ', '')  # unactuated base
    xml = xml.replace('<worldbody>',
                      '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1" '
                      'pos="0 0 0" condim="3" friction="1 0.1 0.1"/>')
    return mujoco, mujoco.MjModel.from_xml_string(xml)


class HumanoidBalanceEnv:
    """gym-5-tuple floating humanoid balance. Action = bounded joint-target offset (12).

    Preconditions: the emitted humanoid model exists and the CLI is built.
    Postconditions: ``reset`` returns a finite obs at a feet-on-floor pose; ``step``
    applies a stable position-servo and returns a finite 5-tuple. Invariant: a = 0 is
    the certified PD-hold-``q0`` scaffold.
    """

    def __init__(self, cfg: BalanceConfig | None = None, *, max_steps: int | None = None,
                 seed: int = 0) -> None:
        self.cfg = cfg or BalanceConfig()
        if max_steps is not None:                                      # back-compat kwarg
            self.cfg = BalanceConfig(**{**self.cfg.__dict__, "max_steps": max_steps})
        self._mj, self.model = _build(self.cfg.model_src)
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
        self._q0j = np.array([self.data.qpos[a] for a in self._act_qadr])   # nominal joint pose
        self._h_ref = self.cfg.h_ref
        self.V = HumanoidCOMLyapunov(h_ref=self._h_ref)
        self.max_steps = self.cfg.max_steps
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._phase = 0.0                                               # periodic-gait phase clock
        obs = self._obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)

    def _com_sig(self) -> dict:
        com = self.data.subtree_com[1]
        support = 0.5 * (self.data.xpos[self._fl][:2] + self.data.xpos[self._fr][:2])
        return {"com_z": float(com[2]),
                "com_xy_off": float(np.linalg.norm(com[:2] - support)),
                "com_speed": float(np.linalg.norm(self.data.cvel[1][:3])),
                "uprightness": float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2])}

    def _obs(self) -> np.ndarray:
        m = self.data.xmat[self._pelvis].reshape(3, 3)
        com = self.data.subtree_com[1]
        support = 0.5 * (self.data.xpos[self._fl][:2] + self.data.xpos[self._fr][:2])
        jq = np.array([self.data.qpos[a] for a in self._act_qadr])
        jv = np.array([self.data.qvel[dof] for dof in self._act_dof])
        tail = [float(com[0] - support[0]), float(self.data.cvel[1][0])]
        if self.cfg.periodic_gait:                                     # phase clock: where in the gait cycle
            tail += [float(np.sin(self._phase)), float(np.cos(self._phase))]
        return np.concatenate([
            [m[2, 2], m[0, 2], float(self.data.xpos[self._pelvis, 2]) - self._h_ref],
            self.data.qvel[0:6], jq, jv, tail,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.data.qpos[:] = self._q0
        self.data.qpos[self._base + 2] = self.cfg.base_z0
        self.data.qvel[:] = 0.0
        sign = 1.0 if self._rng.random() < 0.5 else -1.0
        self.data.qvel[4] = sign * float(self._rng.uniform(self.cfg.perturb_lo, self.cfg.perturb_hi))
        if self.cfg.push_lat_hi > 0.0:                          # lateral push (exercises frontal DOF)
            lsign = 1.0 if self._rng.random() < 0.5 else -1.0
            self.data.qvel[1] = lsign * float(self._rng.uniform(self.cfg.push_lat_lo, self.cfg.push_lat_hi))
        self._mj.mj_forward(self.model, self.data)
        self._t = 0
        self._phase = 0.0
        return self._obs(), {}

    def _capture_step(self) -> tuple[float, float]:
        """LIPM capture-point step shaping: reward a foot placed at the lateral capture point.

        Returns (step_bonus in (0,1], nearest-foot-to-capture-point error). The capture point
        xi_y = com_y + com_y_vel·sqrt(com_z/g) is where a protective foot should land to arrest
        the lateral fall (Pratt/Koolen capturability, the Vukobratović-lineage step criterion).
        """
        com = self.data.subtree_com[1]
        com_y_vel = float(self.data.qvel[1])                       # base lateral velocity ~ COM lateral velocity
        xi_y = float(com[1]) + com_y_vel * float(np.sqrt(max(float(com[2]), 0.1) / 9.81))
        err = min(abs(float(self.data.xpos[self._fl, 1]) - xi_y),
                  abs(float(self.data.xpos[self._fr, 1]) - xi_y))
        return float(np.exp(-err / 0.08)), err

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        tau = np.empty(self.model.nu)
        if self.cfg.torque_action:                                 # DIRECT gravity-compensated torque (no q0
            for i, dof in enumerate(self._act_dof):                #   anchoring) — for a sustained WALKING gait
                cmd = a[i] * self.cfg.tau_max + float(self.data.qfrc_bias[dof])
                tau[i] = np.clip(cmd, -self.cfg.tau_max, self.cfg.tau_max)
        else:
            q_target = self._q0j + a * self.cfg.delta_scale
            for i, (dof, qa) in enumerate(zip(self._act_dof, self._act_qadr)):
                servo = self.cfg.kp * (q_target[i] - self.data.qpos[qa]) - self.cfg.kv * self.data.qvel[dof]
                tau[i] = np.clip(servo + float(self.data.qfrc_bias[dof]), -self.cfg.tau_max, self.cfg.tau_max)
        self.data.ctrl[:] = tau
        self._mj.mj_step(self.model, self.data)
        self._t += 1
        sig = self._com_sig()
        v = self.V(sig)
        upright = (sig["uprightness"] > self.cfg.fall_uprightness
                   and float(self.data.xpos[self._pelvis, 2]) > self.cfg.fall_pelvis_z)
        fell = (not upright) or not np.all(np.isfinite(self.data.qpos))
        reward = 1.0 - 2.0 * v - 0.001 * float(np.sum(a * a))       # alive - Lyapunov - control cost
        info = {"V": v, "upright": upright}
        if self.cfg.w_velocity > 0.0:                              # DIRECT LOCOMOTION: reward forward base velocity
            fwd_v = float(np.clip(self.data.qvel[0], -self.cfg.vel_cap, self.cfg.vel_cap))
            reward += self.cfg.w_velocity * fwd_v
            info["fwd_v"] = fwd_v
        if self.cfg.w_stability > 0.0:                            # viability-band HINGE: only when leaning past
            reward -= self.cfg.w_stability * max(0.0, self.cfg.upright_safe - sig["uprightness"])  # the safe band
        if self.cfg.periodic_gait:                                # PERIODIC GAIT: reward L/R stride alternating
            ref = float(np.sin(self._phase))                      #   with the phase clock (a repeating gait), then
            foot_asym = float(self.data.xpos[self._fl, 0] - self.data.xpos[self._fr, 0])  # advance the clock
            reward += self.cfg.w_gait * ref * foot_asym
            self._phase += 2.0 * np.pi / self.cfg.gait_period
        if self.cfg.w_step > 0.0:                                  # capture-point step shaping (opt-in)
            step_bonus, info["capture_err"] = self._capture_step()
            reward += self.cfg.w_step * step_bonus
        return self._obs(), float(reward), bool(fell), bool(self._t >= self.max_steps), info
