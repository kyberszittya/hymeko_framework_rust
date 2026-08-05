"""Semi-MDP footstep environment — RL over the WBC scaffold (the walking action-space change).

The whole-body controller (`wbc.WholeBodyController`) gives stable low-level tracking; the DCM analysis
gives a marginally-stable fixed-footstep march but the analytical capture-point step adjustment is finicky
for this small-foot robot (`reports/2026-07-29-humanoid-wbc.md`). This env exposes exactly the decision RL
should learn: **where to place the next foot**. One `step()` = one footstep (a semi-MDP option); the WBC
executes it (double-support load transfer → single-support swing to the commanded foothold → land). The
action is a **bounded residual on the nominal (mirror) foothold**, so ``action = 0`` is the certified
fixed-march scaffold (coin-R8 regime: a bounded residual over a scaffold, not learning from scratch), and
the policy learns the small foothold corrections that regulate the DCM into an indefinitely stable gait.

Reward is survival + staying centred (DCM near the walking centre); an episode ends when the humanoid tips.
Reward-independent honesty: ``action = 0`` reproduces the analytical march, so any learned gain is a real
improvement over the scaffold, measurable against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from gymnasium import spaces

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv
from scenarios.humanoid.wbc import Task, WholeBodyController


@dataclass(frozen=True)
class FootstepConfig:
    """Gait + WBC + RL-interface parameters (defaults = the marginally-stable analytical march)."""

    t_step: float = 0.42            # footstep duration (s)
    ds_frac: float = 0.42           # fraction of the step in double support (load transfer)
    kdcm: float = 1.8               # DCM ZMP-tracking gain
    step_h: float = 0.04            # swing-foot apex clearance (m)
    swing_weight: float = 110.0     # WBC swing-foot task weight (raise to actually LIFT the foot, not shuffle)
    wf: float = 0.07                # swing-foot wrench-cost weight (load transfer)
    nominal_dy: float = 0.084       # nominal lateral half-stance (the fixed-march foothold)
    residual_xy: float = 0.05       # action -> foothold residual scale (m), bounded
    forward_stride: float = 0.0     # >0: the nominal foothold ADVANCES +x each step (forward-walking scaffold)
    w_forward: float = 0.0          # reward weight on forward progress (+x pelvis displacement per step)
    forward_cap: float = 0.05       # per-step forward reward is CAPPED here (m) — a single lunge/fall can't game it
    fall_penalty: float = 5.0       # penalty on a fall; large enough that lunging-into-a-fall never pays
    max_footsteps: int = 80         # episode length in footsteps
    fall_uprightness: float = 0.55
    fall_pelvis_z: float = 0.55
    model_src: str = "humanoid.hymeko"   # "humanoid_toe2.hymeko" = the articulated-toe (push-off) model
    toe_off: float = 0.0            # scripted toe-off torque (N·m) on the stance toe during swing (push-off); needs the toe model
    learn_toe: bool = False         # expand the action with a LEARNED toe-off (applied in LATE stance) — the RL finds the push-off
    toe_off_scale: float = 60.0     # action[2] in [-1,1] -> toe-off torque (N·m)
    target_conditioned: bool = False  # append the commanded forward foothold target to the obs (for a
    #   target-conditioned policy that steps WHERE told — see stepping_stone_demo / train_target_footstep)
    w_target: float = 0.0           # reward weight on foot-to-target accuracy (target_conditioned only)


class HumanoidFootstepEnv:
    """gym-style semi-MDP: one ``step`` = one WBC-executed footstep; action = bounded foothold residual.

    # Preconditions the humanoid model emits and the WBC constructs. # Postconditions ``reset`` returns a
    finite obs standing; ``step(a)`` runs one footstep and returns a finite 5-tuple; ``a = 0`` is the
    analytical fixed-march scaffold. # Invariants the WBC guarantees the low-level whole-body tracking.
    """

    def __init__(self, cfg: FootstepConfig | None = None, *, seed: int = 0) -> None:
        self.cfg = cfg or FootstepConfig()
        self._be = HumanoidBalanceEnv(
            BalanceConfig(perturb_lo=0.0, perturb_hi=0.0, model_src=self.cfg.model_src), seed=seed)
        self.model, self.data = self._be.model, self._be.data
        self._mj = mujoco
        self.wbc = WholeBodyController(self.model, self.data, self._be._act_dof, "base")
        self._fl, self._fr, self._pel = self._be._fl, self._be._fr, self._be._pelvis
        self._toe_act = {}                                 # 'L'/'R' -> index (in act_dof) of that foot's toe joint
        for side, jn in (("L", "toe_flex_l"), ("R", "toe_flex_r")):
            jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn))
            if jid >= 0 and int(self.model.jnt_dofadr[jid]) in self._be._act_dof:
                self._toe_act[side] = self._be._act_dof.index(int(self.model.jnt_dofadr[jid]))
        self._q0 = self._be._q0j.copy()
        self._act_q = self._be._act_qadr
        self._dt = float(self.model.opt.timestep)
        self._omega = float(np.sqrt(9.81 / self.cfg.t_step and 0.645)) or 3.9  # set precisely in reset
        self._rng = np.random.default_rng(seed)
        act_dim = 3 if (self.cfg.learn_toe and self._toe_act) else 2   # +1 for the learned toe-off
        self.action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        obs = self.reset(seed=seed)[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    # ---- helpers ----
    def _foot_load(self, body: int) -> float:
        tot = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if body in (self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]):
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, c, f)
                tot += abs(f[0])
        return tot

    def _dcm(self) -> np.ndarray:
        com = np.asarray(self.data.subtree_com[1])
        comv = self.wbc.com_jacobian() @ np.asarray(self.data.qvel)
        return com[:2] + comv[:2] / self._omega

    def _obs(self) -> np.ndarray:
        d = self.data
        com = np.asarray(d.subtree_com[1])
        stance_b = self._fl if self._stance == "L" else self._fr
        sf = np.asarray(d.xpos[stance_b])
        xi = self._dcm()
        m = d.xmat[self._pel].reshape(3, 3)
        tail = [com[2] - self._zc, m[2, 2], m[0, 2], 1.0 if self._stance == "L" else -1.0]
        if self.cfg.target_conditioned:                    # forward target offset from the stance foot
            tx = self._plan_forward_x if self._plan_forward_x is not None else float(sf[0])
            tail.append(float(tx) - float(sf[0]))
        return np.concatenate([
            xi - sf[:2],                                   # DCM relative to stance foot (the key signal)
            com[:2] - sf[:2],                              # CoM relative to stance foot
            (self.wbc.com_jacobian() @ np.asarray(d.qvel))[:2],   # CoM planar velocity
            tail,
        ]).astype(np.float32)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._be.reset(seed=0)
        for _ in range(40):
            self._be.step(np.zeros(self.model.nu, np.float32))
        self._pelR0 = self.data.xmat[self._pel].reshape(3, 3).copy()
        self._zc = float(self.data.subtree_com[1][2])
        self._omega = float(np.sqrt(9.81 / self._zc))
        self._fx = 0.5 * (float(self.data.xpos[self._fl][0]) + float(self.data.xpos[self._fr][0]))
        self._stance = "L"
        self._t = 0
        return self._obs(), {}

    _tick_cb = None                                        # optional callable(env) run after each WBC tick
    _plan_forward_x = None                                 # if set, overrides this step's forward foothold
    #   target (absolute world x) — a planner drives the coarse foothold, the action stays the residual

    def _tick(self, contacts, acc_com, swing_task, force_cost=None, extra_tau=None):
        d = self.data
        _jp, jr = self.wbc.body_jacobian(self._pel)
        acc_pel = 140.0 * self.wbc.orientation_error(d.xmat[self._pel].reshape(3, 3), self._pelR0) \
            - 24.0 * (jr @ np.asarray(d.qvel))
        post = self.wbc.posture_task(self._q0, self._act_q, 8.0, 4.0, 0.5)
        tasks = [Task(self.wbc.com_jacobian(), acc_com, 150.0), Task(jr, acc_pel, 40.0), post]
        if swing_task is not None:
            tasks.insert(1, swing_task)
        tau = self.wbc.solve(contacts, tasks, force_cost=force_cost)
        if extra_tau is not None:                          # e.g. a scripted toe-off push-off torque
            for i, t in extra_tau.items():
                tau[i] += t
        d.ctrl[:] = np.clip(tau, -150.0, 150.0)
        mujoco.mj_step(self.model, d)
        if self._tick_cb is not None:                      # optional per-tick hook (rendering, logging)
            self._tick_cb(self)

    def _dcm_com_accel(self, r_anchor: np.ndarray, s_offset: np.ndarray, t: float) -> np.ndarray:
        """DCM tracking of the nominal periodic reference ``ξ_ref = r_anchor + s·e^{ωt}`` (Englsberger law),
        the ZMP clipped into the stance foot; returns the CoM task acceleration ``ω²(CoM − r_zmp)``."""
        d = self.data
        com = np.asarray(d.subtree_com[1])
        comv = self.wbc.com_jacobian() @ np.asarray(d.qvel)
        xi = com[:2] + comv[:2] / self._omega
        w = self._omega
        xi_ref = r_anchor + s_offset * np.exp(w * t)
        xi_ref_dot = w * s_offset * np.exp(w * t)
        rzmp = xi_ref - xi_ref_dot / w + (1.0 + self.cfg.kdcm / w) * (xi - xi_ref)
        half = np.array([0.09, 0.05])
        rzmp = np.clip(rzmp, r_anchor - half, r_anchor + half)
        axy = w ** 2 * (com[:2] - rzmp)
        return np.array([axy[0], axy[1], 400.0 * (self._zc - com[2]) - 40.0 * comv[2]])

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        stance_b = self._fl if self._stance == "L" else self._fr
        swing = "R" if self._stance == "L" else "L"
        swing_b = self._fr if swing == "R" else self._fl
        # sagittal anchor tracks the ACTUAL stance foot (so the DCM reference advances when walking forward)
        r_anchor = np.array([float(self.data.xpos[stance_b][0]), float(self.data.xpos[stance_b][1])])
        sw0 = np.asarray(self.data.xpos[swing_b]).copy()
        pel_x0 = float(self.data.xpos[self._pel][0])
        # nominal (scaffold) foothold: mirror-lateral, advanced forward by forward_stride; action = bounded residual
        nominal = np.array([r_anchor[0] + cfg.forward_stride,
                            -np.sign(r_anchor[1]) * cfg.nominal_dy if r_anchor[1] != 0
                            else (cfg.nominal_dy if swing == "L" else -cfg.nominal_dy)])
        if self._plan_forward_x is not None:               # a planner commands the coarse forward foothold
            nominal[0] = float(self._plan_forward_x)
        foothold = nominal + cfg.residual_xy * a[:2]
        toe_cmd = float(a[2]) if a.shape[0] >= 3 else 0.0    # learned toe-off magnitude for this step
        # nominal periodic DCM offset from the stance foot (sagittal centred, lateral toward the next foot)
        s_offset = np.array([0.0, -2.0 * r_anchor[1] / (1.0 + np.exp(self._omega * cfg.t_step))])
        n_tick = int(cfg.t_step / self._dt)
        fell = False
        for k in range(n_tick):
            frac = k / n_tick
            acc_com = self._dcm_com_accel(r_anchor, s_offset, k * self._dt)
            if frac < cfg.ds_frac:
                ramp = frac / cfg.ds_frac
                self._tick([stance_b, swing_b], acc_com, None, force_cost=(slice(6, 12), cfg.wf * ramp))
            else:
                ph = (frac - cfg.ds_frac) / (1.0 - cfg.ds_frac)
                zt = sw0[2] + cfg.step_h * np.sin(np.pi * ph)
                yt = sw0[1] + (foothold[1] - sw0[1]) * min(1.0, ph * 1.3)
                xt = sw0[0] + (foothold[0] - sw0[0]) * min(1.0, ph * 1.3)
                jp_sw, _ = self.wbc.body_jacobian(swing_b)
                fpos = np.asarray(self.data.xpos[swing_b])
                fvel = jp_sw @ np.asarray(self.data.qvel)
                acc_sw = 600.0 * (np.array([xt, yt, zt]) - fpos) - 48.0 * fvel
                extra = None
                if self._stance in self._toe_act:          # toe-off push-off on the stance toe
                    toe_t = 0.0
                    if cfg.learn_toe and ph > 0.5:          # LEARNED toe-off, ramped over LATE stance (roll-off)
                        toe_t = toe_cmd * cfg.toe_off_scale * (ph - 0.5) / 0.5
                    elif cfg.toe_off != 0.0:               # or a fixed scripted toe-off
                        toe_t = cfg.toe_off * float(np.sin(np.pi * ph))
                    if toe_t != 0.0:
                        extra = {self._toe_act[self._stance]: toe_t}
                self._tick([stance_b], acc_com, Task(jp_sw, acc_sw, cfg.swing_weight), extra_tau=extra)
            if self._be._com_sig()["uprightness"] < cfg.fall_uprightness:
                fell = True
                break
        self._t += 1
        self._stance = swing
        sig = self._be._com_sig()
        upright = sig["uprightness"] > cfg.fall_uprightness \
            and float(self.data.xpos[self._pel, 2]) > cfg.fall_pelvis_z
        fell = fell or not upright or not np.all(np.isfinite(self.data.qpos))
        xi = self._dcm()
        centre_off = float(abs(xi[1]))                     # DCM LATERAL distance from the walking centre
        fwd = float(self.data.xpos[self._pel, 0]) - pel_x0   # forward (+x) pelvis progress this footstep
        fwd_r = float(np.clip(fwd, -cfg.forward_cap, cfg.forward_cap))   # capped: a single lunge/fall can't game it
        reward = (1.0 - 2.0 * centre_off - 0.01 * float(a @ a)
                  + cfg.w_forward * fwd_r - (cfg.fall_penalty if fell else 0.0))
        if cfg.target_conditioned and self._plan_forward_x is not None:
            swung_b = self._fl if self._stance == "L" else self._fr   # the foot that just landed
            reward -= cfg.w_target * abs(float(self.data.xpos[swung_b, 0]) - float(self._plan_forward_x))
        done = fell
        trunc = self._t >= cfg.max_footsteps
        return self._obs(), float(reward), bool(done), bool(trunc), {
            "centre_off": centre_off, "steps": self._t, "forward": fwd,
            "pelvis_x": float(self.data.xpos[self._pel, 0])}
