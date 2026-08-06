"""CIP-HUM-01 adapter: fixed-base HyMeKo humanoid + scripted PD reach controller.

Depends only on the frozen core ``hymeko_control`` and on ``hymeko_rl`` assets /
the ``hymeko`` CLI (to emit the humanoid MJCF). The core never imports this.

HONESTY CONTRACT: ``data/robotics/humanoid.hymeko`` is FIXED-BASE (welded pelvis),
so the balance / support-margin / no-fall certificate is VACUOUS and is NOT a
genuine test. This adapter genuinely certifies only the REACH (effector reaches a
reachable target config) + joint/velocity limits + no divergence. The balance
gates are reported as blocked, not passed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_control.cip.authority import (
    AuthorityChannel,
    AuthorityMap,
    AuthoritySource,
)
from hymeko_control.cip.certificate import CertificateResult
from hymeko_control.cip.option import (
    AffineAuthorityDecoder,
    ExecutableOption,
    OptionEnd,
    ResponseTrace,
)
from hymeko_control.cip.physical_intent import PhysicalIntent
from hymeko_control.cip.structured_state import ControlState, StructuredStateLike
from hymeko_control.language.ir import ControlModel

from .certificate import hum_certificate_suite

CHAIN_H = ("STAND", "SHIFT_SUPPORT", "REACH", "TOUCH", "RETRACT", "RECOVER")
_INTENT_ORDER = (
    "com_shift", "support_margin_demand", "ee_target_velocity", "torso_orientation",
    "contact_force_demand", "damping_demand", "recovery_demand",
)
_REPO = Path(__file__).resolve().parents[2]
_HUMANOID_SRC = _REPO / "data" / "robotics" / "humanoid.hymeko"


def _cli() -> Path:
    for prof in ("release", "debug"):
        p = _REPO / "target" / prof / "hymeko"
        if p.is_file():
            return p
    raise FileNotFoundError("hymeko CLI not built; run `cargo build -p hymeko_cli`")


class HumanoidSim:
    """Thin MuJoCo wrapper over the emitted fixed-base humanoid."""

    def __init__(self) -> None:
        import mujoco

        xml = subprocess.run(
            [str(_cli()), "emit", "-f", "mjcf", str(_HUMANOID_SRC), "-n", "humanoid"],
            capture_output=True, text=True, check=True,
        ).stdout
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.q0 = self.data.qpos.copy()
        self._ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "forearm_l")
        self._act_jnt = [int(self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)]

    def ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee].copy()

    def fk_ee(self, qpos: np.ndarray) -> np.ndarray:
        d = self._mj.MjData(self.model)
        d.qpos[:] = qpos
        self._mj.mj_forward(self.model, d)
        return d.xpos[self._ee].copy()

    def qadr(self, joint: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, joint)
        return int(self.model.jnt_qposadr[jid])

    def pd_step(self, q_target: np.ndarray, kp: float, kd: float) -> None:
        # computed-torque: PD error + gravity/Coriolis feedforward (qfrc_bias),
        # so the reach converges to the commanded config instead of drooping.
        tau = np.zeros(self.model.nu)
        for i, j in enumerate(self._act_jnt):
            adr = int(self.model.jnt_qposadr[j])
            dof = int(self.model.jnt_dofadr[j])
            tau[i] = (kp * (q_target[adr] - self.data.qpos[adr])
                      - kd * self.data.qvel[dof]
                      + float(self.data.qfrc_bias[dof]))
        self.data.ctrl[:] = tau
        self._mj.mj_step(self.model, self.data)

    def diverged(self) -> bool:
        return not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
                    and float(np.abs(self.data.qvel).max()) < 50.0)

    def limits_ok(self) -> bool:
        for j in range(self.model.njnt):
            if not self.model.jnt_limited[j]:
                continue
            adr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            if hi > lo and not (lo - 1e-3 <= self.data.qpos[adr] <= hi + 1e-3):
                return False
        return True


@dataclass
class HumanoidCIPAdapter:
    """CIP-0 adapter over the fixed-base humanoid; scripted PD stand-reach-recover."""

    model: ControlModel
    kp: float = 120.0
    kd: float = 8.0
    reach_tol: float = 0.04
    home_tol: float = 0.04
    settle_steps: int = 200
    dwell_steps: int = 150
    max_mode_steps: int = 3000

    _sim: HumanoidSim = field(default=None, init=False, repr=False)
    _mode: str = field(default="STAND", init=False, repr=False)
    _next: str = field(default="STAND", init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _q_reach: Any = field(default=None, init=False, repr=False)
    _target: Any = field(default=None, init=False, repr=False)
    _home_pos: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sim = HumanoidSim()
        self._bounds = self.model.intent_bounds()
        q_reach = self._sim.q0.copy()
        q_reach[self._sim.qadr("shoulder_l")] = 1.0
        q_reach[self._sim.qadr("elbow_l")] = -1.3
        self._q_reach = q_reach
        self._target = self._sim.fk_ee(q_reach)
        self._home_pos = self._sim.ee_pos()

    # -- signals --------------------------------------------------------
    def _signals(self) -> dict[str, float]:
        ee = self._sim.ee_pos()
        return {
            "ee_to_target": float(np.linalg.norm(ee - self._target)),
            "ee_to_home": float(np.linalg.norm(ee - self._home_pos)),
            "max_qvel": float(np.abs(self._sim.data.qvel).max()),
            "torso_z": float(self._sim.data.xpos[:, 2].max()),
            "limits_ok": 1.0 if self._sim.limits_ok() else 0.0,
            "diverged": 1.0 if self._sim.diverged() else 0.0,
        }

    def _q_target_for(self, mode: str) -> np.ndarray:
        return self._q_reach if mode in ("REACH", "TOUCH") else self._sim.q0

    # -- CIP-0 protocol -------------------------------------------------
    def observe(self) -> StructuredStateLike:
        s = self._signals()
        state = ControlState(t=self._tick, phase=self._mode, signals=s,
                             contact={"upright": s["torso_z"] > 0.6})
        self._tick += 1
        return state

    def identify_mode(self, state: StructuredStateLike) -> str:
        return self._mode

    def form_intent(self, state: StructuredStateLike, task: Any) -> PhysicalIntent:
        s = self._signals()
        comps = {
            "com_shift": 0.0,
            "support_margin_demand": 1.0,  # demanded; vacuous on fixed base
            "ee_target_velocity": 1.0 if self._mode in ("REACH", "TOUCH") else 0.0,
            "torso_orientation": 0.0,
            "contact_force_demand": 1.0 if self._mode == "TOUCH" else 0.0,
            "damping_demand": min(s["max_qvel"] / 10.0, 1.0),
            "recovery_demand": 1.0 if self._mode in ("RETRACT", "RECOVER") else 0.0,
        }
        return PhysicalIntent.clipped(components=comps, bounds=self._bounds)

    def measure_authority(self, state: StructuredStateLike, mode: str) -> AuthorityMap:
        return AuthorityMap(channels={
            "support_authority": AuthorityChannel(
                "support_authority", 1.0, AuthoritySource.ASSUMED,
                "VACUOUS on fixed base: welded pelvis, no fall possible"),
            "reach_authority": AuthorityChannel(
                "reach_authority", 1.0, AuthoritySource.OBSERVED,
                "left-arm joint-limit reachability of the target config"),
            "actuation_authority": AuthorityChannel(
                "actuation_authority", 1.0, AuthoritySource.OBSERVED,
                "PD torque headroom on actuated joints"),
        })

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption:
        decoder = AffineAuthorityDecoder(
            name="hum_scripted_pd", order=_INTENT_ORDER,
            gain=tuple(1.0 for _ in _INTENT_ORDER), mode=self._mode,
            initiation=frozenset(CHAIN_H), termination="stand_reach_recover",
        )
        opt = decoder.decode(intent, authority)
        prov = dict(opt.provenance)
        prov["realizer"] = "scripted PD reach (fixed-base humanoid)"
        return ExecutableOption(opt.name, opt.mode, opt.command, opt.initiation,
                                opt.termination, prov)

    def _exit_condition(self, mode: str, s: dict[str, float], steps: int) -> bool:
        if mode == "STAND":
            return steps >= self.settle_steps
        if mode == "SHIFT_SUPPORT":
            return steps >= self.settle_steps // 2
        if mode == "REACH":
            return s["ee_to_target"] <= self.reach_tol
        if mode == "TOUCH":
            return steps >= self.dwell_steps
        if mode == "RETRACT":
            return s["ee_to_home"] <= self.home_tol
        return steps >= self.dwell_steps  # RECOVER

    def execute(self, option: ExecutableOption) -> ResponseTrace:
        q_target = self._q_target_for(self._mode)
        signals: list[dict[str, float]] = []
        commands: list[tuple[float, ...]] = []
        diverged = False
        limits_ok = True
        steps = 0
        for steps in range(1, self.max_mode_steps + 1):
            self._sim.pd_step(q_target, self.kp, self.kd)
            s = self._signals()
            signals.append(s)
            commands.append(tuple(float(x) for x in self._sim.data.ctrl))
            limits_ok = limits_ok and bool(s["limits_ok"])
            if s["diverged"]:
                diverged = True
                break
            if self._exit_condition(self._mode, s, steps):
                break
        idx = CHAIN_H.index(self._mode)
        reached = self._mode == "REACH" and signals and signals[-1]["ee_to_target"] <= self.reach_tol
        if diverged:
            self._next, end = self._mode, OptionEnd.ABORTED
        elif self._mode == "RECOVER":
            self._next, end = "RECOVER", OptionEnd.COMPLETED
        else:
            self._next = CHAIN_H[idx + 1]
            end = OptionEnd.HANDOFF
        last = signals[-1] if signals else self._signals()
        recover_home_ok = (
            self._mode == "RECOVER" and last["ee_to_home"] <= self.home_tol
            and not diverged
        )
        return ResponseTrace(
            option=option.name, commands=tuple(commands), signals=tuple(signals),
            end=end,
            provenance={
                "option": option.name, "mode": self._mode, "steps": steps,
                "reached_target": bool(reached or last["ee_to_target"] <= self.reach_tol),
                "recover_home_ok": bool(recover_home_ok),
                "ee_to_target": last["ee_to_target"], "ee_to_home": last["ee_to_home"],
                "limits_ok": limits_ok, "diverged": diverged,
                "max_qvel": max((s["max_qvel"] for s in signals), default=0.0),
                "torso_upright": last["torso_z"] > 0.6,
            },
        )

    def certify(self, state: StructuredStateLike, trace: ResponseTrace) -> CertificateResult:
        return hum_certificate_suite().evaluate(state, trace)

    def transition(self, mode: str, certificate: CertificateResult) -> str:
        nxt = self._next
        if nxt != mode and not self.model.is_legal_transition(mode, nxt):
            nxt = mode
        self._mode = nxt
        return nxt
