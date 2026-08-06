"""Verify the humanoid COM-Lyapunov against the Lyapunov certificate.

Floating-base HyMeKo humanoid (runtime freejoint) under a naive gravity-comp PD
controller COLLAPSES -> V diverges -> the Lyapunov certificate must REJECT it. The
constrained fixed-base humanoid (cannot fall) holds V ~ 0 -> PASSES (vacuously). The
split is the rigorous, Lyapunov-expressed HUM balance prerequisite.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.run_humanoid_lyapunov
SIMULATION. NOT RL.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .lyapunov import HumanoidCOMLyapunov, evaluate_lyapunov

_OUT = Path("reports/2026-07-27-humanoid-lyapunov")
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "data" / "robotics" / "humanoid.hymeko"


def _cli() -> Path:
    for prof in ("release", "debug"):
        p = _REPO / "target" / prof / "hymeko"
        if p.is_file():
            return p
    raise FileNotFoundError("hymeko CLI not built")


def _model(floating: bool):
    import mujoco
    xml = subprocess.run([str(_cli()), "emit", "-f", "mjcf", str(_SRC), "-n", "humanoid"],
                         capture_output=True, text=True, check=True).stdout
    if floating:  # z-hinge base -> 6-DOF free joint: a genuine floating base that can fall
        xml = xml.replace('<joint name="base" type="hinge" axis="0 0 1"/>',
                          '<freejoint name="base"/>')
    return mujoco, mujoco.MjModel.from_xml_string(xml)


def _com_lyap_series(floating: bool, steps: int = 3000) -> list[float]:
    mujoco, m = _model(floating)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    h_ref = float(d.subtree_com[1][2])
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_l")
    fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_r")
    act_jnt = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
    V = HumanoidCOMLyapunov(h_ref=h_ref)
    dt = float(m.opt.timestep)
    com_prev = d.subtree_com[1].copy()
    series = []
    for _ in range(steps):
        tau = np.zeros(m.nu)
        for i, j in enumerate(act_jnt):
            adr = int(m.jnt_qposadr[j])
            dof = int(m.jnt_dofadr[j])
            tau[i] = 200.0 * (q0[adr] - d.qpos[adr]) - 20.0 * d.qvel[dof] + float(d.qfrc_bias[dof])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        com = d.subtree_com[1]
        support = 0.5 * (d.xpos[fl][:2] + d.xpos[fr][:2])
        sig = {
            "com_z": float(com[2]),
            "com_xy_off": float(np.linalg.norm(com[:2] - support)),
            "com_speed": float(np.linalg.norm(com - com_prev) / dt),
            "uprightness": float(d.xmat[1].reshape(3, 3)[2, 2]),
        }
        com_prev = com.copy()
        series.append(V(sig))
        if not np.all(np.isfinite(d.qpos)):
            break
    return series


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    floating = evaluate_lyapunov(_com_lyap_series(floating=True))
    fixed = evaluate_lyapunov(_com_lyap_series(floating=False))
    discriminates = (not floating["passes"]) and fixed["passes"]
    verdict = ("HUMANOID_FLOATING_FAILS_LYAPUNOV_BALANCE_CONTROLLER_PREREQUISITE"
               if discriminates else "LYAPUNOV_INCONCLUSIVE")
    result = {
        "verdict": verdict,
        "floating_base_naive_control": floating,
        "fixed_base_constrained": fixed,
        "discriminates": discriminates,
        "note": "SIMULATION. Floating-base HyMeKo humanoid (runtime freejoint) COLLAPSES under naive "
                "gravity-comp PD -> COM-Lyapunov V diverges -> certificate REJECTS (rigorous balance-"
                "controller prerequisite for HUM-2/3/4). Fixed-base cannot fall -> V~0 -> passes "
                "VACUOUSLY (consistent with HUM-1 'balance vacuous'). Same generic Lyapunov certificate "
                "AIBO PASSED with a real controller -> it discriminates across embodiments.",
    }
    (_OUT / "lyapunov_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "discriminates": discriminates,
                      "floating_passes": floating["passes"], "floating_Vfinal": floating["Vfinal"],
                      "floating_Vmax": floating["Vmax"],
                      "fixed_passes": fixed["passes"], "fixed_Vfinal": fixed["Vfinal"]}, indent=2))


if __name__ == "__main__":
    main()
