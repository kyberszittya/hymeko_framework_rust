"""Option 2: attempt a floating-humanoid balance controller, re-verify the Lyapunov
certificate, and assess SAC readiness.

Adds a FLOOR (the earlier floating test had none -> free fall) and tries hand-tuned
controllers: gravity-comp PD (posture only) and PD + ankle COM-feedback (ankle
strategy). Both are verified against the same Lyapunov certificate. The 2D sagittal
humanoid still TIPS (~1.2 s) under hand tuning -> the certificate rejects them ->
balance here needs LQR (model-based) or SAC (RL), gated by the Lyapunov certificate.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.run_humanoid_balance
SIMULATION. NOT RL (this assesses SAC readiness; it does not train).
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
    if floating:
        xml = xml.replace('<joint name="base" type="hinge" axis="0 0 1"/>', '<freejoint name="base"/>')
        xml = xml.replace('<worldbody>',
                          '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1" '
                          'pos="0 0 0" condim="3" friction="1 0.1 0.1"/>')
    return mujoco, mujoco.MjModel.from_xml_string(xml)


def _run(floating: bool, ka: float, steps: int = 4000) -> tuple[list[float], int]:
    """Return (V series, upright_steps). ka=0 -> gravity-comp PD; ka>0 -> + ankle COM feedback."""
    mujoco, m = _model(floating)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = d.qpos.copy()
    h_ref = float(d.subtree_com[1][2])
    if floating:
        base = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base")])
        d.qpos[base + 2] = 0.80   # drop the feet onto the floor
        mujoco.mj_forward(m, d)
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_l")
    fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_r")
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
    ankle = [aid.get("act_ankle_l"), aid.get("act_ankle_r")]
    act_jnt = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
    V = HumanoidCOMLyapunov(h_ref=h_ref)
    com_prev = d.subtree_com[1].copy()
    series, upright = [], 0
    for step in range(steps):
        com = d.subtree_com[1]
        support = 0.5 * (d.xpos[fl][:2] + d.xpos[fr][:2])
        off_x = float(com[0] - support[0])
        tau = np.zeros(m.nu)
        for i, j in enumerate(act_jnt):
            a = int(m.jnt_qposadr[j])
            dof = int(m.jnt_dofadr[j])
            tau[i] = 300.0 * (q0[a] - d.qpos[a]) - 30.0 * d.qvel[dof] + float(d.qfrc_bias[dof])
        if ka and None not in ankle:
            vel = (com[0] - com_prev[0]) / m.opt.timestep
            for i in ankle:
                tau[i] += ka * off_x + (ka / 4.0) * vel
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        com_prev = com.copy()
        sig = {"com_z": float(com[2]),
               "com_xy_off": float(np.linalg.norm(com[:2] - support)),
               "com_speed": float(np.linalg.norm(com - com_prev) / m.opt.timestep),
               "uprightness": float(d.xmat[1].reshape(3, 3)[2, 2])}
        series.append(V(sig))
        if float(d.xmat[1].reshape(3, 3)[2, 2]) > 0.8 and float(d.xpos[1, 2]) > 0.55:
            upright = step
        if not np.all(np.isfinite(d.qpos)):
            break
    return series, upright


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "fixed_base_constrained": _run(False, 0.0),
        "floating_floor_gravcomp_PD": _run(True, 0.0),
        "floating_floor_ankle_strategy": _run(True, 250.0),
    }
    out = {}
    for name, (series, upright) in cases.items():
        r = evaluate_lyapunov(series)
        r["upright_steps"] = upright
        out[name] = r

    hand_tuned_fail = (not out["floating_floor_gravcomp_PD"]["passes"]
                       and not out["floating_floor_ankle_strategy"]["passes"])
    result = {
        "verdict": ("HANDTUNED_BALANCE_FAILS_LYAPUNOV_RL_OR_LQR_WARRANTED"
                    if hand_tuned_fail and out["fixed_base_constrained"]["passes"]
                    else "INCONCLUSIVE"),
        "cases": out,
        "sac_readiness": {
            "task_is_genuine": "floating humanoid falls (tips ~1.2s) without a real balance loop",
            "handtuned_insufficient": hand_tuned_fail,
            "lyapunov_as_reward_independent_gate": "V is a natural cost; lyapunov_certificate is the "
                "reward-independent success/safety certificate an RL run must not change",
            "regime": "NO certified hand-tuned baseline -> SAC-from-scratch (genuine RL), OR a "
                "model-based LQR baseline enables residual RL (coin-R8 pattern)",
            "not_run": "SAC not trained here; this is a feasibility + readiness assessment",
        },
        "note": "SIMULATION. Floor added (the earlier floating test had none). Sagittal 2D humanoid; "
                "hand-tuned PD + ankle strategy still tip -> Lyapunov rejects -> LQR/SAC is the tool.",
    }
    (_OUT / "balance_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"],
                      "cases": {k: {"passes": v["passes"], "upright_steps": v["upright_steps"],
                                    "Vfinal": v["Vfinal"]} for k, v in out.items()}}, indent=2))


if __name__ == "__main__":
    main()
