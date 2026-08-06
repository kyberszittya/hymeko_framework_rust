"""Plot the spring-legged AIBO: elastic knees launch the body airborne where rigid legs cannot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv  # noqa: E402

from .spring_leg import (  # noqa: E402
    KNEE_MOTOR_TAU_REALISTIC,
    LEGS,
    SpringLegSpec,
    build_spring_legged,
)

_OUT = Path("reports/2026-07-27-aibo-hop")


def _torso_trace(model: mujoco.MjModel, load_angle: float, steps: int = 220) -> np.ndarray:
    d = mujoco.MjData(model)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    base = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base")])
    knee_q = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_{n}")])
              for n in LEGS}
    mujoco.mj_forward(model, d)
    d.qpos[base + 2] = 0.2
    for n in LEGS:
        d.qpos[knee_q[n]] = load_angle
    mujoco.mj_forward(model, d)
    zs = [float(d.xpos[torso, 2])]
    for _ in range(steps):
        mujoco.mj_step(model, d)
        zs.append(float(d.xpos[torso, 2]))
    z = np.array(zs)
    return z - z[0]


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                           max_steps=200)
    mjcf = env._mjcf
    rigid = mujoco.MjModel.from_xml_string(mjcf)
    spring = build_spring_legged(mjcf, SpringLegSpec(stiffness=150.0, springref=0.0, damping=0.05))
    dt = float(rigid.opt.timestep)
    rigid_z = _torso_trace(rigid, -1.0)
    spring_z = _torso_trace(spring, -1.0)
    t = np.arange(len(rigid_z)) * dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, spring_z, "-", color="#2266cc", lw=2, label="spring leg (series-elastic knee)")
    axes[0].plot(t[:len(rigid_z)], rigid_z, "--", color="#cc6622", lw=2, label="rigid geared leg")
    axes[0].axhline(0.03, color="g", ls=":", lw=0.9)
    axes[0].text(t[-1] * 0.5, 0.037, "paw-clearance threshold", color="g", fontsize=8)
    axes[0].set(title="release the loaded knees — torso launch", xlabel="time (s)",
                ylabel="torso rise (m)")
    axes[0].legend(fontsize=8)

    ks = np.linspace(20, 320, 60)
    ceil_cm = np.array([4 * SpringLegSpec(stiffness=float(k)).static_load_limit(
        KNEE_MOTOR_TAU_REALISTIC)[1] for k in ks]) / (5.705 * 9.81) * 100.0
    axes[1].plot(ks, ceil_cm, color="#993399", lw=2)
    axes[1].set(title=f"static-load hop ceiling — realistic {KNEE_MOTOR_TAU_REALISTIC:g} N·m motor",
                xlabel="knee spring stiffness K (N·m/rad)", ylabel="motor-loadable hop (cm)")
    axes[1].text(120, ceil_cm.max() * 0.6,
                 "a small motor can only statically\nload a few cm — a launch spring\nmust be loaded"
                 " dynamically (body weight)", fontsize=8, color="#993399")

    fig.suptitle("Spring-legged AIBO — elastic knees store & return launch energy (SIMULATION)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_OUT / "spring_leg.png", dpi=110)
    print("wrote", _OUT / "spring_leg.png")


if __name__ == "__main__":
    main()
