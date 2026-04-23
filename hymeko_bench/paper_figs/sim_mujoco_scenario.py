"""MuJoCo simulation of the canonical grasping scenario — Rev. 4 figure.

Runs a 4-DOF revolute arm matching the paper's Listing A.6.1 geometry
(base → shoulder → elbow → wrist → flange) under a classical PID
controller tracking a sine-wave setpoint on each joint. The real-valued
joint positions and velocities collected during the simulation are the
"measurement data" REV4 asked for.

Output: `hymeko_bench/paper_figs/fig_mujoco_scenario.pdf` — 2-row figure:
    top row     : three robot snapshots at t ∈ {0.0, 2.5, 5.0} s
                  (matplotlib 3-D links/joints drawn from MuJoCo forward
                   kinematics; no OpenGL render context required)
    bottom row  : joint angle trajectories over the full 5 s sim
                  (4 joints × 500 Hz → 2500 samples per joint)

No training involved. The PID tracks a deterministic setpoint; the robot
is not "doing a grasping task" — it is executing a scripted motion whose
wall-clock physics step cost is what the paper's §5 / Round-3 timing
claims predict.
"""
from __future__ import annotations

import os
# Must be set before `import mujoco` so the renderer picks the headless
# EGL backend (verified available on this host).
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "hymeko_bench" / "paper_figs"


# ----------------------------------------------------------------------
# Canonical 4-DOF arm MJCF (matches the paper's Listing A.6.1 topology).
#
# Kinematic chain (same as examples/paper/hymeko_robot.hymeko):
#   base_link --j1 (Z)-- shoulder_link --j2 (Y)-- elbow_link
#              --j3 (Y)-- wrist_link --j4 (Z)-- flange_link
#
# Joint labels match the paper's j1..j4 naming. Geometry is schematic
# (cylinders + boxes); this scene exists to support the physics sim and
# the paper figure, not to claim a particular robot model.
# ----------------------------------------------------------------------
MJCF = """
<mujoco model="hymeko_grasping_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="135" elevation="-22" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.75" rgb2="0.08 0.10 0.14"
             width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker"
             mark="cross" rgb1="0.78 0.78 0.82" rgb2="0.68 0.68 0.72"
             markrgb="0.3 0.3 0.3" width="300" height="300"/>
    <material name="floormat" texture="grid" texrepeat="4 4" texuniform="true"
              reflectance="0.15"/>
    <material name="link_mat"    rgba="0.55 0.58 0.62 1" specular="0.4" shininess="0.3"/>
    <material name="base_mat"    rgba="0.35 0.40 0.45 1" specular="0.4" shininess="0.3"/>
    <material name="flange_mat"  rgba="0.22 0.46 0.78 1" specular="0.55" shininess="0.5"/>
  </asset>

  <default>
    <joint damping="2.0" limited="true" range="-2.5 2.5"/>
    <geom material="link_mat" friction="0.8 0.05 0.0005"/>
    <position kp="40" kv="4" ctrlrange="-2 2"/>
  </default>

  <worldbody>
    <light pos="0.8 0.8 1.6" dir="-0.4 -0.4 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <light pos="-0.6 0.4 1.3" dir=" 0.3 -0.2 -1" diffuse="0.35 0.35 0.4" specular="0.1 0.1 0.1"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.05" material="floormat"/>

    <!-- Cameras auto-track the elbow link so the full 62-cm-tall arm
         fits in frame across the sine-motion sweep. -->
    <camera name="iso"   pos="1.05 -1.05 0.75" mode="targetbody" target="elbow_link" fovy="35"/>
    <camera name="side"  pos="1.30  0    0.40" mode="targetbody" target="elbow_link" fovy="32"/>

    <!-- base_link: welded to world (no joint on the root body).
         Each j_k is attached to the *distal* link so only the downstream
         sub-chain rotates, matching Listing A.6.1 semantics. -->
    <body name="base_link" pos="0 0 0">
      <inertial pos="0 0 0.04" mass="1.5" diaginertia="0.005 0.005 0.003"/>
      <geom type="cylinder" size="0.06 0.04" pos="0 0 0.04" material="base_mat"/>

      <body name="shoulder_link" pos="0 0 0.08">
        <inertial pos="0 0 0.06" mass="0.8" diaginertia="0.002 0.002 0.001"/>
        <joint name="j1" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <geom type="cylinder" size="0.035 0.06" pos="0 0 0.06"/>

        <body name="elbow_link" pos="0 0 0.12">
          <inertial pos="0 0 0.12" mass="0.6" diaginertia="0.004 0.004 0.0005"/>
          <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom type="box" size="0.03 0.025 0.12" pos="0 0 0.12"/>

          <body name="wrist_link" pos="0 0 0.24">
            <inertial pos="0 0 0.09" mass="0.4" diaginertia="0.002 0.002 0.0003"/>
            <joint name="j3" type="hinge" axis="0 1 0" pos="0 0 0"/>
            <geom type="box" size="0.025 0.02 0.09" pos="0 0 0.09"/>

            <body name="flange_link" pos="0 0 0.18">
              <inertial pos="0 0 0.025" mass="0.15" diaginertia="0.0002 0.0002 0.0001"/>
              <joint name="j4" type="hinge" axis="0 0 1" pos="0 0 0"/>
              <geom type="cylinder" size="0.03 0.025" pos="0 0 0.025" material="flange_mat"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="m1" joint="j1"/>
    <position name="m2" joint="j2"/>
    <position name="m3" joint="j3"/>
    <position name="m4" joint="j4"/>
  </actuator>
</mujoco>
"""


def simulate(sim_seconds: float = 5.0, ctrl_hz: float = 200.0) -> dict:
    """Run the PID-tracked sine-wave motion; return recorded telemetry."""
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)

    dt = model.opt.timestep  # 0.002 s
    steps = int(sim_seconds / dt)
    ctrl_period = max(1, int(round((1.0 / ctrl_hz) / dt)))

    # Sine-wave setpoint (rad) — distinct freq/phase per joint so the
    # motion is visually interesting. MuJoCo `position` actuators close
    # the PID loop internally (kp=40 kv=4 from the MJCF default); we
    # simply push the setpoint into data.ctrl at the control rate.
    def setpoint(t: float) -> np.ndarray:
        return np.array([
             0.7 * np.sin(0.6 * t),             # j1: base yaw
             0.9 * np.sin(0.4 * t + 0.3),       # j2: shoulder pitch
            -0.6 * np.sin(0.5 * t + 0.8),       # j3: elbow pitch
             1.2 * np.sin(0.9 * t + 1.2),       # j4: wrist roll
        ])

    log_t   = np.zeros(steps)
    log_q   = np.zeros((steps, 4))
    log_qd  = np.zeros((steps, 4))
    log_set = np.zeros((steps, 4))
    log_tau = np.zeros((steps, 4))

    for step in range(steps):
        t = step * dt
        log_t[step]   = t
        log_q[step]   = data.qpos.copy()
        log_qd[step]  = data.qvel.copy()
        sp = setpoint(t)
        log_set[step] = sp

        if step % ctrl_period == 0:
            data.ctrl[:] = sp
        log_tau[step] = data.ctrl.copy()

        mujoco.mj_step(model, data)

    # Render MuJoCo snapshots via the EGL offscreen context. Pose the
    # model to the logged q at each snapshot time, run mj_forward, then
    # render with the fixed "iso" camera. Returns RGB uint8 arrays.
    snap_times = [0.0, sim_seconds * 0.25, sim_seconds * 0.5,
                  sim_seconds * 0.75, sim_seconds]
    renderer = mujoco.Renderer(model, height=480, width=640)
    snapshots = []
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "iso")
    for t in snap_times:
        idx = min(int(t / dt), steps - 1)
        data.qpos[:] = log_q[idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render().copy()
        snapshots.append((t, img))

    return {
        "model":     model,
        "t":         log_t,
        "q":         log_q,
        "qd":        log_qd,
        "setpoint":  log_set,
        "tau":       log_tau,
        "snapshots": snapshots,
    }


def make_figure(tel: dict, out_path: Path) -> None:
    n_snap = len(tel["snapshots"])
    fig = plt.figure(figsize=(2.2 * n_snap + 0.4, 5.6))
    gs = fig.add_gridspec(
        nrows=2, ncols=n_snap, height_ratios=[1.2, 1.0],
        hspace=0.30, wspace=0.08,
    )

    for col, (t, img) in enumerate(tel["snapshots"]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(rf"$t{{=}}{t:.2f}$ s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.5")
            spine.set_linewidth(0.6)

    ax_tr = fig.add_subplot(gs[1, :])
    styles = [("j1", "-",  "0.15"),
              ("j2", "--", "0.35"),
              ("j3", "-.", "0.50"),
              ("j4", ":",  "0.70")]
    for idx, (name, ls, c) in enumerate(styles):
        ax_tr.plot(tel["t"], tel["q"][:, idx], ls, color=c,
                   linewidth=1.3, label=f"{name} (measured)")
        ax_tr.plot(tel["t"], tel["setpoint"][:, idx], ls, color=c,
                   linewidth=0.7, alpha=0.45)
    ax_tr.set_xlabel("Simulation time (s)")
    ax_tr.set_ylabel("Joint angle (rad)")
    ax_tr.set_title("PID-tracked joint trajectories "
                    "(measured solid; setpoint faint)", fontsize=9)
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(loc="upper right", ncol=4, frameon=False, fontsize=8)

    fig.suptitle("MuJoCo simulation of the canonical 4-DOF grasping arm "
                 "(Listing A.6.1 topology)", fontsize=10, y=0.995)
    fig.savefig(out_path / "fig_mujoco_scenario.pdf", bbox_inches="tight")
    fig.savefig(out_path / "fig_mujoco_scenario.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_path / 'fig_mujoco_scenario.pdf'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tel = simulate(sim_seconds=5.0)

    # Summary stats for the report.
    settled = tel["t"] > 1.0  # ignore transient
    tracking_err = tel["setpoint"] - tel["q"]
    rmse_rad = np.sqrt(np.mean(tracking_err[settled] ** 2, axis=0))
    print(f"Sim time: {tel['t'][-1]:.2f} s, steps: {len(tel['t'])}")
    print("Per-joint tracking RMSE (rad, post-transient):")
    for name, e in zip(["j1", "j2", "j3", "j4"], rmse_rad):
        print(f"  {name}: {e:.4f}  ({np.degrees(e):.2f}°)")

    # Dump telemetry for independent re-plotting.
    out_csv = REPO / "hymeko_bench" / "results" / "mujoco_scenario.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "t," + ",".join(f"q_{j}" for j in range(4)) \
        + "," + ",".join(f"qd_{j}" for j in range(4)) \
        + "," + ",".join(f"sp_{j}" for j in range(4)) \
        + "," + ",".join(f"tau_{j}" for j in range(4))
    arr = np.hstack([tel["t"][:, None], tel["q"], tel["qd"],
                     tel["setpoint"], tel["tau"]])
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
    print(f"wrote telemetry: {out_csv}")

    make_figure(tel, OUT)


if __name__ == "__main__":
    main()
