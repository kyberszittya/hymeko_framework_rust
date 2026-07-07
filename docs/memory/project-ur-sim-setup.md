---
name: project-ur-sim-setup
description: How the UR5e robot simulation (Gazebo + MoveIt2 + ros2_control) is set up for the hymeko_ros2_demo Tier-1 demo
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f22af6d-3303-4305-8e91-6e4b52c469c5
---

UR5e-in-simulation setup for `hymeko_ros2_demo` (MDPI Technologies Tier-1 live demo). Stack: **ROS 2 Kilted on Ubuntu 24.04**, system Python 3.12 (matches the `hymeko-0.1.0-cp312` wheel ABI).

**Packages (all official apt):** `ros-kilted-ur-description`, `ros-kilted-ur-robot-driver`, `ros-kilted-ur-moveit-config`, `ros-kilted-ur-simulation-gz`, `ros-kilted-moveit-*`, `ros-kilted-gz-*`, `ros-kilted-ros2-controllers`. HyMeKo wheel built locally and `pip install --user --break-system-packages target/wheels/hymeko-0.1.0-cp312-cp312-linux_x86_64.whl`.

**Launch:** `ros2 launch hymeko_ros2_demo ur5e_grasping_demo.launch.py`. Defers to upstream `ur_simulation_gz/ur_sim_moveit.launch.py` (Gazebo + UR5e + ros2_control + MoveIt2 + RViz2). `grasping_context_only.launch.py` is the node-only smoke (synthetic publisher, no Gazebo).

**Non-obvious sim tuning (the part that took work — keep this):**
- **Relaxed controllers YAML** (`config/ur_controllers_relaxed.yaml`) is passed as `controllers_file`. Upstream defaults target a real UR (goal_time=0, per-joint goal=0.1 rad) and report `GOAL_TOLERANCE_VIOLATED` under Gazebo physics even when the trajectory executed. Relaxed: `goal_time 0→2.0s`, `stopped_velocity_tolerance 0.2→0.5 rad/s`, per-joint goal `0.1→0.25 rad`, trajectory `0.2→0.4 rad`. Only the constraints block differs from upstream.
- **MoveIt start tolerance** relaxed at runtime (T+18s): `ros2 param set /move_group trajectory_execution.allowed_start_tolerance 0.2` (upstream 0.01 rad is too tight vs Gazebo joint drift) + `allowed_goal_duration_margin 5.0`.
- **`use_sim_time: True`** on every node — Gazebo publishes `/clock`; mixing wall and sim stamps causes tf2 "jump back in time" warnings.
- **RViz2 SIGSEGV workaround:** on NVIDIA + OGRE-Next, RViz2 can crash. Pass `launch_rviz:=false` → falls back to `ur_sim_control.launch.py` (sim + controllers only, no MoveIt/RViz). Contextual flow + pick_and_place still work without MoveIt.
- **Motion timing:** `pick_and_place` starts at T+20s so `scaled_joint_trajectory_controller` action server has spawned (needs ≥~15s).

**Two motion backends** (`motion_backend` arg): `direct` (default — sends joint trajectories straight to `scaled_joint_trajectory_controller`, deterministic, no MoveIt) and `moveit` (calls `/move_action`, plans+collision-checks, and gates each waypoint on live `/hymeko/grasping/stability_margin > stability_gate` — the closed-loop demo; requires `launch_rviz:=true`).

**The 6 UR joints:** shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.

**No gripper / no F/T in the bare sim** — `topic_pub_sim` provides synthetic upstream-perception topics (`/tool_id`, `/payload_id`, `/grasp_mode`, `/wrench`, `/tcp_pose`). Tier-2 (deferred) would add a real parallel-gripper plugin + F/T sensor to the URDF. Related: [[project-seminar-demos-and-hymeyolo-plan]], [[project-sisy2026-control-paper]].
